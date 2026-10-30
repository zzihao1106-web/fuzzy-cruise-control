"""V4严格串行：双图深度恢复→保存去白PNG→仅从PNG识别边界→两段动画。"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from deep_data import (SpraySynthesizer, channels, correct_photometry, crop_geometry,
                       crop_resize, read_gray, source_connected, write_json)
from deep_spray import font, input_config, overlay, runtime, score_batch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_input(signed, gain=3.0):
    """固定增益的非负观测贡献图；不作逐帧自动拉伸或喷雾掩膜相乘。"""
    return np.clip(np.abs(signed)*gain, 0, 1).astype(np.float32)


def gray_channels(gray):
    """边界阶段只接收已经去白的灰度图，没有原图或原始背景入口。"""
    return channels(gray, np.zeros_like(gray))


def sample_batch(synth, batch, exact_empty=False):
    samples = [synth.sample(force_empty=exact_empty, restoration=True) for _ in range(batch)]
    x = np.stack([s[0] for s in samples])
    semantic = np.stack([s[1] for s in samples])
    target = np.stack([s[2] for s in samples])
    return x, semantic, target


def restoration_metrics(prediction, target, x, semantic):
    foreground = semantic[:, :1] > 0.5
    background = ~foreground
    error = np.abs(prediction-target)
    raw = x[:, :1]-x[:, 1:2]
    white_bg = background & (x[:, 1:2] > 0.45)
    mean = lambda a, m: float(a[m].mean()) if m.any() else None
    foreground_mae = mean(error, foreground) or 0
    background_mae = mean(error, background) or 0
    return {"foreground_mae": foreground_mae, "background_mae": background_mae,
            "white_background_residual_mae": mean(np.abs(prediction), white_bg),
            "plain_difference_foreground_mae": mean(np.abs(raw-target), foreground),
            "plain_difference_background_mae": mean(np.abs(raw-target), background),
            "foreground_energy_ratio": float(np.abs(prediction)[foreground].sum()/max(1e-8, np.abs(target)[foreground].sum())),
            "mse": float(np.square(prediction-target).mean()),
            "score_lower_is_better": foreground_mae+2*background_mae, "synthetic_only": True}


def evaluate_restorer(model, data, source, device, seed=319871, batches=6):
    import torch
    synth = SpraySynthesizer(data/"backgrounds.npz", source, seed, True, curriculum=True)
    predictions, targets, xs, masks = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for _ in range(batches):
            x, semantic, target = sample_batch(synth, 4)
            r, _ = model(torch.from_numpy(x).to(device))
            predictions.append(r.float().cpu().numpy()); targets.append(target); xs.append(x); masks.append(semantic)
    return restoration_metrics(np.concatenate(predictions), np.concatenate(targets), np.concatenate(xs), np.concatenate(masks))


def train(args):
    torch, device = runtime(args.device)
    from restore_model import ReferenceRestorer, RESTORE_VERSION, restoration_loss
    from deep_model import BackgroundUNet, supervised_loss
    data, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.steps < 1 or args.boundary_steps < 1 or args.batch < 1:
        raise ValueError("两阶段训练步数和batch均须为正。")
    info = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    source, seed = info["source_normalized"], args.seed
    torch.manual_seed(seed); np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    restorer = ReferenceRestorer(width=args.width).to(device)
    opt = torch.optim.AdamW(restorer.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    synth = SpraySynthesizer(data/"backgrounds.npz", source, seed, curriculum=True)
    history, best, start = [], float("inf"), time.perf_counter()
    for step in range(1, args.steps+1):
        restorer.train()
        x, semantic, target = sample_batch(synth, args.batch)
        if step % 8 == 0:
            # 精确背景配对也是监督样本，防止零变化被网络变成喷雾。
            x[:, 0] = x[:, 1]; x[:, 2:] = 0; target[:] = 0; semantic[:] = 0
        inputs, truth = torch.from_numpy(x).to(device), torch.from_numpy(target).to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction, nuisance = restorer(inputs)
            loss = restoration_loss(prediction, truth, torch.from_numpy(semantic).to(device))
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(restorer.parameters(), 3)
        scaler.step(opt); scaler.update()
        if step == 1 or step % 50 == 0:
            print(f"深度去白训练 {step}/{args.steps}，损失{loss.item():.5f}", flush=True)
        if step % 100 == 0 or step == args.steps:
            metrics = evaluate_restorer(restorer, data, source, device)
            history.append({"step": step, **metrics})
            state = {"version": RESTORE_VERSION, "restorer_spec": restorer.spec,
                     "restorer": restorer.state_dict(), "cleaner_step": step, "gain": 3.0,
                     "seed": seed, "size": synth.size, "cleaner_validation": metrics,
                     "training_kind": "synthetic_paired_restoration", "human_labels": 0}
            torch.save(state, out/"cleaner_last.pt")
            if metrics["score_lower_is_better"] < best:
                best = metrics["score_lower_is_better"]
                torch.save(state, out/"cleaner_best.pt")
            write_json(out/"cleaner_history.json", history)
            print(f"配对恢复验证：{json.dumps(metrics, ensure_ascii=False)}", flush=True)
    cleaner_seconds = time.perf_counter()-start
    state = torch.load(out/"cleaner_best.pt", map_location="cpu", weights_only=True)
    restorer.load_state_dict(state["restorer"]); restorer.eval()
    # 第二阶段只使用冻结恢复器的输出。原始图和背景不进入边界网络。
    boundary = BackgroundUNet(width=16, iterations=1).to(device)
    opt = torch.optim.AdamW(boundary.parameters(), lr=0.001)
    synth = SpraySynthesizer(data/"backgrounds.npz", source, seed+4000, curriculum=True)
    boundary_history, best_iou = [], -1
    for step in range(1, args.boundary_steps+1):
        x, semantic, target = sample_batch(synth, args.batch)
        with torch.no_grad():
            r, _ = restorer(torch.from_numpy(x).to(device))
            c = (r.abs()*state["gain"]).clamp(0, 1)
            # 与保存PNG及重新读取一致，训练时也量化成8位。
            c = (c*255).round()/255
            inputs = torch.cat([c, torch.zeros_like(c), c, torch.zeros_like(c)], 1)
        boundary.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = supervised_loss(boundary(inputs), torch.from_numpy(semantic).to(device))
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(boundary.parameters(), 5); scaler.step(opt); scaler.update()
        if step == 1 or step % 50 == 0:
            print(f"仅去白图边界训练 {step}/{args.boundary_steps}，损失{loss.item():.4f}", flush=True)
        if step % 100 == 0 or step == args.boundary_steps:
            validation = SpraySynthesizer(data/"backgrounds.npz", source, seed+90000, True, curriculum=True)
            ps, ys = [], []
            boundary.eval()
            with torch.inference_mode():
                for _ in range(4):
                    xx, yy, _ = sample_batch(validation, 4)
                    rr, _ = restorer(torch.from_numpy(xx).to(device))
                    cc = ((rr.abs()*state["gain"]).clamp(0, 1)*255).round()/255
                    z = boundary(torch.cat([cc, torch.zeros_like(cc), cc, torch.zeros_like(cc)], 1))[-1][:, 0].sigmoid()
                    ps.append(z.cpu().numpy()); ys.append(yy[:, 0])
            metrics = score_batch(np.concatenate(ps), np.concatenate(ys))
            boundary_history.append({"step": step, **metrics})
            bundle = {**state, "boundary_spec": boundary.spec, "boundary": boundary.state_dict(),
                      "boundary_step": step, "boundary_validation": metrics,
                      "boundary_input_contract": "saved_cleaned_grayscale_png_only"}
            torch.save(bundle, out/"pipeline_last.pt")
            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]; torch.save(bundle, out/"pipeline_best.pt")
            write_json(out/"boundary_history.json", boundary_history)
    summary = {"version": RESTORE_VERSION, "cleaner_steps": args.steps, "boundary_steps": args.boundary_steps,
               "cleaner_seconds": cleaner_seconds, "total_seconds": time.perf_counter()-start,
               "cleaner_parameters": sum(p.numel() for p in restorer.parameters()),
               "boundary_parameters": sum(p.numel() for p in boundary.parameters()), "batch": args.batch,
               "seed": seed, "size": synth.size, "gain": state["gain"], "learning_rate": args.lr,
               "cleaner_best_metrics": state["cleaner_validation"], "best_synthetic_boundary_iou": best_iou,
               "peak_mib": torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else None,
               "human_labels": 0, "raw_frames_used_for_boundary_input": False,
               "real_accuracy_available": False, "manifest_sha256": digest(data/"manifest.json")}
    write_json(out/"training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def load_bundle(path, device):
    import torch
    from restore_model import ReferenceRestorer
    from deep_model import BackgroundUNet
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    restorer = ReferenceRestorer(**bundle["restorer_spec"]).to(device).eval()
    restorer.load_state_dict(bundle["restorer"])
    boundary = BackgroundUNet(**bundle["boundary_spec"]).to(device).eval()
    boundary.load_state_dict(bundle["boundary"])
    return restorer, boundary, bundle


def boundary_from_cleaned(folder, output, model, cfg, device):
    """这个入口没有原图目录或背景参数，必须从已落盘的去白图开始。"""
    import torch
    from spray_recognition import discover_sequence
    paths, _ = discover_sequence(folder)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    first = read_gray(paths[0]); bounds, roi, _ = crop_geometry(first.shape, cfg)
    size = int(cfg.get("restoration_size", 512))
    small_roi = crop_resize(roi, bounds, size, True) > 0.5
    x0, y0, x1, y1 = bounds; sx, sy = cfg["source_xy"]
    source = ((sx-x0)/(x1-x0)*size, (sy-y0)/(y1-y0)*size)
    empty_count = max(1, int(cfg.get("background_frames", 5)))
    empty = np.stack([crop_resize(read_gray(p), bounds, size) for p in paths[:empty_count]])
    floor = max(3/255, float(np.quantile(empty[:, small_roi], 0.999)))
    records, animation = [], []
    for directory in ("masks", "unknown", "overlays"):
        (output/directory).mkdir(exist_ok=True)
    for index, path in enumerate(paths, 1):
        cleaned = read_gray(path)
        im = crop_resize(cleaned, bounds, size)
        with torch.inference_mode():
            p = model(torch.from_numpy(gray_channels(im)[None]).to(device))[-1].sigmoid().cpu().numpy()[0, 0]
        mask = source_connected((p > 0.5) & small_roi, source, size*0.065)
        unknown = ((p > 0.2) & (p < 0.8) | ((p > 0.5) & (im <= floor))) & small_roi
        valid = int(((im > floor) & small_roi).sum()) >= max(12, int(small_roi.sum()*0.0005))
        if not valid:
            unknown |= mask; mask[:] = False
        # 边界仅叠加在去白图上；原图不参与计算或最终动画。
        frame = overlay(im, mask, unknown)
        draw = ImageDraw.Draw(frame); draw.text((8, 8), f"{index:03d} 去白后边界", font=font(17), fill="white")
        frame.save(output/"overlays"/f"{index:06d}.png")
        animation.append(frame.resize((480, 480)))
        for name, array in (("masks", mask), ("unknown", unknown)):
            full = np.zeros_like(cleaned)
            full[y0:y1, x0:x1] = cv2.resize(array.astype(np.float32), (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST)
            Image.fromarray(np.uint8(full*255)).save(output/name/f"{index:06d}.png")
        records.append({"frame": index, "input_file": path.name, "cleaned_input_sha256": digest(path),
                        "measurement_valid": valid, "mask_pixels_processing": int(mask.sum()), "unknown_pixels": int(unknown.sum())})
    animation[0].save(output/"去白后边界动画.gif", save_all=True, append_images=animation[1:], duration=100, loop=0, disposal=2)
    write_json(output/"boundary_inputs.json", records)
    return {"frames": len(paths), "input_contract": "仅已保存去白PNG", "empty_noise_level": floor,
            "raw_images_read_by_boundary_stage": False}


def process(args):
    torch, device = runtime(args.device)
    from restore_model import RESTORE_VERSION
    from spray_recognition import auto_geometry, discover_sequence
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    folder = args.input or cfg["source_dir"]
    cfg = input_config(cfg, folder, getattr(args, "reuse_geometry", False))
    out = Path(args.output or cfg["output_dir"])
    if out.resolve() == Path(folder).resolve():
        raise ValueError("结果目录不能与原图目录相同。")
    out.mkdir(parents=True, exist_ok=True)
    paths, excluded = discover_sequence(folder)
    count = int(cfg.get("background_frames", 5))
    if count < 2 or count >= len(paths) or args.size < 32:
        raise ValueError("起喷前背景帧数必须至少2且小于总帧数，处理尺寸至少32。")
    if not all(cfg.get(k) is not None for k in ("source_xy", "chamber_xy", "chamber_axes_px")):
        cfg.update(auto_geometry(paths, count))
    cfg["restoration_size"] = args.size
    fit_count = max(1, count-min(2, count-1))
    reference = np.median(np.stack([read_gray(p) for p in paths[:fit_count]]), 0)
    bounds, roi, static = crop_geometry(reference.shape, cfg)
    x0, y0, x1, y1 = bounds
    restorer, boundary, bundle = load_bundle(args.checkpoint, device)
    for name in ("cleaned_frames", "removed_background", "comparison"):
        (out/name).mkdir(exist_ok=True)
    Image.fromarray(np.uint8(reference*255)).save(out/"原始背景参考.png")
    frames, comparisons, records = [], [], []
    start = time.perf_counter()
    for index, path in enumerate(paths, 1):
        original = read_gray(path)
        corrected, spec = correct_photometry(original, reference, static)
        image, bg = crop_resize(original, bounds, args.size), crop_resize(corrected, bounds, args.size)
        with torch.inference_mode():
            signed, nuisance = restorer(torch.from_numpy(channels(image, bg)[None]).to(device))
        signed = signed[0, 0].float().cpu().numpy()
        nuisance = nuisance[0, 0].float().cpu().numpy()
        cleaned = clean_input(signed, bundle["gain"])
        small_roi = crop_resize(roi, bounds, args.size, True) > 0.5
        cleaned *= small_roi
        # 保存的是恢复网络的连续灰度输出，没有用边界或喷雾分割掩膜筛图。
        full = np.zeros_like(original)
        full[y0:y1, x0:x1] = cv2.resize(cleaned, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
        full *= roi
        saved = out/"cleaned_frames"/f"{index:06d}.png"
        Image.fromarray(np.uint8(np.rint(full*255))).save(saved)
        removed = np.clip(corrected, 0, 1)
        removed[y0:y1, x0:x1] = cv2.resize(np.clip(bg+nuisance, 0, 1), (x1-x0, y1-y0))
        Image.fromarray(np.uint8(removed*255)).save(out/"removed_background"/f"{index:06d}.png")
        frame = Image.fromarray(np.uint8(np.rint(cleaned*255))).convert("RGB")
        draw = ImageDraw.Draw(frame); draw.text((8, 8), f"{index:03d} 深度去白", font=font(17), fill="white")
        frames.append(frame.resize((480, 480)))
        compare = Image.new("RGB", (960, 510), "black")
        compare.paste(Image.fromarray(np.uint8(image*255)).convert("RGB").resize((480, 480)), (0, 30))
        compare.paste(frame.resize((480, 480)), (480, 30))
        draw = ImageDraw.Draw(compare); draw.text((8, 3), "原图（仅对照）", font=font(18), fill="white")
        draw.text((488, 3), "深度比对去白：边界模块的实际输入", font=font(18), fill="white")
        comparisons.append(compare)
        if index in {1, 4, 5, 6, 20, 50, len(paths)}:
            compare.save(out/"comparison"/f"{index:06d}.jpg", quality=95)
        records.append({"frame": index, "original_sha256": digest(path), "cleaned_sha256": digest(saved),
                        "mean_learned_correction": float(np.abs(nuisance[small_roi]).mean()),
                        "gain": spec["gain"], "offset": spec["offset"]})
        if index % 10 == 0 or index == len(paths):
            print(f"深度逐帧去白 {index}/{len(paths)}", flush=True)
    frames[0].save(out/"逐帧深度去白.gif", save_all=True, append_images=frames[1:], duration=100, loop=0, disposal=2)
    comparisons[0].save(out/"原图与去白对照.gif", save_all=True, append_images=comparisons[1:], duration=100, loop=0, disposal=2)
    # 先结束全部清理并落盘，再启动独立的边界阶段。
    print("全部去白PNG已保存，现在仅从这些PNG捕捉边界。", flush=True)
    boundary_summary = boundary_from_cleaned(out/"cleaned_frames", out/"boundary", boundary, cfg, device)
    write_json(out/"frame_mapping.json", records)
    write_json(out/"运行配置.json", cfg)
    summary = {"version": RESTORE_VERSION, "frames": len(paths), "reference_frames": fit_count,
               "checkpoint_sha256": digest(args.checkpoint), "checkpoint_version": bundle["version"],
               "cleaner_step": bundle["cleaner_step"], "boundary_step": bundle["boundary_step"],
               "fixed_display_gain": bundle["gain"], "segmentation_mask_used_to_clean_image": False,
               "boundary": boundary_summary, "human_labels": 0, "real_accuracy_available": False,
               "seconds": time.perf_counter()-start, "excluded_attachment_count": len(excluded),
               "restoration_meaning": "保留增亮/变暗的可观测喷雾贡献，黑底为固定背景剔除结果；未恢复不可观测纹理"}
    write_json(out/"restoration_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def process_batch(args):
    from spray_recognition import discover_sequence
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root, output = Path(args.input or cfg["source_dir"]), Path(args.output or cfg["output_dir"])
    if root.resolve() == output.resolve():
        raise ValueError("批处理输出不能与原始输入根目录相同。")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.resolve() == output.resolve():
            continue
        try:
            discover_sequence(child)
        except ValueError:
            records.append({"sequence": child.name, "status": "skipped_no_images"}); continue
        try:
            dest = output/child.name; dest.mkdir(exist_ok=True)
            config_path = dest/"工具运行设置.json"
            write_json(config_path, {**cfg, "source_dir": str(child), "output_dir": str(dest),
                                    "source_xy": None, "chamber_xy": None, "chamber_axes_px": None})
            options = argparse.Namespace(**vars(args))
            options.config, options.input, options.output = str(config_path), str(child), str(dest)
            summary = process(options)
            records.append({"sequence": child.name, "status": "ok", "frames": summary["frames"]})
        except Exception as exc:
            records.append({"sequence": child.name, "status": "error", "message": str(exc)})
    write_json(output/"restoration_batch.json", records)
    if not any(r["status"] == "ok" for r in records):
        raise RuntimeError("没有成功处理的图像序列。")


def main():
    p = argparse.ArgumentParser(description="先深度去白，再从去白图识别边界")
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True); t.add_argument("--output", required=True)
    t.add_argument("--steps", type=int, default=800); t.add_argument("--boundary-steps", type=int, default=500)
    t.add_argument("--batch", type=int, default=4); t.add_argument("--width", type=int, default=16)
    t.add_argument("--seed", type=int, default=20261101); t.add_argument("--lr", type=float, default=0.001)
    t.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    t = sub.add_parser("process")
    t.add_argument("--config", required=True); t.add_argument("--checkpoint", required=True)
    t.add_argument("--input"); t.add_argument("--output"); t.add_argument("--size", type=int, default=320)
    t.add_argument("--reuse-geometry", action="store_true")
    t.add_argument("--batch", action="store_true")
    t.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    if args.command == "train":
        train(args)
    elif args.batch:
        process_batch(args)
    else:
        process(args)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
