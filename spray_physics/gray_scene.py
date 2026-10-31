"""V5灰底场景：准备、三帧分层训练、整段校正，最后从落盘图捕捉边界。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from deep_data import (correct_photometry, crop_geometry, crop_resize, prepare,
                       read_gray, write_json)
from deep_spray import font, input_config, runtime
from gray_background import uniform_gray_variants
from gray_scene_data import GraySceneSynthesizer, temporal_channels


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_gray(path, array):
    Image.fromarray(np.uint8(np.rint(np.clip(array, 0, 1)*255))).save(path)


def prepare_gray(args):
    from spray_recognition import discover_sequence
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    folder = args.input or cfg["source_dir"]
    cfg = input_config(cfg, folder, args.reuse_geometry)
    data = Path(args.output)
    info = prepare(folder, cfg, data, args.size)
    cfg = info["config"]
    paths, _ = discover_sequence(folder)
    reference = np.median(np.stack([read_gray(paths[i-1]) for i in info["reference_frames_1based"]]), 0)
    textured, _, stats, maps = uniform_gray_variants(reference, cfg, return_maps=True)
    np.savez_compressed(data/"gray_reference.npz",
                        gray=crop_resize(textured, info["crop_bounds_xyxy"], args.size))
    save_gray(data/"灰底背景参考.png", textured)
    np.savez_compressed(data/"full_gray_reference.npz", reference=reference, gray=textured, **maps)
    write_json(data/"gray_reference_statistics.json", stats)
    print(json.dumps({"frames": len(paths), "reference_frames": info["reference_frames_1based"],
                      "heldout_empty": info["heldout_empty_frames_1based"], "size": args.size}, ensure_ascii=False))


def evaluate_gray(model, data, source, device, seed=573091, batches=4):
    import torch
    synth = GraySceneSynthesizer(data, source, seed, procedural_only=True)
    totals = {k: [0.0, 0] for k in ("foreground_mae", "background_mae", "weak_foreground_mae",
                                     "static_replacement_foreground_mae", "alpha_mae")}
    model.eval()
    with torch.inference_mode():
        for _ in range(batches):
            x, truth = synth.batch(4)
            inputs = torch.from_numpy(x).to(device)
            z = model.decompose(inputs)
            gray, alpha = z["gray"].cpu().numpy(), z["alpha"].cpu().numpy()
            fg = truth["alpha"] > 0.05
            weak = fg & (np.abs(x[:, 1:2]-x[:, 3:4]) < 0.025)
            errors = {"foreground_mae": (np.abs(gray-truth["gray"]), fg),
                      "background_mae": (np.abs(gray-truth["gray"]), ~fg),
                      "weak_foreground_mae": (np.abs(gray-truth["gray"]), weak),
                      "static_replacement_foreground_mae": (np.abs(x[:, 1:2]-(x[:, 3:4]-x[:, 4:5])-truth["gray"]), fg),
                      "alpha_mae": (np.abs(alpha-truth["alpha"]), np.ones_like(fg))}
            for key, (error, valid) in errors.items():
                totals[key][0] += float(error[valid].sum())
                totals[key][1] += int(valid.sum())
    result = {k: total/max(count, 1) for k, (total, count) in totals.items()}
    result.update(synthetic_only=True, samples=batches*4,
                  selection_score=result["foreground_mae"]+result["background_mae"]*2+result["weak_foreground_mae"])
    return result


def train_gray(args):
    from gray_scene_model import TemporalGrayRestorer, VERSION, gray_scene_loss
    torch, device = runtime(args.device)
    if min(args.steps, args.batch, args.validate_every) < 1:
        raise ValueError("训练步数、批次和验证间隔必须为正。")
    data, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    source = manifest["source_normalized"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = TemporalGrayRestorer(width=args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr*0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    synth = GraySceneSynthesizer(data, source, args.seed)
    start, history, best = time.perf_counter(), [], float("inf")
    for step in range(1, args.steps+1):
        model.train()
        x, truth = synth.batch(args.batch)
        inputs = torch.from_numpy(x).to(device)
        target = {k: torch.from_numpy(v).to(device) for k, v in truth.items()}
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = gray_scene_loss(model.decompose(inputs), target, inputs)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        scaler.step(opt); scaler.update(); schedule.step()
        if step == 1 or step % 50 == 0:
            print(f"灰底三帧训练 {step}/{args.steps}，损失{loss.item():.4f}，耗时{time.perf_counter()-start:.0f}秒", flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            metrics = evaluate_gray(model, data, source, device)
            history.append({"step": step, **metrics})
            state = {"version": VERSION, "kind": "temporal_gray_scene", "spec": model.spec,
                     "model": model.state_dict(), "step": step, "seed": args.seed,
                     "size": synth.synth.size, "validation": metrics,
                     "manifest_sha256": digest(data/"manifest.json"), "human_labels": 0}
            torch.save(state, out/"gray_last.pt")
            if metrics["selection_score"] < best:
                best = metrics["selection_score"]
                torch.save(state, out/"gray_best.pt")
            write_json(out/"training_history.json", history)
            print(json.dumps({"step": step, **metrics}, ensure_ascii=False), flush=True)
    state = torch.load(out/"gray_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    heldout = evaluate_gray(model, data, source, device, seed=args.seed+791039, batches=12)
    write_json(out/"heldout_synthetic.json", heldout)
    write_json(out/"training_summary.json", {"version": VERSION, "steps": args.steps, "best_step": state["step"],
               "batch": args.batch, "size": synth.synth.size, "seed": args.seed, "learning_rate": args.lr,
               "parameters": sum(p.numel() for p in model.parameters()), "seconds": time.perf_counter()-start,
               "human_labels": 0, "real_accuracy_available": False, "heldout": heldout,
               "context": "前一帧、当前帧、后一帧；离线处理，非因果实时模型",
               "training": "三帧程序合成光学混合；单段真实空背景用于参考，不作有喷雾真值"})


def process_gray(args):
    from gray_scene_model import TemporalGrayRestorer, VERSION
    from gray_scene_boundary import gray_boundary
    from spray_recognition import auto_geometry, discover_sequence
    torch, device = runtime(args.device)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    folder = args.input or cfg["source_dir"]
    cfg = input_config(cfg, folder, args.reuse_geometry)
    out = Path(args.output)
    if out.resolve() == Path(folder).resolve():
        raise ValueError("输出目录不能与原图相同。")
    if args.size < 32:
        raise ValueError("处理尺寸至少32。")
    paths, excluded = discover_sequence(folder)
    count = int(cfg.get("background_frames", 5))
    if count < 3 or count >= len(paths):
        raise ValueError("需要至少3张无喷雾背景及后续图像。")
    if not all(cfg.get(k) is not None for k in ("source_xy", "chamber_xy", "chamber_axes_px")):
        cfg.update(auto_geometry(paths, count))
    out.mkdir(parents=True, exist_ok=True)
    for name in ("gray_frames", "opacity", "restoration_unknown", "comparison"):
        (out/name).mkdir(exist_ok=True)
    fit_count = max(1, count-2)
    reference_frames = np.stack([read_gray(path) for path in paths[:fit_count]])
    reference = np.median(reference_frames, 0)
    gray, _, stats, maps = uniform_gray_variants(reference, cfg, return_maps=True)
    bounds, roi, static = crop_geometry(reference.shape, cfg)
    x0, y0, x1, y1 = bounds
    noise = np.median(np.abs(reference_frames-reference), 0)*1.4826+2/255
    b, g = crop_resize(reference, bounds, args.size), crop_resize(gray, bounds, args.size)
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if bundle.get("kind") != "temporal_gray_scene":
        raise ValueError("请选择V5灰底场景权重；旧黑底恢复模型不兼容。")
    model = TemporalGrayRestorer(**bundle["spec"]).to(device).eval()
    model.load_state_dict(bundle["model"])
    with torch.inference_mode():
        base_input = torch.from_numpy(temporal_channels(b, b, b, b, g)[None]).to(device)
        empty_layers = model.decompose(base_input)
        empty_alpha = empty_layers["alpha"][0, 0].cpu().numpy()
        empty_offset = empty_layers["offset"][0, 0].cpu().numpy()
    save_gray(out/"原始背景参考.png", reference)
    save_gray(out/"灰底背景参考.png", gray)
    np.savez_compressed(out/"恢复参考参数.npz", reference=reference, gray=gray,
                        empty_alpha=empty_alpha, empty_offset=empty_offset, **maps)
    # 每帧只从视窗外静态区域归一化曝光，再送入同一网络；不使用后帧构造背景。
    frames = []
    photometry = []
    for path in paths:
        original = read_gray(path)
        _, spec = correct_photometry(original, reference, static)
        frames.append(np.clip((original-spec["offset"])/spec["gain"], 0, 1))
        photometry.append(spec)
    records, animation, comparisons = [], [], []
    start = time.perf_counter()
    yy, xx = np.indices(reference.shape)
    cx, cy = cfg["chamber_xy"]; ax, ay = cfg["chamber_axes_px"]
    sx, sy = cfg["source_xy"]
    rho = np.hypot((xx-cx)/ax, (yy-cy)/ay)
    surface = (rho < 0.9) & (np.hypot(xx-sx, yy-sy) > min(ax, ay)*0.15)
    for i, path in enumerate(paths):
        image = frames[i]
        previous, following = frames[max(0, i-1)], frames[min(len(paths)-1, i+1)]
        inputs = temporal_channels(crop_resize(previous, bounds, args.size), crop_resize(image, bounds, args.size),
                                   crop_resize(following, bounds, args.size), b, g)
        with torch.inference_mode():
            z = model.decompose(torch.from_numpy(inputs[None]).to(device))
        alpha = z["alpha"][0, 0].cpu().numpy()
        # 同一空背景的模型响应校零，减少固定照明纹理的假透明度；不使用喷雾标签。
        alpha = np.clip((alpha-empty_alpha)/np.maximum(1-empty_alpha, 0.05), 0, 1)
        offset = z["offset"][0, 0].cpu().numpy()-empty_offset
        a, off = np.zeros_like(image), np.zeros_like(image)
        a[y0:y1, x0:x1] = cv2.resize(alpha, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
        off[y0:y1, x0:x1] = cv2.resize(offset, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
        estimated_bg = np.clip(reference+off, 0, 1)
        eps = noise*1.5
        physical_tmax = np.clip(np.minimum((image+eps)/(estimated_bg+eps),
                                           (1-image+eps)/(1-estimated_bg+eps)), 0, 1)
        transmission = np.minimum(1-a, physical_tmax)
        replaced = image-transmission*(estimated_bg-gray)
        original = read_gray(path)
        result = np.clip(original*(1-maps["weight"])+replaced*maps["weight"], 0, 1)
        protected = maps["weight"] == 0
        assert np.array_equal(np.uint8(np.rint(result[protected]*255)), np.uint8(np.rint(original[protected]*255)))
        delta = np.abs(image-estimated_bg)
        unknown = ((a > 0.08) & (delta < np.maximum(0.025, noise*3)) & (np.abs(estimated_bg-gray) > 0.04))
        unknown |= ((1-a-transmission) > 0.12) | ((original > 0.985) & (np.abs(reference-gray) > 0.03))
        unknown &= roi & (maps["weight"] > 0.5)
        filename = f"{i+1:06d}.png"
        saved = out/"gray_frames"/filename
        save_gray(saved, result)
        save_gray(out/"opacity"/filename, (1-transmission)*maps["weight"])
        save_gray(out/"restoration_unknown"/filename, unknown)
        preview = Image.open(saved).convert("RGB").resize((640, 400))
        ImageDraw.Draw(preview).text((10, 8), f"{i+1:03d} 灰底场景估计", font=font(18), fill="white", stroke_width=1, stroke_fill="black")
        animation.append(preview)
        if i+1 in {1, 6, 10, 20, 50, len(paths)}:
            canvas = Image.new("RGB", (1280, 430), "#eeeeee")
            canvas.paste(Image.fromarray(np.uint8(np.rint(original*255))).convert("RGB").resize((640, 400)), (0, 30))
            canvas.paste(Image.open(saved).convert("RGB").resize((640, 400)), (640, 30))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 3), f"第{i+1}帧 原图（仅对照）", font=font(18), fill="black")
            draw.text((650, 3), "灰底场景校正（含模型估计）", font=font(18), fill="black")
            canvas.save(out/"comparison"/f"{i+1:06d}.jpg", quality=95)
            comparisons.append((i+1, saved))
        records.append({"frame": i+1, "original_sha256": digest(path), "gray_sha256": digest(saved),
                        "photometry": photometry[i], "unknown_pixels": int(unknown.sum()),
                        "empty_surface_std_gray": float(result[surface].std()*255) if i < count else None,
                        "nozzle_core_preserved": True, "clipped_pixels_roi": int(((replaced < 0) | (replaced > 1))[roi].sum())})
        if (i+1) % 10 == 0 or i+1 == len(paths):
            print(f"灰底逐帧恢复 {i+1}/{len(paths)}", flush=True)
    animation[0].save(out/"灰底场景动画.gif", save_all=True, append_images=animation[1:], duration=100, loop=0, disposal=2)
    print("全部灰底PNG已保存，开始只从处理图捕捉边界。", flush=True)
    boundary = gray_boundary(out/"gray_frames", out/"boundary", cfg, out/"restoration_unknown")
    selected = [1, min(20, len(paths)), min(50, len(paths)), len(paths)]
    sheet = Image.new("RGB", (1440, len(selected)*330), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for row, index in enumerate(selected):
        for col, (title, image_path) in enumerate([
            (f"第{index}帧 原图", paths[index-1]),
            ("三帧深度校正：灰底场景", out/"gray_frames"/f"{index:06d}.png"),
            ("从灰底图捕捉边界（紫色低证据）", out/"boundary/overlays"/f"{index:06d}.png")]):
            x, y = col*480, row*330
            draw.text((x+6, y+4), title, font=font(17), fill="black")
            sheet.paste(Image.open(image_path).convert("RGB").resize((480, 300)), (x, y+30))
    sheet.save(out/"灰底恢复与边界总览.jpg", quality=96)
    write_json(out/"frame_mapping.json", records)
    cfg["engine"] = "gray_scene"; cfg["gray_processing_size"] = args.size
    write_json(out/"运行配置.json", cfg)
    summary = {"version": VERSION, "frames": len(paths), "fit_background_frames": fit_count,
               "heldout_empty_frames": [fit_count+1, count], "checkpoint_step": bundle["step"],
               "checkpoint_sha256": digest(args.checkpoint), "size": args.size,
               "model_input": "前帧、当前帧、后帧、原背景、灰底参考及差分通道",
               "equation": "J=I-T*(B_lit-B_gray)，T由网络及成像范围约束共同估计",
               "network_response_calibrated_on_empty_reference": True, "nozzle_core_preserved": True,
               "boundary": boundary, "human_labels": 0, "real_accuracy_available": False,
               "empty_surface_std_gray": [r["empty_surface_std_gray"] for r in records[:count]],
               "quantitative_measurement_valid": False, "seconds": time.perf_counter()-start,
               "excluded_attachment_count": len(excluded), "gray_reference": stats,
               "physical_pressure_velocity_formula_used": False,
               "limits": "单段真实背景加合成监督；同亮度、过曝区域仍欠定；喷雾灰度及边界为估计，不宣称重建真实纹理"}
    write_json(out/"gray_scene_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def process_gray_batch(args):
    from spray_recognition import discover_sequence
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root, out = Path(args.input or cfg["source_dir"]), Path(args.output)
    if root.resolve() == out.resolve():
        raise ValueError("批处理结果不能与原图根目录相同。")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.resolve() == out.resolve():
            continue
        try:
            discover_sequence(child)
        except ValueError:
            records.append({"sequence": child.name, "status": "skipped_no_images"})
            continue
        destination = out/child.name
        destination.mkdir(exist_ok=True)
        child_cfg = dict(cfg, source_dir=str(child), source_xy=None, chamber_xy=None, chamber_axes_px=None)
        write_json(destination/"批处理设置.json", child_cfg)
        options = argparse.Namespace(**vars(args))
        options.config, options.input, options.output = str(destination/"批处理设置.json"), None, str(destination)
        try:
            process_gray(options)
            records.append({"sequence": child.name, "status": "ok"})
        except (ValueError, OSError) as exc:
            records.append({"sequence": child.name, "status": "failed", "reason": str(exc)})
    write_json(out/"gray_scene_batch.json", records)
    if not any(r["status"] == "ok" for r in records):
        raise ValueError("没有序列成功处理，请检查子文件夹与背景帧设置。")


def main():
    p = argparse.ArgumentParser(description="保留灰底场景的三帧深度去白工具")
    sub = p.add_subparsers(dest="operation", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", required=True); prep.add_argument("--input")
    prep.add_argument("--output", required=True); prep.add_argument("--size", type=int, default=256)
    prep.add_argument("--reuse-geometry", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--data", required=True); train.add_argument("--output", required=True)
    train.add_argument("--steps", type=int, default=1000); train.add_argument("--batch", type=int, default=4)
    train.add_argument("--width", type=int, default=16); train.add_argument("--lr", type=float, default=0.0015)
    train.add_argument("--seed", type=int, default=20261031); train.add_argument("--device", default="auto")
    train.add_argument("--validate-every", type=int, default=200)
    process = sub.add_parser("process")
    process.add_argument("--config", required=True); process.add_argument("--input")
    process.add_argument("--output", required=True); process.add_argument("--checkpoint", required=True)
    process.add_argument("--size", type=int, default=256); process.add_argument("--device", default="auto")
    process.add_argument("--reuse-geometry", action="store_true")
    process.add_argument("--batch", action="store_true")
    args = p.parse_args()
    if args.operation == "prepare":
        prepare_gray(args)
    elif args.operation == "train":
        train_gray(args)
    else:
        (process_gray_batch if args.batch else process_gray)(args)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
