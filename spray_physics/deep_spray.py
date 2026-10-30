"""V3试验入口：准备数据、真实背景合成训练、整段推理及透明评估。"""

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from deep_data import (SpraySynthesizer, channels, correct_photometry, crop_geometry,
                       crop_resize, prepare, read_gray, source_connected, write_json)

HERE = Path(__file__).resolve().parent


def runtime(device="auto"):
    import torch
    torch.set_num_threads(4)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境不能使用CUDA，请检查独立环境或显式选择cpu。")
    return torch, torch.device(device)


def score_batch(probability, target, roi=None):
    pred, truth = probability > 0.5, target > 0.5
    valid = np.ones_like(pred, bool) if roi is None else np.broadcast_to(roi, pred.shape)
    tp = int((pred & truth & valid).sum())
    fp = int((pred & ~truth & valid).sum())
    fn = int((~pred & truth & valid).sum())
    tn = int((~pred & ~truth & valid).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "iou": tp / max(1, tp+fp+fn), "recall": tp/max(1, tp+fn),
            "background_false_positive_rate": fp/max(1, fp+tn)}


def boundary_f1(probability, target, valid, tolerance=2):
    pred, truth = probability > 0.5, target > 0.5
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    a = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool) & erosion
    b = cv2.morphologyEx(truth.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool) & erosion
    if not a.any() or not b.any():
        return None
    width = 2*tolerance+1
    near_a = cv2.dilate(a.astype(np.uint8), np.ones((width, width), np.uint8)) > 0
    near_b = cv2.dilate(b.astype(np.uint8), np.ones((width, width), np.uint8)) > 0
    precision = float(near_b[a].mean())
    recall = float(near_a[b].mean())
    return 2*precision*recall/max(1e-9, precision+recall)


def evaluate_synthetic(model, synth, device, batches=6, batch_size=4):
    import torch
    xs, ys, probabilities = [], [], []
    model.eval()
    with torch.inference_mode():
        for _ in range(batches):
            samples = [synth.sample() for _ in range(batch_size)]
            x = np.stack([a for a, _ in samples])
            y = np.stack([b for _, b in samples])
            z = model(torch.from_numpy(x).to(device))[-1][:, 0].sigmoid().cpu().numpy()
            probabilities.append(z)
            xs.append(x)
            ys.append(y)
    p, x, y = np.concatenate(probabilities), np.concatenate(xs), np.concatenate(ys)
    scores = score_batch(p, y[:, 0], synth.roi)
    # 固定差分阈值仅作同一合成集上的简单对照，不称为V2完整算法。
    d = np.maximum(x[:, 2], x[:, 3])
    plain = score_batch(d > 0.04, y[:, 0], synth.roi)
    invisible = (y[:, 0] > 0.5) & (np.abs(x[:, 0]-x[:, 1]) < 0.025) & synth.roi
    scores["low_contrast_recall"] = float((p[invisible] > 0.5).mean()) if invisible.any() else None
    scores["low_contrast_pixels"] = int(invisible.sum())
    scores["absolute_difference_baseline"] = plain
    boundaries = [boundary_f1(a, b, synth.roi) for a, b in zip(p, y[:, 0])]
    scores["boundary_f1_tolerance_2_processing_pixels"] = float(np.mean([b for b in boundaries if b is not None])) if any(b is not None for b in boundaries) else None
    scores["synthetic_only"] = True
    return scores


def load_annotations(folder, manifest, size):
    """真实监督仅从显式掩膜载入；255未知，0背景，1喷雾。"""
    if not folder:
        return []
    cfg = manifest["config"]
    source = Path(manifest["source_dir"])
    background_paths = [source/manifest["files"][i-1]["name"] for i in manifest["reference_frames_1based"]]
    bg = np.median(np.stack([read_gray(p) for p in background_paths]), 0)
    bounds, roi, static = crop_geometry(bg.shape, cfg)
    samples = []
    for entry in manifest["files"]:
        label_path = Path(folder) / (Path(entry["name"]).stem+".png")
        if not label_path.exists():
            continue
        image = read_gray(source/entry["name"])
        label = np.asarray(Image.open(label_path).convert("L"))
        if label.shape != bg.shape or not set(np.unique(label)).issubset({0, 1, 255}):
            raise ValueError(f"标注尺寸或类别非法：{label_path.name}")
        valid = (label != 255) & roi
        if valid.sum() == 0:
            continue
        corr, _ = correct_photometry(image, bg, static)
        x = channels(crop_resize(image, bounds, size), crop_resize(corr, bounds, size))
        mask = crop_resize(label == 1, bounds, size, True)
        v = crop_resize(valid, bounds, size, True)[None]
        # 稀疏标注不生成未经标注的轮廓真值，边缘辅助监督仅作零权重处理。
        target = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)])
        samples.append((x, target, v, entry["name"]))
    return samples


def train(args):
    torch, device = runtime(args.device)
    from deep_model import BackgroundUNet, DEEP_VERSION, supervised_loss
    if args.steps < 1 or args.batch < 1 or args.eval_every < 1 or args.lr <= 0:
        raise ValueError("训练步数、batch、评估间隔和学习率必须为正。")
    out, data = Path(args.output), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    model = BackgroundUNet(args.width, args.iterations).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["model_spec"] != model.spec:
            raise ValueError("继续训练的模型结构与参数不一致。")
        model.load_state_dict(checkpoint["state_dict"])
    synth = SpraySynthesizer(data/"backgrounds.npz", manifest["source_normalized"], args.seed)
    annotations = load_annotations(args.labels, manifest, synth.size)
    if args.labels and not annotations:
        raise ValueError("指定的标注目录没有确定标注；不能将未知标签当作监督。")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rows, best, start = [], -1, time.perf_counter()
    for step in range(1, args.steps+1):
        model.train()
        samples = [synth.sample() for _ in range(args.batch)]
        x = torch.from_numpy(np.stack([a for a, _ in samples])).to(device)
        target = torch.from_numpy(np.stack([b for _, b in samples])).to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = supervised_loss(model(x), target)
            if annotations and step % 2 == 0:
                a, b, v, _ = annotations[step//2 % len(annotations)]
                real_z = model(torch.from_numpy(a[None]).to(device))[-1][:, :1]
                # 稀疏真实标注只监督分割，不给辅助头制造伪标签。
                real_ce = torch.nn.functional.binary_cross_entropy_with_logits(real_z,
                    torch.from_numpy(b[None, :1]).to(device), reduction="none")
                valid = torch.from_numpy(v[None]).to(device)
                loss = loss + (real_ce*valid).sum()/valid.sum().clamp_min(1)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        scaler.step(opt)
        scaler.update()
        if step == 1 or step % 25 == 0:
            print(f"训练 {step}/{args.steps}，损失 {loss.item():.4f}，用时 {time.perf_counter()-start:.1f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            # 每次同一验证随机种子；只用程序生成背景，与训练真实背景分开。
            val = SpraySynthesizer(data/"backgrounds.npz", manifest["source_normalized"], args.seed+100000, True)
            metrics = evaluate_synthetic(model, val, device)
            row = {"step": step, "loss": float(loss.item()), **metrics}
            rows.append(row)
            checkpoint = {"state_dict": model.state_dict(), "model_spec": model.spec, "version": DEEP_VERSION,
                          "seed": args.seed, "step": step, "training_kind": "synthetic_plus_reviewed" if annotations else "synthetic_only",
                          "annotation_count": len(annotations), "annotation_files": [a[3] for a in annotations],
                          "size": synth.size, "validation": metrics}
            torch.save(checkpoint, out/"last.pt")
            if metrics["iou"] > best:
                best = metrics["iou"]
                torch.save(checkpoint, out/"best.pt")
            write_json(out/"training_log.json", rows)
            print(f"程序生成背景验证：IoU={metrics['iou']:.3f}，低对比召回={metrics['low_contrast_recall']}（仅合成）", flush=True)
    summary = {"version": DEEP_VERSION, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
               "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "model_spec": model.spec,
               "parameters": sum(p.numel() for p in model.parameters()), "steps": args.steps, "batch": args.batch,
               "seed": args.seed, "learning_rate": args.lr, "training_size": synth.size,
               "training_kind": "synthetic_plus_reviewed" if annotations else "synthetic_only",
               "human_annotation_frames": len(annotations), "annotation_files": [a[3] for a in annotations],
               "seconds": time.perf_counter()-start, "best_synthetic_iou": best,
               "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else None,
               "real_accuracy_available": False, "reference_frames_1based": manifest["reference_frames_1based"],
               "heldout_empty_frames_1based": manifest["heldout_empty_frames_1based"],
               "data_manifest_sha256": hashlib.sha256((data/"manifest.json").read_bytes()).hexdigest()}
    write_json(out/"training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def load_model(checkpoint_path, device):
    import torch
    from deep_model import BackgroundUNet
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = BackgroundUNet(**checkpoint["model_spec"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def font(size=19):
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def overlay(image, mask, unknown):
    rgb = np.repeat((image*255).astype(np.uint8)[..., None], 3, 2)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (50, 240, 120), 2)
    rgb[unknown] = (rgb[unknown]*0.60+np.array([190, 70, 230])*0.40).astype(np.uint8)
    return Image.fromarray(rgb)


def make_panel(image, corrected, probability, mask, unknown, index, v2=None):
    panels = [Image.fromarray(np.uint8(image*255)).convert("RGB"),
              Image.fromarray(np.uint8(np.clip(np.abs(image-corrected)*3, 0, 1)*255)).convert("RGB"),
              Image.fromarray(cv2.applyColorMap(np.uint8(probability*255), cv2.COLORMAP_TURBO)[..., ::-1]),
              overlay(image, mask, unknown)]
    titles = [f"第{index}帧 原图", "双极性变化 ×3（非去白结果）", "V3喷雾概率（未校准）", "V3边界；紫色=未知提示"]
    if v2 is not None:
        panels.append(overlay(image, v2, np.zeros_like(mask)))
        titles.append("V2包络（同一原图）")
    canvas = Image.new("RGB", (360*len(panels), 392), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (im, title) in enumerate(zip(panels, titles)):
        canvas.paste(im.resize((360, 360)), (i*360, 32))
        draw.text((i*360+5, 5), title, fill="black", font=font(15))
    return canvas


def predict(args):
    torch, device = runtime(args.device)
    from deep_model import DEEP_VERSION
    from spray_recognition import auto_geometry, discover_sequence
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    folder = args.input or cfg["source_dir"]
    output = Path(args.output or cfg["output_dir"])
    if output.resolve() == Path(folder).resolve():
        raise ValueError("输出不能与原图文件夹相同。")
    output.mkdir(parents=True, exist_ok=True)
    paths, excluded = discover_sequence(folder)
    count = int(cfg.get("background_frames", 5))
    if count < 2 or count >= len(paths) or args.size < 32:
        raise ValueError("背景帧至少2帧且须少于总帧数，处理尺寸至少32。")
    if not all(cfg.get(k) is not None for k in ("source_xy", "chamber_xy", "chamber_axes_px")):
        cfg.update(auto_geometry(paths, count))
    model, checkpoint = load_model(args.checkpoint, device)
    fit_count = max(1, count-min(2, count-1))
    background = np.median(np.stack([read_gray(p) for p in paths[:fit_count]]), 0)
    bounds, roi, static = crop_geometry(background.shape, cfg)
    x0, y0, x1, y1 = bounds
    small_roi = crop_resize(roi, bounds, args.size, True) > 0.5
    sx, sy = cfg["source_xy"]
    small_source = ((sx-x0)/(x1-x0)*args.size, (sy-y0)/(y1-y0)*args.size)
    noise_stack = np.stack([read_gray(p)-background for p in paths[:fit_count]])
    noise = max(2/255, float(np.quantile(np.abs(noise_stack[:, roi]), 0.995))*2)
    for directory in ("probability", "edge_probability", "model_masks", "unknown_masks", "overlays", "signal_on_black", "iterations"):
        (output/directory).mkdir(exist_ok=True)
    rows, previews, empty_checks = [], [], []
    start = time.perf_counter()
    selected = set(cfg.get("selected_frames", [1, 6, 10, 20, 50, 70])) | {4, 5, len(paths)}
    model_counts = []
    for i, path in enumerate(paths, 1):
        image = read_gray(path)
        if image.shape != background.shape:
            raise ValueError("同一序列分辨率不一致。")
        corrected, photometry = correct_photometry(image, background, static)
        im, bg = crop_resize(image, bounds, args.size), crop_resize(corrected, bounds, args.size)
        inp = torch.from_numpy(channels(im, bg)[None]).to(device)
        with torch.inference_mode():
            stages = [z.sigmoid().float().cpu().numpy()[0] for z in model(inp)]
        p, edge, ambiguity = stages[-1]
        raw_mask = (p > 0.5) & small_roi
        model_counts.append(int(raw_mask.sum()))
        mask = source_connected(raw_mask, small_source, max(4, args.size*0.065))
        delta = np.abs(im-bg)
        observed = (delta > max(noise, 0.035)) & small_roi
        valid_observation = int(observed.sum()) >= max(12, int(small_roi.sum()*0.0005))
        # 无观测时清空确定测量，模型原始概率仍然保存供反事实检查。
        unknown = (((p > 0.2) & (p < 0.8)) | ((ambiguity > 0.5) & (p > 0.25))) & small_roi
        # 几何筛除与真实背景不是同一概念，保留被拒绝模型区域供复核。
        unknown |= raw_mask & ~mask
        if not valid_observation:
            unknown |= mask
            mask[:] = False
        current = (p-stages[0][0]) if len(stages) > 1 else np.zeros_like(p)
        row = {"frame": i, "name": path.name, "measurement_valid": valid_observation,
               "model_pixels_before_gate": int(raw_mask.sum()), "source_connected_pixels": int(mask.sum()),
               "unknown_pixels": int(unknown.sum()), "observed_pixels": int(observed.sum()),
               "refinement_mean_absolute_change": float(np.abs(current[small_roi]).mean()),
               "gain": photometry["gain"], "offset": photometry["offset"]}
        rows.append(row)
        if fit_count < i <= count:
            empty_checks.append({"frame": i, "raw_model_false_positive_fraction": float(raw_mask.sum()/small_roi.sum()),
                                 "final_mask_fraction": float(mask.sum()/small_roi.sum()), "independent_experiment": False})
        for name, array in (("probability", p), ("edge_probability", edge), ("model_masks", mask), ("unknown_masks", unknown)):
            full = np.zeros_like(background)
            full[y0:y1, x0:x1] = cv2.resize(array.astype(np.float32), (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR if "probability" in name else cv2.INTER_NEAREST)
            Image.fromarray(np.uint8(np.clip(full, 0, 1)*255)).save(output/name/f"{i:06d}.png")
        # 只显示观测变化，不能把p乘原图称为真实恢复的无背景喷雾。
        signal = np.clip(np.abs(image-corrected)*3, 0, 1)
        full_mask = np.asarray(Image.open(output/"model_masks"/f"{i:06d}.png")) > 0
        signal *= full_mask
        Image.fromarray(np.uint8(signal*255)).save(output/"signal_on_black"/f"{i:06d}.png")
        if i in selected:
            for stage_index, stage in enumerate(stages):
                Image.fromarray(np.uint8(stage[0]*255)).save(output/"iterations"/f"{i:06d}_stage{stage_index}.png")
            v2 = None
            if args.v2:
                candidate = Path(args.v2)/"envelope_masks"/f"{i:06d}.png"
                if candidate.exists():
                    v2 = crop_resize(np.asarray(Image.open(candidate).convert("L")) > 0, bounds, args.size, True) > 0.5
            panel = make_panel(im, bg, p, mask, unknown, i, v2)
            panel.save(output/"overlays"/f"{i:06d}.jpg", quality=93)
            if i in {6, 20, len(paths)}:
                previews.append(panel)
        if i % 10 == 0 or i == len(paths):
            print(f"V3推理 {i}/{len(paths)}", flush=True)
    # 精确背景反事实不经过观测门控，直接检查网络是否会凭空画喷雾。
    b = crop_resize(background, bounds, args.size)
    with torch.inference_mode():
        null_p = model(torch.from_numpy(channels(b, b)[None]).to(device))[-1][0, 0].sigmoid().cpu().numpy()
    null_fraction = float(((null_p > 0.5) & small_roi).sum()/small_roi.sum())
    if previews:
        canvas = Image.new("RGB", (previews[0].width, sum(p.height for p in previews)), "white")
        offset = 0
        for panel in previews:
            canvas.paste(panel, (0, offset)); offset += panel.height
        canvas.save(output/"V3整段效果对照.jpg", quality=94)
    with (output/"frame_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"version": DEEP_VERSION, "frames": len(paths), "background_reference_frames": fit_count,
               "heldout_empty_checks": empty_checks, "exact_background_raw_false_positive_fraction": null_fraction,
               "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
               "checkpoint_training_kind": checkpoint["training_kind"], "checkpoint_step": checkpoint["step"],
               "model_spec": checkpoint["model_spec"], "inference_size": args.size,
               "real_boundary_accuracy": None, "uncertainty_calibrated": False,
               "timing_seconds_including_io": time.perf_counter()-start,
               "measurement_definition": "模型喷雾区域；不是已标定的液滴边界或V2外包络，不导出物理面积/速度结论",
               "physics_constraints": ["标定视窗", "喷嘴连通分量", "无观测测量降级"],
               "not_implemented": ["真实边界监督训练", "时序状态网络", "定量喷速约束", "物理透明度反演"],
               "excluded_attachment_count": len(excluded)}
    write_json(output/"deep_run_summary.json", summary)
    write_json(output/"运行配置.json", cfg)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def evaluate(args):
    if args.batches < 1:
        raise ValueError("评估批次数必须为正。")
    torch, device = runtime(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    data = Path(args.data)
    manifest = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    synth = SpraySynthesizer(data/"backgrounds.npz", manifest["source_normalized"], args.seed, True)
    scores = evaluate_synthetic(model, synth, device, batches=args.batches)
    scores.update(seed=args.seed, checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  split="程序生成的新背景与新喷束；未用于训练或best选择的合成测试种子",
                  real_boundary_accuracy=None, training_kind=checkpoint["training_kind"])
    write_json(args.output, scores)
    # 合成图明确保留真值列，不把示意数据冒充真实实验处理效果。
    viz = SpraySynthesizer(data/"backgrounds.npz", manifest["source_normalized"], args.seed, True)
    canvas = Image.new("RGB", (1280, 3*350), "white")
    draw = ImageDraw.Draw(canvas)
    with torch.inference_mode():
        for row in range(3):
            x, y = viz.sample()
            p = model(torch.from_numpy(x[None]).to(device))[-1][0, 0].sigmoid().cpu().numpy()
            arrays = [x[0], x[1], y[0]]
            pictures = [Image.fromarray(np.uint8(a*255)).convert("RGB") for a in arrays]
            pictures.append(overlay(x[0], p > 0.5, (p > 0.2) & (p < 0.8)))
            for col, (title, picture) in enumerate(zip(["合成输入", "程序生成背景", "合成真值", "模型预测边界"], pictures)):
                draw.text((col*320+4, row*350+5), title, font=font(16), fill="black")
                canvas.paste(picture.resize((320, 320)), (col*320, row*350+30))
    canvas.save(Path(args.output).parent/"合成测试可视化.jpg", quality=92)
    print(json.dumps(scores, ensure_ascii=False), flush=True)


def evaluate_labels(args):
    torch, device = runtime(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    manifest = json.loads((Path(args.data)/"manifest.json").read_text(encoding="utf-8"))
    samples = load_annotations(args.labels, manifest, checkpoint["size"])
    if not samples:
        raise ValueError("没有已标确定像素的真实标注，不生成虚构精度。")
    records = []
    with torch.inference_mode():
        for x, y, valid, name in samples:
            p = model(torch.from_numpy(x[None]).to(device))[-1][0, 0].sigmoid().cpu().numpy()
            record = {"file": name, **score_batch(p, y[0], valid[0] > 0.5),
                      "labeled_fraction_of_crop": float(valid.mean()),
                      "boundary_f1_tolerance_2_processing_pixels": boundary_f1(p, y[0], valid[0] > 0.5),
                      "used_for_real_supervision": name in checkpoint.get("annotation_files", [])}
            records.append(record)
    write_json(args.output, {"records": records, "independent_experiment": False,
                            "note": "仅在确定标注像素评分；稀疏标注不代表整幅边界精度，训练用标注不是留出测试。"})


def predict_batch(args):
    from spray_recognition import discover_sequence
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    folder = Path(args.input or cfg["source_dir"])
    output = Path(args.output or cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for child in sorted(folder.iterdir()):
        if not child.is_dir() or child.resolve() == output.resolve():
            continue
        try:
            discover_sequence(child)
        except ValueError:
            records.append({"folder": child.name, "status": "skipped_no_images"})
            continue
        try:
            dest = output/child.name; dest.mkdir(exist_ok=True)
            child_cfg = {**cfg, "source_dir": str(child), "output_dir": str(dest),
                         "source_xy": None, "chamber_xy": None, "chamber_axes_px": None}
            config_path = dest/"工具运行设置.json"
            write_json(config_path, child_cfg)
            options = argparse.Namespace(**vars(args))
            options.config, options.input, options.output = str(config_path), str(child), str(dest)
            summary = predict(options)
            records.append({"folder": child.name, "status": "ok", "frames": summary["frames"]})
        except Exception as exc:
            records.append({"folder": child.name, "status": "error", "message": str(exc)})
    write_json(output/"deep_batch_summary.json", records)
    if not any(r["status"] == "ok" for r in records):
        raise RuntimeError("未成功处理任何子序列，请检查批处理结果。")


def main():
    parser = argparse.ArgumentParser(description="喷雾背景参考深度学习试验工具")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="生成背景训练池与帧清单")
    p.add_argument("--config", required=True); p.add_argument("--input")
    p.add_argument("--output", required=True); p.add_argument("--size", type=int, default=320)
    p = sub.add_parser("train", help="合成预训练，可混合显式复核标注")
    p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=1000); p.add_argument("--batch", type=int, default=4)
    p.add_argument("--width", type=int, default=16); p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.001); p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--eval-every", type=int, default=100); p.add_argument("--resume"); p.add_argument("--labels")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p = sub.add_parser("predict", help="用已训练权重处理完整序列")
    p.add_argument("--config", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--input"); p.add_argument("--output"); p.add_argument("--size", type=int, default=512)
    p.add_argument("--v2", help="可选V2结果目录用于同图对照")
    p.add_argument("--batch", action="store_true", help="对子文件夹分别重估背景和几何")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p = sub.add_parser("evaluate", help="独立合成随机种子评估，不冒充真实精度")
    p.add_argument("--data", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True); p.add_argument("--batches", type=int, default=16)
    p.add_argument("--seed", type=int, default=912730)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p = sub.add_parser("evaluate-labels", help="对显式真实标注评分，报告覆盖率与训练重叠")
    p.add_argument("--data", required=True); p.add_argument("--labels", required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.size < 32:
            raise ValueError("训练尺寸至少32像素。")
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        info = prepare(args.input or cfg["source_dir"], cfg, args.output, args.size)
        print(f"已准备{len(info['files'])}帧，背景参考{info['reference_frames_1based']}，留出空背景{info['heldout_empty_frames_1based']}")
    elif args.command == "train":
        train(args)
    elif args.command == "predict":
        predict_batch(args) if args.batch else predict(args)
    elif args.command == "evaluate-labels":
        evaluate_labels(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
