"""边界阶段只读取已保存灰底PNG及恢复不确定性，不访问原图。"""

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from deep_data import read_gray, source_connected, write_json
from deep_spray import font


def gray_boundary(folder, output, cfg, uncertainty_folder=None):
    from spray_recognition import discover_sequence
    paths, _ = discover_sequence(folder)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    for name in ("masks", "unknown", "overlays"):
        (out/name).mkdir(exist_ok=True)
    count = int(cfg.get("background_frames", 5))
    fit_count = max(1, count-2)
    empty = np.stack([read_gray(p) for p in paths[:fit_count]])
    base = np.median(empty, 0)
    yy, xx = np.indices(base.shape)
    cx, cy = cfg["chamber_xy"]; ax, ay = cfg["chamber_axes_px"]
    sx, sy = cfg["source_xy"]; radius = min(ax, ay)
    rho = np.hypot((xx-cx)/ax, (yy-cy)/ay)
    roi = (rho < 0.97) & (np.hypot(xx-sx, yy-sy) > radius*0.065)
    noise = np.median(np.abs(empty-base), 0)*1.4826+1/255
    high = np.maximum(float(cfg.get("threshold_gray", 10))/255, noise*4)
    low = np.maximum(high*0.5, noise*2.5)
    frames, records, previous_signal = [], [], np.zeros_like(base)
    for index, path in enumerate(paths, 1):
        image = read_gray(path)
        # 所有图像证据均来自实际落盘灰底图；透明度网络不直接输出此掩膜。
        signal = cv2.GaussianBlur(np.abs(image-base), (0, 0), 0.8)
        strong = (signal > high) & roi
        weak = (signal > low) & roi
        linked = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        _, labels = cv2.connectedComponents(linked)
        supported_ids = np.unique(labels[strong]); supported_ids = supported_ids[supported_ids > 0]
        mask = source_connected(np.isin(labels, supported_ids), (sx, sy), reach=radius*0.14) & roi
        # 时序只标记弱证据，不强制每帧包络单调增长或补全整条喷束。
        uncertain = mask & ((signal < high) | (np.abs(signal-previous_signal) > 0.12))
        if uncertainty_folder:
            uncertain |= (read_gray(Path(uncertainty_folder)/path.name) > 0.5) & mask
        if index <= count:
            # 背景帧本应为空；保留原始误检统计，不强制清空掩膜。
            state = "empty_background_check"
        else:
            state = "estimated_boundary" if mask.any() else "no_supported_boundary"
        canvas = Image.fromarray(np.uint8(np.rint(image*255))).convert("RGB")
        canvas_array = np.array(canvas)
        edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        unknown_edge = edge & (cv2.dilate(uncertain.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        canvas_array[edge] = [48, 231, 130]
        canvas_array[unknown_edge] = [207, 110, 255]
        canvas = Image.fromarray(canvas_array)
        ImageDraw.Draw(canvas).text((16, 12), f"{index:03d} 灰底校正后的边界估计；紫色为低证据", font=font(20), fill="white", stroke_width=1, stroke_fill="black")
        canvas.save(out/"overlays"/path.name)
        frames.append(canvas.resize((640, 400)))
        for name, field in (("masks", mask), ("unknown", uncertain)):
            Image.fromarray(np.uint8(field*255)).save(out/name/path.name)
        records.append({"frame": index, "state": state, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "mask_pixels": int(mask.sum()), "uncertain_pixels": int(uncertain.sum()),
                        "quantitative_measurement_valid": False})
        previous_signal = signal
    frames[0].save(out/"灰底边界动画.gif", save_all=True, append_images=frames[1:], duration=100, loop=0, disposal=2)
    write_json(out/"boundary_inputs.json", records)
    summary = {"frames": len(paths), "raw_images_read_by_boundary_stage": False,
               "input_contract": "saved_gray_scene_png_only", "human_labels": 0,
               "empty_frame_false_positive_pixels": [r["mask_pixels"] for r in records[:count]],
               "quantitative_measurement_valid": False,
               "reason": "恢复与边界均未通过真实标注精度验证；紫色不作为确定测量"}
    write_json(out/"boundary_summary.json", summary)
    return summary
