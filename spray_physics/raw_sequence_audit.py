"""原始序列路线试验：比较固定背景差分、三帧时序纹理及证据融合。

这里只输出候选区域，不恢复所谓真实灰底，也不把候选轮廓当作测量真值。
三帧时序标准差再取空间标准差的思路参考 ECN；本模块为独立实现，
未移植其单喷束最大连通域、填洞、液芯补画或施里伦专用阈值。
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d

from deep_data import correct_photometry, crop_geometry, read_gray, source_connected, write_json
from deep_spray import font
from spray_recognition import auto_geometry, discover_sequence


def temporal_texture(window, size=5):
    """计算总体时间标准差，再计算局部样本空间标准差；输入为灰度级。"""
    time_std = np.std(window, axis=0, ddof=0)
    mean = cv2.boxFilter(time_std, -1, (size, size), borderType=cv2.BORDER_REFLECT)
    square = cv2.boxFilter(time_std**2, -1, (size, size), borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(square-mean**2, 0) * (size*size)/(size*size-1)
    return np.sqrt(variance)


def angular_contrast(difference, source, smooth_degrees=20):
    """移除每一半径上的宽角度趋势，只作为喷束结构特征，不恢复灰底。"""
    if not 0 < smooth_degrees < 90:
        raise ValueError("角向平滑尺度应在0至90度之间。")
    yy, xx = np.indices(difference.shape)
    extent = int(np.ceil(np.hypot(xx-source[0], yy-source[1]).max()))+2
    flags = cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS | cv2.INTER_LINEAR
    polar = cv2.warpPolar(difference.astype(np.float32), (extent, 720), tuple(map(float, source)), extent, flags)
    trend = gaussian_filter1d(polar, smooth_degrees*2, axis=0, mode="wrap")
    residual = polar-trend
    return cv2.warpPolar(residual, (difference.shape[1], difference.shape[0]), tuple(map(float, source)), extent, flags | cv2.WARP_INVERSE_MAP)


def candidate_mask(score, roi, source, reach, scale=1.0, require_source=True):
    """滞后阈值保留强证据相连的弱证据；不填洞、不强制增长。"""
    strong = (score >= scale) & roi
    weak = (score >= scale*0.55) & roi
    linked = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    linked = (linked > 0) & roi
    _, labels = cv2.connectedComponents(linked.astype(np.uint8))
    ids = np.unique(labels[strong]); ids = ids[ids > 0]
    supported = np.isin(labels, ids) if ids.size else np.zeros_like(roi)
    if require_source:
        return source_connected(supported, source, reach=reach) & roi
    # 喷束与源点之间的低对比缺口不能证明远端喷束不存在；结构分支仅滤小孤岛。
    _, components, stats, _ = cv2.connectedComponentsWithStats(supported.astype(np.uint8))
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= 64)
    keep = keep[keep > 0]
    return np.isin(components, keep) & roi


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum()/union) if union else 1.0


def overlay(image, mask, interior, uncertain=None):
    rgb = np.repeat(np.uint8(np.clip(image, 0, 255))[..., None], 3, 2)
    edge = (cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0) & interior
    rgb[edge] = [45, 237, 128]
    if uncertain is not None:
        rgb[edge & uncertain] = [233, 155, 45]
    return Image.fromarray(rgb)


def run(config, output, input_dir=None, background_frames=3, empty_frames=5):
    cfg = dict(config)
    folder = input_dir or cfg.get("source_dir")
    paths, excluded = discover_sequence(folder)
    if not 3 <= background_frames < empty_frames < len(paths):
        raise ValueError("至少三帧建背景，且需留下独立空背景和后续序列。")
    # 切换输入时不静默继承旧几何；此命令没有隐式复用开关。
    if input_dir and cfg.get("source_dir") and Path(input_dir).resolve() != Path(cfg["source_dir"]).resolve():
        cfg.update(auto_geometry(paths, empty_frames))
        cfg.update(illumination_geometry="unspecified", illumination_geometry_confirmed=False)
    if not all(cfg.get(k) is not None for k in ("source_xy", "chamber_xy", "chamber_axes_px")):
        cfg.update(auto_geometry(paths, empty_frames))
    out = Path(output).resolve()
    if out == Path(folder).resolve() or Path(folder).resolve() in out.parents:
        raise ValueError("研究输出须位于原图目录之外，避免下一次运行混入输入。")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("intensity", "texture", "fusion", "directional", "disagreement", "overlays", "directional_overlays"):
        (out/name).mkdir(exist_ok=True)
    base = np.median(np.stack([read_gray(p) for p in paths[:background_frames]]), axis=0)
    bounds, full_roi, static = crop_geometry(base.shape, cfg)
    x0, y0, x1, y1 = bounds
    b = base[y0:y1, x0:x1]*255
    sx, sy = np.array(cfg["source_xy"])-[x0, y0]
    yy, xx = np.indices(b.shape)
    radius = min(cfg["chamber_axes_px"])
    distance = np.hypot(xx-sx, yy-sy)
    # 只排除实际喷孔附近小圆，不将后期内圈大片区域排除。
    roi = full_roi[y0:y1, x0:x1] & (distance > radius*0.06)
    interior = cv2.erode(roi.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    white = roi & (b > 0.65*255)
    inner = roi & (distance >= radius*0.10) & (distance <= radius*0.44)
    normalized, photometry = [], []
    for path in paths:
        im = read_gray(path)
        _, spec = correct_photometry(im, base, static)
        normalized.append(((im[y0:y1, x0:x1]-spec["offset"])/spec["gain"]*255).astype(np.float32))
        photometry.append(spec)
    images = np.stack(normalized)
    # 阈值仅由建模空帧确定；后续空帧只评估，后段喷雾不参与选择阈值。
    blank_difference = np.stack([cv2.GaussianBlur(np.abs(im-b), (0, 0), 0.8) for im in images[:background_frames]])
    blank_texture = np.stack([temporal_texture(images[i-2:i+1]) for i in range(2, background_frames)])
    intensity_high = max(6.0, float(np.quantile(blank_difference[:, roi], 0.999))*1.25)
    texture_high = max(1.5, float(np.quantile(blank_texture[:, roi], 0.999))*1.25)
    blank_directional = np.stack([angular_contrast(im-b, (sx, sy)) for im in images[:background_frames]])
    directional_high = max(8.0, float(np.quantile(blank_directional[:, roi], 0.999))*1.25)
    # 保留阈值选择数据和独立空帧的区别，不报告真实喷雾准确率。
    records, thumbnails, directional_thumbnails, panels, manifest = [], [], [], [], []
    selected = sorted({empty_frames-1, min(12, len(paths)-1), min(23, len(paths)-1), min(49, len(paths)-1), len(paths)-1})
    for index, (path, im) in enumerate(zip(paths, images)):
        window = images[max(0, index-2):index+1]
        if len(window) < 3:
            window = np.concatenate([np.repeat(window[:1], 3-len(window), axis=0), window])
        intensity = cv2.GaussianBlur(np.abs(im-b), (0, 0), 0.8)/intensity_high
        texture = temporal_texture(window)/texture_high
        directional_feature = angular_contrast(im-b, (sx, sy))
        scores = {"intensity": intensity, "texture": texture, "fusion": np.maximum(intensity, texture),
                  "directional": np.maximum(directional_feature, 0)/directional_high}
        masks = {name: candidate_mask(score, roi, (sx, sy), radius*0.10, require_source=name != "directional") for name, score in scores.items()}
        disagreement = masks["intensity"] ^ masks["texture"]
        frame_record = {"frame": index+1, "heldout_empty": background_frames <= index < empty_frames,
                        "methods": {}, "quantitative_measurement_valid": False}
        for name, mask in masks.items():
            low = candidate_mask(scores[name], roi, (sx, sy), radius*0.10, 0.8, require_source=name != "directional")
            high = candidate_mask(scores[name], roi, (sx, sy), radius*0.10, 1.2, require_source=name != "directional")
            frame_record["methods"][name] = {
                "candidate_pixels": int(mask.sum()),
                "inner_candidate_fraction": float(mask[inner].mean()),
                "white_candidate_fraction": float(mask[white].mean()) if white.any() else None,
                "threshold_80_120_iou": iou(low, high)}
            Image.fromarray(np.uint8(mask*255)).save(out/name/f"{index+1:06d}.png")
        Image.fromarray(np.uint8(disagreement*255)).save(out/"disagreement"/f"{index+1:06d}.png")
        display = overlay(im, masks["fusion"], interior, disagreement)
        ImageDraw.Draw(display).text((12, 10), f"第{index+1}帧：原图上的候选边界；橙色为方法分歧，未验收", font=font(18), fill="white", stroke_width=1, stroke_fill="black")
        display.save(out/"overlays"/f"{index+1:06d}.jpg")
        thumbnails.append(display.resize((508, 500)))
        directional_display = overlay(im, masks["directional"], interior)
        ImageDraw.Draw(directional_display).text((12, 10), f"第{index+1}帧：角向喷束结构候选；不代表全部喷雾", font=font(18), fill="white", stroke_width=1, stroke_fill="black")
        directional_display.save(out/"directional_overlays"/f"{index+1:06d}.jpg")
        directional_thumbnails.append(directional_display.resize((508, 500)))
        if index in selected:
            row = Image.new("RGB", (4*508, 548), "#181c22")
            columns = [(Image.fromarray(np.uint8(np.clip(im, 0, 255))).convert("RGB"), "原图：保持观测数据"),
                       (overlay(im, masks["intensity"], interior), "A 固定背景亮度变化"),
                       (overlay(im, masks["texture"], interior), "B 三帧时序纹理"),
                       (overlay(im, masks["fusion"], interior, disagreement), "C 融合候选（橙色为分歧）")]
            for col, (picture, title) in enumerate(columns):
                row.paste(picture.resize((508, 500)), (col*508, 48))
                ImageDraw.Draw(row).text((col*508+10, 10), f"{index+1:02d}帧 {title}", font=font(19), fill="white")
            row.save(out/f"第{index+1:02d}帧_原始证据对照.jpg")
            panels.append(row)
            directional_row = Image.new("RGB", (1524, 558), "#181c22")
            feature_picture = Image.fromarray(np.uint8(np.clip(directional_feature+100, 0, 255))).convert("RGB")
            for col, (picture, title) in enumerate([(columns[0][0], "原图"), (feature_picture, "有符号角向结构特征+100"),
                                                   (directional_display, "喷束结构候选（不用于删像素）")]):
                directional_row.paste(picture.resize((508, 500)), (col*508, 48))
                ImageDraw.Draw(directional_row).text((col*508+10, 10), f"{index+1:02d}帧 {title}", font=font(19), fill="white")
            directional_row.save(out/f"第{index+1:02d}帧_角向结构候选.jpg")
        records.append(frame_record)
        manifest.append({"frame": index+1, "name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "photometry": photometry[index]})
    summary = {"status": "research_candidates_not_accepted", "frames": len(paths), "human_labels": 0,
               "illumination": cfg.get("illumination_geometry", "unspecified"),
               "illumination_confirmed": cfg.get("illumination_geometry_confirmed", False),
               "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "reference_frames": list(range(1, background_frames+1)),
               "heldout_empty_frames": list(range(background_frames+1, empty_frames+1)),
               "intensity_high_gray": intensity_high, "texture_high_gray": texture_high,
               "directional_high_gray": directional_high, "directional_smooth_degrees": 20,
               "directional_polarity": "positive_bright_jet_candidate_only",
               "directional_note": "角向对比会移除轴对称或宽角度喷雾，不能拿此候选掩膜删除原图像素",
               "uses_future_frames": False, "fills_holes": False, "monotone_growth": False,
               "optical_layer_reconstruction": False, "physics_completion": False,
               "quantitative_measurement_valid": False,
               "threshold_iou_note": "只表示阈值扰动稳定性，不是相对真值的准确率",
               "empty_false_positive_pixels": {name: [r["methods"][name]["candidate_pixels"] for r in records[background_frames:empty_frames]] for name in masks},
               "crop_bounds_xyxy": bounds, "roi_pixels": int(roi.sum()), "inner_pixels": int(inner.sum()),
               "excluded_attachment_count": len(excluded)}
    write_json(out/"summary.json", summary)
    write_json(out/"frame_metrics.json", records)
    write_json(out/"input_manifest.json", manifest)
    write_json(out/"geometry.json", {k: cfg[k] for k in ("source_xy", "chamber_xy", "chamber_axes_px")})
    with (out/"frame_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "method", "candidate_pixels", "inner_candidate_fraction", "white_candidate_fraction", "threshold_80_120_iou"])
        writer.writeheader()
        for r in records:
            for name, values in r["methods"].items():
                writer.writerow({"frame": r["frame"], "method": name, **values})
    contact = Image.new("RGB", (2032, len(panels)*548))
    for i, panel in enumerate(panels):
        contact.paste(panel, (0, i*548))
    contact.save(out/"原始序列路线对照总览.jpg")
    thumbnails[0].save(out/"原图候选边界_未验收.gif", save_all=True, append_images=thumbnails[1:], duration=100, loop=0)
    directional_thumbnails[0].save(out/"角向喷束结构候选_未验收.gif", save_all=True, append_images=directional_thumbnails[1:], duration=100, loop=0)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-frames", type=int, default=3)
    parser.add_argument("--empty-frames", type=int, default=5)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    print(json.dumps(run(cfg, args.output, args.input, args.reference_frames, args.empty_frames), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
