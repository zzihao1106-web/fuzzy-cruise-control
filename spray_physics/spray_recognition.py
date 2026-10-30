"""用背景证据、喷束增长模型与跨帧运动约束识别喷雾外包络。"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from recognition_core import EvidenceTracker

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_gray(path):
    # Pillow 可以可靠读取含中文的 Windows 路径。
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)


def discover_sequence(folder):
    """按自然顺序读取主分辨率图像组，排除设置照片等不同尺寸附件。"""
    paths = [p for p in Path(folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}]
    groups = {}
    for p in paths:
        try:
            with Image.open(p) as im:
                groups.setdefault(im.size, []).append(p)
        except (OSError, ValueError):
            continue
    if not groups:
        raise ValueError("文件夹内没有可读取的图像。")
    main_size = max(groups, key=lambda size: len(groups[size]))
    def natural(p):
        return [int(v) if v.isdigit() else v.lower() for v in re.split(r"(\d+)", p.name)]
    selected = sorted(groups[main_size], key=natural)
    return selected, [str(p) for size, group in groups.items() if size != main_size for p in group]


def auto_geometry(paths, n_background=5):
    """自动估计圆形视窗及早期变化区域；用户可在界面中一次性修正。"""
    images = np.stack([read_gray(p) for p in paths[:n_background]])
    bg = np.median(images, axis=0)
    h, w = bg.shape
    yy, xx = np.indices(bg.shape)
    central = ((xx - w / 2) / (w * 0.30)) ** 2 + ((yy - h / 2) / (h * 0.35)) ** 2 < 1
    source = [w / 2, h / 2]
    for path in paths[n_background:]:
        d = read_gray(path) - bg
        weight = np.maximum(np.abs(d) - 15, 0) * central
        if (weight > 0).sum() > max(40, h * w * 0.0004):
            source = [float((weight * xx).sum() / weight.sum()), float((weight * yy).sum() / weight.sum())]
            break
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(bg.astype(np.uint8), (7, 7), 2), cv2.HOUGH_GRADIENT,
        dp=1.5, minDist=min(h, w) * 0.2, param1=45, param2=30,
        minRadius=int(min(h, w) * 0.34), maxRadius=int(min(h, w) * 0.49),
    )
    warnings = []
    if circles is not None:
        x, y, r = min(circles[0], key=lambda c: (c[0] - source[0]) ** 2 + (c[1] - source[1]) ** 2)
        chamber, axes = [float(x), float(y)], [float(r), float(r)]
    else:
        chamber, axes = [w / 2, h / 2], [min(h, w) * 0.42] * 2
        warnings.append("未自动找到可靠圆形视窗，请在标定界面点击圆周三点。")
    return {"source_xy": source, "chamber_xy": chamber, "chamber_axes_px": axes, "geometry_warnings": warnings}


def polar_sample(image, xy, n_angles, n_radius):
    return cv2.warpPolar(
        image.astype(np.float32), (n_radius, n_angles), tuple(xy), n_radius,
        cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS | cv2.INTER_LINEAR,
    )


def estimate_jet_directions(residuals, xy, n_angles, n_radius, onset):
    # 只用早期高对比度的近喷嘴区域估计方向，不规定喷孔数量。
    signal = np.zeros(n_angles, np.float32)
    for d in residuals[onset + 2:onset + 7]:
        lo, hi = max(5, int(n_radius * 0.125)), max(10, int(n_radius * 0.31))
        signal += polar_sample(np.maximum(d, 0), xy, n_angles, n_radius)[:, lo:hi].mean(1)
    signal = gaussian_filter1d(signal, 1.0, mode="wrap")
    triple = np.tile(signal, 3)
    peaks, _ = find_peaks(triple, distance=max(8, n_angles // 20), prominence=max(2, signal.max() * 0.06))
    peaks = peaks[(peaks >= n_angles) & (peaks < 2 * n_angles)] - n_angles
    if len(peaks) < 1:
        raise ValueError("无法可靠估计喷束方向，请检查喷嘴中心或起喷前背景帧。")
    return peaks, signal


def fit_growth(profiles, wall_radius, onset, last_frame):
    # 用早期可见喷束拟合像素域幂律；指数允许变化，避免强迫采用平方根律。
    n_frames, n_jets, n_radius = profiles.shape
    rscale = n_radius / 360
    measured = np.zeros((n_frames, n_jets))
    for t in range(onset, n_frames):
        for j in range(n_jets):
            p = gaussian_filter1d(profiles[t, j], 2)
            valid = (p > 16) & (np.arange(n_radius) < wall_radius[j] - 8*rscale)
            valid[:max(3, int(25*rscale))] = False
            runs = np.convolve(valid.astype(float), np.ones(5), "same") >= 4
            inds = np.flatnonzero(runs)
            if len(inds):
                measured[t, j] = inds[-1]
    models, predictions = [], []
    for j in range(n_jets):
        inds = np.arange(onset + 1, min(last_frame, n_frames))
        good = (measured[inds, j] > 35*rscale) & (measured[inds, j] < wall_radius[j] * 0.88)
        tt = inds[good].astype(float)
        yy = measured[inds[good], j]
        if len(tt) < 4:
            raise ValueError(f"第 {j + 1} 条喷束的可靠早期观测不足。")

        def residual(p):
            amp, exponent, t0 = p
            return (amp * np.maximum(tt - t0, 0.05) ** exponent - yy) / 8

        fit = least_squares(
            residual, [40*rscale, 0.8, onset - 0.5],
            bounds=([0.1, 0.4, onset - 2], [150*rscale, 1.6, onset + 0.8]), loss="soft_l1",
        )
        amp, exponent, t0 = fit.x
        pred = amp * np.maximum(np.arange(n_frames) - t0, 0) ** exponent
        pred[:onset] = 0
        predictions.append(np.minimum(pred, wall_radius[j]))
        models.append({"jet": j + 1, "amplitude_px_frame_power": float(amp), "exponent": float(exponent),
                       "onset_index0": float(t0), "fit_samples": len(tt),
                       "fit_rmse_px": float(np.sqrt(np.mean((residual(fit.x) * 8) ** 2)))})
    return measured, np.stack(predictions, axis=1), models


def optimize_boundary(polar_residual, polar_threshold, wall, prediction, previous, expected_step,
                      max_step, cfg, use_physics=True, evidence_weight=None):
    n_angles, n_radius = polar_residual.shape
    rscale = n_radius / 360
    rr = np.arange(n_radius)[None, :]
    # 内外二分类的累计代价。内侧数据不要求所有像素都是液相，结果定义为外包络。
    local = np.clip((polar_threshold - polar_residual) / 7, -3, 3)
    local[:, :max(3, int(24*rscale))] = 0
    costs = np.cumsum(local, axis=1) / n_radius
    costs[rr > wall[:, None]] = 1e6
    if not use_physics:
        return costs.argmin(1).astype(float)
    width = 22*rscale + 0.15 * prediction
    prior_weight = np.ones(n_angles) if evidence_weight is None else evidence_weight
    costs += cfg["physics_weight"] * prior_weight[:, None] * ((rr - prediction[:, None]) / width[:, None]) ** 2
    # 预期增量来自增长模型；到达视窗边界后自然趋于零，不强制持续增大。
    target = previous + expected_step
    costs += cfg["temporal_weight"] * ((rr - target[:, None]) / (24*rscale)) ** 2
    costs[rr > (previous + max_step)[:, None]] = 1e6
    costs[rr < np.maximum(previous - max_step, 0)[:, None]] = 1e6
    boundary = costs.argmin(1).astype(float)
    for _ in range(3):
        neighbor = (np.roll(boundary, 1) + np.roll(boundary, -1)) / 2
        adjusted = costs + cfg["angular_weight"] * ((rr - neighbor[:, None]) / (18*rscale)) ** 2
        boundary = adjusted.argmin(1).astype(float)
    return boundary


def boundary_to_mask(boundary, radius, angle_bin, roi, min_radius=24):
    return (radius <= boundary[angle_bin]) & roi & (boundary[angle_bin] > min_radius)


def contour_points(boundary, source_xy):
    angle = np.arange(len(boundary)) * 2 * np.pi / len(boundary)
    return np.round(np.c_[source_xy[0] + boundary * np.cos(angle),
                          source_xy[1] + boundary * np.sin(angle)]).astype(np.int32)


def main(config_path, overrides=None, frame_observer=None):
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg.update(overrides or {})
    if cfg["fps"] <= 0 or cfg["background_frames"] < 2 or cfg["threshold_gray"] <= 0:
        raise ValueError("帧率和阈值必须大于0，背景帧数至少为2。")
    if cfg.get("known_diameter_mm") is not None and cfg["known_diameter_mm"] <= 0:
        raise ValueError("已知标定直径必须大于0，或留空。")
    root = Path(cfg["source_dir"])
    out = (config_path.parent / cfg["output_dir"]).resolve()
    if out == root.resolve():
        raise ValueError("输出目录不能与输入图像目录相同。")
    for directory in [out, out / "envelope_masks", out / "evidence_masks", out / "inferred_masks", out / "unknown_masks", out / "model_added_masks", out / "background_removed", out / "overlays"]:
        directory.mkdir(parents=True, exist_ok=True)
    paths, excluded = discover_sequence(root)
    if len(paths) < cfg["background_frames"] + 10:
        raise ValueError("图像数量不足，或输入文件名不匹配。")
    images = np.stack([read_gray(p) for p in paths])
    n_frames, height, width = images.shape
    n_bg = cfg["background_frames"]
    if not cfg.get("source_xy") or not cfg.get("chamber_xy") or not cfg.get("chamber_axes_px"):
        geometry = auto_geometry(paths, n_bg)
        for key, value in geometry.items():
            if not cfg.get(key):
                cfg[key] = value
        cfg["calibration_source"] = "自动估计视窗；请检查标定与视窗.png，必要时一次修正"
        for warning in geometry["geometry_warnings"]:
            print("标定提示：" + warning, flush=True)
    if cfg.get("known_diameter_mm"):
        cfg["mm_per_pixel"] = cfg["known_diameter_mm"] / (2 * np.sqrt(np.prod(cfg["chamber_axes_px"])))
    bg = np.median(images[:n_bg], axis=0)
    noise = 1.4826 * np.median(np.abs(images[:n_bg] - bg), axis=0)
    noise = cv2.GaussianBlur(np.maximum(noise, cfg["noise_floor_gray"]), (0, 0), 1.5)
    threshold = np.maximum(cfg["threshold_gray"], 3.5 * noise)
    yy, xx = np.indices((height, width))
    sx, sy = cfg["source_xy"]
    cx, cy = cfg["chamber_xy"]
    ax, ay = cfg["chamber_axes_px"]
    if ax <= 0 or ay <= 0 or ((sx-cx)/ax)**2 + ((sy-cy)/ay)**2 >= 1:
        raise ValueError("视窗半径必须为正，喷嘴必须位于所选视窗内。")
    roi = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 < 1
    reference_roi = ((xx - cx) / (ax + 65)) ** 2 + ((yy - cy) / (ay + 65)) ** 2 > 1
    if reference_roi.sum() < 100:
        reference_roi = (xx < max(2, width//30)) | (xx >= width-max(2, width//30)) | (yy < max(2, height//30)) | (yy >= height-max(2, height//30))
    drift = np.median((images - bg)[:, reference_roi], axis=1)
    residuals = images - bg - drift[:, None, None]
    residuals = np.stack([cv2.GaussianBlur(d, (0, 0), 1.1) for d in residuals])
    radius = np.hypot(xx - sx, yy - sy)
    n_angles = cfg["angular_bins"]
    n_radius = cfg.get("radial_bins") or int(max(ax, ay) + np.hypot(sx - cx, sy - cy) + 10)
    rscale = n_radius / 360
    near_radius = max(3, int(24*rscale))
    angle_bin = ((np.arctan2(yy - sy, xx - sx) % (2 * np.pi)) * n_angles / (2 * np.pi)).astype(int)
    center_roi = radius < 100*rscale
    polarity = cfg.get("signal_polarity", "auto")
    if polarity == "auto":
        activity_limit = max(50, 400*rscale*rscale)
        changed = ((np.abs(residuals) > threshold) & center_roi).sum(axis=(1, 2))
        first_changed = np.flatnonzero((np.arange(n_frames) >= n_bg) & (changed > activity_limit))
        first = int(first_changed[0]) if len(first_changed) else n_bg
        early = residuals[first:min(first+5, n_frames), center_roi]
        pos_energy = float(np.maximum(early-10, 0).sum())
        neg_energy = float(np.maximum(-early-10, 0).sum())
        polarity = "dark" if neg_energy > pos_energy*1.25 else "bright"
    if polarity not in {"bright", "dark"}:
        raise ValueError("信号极性必须是auto、bright或dark。")
    if polarity == "dark":
        residuals = -residuals
    print("图像变化极性：" + ("暗喷雾" if polarity == "dark" else "亮喷雾"), flush=True)
    activity = ((residuals > threshold) & center_roi).sum(axis=(1, 2))
    active = np.flatnonzero((np.arange(n_frames) >= n_bg) & (activity > max(50, 400*rscale*rscale)))
    if not len(active):
        raise ValueError("未检测到起喷。")
    onset = int(active[0])
    if n_frames-onset < 9:
        raise ValueError("起喷后的图像不足9帧，无法可靠估计发展模型，请增加序列长度。")
    print(f"检测到可见起喷：第 {onset + 1} 帧", flush=True)
    polar_roi = polar_sample(roi.astype(float), cfg["source_xy"], n_angles, n_radius)
    wall = np.maximum((polar_roi > 0.95).sum(1) - 2, 0)
    polars = np.stack([polar_sample(d, cfg["source_xy"], n_angles, n_radius) for d in residuals])
    polar_bg = polar_sample(bg, cfg["source_xy"], n_angles, n_radius)
    glare_fraction = (polar_bg[:, int(n_radius * 0.3):int(n_radius * 0.85)] >= 130).mean(1)
    evidence_weight = 1 + 6 * np.sqrt(glare_fraction)
    polar_threshold = polar_sample(threshold, cfg["source_xy"], n_angles, n_radius)
    peaks, angular_signal = estimate_jet_directions(residuals, cfg["source_xy"], n_angles, n_radius, onset)
    distances = np.abs((np.arange(n_angles)[:, None] - peaks[None, :] + n_angles / 2) % n_angles - n_angles / 2)
    nearest = distances.argmin(1)
    profiles = np.stack([np.mean(polars[:, (p + np.arange(-3, 4)) % n_angles], axis=1) for p in peaks], axis=1)
    fit_end = min(n_frames, onset + cfg.get("growth_fit_frames", 13))
    try:
        observed, predicted_jets, models = fit_growth(profiles, wall[peaks], onset, fit_end)
    except ValueError as exc:
        print(f"增长模型不可用，降级为图像证据识别：{exc}", flush=True)
        observed = np.zeros((n_frames, len(peaks)))
        for t in range(onset, n_frames):
            for j in range(len(peaks)):
                positions = np.flatnonzero((profiles[t, j] > 16) & (np.arange(n_radius) < wall[peaks[j]]))
                observed[t, j] = positions[-1] if len(positions) else 0
        predicted_jets = np.zeros_like(observed)
        models = [{"jet": j+1, "amplitude_px_frame_power": 0., "exponent": 1., "onset_index0": float(onset),
                   "fit_samples": 0, "fit_rmse_px": None} for j in range(len(peaks))]
    reliable_models = [m for k, m in enumerate(models) if m["fit_rmse_px"] is not None and glare_fraction[peaks[k]] < 0.35 and m["fit_rmse_px"] < 6 and 0.41 < m["exponent"] < 1.59]
    if reliable_models:
        common_exponent = float(np.median([m["exponent"] for m in reliable_models]))
        common_t0 = float(np.median([m["onset_index0"] for m in reliable_models]))
        for j, model in enumerate(models):
            borrowed = model["fit_rmse_px"] is None or glare_fraction[peaks[j]] >= 0.35 or model["fit_rmse_px"] >= 6 or model["exponent"] <= 0.41
            model["borrowed_clear_jet_growth_shape"] = bool(borrowed)
            if borrowed:
                indices = np.arange(onset + 1, min(onset + 5, n_frames))
                values = observed[indices, j]
                valid = values > 30*rscale
                amp = float(np.median(values[valid] / np.maximum(indices[valid] - common_t0, 0.1) ** common_exponent)) if valid.any() else float(np.median([m["amplitude_px_frame_power"] for m in reliable_models]))
                pred = amp * np.maximum(np.arange(n_frames) - common_t0, 0) ** common_exponent
                pred[:onset] = 0
                predicted_jets[:, j] = np.minimum(pred, wall[peaks[j]])
                model["prior_amplitude"] = amp
                model["prior_exponent"] = common_exponent
                model["prior_onset_index0"] = common_t0
    for model in models:
        model.setdefault("borrowed_clear_jet_growth_shape", False)
        model.setdefault("prior_amplitude", model["amplitude_px_frame_power"])
        model.setdefault("prior_exponent", model["exponent"])
        model.setdefault("prior_onset_index0", model["onset_index0"])
    raw_boundaries = np.zeros((n_frames, n_angles))
    for t in range(onset, n_frames):
        raw_boundaries[t] = optimize_boundary(polars[t], polar_threshold, wall, wall * 0, wall * 0, wall * 0, 50, cfg, False)
    # 用早期非反光角度估计角向轮廓，保留喷束之间的凹陷。
    shape_samples = []
    for t in range(onset + 3, min(onset + 9, n_frames)):
        denom = np.maximum(observed[t, nearest], 40*rscale)
        shape_samples.append(np.clip(raw_boundaries[t] / denom, 0, 1.15))
    shape = gaussian_filter1d(np.median(shape_samples, axis=0), 1, mode="wrap")
    prediction = np.minimum(predicted_jets[:, nearest] * shape[None], wall[None])
    # V2不再按帧号将形状压平到视窗，模型只提出局部提议。
    jet_strength = np.array([0. if m["fit_rmse_px"] is None else 1/(1+(m["fit_rmse_px"]/6)**2) for m in models])
    jet_strength *= np.array([0.6 if m["borrowed_clear_jet_growth_shape"] else 1. for m in models])
    model_strength = jet_strength[nearest]
    observed_steps = np.diff(observed[onset:fit_end], axis=0)
    positive_steps = observed_steps[(observed_steps > 0) & (observed_steps < 80*rscale)]
    estimate = np.percentile(positive_steps, 98)*1.3 if len(positive_steps) else 40*rscale
    max_step = float(np.clip(estimate, 30*rscale, 65*rscale))
    speed_bound_mode = "早期可见前沿增量的98分位数乘1.3，单位px/帧；不是已标定物理速度"
    if cfg.get("mm_per_pixel") and cfg.get("liquid_density_kg_m3") and cfg.get("injection_pressure_confirmed") and cfg.get("velocity_coefficient"):
        delta_pa = cfg["injection_pressure_MPa_label"] * 1e6 - cfg["ambient_pressure_bar"] * 1e5
        if delta_pa <= 0:
            raise ValueError("喷射压力必须大于环境压力。")
        if cfg["liquid_density_kg_m3"] <= 0:
            raise ValueError("液体密度必须大于0。")
        if not 0 < cfg["velocity_coefficient"] <= 1:
            raise ValueError("速度系数Cv必须在0到1之间。")
        u0 = cfg["velocity_coefficient"] * np.sqrt(2 * delta_pa / cfg["liquid_density_kg_m3"])
        max_step = min(max_step, u0 * 1000 / cfg["fps"] / cfg["mm_per_pixel"])
        speed_bound_mode = "数据增量上界与伯努利出口速度上界共同约束"
    print(f"检测到 {len(peaks)} 条喷束方向；前沿增量上界 {max_step:.2f} px/帧", flush=True)
    boundaries = np.zeros_like(raw_boundaries)
    tracker = EvidenceTracker(roi, cfg["source_xy"], threshold, n_angles, n_radius, max_step, cfg)
    statuses = []
    stats, contour_rows, previews = [], [], []
    selected = set(cfg["selected_frames"]) | {1, onset + 1, n_frames}
    glare = (bg >= 130) & roi
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    gif_frames = []
    for t in range(n_frames):
        previous_tracker = copy.copy(tracker) if frame_observer else None
        recognized = tracker.process(residuals[t], wall, prediction[t], model_strength, glare_fraction, before_onset=t<onset)
        if frame_observer:
            frame_observer(t, recognized, {"tracker_before": previous_tracker, "raw": images[t], "background": bg,
                                           "roi": roi, "wall": wall, "prediction": prediction[t],
                                           "model_strength": model_strength, "glare_fraction": glare_fraction,
                                           "config": cfg, "frame_count": n_frames, "signal_polarity": polarity})
        boundary = recognized.boundary
        boundaries[t] = boundary
        statuses.append(recognized.status)
        envelope = recognized.envelope
        candidates = (residuals[t] > threshold) & roi
        evidence, inferred = recognized.evidence, recognized.inferred
        classification = np.zeros((height, width, 3), np.uint8)
        classification[evidence] = (30, 220, 120)
        classification[inferred] = (255, 150, 30)
        classification[recognized.unknown] = (170, 110, 255)
        clean = np.clip(residuals[t], 0, 150) / 150 * 255
        clean *= envelope
        if t < onset:
            clean[:] = 0
        reliable_angle = recognized.reliable
        points = contour_points(boundary, cfg["source_xy"])
        overlay = cv2.cvtColor(images[t].astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if recognized.status == "observed":
            for k in range(n_angles):
                color = (25, 230, 120) if reliable_angle[k] else (255, 170, 30)
                cv2.line(overlay, tuple(points[k]), tuple(points[(k + 1) % n_angles]), color, 2, cv2.LINE_AA)
        elif recognized.status.startswith("unknown"):
            predicted_points = contour_points(recognized.prediction, cfg["source_xy"])
            for k in range(0, n_angles-1, 4):
                cv2.line(overlay, tuple(predicted_points[k]), tuple(predicted_points[k+1]), (170, 110, 255), 2)
        cv2.drawMarker(overlay, (round(sx), round(sy)), (80, 180, 255), cv2.MARKER_CROSS, 15, 1)
        stem = f"{t+1:06d}"
        Image.fromarray((envelope * 255).astype(np.uint8)).save(out / "envelope_masks" / f"{stem}.png")
        Image.fromarray((evidence * 255).astype(np.uint8)).save(out / "evidence_masks" / f"{stem}.png")
        Image.fromarray((inferred * 255).astype(np.uint8)).save(out / "inferred_masks" / f"{stem}.png")
        Image.fromarray((recognized.unknown * 255).astype(np.uint8)).save(out / "unknown_masks" / f"{stem}.png")
        Image.fromarray((recognized.model_added * 255).astype(np.uint8)).save(out / "model_added_masks" / f"{stem}.png")
        Image.fromarray(clean.astype(np.uint8)).save(out / "background_removed" / f"{stem}.png")
        if t + 1 in selected:
            Image.fromarray(overlay).save(out / "overlays" / f"{stem}.png")
            panels = [images[t].astype(np.uint8), np.clip(np.maximum(residuals[t], 0) * 2, 0, 255).astype(np.uint8), classification, overlay]
            labels = [f"第{t+1}帧 原图", "背景校正差分", "绿=证据 橙=包络 紫=未知", "V2边界 / 紫色虚线仅为预测"]
            panel = Image.new("RGB", (1920, 330), (247, 247, 247))
            draw = ImageDraw.Draw(panel)
            for k, (a, label) in enumerate(zip(panels, labels)):
                im = Image.fromarray(a).convert("RGB").resize((480, 300))
                panel.paste(im, (k * 480, 30))
                draw.text((k * 480 + 8, 1), label, font=font, fill=(20, 20, 20))
            previews.append(panel)
        if t + 1 >= onset:
            pair = Image.new("RGB", (960, 330), "white")
            pair.paste(Image.fromarray(images[t].astype(np.uint8)).resize((480, 300)), (0, 30))
            pair.paste(Image.fromarray(overlay).resize((480, 300)), (480, 30))
            ImageDraw.Draw(pair).text((10, 2), f"第 {t+1:02d} 帧     原图 / 物理约束外包络", font=font, fill="black")
            gif_frames.append(pair)
        vals = boundary[boundary > 0]
        valid = recognized.measurement_valid
        stats.append({"frame": t + 1, "source": paths[t].name,
                      "status": recognized.status, "measurement_valid": valid,
                      "relative_time_ms_assumed": round(t * 1000 / cfg["fps"], 4) if cfg.get("consecutive_frames_assumed", True) else "",
                      "raw_gray_threshold_area_px": int(((images[t] > 130) & roi).sum()),
                      "background_only_area_px": int(candidates.sum()), "envelope_area_px": int(envelope.sum()) if valid else None,
                      "evidence_area_px": int(evidence.sum()), "inferred_area_px": int(inferred.sum()),
                      "unknown_area_px": int(recognized.unknown.sum()), "model_added_area_px": int(recognized.model_added.sum()),
                      "model_conflict_ratio": round(recognized.model_conflict_ratio, 4),
                      "glare_background_only_area_px": int((candidates & glare).sum()),
                      "glare_evidence_area_px": int((evidence & glare).sum()),
                      "front_p95_px": round(float(np.percentile(vals, 95)), 3) if len(vals) else (0 if valid else None),
                      "boundary_high_evidence_ratio": round(float(reliable_angle.mean()), 4) if t >= onset else 0})
        for k, (x, y) in enumerate(points):
            scale = cfg.get("mm_per_pixel")
            contour_rows.append([t + 1, k * 360 / n_angles, int(x) if valid else "", int(y) if valid else "", float(boundary[k]) if valid else "",
                                 float(boundary[k] * scale) if scale and valid else "", int(reliable_angle[k]), recognized.status,
                                 float(recognized.prediction[k])])
        scale = cfg.get("mm_per_pixel")
        stats[-1]["front_p95_mm"] = round(stats[-1]["front_p95_px"] * scale, 4) if scale and valid else ""
        stats[-1]["envelope_area_mm2"] = round(envelope.sum() * scale ** 2, 4) if scale and valid else ""
        stats[-1]["evidence_area_mm2"] = round(evidence.sum() * scale ** 2, 4) if scale else ""
        if (t+1) % 10 == 0 or t+1 == n_frames:
            print(f"处理进度：{t+1}/{n_frames}", flush=True)
    def save_csv(name, rows):
        with (out / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save_csv("frame_metrics.csv", stats)
    save_csv("growth_models.csv", models)
    jet_rows = []
    for j, peak in enumerate(peaks):
        jet_front = boundaries[:, (peak+np.arange(-2, 3)) % n_angles].mean(1)
        # 不跨过未知帧平滑或计算速度，避免补零制造虚假回缩。
        speed = np.full(n_frames, np.nan)
        indices = np.flatnonzero((np.array(statuses) == "observed") & (jet_front > 0))
        for segment in np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1):
            if len(segment) >= 2:
                speed[segment] = np.gradient(gaussian_filter1d(jet_front[segment], 1)) * cfg["fps"] * (cfg.get("mm_per_pixel") or 1)/1000
        for t in range(n_frames):
            jet_rows.append({"frame": t+1, "jet": j+1, "angle_deg_clockwise": float(peak*360/n_angles),
                             "status": statuses[t], "front_px": round(float(jet_front[t]), 4) if statuses[t] == "observed" else "",
                             "front_mm": round(float(jet_front[t]*cfg["mm_per_pixel"]), 4) if cfg.get("mm_per_pixel") and statuses[t] == "observed" else "",
                             "smoothed_projected_speed_m_s_assumed": round(float(speed[t]), 4) if cfg.get("mm_per_pixel") and cfg.get("consecutive_frames_assumed", True) and np.isfinite(speed[t]) else ""})
    save_csv("jet_metrics.csv", jet_rows)
    with (out / "contours.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f); writer.writerow(["frame", "angle_deg_clockwise", "x_px", "y_px", "radius_px", "radius_mm", "high_image_evidence", "status", "predicted_radius_px"]); writer.writerows(contour_rows)
    collage = Image.new("RGB", (1920, len(previews) * 330), "white")
    for k, panel in enumerate(previews):
        collage.paste(panel, (0, k * 330))
    collage.save(out / "方法对照总览.jpg", quality=93)
    previews[-1].save(out / "末帧识别对照.jpg", quality=95)
    gif_frames[0].save(out / "识别过程慢放.gif", save_all=True, append_images=gif_frames[1:], duration=120, loop=0, optimize=True)
    Image.fromarray(bg.astype(np.uint8)).save(out / "background.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    frames = np.arange(1, n_frames + 1)
    axes[0].plot(frames, np.percentile(observed, 95, axis=1), label="Measured jet fronts")
    axes[0].plot(frames, np.percentile(predicted_jets, 95, axis=1), "--", label="Fitted growth prior")
    axes[0].plot(frames, [s["front_p95_px"] for s in stats], label="Constrained envelope")
    axes[0].set(xlabel="Frame", ylabel="Projected radius (px)", title="Front development (pixel domain)")
    axes[0].legend(fontsize=8)
    for field, label in [("raw_gray_threshold_area_px", "Raw threshold"), ("background_only_area_px", "Background subtraction"), ("envelope_area_px", "Constrained envelope"), ("evidence_area_px", "Image-supported foreground")]:
        axes[1].plot(frames, [s[field] for s in stats], label=label)
    axes[1].set(xlabel="Frame", ylabel="Area (px)", title="Method comparison (not accuracy scores)")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "发展曲线与方法对照.png", dpi=180); plt.close(fig)
    actual_step = np.diff(boundaries, axis=0)
    observed_pair = (np.array(statuses[1:]) == "observed") & (np.array(statuses[:-1]) == "observed")
    diagnostic_steps = actual_step[observed_pair]
    jet_fronts = np.stack([boundaries[:, (peak+np.arange(-2, 3)) % n_angles].mean(1) for peak in peaks], axis=1)
    jet_steps = np.diff(jet_fronts, axis=0)
    jet_pairs = observed_pair[:, None] & (jet_fronts[1:] > 0) & (jet_fronts[:-1] > 0)
    negative = stats[:n_bg]
    summary = {
        "version": "2.0.0", "frames": n_frames, "onset_frame": onset + 1, "jet_count_from_image": len(peaks),
        "unknown_frames": [i+1 for i, status in enumerate(statuses) if status.startswith("unknown")],
        "detected_signal_polarity": polarity,
        "jet_angles_deg_clockwise": (peaks * 360 / n_angles).tolist(), "max_front_step_px_per_frame": max_step,
        "speed_bound_mode": speed_bound_mode, "max_actual_step_px": float(np.abs(diagnostic_steps).max()) if diagnostic_steps.size else 0.,
        "step_constraint_violations": int((np.abs(diagnostic_steps) > max_step + 1).sum()),
        "jet_step_exceedances": int(((np.abs(jet_steps) > max_step+1) & jet_pairs).sum()),
        "max_observed_jet_step_px": float(np.max(np.abs(jet_steps)[jet_pairs])) if jet_pairs.any() else 0.,
        "step_note": "强图像证据允许否决模型，超界仅作冲突诊断，不再裁掉观测。",
        "background_gray_threshold_area_mean_px": float(np.mean([s["raw_gray_threshold_area_px"] for s in negative])),
        "background_subtraction_area_mean_px": float(np.mean([s["background_only_area_px"] for s in negative])),
        "background_physics_area_mean_px": float(np.mean([s["envelope_area_px"] for s in negative])),
        "annotation_accuracy_evaluated": False,
        "excluded_different_resolution_images": excluded,
        "output_directory": str(out),
        "note": "背景帧也参与背景估计，因此其零误检只是重建检查；不能代替留出验证或IoU。外包络不是液相像素真值。",
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "configuration": cfg,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "运行配置.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    calibration = Image.fromarray(bg.astype(np.uint8)).convert("RGB")
    cd = ImageDraw.Draw(calibration)
    cd.ellipse((cx-ax, cy-ay, cx+ax, cy+ay), outline=(255, 80, 80), width=3)
    cd.ellipse((sx-6, sy-6, sx+6, sy+6), fill=(50, 180, 255))
    label = f"视窗直径 {cfg.get('known_diameter_mm') or '未标定'} mm"
    cd.text((20, 15), label, font=font, fill=(255, 100, 80))
    if cfg.get("mm_per_pixel"):
        length_px = 10 / cfg["mm_per_pixel"]
        cd.line((30, height-35, 30+length_px, height-35), fill="white", width=5)
        cd.text((30, height-70), "10 mm", font=font, fill="white")
    calibration.save(out / "标定与视窗.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json") if Path(__file__).with_name("config.json").exists() else Path(__file__).with_name("config.example.json")))
    parser.add_argument("--input", help="图像序列文件夹；会覆盖配置中的输入路径")
    parser.add_argument("--output", help="结果文件夹")
    parser.add_argument("--auto-geometry", action="store_true", help="重新估计喷嘴位置和视窗")
    parser.add_argument("--reuse-geometry", action="store_true", help="新输入仍复用配置中的一次标定")
    parser.add_argument("--diameter-mm", type=float)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--background-frames", type=int)
    parser.add_argument("--batch", action="store_true", help="批量处理输入文件夹的直接子文件夹")
    args = parser.parse_args()
    changes = {}
    for cli, key in [(args.input, "source_dir"), (args.output, "output_dir"), (args.diameter_mm, "known_diameter_mm"), (args.fps, "fps"), (args.background_frames, "background_frames")]:
        if cli is not None:
            changes[key] = cli
    if args.auto_geometry or (args.input and not args.reuse_geometry):
        changes.update(source_xy=None, chamber_xy=None, chamber_axes_px=None, mm_per_pixel=None, radial_bins=None)
    if args.batch:
        if not args.input or not args.output:
            parser.error("批处理必须同时提供 --input 和 --output")
        outcomes = []
        for child in sorted(Path(args.input).iterdir()):
            if child.is_dir() and child.resolve() != Path(args.output).resolve():
                case = dict(changes, source_dir=str(child.resolve()), output_dir=str((Path(args.output) / child.name).resolve()), source_xy=None, chamber_xy=None, chamber_axes_px=None, mm_per_pixel=None, radial_bins=None)
                try:
                    main(args.config, case)
                    outcomes.append({"case": child.name, "ok": True})
                except Exception as exc:
                    print(f"跳过 {child.name}：{exc}", flush=True)
                    outcomes.append({"case": child.name, "ok": False, "reason": str(exc)})
        batch_out = Path(args.output); batch_out.mkdir(parents=True, exist_ok=True)
        (batch_out / "batch_summary.json").write_text(json.dumps(outcomes, ensure_ascii=False, indent=2), encoding="utf-8")
        if not outcomes or not any(case["ok"] for case in outcomes):
            raise SystemExit("批处理中没有成功识别的序列，请检查batch_summary.json。")
    else:
        main(args.config, changes)
