"""原始背景保留、静态区域光度校正及可控增亮/变暗喷雾合成。"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def channels(image, background):
    delta = image - background
    return np.stack([image, background, np.maximum(delta, 0), np.maximum(-delta, 0)]).astype(np.float32)


def correct_photometry(image, background, static):
    """仅用视窗外静态区域拟合有限增益与偏置，不拟合喷雾内部。"""
    selected = static & np.isfinite(image) & np.isfinite(background)
    if selected.sum() < 100:
        return background.copy(), {"gain": 1.0, "offset": 0.0, "fitted": False}
    x, y = background[selected][::8], image[selected][::8]
    gain, offset = 1.0, float(np.clip(np.median(y-x), -0.04, 0.04))
    for _ in range(3):
        residual = y - (x * gain + offset)
        good = np.abs(residual - np.median(residual)) < max(0.01, float(np.median(np.abs(residual))) * 3)
        if good.sum() < 30 or np.std(x[good]) < 0.015:
            break
        gain, offset = np.linalg.lstsq(np.c_[x[good], np.ones(good.sum())], y[good], rcond=None)[0]
        gain, offset = float(np.clip(gain, 0.9, 1.1)), float(np.clip(offset, -0.04, 0.04))
    return np.clip(background * gain + offset, 0, 1), {"gain": gain, "offset": offset, "fitted": True}


def crop_geometry(shape, cfg):
    h, w = shape
    cx, cy = cfg["chamber_xy"]
    ax, ay = cfg["chamber_axes_px"]
    if ax <= 0 or ay <= 0:
        raise ValueError("视窗半轴必须为正。")
    y, x = np.indices(shape)
    rho = np.sqrt(((x-cx)/ax)**2 + ((y-cy)/ay)**2)
    # 保留视窗边缘，避免缩小ROI后隐藏真实误检。
    roi = rho <= 0.995
    bounds = [max(0, int(cx-ax-16)), max(0, int(cy-ay-16)), min(w, int(cx+ax+17)), min(h, int(cy+ay+17))]
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1] or roi.sum() < 100:
        raise ValueError("标定视窗不在图像内。")
    return bounds, roi, rho > 1.12


def crop_resize(array, bounds, size, nearest=False):
    x0, y0, x1, y1 = bounds
    return cv2.resize(array[y0:y1, x0:x1].astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def prepare(folder, cfg, output, size=320):
    from spray_recognition import auto_geometry, discover_sequence
    output = Path(output)
    if output.resolve() == Path(folder).resolve():
        raise ValueError("数据准备输出不能与原图目录相同。")
    cfg = {**cfg, "source_dir": str(Path(folder).resolve())}
    output.mkdir(parents=True, exist_ok=True)
    paths, excluded = discover_sequence(folder)
    count = int(cfg.get("background_frames", 5))
    if count < 2 or count >= len(paths):
        raise ValueError("背景帧至少2帧，且必须保留后续图像。")
    if not all(cfg.get(k) is not None for k in ("source_xy", "chamber_xy", "chamber_axes_px")):
        cfg = {**cfg, **auto_geometry(paths, count)}
    backgrounds = np.stack([read_gray(p) for p in paths[:count]])
    # 留出空背景用于内部误检检查；不将这些帧投入合成训练。
    fit_count = max(1, count - min(2, count-1))
    background = np.median(backgrounds[:fit_count], axis=0)
    bounds, roi, static = crop_geometry(background.shape, cfg)
    bg = crop_resize(background, bounds, size)
    pool = np.stack([crop_resize(im, bounds, size) for im in backgrounds[:fit_count]])
    small_roi = crop_resize(roi, bounds, size, True) > 0.5
    np.savez_compressed(output / "backgrounds.npz", background=bg, pool=pool, roi=small_roi)
    sx, sy = cfg["source_xy"]
    x0, y0, x1, y1 = bounds
    info = {"source_dir": str(Path(folder).resolve()), "config": cfg, "size": size,
            "crop_bounds_xyxy": bounds, "source_normalized": [(sx-x0)/(x1-x0), (sy-y0)/(y1-y0)],
            "reference_frames_1based": list(range(1, fit_count+1)),
            "heldout_empty_frames_1based": list(range(fit_count+1, count+1)),
            "split_note": "只有一组真实序列；真实帧不作随机训练/测试划分；训练仅用参考帧及程序生成背景合成。",
            "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
            "excluded": excluded, "human_labels": 0}
    write_json(output / "manifest.json", info)
    Image.fromarray(np.uint8(background*255)).save(output / "原始背景.png")
    return info


class SpraySynthesizer:
    """合成监督有已知掩膜；随机种子分离，不将其称为真实喷雾标注。"""
    def __init__(self, background_path, source_normalized=(0.5, 0.5), seed=20261030, procedural_only=False, curriculum=False):
        with np.load(background_path) as z:
            self.pool = z["pool"].astype(np.float32)
            self.background = z["background"].astype(np.float32)
            self.roi = z["roi"].astype(bool)
        self.size = self.background.shape[0]
        self.source = source_normalized
        self.rng = np.random.default_rng(seed)
        self.procedural_only = procedural_only
        self.curriculum = curriculum
        self.yy, self.xx = np.indices(self.background.shape, dtype=np.float32)

    def smooth_noise(self, scale):
        return cv2.resize(self.rng.normal(size=(scale, scale)).astype(np.float32), (self.size, self.size), interpolation=cv2.INTER_CUBIC)

    def sample(self, force_empty=False, restoration=False):
        rng, n = self.rng, self.size
        use_procedural = self.procedural_only or rng.random() < 0.4
        if use_procedural:
            b = np.clip(rng.uniform(0.12, 0.45) + self.smooth_noise(5)*0.04, 0, 1)
            for _ in range(int(rng.integers(1, 5))):
                px, py = rng.uniform(0, n, 2)
                sx, sy = rng.uniform(0.04*n, 0.6*n, 2)
                b += rng.uniform(0.1, 0.6)*np.exp(-((self.xx-px)/sx)**2-((self.yy-py)/sy)**2)
            b = np.clip(b, 0, 0.95).astype(np.float32)
        else:
            b = self.pool[int(rng.integers(len(self.pool)))].copy()
            if rng.random() < 0.5:
                b = np.rot90(b, int(rng.integers(4))).copy()
            b = np.clip(b*rng.uniform(0.8, 1.1) + rng.uniform(-0.03, 0.03), 0, 1)
        b_ref = b.copy()
        # 没有喷雾时也模拟光照漂移，防止绝对差被直接学成前景。
        current_bg = np.clip(b*rng.uniform(0.97, 1.03) + rng.uniform(-0.012, 0.012) + self.smooth_noise(3)*0.008, 0, 1)
        if self.curriculum:
            radius = np.hypot(self.xx-n/2, self.yy-n/2)
            ring = np.exp(-((radius-n*0.47)/(n*0.009))**2)
            current_bg = np.clip(current_bg+ring*rng.uniform(-0.07, 0.07), 0, 1)
        mask = np.zeros_like(b)
        if not force_empty and rng.random() > 0.25:
            cx = (self.source[0] + rng.uniform(-0.08, 0.08))*n
            cy = (self.source[1] + rng.uniform(-0.08, 0.08))*n
            dx, dy = self.xx-cx, self.yy-cy
            r = np.hypot(dx, dy)
            angle = np.arctan2(dy, dx)
            jets = int(rng.integers(1, 15))
            phase = rng.uniform(-np.pi, np.pi)
            base_length = rng.uniform(0.04*n, 0.55*n)
            rough = self.smooth_noise(25)*rng.uniform(0.5, 2.5)
            for j in range(jets):
                theta = phase + j*2*np.pi/jets + rng.normal(0, 0.03)
                da = np.abs(np.arctan2(np.sin(angle-theta), np.cos(angle-theta)))
                opening = rng.uniform(0.07, min(0.25, 2.5/jets))
                length = base_length*rng.uniform(0.8, 1.2)
                cap = length*(0.83+0.17*np.sqrt(np.maximum(0, 1-(da/opening)**2)))
                lateral = r*(opening-da)
                signed = np.minimum(cap-r, lateral) + rough
                plume = np.clip((signed+1.2)/2.4, 0, 1)*(r > 2)
                mask = np.maximum(mask, plume)
            if self.curriculum and base_length > n*0.2:
                # 晚期可出现合并云团，避免训练集永远只有互不相交的细射线。
                core_radius = rng.uniform(0.05, 0.20)*n
                mask = np.maximum(mask, np.clip((core_radius-r+rough+1)/2, 0, 1))
        mask *= self.roi
        opacity = rng.uniform(0.3, 0.95)
        scatter = rng.uniform(0.15, 0.85)
        if rng.random() < 0.15:
            scatter = 0.0
        texture = np.clip(0.8 + 0.12*self.smooth_noise(36) + 0.08*self.smooth_noise(10), 0.3, 1.2)
        signal = mask*texture*(scatter-opacity*current_bg)
        if self.curriculum:
            # 人工制造白斑遮断；保持潜在掩膜，并把不可见性作为辅助目标。
            threshold = max(0.45, float(np.quantile(b[self.roi], 0.80)))
            white = np.clip((b-threshold)*20, 0, 1)
            white = cv2.GaussianBlur(white, (5, 5), 1)
            signal *= 1-white*rng.uniform(0.85, 1.0)
            if rng.random() < 0.35:
                center = np.exp(-((self.xx-n/2)**2+(self.yy-n/2)**2)/(n*0.12)**2)
                signal *= 1-center*rng.uniform(0.70, 1.0)
        image = np.clip(current_bg+signal+rng.normal(0, rng.uniform(0.002, 0.009), b.shape), 0, 1).astype(np.float32)
        hard = (mask > 0.15).astype(np.float32)
        edge = cv2.morphologyEx(hard, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        ambiguity = np.maximum(edge*0.65, hard*(np.abs(signal) < 0.025))
        target = np.stack([hard, edge, ambiguity]).astype(np.float32)
        if restoration:
            # 监督的是可观测的有符号喷雾贡献，不凭空补回完全相消的纹理。
            clean_signed = np.clip(current_bg+signal, 0, 1)-current_bg
            return channels(image, b_ref), target, clean_signed[None].astype(np.float32)
        return channels(image, b_ref), target


def source_connected(mask, source_xy, reach=12):
    """保留靠近喷嘴的连通分量；返回掩膜而非用先验填满缺口。"""
    mask = np.asarray(mask, dtype=np.uint8)
    _, labels = cv2.connectedComponents(mask)
    h, w = mask.shape
    y, x = np.indices(mask.shape)
    near = (x-source_xy[0])**2+(y-source_xy[1])**2 <= reach**2
    ids = np.unique(labels[near & (mask > 0)])
    ids = ids[ids > 0]
    return np.isin(labels, ids) if len(ids) else np.zeros((h, w), bool)
