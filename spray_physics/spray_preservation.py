"""对近喷孔、非强光区域设置有证据支持的保守扣除上限。"""

import cv2
import numpy as np

from deep_data import source_connected


def preserve_source_spray(image, previous, following, background, gray, transmission, noise, cfg, roi):
    """只限制扣除，不补画纹理；返回新透射系数及应标不确定的保护范围。"""
    radius = min(cfg["chamber_axes_px"])
    sx, sy = cfg["source_xy"]
    yy, xx = np.indices(image.shape)
    distance = np.hypot(xx-sx, yy-sy)
    # 三帧中位变化减少噪声触发；空背景及新出现单帧噪点不能单独激活保护。
    change = np.median(np.stack([cv2.GaussianBlur(np.abs(frame-background), (0, 0), 3)
                                for frame in (previous, image, following)]), 0)
    excess = np.maximum(change-cv2.GaussianBlur(noise, (0, 0), 3)*1.5, 0)
    activity = np.clip(excess/(noise*3+2/255), 0, 1)
    # 对小范围的相消缺口做保守连续性传播，避免保护强度忽高忽低形成新的暗洞。
    local_radius = max(2, int(radius*0.045))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (local_radius*2+1, local_radius*2+1))
    activity = cv2.morphologyEx(activity.astype(np.float32), cv2.MORPH_CLOSE, kernel)
    activity = cv2.GaussianBlur(activity, (0, 0), max(1, radius*0.018))
    supported = source_connected((activity > 0.18) & roi, (sx, sy), reach=radius*0.20)
    supported = cv2.GaussianBlur(supported.astype(np.float32), (0, 0), 5)
    # 强白斑仍由恢复模型处理。此处不把整块原图贴回，以免重新引入打光。
    glare = np.maximum(background-gray, 0)
    coarse_glare = cv2.GaussianBlur(glare, (0, 0), max(1, radius*0.015))
    x = np.clip((coarse_glare-20/255)/(80/255-20/255), 0, 1)
    low_glare = 1-x*x*(3-2*x)
    source_weight = np.exp(-(distance/(radius*0.65))**4)
    strength = float(0.95 if cfg.get("source_preservation_strength") is None else cfg["source_preservation_strength"])
    if not 0 <= strength <= 1:
        raise ValueError("喷雾保留强度必须在0—1之间。")
    opacity_floor = strength*activity*supported*low_glare*source_weight
    opacity_floor *= roi & (glare > 3/255) & (glare < 80/255)
    guarded = np.minimum(transmission, 1-opacity_floor)
    adjustment = (transmission-guarded)*glare
    # 有保护介入的数值仍是保守估计，不能因变亮就标为可靠测量。
    uncertain = adjustment > 2/255
    return guarded.astype(np.float32), adjustment.astype(np.float32), uncertain
