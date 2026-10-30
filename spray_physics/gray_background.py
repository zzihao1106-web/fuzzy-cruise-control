"""空背景灰底校正：比较局部去白、全基座均匀化与纯灰参考模板。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from deep_spray import font
from spray_recognition import discover_sequence


def fit_gray_base(image, cfg):
    """从非水平打光方向的底座拟合平滑灰度，不生成或补画纹理。"""
    h, w = image.shape
    yy, xx = np.indices(image.shape, dtype=np.float32)
    cx, cy = cfg["chamber_xy"]; ax, ay = cfg["chamber_axes_px"]
    sx, sy = cfg["source_xy"]
    u, v = (xx-cx)/ax, (yy-cy)/ay
    rho = np.hypot(u, v)
    nozzle_distance = np.hypot(xx-sx, yy-sy)
    base = (rho < 0.92) & (nozzle_distance > min(ax, ay)*0.16)
    base &= np.abs(yy-sy) > np.abs(xx-sx)*0.65
    smooth = cv2.GaussianBlur(image, (0, 0), max(3, min(ax, ay)*0.045))
    design = np.stack([np.ones_like(u), u, v, u*u, u*v, v*v], -1)
    positions = np.flatnonzero(base.ravel())[::6]
    if len(positions) < 100:
        raise ValueError("可用灰色底座区域不足，请检查几何标定。")
    x, y = design.reshape(-1, 6)[positions], smooth.ravel()[positions]
    weight = np.ones_like(y)
    for _ in range(10):
        coef = np.linalg.lstsq(x*weight[:, None], y*weight, rcond=None)[0]
        residual = y-x@coef
        scale = max(3/255, float(np.median(np.abs(residual-np.median(residual))))*1.4826)
        # 正向异常主要来自照明，降低其拟合权重。
        normalized = residual/scale
        weight = np.where(normalized > 1.5, 1.5/np.maximum(normalized, 1.5), 1.0)
        weight *= np.where(normalized < -3, 3/np.maximum(-normalized, 3), 1.0)
    target = np.einsum("hwc,c->hw", design, coef)
    low, high = np.quantile(y, [0.10, 0.85])
    target = np.clip(target, max(0.05, low-0.02), min(0.85, high+0.02)).astype(np.float32)
    return target, smooth, rho, nozzle_distance


def calibrate_gray_background(image, cfg):
    """只用于已确认无喷雾的背景，不能直接套用在含喷雾图上。"""
    target, smooth, rho, nozzle_distance = fit_gray_base(image, cfg)
    radius = min(cfg["chamber_axes_px"])
    excess = smooth-target
    candidates = ((excess > 12/255) & (rho < 0.975) & (nozzle_distance > radius*0.12)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates)
    region = np.zeros_like(image, np.float32)
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > radius*radius*0.015 and float(excess[labels == i].max()) > 32/255:
            region[labels == i] = 1
    feather_sigma = max(2, radius*0.018)
    region = cv2.GaussianBlur(region, (0, 0), feather_sigma)
    strength = np.clip((excess-8/255)/(22/255), 0, 1)
    strength = strength*strength*(3-2*strength)
    outer_protection = np.clip((0.98-rho)/0.025, 0, 1)
    nozzle_protection = np.clip((nozzle_distance-radius*0.10)/(radius*0.05), 0, 1)
    blend = region*strength*outer_protection*nozzle_protection
    ratio = np.clip(target/np.maximum(smooth, 1/255), 0.2, 1)
    gain = 1-blend+blend*ratio
    corrected = np.clip(image*gain, 0, 1)
    return corrected, gain, blend, target


def uniform_gray_variants(image, cfg, return_maps=False):
    """将整个圆窗内基座归一化；细纹理版与纯模板版分别保存。"""
    _, smooth, rho, nozzle_distance = fit_gray_base(image, cfg)
    yy, xx = np.indices(image.shape, dtype=np.float32)
    sx, sy = cfg["source_xy"]
    radius = min(cfg["chamber_axes_px"])
    unlit = (rho < 0.88) & (nozzle_distance > radius*0.20)
    unlit &= np.abs(yy-sy) > np.abs(xx-sx)*1.25
    if unlit.sum() < 100:
        raise ValueError("可用底座区域不足，无法估计统一灰度，请检查几何标定。")
    gray = float(np.median(smooth[unlit]))
    # 用无直射光的圆周剖面保留圆窗结构，消除环边角向光照差异。
    profile_r = np.linspace(0.90, 1.20, 151)
    ring_valid = (np.abs(yy-sy) > np.abs(xx-sx)*0.85)
    profile = []
    for r in profile_r:
        take = (np.abs(rho-r) < 0.003) & ring_valid
        profile.append(float(np.median(image[take])) if take.any() else gray)
    profile = cv2.GaussianBlur(np.array(profile, np.float32).reshape(1, -1), (0, 0), 1.5).ravel()
    radial = np.interp(rho, profile_r, profile).astype(np.float32)
    transition = np.clip((rho-0.94)/0.035, 0, 1)
    transition = transition*transition*(3-2*transition)
    surface = gray*(1-transition)+radial*transition
    # 仅从原图提取很细的纹理，限制残余阴影幅度，不生成新纹理。
    detail = image-cv2.GaussianBlur(image, (0, 0), 1.5)
    detail = (4/255)*np.tanh(detail/(4/255))
    inside = np.clip((1.18-rho)/0.06, 0, 1)
    inside = inside*inside*(3-2*inside)
    core_radius = max(18, radius*0.065)
    nozzle_blend = np.clip((nozzle_distance-core_radius)/max(8, radius*0.03), 0, 1)
    nozzle_blend = nozzle_blend*nozzle_blend*(3-2*nozzle_blend)
    weight = inside*nozzle_blend
    textured = image*(1-weight)+np.clip(surface+detail, 0, 1)*weight
    template = image*(1-weight)+surface*weight
    protected = nozzle_distance <= core_radius
    assert np.array_equal(np.uint8(np.rint(textured[protected]*255)), np.uint8(np.rint(image[protected]*255)))
    assert np.array_equal(np.uint8(np.rint(template[protected]*255)), np.uint8(np.rint(image[protected]*255)))
    valid_surface = (rho < 0.90) & (nozzle_distance > radius*0.15)
    statistics = {"estimated_base_gray_0_255": gray*255,
                  "surface_std_original": float(image[valid_surface].std()*255),
                  "surface_std_textured": float(textured[valid_surface].std()*255),
                  "surface_std_template": float(template[valid_surface].std()*255),
                  "nozzle_core_radius_px": core_radius,
                  "nozzle_pixels_preserved": True,
                  "ring_geometry_preserved_but_illumination_normalized": True,
                  "template_is_estimated_reference_not_measured_texture": True}
    results = np.clip(textured, 0, 1), np.clip(template, 0, 1), statistics
    return (*results, {"surface": surface, "weight": weight}) if return_maps else results


def main():
    p = argparse.ArgumentParser(description="从无喷雾背景建立保留场景的灰底参考")
    p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    p.add_argument("--uniform-variants", action="store_true", help="附加全基座平场及均匀灰底模板对照")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    paths, _ = discover_sequence(cfg["source_dir"])
    count = int(cfg.get("background_frames", 5))
    if count < 1 or count > len(paths):
        raise ValueError("空背景帧数无效。")
    out = Path(args.output)
    if out.resolve() == Path(cfg["source_dir"]).resolve():
        raise ValueError("输出目录不能与原图相同。")
    out.mkdir(parents=True, exist_ok=True)
    originals = [np.asarray(Image.open(path).convert("L"), np.float32)/255 for path in paths[:count]]
    reference = np.median(np.stack(originals), 0)
    corrected, gain, blend, target = calibrate_gray_background(reference, cfg)
    Image.fromarray(np.uint8(np.rint(reference*255))).save(out/"原始空背景.png")
    Image.fromarray(np.uint8(np.rint(corrected*255))).save(out/"灰底背景参考.png")
    Image.fromarray(np.uint8(np.rint(blend*255))).save(out/"光斑校正范围.png")
    np.savez_compressed(out/"灰底校正参数.npz", gain=gain, blend=blend, target=target)
    for i, original in enumerate(originals, 1):
        frame = np.uint8(np.rint(np.clip(original*gain, 0, 1)*255))
        Image.fromarray(frame).save(out/f"第{i:02d}帧_灰底去白.png")
    canvas = Image.new("RGB", (1280, 440), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for col, (title, file) in enumerate([("原始空背景", "原始空背景.png"), ("白斑校正为灰色底座（样图）", "灰底背景参考.png")]):
        draw.text((col*640+12, 8), title, font=font(22), fill="black")
        im = Image.open(out/file).convert("RGB").resize((640, 400))
        canvas.paste(im, (col*640, 40))
    canvas.save(out/"灰底去白_前后对照.jpg", quality=96)
    nozzle_x, nozzle_y = cfg["source_xy"]
    yy, xx = np.indices(reference.shape)
    protected = np.hypot(xx-nozzle_x, yy-nozzle_y) <= min(cfg["chamber_axes_px"])*0.10
    outside = blend < 1e-8
    before = np.uint8(np.rint(reference*255)); after = np.uint8(np.rint(corrected*255))
    assert np.array_equal(before[protected], after[protected])
    assert np.array_equal(before[outside], after[outside])
    summary = {"kind": "空背景灰底光照校正样图", "background_frames": count,
               "primary_method": "A局部光照校正；均匀方案另见均匀灰底统计.json",
               "scene_preserved": True, "nozzle_pixels_unchanged": True,
               "outside_correction_pixels_unchanged": True,
               "corrected_pixels": int((blend > 0.01).sum()),
               "target_gray_estimated_from_base": True, "true_unlit_background_measured": False,
               "spray_frames_processed": 0,
               "note": "这是待用户确认的背景外观参考，不是黑底前景，也未将增益图直接套用到含喷雾帧。"}
    (out/"样图说明.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.uniform_variants:
        textured, template, statistics, maps = uniform_gray_variants(reference, cfg, return_maps=True)
        np.savez_compressed(out/"均匀灰底参考参数.npz", **maps)
        for directory in (f"方案B_前{count}帧", f"方案C_前{count}帧"):
            (out/directory).mkdir(exist_ok=True)
        for i, original in enumerate(originals, 1):
            detail = original-cv2.GaussianBlur(original, (0, 0), 1.5)
            detail = (4/255)*np.tanh(detail/(4/255))
            for directory, surface in ((f"方案B_前{count}帧", np.clip(maps["surface"]+detail, 0, 1)),
                                       (f"方案C_前{count}帧", maps["surface"])):
                array = original*(1-maps["weight"])+surface*maps["weight"]
                frame = np.uint8(np.rint(np.clip(array, 0, 1)*255))
                # 每张背景都保护喷孔核心，不用中位数参考覆盖该处原像素。
                core = maps["weight"] == 0
                assert np.array_equal(frame[core], np.uint8(np.rint(original[core]*255)))
                Image.fromarray(frame).save(out/directory/f"{i:06d}.png")
        files = [("原始空背景", "原始空背景.png", reference),
                 ("A 局部去白：保留原底座明暗", "方案A_局部去白.png", corrected),
                 ("B 全基座平场：保留细纹理", "方案B_均匀灰底_细纹理.png", textured),
                 ("C 均匀灰底模板：保留喷孔及圆窗结构", "方案C_均匀灰底模板.png", template)]
        sheet = Image.new("RGB", (1280, 880), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for i, (title, name, array) in enumerate(files):
            pic = Image.fromarray(np.uint8(np.rint(array*255)))
            pic.save(out/name)
            x, y = i % 2*640, i // 2*440
            draw.text((x+10, y+8), title, font=font(20), fill="black")
            sheet.paste(pic.convert("RGB").resize((640, 400)), (x, y+40))
        sheet.save(out/"三种灰底方案对照.jpg", quality=96)
        (out/"均匀灰底统计.json").write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(statistics, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
