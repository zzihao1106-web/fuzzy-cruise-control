"""空气中自由喷雾的低阶贯穿距约束：先盲测，再决定是否接入边界。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from deep_data import read_gray, write_json
from deep_spray import font
from spray_recognition import discover_sequence, polar_sample

VERSION = "physics-prior-0.1"


def air_scenarios(cfg):
    """压力基准未知时输出多情形，不把名义工况改成已确认绝压。"""
    if cfg.get("ambient_gas") != "air":
        raise ValueError("此入口目前只定义了空气；请确认环境介质。")
    temperature = float(cfg["ambient_temperature_K"])
    if temperature <= 0:
        raise ValueError("空气温度必须为正开尔文。")
    ambient = float(cfg["ambient_pressure_bar"])*1e5
    injection = float(cfg["injection_pressure_MPa_label"])*1e6
    area = np.pi*(float(cfg["nozzle_diameter_mm"])*1e-3)**2/4
    ambient_bases = [cfg["ambient_pressure_basis"]] if cfg.get("ambient_pressure_basis") in {"absolute", "gauge"} else ["absolute", "gauge"]
    injection_bases = [cfg["injection_pressure_basis"]] if cfg.get("injection_pressure_basis") in {"absolute", "gauge"} else ["absolute", "gauge"]
    results = []
    for ab in ambient_bases:
        for ib in injection_bases:
            pa = ambient+(101325 if ab == "gauge" else 0)
            pi = injection+(101325 if ib == "gauge" else 0)
            if pi <= pa:
                raise ValueError("喷油端压力必须大于环境压力。")
            rho = pa/(287.05*temperature)
            results.append({"ambient_basis_scenario": ab, "injection_basis_scenario": ib,
                            "ambient_absolute_pa": pa, "delta_pa": pi-pa, "temperature_K": temperature,
                            "air_density_kg_m3": rho, "density_model": "ideal_dry_air_Z_1_assumed",
                            "penetration_scale_m_sqrt_s": float((area*(pi-pa)/rho)**0.25)})
    return results


def penetration(times_ms, coefficients, transition_ms, onset_ms, scale):
    """本项目渐近插值：小时间线性，大时间平方根；不是经验公式原文复现。"""
    tau = np.maximum(np.asarray(times_ms)-onset_ms, 0)/1000
    tb = transition_ms/1000
    return np.asarray(coefficients)*scale*tau/np.sqrt(tau+tb)*1000


def extract_fronts(folder, cfg, train_end=12, uncertainty_folder=None):
    paths, _ = discover_sequence(folder)
    bg_count = int(cfg.get("background_frames", 5))
    if train_end <= bg_count+3 or train_end >= len(paths):
        raise ValueError("拟合结束帧需保留至少4张发展图及后续留出图。")
    reference = np.median(np.stack([read_gray(p) for p in paths[:max(1, bg_count-2)]]), 0)
    sx, sy = cfg["source_xy"]; cx, cy = cfg["chamber_xy"]; ax, ay = cfg["chamber_axes_px"]
    nr, na = int(np.ceil(max(ax, ay)*1.2)), 720
    rr = np.arange(nr, dtype=np.float32)
    angles = np.arange(na)*2*np.pi/na
    mapx = sx+np.cos(angles[:, None])*rr
    mapy = sy+np.sin(angles[:, None])*rr
    visible = ((mapx-cx)/ax)**2+((mapy-cy)/ay)**2 < 0.97**2
    limits = visible.sum(1).astype(float)
    signal = np.zeros(na)
    polar = []
    for index, path in enumerate(paths, 1):
        current = np.maximum(read_gray(path)-reference, 0)
        pp = polar_sample(current, (sx, sy), na, nr)
        polar.append(pp)
        if bg_count+1 <= index <= train_end:
            signal += pp[:, int(min(ax, ay)*0.12):int(min(ax, ay)*0.40)].mean(1)
    signal = gaussian_filter1d(signal, 2, mode="wrap")
    peaks, _ = find_peaks(np.tile(signal, 3), distance=36, prominence=max(0.008, float(signal.max())*0.10))
    peaks = peaks[(peaks >= na) & (peaks < na*2)]-na
    if not 2 <= len(peaks) <= 24:
        raise ValueError(f"早期图像无法可靠确定喷束方向：检出{len(peaks)}条。")
    njet = len(peaks)
    fronts = np.full((len(paths), njet), np.nan)
    confidence = np.zeros_like(fronts)
    censored = np.zeros_like(fronts, dtype=bool)
    profiles = np.zeros((len(paths), njet, nr), np.float32)
    for t, pp in enumerate(polar):
        unknown = None
        if uncertainty_folder:
            unknown = polar_sample(read_gray(Path(uncertainty_folder)/paths[t].name), (sx, sy), na, nr)
        for j, peak in enumerate(peaks):
            rows = (peak+np.arange(-4, 5)) % na
            profile = gaussian_filter1d(np.median(pp[rows], 0), 1.2)
            profiles[t, j] = profile
            lo, hi = int(min(ax, ay)*0.08), int(limits[peak])-4
            valid = (profile > 12/255)
            valid[:lo] = False; valid[hi:] = False
            linked = cv2.morphologyEx(valid[None].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8))[0] > 0
            starts = np.flatnonzero(linked & ~np.r_[False, linked[:-1]])
            ends = np.flatnonzero(linked & ~np.r_[linked[1:], False])
            blocks = [(a, b) for a, b in zip(starts, ends) if a < lo+55 and b-a >= 6]
            if not blocks or t < bg_count:
                continue
            a, front = max(blocks, key=lambda ab: ab[1])
            fronts[t, j] = front
            tail = slice(max(a, front-12), front+1)
            local_uncertain = float(np.median(unknown[rows, tail])) if unknown is not None else 0
            # 前沿本来由阈值定义，不能因其灰度接近阈值就把所有清晰边缘降为低置信。
            # 对比度用前沿内侧的喷雾区域估计，恢复不确定性仍在前沿处检查。
            interior = slice(max(a, front-25), max(a+1, front-5))
            confidence[t, j] = (1-local_uncertain)*min(1, float(profile[interior].mean())/(20/255))
            censored[t, j] = front >= limits[peak]-14
    return {"paths": paths, "reference": reference, "peaks": peaks, "angles": angles[peaks], "limits_px": limits[peaks],
            "fronts_px": fronts, "confidence": confidence, "censored": censored, "profiles": profiles}


def fit_model(times_ms, observed_mm, good, scale, onset_bounds, initial=None):
    njet = observed_mm.shape[1]
    rows, jets = np.where(good)
    if any(good[:, j].sum() < 3 for j in range(njet)):
        raise ValueError("每条参与拟合的清晰喷束至少需要3个可靠前沿。")
    def residual(p):
        prediction = penetration(times_ms[rows], np.exp(p[jets]), np.exp(p[-2]), p[-1], scale)
        return (prediction-observed_mm[rows, jets])/0.4
    guess = np.r_[np.full(njet, np.log(1.5)), np.log(0.10), np.mean(onset_bounds)] if initial is None else initial
    lower = np.r_[np.full(njet, np.log(0.05)), np.log(0.005), onset_bounds[0]]
    upper = np.r_[np.full(njet, np.log(12)), np.log(3.0), onset_bounds[1]]
    fit = least_squares(residual, np.clip(guess, lower+1e-8, upper-1e-8), bounds=(lower, upper), loss="soft_l1", max_nfev=2000)
    predicted = penetration(times_ms[:, None], np.exp(fit.x[:-2])[None], np.exp(fit.x[-2]), fit.x[-1], scale)
    return fit, predicted


def summarize_error(prediction, reference, mask):
    error = (prediction-reference)[mask]
    return {"samples": int(mask.sum()), "mae_mm": float(np.mean(np.abs(error))) if len(error) else None,
            "rmse_mm": float(np.sqrt(np.mean(error**2))) if len(error) else None}


def run(args):
    if args.bootstrap < 0:
        raise ValueError("自举次数不能为负。")
    if args.test_end <= args.train_end:
        raise ValueError("留出结束帧必须晚于拟合结束帧。")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not all(cfg.get(key) is not None for key in ("source_xy", "chamber_xy", "chamber_axes_px")):
        raise ValueError("请载入已经检查几何标定的V6运行配置，再建立物理前沿。")
    scenarios = air_scenarios(cfg)
    nominal = scenarios[0]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if out.resolve() == Path(args.cleaned).resolve():
        raise ValueError("物理诊断结果不能写入灰底输入图目录。")
    series = extract_fronts(args.cleaned, cfg, args.train_end, args.uncertainty)
    mm = float(cfg.get("mm_per_pixel") or cfg["known_diameter_mm"]/(2*np.sqrt(np.prod(cfg["chamber_axes_px"]))))
    observed = series["fronts_px"]*mm
    times = np.arange(len(observed))/float(cfg["fps"])*1000
    count = int(cfg.get("background_frames", 5))
    train = np.zeros_like(observed, bool); train[count:args.train_end] = True
    reliable = np.isfinite(observed) & (series["confidence"] > 0.6) & ~series["censored"]
    reliable &= series["fronts_px"] < series["limits_px"][None]*0.9
    train &= reliable
    clear = np.flatnonzero(train.sum(0) >= 4)
    if len(clear) < 3:
        raise ValueError("至少需要3条具有4个早期可靠观测的喷束，暂不启用物理补全。")
    train[:, np.setdiff1d(np.arange(observed.shape[1]), clear)] = False
    dt = 1000/float(cfg["fps"])
    onset_bounds = [times[count]-dt*1.2, times[count]-dt*0.01]
    fit, clear_prediction = fit_model(times, observed[:, clear], train[:, clear], nominal["penetration_scale_m_sqrt_s"], onset_bounds)
    coefficients = np.full(observed.shape[1], np.median(np.exp(fit.x[:-2])))
    coefficients[clear] = np.exp(fit.x[:-2])
    predicted = penetration(times[:, None], coefficients[None], np.exp(fit.x[-2]), fit.x[-1], nominal["penetration_scale_m_sqrt_s"])
    test = reliable.copy(); test[:args.train_end] = False; test[args.test_end:] = False
    test[:, np.setdiff1d(np.arange(observed.shape[1]), clear)] = False
    baseline = np.full_like(observed, np.nan)
    power = np.full_like(observed, np.nan)
    for j in clear:
        ids = np.flatnonzero(train[:, j]); baseline[:, j] = observed[ids[-1], j]
        def residual(p):
            return p[0]*np.maximum(times[ids]-p[2], 1e-4)**p[1]-observed[ids, j]
        f = least_squares(residual, [20, 0.7, np.mean(onset_bounds)],
                          bounds=([0.01, 0.2, onset_bounds[0]], [1000, 1.8, onset_bounds[1]]), loss="soft_l1")
        power[:, j] = f.x[0]*np.maximum(times-f.x[2], 0)**f.x[1]
    # 自举只使用拟合帧，不用留出前沿调整区间。
    rng = np.random.default_rng(20261102)
    boot = []
    for _ in range(args.bootstrap):
        sampled = observed[:, clear].copy()
        for k, j in enumerate(clear):
            ids = np.flatnonzero(train[:, j])
            residuals = observed[ids, j]-predicted[ids, j]
            sampled[ids, k] = predicted[ids, j]+rng.choice(residuals, len(ids), replace=True)
        _, pp = fit_model(times, sampled, train[:, clear], nominal["penetration_scale_m_sqrt_s"], onset_bounds, fit.x)
        boot.append(pp)
    sigma = np.maximum(np.std(np.stack(boot), 0) if boot else np.zeros_like(clear_prediction), 0.5)
    intervals = np.full_like(predicted, 2.0)
    intervals[:, clear] = 1.96*sigma
    # 预测越远置信带越宽，且不把区间说成已校准的统计覆盖率。
    horizon = np.maximum(times-times[args.train_end-1], 0)
    intervals += horizon[:, None]*0.75
    coverage = float((np.abs(predicted-observed)[test] <= intervals[test]).mean()) if test.any() else None
    summary = {"version": VERSION, "ambient_gas": "air", "air_temperature_K": float(cfg["ambient_temperature_K"]),
               "air_temperature_measured": bool(cfg.get("ambient_temperature_measured", False)),
               "temperature_assumption_source": cfg.get("ambient_temperature_source"), "pressure_scenarios": scenarios,
               "nominal_scenario_is_assumption": True, "mm_per_pixel": mm, "fps": cfg["fps"],
               "consecutive_frames_assumed": cfg.get("consecutive_frames_assumed", False),
               "fit_frames_1based": [count+1, args.train_end], "test_frames_1based": [args.train_end+1, args.test_end],
               "directions_estimated_from_training_frames_only": True,
               "angles_deg": (series["angles"]*180/np.pi).tolist(), "clear_jet_indices_1based": (clear+1).tolist(),
               "fitted_onset_ms_relative_first_frame": float(fit.x[-1]), "transition_ms": float(np.exp(fit.x[-2])),
               "transition_is_effective_interpolation_parameter_not_measured_breakup": True,
               "coefficients": coefficients.tolist(), "training_error": summarize_error(predicted, observed, train),
               "blind_test_physics": summarize_error(predicted, observed, test),
               "blind_test_persistence": summarize_error(baseline, observed, test),
               "blind_test_free_power": summarize_error(power, observed, test),
               "blind_test_interval_coverage": coverage, "interval_coverage_is_not_calibrated": True,
               "hard_boundary_constraints_enabled": False, "restored_appearance_modified": False,
               "viewport_is_not_wall": True, "predictions_are_not_clamped_to_viewport": True,
               "raw_images_read": False, "reference": "清晰去白图阈值前沿，非人工真值", "human_labels": 0,
               "input_hashes": [{"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in series["paths"]],
               "extrapolation_after_test_is_unvalidated": True,
               "physical_model": "S=C*(A*deltaP/rho_air)^0.25*tau/sqrt(tau+tau_b)",
               "limits": "二维投影自由射流低阶插值；未验证喷射结束、碰壁、强蒸发阶段；不预测液滴纹理"}
    write_json(out/"physics_summary.json", summary)
    np.savez_compressed(out/"physics_tracks.npz", times_ms=times, observed_mm=observed, predicted_mm=predicted,
                        interval_halfwidth_mm=intervals, reliable=reliable, train=train, test=test,
                        angles_rad=series["angles"], visibility_limit_mm=series["limits_px"]*mm,
                        confidence=series["confidence"], censored=series["censored"])
    with (out/"physics_tracks.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame", "jet", "time_ms", "observed_mm", "predicted_mm", "interval_halfwidth_mm", "status", "censored"])
        writer.writeheader()
        for t in range(len(times)):
            for j in range(observed.shape[1]):
                status = "fit" if train[t, j] else "blind_test" if test[t, j] else "prediction_only"
                if t < count:
                    status = "empty_reference_not_a_measured_front"
                if predicted[t, j] > series["limits_px"][j]*mm:
                    status = "predicted_outside_viewport"
                writer.writerow({"frame": t+1, "jet": j+1, "time_ms": times[t],
                                 "observed_mm": observed[t, j] if reliable[t, j] else "",
                                 "predicted_mm": predicted[t, j], "interval_halfwidth_mm": intervals[t, j],
                                 "status": status, "censored": bool(series["censored"][t, j])})
    render(out, series, cfg, times, predicted, intervals, train, test)
    print(json.dumps({k: v for k, v in summary.items() if k != "input_hashes"}, ensure_ascii=False), flush=True)
    return summary


def render(out, series, cfg, times, prediction, intervals, train, test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mm = float(cfg.get("mm_per_pixel") or cfg["known_diameter_mm"]/(2*np.sqrt(np.prod(cfg["chamber_axes_px"]))))
    observed = series["fronts_px"]*mm
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    selected = sorted(np.flatnonzero(train.sum(0) >= 4), key=lambda j: -int(test[:, j].sum()))[:4]
    checked_last = int(np.flatnonzero(test.any(1))[-1]) if test.any() else -1
    for ax, j in zip(axes.ravel(), selected):
        ax.plot(times[:checked_last+1], prediction[:checked_last+1, j], color="#e99721", label="Physical prior (projection)")
        if checked_last >= 0:
            ax.plot(times[checked_last:], prediction[checked_last:, j], color="#e99721", linestyle=":", label="Unvalidated extrapolation")
        ax.fill_between(times, prediction[:, j]-intervals[:, j], prediction[:, j]+intervals[:, j], color="#e99721", alpha=.18)
        ax.scatter(times[train[:, j]], observed[train[:, j], j], color="#23864b", label="Fit observations", s=25)
        ax.scatter(times[test[:, j]], observed[test[:, j], j], color="#4276b7", label="Held-out observations", s=25)
        ax.axhline(series["limits_px"][j]*mm, color="gray", linestyle="--", label="Visibility limit")
        ax.set(xlim=(0, times[min(23, len(times)-1)]), ylim=(0, max(35, float(series["limits_px"][j]*mm+5))),
               xlabel="Time from first frame (ms)", ylabel="Projected penetration (mm)", title=f"Jet {j+1} / {series['angles'][j]*180/np.pi:.1f} deg")
        ax.grid(alpha=.2)
    axes.ravel()[0].legend(fontsize=7)
    fig.savefig(out/"物理贯穿距盲测.png", dpi=160); plt.close(fig)
    sx, sy = cfg["source_xy"]
    frames = []
    (out/"overlays").mkdir(exist_ok=True)
    validated_end = int(np.flatnonzero(test.any(1))[-1]+1) if test.any() else 0
    for t, path in enumerate(series["paths"]):
        im = Image.open(path).convert("RGB"); draw = ImageDraw.Draw(im)
        # 仅在已检验时间窗口内显示预测带；不把超出视窗的位置压在圆周上。
        if int(cfg.get("background_frames", 5)) <= t and t+1 <= validated_end:
            for j, angle in enumerate(series["angles"]):
                if prediction[t, j] <= 0 or prediction[t, j] > series["limits_px"][j]*mm:
                    continue
                lo, hi = np.maximum(prediction[t, j]-intervals[t, j], 0)/mm, (prediction[t, j]+intervals[t, j])/mm
                hi = min(hi, series["limits_px"][j])
                point = lambda r: (sx+r*np.cos(angle), sy+r*np.sin(angle))
                draw.line((*point(lo), *point(hi)), fill="#ffbd45", width=4)
                x, y = point(prediction[t, j]/mm)
                draw.ellipse((x-4, y-4, x+4, y+4), outline="#ffbd45", width=2)
                if np.isfinite(observed[t, j]) and not series["censored"][t, j]:
                    x, y = point(observed[t, j]/mm)
                    draw.ellipse((x-3, y-3, x+3, y+3), outline="#4bdb90", width=2)
        draw.text((16, 12), f"{t+1:03d} 橙：物理预测带；绿：图像前沿；非真实标注", font=font(20), fill="white", stroke_width=1, stroke_fill="black")
        im.save(out/"overlays"/f"{t+1:06d}.png")
        frames.append(im.resize((640, 400)))
    frames[0].save(out/"物理前沿预测.gif", save_all=True, append_images=frames[1:], duration=100, loop=0, disposal=2)


def main():
    p = argparse.ArgumentParser(description="对灰底处理图建立自由喷雾物理贯穿距预测并做时间留出盲测")
    p.add_argument("--config", required=True); p.add_argument("--cleaned", required=True)
    p.add_argument("--uncertainty"); p.add_argument("--output", required=True)
    p.add_argument("--train-end", type=int, default=12); p.add_argument("--test-end", type=int, default=18)
    p.add_argument("--bootstrap", type=int, default=20)
    args = p.parse_args(); run(args)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
