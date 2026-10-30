"""读取已有结果，定位边界突跳与优先标注帧；不修改识别输出。"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def analyze_jumps(radii, valid_frames, angles, jet_angles, max_step, mm_per_pixel=None):
    events = []
    for t in range(1, len(radii)):
        if not valid_frames[t] or not valid_frames[t-1]:
            continue
        for k, angle in enumerate(angles):
            previous, current = radii[t-1, k], radii[t, k]
            if not np.isfinite(previous) or not np.isfinite(current) or abs(current-previous) <= max_step+1:
                continue
            distance = float(np.min(np.abs((np.asarray(jet_angles)-angle+180) % 360-180))) if len(jet_angles) else 180.
            if previous == 0 or current == 0:
                category = "角向可见性出现或消失"
            elif distance <= 8:
                category = "喷束方向前沿待核查"
            else:
                category = "喷束间隙轮廓待核查"
            events.append({"frame": t+1, "angle_deg": float(angle), "previous_radius_px": float(previous),
                           "current_radius_px": float(current), "delta_px": float(current-previous),
                           "delta_mm": float((current-previous)*mm_per_pixel) if mm_per_pixel else "",
                           "distance_to_jet_deg": distance, "review_hint": category})
    return events


def run(results):
    root = Path(results).resolve()
    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    with (root / "contours.csv").open(encoding="utf-8-sig", newline="") as file:
        contours = list(csv.DictReader(file))
    with (root / "frame_metrics.csv").open(encoding="utf-8-sig", newline="") as file:
        metrics = list(csv.DictReader(file))
    angles = sorted({float(row["angle_deg_clockwise"]) for row in contours})
    angle_index = {angle: i for i, angle in enumerate(angles)}
    radii = np.full((summary["frames"], len(angles)), np.nan)
    for row in contours:
        if row["radius_px"]:
            radii[int(row["frame"])-1, angle_index[float(row["angle_deg_clockwise"])]] = float(row["radius_px"])
    valid = [row.get("status", "observed") == "observed" for row in metrics]
    events = analyze_jumps(radii, valid, angles, summary["jet_angles_deg_clockwise"],
                           summary["max_front_step_px_per_frame"], summary["configuration"].get("mm_per_pixel"))
    counts = Counter(event["review_hint"] for event in events)
    per_frame = Counter(event["frame"] for event in events)
    important = [frame for frame, _ in per_frame.most_common(6)]
    annotation_frames = sorted(set(important + [summary["onset_frame"], min(10, summary["frames"]),
                                                min(20, summary["frames"]), summary["frames"]]))
    report = {"algorithm_result_version": summary.get("version"), "frames": summary["frames"],
              "event_count": len(events), "event_categories": dict(counts),
              "events_by_frame": dict(per_frame), "annotation_priority_frames": annotation_frames,
              "jet_centerline_step_exceedances": summary.get("jet_step_exceedances"),
              "note": "类别仅为人工复核提示，不是误检真值；角向轮廓突跳不能直接解释为流体速度。"}
    (root / "边界诊断.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "边界突跳事件.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["frame", "angle_deg", "previous_radius_px", "current_radius_px", "delta_px", "delta_mm", "distance_to_jet_deg", "review_hint"])
        writer.writeheader(); writer.writerows(events)
    rows = "\n".join(f"| {frame} | {count} |" for frame, count in sorted(per_frame.items()))
    text = f"""# V2后续边界诊断报告

该报告读取已有识别结果，不改动算法和原始图像。

- 输入结果版本：{summary.get('version')}，共{summary['frames']}帧。
- 超过当前逐角度位移诊断上界的事件：{len(events)}个。
- 触发类型：{dict(counts)}。
- 按同一喷束中心方向统计的超界事件：{summary.get('jet_step_exceedances')}个。
- 建议优先标注帧：{', '.join(map(str, annotation_frames))}。

| 当前帧号 | 待核查角度数量 |
|---|---:|
{rows}

这些事件不等于已经确认的识别错误。喷束之间的可见区域出现、消失，可能让某一角度的包络半径突变；其含义与沿同一喷束的前沿位移不同。需要逐个结合原图、反光区和人工边界带检查。

下一步应先检查事件最密集的帧，以及起喷和白色重合的代表帧。记录边界可接受范围后，再评估阈值、窗框过滤和局部补全的收益与代价。无人工标注前，不给出IoU或真实误差结论。
"""
    (root / "V2后续边界诊断报告.md").write_text(text, encoding="utf-8")
    # 只在本机原图仍可用时生成复核联系图，图片不随诊断JSON自动上传。
    source = Path(summary["configuration"]["source_dir"])
    selected = important[:4]
    if selected and source.exists():
        canvas = Image.new("RGB", (1200, 285*len(selected)), "white")
        draw = ImageDraw.Draw(canvas); font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
        sx, sy = summary["configuration"]["source_xy"]
        for r, frame in enumerate(selected):
            with Image.open(source / metrics[frame-2]["source"]) as image:
                previous = image.convert("RGB")
            with Image.open(source / metrics[frame-1]["source"]) as image:
                current = image.convert("RGB")
            overlay = current.copy(); painter = ImageDraw.Draw(overlay)
            for event in [item for item in events if item["frame"] == frame]:
                angle = np.deg2rad(event["angle_deg"])
                for radius, color in [(event["previous_radius_px"], "yellow"), (event["current_radius_px"], "red")]:
                    x, y = sx+radius*np.cos(angle), sy+radius*np.sin(angle)
                    painter.ellipse((x-4, y-4, x+4, y+4), outline=color, width=3)
            for c, (image, title) in enumerate([(previous, f"第{frame-1}帧"), (current, f"第{frame}帧"), (overlay, "黄=前帧半径 红=当前半径")]):
                canvas.paste(image.resize((400, 250)), (c*400, r*285+35))
                draw.text((c*400+6, r*285+3), title, font=font, fill="black")
        canvas.save(root / "边界异常复核.jpg", quality=94)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    run(parser.parse_args().results)
