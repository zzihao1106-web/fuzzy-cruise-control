"""可复现的V2检查：强变化保留与末帧无信号反事实，均不冒充人工准确率。"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spray_recognition import main


def run(config_path, output=None, baseline=None):
    checks = []
    counterfactual = {}

    def inspect(t, result, context):
        cfg = context["config"]
        yy, xx = np.indices(context["raw"].shape)
        cx, cy = cfg["chamber_xy"]; ax, ay = cfg["chamber_axes_px"]
        inner = ((xx-cx)/ax)**2+((yy-cy)/ay)**2 < 0.85**2
        change = context["raw"]-context["background"]
        if context["signal_polarity"] == "dark":
            change = -change
        strong = (change > 30) & inner
        count = int(strong.sum())
        entry = {"frame": t+1, "strong_change_pixels": count,
                 "v2_retained": int((strong & result.envelope).sum()),
                 "v2_retained_fraction": float((strong & result.envelope).sum()/count) if count else None,
                 "model_added_pixels": int(result.model_added.sum()), "unknown_pixels": int(result.unknown.sum())}
        if baseline:
            path = Path(baseline) / "envelope_masks" / f"{t+1:06d}.png"
            if path.exists():
                with Image.open(path) as image:
                    mask = np.asarray(image) > 0
                entry["v1_retained_fraction"] = float((mask & strong).sum()/count) if count else None
        checks.append(entry)
        if t+1 == context["frame_count"]:
            replaced = context["tracker_before"].process(np.zeros_like(change), context["wall"], context["prediction"],
                                                         context["model_strength"], context["glare_fraction"])
            counterfactual.update(status=replaced.status, measurement_valid=replaced.measurement_valid,
                                  envelope_pixels=int(replaced.envelope.sum()), unknown_prediction_pixels=int(replaced.unknown.sum()),
                                  note="只替换末帧的变化信号，保留此前轨迹；这是压力测试，不是真实准确率。")

    changes = {"output_dir": str(Path(output).resolve())} if output else {}
    summary = main(config_path, changes, frame_observer=inspect)
    out = Path(summary["output_directory"])
    report = {"version": summary["version"], "frames": summary["frames"], "checks": checks,
              "counterfactual": counterfactual, "annotation_accuracy": None,
              "note": "强变化保留不是IoU，不验证误检；需结合对照图与人工标注。"}
    (out / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    canvas = Image.new("RGB", (1200, 580), "white"); draw = ImageDraw.Draw(canvas)
    source = Path(summary["configuration"]["source_dir"])
    for row, frame in enumerate([summary["onset_frame"], min(20, summary["frames"]) ]):
        # 输入映射来自逐帧表，兼容不同文件命名方式。
        import csv
        with (out / "frame_metrics.csv").open(encoding="utf-8-sig", newline="") as file:
            mapping = list(csv.DictReader(file))
        images = [source / mapping[frame-1]["source"],
                  Path(baseline) / "overlays" / f"{frame:06d}.png" if baseline else out / "background_removed" / f"{frame:06d}.png",
                  out / "overlays" / f"{frame:06d}.png"]
        labels = [f"第{frame}帧 原图", "V1识别" if baseline else "背景变化", "V2：观测优先，预测单独标记"]
        for column, (path, label) in enumerate(zip(images, labels)):
            if path.exists():
                with Image.open(path) as image:
                    if row == 0:
                        sx, sy = summary["configuration"]["source_xy"]
                        image = image.crop((sx-160, sy-100, sx+160, sy+100))
                    image = image.convert("RGB").resize((400, 250))
                    canvas.paste(image, (400*column, row*290+35))
            draw.text((column*400+6, row*290+4), label, font=font, fill="black")
    canvas.save(out / "V1_V2效果对照.jpg", quality=95)
    print(json.dumps({"起喷检查": checks[summary["onset_frame"]-1], "无信号检查": counterfactual}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--baseline")
    arguments = parser.parse_args()
    run(arguments.config, arguments.output, arguments.baseline)
