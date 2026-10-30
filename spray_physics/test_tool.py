"""验证读取、约束求解及不同尺寸/明暗极性的完整序列处理。"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from spray_recognition import discover_sequence, main, optimize_boundary
from recognition_core import EvidenceTracker


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / ("config.json" if (HERE / "config.json").exists() else "config.example.json")
BASE_CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RESULTS = []
TEST_TMP = HERE / "_test_tmp"
TEST_TMP.mkdir(exist_ok=True)


class SolverTests(unittest.TestCase):
    def test_missing_glare_boundary_uses_prior(self):
        # 真值前沿160px，在局部人为擦去90—160px的成像证据。
        data = np.zeros((72, 360), np.float32)
        data[:, :160] = 40
        data[:9, 90:160] = 0
        threshold = np.full_like(data, 10)
        wall = np.full(72, 330.)
        prediction = np.full(72, 160.)
        previous = np.full(72, 155.)
        plain = optimize_boundary(data, threshold, wall, prediction, previous, np.full(72, 5), 35, BASE_CFG, False)
        constrained = optimize_boundary(data, threshold, wall, prediction, previous, np.full(72, 5), 35, BASE_CFG, evidence_weight=np.full(72, 5))
        plain_error = float(np.abs(plain[:9]-160).mean())
        constrained_error = float(np.abs(constrained[:9]-160).mean())
        self.assertLess(constrained_error, plain_error * 0.5)
        RESULTS.append({"check": "合成局部信号缺失", "baseline_boundary_error_px": plain_error,
                        "constrained_boundary_error_px": constrained_error, "synthetic_only": True})

    def test_front_jump_and_wall_limit(self):
        data = np.full((36, 360), 90, np.float32)
        prior = np.full(36, 300.)
        previous = np.full(36, 80.)
        boundary = optimize_boundary(data, np.full_like(data, 10), np.full(36, 82.), prior, previous, np.zeros(36), 6, BASE_CFG)
        self.assertTrue(np.all(boundary <= 82))
        self.assertTrue(np.all(np.abs(boundary-previous) <= 6))

    def test_natural_order_and_photo_exclusion(self):
        with tempfile.TemporaryDirectory(prefix="spray_input_", dir=TEST_TMP) as folder:
            path = Path(folder)
            self.assertTrue(path.resolve().is_relative_to(TEST_TMP.resolve()))
            for number in [10, 2, 1]:
                Image.new("L", (64, 48)).save(path / f"图像_{number}.png")
            Image.new("L", (30, 80)).save(path / "相机设置.png")
            sequence, excluded = discover_sequence(path)
            self.assertEqual([p.stem for p in sequence], ["图像_1", "图像_2", "图像_10"])
            self.assertEqual(len(excluded), 1)


def make_synthetic_sequence(folder, width, n_jets, dark, delay=5):
    height = 512
    yy, xx = np.indices((height, width))
    cx, cy, chamber_radius = width/2, height/2, 211
    radius = np.hypot(xx-cx, yy-cy)
    angle = np.arctan2(yy-cy, xx-cx)
    base = np.full((height, width), 155 if dark else 45, np.float32)
    base += (35 if dark else 110) * np.exp(-((yy-cy)/35)**2) * (radius < chamber_radius)
    base += 30 * np.exp(-((radius-chamber_radius)/2.5)**2)
    jets = np.cos(n_jets * (angle-0.07)) > 0.58
    rng = np.random.default_rng(71+n_jets)
    for t in range(24+delay-5):
        image = base.copy()
        if t >= delay:
            tip = min(33*(t-delay+1)**0.62, chamber_radius-8)
            spray = jets & (radius < tip) & (radius > 9)
            image[spray] += -75 if dark else 75
        image += rng.normal(0, 0.7, image.shape)
        Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(folder / f"image_{t+1}.png")
    Image.new("L", (80, 120), 160).save(folder / "camera_photo.png")


class IntegrationTests(unittest.TestCase):
    def run_case(self, width, jets, dark, delay=5, blank_end=False):
        with tempfile.TemporaryDirectory(prefix="spray_integration_", dir=TEST_TMP) as folder:
            folder = Path(folder)
            self.assertTrue(folder.resolve().is_relative_to(TEST_TMP.resolve()))
            source, output = folder / "input", folder / "output"
            source.mkdir()
            make_synthetic_sequence(source, width, jets, dark, delay)
            if blank_end:
                final_frame = 24+delay-5
                with Image.open(source / "image_1.png") as image:
                    image.save(source / f"image_{final_frame}.png")
            with contextlib.redirect_stdout(io.StringIO()):
                summary = main(CONFIG_PATH, {
                    "source_dir": str(source), "output_dir": str(output), "source_xy": None,
                    "chamber_xy": None, "chamber_axes_px": None, "mm_per_pixel": None,
                    "radial_bins": None, "known_diameter_mm": 60, "signal_polarity": "auto",
                })
            self.assertEqual(summary["frames"], 24+delay-5)
            self.assertEqual(summary["onset_frame"], delay+1)
            self.assertEqual(summary["jet_count_from_image"], jets)
            self.assertEqual(summary["detected_signal_polarity"], "dark" if dark else "bright")
            self.assertIsInstance(summary["step_constraint_violations"], int)
            self.assertEqual(summary["version"], "2.0.0")
            self.assertEqual(len(list((output / "envelope_masks").glob("*.png"))), 24+delay-5)
            self.assertEqual(len(summary["excluded_different_resolution_images"]), 1)
            self.assertTrue((output / "jet_metrics.csv").exists())
            if blank_end:
                import csv
                self.assertEqual(summary["unknown_frames"], [final_frame])
                with (output / "frame_metrics.csv").open(encoding="utf-8-sig", newline="") as file:
                    last = list(csv.DictReader(file))[-1]
                self.assertEqual(last["envelope_area_px"], "")
                self.assertEqual(last["measurement_valid"], "False")
                with (output / "jet_metrics.csv").open(encoding="utf-8-sig", newline="") as file:
                    rows = [r for r in csv.DictReader(file) if int(r["frame"]) == final_frame]
                self.assertTrue(all(r["smoothed_projected_speed_m_s_assumed"] == "" for r in rows))
            for name in ["frame_metrics.csv", "contours.csv", "jet_metrics.csv"]:
                self.assertNotIn("nan", (output / name).read_text(encoding="utf-8-sig").lower())
            RESULTS.append({"check": "完整自动读取与识别", "width": width, "height": 512,
                            "jets": jets, "polarity": summary["detected_signal_polarity"],
                            "frames": 24+delay-5, "onset_frame":delay+1, "ok": True, "synthetic_only": True})

    def test_bright_six_jets(self):
        self.run_case(512, 6, False)

    def test_dark_eight_jets_wider_image(self):
        self.run_case(640, 8, True)

    def test_late_onset_dark_spray(self):
        self.run_case(512, 6, True, delay=15)

    def test_blank_last_frame_has_no_exported_measurement(self):
        self.run_case(512, 6, False, blank_end=True)

    def test_cli_batch_continues_after_invalid_folder(self):
        with tempfile.TemporaryDirectory(prefix="spray_batch_", dir=TEST_TMP) as temporary:
            base = Path(temporary)
            self.assertTrue(base.resolve().is_relative_to(TEST_TMP.resolve()))
            inputs, output = base / "inputs", base / "outputs"
            inputs.mkdir()
            (inputs / "空目录").mkdir()
            case = inputs / "其他尺寸亮喷雾"; case.mkdir()
            make_synthetic_sequence(case, 512, 6, False)
            command = [sys.executable, str(HERE / "spray_recognition.py"), "--batch", "--config", str(CONFIG_PATH), "--input", str(inputs),
                       "--output", str(output), "--diameter-mm", "60", "--fps", "5000"]
            env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            outcomes = json.loads((output / "batch_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(sum(c["ok"] for c in outcomes), 1)
            self.assertEqual(sum(not c["ok"] for c in outcomes), 1)
            summary = json.loads((output / "其他尺寸亮喷雾" / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["configuration"]["fps"], 5000)
            RESULTS.append({"check": "命令行批处理容错", "success": 1, "skipped_invalid": 1, "ok": True})


class V2RegressionTests(unittest.TestCase):
    def tracker(self):
        yy, xx = np.indices((320, 320))
        radius = np.hypot(xx-160, yy-160)
        roi = radius < 149
        cfg = dict(BASE_CFG, max_completion_px=14, mm_per_pixel=None)
        return EvidenceTracker(roi, (160, 160), np.full(radius.shape, 10.), 360, 160, 25, cfg), radius

    def test_clear_start_is_not_cut_by_model(self):
        tracker, radius = self.tracker()
        residual = np.where(radius < 24, 40., 0.)
        result = tracker.process(residual, np.full(360, 149.), np.zeros(360), np.ones(360))
        self.assertEqual(int((result.evidence & (residual > 30)).sum()), int((residual > 30).sum()))
        self.assertTrue(np.all(result.envelope[result.evidence]))

    def test_no_observation_produces_unknown_not_measurement(self):
        tracker, radius = self.tracker()
        observed = np.where(radius < 80, 40., 0.)
        tracker.process(observed, np.full(360, 149.))
        result = tracker.process(np.zeros_like(radius), np.full(360, 149.), np.full(360, 145.), np.ones(360))
        self.assertFalse(result.measurement_valid)
        self.assertEqual(result.status, "unknown_no_observation")
        self.assertEqual(int(result.envelope.sum()), 0)
        self.assertGreater(int(result.unknown.sum()), 1000)

    def test_strong_observation_can_override_growth_prior(self):
        tracker, radius = self.tracker()
        residual = np.where(radius < 100, 40., 0.)
        for _ in range(4):
            result = tracker.process(residual, np.full(360, 149.), np.full(360, 20.), np.ones(360))
        self.assertEqual(int(result.evidence.sum()), int((residual > 30).sum()))
        self.assertGreater(np.median(result.boundary), 95)

    def test_model_only_fills_small_currently_supported_gap(self):
        tracker, radius = self.tracker()
        yy, xx = np.indices(radius.shape)
        angles = np.arctan2(yy-160, xx-160)
        residual = np.where(radius < 90, 40., 0.)
        wall = np.full(360, 149.)
        for _ in range(3):
            tracker.process(residual, wall)
        weak = residual.copy()
        band = (np.abs(angles) < np.deg2rad(1.5)) & (radius >= 80) & (radius < 90)
        weak[band] = 8
        result = tracker.process(weak, wall, np.full(360, 90.), np.ones(360), np.ones(360))
        self.assertGreater(int(result.model_added.sum()), 0)
        self.assertLess(np.max(result.boundary), 93)

    def test_unrelated_remote_reflection_is_not_a_spray_seed(self):
        tracker, radius = self.tracker()
        residual = np.where(radius < 30, 40., 0.)
        residual[50:60, 150:170] = 50
        result = tracker.process(residual, np.full(360, 149.))
        self.assertEqual(int(result.evidence[50:60, 150:170].sum()), 0)

    def test_thin_rim_and_bridge_are_rejected(self):
        tracker, radius = self.tracker()
        residual = np.where(radius < 60, 40., 0.)
        residual[(radius > 132) & (radius < 135)] = 25
        residual[159:162, 215:295] = 15
        result = tracker.process(residual, np.full(360, 149.))
        self.assertLess(int(result.evidence[radius > 125].sum()), 10)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "checks": RESULTS, "note": "合成结果用于验证流程和约束，不能作为真实喷雾准确率。"}
    (HERE / "验证结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
