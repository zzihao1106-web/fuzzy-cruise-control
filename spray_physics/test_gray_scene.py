"""验证灰底替换不抹去不透明喷雾、串行输入及独立空背景检查。"""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gray_scene_model import TemporalGrayRestorer
from gray_scene_data import temporal_channels
from gray_scene import prepare_gray, process_gray, train_gray
from gray_scene_boundary import gray_boundary

ROOT = Path(__file__).parent/"_test_tmp"
ROOT.mkdir(exist_ok=True)
torch.set_num_threads(2)


class GraySceneTests(unittest.TestCase):
    def test_same_spray_under_different_background_lights(self):
        model = TemporalGrayRestorer(width=8, depth=2).eval()
        alpha, foreground, gray = 0.6, 0.9, 0.25
        with torch.no_grad():
            model.ending.bias[0] = torch.logit(torch.tensor(alpha))
        estimates = []
        for light in (0.3, 0.7):
            b = np.full((35, 47), light, np.float32)
            image = (1-alpha)*b+alpha*foreground
            x = temporal_channels(image, image, image, b, np.full_like(b, gray))
            with torch.inference_mode():
                estimates.append(model(torch.from_numpy(x[None])))
        torch.testing.assert_close(estimates[0], estimates[1])
        self.assertAlmostEqual(float(estimates[0].mean()), (1-alpha)*gray+alpha*foreground, places=5)

    def test_opaque_foreground_survives_replacement(self):
        model = TemporalGrayRestorer(width=8, depth=2).eval()
        with torch.no_grad():
            model.ending.bias[0] = 20
        image = np.full((32, 32), 0.7, np.float32)
        x = temporal_channels(image, image, image, image, np.full_like(image, 0.25))
        with torch.inference_mode():
            result = model(torch.from_numpy(x[None])).numpy()[0, 0]
        np.testing.assert_allclose(result, image, atol=1e-6)

    def test_boundary_has_no_original_image_dependency(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            p = Path(folder); frames = p/"gray"; frames.mkdir()
            for i in range(6):
                image = np.full((96, 128), 70, np.uint8)
                if i == 5:
                    image[45:52, 65:95] = 140
                Image.fromarray(image).save(frames/f"{i+1:06d}.png")
            cfg = {"source_dir": "missing_original_folder", "background_frames": 5,
                   "source_xy": [64, 48], "chamber_xy": [64, 48], "chamber_axes_px": [43, 43]}
            summary = gray_boundary(frames, p/"boundary", cfg)
            self.assertFalse(summary["raw_images_read_by_boundary_stage"])
            self.assertEqual(summary["empty_frame_false_positive_pixels"], [0]*5)
            record = json.loads((p/"boundary/boundary_inputs.json").read_text(encoding="utf-8"))[-1]
            self.assertGreater(record["mask_pixels"], 0)
            self.assertFalse(record["quantitative_measurement_valid"])

    def test_prepare_train_save_then_segment_pipeline(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            p = Path(folder); source = p/"raw"; source.mkdir()
            yy, xx = np.indices((96, 128))
            for i in range(7):
                im = np.full((96, 128), 70, np.uint8)
                im[(abs(yy-48) < 9) & (abs(xx-64) > 20)] = 160
                if i >= 5:
                    im[44:53, 65:86+i] = 200
                Image.fromarray(im).save(source/f"{i+1:06d}.png")
            cfg = {"source_dir": str(source), "background_frames": 5,
                   "source_xy": [64, 48], "chamber_xy": [64, 48], "chamber_axes_px": [43, 43]}
            config = p/"cfg.json"; config.write_text(json.dumps(cfg), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_gray(argparse.Namespace(config=str(config), input=None, output=str(p/"data"), size=64, reuse_geometry=False))
                train_gray(argparse.Namespace(data=str(p/"data"), output=str(p/"train"), steps=2, batch=1,
                           width=8, lr=0.001, seed=12, device="cpu", validate_every=2))
                summary = process_gray(argparse.Namespace(config=str(config), input=None, output=str(p/"result"),
                           checkpoint=str(p/"train/gray_best.pt"), size=64, device="cpu", reuse_geometry=False))
            self.assertEqual(summary["frames"], 7)
            self.assertFalse(summary["boundary"]["raw_images_read_by_boundary_stage"])
            self.assertFalse(summary["real_accuracy_available"])
            self.assertEqual(len(list((p/"result/gray_frames").glob("*.png"))), 7)
            original = np.asarray(Image.open(source/"000007.png"))
            corrected = np.asarray(Image.open(p/"result/gray_frames/000007.png"))
            np.testing.assert_array_equal(original[46:51, 62:67], corrected[46:51, 62:67])
            self.assertGreater(float(corrected[20:75, 40:90].mean()), 20)
            with Image.open(p/"result/灰底场景动画.gif") as animation:
                self.assertEqual(animation.n_frames, 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
