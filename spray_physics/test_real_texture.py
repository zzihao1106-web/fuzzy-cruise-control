"""验证无光斑时保持细节，以及小喷雾不会被背景数量淹没。"""

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

from gray_scene_data import temporal_channels
from real_texture_training import RealTextureRestorer, covariance_gray, pair_loss
from spray_preservation import preserve_source_spray
from gray_scene import process_gray

torch.set_num_threads(2)


class RealTextureTests(unittest.TestCase):
    def test_no_highlight_keeps_every_detail(self):
        model = RealTextureRestorer(width=8, depth=2).eval()
        rng = np.random.default_rng(12)
        image = rng.uniform(0.1, 0.9, (35, 47)).astype(np.float32)
        background = np.full_like(image, 0.27)
        x = temporal_channels(image, image, image, background, background)
        with torch.inference_mode():
            output = model(torch.from_numpy(x[None])).numpy()[0, 0]
        np.testing.assert_array_equal(output, image)

    def test_correlation_empty_scene_returns_gray(self):
        yy, xx = np.indices((64, 64), dtype=np.float32)
        b = 0.27+0.5*np.exp(-((xx-30)/20)**2-((yy-30)/8)**2)
        gray = np.full_like(b, 0.27)
        x = temporal_channels(b, b, b, b, gray)[None]
        np.testing.assert_allclose(covariance_gray(x)[0, 0], gray, atol=3e-5)

    def test_erasing_small_spray_does_not_win_by_background_count(self):
        x = torch.zeros(1, 9, 64, 64); x[:, 4:5] = 0.27
        target = torch.full((1, 1, 64, 64), 0.27)
        target[..., 30:33, 30:33] = 0.8
        t = torch.ones_like(target); t[..., 30:33, 30:33] = 0.2
        truth = {"gray": target, "transmission": t, "valid": torch.ones_like(t)}
        correct = {"gray": target, "transmission": t}
        erased = {"gray": target.clone(), "transmission": torch.ones_like(t)}
        erased["gray"][..., 30:33, 30:33] = 0.27
        self.assertGreater(float(pair_loss(erased, truth, x)), float(pair_loss(correct, truth, x))+1)

    def guard(self, image, background, gray):
        shape = image.shape
        return preserve_source_spray(image, image, image, background, gray,
                                     np.full(shape, 0.95, np.float32), np.full(shape, 0.004, np.float32),
                                     {"source_xy": [shape[1]/2, shape[0]/2], "chamber_axes_px": [40, 40]},
                                     np.ones(shape, bool))

    def test_empty_background_does_not_activate_preservation(self):
        b = np.full((96, 96), 0.38, np.float32)
        t, change, unknown = self.guard(b, b, np.full_like(b, 0.27))
        np.testing.assert_allclose(t, 0.95)
        self.assertFalse(change.any()); self.assertFalse(unknown.any())

    def test_strong_white_spot_is_not_pasted_back(self):
        b = np.full((96, 96), 0.8, np.float32)
        t, change, _ = self.guard(b+0.12, b, np.full_like(b, 0.27))
        np.testing.assert_allclose(t, 0.95)
        self.assertFalse(change.any())

    def test_supported_inner_spray_limits_background_subtraction(self):
        yy, xx = np.indices((96, 96))
        b = np.full((96, 96), 0.38, np.float32); g = np.full_like(b, 0.27)
        image = b+0.14*np.exp(-((xx-48)**2+(yy-48)**2)/400)
        t, change, unknown = self.guard(image.astype(np.float32), b, g)
        self.assertGreater(float(change.max()), 0.02)
        self.assertTrue(unknown.any())
        self.assertTrue((image-t*(b-g) >= image-0.95*(b-g)-1e-7).all())

    def test_remote_isolated_change_does_not_activate_source_guard(self):
        yy, xx = np.indices((96, 96))
        b = np.full((96, 96), 0.38, np.float32)
        image = b+0.16*np.exp(-((xx-86)**2+(yy-48)**2)/12)
        t, change, _ = self.guard(image.astype(np.float32), b, np.full_like(b, 0.27))
        np.testing.assert_allclose(t, 0.95)
        self.assertFalse(change.any())

    def test_guard_bundle_saved_scene_pipeline(self):
        root = Path(__file__).parent/"_test_tmp"; root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            p = Path(folder); source = p/"raw"; source.mkdir()
            yy, xx = np.indices((96, 128)); radius = np.hypot(xx-64, yy-48)
            for i in range(7):
                image = 70+18*np.exp(-radius**2/800)
                image[(abs(yy-48) < 8) & (abs(xx-64) > 24)] = 170
                if i >= 5:
                    image += 45*np.exp(-radius**2/600)
                Image.fromarray(np.uint8(np.clip(image, 0, 255))).save(source/f"{i+1:06d}.png")
            cfg = {"source_dir": str(source), "background_frames": 5, "source_xy": [64, 48],
                   "chamber_xy": [64, 48], "chamber_axes_px": [43, 43]}
            (p/"config.json").write_text(json.dumps(cfg), encoding="utf-8")
            model = RealTextureRestorer(width=8, depth=2)
            torch.save({"version": "6.1-test", "kind": "counterfactual_real_texture", "spec": model.spec,
                        "model": model.state_dict(), "step": 0, "preservation_guard_enabled": True}, p/"model.pt")
            with contextlib.redirect_stdout(io.StringIO()):
                summary = process_gray(argparse.Namespace(config=str(p/"config.json"), input=None,
                          output=str(p/"result"), checkpoint=str(p/"model.pt"), size=64, device="cpu", reuse_geometry=False))
            self.assertTrue(summary["spray_preservation_enabled"])
            self.assertEqual(len(list((p/"result/preservation_adjustment").glob("*.png"))), 7)
            original = np.asarray(Image.open(source/"000007.png"))
            corrected = np.asarray(Image.open(p/"result/gray_frames/000007.png"))
            np.testing.assert_array_equal(original[46:51, 62:67], corrected[46:51, 62:67])
            self.assertFalse(summary["boundary"]["raw_images_read_by_boundary_stage"])

    def test_zero_strength_preserves_legacy_transmission(self):
        b = np.full((96, 96), 0.4, np.float32); image = b+0.15
        t = np.full_like(b, 0.9)
        guarded, change, _ = preserve_source_spray(image, image, image, b, b*0+0.27, t, b*0+0.004,
                         {"source_xy": [48, 48], "chamber_axes_px": [40, 40], "source_preservation_strength": 0}, np.ones_like(b, bool))
        np.testing.assert_array_equal(guarded, t)
        self.assertFalse(change.any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
