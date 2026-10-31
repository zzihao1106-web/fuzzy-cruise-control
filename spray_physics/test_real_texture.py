"""验证无光斑时保持细节，以及小喷雾不会被背景数量淹没。"""

import unittest

import numpy as np
import torch

from gray_scene_data import temporal_channels
from real_texture_training import RealTextureRestorer, covariance_gray, pair_loss

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
