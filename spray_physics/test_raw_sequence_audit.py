"""检查新特征的适用边界；不以这些受控测试代替真实实验验收。"""

import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from raw_sequence_audit import angular_contrast, candidate_mask, run, temporal_texture


class RawEvidenceTests(unittest.TestCase):
    def test_static_white_pattern_does_not_create_temporal_signal(self):
        yy, xx = np.indices((96, 96))
        image = (80+150*np.exp(-((yy-48)/10)**2)).astype(np.float32)
        response = temporal_texture(np.stack([image]*3))
        self.assertLess(float(response.max()), 1e-4)

    def test_uniform_brightness_change_has_no_spatial_temporal_texture(self):
        images = np.stack([np.full((96, 96), value, np.float32) for value in (80, 90, 100)])
        self.assertLess(float(temporal_texture(images).max()), 0.02)

    def test_static_foreground_is_lost_by_motion_but_retained_by_intensity(self):
        signal = np.zeros((96, 96), np.float32)
        signal[42:55, 48:82] = 40
        window = np.stack([signal]*3)
        motion = temporal_texture(window)/2
        intensity = cv2.GaussianBlur(signal, (0, 0), 0.8)/6
        roi = np.ones(signal.shape, bool)
        mask_motion = candidate_mask(motion, roi, (48, 48), 10)
        mask_fusion = candidate_mask(np.maximum(intensity, motion), roi, (48, 48), 10)
        self.assertFalse(mask_motion.any())
        self.assertTrue(mask_fusion[48, 65])

    def test_angular_feature_rejects_broad_trend_but_preserves_narrow_jet(self):
        yy, xx = np.indices((256, 256))
        r = np.hypot(xx-128, yy-128)
        theta = np.arctan2(yy-128, xx-128)
        broad = (15+10*np.cos(theta)) * (r > 8)
        jet = 70*np.exp(-(theta/0.07)**2) * (r > 20) * (r < 90)
        feature = angular_contrast((broad+jet).astype(np.float32), (128, 128))
        self.assertGreater(float(feature[128, 190]), 40)
        self.assertLess(abs(float(feature[190, 128])), 5)

    def test_angular_feature_cannot_define_symmetric_dense_cloud(self):
        yy, xx = np.indices((256, 256))
        cloud = ((np.hypot(xx-128, yy-128) < 65)*50).astype(np.float32)
        feature = angular_contrast(cloud, (128, 128))
        # 记录明确的失效模式，防止以后把特征低响应解释为没有喷雾。
        self.assertLess(abs(float(feature[128, 160])), 1)

    def test_mask_has_no_forced_growth_or_hole_filling(self):
        score = np.zeros((96, 96), np.float32)
        score[30:70, 30:70] = 2
        score[44:55, 44:55] = 0
        mask = candidate_mask(score, np.ones_like(score, bool), (31, 31), 8)
        self.assertTrue(mask[35, 35])
        self.assertFalse(mask[49, 49])
        empty = candidate_mask(np.zeros_like(score), np.ones_like(score, bool), (31, 31), 8)
        self.assertFalse(empty.any())

    def test_directional_component_is_not_deleted_due_to_low_contrast_root(self):
        score = np.zeros((128, 128), np.float32)
        score[58:69, 90:112] = 2
        roi = np.ones_like(score, bool)
        rooted = candidate_mask(score, roi, (64, 64), 10)
        structure = candidate_mask(score, roi, (64, 64), 10, require_source=False)
        self.assertFalse(rooted.any())
        self.assertTrue(structure[64, 100])

    def test_end_to_end_has_no_future_frame_leakage_and_preserves_input(self):
        workspace = Path(__file__).resolve().parent
        scratch = workspace/"_test_tmp"/"raw_audit"
        if not scratch.resolve().is_relative_to(workspace):
            raise RuntimeError("测试临时目录必须在项目内部。")
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary)
            source = root/"source"; source.mkdir()
            for i in range(8):
                image = np.full((128, 128), 70, np.uint8)
                image[55:74, :] = 180
                if i >= 5:
                    image[40:50, 65:90] = 230
                Image.fromarray(image).save(source/f"{i+1:04d}.png")
            original = {p.name: p.read_bytes() for p in source.iterdir()}
            config = {"source_dir": str(source), "source_xy": [64, 64],
                      "chamber_xy": [64, 64], "chamber_axes_px": [50, 50]}
            a = run(config, root/"a")
            self.assertEqual(original, {p.name: p.read_bytes() for p in source.iterdir()})
            Image.fromarray(np.full((128, 128), 255, np.uint8)).save(source/"0008.png")
            b = run(config, root/"b")
            self.assertEqual(a["intensity_high_gray"], b["intensity_high_gray"])
            for method in ("intensity", "texture", "fusion", "directional"):
                for frame in range(1, 8):
                    self.assertEqual((root/"a"/method/f"{frame:06d}.png").read_bytes(),
                                     (root/"b"/method/f"{frame:06d}.png").read_bytes())


if __name__ == "__main__":
    unittest.main()
