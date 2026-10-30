"""V3有意义的回归：合成标签、未知监督、共享细化及观测降级基础。"""

import tempfile
import unittest
import argparse
import contextlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from deep_data import SpraySynthesizer, channels, correct_photometry, source_connected
from deep_spray import boundary_f1, score_batch, input_config


class DataTests(unittest.TestCase):
    def test_new_input_does_not_reuse_old_geometry_implicitly(self):
        cfg = {"source_dir": "sequence_a", "source_xy": [20, 20], "chamber_xy": [20, 20], "chamber_axes_px": [15, 15]}
        self.assertIsNone(input_config(cfg, "sequence_b")["source_xy"])
        self.assertEqual(input_config(cfg, "sequence_b", True)["source_xy"], [20, 20])
        self.assertEqual(cfg["source_xy"], [20, 20])
    def test_metrics_do_not_score_unknown_as_background(self):
        prediction = np.ones((32, 32), np.float32)
        target = np.zeros_like(prediction); target[:, 16:] = 1
        valid = np.zeros_like(prediction, bool); valid[:, 16:] = True
        scores = score_batch(prediction, target, valid)
        self.assertEqual(scores["iou"], 1)
        self.assertEqual(scores["fp"], 0)

    def test_boundary_score_and_empty_boundary(self):
        a = np.zeros((40, 40), np.float32); a[12:28, 12:28] = 1
        self.assertEqual(boundary_f1(a, a, np.ones_like(a, bool)), 1)
        self.assertIsNone(boundary_f1(np.zeros_like(a), a, np.ones_like(a, bool)))

    def test_both_polarities_retained(self):
        a = channels(np.array([[0.8, 0.2]], np.float32), np.array([[0.4, 0.4]], np.float32))
        np.testing.assert_allclose(a[2], [[0.4, 0]])
        np.testing.assert_allclose(a[3], [[0, 0.2]])

    def test_static_fit_does_not_absorb_foreground(self):
        rng = np.random.default_rng(1)
        bg = rng.uniform(0.1, 0.7, (64, 64)).astype(np.float32)
        image = bg*1.02+0.01
        image[20:45, 20:45] += 0.2
        static = np.ones_like(bg, bool); static[15:50, 15:50] = False
        corr, spec = correct_photometry(image, bg, static)
        self.assertAlmostEqual(spec["gain"], 1.02, places=4)
        self.assertGreater(float((image-corr)[20:45, 20:45].mean()), 0.19)

    def test_large_uniform_drift_is_bounded(self):
        bg = np.full((64, 64), 0.2, np.float32)
        _, spec = correct_photometry(bg+0.5, bg, np.ones_like(bg, bool))
        self.assertLessEqual(spec["offset"], 0.04)

    def test_source_constraint_removes_remote_blob_without_filling(self):
        m = np.zeros((80, 80), bool); m[38:43, 38:60] = True; m[5:15, 5:15] = True
        result = source_connected(m, (40, 40), 5)
        self.assertEqual(int(result.sum()), 110)
        self.assertFalse((result & ~m).any())

    def test_synthetic_empty_and_reproducible(self):
        # 显式选择工作区临时目录，避免Windows沙箱的系统临时目录权限问题。
        root = Path(__file__).parent/"_test_tmp"; root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            p = Path(folder)/"backgrounds.npz"
            b = np.full((64, 64), 0.4, np.float32)
            np.savez(p, pool=b[None], background=b, roi=np.ones_like(b, bool))
            s1, s2 = SpraySynthesizer(p, seed=42), SpraySynthesizer(p, seed=42)
            x, y = s1.sample(force_empty=True)
            x2, y2 = s2.sample(force_empty=True)
            self.assertEqual(float(y.sum()), 0)
            np.testing.assert_array_equal(x, x2)
            np.testing.assert_array_equal(y, y2)
            self.assertTrue(np.isfinite(x).all())
            hard = SpraySynthesizer(p, seed=42, curriculum=True)
            for _ in range(8):
                difficult_x, difficult_y = hard.sample()
                self.assertTrue(np.isfinite(difficult_x).all())
                self.assertTrue(np.isfinite(difficult_y).all())


try:
    import torch
    from deep_model import BackgroundUNet, supervised_loss
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "需要独立PyTorch环境")
class ModelTests(unittest.TestCase):
    def test_multiscale_restores_original_shape_and_reports_disagreement(self):
        from deep_spray import network_prediction
        model = BackgroundUNet(8, 1).eval()
        stages, fused, disagreement = network_prediction(model, torch.rand(1, 4, 35, 47), 32)
        self.assertEqual(fused.shape, (3, 35, 47))
        self.assertEqual(disagreement.shape, (35, 47))
        self.assertTrue(np.isfinite(fused).all())
        self.assertGreaterEqual(float(disagreement.min()), 0)

    def test_low_contrast_supervision_has_higher_gradient(self):
        z = torch.zeros(1, 3, 16, 16, requires_grad=True)
        target = torch.zeros_like(z); target[:, 0] = 1; target[:, 2, 8, 8] = 1
        supervised_loss([z], target, weak_weight=3).backward()
        self.assertGreater(float(z.grad[0, 0, 8, 8].abs()), float(z.grad[0, 0, 8, 9].abs()))

    def test_training_checkpoint_and_whole_sequence_export(self):
        from deep_data import prepare
        from deep_spray import predict, predict_batch, train
        root = Path(__file__).parent/"_test_tmp"; root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            path = Path(folder)
            source = path/"原图"; source.mkdir()
            for i in range(6):
                image = np.full((48, 64), 80, np.uint8)
                if i >= 2:
                    image[22:27, 32:35+i*2] = 140
                Image.fromarray(image).save(source/f"image_{i+1}.png")
            cfg = {"source_dir": str(source), "background_frames": 2, "source_xy": [32, 24],
                   "chamber_xy": [32, 24], "chamber_axes_px": [20, 20], "selected_frames": [1, 3, 6]}
            data, weights, output = path/"data", path/"weights", path/"output"
            config_path = path/"config.json"
            config_path.write_text(json.dumps(cfg), encoding="utf-8")
            prepare(source, cfg, data, 64)
            options = argparse.Namespace(data=str(data), output=str(weights), device="cpu", steps=2,
                        batch=1, eval_every=2, lr=0.001, seed=7, width=8, iterations=1, resume=None, labels=None)
            with contextlib.redirect_stdout(io.StringIO()):
                train(options)
                summary = predict(argparse.Namespace(config=str(config_path), checkpoint=str(weights/"best.pt"),
                    device="cpu", input=None, output=str(output), size=64, v2=None))
            self.assertEqual(summary["frames"], 6)
            self.assertEqual(summary["checkpoint_training_kind"], "synthetic_only")
            self.assertIsNone(summary["real_boundary_accuracy"])
            self.assertEqual(len(list((output/"probability").glob("*.png"))), 6)
            self.assertEqual(int(np.asarray(Image.open(output/"model_masks/000001.png")).sum()), 0)
            self.assertEqual(len(list(source.glob("*.png"))), 6)
            batch_input = path/"批处理输入"; batch_input.mkdir()
            shutil.copytree(source, batch_input/"有效序列")
            (batch_input/"空文件夹").mkdir()
            batch_output = path/"批处理结果"
            with contextlib.redirect_stdout(io.StringIO()):
                predict_batch(argparse.Namespace(config=str(config_path), checkpoint=str(weights/"best.pt"),
                    device="cpu", input=str(batch_input), output=str(batch_output), size=64, v2=None))
            records = json.loads((batch_output/"deep_batch_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(sum(r["status"] == "ok" for r in records), 1)
            self.assertEqual(sum(r["status"] == "skipped_no_images" for r in records), 1)

    def test_refinement_and_odd_size_gradient(self):
        torch.set_num_threads(2)
        model = BackgroundUNet(8, 2)
        x = torch.rand(1, 4, 35, 47)
        outputs = model(x)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(tuple(outputs[-1].shape), (1, 3, 35, 47))
        loss = supervised_loss(outputs, torch.zeros_like(outputs[-1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.refiner[-1].weight.grad.abs().sum()), 0)

    def test_unknown_pixels_have_zero_gradient(self):
        z = torch.randn(1, 3, 32, 32, requires_grad=True)
        valid = torch.ones(1, 1, 32, 32); valid[..., :16] = 0
        loss = supervised_loss([z], torch.ones_like(z), valid)
        loss.backward()
        self.assertEqual(float(z.grad[..., :16].abs().sum()), 0)
        self.assertGreater(float(z.grad[..., 16:].abs().sum()), 0)

    def test_checkpoint_round_trip(self):
        root = Path(__file__).parent/"_test_tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            model = BackgroundUNet(8, 1).eval()
            x = torch.rand(1, 4, 32, 32)
            path = Path(folder)/"m.pt"
            torch.save({"state_dict": model.state_dict(), "model_spec": model.spec}, path)
            checkpoint = torch.load(path, weights_only=True)
            other = BackgroundUNet(**checkpoint["model_spec"]).eval()
            other.load_state_dict(checkpoint["state_dict"])
            torch.testing.assert_close(model(x)[-1], other(x)[-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
