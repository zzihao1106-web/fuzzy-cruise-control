"""验证双图恢复、喷雾保留目标及边界只消费已落盘去白图的合同。"""

import argparse
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from deep_data import prepare
from deep_model import BackgroundUNet
from restore_model import ReferenceRestorer, restoration_loss
from restore_spray import boundary_from_cleaned, clean_input, digest, process, process_batch, train

ROOT = Path(__file__).parent/"_test_tmp"
ROOT.mkdir(exist_ok=True)
torch.set_num_threads(2)


class RestorationTests(unittest.TestCase):
    def test_reference_reconstruction_and_odd_size_gradient(self):
        model = ReferenceRestorer(width=8, depth=2)
        x = torch.rand(1, 4, 35, 47)
        restored, nuisance = model(x)
        self.assertEqual(tuple(restored.shape), (1, 1, 35, 47))
        torch.testing.assert_close(restored+nuisance+x[:, 1:2], x[:, :1])
        loss = restored.square().mean(); loss.backward()
        self.assertTrue(torch.isfinite(model.ending.weight.grad).all())
        self.assertGreater(float(model.ending.weight.grad.abs().sum()), 0)

    def test_erasing_spray_is_penalized(self):
        truth = torch.zeros(1, 1, 32, 32); truth[..., 12:20, 12:20] = 0.2
        semantic = torch.zeros(1, 3, 32, 32); semantic[:, :1] = truth > 0
        good = restoration_loss(truth, truth, semantic)
        black = restoration_loss(torch.zeros_like(truth), truth, semantic)
        self.assertGreater(float(black), float(good)+0.1)

    def test_positive_and_negative_spray_remain_visible(self):
        result = clean_input(np.array([[-0.1, 0, 0.1]], np.float32), 3)
        np.testing.assert_allclose(result, [[0.3, 0, 0.3]])

    def test_boundary_runs_without_any_raw_image_access(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            p = Path(folder); frames = p/"cleaned"; frames.mkdir()
            for i in range(3):
                a = np.zeros((64, 64), np.uint8)
                if i == 2:
                    a[30:35, 31:50] = 180
                Image.fromarray(a).save(frames/f"{i+1:06d}.png")
            cfg = {"source_dir": "this_raw_folder_does_not_exist", "background_frames": 2,
                   "source_xy": [32, 32], "chamber_xy": [32, 32], "chamber_axes_px": [28, 28], "restoration_size": 64}
            report = boundary_from_cleaned(frames, p/"boundary", BackgroundUNet(8, 1).eval(), cfg, torch.device("cpu"))
            self.assertFalse(report["raw_images_read_by_boundary_stage"])
            inputs = json.loads((p/"boundary/boundary_inputs.json").read_text(encoding="utf-8"))
            self.assertEqual(inputs[2]["cleaned_input_sha256"], digest(frames/"000003.png"))
            with Image.open(p/"boundary/去白后边界动画.gif") as im:
                self.assertEqual(im.n_frames, 3)

    def test_train_restore_save_then_boundary_full_pipeline(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            p = Path(folder); source = p/"原图"; source.mkdir()
            for i in range(6):
                a = np.full((64, 64), 70, np.uint8)
                a[28:37, 4:22] = 180; a[28:37, 43:60] = 180
                if i >= 2:
                    a[30:35, 30:37+i*2] += 40
                Image.fromarray(a).save(source/f"{i+1:06d}.png")
            cfg = {"source_dir": str(source), "background_frames": 2, "source_xy": [32, 32],
                   "chamber_xy": [32, 32], "chamber_axes_px": [29, 29]}
            config = p/"config.json"; config.write_text(json.dumps(cfg), encoding="utf-8")
            prepare(source, cfg, p/"data", 64)
            with contextlib.redirect_stdout(io.StringIO()):
                train(argparse.Namespace(data=str(p/"data"), output=str(p/"train"), steps=2,
                    boundary_steps=2, batch=1, width=8, seed=5, lr=0.001, device="cpu"))
                summary = process(argparse.Namespace(config=str(config), checkpoint=str(p/"train/pipeline_best.pt"),
                    input=None, output=str(p/"result"), size=64, device="cpu", reuse_geometry=False))
            self.assertFalse(summary["segmentation_mask_used_to_clean_image"])
            self.assertEqual(summary["boundary"]["frames"], 6)
            self.assertEqual(len(list((p/"result/cleaned_frames").glob("*.png"))), 6)
            with Image.open(p/"result/逐帧深度去白.gif") as im:
                self.assertEqual(im.n_frames, 6)
            self.assertEqual(len(list(source.glob("*.png"))), 6)
            batch_input = p/"batch_input"; batch_input.mkdir()
            shutil.copytree(source, batch_input/"valid")
            (batch_input/"empty").mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                process_batch(argparse.Namespace(config=str(config), checkpoint=str(p/"train/pipeline_best.pt"),
                    input=str(batch_input), output=str(p/"batch_output"), size=64, device="cpu", reuse_geometry=False))
            records = json.loads((p/"batch_output/restoration_batch.json").read_text(encoding="utf-8"))
            self.assertEqual(sum(r["status"] == "ok" for r in records), 1)
            self.assertEqual(sum(r["status"] == "skipped_no_images" for r in records), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
