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

from deep_data import prepare, SpraySynthesizer
from deep_model import BackgroundUNet
from restore_model import ReferenceRestorer, OpticalRestorer, restoration_loss, optical_loss
from restore_spray import boundary_from_cleaned, clean_input, digest, process, process_batch, train

ROOT = Path(__file__).parent/"_test_tmp"
ROOT.mkdir(exist_ok=True)
torch.set_num_threads(2)


class RestorationTests(unittest.TestCase):
    def test_physical_projection_preserves_visible_positive_changes(self):
        model = OpticalRestorer(width=8, depth=2, physical_projection=True).eval()
        x = torch.rand(1, 4, 32, 32)
        z = model.decompose(x)
        r, n = model(x)
        torch.testing.assert_close(r+n+x[:, 1:2], x[:, :1])
        background = (x[:, 1:2]+z["offset"]).clamp(0, 1)
        self.assertTrue((r+1e-6 >= (x[:, :1]-background).clamp_min(0)).all())

    def test_optical_synthesis_matches_mixing_equation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            p = Path(folder)/"backgrounds.npz"
            b = np.full((64, 64), 0.6, np.float32)
            np.savez(p, pool=b[None], background=b, roi=np.ones_like(b, bool))
            synth = SpraySynthesizer(p, seed=77, curriculum=True)
            for _ in range(8):
                x, _, target = synth.sample(optical=True)
                reconstructed = (1-target["alpha"][0])*(x[1]+target["background_offset"][0])+target["foreground"][0]
                self.assertLess(float(np.abs(reconstructed-x[0]).mean()), 0.02)
                self.assertTrue((target["foreground"] <= target["alpha"]+1e-6).all())

    def test_optical_layers_are_bounded_and_reconstruct(self):
        model = OpticalRestorer(width=8, depth=2)
        x = torch.rand(1, 4, 35, 47)
        layers = model.decompose(x)
        self.assertTrue((layers["foreground"] >= 0).all())
        self.assertTrue((layers["foreground"] <= layers["alpha"]).all())
        torch.testing.assert_close(layers["foreground"]+layers["background_contribution"], layers["reconstruction"])
        r, n = model(x)
        torch.testing.assert_close(r+n+x[:, 1:2], layers["reconstruction"])

    def test_zero_contrast_does_not_make_zero_foreground_valid(self):
        # B=F=0.7时观测不变，但alpha=0.6对应前景层0.42，不能把黑图当恢复正确。
        x = torch.zeros(1, 4, 16, 16); x[:, :2] = 0.7
        alpha = torch.full((1, 1, 16, 16), 0.6)
        target = {"alpha": alpha, "foreground": alpha*0.7, "background_offset": alpha*0}
        good = {"alpha_logits": torch.logit(alpha), "foreground": alpha*0.7, "offset": alpha*0, "reconstruction": x[:, :1]}
        black = {"alpha_logits": torch.full_like(alpha, -20), "foreground": alpha*0, "offset": alpha*0, "reconstruction": x[:, :1]}
        self.assertGreater(float(optical_loss(black, target, x)), float(optical_loss(good, target, x))+0.5)

    def test_optical_training_bundle_and_saved_png_pipeline(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            p = Path(folder); source = p/"raw"; source.mkdir()
            for i in range(4):
                a = np.full((48, 64), 90, np.uint8)
                if i >= 2:
                    a[22:27, 30:48] = 140
                Image.fromarray(a).save(source/f"{i+1:06d}.png")
            cfg = {"source_dir": str(source), "background_frames": 2, "source_xy": [32, 24],
                   "chamber_xy": [32, 24], "chamber_axes_px": [20, 20]}
            config = p/"config.json"; config.write_text(json.dumps(cfg), encoding="utf-8")
            prepare(source, cfg, p/"data", 64)
            with contextlib.redirect_stdout(io.StringIO()):
                train(argparse.Namespace(data=str(p/"data"), output=str(p/"train"), model="optical", baseline=None,
                      steps=2, boundary_steps=2, batch=1, width=8, seed=5, lr=0.001, device="cpu"))
                report = process(argparse.Namespace(config=str(config), checkpoint=str(p/"train/pipeline_best.pt"),
                    input=None, output=str(p/"result"), size=64, device="cpu", reuse_geometry=False))
            self.assertEqual(report["restore_kind"], "optical")
            self.assertFalse(report["boundary"]["raw_images_read_by_boundary_stage"])
            self.assertEqual(len(list((p/"result/restoration_unknown").glob("*.png"))), 4)
            with contextlib.redirect_stdout(io.StringIO()):
                train(argparse.Namespace(data=str(p/"data"), output=str(p/"projected_train"), model="optical", baseline=None,
                      resume_cleaner=str(p/"train/cleaner_best.pt"), physical_projection=True, foreground_gain=1.0,
                      steps=0, boundary_steps=2, batch=1, width=8, seed=5, lr=0.001, device="cpu"))
                projected = process(argparse.Namespace(config=str(config), checkpoint=str(p/"projected_train/pipeline_best.pt"),
                    input=None, output=str(p/"projected_result"), size=64, device="cpu", reuse_geometry=False))
            self.assertTrue(projected["physical_projection"])
            self.assertEqual(projected["fixed_display_gain"], 1)

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
            saved_cfg = json.loads((p/"result/运行配置.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(saved_cfg["output_dir"]).resolve(), (p/"result").resolve())
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
