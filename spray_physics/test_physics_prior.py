"""检查压力基准、温度标识、渐近物理规律和留出方向信息隔离。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from physics_prior import air_scenarios, penetration, extract_fronts

ROOT = Path(__file__).parent/"_test_tmp"
ROOT.mkdir(exist_ok=True)


class PhysicsPriorTests(unittest.TestCase):
    def config(self):
        return {"ambient_gas": "air", "ambient_temperature_K": 298.15,
                "ambient_pressure_bar": 40, "injection_pressure_MPa_label": 140,
                "nozzle_diameter_mm": 0.23, "ambient_pressure_basis": None,
                "injection_pressure_basis": None}

    def test_unknown_pressure_basis_retains_four_scenarios(self):
        scenarios = air_scenarios(self.config())
        self.assertEqual(len(scenarios), 4)
        self.assertAlmostEqual(scenarios[0]["air_density_kg_m3"], 46.737731, places=5)
        self.assertGreater(scenarios[2]["air_density_kg_m3"], scenarios[0]["air_density_kg_m3"])
        self.assertEqual(scenarios[0]["delta_pa"], 136e6)

    def test_known_pressure_basis_is_respected(self):
        cfg = self.config(); cfg.update(ambient_pressure_basis="gauge", injection_pressure_basis="gauge")
        scenarios = air_scenarios(cfg)
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0]["ambient_absolute_pa"], 4101325)
        self.assertEqual(scenarios[0]["delta_pa"], 136e6)

    def test_bad_temperature_is_rejected(self):
        cfg = self.config(); cfg["ambient_temperature_K"] = 0
        with self.assertRaises(ValueError):
            air_scenarios(cfg)

    def test_early_linear_late_square_root_and_no_viewport_plateau(self):
        early = penetration(np.array([0.00001, 0.00002]), 1.5, 0.1, 0, 0.59)
        late = penetration(np.array([1e4, 4e4]), 1.5, 0.1, 0, 0.59)
        self.assertAlmostEqual(float(early[1]/early[0]), 2, places=3)
        self.assertAlmostEqual(float(late[1]/late[0]), 2, places=4)
        self.assertGreater(float(late[0]), 30)
        self.assertEqual(float(penetration(-0.1, 1.5, 0.1, 0, 0.59)), 0)

    def test_future_images_do_not_change_fitted_directions(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            p = Path(folder)
            cfg = {"source_xy": [64, 64], "chamber_xy": [64, 64], "chamber_axes_px": [52, 52], "background_frames": 3}
            yy, xx = np.indices((128, 128)); dx, dy = xx-64, yy-64
            radius, angle = np.hypot(dx, dy), np.arctan2(dy, dx)
            for variant in (0, 1):
                directory = p/str(variant); directory.mkdir()
                for frame in range(1, 15):
                    image = np.full((128, 128), 70, np.uint8)
                    if frame > 3:
                        offset = np.pi/4 if variant and frame > 8 else 0
                        shape = np.zeros_like(radius, bool)
                        for direction in np.arange(4)*np.pi/2+offset:
                            delta = np.arctan2(np.sin(angle-direction), np.cos(angle-direction))
                            shape |= (np.abs(delta) < 0.14) & (radius < 13+(frame-3)*3)
                        image[shape] = 180
                    Image.fromarray(image).save(directory/f"{frame:06d}.png")
            a = extract_fronts(p/"0", cfg, train_end=8)
            b = extract_fronts(p/"1", cfg, train_end=8)
            np.testing.assert_array_equal(a["peaks"], b["peaks"])
            self.assertGreaterEqual(len(a["peaks"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
