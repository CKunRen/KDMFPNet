from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kdmfpnet.physics_gate import case_factor, local_bias_compensation_cp


class LocalBiasCompensationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points = np.array(
            [[0.0, 0.0, 0.0], [0.002, 0.01, 0.0], [0.2, 0.02, 0.0]],
            dtype=np.float64,
        )
        self.baseline = np.array([1.2, 0.9, 0.2])

    def test_preserves_shape_and_finiteness(self) -> None:
        result = local_bias_compensation_cp(self.points, self.baseline, mach=3.5, nose_tip_radius_m=0.0045)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.isfinite(result).all())

    def test_inactive_tail_point_is_unchanged(self) -> None:
        result = local_bias_compensation_cp(self.points, self.baseline, mach=3.5, nose_tip_radius_m=0.0045)
        self.assertAlmostEqual(float(result[2]), float(self.baseline[2]))

    def test_active_nose_point_receives_bounded_positive_offset(self) -> None:
        result = local_bias_compensation_cp(self.points, self.baseline, mach=3.5, nose_tip_radius_m=0.0045)
        gate = case_factor(3.5, 0.0045)  # point gate is 1 at s=0 with baseline>0.8
        expected = float(self.baseline[0]) + gate * 0.2
        self.assertAlmostEqual(float(result[0]), expected, places=9)
        self.assertGreater(float(result[0]), float(self.baseline[0]))

    def test_subthreshold_condition_leaves_prediction_unchanged(self) -> None:
        # Mach below the activation onset -> case factor zero everywhere.
        result = local_bias_compensation_cp(self.points, self.baseline, mach=2.0, nose_tip_radius_m=0.006)
        self.assertTrue(np.allclose(result, self.baseline))

    def test_mismatched_prediction_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            local_bias_compensation_cp(self.points, self.baseline[:2], mach=3.5, nose_tip_radius_m=0.0045)


if __name__ == "__main__":
    unittest.main()
