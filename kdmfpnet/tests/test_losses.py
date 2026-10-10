from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kdmfpnet.losses import build_hard_example_weights, ramp_weight, weighted_pressure_loss


class LossTests(unittest.TestCase):
    def test_build_hard_example_weights_focuses_largest_residual(self) -> None:
        prediction = torch.tensor([[[0.0], [0.2], [2.0], [0.1]]], dtype=torch.float32)
        target = torch.zeros_like(prediction)

        weights = build_hard_example_weights(prediction, target, topk_fraction=0.25, hard_point_alpha=2.0)

        self.assertGreater(float(weights[0, 2, 0]), float(weights[0, 0, 0]))

    def test_build_hard_example_weights_focuses_largest_target_magnitude(self) -> None:
        prediction = torch.tensor([[[1.1], [3.0], [1.2], [2.5]]], dtype=torch.float32)
        target = torch.tensor([[[0.1], [2.0], [0.2], [1.5]]], dtype=torch.float32)

        weights = build_hard_example_weights(
            prediction,
            target,
            hard_point_alpha=0.0,
            target_peak_alpha=2.0,
            target_peak_fraction=0.25,
        )

        self.assertGreater(float(weights[0, 1, 0]), float(weights[0, 0, 0]))

    def test_weighted_pressure_loss_is_zero_when_prediction_matches_target(self) -> None:
        prediction = torch.zeros(1, 4, 1)
        target = torch.zeros(1, 4, 1)
        weights = torch.ones_like(target)

        loss = weighted_pressure_loss(prediction, target, weights)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=7)

    def test_ramp_weight_reaches_full_value_after_ramp(self) -> None:
        self.assertEqual(ramp_weight(epoch=0, warmup_epochs=10, ramp_epochs=20, max_value=1.0), 0.0)
        self.assertEqual(ramp_weight(epoch=30, warmup_epochs=10, ramp_epochs=20, max_value=1.0), 1.0)


if __name__ == "__main__":
    unittest.main()
