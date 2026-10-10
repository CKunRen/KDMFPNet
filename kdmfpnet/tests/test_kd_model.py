from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kdmfpnet.kd_model import EdgeConvBlock, build_student_model, build_teacher_model


class EdgeConvBlockTests(unittest.TestCase):
    def test_output_shape_matches_input_channels(self) -> None:
        block = EdgeConvBlock(in_channels=8, out_channels=8, hidden=16, k=4)
        points = torch.randn(2, 20, 3)
        feat = torch.randn(2, 8, 20)

        out = block(points, feat)

        self.assertEqual(tuple(out.shape), (2, 8, 20))

    def test_ignores_padding_and_clamps_k_to_valid_neighbours(self) -> None:
        # 6 valid points in row 0 (fewer than k=16), all-valid in row 1.
        block = EdgeConvBlock(in_channels=4, out_channels=4, hidden=8, k=16)
        points = torch.randn(2, 20, 3)
        feat = torch.randn(2, 4, 20)
        mask = torch.zeros(2, 20, dtype=torch.bool)
        mask[0, :6] = True
        mask[1, :] = True

        out = block(points, feat, mask=mask)

        self.assertEqual(tuple(out.shape), (2, 4, 20))
        self.assertTrue(torch.isfinite(out).all())

    def test_gradients_flow_to_input_features(self) -> None:
        block = EdgeConvBlock(in_channels=6, out_channels=6, hidden=8, k=4)
        points = torch.randn(1, 10, 3)
        feat = torch.randn(1, 6, 10, requires_grad=True)

        out = block(points, feat)
        out.sum().backward()

        self.assertIsNotNone(feat.grad)
        self.assertGreater(feat.grad.abs().sum().item(), 0.0)

    def test_precomputed_knn_idx_matches_online_search(self) -> None:
        # kd_data.py precomputes knn_idx once with cKDTree on physical
        # coordinates to skip the ~74ms/step O(N^2) online search inside the
        # training loop (see EdgeConvBlock.forward docstring). This must
        # produce the same neighbourhood -- and hence the same output -- as
        # the online fallback, given the same (physical) coordinates.
        from kdmfpnet.kd_data import _compute_knn_indices

        block = EdgeConvBlock(in_channels=5, out_channels=5, hidden=8, k=4)
        block.eval()
        points = torch.randn(1, 15, 3)
        feat = torch.randn(1, 5, 15)
        knn_idx = torch.from_numpy(_compute_knn_indices(points[0].numpy(), k=4)).unsqueeze(0)

        with torch.no_grad():
            out_online = block(points, feat)
            out_precomputed = block(points, feat, knn_idx=knn_idx)

        self.assertTrue(torch.allclose(out_online, out_precomputed, atol=1e-5))

    def test_permuting_points_permutes_output_consistently(self) -> None:
        # EdgeConv should be a function of local geometry, not point order:
        # permuting the input points/features should permute the output the
        # same way (up to floating point tolerance), confirming the kNN
        # gather is wired to the correct point, not a stale/mismatched index.
        torch.manual_seed(0)
        block = EdgeConvBlock(in_channels=5, out_channels=5, hidden=8, k=4)
        block.eval()
        points = torch.randn(1, 15, 3)
        feat = torch.randn(1, 5, 15)
        perm = torch.randperm(15)

        with torch.no_grad():
            out = block(points, feat)
            out_permuted_input = block(points[:, perm, :], feat[:, :, perm])

        self.assertTrue(torch.allclose(out[:, :, perm], out_permuted_input, atol=1e-5))


class KDPressureModelTests(unittest.TestCase):
    def test_teacher_and_student_forward_shape(self) -> None:
        teacher = build_teacher_model()
        student = build_student_model()
        points = torch.randn(2, 128, 3)
        condition = torch.randn(2, 2)

        teacher_out = teacher(points, condition)
        student_out = student(points, condition)

        self.assertEqual(tuple(teacher_out.shape), (2, 128, 1))
        self.assertEqual(tuple(student_out.shape), (2, 128, 1))

    def test_distill_embedding_shape_is_shared(self) -> None:
        teacher = build_teacher_model()
        student = build_student_model()
        points = torch.randn(1, 64, 3)
        condition = torch.randn(1, 2)

        teacher_features = teacher.forward_features(points, condition)
        student_features = student.forward_features(points, condition)

        self.assertEqual(tuple(teacher_features["distill"].shape), (1, 256))
        self.assertEqual(tuple(student_features["distill"].shape), (1, 256))
        self.assertEqual(tuple(teacher_features["hard_geo"].shape), (1, 256))
        self.assertEqual(tuple(student_features["hard_geo"].shape), (1, 256))
        self.assertGreater(sum(p.numel() for p in teacher.parameters()), sum(p.numel() for p in student.parameters()))

    def test_forward_supports_padded_masked_batches(self) -> None:
        student = build_student_model()
        points = torch.randn(2, 8, 3)
        condition = torch.randn(2, 2)
        mask = torch.tensor(
            [
                [True, True, True, True, False, False, False, False],
                [True, True, True, True, True, True, False, False],
            ]
        )

        out = student(points, condition, mask=mask)

        self.assertEqual(tuple(out.shape), (2, 8, 1))


if __name__ == "__main__":
    unittest.main()
