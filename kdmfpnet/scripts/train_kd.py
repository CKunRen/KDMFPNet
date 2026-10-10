from __future__ import annotations

import argparse
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import torch

from kdmfpnet.kd_train import train_kd_pressure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-fidelity point-cloud Cp model.")
    parser.add_argument("--data-root", type=Path, required=True, help="External root containing G2/ and G4/.")
    parser.add_argument("--g2-manifest", type=Path, help="Optional physics-vetted G2 train manifest.")
    parser.add_argument("--g4-train-manifest", type=Path, help="Optional physics-vetted G4 train manifest.")
    parser.add_argument("--g4-test-manifest", type=Path, help="Optional physics-vetted G4 test manifest.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kd_g4_cp"))
    parser.add_argument("--device", default="", help="Torch device, e.g. cuda or cpu. Empty selects automatically.")
    parser.add_argument("--num-points", type=int, default=0, help="Point cap; 0 keeps native resolution.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-epochs", type=int, default=120)
    parser.add_argument("--student-pretrain-epochs", type=int, default=60)
    parser.add_argument("--student-epochs", type=int, default=100)
    parser.add_argument(
        "--architecture-variant",
        choices=("baseline", "multiscale_edge", "highfreq_coords"),
        default="baseline",
        help="Network variant. 'baseline' is the released method; the others are optional ablations.",
    )
    parser.add_argument("--knn-k", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    result = train_kd_pressure(
        data_root=args.data_root,
        output_dir=args.output_dir,
        g2_manifest=args.g2_manifest,
        g4_train_manifest=args.g4_train_manifest,
        g4_test_manifest=args.g4_test_manifest,
        num_points=args.num_points or None,
        batch_size=args.batch_size,
        teacher_epochs=args.teacher_epochs,
        student_pretrain_epochs=args.student_pretrain_epochs,
        student_pretrain_fidelity="g2",
        student_epochs=args.student_epochs,
        architecture_variant=args.architecture_variant,
        teacher_lr=8e-4,
        student_lr=4e-4,
        weight_decay=5e-4,
        seed=args.seed,
        val_fraction=0.1,
        student_val_fraction=0.18,
        device=device,
        grad_clip_norm=1.0,
        target_column="Cp",
        dropout=0.14,
        student_dropout=0.14,
        student_weight_decay=5e-4,
        fourier_bands=4,
        use_residual_decoder=True,
        knn_k=args.knn_k,
        response_alpha=0.35,
        feature_alpha=0.15,
        hard_point_alpha=1.5,
        hard_point_fraction=0.1,
        target_peak_alpha=0.35,
        target_peak_fraction=0.08,
        mach_focus_alpha=0.60,
        mach_focus_start=3.2,
        mach_focus_end=4.5,
        low_signal_alpha=0.35,
        low_signal_cap=1.75,
        kd_warmup_epochs=10,
        kd_ramp_epochs=60,
        point_alpha=0.3,
        point_feat_distill_channels=384,
        early_stop_patience=10**9,
        early_stop_min_epochs=60,
    )
    print(f"Teacher checkpoint: {result.teacher_best_checkpoint}")
    print(f"Student checkpoint: {result.student_best_checkpoint}")
    print(f"Metrics: {result.student_metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
