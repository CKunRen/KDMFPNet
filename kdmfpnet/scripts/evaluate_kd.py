from __future__ import annotations

import argparse
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from kdmfpnet.kd_train import evaluate_kd_pressure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved KD student checkpoint.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kd_eval"))
    parser.add_argument("--device", default="")
    parser.add_argument("--num-points", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-column", default="Cp")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_kd_pressure(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_points=args.num_points or None,
        seed=args.seed,
        device=args.device,
        target_column=args.target_column,
    )
    metrics = result["metrics"]
    print(
        "Evaluation complete | "
        f"mae={metrics['mae']:.6f} | "
        f"rmse={metrics['rmse']:.6f} | "
        f"nmae={metrics['nmae']:.6f} | "
        f"nrmse={metrics['nrmse']:.6f}"
    )
    print(f"Metrics: {result['metrics_path']}")
    print(f"Predictions: {result['predictions_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
