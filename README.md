# KDMFPNet

Knowledge-Distilled Multi-Fidelity Point-Cloud Network for point-wise surface pressure-coefficient (`Cp`) prediction.

This repository contains only reusable neural-network, data-interface, training, evaluation, inference-gate, and test code. Training data, generated simulation cases, solver files, checkpoints, logs, and experiment outputs are external and are not included.

The release covers the PointNet/EdgeConv pressure model, G2-to-G4 knowledge distillation, point-level data interfaces, and the inference-only, validation-calibrated local bias compensation used for nose high-`Cp` correction. Training samples and generated simulation artifacts are not included.

## Layout

```text
kdmfpnet/                  importable Python package
  kdmfpnet/                models, loaders, losses, KD trainer, evaluator
    physics_gate.py        frozen single-model local bias-compensation interface
    nose_calibration.py    reusable local calibration interface
  scripts/                 portable train/evaluate entry points
  tests/                   source-only unit tests
docs/model_design.md       model and data-interface notes
data/README.md             external-data contract
pyproject.toml            package metadata and dependencies
requirements.txt           optional dependency list
```

## Install

Python 3.10 or newer is recommended:

```bash
python -m pip install -e .
```

Install the PyTorch build appropriate for the target CPU or GPU separately when needed.

## External data contract

No samples are shipped. Keep data, checkpoints, and generated outputs outside this source tree. The training interface expects an external data root with `G2/` and `G4/` fidelity directories. Each sample directory supplies point coordinates, flight-condition metadata, and the target field through the files documented in [`data/README.md`](data/README.md). Manifest files are plain text lists of sample IDs.

## Train and evaluate

```bash
python kdmfpnet/scripts/train_kd.py \
  --data-root /path/to/data-root \
  --g2-manifest /path/to/G2/train_manifest.txt \
  --g4-train-manifest /path/to/G4/train_manifest.txt \
  --g4-test-manifest /path/to/G4/test_manifest.txt \
  --architecture-variant baseline \
  --output-dir /path/to/outputs/kd_run

python kdmfpnet/scripts/evaluate_kd.py \
  --data-root /path/to/data-root \
  --checkpoint /path/to/outputs/kd_run/student_best.pt \
  --output-dir /path/to/outputs/evaluation
```

Use `--architecture-variant baseline` for the released method. The other variants are optional ablations and are not required for the released single-model method. Train the baseline in one output directory with the external manifests and seed; no pretrained weights are included in this repository.

## Inference calibration

The released method applies a frozen local bias compensation to the baseline prediction:

```python
from kdmfpnet.physics_gate import local_bias_compensation_cp

cp = local_bias_compensation_cp(
    points_m=points_xyz,             # [N, 3], physical coordinates
    baseline_cp=baseline_cp,         # [N]
    mach=mach,
    nose_tip_radius_m=nose_tip_radius_m,
)
```

The compensation localizes a small positive pressure bias to the very front of the body, smoothly activated by Mach number and nose-tip radius. It reads only the baseline prediction, the geometry, and the flight condition, so a single trained predictor is sufficient at inference. The constants are selected on validation data and frozen before testing.

The scripts do not generate data. They consume the external point-cloud records and write metrics, predictions, and checkpoints to the requested output directory.

## Tests

From the repository root:

```bash
python -m unittest discover -s kdmfpnet/tests -p "test_*.py"
```

All included tests use synthetic tensors or temporary metadata and do not require released samples.

## Reproducibility

- G2 and G4 use separate standardizers and manifests.
- The KD path trains a teacher, warms the student on the low-fidelity split, then performs high-fidelity distillation.
- The model accepts padded batches through a point mask.
- Inference uses one trained predictor plus the frozen local bias compensation; integrated aerodynamic quantities can be computed by a downstream application from the predicted field.
