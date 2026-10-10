# KDMFPNet model and data interface

## Prediction task

The model predicts point-wise surface pressure coefficient `Cp` from one surface point cloud and its flight condition:

- `points_m`: `[N, 3]` physical `x, y, z` coordinates.
- `condition`: `[mach, aoa_deg]`.
- `Cp`: `[N]`, aligned by `PointID`.

G2 and G4 are separate fidelity levels. The teacher is trained on G2; the student is initialized with G2 supervision and then trained on G4 with response and feature distillation. Each fidelity uses its own standardizers and manifests.

## Network

The reusable core lives in `kdmfpnet/kd_model.py`:

1. Fourier features encode coordinates.
2. Shared point MLPs and an early EdgeConv block encode local geometry.
3. Global max, mean, and hard-point pooling summarize each cloud.
4. Mach/AoA condition features modulate point features and enter the decoder.
5. A point-wise decoder predicts `Cp`; an optional residual decoder adds a second prediction path.
6. Optional architecture variants (for ablations only) add a learned gated local residual branch.

`kdmfpnet/kd_train.py` handles teacher training, student pretraining, G4 distillation, checkpoint selection, and evaluation. `kdmfpnet/kd_data.py` loads external records, reconstructs `Cp`, applies point-level physical screening, and precomputes physical-coordinate k-nearest neighbors. No sample or trained weight is included here.

## Single-model local bias compensation

The released inference applies a frozen, validation-calibrated local bias compensation to the baseline prediction through `local_bias_compensation_cp` in `kdmfpnet/physics_gate.py`:

```
C_hat_i = C_{p,i} + g_{p,i} * g_c * delta
```

The point factor `g_{p,i}` uses normalized axial position from minimum `x` and the predicted `Cp`; the case factor `g_c` uses Mach number and nose radius. The compensation changes only the very front of the body under activated conditions. Only the baseline prediction, the geometry, and the flight condition are required, so a single trained predictor is sufficient. All constants are selected on validation data and frozen before testing. `kdmfpnet/nose_calibration.py` exposes the same interface under the historical nose naming.

## External data contract

See `data/README.md`. The package accepts paths to external G2/G4 directories and ID manifests. It does not create or package simulation cases, grids, solver runs, or training samples.
