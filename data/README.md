# External data contract

This public package contains no training samples. Keep the data root outside the repository and pass it to the training and evaluation scripts.

## Sample directory

Each sample directory must provide:

- `parameters.json`: at least `mach` and `aoa_deg`, plus any geometry metadata needed by the loader.
- `status.json`: a completed/usable status record.
- `run.log`: the computation history used to validate the record.
- `<sample_id>_history.csv`: convergence history.
- `<sample_id>_surface_flow.csv`: point coordinates and conservative variables from which `Cp` is reconstructed.

The loader checks finite values, positive density/pressure, and the configured physical `Cp` range before accepting a sample.

## Fidelity layout

```text
<data-root>/
  G2/<sample_id>/...
  G4/<sample_id>/...
  G2/train_manifest.txt
  G4/train_manifest.txt
  G4/test_manifest.txt
```

Manifest files contain one sample ID per line. They are metadata only and are not included in this repository.
