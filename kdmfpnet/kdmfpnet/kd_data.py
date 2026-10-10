from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from .utils import Standardizer, stable_sample_seed

DEFAULT_TARGET_COLUMN = "Cp"
MAX_ABS_CONSERVATIVE_VALUE = 1.0e4
MAX_ABS_CP_VALUE = 20.0
# Remove isolated nonphysical Cp points while retaining the rest of each sample.
MAX_UNPHYSICAL_CP_FRACTION = 5.0e-4
# Point screening uses a Mach-dependent stagnation Cp ceiling with tolerance.
CP_STAGNATION_OVERSHOOT = 1.5
HIGH_CP_THERMODYNAMIC_THRESHOLD = 1.5
THERMODYNAMIC_OVERSHOOT = 1.2

# Solver status counters are interpreted with a tolerance for transient points.
MAX_NON_PHYSICAL_POINTS = 100
_NON_PHYSICAL_RE = re.compile(r"there are (\d+) non-physical points")
_NON_PHYSICAL_RECONSTRUCTED_RE = re.compile(
    r"(\d+) reconstructed states for upwinding are non-physical"
)
# MUSCL reconstruction is limiter-clipped, so a handful of non-physical
# reconstructed states is normal and never enters the solution vector. Allow
# more headroom here than for solution points.
MAX_NON_PHYSICAL_RECONSTRUCTED = 2000


def _non_physical_counts(text: str) -> tuple[int, int]:
    """Return (max solution points, max reconstructed states) reported by SU2.

    Returns (-1, -1) when the warning string is present but no count could be
    parsed, so callers can treat an unrecognised format as unsafe.
    """
    points = [int(m) for m in _NON_PHYSICAL_RE.findall(text)]
    recon = [int(m) for m in _NON_PHYSICAL_RECONSTRUCTED_RE.findall(text)]
    if "non-physical points" in text and not points:
        return -1, max(recon, default=0)
    return max(points, default=0), max(recon, default=0)


def _sample_indices(total: int, num_points: int, seed: int) -> np.ndarray:
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if total <= 0:
        raise ValueError("Point cloud is empty")
    rng = np.random.default_rng(seed)
    replace = total < num_points
    return rng.choice(total, size=num_points, replace=replace)


def _take_top_count(total: int, fraction: float, num_points: int) -> int:
    if fraction <= 0 or total <= 0:
        return 0
    return min(total, max(1, int(round(num_points * fraction))))


def _pressure_aware_sample_indices(
    points: np.ndarray,
    target: np.ndarray,
    num_points: int,
    seed: int,
    peak_fraction: float = 0.20,
    tail_fraction: float = 0.20,
    outer_fraction: float = 0.15,
) -> np.ndarray:
    total = int(points.shape[0])
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if total <= 0:
        raise ValueError("Point cloud is empty")

    rng = np.random.default_rng(seed)
    selected: list[int] = []

    def add(indices: np.ndarray, count: int) -> None:
        if count <= 0 or indices.size == 0:
            return
        chosen = indices[: min(count, indices.size)]
        selected.extend(int(i) for i in chosen)

    target_flat = np.asarray(target, dtype=np.float32).reshape(-1)
    peak_order = np.argsort(-np.abs(target_flat))
    add(peak_order, _take_top_count(total, peak_fraction, num_points))

    x = np.asarray(points[:, 0], dtype=np.float32)
    tail_order = np.argsort(x)
    add(tail_order, _take_top_count(total, tail_fraction, num_points))

    radius = np.linalg.norm(np.asarray(points[:, 1:3], dtype=np.float32), axis=1)
    outer_order = np.argsort(-radius)
    add(outer_order, _take_top_count(total, outer_fraction, num_points))

    unique_selected = list(dict.fromkeys(selected))
    if len(unique_selected) >= num_points:
        return np.asarray(unique_selected[:num_points], dtype=np.int64)

    selected_set = set(unique_selected)
    remaining = np.asarray([i for i in range(total) if i not in selected_set], dtype=np.int64)
    needed = num_points - len(unique_selected)
    if remaining.size > 0:
        replace = remaining.size < needed
        fill = rng.choice(remaining, size=needed, replace=replace)
    else:
        fill = rng.choice(np.arange(total, dtype=np.int64), size=needed, replace=True)
    return np.asarray(unique_selected + [int(i) for i in fill], dtype=np.int64)


def _sample_is_clean(sample_dir: Path) -> bool:
    status_path = sample_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if status.get("status") != "done":
        return False

    sample_id = sample_dir.name
    if not (
        (sample_dir / f"{sample_id}_history.csv").is_file()
        and (sample_dir / f"{sample_id}_surface_flow.csv").is_file()
        # parameters.json is required downstream for the Mach-focus sampling
        # weights and the Cp non-dimensionalisation; some G4 directories were
        # left with only a partial download and are missing it even though
        # status.json says "done" and the CSVs are present.
        and (sample_dir / "parameters.json").is_file()
    ):
        return False

    run_log_path = sample_dir / "run.log"
    if not run_log_path.is_file():
        return False
    text = run_log_path.read_text(encoding="utf-8", errors="ignore")
    if not (
        "All convergence criteria satisfied." in text
        and "Maximum number of iterations reached" not in text
        and "SU2 has diverged" not in text
        and "MPI_ABORT" not in text
        and "Exit Success (SU2_CFD)" in text
    ):
        return False

    non_physical_points, non_physical_recon = _non_physical_counts(text)
    if non_physical_points < 0 or non_physical_points > MAX_NON_PHYSICAL_POINTS:
        return False
    if non_physical_recon > MAX_NON_PHYSICAL_RECONSTRUCTED:
        return False

    # SU2 can emit an exit-success footer after a numerically unusable run.
    # Reject those pseudo-successes from the training pool.
    try:
        surface_path = sample_dir / f"{sample_id}_surface_flow.csv"
        values = pd.read_csv(
            surface_path,
            usecols=["Density", "Momentum_x", "Momentum_y", "Momentum_z", "Energy"],
        ).to_numpy(dtype=np.float64)
    except (OSError, ValueError, TypeError):
        return False
    if values.size == 0 or not np.isfinite(values).all():
        return False

    density = values[:, 0]
    momentum = values[:, 1:4]
    energy = values[:, 4]
    if np.any(density <= 0.0):
        return False
    if np.max(np.abs(values)) > MAX_ABS_CONSERVATIVE_VALUE:
        return False
    pressure = _pressure_from_conservative_vars(
        density=density,
        momentum_x=momentum[:, 0],
        momentum_y=momentum[:, 1],
        momentum_z=momentum[:, 2],
        energy=energy,
    )
    return bool(
        np.isfinite(pressure).all()
        and np.all(pressure > 0.0)
        and np.max(np.abs(pressure)) <= MAX_ABS_CONSERVATIVE_VALUE
    )


def list_fidelity_sample_ids(
    fidelity_root: str | Path,
    split: str | None = None,
    clean_only: bool = False,
) -> list[str]:
    fidelity_root = Path(fidelity_root)
    if not fidelity_root.is_dir():
        raise FileNotFoundError(fidelity_root)
    sample_ids = []
    for child in fidelity_root.iterdir():
        if not child.is_dir():
            continue
        parts = child.name.split("_")
        if split is not None and (len(parts) < 2 or parts[1] != split):
            continue
        surface_path = child / f"{child.name}_surface_flow.csv"
        if not surface_path.exists():
            continue
        if clean_only and not _sample_is_clean(child):
            continue
        sample_ids.append(child.name)
    return sorted(sample_ids)


def load_sample_id_manifest(manifest_path: str | Path) -> list[str]:
    """Load a fixed list of sample ids, one per line, ignoring blanks.

    Used for the G2 training set: `usable_g2_train_800.txt` is the output of
    check_validation.py's physics survey (convergence, orientation, CD
    magnitude, fin drag share) run over the combined original-800 + top-up-244
    submission, then trimmed to exactly 800 with a Mach-uniform selection. That
    is a stronger and more current filter than `_sample_is_clean` in this file,
    which only checks SU2's own exit status and conservative-variable sanity and
    knows nothing about orientation or drag magnitude. Reading the manifest
    means training uses exactly the samples that were vetted, rather than
    re-deriving a (weaker, possibly stale) clean set from the directory tree.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ids = [line.strip() for line in path.read_text(encoding="ascii").splitlines()]
    ids = [sid for sid in ids if sid]
    if not ids:
        raise ValueError(f"Manifest is empty: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Manifest contains duplicate ids: {path}")
    return sorted(ids)


def _parse_freestream_pressure(run_log_path: Path, fallback: float = 1.0) -> float:
    if not run_log_path.exists():
        return float(fallback)
    for line in run_log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Static Pressure" not in line:
            continue
        values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if values:
            # SU2's table contains dimensional pressure first and the
            # nondimensional pressure in the final numeric column.
            return float(values[-1])
    return float(fallback)


def _pressure_from_conservative_vars(
    density: np.ndarray,
    momentum_x: np.ndarray,
    momentum_y: np.ndarray,
    momentum_z: np.ndarray,
    energy: np.ndarray,
    gamma: float = 1.4,
) -> np.ndarray:
    rho = np.asarray(density, dtype=np.float64)
    mom_x = np.asarray(momentum_x, dtype=np.float64)
    mom_y = np.asarray(momentum_y, dtype=np.float64)
    mom_z = np.asarray(momentum_z, dtype=np.float64)
    total_energy = np.asarray(energy, dtype=np.float64)
    kinetic = 0.5 * (mom_x * mom_x + mom_y * mom_y + mom_z * mom_z) / np.clip(rho, 1e-12, None)
    pressure = (gamma - 1.0) * (total_energy - kinetic)
    return pressure.astype(np.float32)


def stagnation_cp(mach: float, gamma: float = 1.4) -> float:
    """Cp at a stagnation point behind a normal shock (Rayleigh pitot formula).

    This is the largest Cp inviscid flow can produce on the surface, and it
    depends on Mach: 1.56 at Ma 1.5, 1.77 at Ma 3, 1.81 at Ma 5. Below Ma 1 the
    shock term does not apply, so fall back to the isentropic expression.
    """
    mach = float(mach)
    if mach <= 0.0:
        raise ValueError("Mach must be positive")
    if mach < 1.0:
        ratio = (1.0 + 0.5 * (gamma - 1.0) * mach**2) ** (gamma / (gamma - 1.0))
    else:
        ratio = ((gamma + 1.0) ** 2 * mach**2 / (4.0 * gamma * mach**2 - 2.0 * (gamma - 1.0))) ** (
            gamma / (gamma - 1.0)
        ) * ((1.0 - gamma + 2.0 * gamma * mach**2) / (gamma + 1.0))
    return 2.0 / (gamma * mach**2) * (ratio - 1.0)


def _cp_from_pressure(pressure: np.ndarray, mach: float, p_inf: float, gamma: float = 1.4) -> np.ndarray:
    pressure = np.asarray(pressure, dtype=np.float32)
    q_inf = 0.5 * gamma * float(p_inf) * float(mach) ** 2
    if abs(q_inf) < 1e-12:
        raise ValueError("Freestream dynamic pressure is zero")
    return ((pressure - float(p_inf)) / q_inf).astype(np.float32)


def _high_cp_thermodynamic_outliers(
    density: np.ndarray,
    pressure: np.ndarray,
    cp: np.ndarray,
    mach: float,
    p_inf: float,
    gamma: float = 1.4,
) -> np.ndarray:
    """Flag severe high-Cp state violations for freestream rho=1, T=1 Euler runs."""
    rho = np.asarray(density, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)
    total_temperature = 1.0 + 0.5 * (gamma - 1.0) * float(mach) ** 2
    isentropic_stagnation_density = total_temperature ** (1.0 / (gamma - 1.0))
    high_cp = np.asarray(cp) > HIGH_CP_THERMODYNAMIC_THRESHOLD
    return high_cp & (
        (rho <= 0.0)
        | (p > THERMODYNAMIC_OVERSHOOT * total_temperature * float(p_inf) * rho)
        | (rho > THERMODYNAMIC_OVERSHOOT * isentropic_stagnation_density)
    )


def _resolve_sample_dir(fidelity_root: Path, sample_id: str) -> Path:
    candidate = fidelity_root / sample_id
    if candidate.is_dir():
        return candidate
    if (fidelity_root / f"{sample_id}_surface_flow.csv").exists():
        return fidelity_root
    raise FileNotFoundError(candidate)


# EdgeConvBlock's local-neighbourhood aggregation needs each point's k nearest
# neighbours. A given sample's point cloud (and hence its neighbour
# structure) is fixed across the whole run at num_points=None -- the
# production setting -- so paying an O(N^2) brute-force GPU search inside the
# model's forward pass on every single training step (measured: ~74 ms at
# N=20000, more than the rest of the forward+backward pass combined) is pure
# waste. Precomputing it once here with a CPU cKDTree (~50 ms one-time,
# amortised over hundreds of epochs via FidelityPressureDataset's existing
# per-sample record cache) and shipping the fixed indices alongside the
# points removes that cost from the training loop entirely.
DEFAULT_KNN_K = 16


def _compute_knn_indices(points: np.ndarray, k: int = DEFAULT_KNN_K) -> np.ndarray:
    n = points.shape[0]
    k_eff = min(k, max(n - 1, 1))
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k_eff + 1)
    idx = np.atleast_2d(idx)
    # tree.query's first column is each point itself (distance 0); drop it so
    # the returned indices are strictly the k_eff *other* nearest points, and
    # pad by repeating the last neighbour if k_eff < k (tiny point clouds).
    idx = idx[:, 1:]
    if idx.shape[1] < k:
        pad = np.repeat(idx[:, -1:], k - idx.shape[1], axis=1)
        idx = np.concatenate([idx, pad], axis=1)
    return idx.astype(np.int64)


def load_fidelity_pressure_record(
    sample_id: str,
    fidelity_root: str | Path,
    num_points: int | None,
    seed: int,
    target_column: str = DEFAULT_TARGET_COLUMN,
    sampling_mode: str = "random",
    peak_sample_fraction: float = 0.20,
    tail_sample_fraction: float = 0.20,
    outer_sample_fraction: float = 0.15,
    knn_k: int = DEFAULT_KNN_K,
) -> dict[str, object]:
    fidelity_root = Path(fidelity_root)
    sample_dir = _resolve_sample_dir(fidelity_root, sample_id)
    surface_path = sample_dir / f"{sample_id}_surface_flow.csv"
    params_path = sample_dir / "parameters.json"
    run_log_path = sample_dir / "run.log"
    if not surface_path.exists():
        raise FileNotFoundError(surface_path)
    if not params_path.exists():
        raise FileNotFoundError(params_path)

    df = pd.read_csv(surface_path, usecols=["PointID", "x", "y", "z", "Density", "Momentum_x", "Momentum_y", "Momentum_z", "Energy"])
    with params_path.open("r", encoding="utf-8") as f:
        params = json.load(f)
    mach = float(params["mach"])
    p_inf = _parse_freestream_pressure(run_log_path)

    local_pressure = _pressure_from_conservative_vars(
        density=df["Density"].to_numpy(dtype=np.float32),
        momentum_x=df["Momentum_x"].to_numpy(dtype=np.float32),
        momentum_y=df["Momentum_y"].to_numpy(dtype=np.float32),
        momentum_z=df["Momentum_z"].to_numpy(dtype=np.float32),
        energy=df["Energy"].to_numpy(dtype=np.float32),
    )
    cp = _cp_from_pressure(local_pressure, mach=mach, p_inf=p_inf)
    cp_ceiling = min(CP_STAGNATION_OVERSHOOT * stagnation_cp(mach), MAX_ABS_CP_VALUE)
    bad = (
        ~np.isfinite(cp)
        | (np.abs(cp) > cp_ceiling)
        | _high_cp_thermodynamic_outliers(df["Density"].to_numpy(), local_pressure, cp, mach, p_inf)
    )
    bad_count = int(bad.sum())
    if bad_count:
        bad_fraction = bad_count / bad.size
        if bad_fraction > MAX_UNPHYSICAL_CP_FRACTION:
            raise ValueError(
                f"Unphysical Cp in {sample_id}: {bad_count}/{bad.size} nodes "
                f"({bad_fraction:.3g}) exceed |Cp|={cp_ceiling:.4g} "
                f"({CP_STAGNATION_OVERSHOOT:g}x stagnation Cp at Ma {mach:.3f}, "
                f"or high-Cp thermodynamic limit), "
                f"limit={MAX_UNPHYSICAL_CP_FRACTION:.3g}; "
                f"max_abs={float(np.nanmax(np.abs(cp))):.6g}"
            )
        # Below the fraction limit these are isolated nonphysical nodes
        # (see MAX_UNPHYSICAL_CP_FRACTION). Drop them from the cloud before
        # sampling so the row indices stay aligned with df.
        keep = ~bad
        df = df.loc[keep].reset_index(drop=True)
        local_pressure = local_pressure[keep]
        cp = cp[keep]
    if target_column.lower() in {"cp", "cp_coeff", "pressure_coefficient"}:
        target_values = cp.reshape(-1, 1)
    elif target_column.lower() in {"pressure", "p"}:
        target_values = local_pressure.reshape(-1, 1)
    else:
        raise ValueError(f"Unknown target_column: {target_column!r}")

    all_points = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    all_target = target_values
    if num_points is None:
        indices = np.arange(len(df), dtype=np.int64)
    elif sampling_mode == "pressure_aware":
        indices = _pressure_aware_sample_indices(
            points=all_points,
            target=all_target,
            num_points=num_points,
            seed=seed,
            peak_fraction=peak_sample_fraction,
            tail_fraction=tail_sample_fraction,
            outer_fraction=outer_sample_fraction,
        )
    elif sampling_mode == "random":
        indices = _sample_indices(len(df), num_points, seed)
    else:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode!r}")
    sampled = df.iloc[indices].reset_index(drop=True)

    points = sampled[["x", "y", "z"]].to_numpy(dtype=np.float32)
    point_ids = sampled["PointID"].to_numpy(dtype=np.int64)
    pressure = target_values[indices].astype(np.float32)
    condition = np.array([mach, float(params["aoa_deg"])], dtype=np.float32)
    knn_idx = _compute_knn_indices(points, k=knn_k)
    return {
        "sample_id": sample_id,
        "point_ids": point_ids,
        "points": points,
        "condition": condition,
        "cp": pressure,
        "knn_idx": knn_idx,
    }


class FidelityPressureDataset(Dataset):
    def __init__(
        self,
        fidelity_root: str | Path,
        sample_ids: Sequence[str],
        num_points: int | None,
        seed: int = 42,
        target_column: str = DEFAULT_TARGET_COLUMN,
        sampling_mode: str = "random",
        peak_sample_fraction: float = 0.20,
        tail_sample_fraction: float = 0.20,
        outer_sample_fraction: float = 0.15,
        cache_records: bool = True,
        knn_k: int = DEFAULT_KNN_K,
    ) -> None:
        self.fidelity_root = Path(fidelity_root)
        self.sample_ids = list(sample_ids)
        self.num_points = None if num_points is None else int(num_points)
        self.seed = int(seed)
        self.target_column = target_column
        self.sampling_mode = sampling_mode
        self.peak_sample_fraction = float(peak_sample_fraction)
        self.tail_sample_fraction = float(tail_sample_fraction)
        self.outer_sample_fraction = float(outer_sample_fraction)
        self.cache_records = bool(cache_records)
        self.knn_k = int(knn_k)
        self._record_cache: dict[str, dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.sample_ids[index]
        if self.cache_records:
            cached = self._record_cache.get(sample_id)
            if cached is not None:
                return cached
        record = load_fidelity_pressure_record(
            sample_id=sample_id,
            fidelity_root=self.fidelity_root,
            num_points=self.num_points,
            seed=stable_sample_seed(sample_id, self.seed),
            target_column=self.target_column,
            sampling_mode=self.sampling_mode,
            peak_sample_fraction=self.peak_sample_fraction,
            tail_sample_fraction=self.tail_sample_fraction,
            outer_sample_fraction=self.outer_sample_fraction,
            knn_k=self.knn_k,
        )
        if self.cache_records:
            self._record_cache[sample_id] = record
        return record


def collate_fidelity_pressure_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    lengths = np.asarray([np.asarray(item["points"]).shape[0] for item in batch], dtype=np.int64)
    max_len = int(lengths.max()) if lengths.size else 0
    batch_size = len(batch)

    points = np.zeros((batch_size, max_len, 3), dtype=np.float32)
    point_ids = np.full((batch_size, max_len), -1, dtype=np.int64)
    cp = np.zeros((batch_size, max_len, 1), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=bool)
    condition = np.stack([np.asarray(item["condition"], dtype=np.float32) for item in batch], axis=0)

    knn_k = np.asarray(batch[0]["knn_idx"]).shape[1] if batch and "knn_idx" in batch[0] else 0
    # Padding rows keep knn_idx=0 (a harmless self-loop into slot 0, itself
    # zero-padded) rather than -1 -- these rows are masked out by `mask`
    # before EdgeConvBlock ever sees them, but an out-of-range index would
    # still crash torch.gather if something reads it before masking.
    knn_idx = np.zeros((batch_size, max_len, knn_k), dtype=np.int64) if knn_k else None

    for batch_index, item in enumerate(batch):
        item_points = np.asarray(item["points"], dtype=np.float32)
        item_point_ids = np.asarray(item["point_ids"], dtype=np.int64)
        item_cp = np.asarray(item["cp"], dtype=np.float32)
        count = item_points.shape[0]
        points[batch_index, :count] = item_points
        point_ids[batch_index, :count] = item_point_ids
        cp[batch_index, :count] = item_cp
        mask[batch_index, :count] = True
        if knn_idx is not None:
            knn_idx[batch_index, :count] = np.asarray(item["knn_idx"], dtype=np.int64)

    result = {
        "sample_id": [item["sample_id"] for item in batch],
        "point_ids": torch.from_numpy(point_ids),
        "points": torch.from_numpy(points),
        "condition": torch.from_numpy(condition),
        "cp": torch.from_numpy(cp),
        "mask": torch.from_numpy(mask),
        "lengths": torch.from_numpy(lengths),
    }
    if knn_idx is not None:
        result["knn_idx"] = torch.from_numpy(knn_idx)
    return result


def build_fidelity_pressure_loader(
    fidelity_root: str | Path,
    sample_ids: Sequence[str],
    num_points: int | None,
    batch_size: int,
    seed: int = 42,
    shuffle: bool = False,
    drop_last: bool = False,
    sampler=None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    sampling_mode: str = "random",
    peak_sample_fraction: float = 0.20,
    tail_sample_fraction: float = 0.20,
    outer_sample_fraction: float = 0.15,
    knn_k: int = DEFAULT_KNN_K,
) -> DataLoader:
    dataset = FidelityPressureDataset(
        fidelity_root=fidelity_root,
        sample_ids=sample_ids,
        num_points=num_points,
        seed=seed,
        target_column=target_column,
        sampling_mode=sampling_mode,
        peak_sample_fraction=peak_sample_fraction,
        tail_sample_fraction=tail_sample_fraction,
        outer_sample_fraction=outer_sample_fraction,
        knn_k=knn_k,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fidelity_pressure_batch,
    )


def fit_fidelity_standardizers(
    fidelity_root: str | Path,
    sample_ids: Sequence[str],
    num_points: int | None,
    seed: int,
    target_column: str = DEFAULT_TARGET_COLUMN,
    sampling_mode: str = "random",
    peak_sample_fraction: float = 0.20,
    tail_sample_fraction: float = 0.20,
    outer_sample_fraction: float = 0.15,
) -> dict[str, Standardizer]:
    point_blocks = []
    condition_blocks = []
    pressure_blocks = []
    for sample_id in sample_ids:
        record = load_fidelity_pressure_record(
            sample_id=sample_id,
            fidelity_root=fidelity_root,
            num_points=num_points,
            seed=stable_sample_seed(sample_id, seed),
            target_column=target_column,
            sampling_mode=sampling_mode,
            peak_sample_fraction=peak_sample_fraction,
            tail_sample_fraction=tail_sample_fraction,
            outer_sample_fraction=outer_sample_fraction,
        )
        point_blocks.append(np.asarray(record["points"], dtype=np.float32))
        condition_blocks.append(np.asarray(record["condition"], dtype=np.float32))
        pressure_blocks.append(np.asarray(record["cp"], dtype=np.float32))
    points = np.concatenate(point_blocks, axis=0)
    conditions = np.stack(condition_blocks, axis=0)
    pressures = np.concatenate(pressure_blocks, axis=0)
    return {
        "points": Standardizer.fit(points),
        "condition": Standardizer.fit(conditions),
        "cp": Standardizer.fit(pressures),
    }
