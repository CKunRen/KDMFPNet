"""Inference-only, validation-calibrated local bias compensation for surface Cp.

The final method is a single point-cloud predictor plus a physics-guided local
bias compensation:

    C_hat_i = C_{p,i} + g_{p,i} * g_{c} * delta

    g_{p,i} = clip(1 - s_i / ell_g, 0, 1) * clip((C_{p,i} - C0) / dC, 0, 1)
    g_{c}   = clip((Ma - M_g) / dM, 0, 1) * clip((r_n - r0) / dr, 0, 1)

where ``s_i`` is the normalized axial position (nose at ``s = 0``), ``C_{p,i}``
is the predicted pressure coefficient, ``r_n`` is the nose-tip radius, and
``delta`` is a bounded bias compensation. It changes only the very front of the
body under activated flight and geometry conditions. The function reads only
the baseline prediction, the geometry, and the flight condition, so it can be
applied after running a single predictor. All constants are selected on
validation data and frozen before testing.
"""
from __future__ import annotations

import numpy as np


def _validate_inputs(points_m: np.ndarray, baseline_cp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_m, dtype=np.float64)
    baseline = np.asarray(baseline_cp, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points_m must be a nonempty [N, 3] array")
    if len(baseline) != len(points):
        raise ValueError("baseline_cp must align one-to-one with points_m")
    return points, baseline


def _axial_position(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    return np.clip((x - x.min()) / max(float(x.max() - x.min()), 1e-12), 0.0, 1.0)


def case_factor(
    mach: float,
    nose_tip_radius_m: float,
    mach_start: float = 3.0,
    mach_width: float = 1.2,
    radius_start: float = 0.0035,
    radius_width: float = 0.0025,
) -> float:
    """Smoothly activate the correction for fast, blunt-nose conditions."""
    mach_score = np.clip((float(mach) - mach_start) / mach_width, 0.0, 1.0)
    radius_score = np.clip((float(nose_tip_radius_m) - radius_start) / radius_width, 0.0, 1.0)
    return float(mach_score * radius_score)


def local_bias_compensation_cp(
    points_m: np.ndarray,
    baseline_cp: np.ndarray,
    mach: float,
    nose_tip_radius_m: float,
    delta: float = 0.2,
    ell_g: float = 0.005,
    c0: float = 0.8,
    d_c: float = 0.4,
) -> np.ndarray:
    """Apply the frozen single-model local bias compensation.

    ``points_m`` has shape ``[N, 3]`` and includes the full cloud so that
    minimum and maximum ``x`` locate the nose and aft end. Returns the
    compensated ``Cp`` with the same shape as ``baseline_cp``.
    """
    points, baseline = _validate_inputs(points_m, baseline_cp)
    axial = _axial_position(points)
    point_gate = np.clip(1.0 - axial / ell_g, 0.0, 1.0) * np.clip((baseline - c0) / d_c, 0.0, 1.0)
    gate = point_gate * case_factor(mach, nose_tip_radius_m)
    return baseline + gate * float(delta)


# Backward-compatible alias kept for callers that used the previous name.
calibrated_cp = local_bias_compensation_cp
