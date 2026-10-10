"""Reusable single-model local calibration interface.

This module exposes the same validation-calibrated local bias compensation as
``kdmfpnet.physics_gate`` under the historical ``nose`` naming. It depends only
on the baseline prediction, the geometry, and the flight condition.
"""
from __future__ import annotations

import numpy as np

from .physics_gate import case_factor, local_bias_compensation_cp

__all__ = ["nose_case_factor", "calibrated_nose_cp", "calibrated_cp"]


def nose_case_factor(mach: float, nose_tip_radius_m: float) -> float:
    """Smoothly activate the correction for fast, blunt-nose conditions."""
    return case_factor(mach, nose_tip_radius_m)


def calibrated_nose_cp(
    points_m: np.ndarray,
    baseline_cp: np.ndarray,
    mach: float,
    nose_tip_radius_m: float,
    delta: float = 0.2,
) -> np.ndarray:
    """Return Cp with the frozen spatial and condition-gated local correction."""
    return local_bias_compensation_cp(points_m, baseline_cp, mach, nose_tip_radius_m, delta=delta)


calibrated_cp = calibrated_nose_cp
