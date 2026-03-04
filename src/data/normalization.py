# -*- coding: utf-8 -*-
"""
src.data.normalization — Condition & signal normalization utilities.

This module centralises every normalization operation used by the cVAE
training and evaluation pipelines.  All functions are **pure** (no
side-effects) and operate on NumPy arrays.

Two families of normalization live here:

1. **Condition normalization** (``normalize_conditions`` / ``apply_condition_norm``):
   Min-max scaling of distance *D* and current *C* arrays to [0, 1].
   Parameters are computed on the **training** split to avoid leakage.

2. **Signal normalization helpers** (``normalize_peak``, ``normalize_power``):
   Optional IQ signal scaling (peak or power).  The current pipeline
   does **not** apply these in-code (they happen upstream in GNU Radio),
   but the functions are provided for completeness and future use.

Metadata returned by ``compute_condition_norm_params`` is JSON-
serialisable and stored in ``state_run.json["normalization"]``.

Commit: refactor(step2).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


# =====================================================================
# 1.  Condition normalization  (D, C → [0, 1])
# =====================================================================

def compute_condition_norm_params(
    D_train: np.ndarray,
    C_train: np.ndarray,
) -> Dict[str, float]:
    """Compute min-max parameters from the **training** split.

    Parameters
    ----------
    D_train : ndarray, shape (N,)
        Raw distance values (metres).
    C_train : ndarray, shape (N,)
        Raw current values (mA).

    Returns
    -------
    dict
        ``{"D_min", "D_max", "C_min", "C_max"}`` — all Python floats,
        ready for JSON serialisation.
    """
    return {
        "D_min": float(D_train.min()),
        "D_max": float(D_train.max()),
        "C_min": float(C_train.min()),
        "C_max": float(C_train.max()),
    }


def apply_condition_norm(
    D: np.ndarray,
    C: np.ndarray,
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Min-max scale *D* and *C* using pre-computed parameters.

    If ``D_max == D_min`` the result is all-zeros.
    If ``C_max == C_min`` the result is all-0.5.

    Parameters
    ----------
    D, C : ndarray, shape (N,)
    params : dict with keys ``D_min, D_max, C_min, C_max``

    Returns
    -------
    (Dn, Cn) : tuple of ndarray, shape (N,)
    """
    D_min, D_max = float(params["D_min"]), float(params["D_max"])
    C_min, C_max = float(params["C_min"]), float(params["C_max"])

    if D_max > D_min:
        Dn = (D.astype(np.float64) - D_min) / (D_max - D_min)
    else:
        Dn = np.zeros(D.shape, dtype=np.float64)

    if C_max > C_min:
        Cn = (C.astype(np.float64) - C_min) / (C_max - C_min)
    else:
        Cn = np.full(C.shape, 0.5, dtype=np.float64)

    return Dn, Cn


def normalize_conditions(
    D_train: np.ndarray,
    C_train: np.ndarray,
    D_val: np.ndarray,
    C_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """One-shot: compute params on train, apply to both splits.

    Returns
    -------
    (Dn_train, Cn_train, Dn_val, Cn_val, norm_params)
    """
    params = compute_condition_norm_params(D_train, C_train)
    Dn_train, Cn_train = apply_condition_norm(D_train, C_train, params)
    Dn_val, Cn_val = apply_condition_norm(D_val, C_val, params)
    return Dn_train, Cn_train, Dn_val, Cn_val, params


# =====================================================================
# 2.  Signal normalization helpers  (IQ arrays)
# =====================================================================

def compute_signal_power(x: np.ndarray) -> float:
    """Mean power of an IQ signal array, shape (N, 2).

    .. math::
        P = \\frac{1}{N} \\sum_n (I_n^2 + Q_n^2)
    """
    return float(np.mean(np.sum(x ** 2, axis=-1)))


def normalize_peak(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Scale *x* so that ``max(|x|) == 1``.

    Returns ``(x_normed, peak_factor)`` where
    ``x_normed = x / peak_factor``.
    """
    peak = float(np.max(np.abs(x)))
    if peak == 0.0:
        return x.copy(), 1.0
    return x / peak, peak


def normalize_power(
    x: np.ndarray,
    target_power: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Scale *x* so that its mean power equals *target_power*.

    Returns ``(x_normed, scale_factor)`` where
    ``x_normed = x * scale_factor``.
    """
    p = compute_signal_power(x)
    if p == 0.0:
        return x.copy(), 1.0
    scale = float(np.sqrt(target_power / p))
    return x * scale, scale


# =====================================================================
# 3.  state_run.json I/O helper
# =====================================================================

def load_normalization_from_state(
    state_run: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """Extract condition-normalization params from a ``state_run`` dict.

    Returns ``None`` if the required keys are missing (old-format run).
    """
    norm = state_run.get("normalization")
    if not isinstance(norm, dict):
        return None
    required = {"D_min", "D_max", "C_min", "C_max"}
    if not required.issubset(norm.keys()):
        return None
    return {k: float(norm[k]) for k in required}
