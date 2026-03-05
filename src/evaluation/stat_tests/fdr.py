# -*- coding: utf-8 -*-
"""
Benjamini–Hochberg FDR correction utility.

Takes a sequence of raw p-values and returns adjusted q-values that
control the False Discovery Rate.

Public API
----------
benjamini_hochberg(pvalues) → np.ndarray of q-values (same order)

Commit: refactor(etapaA1).
"""
from __future__ import annotations

import numpy as np


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini–Hochberg FDR correction.

    Parameters
    ----------
    pvalues : array-like of float
        Raw p-values (1-D).

    Returns
    -------
    np.ndarray
        Adjusted q-values in the *same order* as the input.
        q[i] = min_{k≥i}(p_{(k)} · m / k)  (monotonised).
    """
    pv = np.asarray(pvalues, dtype=np.float64).ravel()
    m = len(pv)
    if m == 0:
        return np.array([], dtype=np.float64)

    # Sort p-values ascending
    order = np.argsort(pv)
    sorted_p = pv[order]

    # BH adjusted: q_{(i)} = p_{(i)} * m / i   (1-indexed)
    ranks = np.arange(1, m + 1, dtype=np.float64)
    adjusted = sorted_p * m / ranks

    # Enforce monotonicity (cumulative min from right)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Restore original order
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result
