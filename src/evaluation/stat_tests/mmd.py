# -*- coding: utf-8 -*-
"""
Two-sample Maximum Mean Discrepancy (MMD) test with RBF kernel.

Implements the unbiased estimator MMD²_u (Gretton et al., 2012) with a
permutation-based p-value.  Bandwidth is chosen via the *median heuristic*
with a robust fallback.

Public API
----------
mmd_rbf(Y_real, Y_pred, *, n_perm=200, seed=42)
    → dict  {"mmd2": float, "pval": float, "bandwidth": float,
             "n_perm": int, "n_real": int, "n_pred": int}

Commit: refactor(etapaA1).
"""
from __future__ import annotations

from typing import Dict

import numpy as np


# ------------------------------------------------------------------
# Kernel helpers
# ------------------------------------------------------------------

def _median_bandwidth(X: np.ndarray, Y: np.ndarray,
                      subsample: int = 5000) -> float:
    """Median heuristic for RBF bandwidth σ².

    Returns σ² = median(||x_i − y_j||²) over a random sub-sample.
    Falls back to 1.0 if the median is degenerate (0 or non-finite).
    """
    rng = np.random.default_rng(0)
    nx = min(subsample, len(X))
    ny = min(subsample, len(Y))
    Xs = X[rng.choice(len(X), nx, replace=False)]
    Ys = Y[rng.choice(len(Y), ny, replace=False)]

    # Pair-wise squared distances (cross-set only → cheaper & unbiased)
    # ||x-y||² = ||x||² + ||y||² − 2 x·y
    xx = np.sum(Xs ** 2, axis=1, keepdims=True)       # (nx,1)
    yy = np.sum(Ys ** 2, axis=1, keepdims=True).T     # (1,ny)
    dists2 = np.clip(xx + yy - 2.0 * Xs @ Ys.T, 0.0, None)

    med = float(np.median(dists2))
    if med <= 0.0 or not np.isfinite(med):
        # Robust fallback: mean of squared norms
        med = float(np.mean(np.sum((Xs - Ys[:nx]) ** 2, axis=1)))
    if med <= 0.0 or not np.isfinite(med):
        med = 1.0
    return med


def _gram_rbf(X: np.ndarray, Y: np.ndarray, bw: float) -> np.ndarray:
    """Compute RBF Gram matrix  K(X, Y) = exp(−||x−y||² / (2·bw))."""
    xx = np.sum(X ** 2, axis=1, keepdims=True)
    yy = np.sum(Y ** 2, axis=1, keepdims=True).T
    D2 = np.clip(xx + yy - 2.0 * X @ Y.T, 0.0, None)
    return np.exp(-D2 / (2.0 * bw))


# ------------------------------------------------------------------
# Unbiased MMD² estimator
# ------------------------------------------------------------------

def _mmd2_unbiased(Kxx: np.ndarray, Kyy: np.ndarray,
                   Kxy: np.ndarray) -> float:
    """Unbiased estimator of MMD² given pre-computed Gram blocks."""
    m = Kxx.shape[0]
    n = Kyy.shape[0]
    # Zero out diagonals (unbiased)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_xx = Kxx.sum() / max(1, m * (m - 1))
    term_yy = Kyy.sum() / max(1, n * (n - 1))
    term_xy = Kxy.sum() / max(1, m * n)
    return float(term_xx + term_yy - 2.0 * term_xy)


# ------------------------------------------------------------------
# Public entry-point
# ------------------------------------------------------------------

def mmd_rbf(
    Y_real: np.ndarray,
    Y_pred: np.ndarray,
    *,
    n_perm: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """Two-sample MMD test with RBF kernel and permutation p-value.

    Parameters
    ----------
    Y_real, Y_pred : ndarray (n, 2)
        IQ samples from the real channel and the twin, respectively.
    n_perm : int
        Number of permutations (200 = quick, 2000 = full).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys ``mmd2``, ``pval``, ``bandwidth``, ``n_perm``,
    ``n_real``, ``n_pred``.
    """
    X = np.asarray(Y_real, dtype=np.float64)
    Y = np.asarray(Y_pred, dtype=np.float64)
    assert X.ndim == 2 and Y.ndim == 2 and X.shape[1] == Y.shape[1]

    bw = _median_bandwidth(X, Y)

    m, n = len(X), len(Y)
    Z = np.concatenate([X, Y], axis=0)  # pooled

    # Pre-compute full Gram on pooled data (symmetric)
    K = _gram_rbf(Z, Z, bw)

    # Observed statistic
    Kxx = K[:m, :m].copy()
    Kyy = K[m:, m:].copy()
    Kxy = K[:m, m:]
    mmd2_obs = _mmd2_unbiased(Kxx, Kyy, Kxy)

    # Permutation null distribution
    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(m + n)
        Ki_xx = K[np.ix_(idx[:m], idx[:m])].copy()
        Ki_yy = K[np.ix_(idx[m:], idx[m:])].copy()
        Ki_xy = K[np.ix_(idx[:m], idx[m:])]
        mmd2_perm = _mmd2_unbiased(Ki_xx, Ki_yy, Ki_xy)
        if mmd2_perm >= mmd2_obs:
            count_ge += 1

    pval = (count_ge + 1) / (n_perm + 1)  # conservative estimate

    return {
        "mmd2": float(mmd2_obs),
        "pval": float(pval),
        "bandwidth": float(bw),
        "n_perm": int(n_perm),
        "n_real": int(m),
        "n_pred": int(n),
    }
