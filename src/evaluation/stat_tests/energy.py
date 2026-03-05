# -*- coding: utf-8 -*-
"""
Two-sample Energy Distance test with permutation p-value.

The energy distance (Székely & Rizzo, 2004) is a distribution-free
metric that equals zero iff the two distributions are identical:

    E(X,Y) = 2·E||X−Y|| − E||X−X'|| − E||Y−Y'||

Public API
----------
energy_test(Y_real, Y_pred, *, n_perm=200, seed=42)
    → dict  {"energy": float, "pval": float, "n_perm": int,
             "n_real": int, "n_pred": int}

Commit: refactor(etapaA1).
"""
from __future__ import annotations

from typing import Dict

import numpy as np


# ------------------------------------------------------------------
# Core statistic
# ------------------------------------------------------------------

def _energy_statistic(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute the (unnormalised) energy distance between X and Y."""
    # E||X−Y||
    # Using the identity: E||a−b|| over pairs via cdist
    # For moderate n we compute pairwise norms directly.
    # For large n we sub-sample to keep O(n²) tractable.
    from scipy.spatial.distance import cdist

    m, n = len(X), len(Y)

    dxy = cdist(X, Y, metric="euclidean")
    mean_xy = dxy.mean()

    dxx = cdist(X, X, metric="euclidean")
    mean_xx = dxx.mean()

    dyy = cdist(Y, Y, metric="euclidean")
    mean_yy = dyy.mean()

    # Energy distance (scaled by mn/(m+n) for the test statistic)
    e_dist = 2.0 * mean_xy - mean_xx - mean_yy
    return float(e_dist)


def _energy_statistic_fast(X: np.ndarray, Y: np.ndarray,
                           max_pairs: int = 8000) -> float:
    """Sub-sampled energy statistic for large datasets.

    When m or n > max_pairs we draw random sub-samples to cap
    computation at O(max_pairs²).
    """
    rng = np.random.default_rng(1234)
    m, n = len(X), len(Y)
    if m > max_pairs:
        X = X[rng.choice(m, max_pairs, replace=False)]
    if n > max_pairs:
        Y = Y[rng.choice(n, max_pairs, replace=False)]
    return _energy_statistic(X, Y)


# ------------------------------------------------------------------
# Public entry-point
# ------------------------------------------------------------------

def energy_test(
    Y_real: np.ndarray,
    Y_pred: np.ndarray,
    *,
    n_perm: int = 200,
    seed: int = 42,
    max_pairs: int = 8000,
) -> Dict[str, float]:
    """Two-sample energy distance test with permutation p-value.

    Parameters
    ----------
    Y_real, Y_pred : ndarray (n, 2)
        IQ samples from the real channel and the twin, respectively.
    n_perm : int
        Number of permutations (200 = quick, 2000 = full).
    seed : int
        RNG seed for reproducibility.
    max_pairs : int
        Sub-sample cap per set (controls O(n²) cost).

    Returns
    -------
    dict with keys ``energy``, ``pval``, ``n_perm``, ``n_real``,
    ``n_pred``.
    """
    X = np.asarray(Y_real, dtype=np.float64)
    Y = np.asarray(Y_pred, dtype=np.float64)
    assert X.ndim == 2 and Y.ndim == 2 and X.shape[1] == Y.shape[1]

    m, n = len(X), len(Y)

    # Observed statistic (sub-sampled if large)
    e_obs = _energy_statistic_fast(X, Y, max_pairs=max_pairs)

    # Permutation null
    rng = np.random.default_rng(seed)
    Z = np.concatenate([X, Y], axis=0)
    count_ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(m + n)
        Xp = Z[idx[:m]]
        Yp = Z[idx[m:]]
        e_perm = _energy_statistic_fast(Xp, Yp, max_pairs=max_pairs)
        if e_perm >= e_obs:
            count_ge += 1

    pval = (count_ge + 1) / (n_perm + 1)

    return {
        "energy": float(e_obs),
        "pval": float(pval),
        "n_perm": int(n_perm),
        "n_real": int(m),
        "n_pred": int(n),
    }
