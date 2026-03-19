# -*- coding: utf-8 -*-
"""
src/evaluation/stat_tests/jsd.py — Jensen-Shannon Divergence for 2-D I/Q residuals.

jsd_2d  — Estimate JSD between two 2-D I/Q point clouds via joint histogram.
"""

import numpy as np


def jsd_2d(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """
    Estimate Jensen-Shannon Divergence between two 2-D I/Q point clouds.

    Uses a joint 2-D histogram on the combined data range as a non-parametric
    density estimator.  Returns JSD in nats ∈ [0, ln 2 ≈ 0.693].

    Parameters
    ----------
    a    : ndarray (N, 2) — reference I/Q (e.g. real channel residuals)
    b    : ndarray (M, 2) — test I/Q     (e.g. predicted residuals)
    bins : int — number of histogram bins per axis (default 64)

    Returns
    -------
    float — JSD in nats.  0 = identical distributions; ln(2) ≈ 0.693 = maximally different.

    Notes
    -----
    JSD = 0.5 · KL(p‖m) + 0.5 · KL(q‖m)  where m = 0.5·(p + q).
    Unlike KL divergence, JSD is symmetric and always finite.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.ndim != 2 or a.shape[1] < 2 or b.ndim != 2 or b.shape[1] < 2:
        return float("nan")

    # Shared bin edges spanning both distributions
    all_data = np.concatenate([a[:, :2], b[:, :2]], axis=0)
    lo = all_data.min(axis=0)
    hi = all_data.max(axis=0)

    # Guard against degenerate (all-same) data
    rng = hi - lo
    eps = 1e-12
    hi = np.where(rng < eps, lo + eps, hi)

    edges = [np.linspace(lo[i], hi[i], bins + 1) for i in range(2)]

    H_a, _ = np.histogramdd(a[:, :2], bins=edges)
    H_b, _ = np.histogramdd(b[:, :2], bins=edges)

    p = H_a.ravel().astype(np.float64)
    q = H_b.ravel().astype(np.float64)
    p /= (p.sum() + 1e-300)
    q /= (q.sum() + 1e-300)

    m = 0.5 * (p + q)

    # KL(p‖m) + KL(q‖m) with numerical guards
    def _kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log(x[mask] / (y[mask] + 1e-300))))

    jsd = 0.5 * (_kl(p, m) + _kl(q, m))
    return float(max(0.0, jsd))
