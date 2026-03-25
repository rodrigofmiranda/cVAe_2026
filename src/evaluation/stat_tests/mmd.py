# -*- coding: utf-8 -*-
"""
Two-sample Maximum Mean Discrepancy (MMD) test with RBF kernel.

Implements the unbiased estimator MMD²_u (Gretton et al., 2012) with a
permutation-based p-value.  Bandwidth is chosen via the *median heuristic*
with a robust fallback.

When a TensorFlow GPU is available the permutation null distribution is
computed via a single batched matmul on the GPU — typically 10-20× faster
than the multi-threaded BLAS path on CPU.

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
# GPU availability (lazy, cached)
# ------------------------------------------------------------------

_GPU_AVAILABLE: bool | None = None


def _check_gpu() -> bool:
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        try:
            import tensorflow as tf
            _GPU_AVAILABLE = len(tf.config.list_physical_devices("GPU")) > 0
        except Exception:
            _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


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
# GPU-accelerated permutation null
# ------------------------------------------------------------------

def _perm_pval_gpu(
    K_np: np.ndarray,
    m: int,
    n: int,
    n_perm: int,
    seed: int,
    mmd2_obs: float,
    chunk_size: int = 500,
) -> float:
    """GPU-batched permutation p-value for MMD².

    Generates indicator vectors on CPU (preserving RNG sequence) and
    performs batched K @ E matmul on GPU in chunks to control memory.
    """
    import tensorflow as tf

    N = m + n

    # Move Gram matrix to GPU once
    K = tf.constant(K_np, dtype=tf.float64)
    diag_K = tf.linalg.diag_part(K)        # (N,)
    row_sums = tf.reduce_sum(K, axis=1)     # (N,)

    denom_xx = tf.constant(float(m * (m - 1)), dtype=tf.float64)
    denom_yy = tf.constant(float(n * (n - 1)), dtype=tf.float64)
    denom_xy = tf.constant(float(m * n), dtype=tf.float64)

    rng = np.random.default_rng(seed)
    count_ge = 0

    for chunk_start in range(0, n_perm, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_perm)
        cs = chunk_end - chunk_start

        # Build indicator matrix on CPU (same RNG sequence as CPU path)
        E_np = np.zeros((N, cs), dtype=np.float64)
        for p in range(cs):
            idx = rng.permutation(N)
            E_np[idx[:m], p] = 1.0

        E = tf.constant(E_np)

        # Single batched matmul — replaces cs individual dgemv calls
        KE = tf.matmul(K, E)               # (N, cs)

        E_comp = 1.0 - E
        s_a  = tf.reduce_sum(KE * E, axis=0)
        s_ab = tf.reduce_sum(KE * E_comp, axis=0)
        s_b  = tf.reduce_sum((row_sums[:, None] - KE) * E_comp, axis=0)
        da   = tf.reduce_sum(diag_K[:, None] * E, axis=0)
        db   = tf.reduce_sum(diag_K[:, None] * E_comp, axis=0)

        term_xx = (s_a - da) / denom_xx
        term_yy = (s_b - db) / denom_yy
        term_xy = s_ab / denom_xy
        mmd2_perms = term_xx + term_yy - 2.0 * term_xy     # (cs,)

        count_ge += int(tf.reduce_sum(
            tf.cast(mmd2_perms >= mmd2_obs, tf.int32)
        ).numpy())

    return (count_ge + 1) / (n_perm + 1)


# ------------------------------------------------------------------
# CPU permutation null (original)
# ------------------------------------------------------------------

def _perm_pval_cpu(
    K: np.ndarray,
    m: int,
    n: int,
    n_perm: int,
    seed: int,
    mmd2_obs: float,
) -> float:
    """CPU permutation p-value (BLAS dgemv loop)."""
    N = m + n
    diag_K = np.diag(K).copy()
    row_sums = K.sum(axis=1)
    rng = np.random.default_rng(seed)
    count_ge = 0
    e = np.empty(N, dtype=np.float64)

    for _ in range(n_perm):
        idx = rng.permutation(N)
        a = idx[:m]
        b = idx[m:]

        e[:] = 0.0
        e[a] = 1.0

        Ke = K @ e
        s_a = float(Ke[a].sum())
        s_ab = float(Ke[b].sum())
        s_b = float((row_sums[b] - Ke[b]).sum())

        da = float(diag_K[a].sum())
        db = float(diag_K[b].sum())

        term_xx = (s_a - da) / max(1, m * (m - 1))
        term_yy = (s_b - db) / max(1, n * (n - 1))
        term_xy = s_ab / max(1, m * n)
        mmd2_perm = term_xx + term_yy - 2.0 * term_xy

        if mmd2_perm >= mmd2_obs:
            count_ge += 1

    return (count_ge + 1) / (n_perm + 1)


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
    N = m + n

    # Pre-compute full Gram on pooled data (symmetric)
    K = _gram_rbf(Z, Z, bw)

    # Observed statistic
    Kxx = K[:m, :m].copy()
    Kyy = K[m:, m:].copy()
    Kxy = K[:m, m:]
    mmd2_obs = _mmd2_unbiased(Kxx, Kyy, Kxy)

    # Permutation p-value — GPU or CPU
    use_gpu = _check_gpu()
    if use_gpu:
        try:
            pval = _perm_pval_gpu(K, m, n, n_perm, seed, mmd2_obs)
        except Exception as exc:
            print(f"⚠️  MMD GPU fallback to CPU: {exc}")
            pval = _perm_pval_cpu(K, m, n, n_perm, seed, mmd2_obs)
    else:
        pval = _perm_pval_cpu(K, m, n, n_perm, seed, mmd2_obs)

    return {
        "mmd2": float(mmd2_obs),
        "pval": float(pval),
        "bandwidth": float(bw),
        "n_perm": int(n_perm),
        "n_real": int(m),
        "n_pred": int(n),
    }
