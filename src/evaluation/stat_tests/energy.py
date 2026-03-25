# -*- coding: utf-8 -*-
"""
Two-sample Energy Distance test with permutation p-value.

The energy distance (Székely & Rizzo, 2004) is a distribution-free
metric that equals zero iff the two distributions are identical:

    E(X,Y) = 2·E||X−Y|| − E||X−X'|| − E||Y−Y'||

When a TensorFlow GPU is available the permutation null distribution is
computed via a single batched matmul on the GPU — typically 10-20× faster
than the multi-threaded BLAS path on CPU.

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
# GPU-accelerated permutation null
# ------------------------------------------------------------------

def _perm_pval_gpu(
    D_np: np.ndarray,
    ms: int,
    ns: int,
    n_perm: int,
    seed: int,
    e_obs: float,
    chunk_size: int = 500,
) -> float:
    """GPU-batched permutation p-value for Energy distance.

    Generates indicator vectors on CPU (preserving RNG sequence) and
    performs batched D @ E matmul on GPU in chunks to control memory.
    """
    import tensorflow as tf

    Ns = ms + ns

    # Move distance matrix to GPU once
    D = tf.constant(D_np, dtype=tf.float64)
    row_sums_D = tf.reduce_sum(D, axis=1)     # (Ns,)

    denom_xy = tf.constant(float(ms * ns), dtype=tf.float64)
    denom_xx = tf.constant(float(ms * ms), dtype=tf.float64)
    denom_yy = tf.constant(float(ns * ns), dtype=tf.float64)

    rng = np.random.default_rng(seed)
    count_ge = 0

    for chunk_start in range(0, n_perm, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_perm)
        cs = chunk_end - chunk_start

        # Build indicator matrix on CPU (same RNG sequence as CPU path)
        E_np = np.zeros((Ns, cs), dtype=np.float64)
        for p in range(cs):
            idx = rng.permutation(Ns)
            E_np[idx[:ms], p] = 1.0

        E = tf.constant(E_np)

        # Single batched matmul
        DE = tf.matmul(D, E)                # (Ns, cs)

        E_comp = 1.0 - E
        sum_xy = tf.reduce_sum(DE * E_comp, axis=0)        # (cs,)
        sum_xx = tf.reduce_sum(DE * E, axis=0)              # (cs,)
        sum_yy = tf.reduce_sum(
            (row_sums_D[:, None] - DE) * E_comp, axis=0
        )                                                    # (cs,)

        mean_xy = sum_xy / denom_xy
        mean_xx = sum_xx / denom_xx
        mean_yy = sum_yy / denom_yy
        e_perms = 2.0 * mean_xy - mean_xx - mean_yy        # (cs,)

        count_ge += int(tf.reduce_sum(
            tf.cast(e_perms >= e_obs, tf.int32)
        ).numpy())

    return (count_ge + 1) / (n_perm + 1)


# ------------------------------------------------------------------
# CPU permutation null (original)
# ------------------------------------------------------------------

def _perm_pval_cpu(
    D: np.ndarray,
    ms: int,
    ns: int,
    n_perm: int,
    seed: int,
    e_obs: float,
) -> float:
    """CPU permutation p-value (BLAS dgemv loop)."""
    Ns = ms + ns
    row_sums_D = D.sum(axis=1)
    rng = np.random.default_rng(seed)
    count_ge = 0
    e = np.empty(Ns, dtype=np.float64)

    for _ in range(n_perm):
        idx = rng.permutation(Ns)
        a = idx[:ms]
        b = idx[ms:]

        e[:] = 0.0
        e[a] = 1.0

        De = D @ e
        sum_xy = float(De[b].sum())
        sum_xx = float(De[a].sum())
        sum_yy = float((row_sums_D[b] - De[b]).sum())

        mean_xy = sum_xy / max(1, ms * ns)
        mean_xx = sum_xx / max(1, ms * ms)
        mean_yy = sum_yy / max(1, ns * ns)
        e_perm = 2.0 * mean_xy - mean_xx - mean_yy

        if e_perm >= e_obs:
            count_ge += 1

    return (count_ge + 1) / (n_perm + 1)


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

    # ----- Permutation null -----
    from scipy.spatial.distance import cdist as _cdist

    rng = np.random.default_rng(seed)
    Z = np.concatenate([X, Y], axis=0)
    N = m + n

    # Sub-sample the pool if needed (same budget as the statistic)
    _ms = min(m, max_pairs)
    _ns = min(n, max_pairs)
    _Ns = _ms + _ns
    if N > _Ns:
        _pool_idx = rng.choice(N, _Ns, replace=False)
        Z_sub = Z[_pool_idx]
    else:
        _Ns = N
        _ms, _ns = m, n
        Z_sub = Z

    D = _cdist(Z_sub, Z_sub, metric="euclidean").astype(np.float64)

    # GPU or CPU dispatch
    use_gpu = _check_gpu()
    if use_gpu:
        try:
            pval = _perm_pval_gpu(D, _ms, _ns, n_perm, seed, e_obs)
        except Exception as exc:
            print(f"⚠️  Energy GPU fallback to CPU: {exc}")
            pval = _perm_pval_cpu(D, _ms, _ns, n_perm, seed, e_obs)
    else:
        pval = _perm_pval_cpu(D, _ms, _ns, n_perm, seed, e_obs)

    return {
        "energy": float(e_obs),
        "pval": float(pval),
        "n_perm": int(n_perm),
        "n_real": int(m),
        "n_pred": int(n),
    }
