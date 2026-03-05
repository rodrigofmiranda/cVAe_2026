# -*- coding: utf-8 -*-
"""
PSD distance with bootstrap confidence interval.

Re-uses the existing ``_psd_log`` estimator from
``src.evaluation.metrics`` (Hanning-windowed, Welch-like overlap) to
keep numbers consistent with the rest of the pipeline.

The distance is the normalised L2 between the log10-PSD vectors of
the complex IQ residual:

    psd_dist = ||PSD_real − PSD_pred||₂ / √nfft

A bootstrap is used to estimate CI without distributional assumptions.

Public API
----------
psd_distance(Y_real, Y_pred, X=None, *, nfft=2048, n_boot=500, ci=0.95, seed=42)
    → dict  {"psd_dist": float, "psd_ci_low": float, "psd_ci_high": float,
             "nfft": int, "n_boot": int}

If *X* is provided the PSD is computed on the residual Δ=Y−X (consistent
with the existing ``residual_distribution_metrics``).  If *X* is None the
PSD is computed directly on Y.

Commit: refactor(etapaA1).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _iq_to_complex(Y: np.ndarray) -> np.ndarray:
    """Convert (n, 2) IQ array to complex-valued 1-D array."""
    Y = np.asarray(Y)
    return Y[:, 0] + 1j * Y[:, 1]


def _psd_log_local(xc: np.ndarray, nfft: int = 2048,
                   eps: float = 1e-12) -> np.ndarray:
    """Welch-like log10 PSD — mirrors ``src.evaluation.metrics._psd_log``."""
    xc = np.asarray(xc, dtype=np.complex128).ravel()
    n = len(xc)
    nfft = int(min(max(256, nfft), n)) if n > 0 else int(nfft)
    if nfft < 256 or n < 256:
        nfft = max(1, n)
    win = np.hanning(nfft) if nfft >= 8 else np.ones(nfft)
    nseg = 4
    hop = max(1, (n - nfft) // max(1, nseg - 1)) if n > nfft else nfft
    acc = None
    cnt = 0
    for start in range(0, max(1, n - nfft + 1), hop):
        seg = xc[start : start + nfft]
        if len(seg) < nfft:
            break
        segw = seg * win
        X = np.fft.fft(segw, n=nfft)
        P = (np.abs(X) ** 2) / (np.sum(win ** 2) + eps)
        acc = P if acc is None else (acc + P)
        cnt += 1
        if cnt >= nseg:
            break
    if acc is None:
        acc = (np.abs(np.fft.fft(xc, n=nfft)) ** 2) / max(1, nfft)
        cnt = 1
    psd = acc / max(1, cnt)
    return np.log10(psd + eps)


def _psd_l2(c_real: np.ndarray, c_pred: np.ndarray,
            nfft: int) -> float:
    """Normalised L2 between log-PSD vectors."""
    p_r = _psd_log_local(c_real, nfft=nfft)
    p_p = _psd_log_local(c_pred, nfft=nfft)
    return float(np.linalg.norm(p_p - p_r) / np.sqrt(max(1, len(p_r))))


# ------------------------------------------------------------------
# Public entry-point
# ------------------------------------------------------------------

def psd_distance(
    Y_real: np.ndarray,
    Y_pred: np.ndarray,
    X: Optional[np.ndarray] = None,
    *,
    nfft: int = 2048,
    n_boot: int = 500,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """PSD distance between real and predicted IQ with bootstrap CI.

    Parameters
    ----------
    Y_real, Y_pred : ndarray (n, 2)
        IQ samples.
    X : ndarray (n, 2), optional
        Transmitted IQ.  If given, PSD is computed on residual Δ=Y−X.
    nfft : int
        FFT size (passed to Welch estimator).
    n_boot : int
        Number of bootstrap replicates for the CI.
    ci : float
        Confidence level (default 0.95 → 2.5%–97.5% interval).
    seed : int
        RNG seed.

    Returns
    -------
    dict with keys ``psd_dist``, ``psd_ci_low``, ``psd_ci_high``,
    ``nfft``, ``n_boot``.
    """
    Yr = np.asarray(Y_real, dtype=np.float64)
    Yp = np.asarray(Y_pred, dtype=np.float64)

    if X is not None:
        Xr = np.asarray(X, dtype=np.float64)
        Yr = Yr - Xr
        Yp = Yp - Xr

    cr = _iq_to_complex(Yr)
    cp = _iq_to_complex(Yp)

    # Point estimate
    dist_obs = _psd_l2(cr, cp, nfft)

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    n = min(len(cr), len(cp))
    boot_dists = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx_r = rng.choice(n, n, replace=True)
        idx_p = rng.choice(n, n, replace=True)
        boot_dists[b] = _psd_l2(cr[idx_r], cp[idx_p], nfft)

    alpha = 1.0 - ci
    lo = float(np.percentile(boot_dists, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_dists, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "psd_dist": dist_obs,
        "psd_ci_low": lo,
        "psd_ci_high": hi,
        "nfft": int(nfft),
        "n_boot": int(n_boot),
    }
