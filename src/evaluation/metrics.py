# -*- coding: utf-8 -*-
"""
src/evaluation/metrics.py — Shared signal-quality and residual-distribution metrics.

Extracted from cvae_TRAIN_documented.py and analise_cvae_reviewed.py
(Commit 3E).  Both monoliths contained identical copies; this module is
the single source of truth.

Functions
---------
calculate_evm   – EVM (%) and EVM (dB) between reference and test I/Q.
calculate_snr   – SNR (dB) between reference and test I/Q.
_skew_kurt      – Per-axis skewness and excess kurtosis.
_psd_log        – Log10 PSD estimate (Welch-like, Hanning window).
residual_distribution_metrics – Aggregate residual Δ statistics.
"""

import numpy as np


# ==========================================================================
# EVM / SNR
# ==========================================================================
def calculate_evm(ref, test):
    ref = np.asarray(ref); test = np.asarray(test)
    rc = ref[:,0] + 1j*ref[:,1]
    tc = test[:,0] + 1j*test[:,1]
    mean_power = np.mean(np.abs(rc)**2)
    if mean_power == 0:
        return float("inf"), float("-inf")
    evm = np.sqrt(np.mean(np.abs(tc-rc)**2) / mean_power)
    return float(evm*100), float(20*np.log10(max(evm, 1e-12)))

def calculate_snr(ref, test):
    ref = np.asarray(ref); test = np.asarray(test)
    rc = ref[:,0] + 1j*ref[:,1]
    tc = test[:,0] + 1j*test[:,1]
    sp = np.mean(np.abs(rc)**2)
    npow = np.mean(np.abs(rc-tc)**2)
    if npow == 0:
        return float("inf")
    return float(10*np.log10(max(sp/npow, 1e-12)))


# ==========================================================================
# Numeric helpers (used by residual_distribution_metrics and plots)
# ==========================================================================
def _skew_kurt(x: np.ndarray, eps: float = 1e-12):
    x = np.asarray(x, dtype=np.float64)
    m = np.mean(x, axis=0)
    v = np.var(x, axis=0)
    s = np.sqrt(v + eps)
    z = (x - m) / s
    skew = np.mean(z**3, axis=0)
    kurt = np.mean(z**4, axis=0) - 3.0
    return skew, kurt

def _psd_log(xc: np.ndarray, nfft: int = 2048, eps: float = 1e-12):
    xc = np.asarray(xc, dtype=np.complex128).ravel()
    n = len(xc)
    nfft = int(min(max(256, nfft), n)) if n > 0 else int(nfft)
    if nfft < 256 or n < 256:
        nfft = max(1, n)
    win = np.hanning(nfft) if nfft >= 8 else np.ones(nfft)
    nseg = 4
    hop = max(1, (n - nfft) // max(1, nseg-1)) if n > nfft else nfft
    acc = None
    cnt = 0
    for start in range(0, max(1, n - nfft + 1), hop):
        seg = xc[start:start+nfft]
        if len(seg) < nfft:
            break
        segw = seg * win
        X = np.fft.fft(segw, n=nfft)
        P = (np.abs(X)**2) / (np.sum(win**2) + eps)
        acc = P if acc is None else (acc + P)
        cnt += 1
        if cnt >= nseg:
            break
    if acc is None:
        acc = (np.abs(np.fft.fft(xc, n=nfft))**2) / max(1, nfft)
        cnt = 1
    psd = acc / max(1, cnt)
    return np.log10(psd + eps)


# ==========================================================================
# Residual distribution comparison
# ==========================================================================
def residual_distribution_metrics(
    X: np.ndarray,
    Y: np.ndarray,
    Yp: np.ndarray,
    psd_nfft: int = 2048,
    gauss_alpha: float = 0.01,
):
    d_real = np.asarray(Y) - np.asarray(X)
    d_pred = np.asarray(Yp) - np.asarray(X)

    mean_l2 = float(np.linalg.norm(np.mean(d_pred, axis=0) - np.mean(d_real, axis=0)))
    cov_fro = float(np.linalg.norm(np.cov(d_pred.T) - np.cov(d_real.T), ord="fro"))

    var_real = float(np.mean(np.var(d_real, axis=0)))
    var_pred = float(np.mean(np.var(d_pred, axis=0)))

    skew_r, kurt_r = _skew_kurt(d_real)
    skew_p, kurt_p = _skew_kurt(d_pred)
    skew_l2 = float(np.linalg.norm(skew_p - skew_r))
    kurt_l2 = float(np.linalg.norm(kurt_p - kurt_r))

    cr = d_real[:, 0] + 1j * d_real[:, 1]
    cp = d_pred[:, 0] + 1j * d_pred[:, 1]
    psd_r = _psd_log(cr, nfft=int(psd_nfft))
    psd_p = _psd_log(cp, nfft=int(psd_nfft))
    psd_l2 = float(np.linalg.norm(psd_p - psd_r) / np.sqrt(len(psd_r) if len(psd_r) else 1))

    out = {
        "delta_mean_l2": mean_l2,
        "delta_cov_fro": cov_fro,
        "var_real_delta": var_real,
        "var_pred_delta": var_pred,
        "delta_skew_l2": skew_l2,
        "delta_kurt_l2": kurt_l2,
        "delta_psd_l2": psd_l2,
    }

    # Keep JB fields aligned with src.metrics.distribution (auditability / underflow-safe log10(p)).
    try:
        from src.metrics.distribution import gaussianity_tests

        g = gaussianity_tests(d_pred, alpha=float(gauss_alpha))
        out.update({
            "jb_stat_I": float(g.get("jb_stat_I")),
            "jb_stat_Q": float(g.get("jb_stat_Q")),
            "jb_p_I": float(g.get("jb_p_I")),
            "jb_p_Q": float(g.get("jb_p_Q")),
            "jb_p_min": float(g.get("jb_p_min")),
            "jb_log10p_I": float(g.get("jb_log10p_I")),
            "jb_log10p_Q": float(g.get("jb_log10p_Q")),
            "jb_log10p_min": float(g.get("jb_log10p_min")),
            "reject_gaussian": bool(g.get("reject_gaussian", False)),
        })
    except Exception:
        # Evaluation should stay robust even if optional JB computation fails.
        out.update({
            "jb_stat_I": float("nan"),
            "jb_stat_Q": float("nan"),
            "jb_p_I": float("nan"),
            "jb_p_Q": float("nan"),
            "jb_p_min": float("nan"),
            "jb_log10p_I": float("nan"),
            "jb_log10p_Q": float("nan"),
            "jb_log10p_min": float("nan"),
            "reject_gaussian": False,
        })
    return out
