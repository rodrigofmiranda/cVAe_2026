# -*- coding: utf-8 -*-
"""
src/evaluation/metrics.py — Shared signal-quality and residual-distribution metrics.

Shared metrics for the canonical training and evaluation pipelines.

Canonical δ definition
----------------------
δ = Y − X (element-wise).  X and Y are the float32 (N, 2) arrays stored per
experiment; their signal-chain semantics are defined by the dataset preparation
pipeline (see src/data/channel_dataset.py), not asserted here.

Functions
---------
calculate_evm   – EVM (%) and EVM (dB) between reference and test I/Q.
calculate_snr   – SNR (dB) between reference and test I/Q.
_skew_kurt      – Per-axis skewness and excess kurtosis.
_psd_log        – Log10 PSD estimate (Welch-like, Hanning window).
_acf_curve_complex – Normalised residual ACF curve on complex IQ.
residual_distribution_metrics – Aggregate residual Δ statistics.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.data.support_geometry import support_feature_dict, support_region_labels


_SIGNATURE_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
_COVERAGE_LEVELS = (0.50, 0.80, 0.95)


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
    hop = max(1, nfft // 2)   # 50 % overlap — standard Welch
    max_segs = 256             # cap to bound runtime
    acc = None
    cnt = 0
    for start in range(0, max(1, n - nfft + 1), hop):
        seg = xc[start:start + nfft]
        if len(seg) < nfft:
            break
        X = np.fft.fft(seg * win, n=nfft)
        P = (np.abs(X) ** 2) / (np.sum(win ** 2) + eps)
        acc = P if acc is None else (acc + P)
        cnt += 1
        if cnt >= max_segs:
            break
    if acc is None:
        acc = (np.abs(np.fft.fft(xc, n=nfft)) ** 2) / max(1, nfft)
        cnt = 1
    psd = acc / max(1, cnt)
    return np.log10(psd + eps)


def _acf_curve_complex(xc: np.ndarray, max_lag: int = 128) -> np.ndarray:
    xc = np.asarray(xc, dtype=np.complex128).ravel()
    xc = xc - np.mean(xc)
    if len(xc) == 0:
        return np.zeros(1, dtype=np.float64)
    denom = float(np.vdot(xc, xc).real)
    if denom <= 1e-12:
        return np.zeros(max_lag + 1, dtype=np.float64)
    out = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        if lag == 0:
            out[lag] = 1.0
        else:
            out[lag] = float(np.vdot(xc[:-lag], xc[lag:]).real / denom)
    return out


def _quantile_interp(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64)
    try:
        return np.quantile(x, q, method="linear")
    except TypeError:
        return np.quantile(x, q, interpolation="linear")


def _wasserstein_1d(x: np.ndarray, y: np.ndarray, max_quantiles: int = 2048) -> float:
    """Approximate 1D Wasserstein-1 distance by quantile matching."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    n_q = int(min(max(len(x), len(y), 256), max_quantiles))
    q = np.linspace(0.0, 1.0, n_q, endpoint=True)
    xq = _quantile_interp(x, q)
    yq = _quantile_interp(y, q)
    return float(np.mean(np.abs(xq - yq)))


def _quantile_map(x: np.ndarray, quantiles: Sequence[float]) -> Dict[str, float]:
    vals = _quantile_interp(x, np.asarray(quantiles, dtype=np.float64))
    out: Dict[str, float] = {}
    for q, v in zip(quantiles, vals):
        qlab = f"q{int(round(float(q) * 100)):02d}"
        out[qlab] = float(v)
    return out


def _tail_probability(x: np.ndarray, threshold: float) -> float:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0 or not np.isfinite(threshold):
        return float("nan")
    return float(np.mean(np.abs(arr) > float(threshold)))


def _corr_iq(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
        return float("nan")
    std = np.std(arr, axis=0)
    if np.any(std < 1e-12):
        return float("nan")
    return float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])


def _ellipse_axis_ratio(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
        return float("nan")
    cov = np.cov(arr.T)
    try:
        eigvals = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return float("nan")
    eigvals = np.sort(np.asarray(eigvals, dtype=np.float64))
    if eigvals.size != 2 or eigvals[0] <= 1e-12:
        return float("nan")
    return float(np.sqrt(eigvals[1] / eigvals[0]))


def _coverage_from_samples(
    y_true: np.ndarray,
    y_samples: Optional[np.ndarray],
    levels: Sequence[float] = _COVERAGE_LEVELS,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if y_samples is None:
        for level in levels:
            tag = int(round(float(level) * 100))
            out[f"coverage_{tag}_I"] = float("nan")
            out[f"coverage_{tag}_Q"] = float("nan")
            out[f"coverage_{tag}"] = float("nan")
        return out

    ys = np.asarray(y_samples, dtype=np.float64)
    yt = np.asarray(y_true, dtype=np.float64)
    if ys.ndim != 3 or ys.shape[1:] != yt.shape:
        for level in levels:
            tag = int(round(float(level) * 100))
            out[f"coverage_{tag}_I"] = float("nan")
            out[f"coverage_{tag}_Q"] = float("nan")
            out[f"coverage_{tag}"] = float("nan")
        return out

    for level in levels:
        alpha = (1.0 - float(level)) / 2.0
        lo = np.quantile(ys, alpha, axis=0)
        hi = np.quantile(ys, 1.0 - alpha, axis=0)
        cov_i = np.mean((yt[:, 0] >= lo[:, 0]) & (yt[:, 0] <= hi[:, 0]))
        cov_q = np.mean((yt[:, 1] >= lo[:, 1]) & (yt[:, 1] <= hi[:, 1]))
        tag = int(round(float(level) * 100))
        out[f"coverage_{tag}_I"] = float(cov_i)
        out[f"coverage_{tag}_Q"] = float(cov_q)
        out[f"coverage_{tag}"] = float(0.5 * (cov_i + cov_q))
    return out


def residual_stat_signature(
    X: np.ndarray,
    Y: np.ndarray,
    Yp: np.ndarray,
    *,
    Y_samples: Optional[np.ndarray] = None,
    coverage_target: Optional[np.ndarray] = None,
    quantiles: Sequence[float] = _SIGNATURE_QUANTILES,
    coverage_levels: Sequence[float] = _COVERAGE_LEVELS,
) -> Dict[str, float]:
    """Return the canonical residual-noise signature for real vs predicted data."""
    X_arr = np.asarray(X, dtype=np.float64)
    d_real = np.asarray(Y, dtype=np.float64) - X_arr
    d_pred = np.asarray(Yp, dtype=np.float64) - X_arr

    mean_real = np.mean(d_real, axis=0)
    mean_pred = np.mean(d_pred, axis=0)
    std_real = np.std(d_real, axis=0)
    std_pred = np.std(d_pred, axis=0)
    var_real = np.var(d_real, axis=0)
    var_pred = np.var(d_pred, axis=0)
    iqr_real = np.quantile(d_real, 0.75, axis=0) - np.quantile(d_real, 0.25, axis=0)
    iqr_pred = np.quantile(d_pred, 0.75, axis=0) - np.quantile(d_pred, 0.25, axis=0)

    r_real = np.linalg.norm(d_real, axis=1)
    r_pred = np.linalg.norm(d_pred, axis=1)
    radial_real = _quantile_map(r_real, (0.05, 0.50, 0.95))
    radial_pred = _quantile_map(r_pred, (0.05, 0.50, 0.95))

    out: Dict[str, float] = {
        "var_ratio_I": float(var_pred[0] / var_real[0]) if var_real[0] > 0 else float("nan"),
        "var_ratio_Q": float(var_pred[1] / var_real[1]) if var_real[1] > 0 else float("nan"),
        "iqr_real_I": float(iqr_real[0]),
        "iqr_real_Q": float(iqr_real[1]),
        "iqr_pred_I": float(iqr_pred[0]),
        "iqr_pred_Q": float(iqr_pred[1]),
        "delta_iqr_I": float(iqr_pred[0] - iqr_real[0]),
        "delta_iqr_Q": float(iqr_pred[1] - iqr_real[1]),
        "radial_wasserstein": _wasserstein_1d(r_real, r_pred),
        "radial_q05_real": radial_real["q05"],
        "radial_q50_real": radial_real["q50"],
        "radial_q95_real": radial_real["q95"],
        "radial_q05_pred": radial_pred["q05"],
        "radial_q50_pred": radial_pred["q50"],
        "radial_q95_pred": radial_pred["q95"],
        "delta_radial_q05": float(radial_pred["q05"] - radial_real["q05"]),
        "delta_radial_q50": float(radial_pred["q50"] - radial_real["q50"]),
        "delta_radial_q95": float(radial_pred["q95"] - radial_real["q95"]),
        "corr_iq_real": _corr_iq(d_real),
        "corr_iq_pred": _corr_iq(d_pred),
        "ellipse_axis_ratio_real": _ellipse_axis_ratio(d_real),
        "ellipse_axis_ratio_pred": _ellipse_axis_ratio(d_pred),
    }
    out["delta_corr_IQ"] = (
        float(out["corr_iq_pred"] - out["corr_iq_real"])
        if np.isfinite(out["corr_iq_pred"]) and np.isfinite(out["corr_iq_real"])
        else float("nan")
    )
    out["delta_ellipse_axis_ratio"] = (
        float(out["ellipse_axis_ratio_pred"] - out["ellipse_axis_ratio_real"])
        if np.isfinite(out["ellipse_axis_ratio_pred"]) and np.isfinite(out["ellipse_axis_ratio_real"])
        else float("nan")
    )

    for axis, idx in (("I", 0), ("Q", 1)):
        real_q = _quantile_map(d_real[:, idx], quantiles)
        pred_q = _quantile_map(d_pred[:, idx], quantiles)
        for qlab, real_v in real_q.items():
            pred_v = pred_q[qlab]
            out[f"{qlab}_real_{axis}"] = float(real_v)
            out[f"{qlab}_pred_{axis}"] = float(pred_v)
            out[f"delta_{qlab}_{axis}"] = float(pred_v - real_v)
        tail_2_real = _tail_probability(d_real[:, idx], 2.0 * std_real[idx])
        tail_2_pred = _tail_probability(d_pred[:, idx], 2.0 * std_real[idx])
        tail_3_real = _tail_probability(d_real[:, idx], 3.0 * std_real[idx])
        tail_3_pred = _tail_probability(d_pred[:, idx], 3.0 * std_real[idx])
        out[f"tail_p2sigma_real_{axis}"] = tail_2_real
        out[f"tail_p2sigma_pred_{axis}"] = tail_2_pred
        out[f"tail_p3sigma_real_{axis}"] = tail_3_real
        out[f"tail_p3sigma_pred_{axis}"] = tail_3_pred
        out[f"delta_tail_p2sigma_{axis}"] = float(tail_2_pred - tail_2_real)
        out[f"delta_tail_p3sigma_{axis}"] = float(tail_3_pred - tail_3_real)

    cov_target = np.asarray(coverage_target if coverage_target is not None else Y, dtype=np.float64)
    out.update(_coverage_from_samples(cov_target, Y_samples, levels=coverage_levels))
    for level in coverage_levels:
        tag = int(round(float(level) * 100))
        target = float(level)
        for suffix in ("", "_I", "_Q"):
            key = f"coverage_{tag}{suffix}"
            val = out.get(key, float("nan"))
            out[f"delta_coverage_{tag}{suffix}"] = (
                float(val - target) if np.isfinite(float(val)) else float("nan")
            )
    return out


def residual_signature_by_amplitude_bin(
    *,
    X_real: np.ndarray,
    Y_real: np.ndarray,
    X_pred: np.ndarray,
    Y_pred: np.ndarray,
    regime_id: str,
    regime_label: str = "",
    study: str = "",
    run_id: str = "",
    run_dir: str = "",
    model_run_dir: str = "",
    best_grid_tag: str = "",
    dist_target_m: float = float("nan"),
    curr_target_mA: float = float("nan"),
    amplitude_bins: int = 4,
    min_samples_per_bin: int = 512,
    stat_mode: str = "quick",
    stat_n_perm: int = 200,
    stat_seed: int = 42,
    stat_execution_backend: str = "cpu",
) -> List[Dict[str, Any]]:
    """Return amplitude-conditioned residual signature rows for one regime."""
    from src.evaluation.stat_tests import benjamini_hochberg, energy_test, mmd_rbf
    from src.metrics.distribution import residual_fidelity_metrics

    Xr = np.asarray(X_real, dtype=np.float64)
    Yr = np.asarray(Y_real, dtype=np.float64)
    Xp = np.asarray(X_pred, dtype=np.float64)
    Yp = np.asarray(Y_pred, dtype=np.float64)
    if len(Xr) == 0 or len(Xp) == 0:
        return []

    amp_real = np.linalg.norm(Xr, axis=1)
    quant_edges = np.quantile(
        amp_real,
        np.linspace(0.0, 1.0, int(max(2, amplitude_bins)) + 1),
    )
    edges = np.asarray(quant_edges, dtype=np.float64)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12

    amp_pred = np.linalg.norm(Xp, axis=1)
    rows: List[Dict[str, Any]] = []
    mmd_pvals: List[float] = []
    energy_pvals: List[float] = []

    for bi in range(len(edges) - 1):
        lo = float(edges[bi])
        hi = float(edges[bi + 1])
        is_last = bi == len(edges) - 2
        mask_real = (amp_real >= lo) & ((amp_real <= hi) if is_last else (amp_real < hi))
        mask_pred = (amp_pred >= lo) & ((amp_pred <= hi) if is_last else (amp_pred < hi))

        row: Dict[str, Any] = {
            "study": study,
            "regime_id": regime_id,
            "regime_label": regime_label,
            "run_id": run_id,
            "run_dir": run_dir,
            "model_run_dir": model_run_dir,
            "best_grid_tag": best_grid_tag,
            "dist_target_m": float(dist_target_m),
            "curr_target_mA": float(curr_target_mA),
            "amplitude_bin_index": int(bi),
            "amplitude_bin_label": f"q{int(100 * bi / (len(edges) - 1))}-{int(100 * (bi + 1) / (len(edges) - 1))}",
            "amplitude_lo": lo,
            "amplitude_hi": hi,
            "n_samples_real": int(mask_real.sum()),
            "n_samples_pred": int(mask_pred.sum()),
            "stat_mode": stat_mode,
        }

        if int(mask_real.sum()) < int(min_samples_per_bin) or int(mask_pred.sum()) < int(min_samples_per_bin):
            row.update({
                "std_real_delta_I": float("nan"),
                "std_real_delta_Q": float("nan"),
                "std_pred_delta_I": float("nan"),
                "std_pred_delta_Q": float("nan"),
                "delta_wasserstein_I": float("nan"),
                "delta_wasserstein_Q": float("nan"),
                "delta_jb_stat_rel_I": float("nan"),
                "delta_jb_stat_rel_Q": float("nan"),
                "stat_mmd_pval": float("nan"),
                "stat_mmd_qval": float("nan"),
                "stat_energy_pval": float("nan"),
                "stat_energy_qval": float("nan"),
            })
            rows.append(row)
            continue

        rr = Yr[mask_real] - Xr[mask_real]
        rp = Yp[mask_pred] - Xp[mask_pred]
        sig = residual_fidelity_metrics(rr, rp, max_samples=min(len(rr), len(rp)), X=Xr[mask_real], X_pred=Xp[mask_pred])
        row.update({
            "std_real_delta_I": float(sig.get("std_real_delta_I", float("nan"))),
            "std_real_delta_Q": float(sig.get("std_real_delta_Q", float("nan"))),
            "std_pred_delta_I": float(sig.get("std_pred_delta_I", float("nan"))),
            "std_pred_delta_Q": float(sig.get("std_pred_delta_Q", float("nan"))),
            "delta_wasserstein_I": float(sig.get("delta_wasserstein_I", float("nan"))),
            "delta_wasserstein_Q": float(sig.get("delta_wasserstein_Q", float("nan"))),
            "delta_jb_stat_rel_I": float(sig.get("delta_jb_stat_rel_I", float("nan"))),
            "delta_jb_stat_rel_Q": float(sig.get("delta_jb_stat_rel_Q", float("nan"))),
        })

        n_cmp = min(len(rr), len(rp))
        rng = np.random.RandomState(int(stat_seed) + bi)
        if n_cmp < len(rr):
            rr = rr[rng.choice(len(rr), n_cmp, replace=False)]
        if n_cmp < len(rp):
            rp = rp[rng.choice(len(rp), n_cmp, replace=False)]
        try:
            sf_mmd = mmd_rbf(
                rr,
                rp,
                n_perm=int(stat_n_perm),
                seed=int(stat_seed) + bi,
                execution_backend=stat_execution_backend,
            )
            sf_energy = energy_test(
                rr,
                rp,
                n_perm=int(stat_n_perm),
                seed=int(stat_seed) + bi,
                execution_backend=stat_execution_backend,
            )
            row["stat_mmd_pval"] = float(sf_mmd["pval"])
            row["stat_energy_pval"] = float(sf_energy["pval"])
            row["stat_mmd_qval"] = float("nan")
            row["stat_energy_qval"] = float("nan")
            rows.append(row)
            mmd_pvals.append(float(sf_mmd["pval"]))
            energy_pvals.append(float(sf_energy["pval"]))
        except Exception as exc:
            print(
                "⚠️  Residual signature amplitude-bin stat tests failed "
                f"for regime '{regime_id}' bin={bi}: {exc}"
            )
            row["stat_mmd_pval"] = float("nan")
            row["stat_energy_pval"] = float("nan")
            row["stat_mmd_qval"] = float("nan")
            row["stat_energy_qval"] = float("nan")
            rows.append(row)

    if rows:
        valid_mmd_idx = [i for i, r in enumerate(rows) if np.isfinite(float(r.get("stat_mmd_pval", float("nan"))))]
        valid_energy_idx = [i for i, r in enumerate(rows) if np.isfinite(float(r.get("stat_energy_pval", float("nan"))))]
        if valid_mmd_idx:
            qvals = benjamini_hochberg(np.asarray([rows[i]["stat_mmd_pval"] for i in valid_mmd_idx], dtype=float))
            for idx, qv in zip(valid_mmd_idx, qvals):
                rows[idx]["stat_mmd_qval"] = float(qv)
        if valid_energy_idx:
            qvals = benjamini_hochberg(np.asarray([rows[i]["stat_energy_pval"] for i in valid_energy_idx], dtype=float))
            for idx, qv in zip(valid_energy_idx, qvals):
                rows[idx]["stat_energy_qval"] = float(qv)
    return rows


def residual_signature_by_support_bin(
    *,
    X_real: np.ndarray,
    Y_real: np.ndarray,
    X_pred: np.ndarray,
    Y_pred: np.ndarray,
    a_train: float,
    regime_id: str,
    regime_label: str = "",
    study: str = "",
    run_id: str = "",
    run_dir: str = "",
    model_run_dir: str = "",
    best_grid_tag: str = "",
    dist_target_m: float = float("nan"),
    curr_target_mA: float = float("nan"),
    support_bins: int = 4,
    min_samples_per_bin: int = 512,
    stat_mode: str = "quick",
    stat_n_perm: int = 200,
    stat_seed: int = 42,
    stat_execution_backend: str = "cpu",
) -> List[Dict[str, Any]]:
    """Return support-conditioned residual signature rows for one regime."""
    from src.evaluation.stat_tests import benjamini_hochberg, energy_test, mmd_rbf
    from src.metrics.distribution import residual_fidelity_metrics

    Xr = np.asarray(X_real, dtype=np.float64)
    Yr = np.asarray(Y_real, dtype=np.float64)
    Xp = np.asarray(X_pred, dtype=np.float64)
    Yp = np.asarray(Y_pred, dtype=np.float64)
    if len(Xr) == 0 or len(Xp) == 0:
        return []

    feats_real = support_feature_dict(Xr, a_train=float(a_train))
    feats_pred = support_feature_dict(Xp, a_train=float(a_train))
    region_real = support_region_labels(Xr, a_train=float(a_train))
    region_pred = support_region_labels(Xp, a_train=float(a_train))

    rows: List[Dict[str, Any]] = []
    valid_mmd_idx: List[int] = []
    valid_energy_idx: List[int] = []

    def _append_row(
        *,
        axis_name: str,
        bin_index: int,
        bin_label: str,
        lo: float,
        hi: float,
        mask_real: np.ndarray,
        mask_pred: np.ndarray,
        region_label: str = "",
    ) -> None:
        row: Dict[str, Any] = {
            "study": study,
            "regime_id": regime_id,
            "regime_label": regime_label,
            "run_id": run_id,
            "run_dir": run_dir,
            "model_run_dir": model_run_dir,
            "best_grid_tag": best_grid_tag,
            "dist_target_m": float(dist_target_m),
            "curr_target_mA": float(curr_target_mA),
            "support_axis": axis_name,
            "support_bin_index": int(bin_index),
            "support_bin_label": str(bin_label),
            "support_lo": float(lo),
            "support_hi": float(hi),
            "support_region": str(region_label),
            "n_samples_real": int(mask_real.sum()),
            "n_samples_pred": int(mask_pred.sum()),
            "stat_mode": stat_mode,
        }
        if int(mask_real.sum()) < int(min_samples_per_bin) or int(mask_pred.sum()) < int(min_samples_per_bin):
            row.update({
                "std_real_delta_I": float("nan"),
                "std_real_delta_Q": float("nan"),
                "std_pred_delta_I": float("nan"),
                "std_pred_delta_Q": float("nan"),
                "delta_wasserstein_I": float("nan"),
                "delta_wasserstein_Q": float("nan"),
                "delta_jb_stat_rel_I": float("nan"),
                "delta_jb_stat_rel_Q": float("nan"),
                "stat_mmd_pval": float("nan"),
                "stat_mmd_qval": float("nan"),
                "stat_energy_pval": float("nan"),
                "stat_energy_qval": float("nan"),
            })
            rows.append(row)
            return

        rr = Yr[mask_real] - Xr[mask_real]
        rp = Yp[mask_pred] - Xp[mask_pred]
        sig = residual_fidelity_metrics(
            rr,
            rp,
            max_samples=min(len(rr), len(rp)),
            X=Xr[mask_real],
            X_pred=Xp[mask_pred],
        )
        row.update({
            "std_real_delta_I": float(sig.get("std_real_delta_I", float("nan"))),
            "std_real_delta_Q": float(sig.get("std_real_delta_Q", float("nan"))),
            "std_pred_delta_I": float(sig.get("std_pred_delta_I", float("nan"))),
            "std_pred_delta_Q": float(sig.get("std_pred_delta_Q", float("nan"))),
            "delta_wasserstein_I": float(sig.get("delta_wasserstein_I", float("nan"))),
            "delta_wasserstein_Q": float(sig.get("delta_wasserstein_Q", float("nan"))),
            "delta_jb_stat_rel_I": float(sig.get("delta_jb_stat_rel_I", float("nan"))),
            "delta_jb_stat_rel_Q": float(sig.get("delta_jb_stat_rel_Q", float("nan"))),
        })
        n_cmp = min(len(rr), len(rp))
        rng = np.random.RandomState(int(stat_seed) + len(rows))
        if n_cmp < len(rr):
            rr = rr[rng.choice(len(rr), n_cmp, replace=False)]
        if n_cmp < len(rp):
            rp = rp[rng.choice(len(rp), n_cmp, replace=False)]
        try:
            sf_mmd = mmd_rbf(
                rr,
                rp,
                n_perm=int(stat_n_perm),
                seed=int(stat_seed) + len(rows),
                execution_backend=stat_execution_backend,
            )
            sf_energy = energy_test(
                rr,
                rp,
                n_perm=int(stat_n_perm),
                seed=int(stat_seed) + len(rows),
                execution_backend=stat_execution_backend,
            )
            row["stat_mmd_pval"] = float(sf_mmd["pval"])
            row["stat_mmd_qval"] = float("nan")
            row["stat_energy_pval"] = float(sf_energy["pval"])
            row["stat_energy_qval"] = float("nan")
            rows.append(row)
            valid_mmd_idx.append(len(rows) - 1)
            valid_energy_idx.append(len(rows) - 1)
        except Exception as exc:
            print(
                "⚠️  Residual signature support-bin stat tests failed "
                f"for regime '{regime_id}' axis='{axis_name}' bin={bin_index}: {exc}"
            )
            row["stat_mmd_pval"] = float("nan")
            row["stat_mmd_qval"] = float("nan")
            row["stat_energy_pval"] = float("nan")
            row["stat_energy_qval"] = float("nan")
            rows.append(row)

    for axis_name in ("r_l2_norm", "r_inf_norm"):
        values_real = np.asarray(feats_real[axis_name], dtype=np.float64)
        values_pred = np.asarray(feats_pred[axis_name], dtype=np.float64)
        edges = np.quantile(
            values_real,
            np.linspace(0.0, 1.0, int(max(2, support_bins)) + 1),
        ).astype(np.float64)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-12
        for bi in range(len(edges) - 1):
            lo = float(edges[bi])
            hi = float(edges[bi + 1])
            is_last = bi == len(edges) - 2
            mask_real = (values_real >= lo) & ((values_real <= hi) if is_last else (values_real < hi))
            mask_pred = (values_pred >= lo) & ((values_pred <= hi) if is_last else (values_pred < hi))
            _append_row(
                axis_name=axis_name,
                bin_index=bi,
                bin_label=f"q{int(100 * bi / (len(edges) - 1))}-{int(100 * (bi + 1) / (len(edges) - 1))}",
                lo=lo,
                hi=hi,
                mask_real=mask_real,
                mask_pred=mask_pred,
            )

    for region_name in ("center", "edge", "corner"):
        mask_real = np.asarray(region_real == region_name, dtype=bool)
        mask_pred = np.asarray(region_pred == region_name, dtype=bool)
        _append_row(
            axis_name="support_region",
            bin_index={"center": 0, "edge": 1, "corner": 2}[region_name],
            bin_label=region_name,
            lo=float("nan"),
            hi=float("nan"),
            mask_real=mask_real,
            mask_pred=mask_pred,
            region_label=region_name,
        )

    if valid_mmd_idx:
        qvals = benjamini_hochberg(
            np.asarray([rows[i]["stat_mmd_pval"] for i in valid_mmd_idx], dtype=float)
        )
        for idx, qv in zip(valid_mmd_idx, qvals):
            rows[idx]["stat_mmd_qval"] = float(qv)
    if valid_energy_idx:
        qvals = benjamini_hochberg(
            np.asarray([rows[i]["stat_energy_pval"] for i in valid_energy_idx], dtype=float)
        )
        for idx, qv in zip(valid_energy_idx, qvals):
            rows[idx]["stat_energy_qval"] = float(qv)
    return rows


# ==========================================================================
# Residual distribution comparison
# ==========================================================================
def residual_distribution_metrics(
    X: np.ndarray,
    Y: np.ndarray,
    Yp: np.ndarray,
    psd_nfft: int = 2048,
    gauss_alpha: float = 0.01,
    acf_max_lag: int = 128,
    Y_samples: Optional[np.ndarray] = None,
    coverage_target: Optional[np.ndarray] = None,
):
    X_arr  = np.asarray(X)
    d_real = np.asarray(Y) - X_arr
    d_pred = np.asarray(Yp) - X_arr

    # Heteroscedasticity: Pearson(|X_c|, |δ_c|²).
    # rho > 0 → noise power grows with input amplitude (typical IM/DD shot noise).
    # rho ≈ 0 → noise power is independent of drive level (AWGN regime).
    def _rho_hetero(x_arr: np.ndarray, d_arr: np.ndarray) -> float:
        xc = x_arr[:, 0] + 1j * x_arr[:, 1]
        dc = d_arr[:, 0] + 1j * d_arr[:, 1]
        amp_x  = np.abs(xc)
        amp_d2 = np.abs(dc) ** 2
        if np.std(amp_x) < 1e-12 or np.std(amp_d2) < 1e-12:
            return float("nan")
        return float(np.corrcoef(amp_x, amp_d2)[0, 1])

    mean_l2 = float(np.linalg.norm(np.mean(d_pred, axis=0) - np.mean(d_real, axis=0)))
    cov_fro = float(np.linalg.norm(np.cov(d_pred.T) - np.cov(d_real.T), ord="fro"))

    mean_real = np.mean(d_real, axis=0)
    mean_pred = np.mean(d_pred, axis=0)
    std_real = np.std(d_real, axis=0)
    std_pred = np.std(d_pred, axis=0)

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
    acf_r = _acf_curve_complex(cr, max_lag=int(acf_max_lag))
    acf_p = _acf_curve_complex(cp, max_lag=int(acf_max_lag))
    acf_l2 = float(np.linalg.norm(acf_p - acf_r) / np.sqrt(len(acf_r) if len(acf_r) else 1))

    try:
        from src.evaluation.stat_tests.jsd import jsd_2d
    except ImportError:
        jsd_2d = None  # type: ignore[assignment]
    stat_jsd = jsd_2d(d_real, d_pred) if jsd_2d is not None else float("nan")

    out = {
        "delta_mean_l2": mean_l2,
        "delta_cov_fro": cov_fro,
        "mean_real_delta_I": float(mean_real[0]),
        "mean_real_delta_Q": float(mean_real[1]),
        "mean_pred_delta_I": float(mean_pred[0]),
        "mean_pred_delta_Q": float(mean_pred[1]),
        "std_real_delta_I": float(std_real[0]),
        "std_real_delta_Q": float(std_real[1]),
        "std_pred_delta_I": float(std_pred[0]),
        "std_pred_delta_Q": float(std_pred[1]),
        "delta_mean_I": float(mean_pred[0] - mean_real[0]),
        "delta_mean_Q": float(mean_pred[1] - mean_real[1]),
        "delta_std_I": float(std_pred[0] - std_real[0]),
        "delta_std_Q": float(std_pred[1] - std_real[1]),
        "var_real_delta": var_real,
        "var_pred_delta": var_pred,
        "delta_skew_l2": skew_l2,
        "delta_kurt_l2": kurt_l2,
        "delta_skew_I": float(skew_p[0] - skew_r[0]),
        "delta_skew_Q": float(skew_p[1] - skew_r[1]),
        "delta_kurt_I": float(kurt_p[0] - kurt_r[0]),
        "delta_kurt_Q": float(kurt_p[1] - kurt_r[1]),
        "delta_wasserstein_I": _wasserstein_1d(d_real[:, 0], d_pred[:, 0]),
        "delta_wasserstein_Q": _wasserstein_1d(d_real[:, 1], d_pred[:, 1]),
        "delta_psd_l2": psd_l2,
        "delta_acf_l2": acf_l2,
        "rho_hetero_real": _rho_hetero(X_arr, d_real),
        "rho_hetero_pred": _rho_hetero(X_arr, d_pred),
        "stat_jsd": stat_jsd,
    }
    out.update(
        residual_stat_signature(
            X_arr,
            Y,
            Yp,
            Y_samples=Y_samples,
            coverage_target=coverage_target,
        )
    )

    # Keep JB fields aligned with src.metrics.distribution (auditability / underflow-safe log10(p)).
    try:
        from src.metrics.distribution import gaussianity_tests

        g_real = gaussianity_tests(d_real, alpha=float(gauss_alpha))
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
            "jb_real_stat_I": float(g_real.get("jb_stat_I")),
            "jb_real_stat_Q": float(g_real.get("jb_stat_Q")),
            "jb_real_p_I": float(g_real.get("jb_p_I")),
            "jb_real_p_Q": float(g_real.get("jb_p_Q")),
            "jb_real_p_min": float(g_real.get("jb_p_min")),
            "jb_real_log10p_I": float(g_real.get("jb_log10p_I")),
            "jb_real_log10p_Q": float(g_real.get("jb_log10p_Q")),
            "jb_real_log10p_min": float(g_real.get("jb_log10p_min")),
            "jb_real_reject_gaussian": bool(g_real.get("reject_gaussian", False)),
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
            "jb_real_stat_I": float("nan"),
            "jb_real_stat_Q": float("nan"),
            "jb_real_p_I": float("nan"),
            "jb_real_p_Q": float("nan"),
            "jb_real_p_min": float("nan"),
            "jb_real_log10p_I": float("nan"),
            "jb_real_log10p_Q": float("nan"),
            "jb_real_log10p_min": float("nan"),
            "jb_real_reject_gaussian": False,
        })

    for axis in ("I", "Q"):
        pred = out.get(f"jb_log10p_{axis}", float("nan"))
        real = out.get(f"jb_real_log10p_{axis}", float("nan"))
        pred_f = float(pred) if pred is not None else float("nan")
        real_f = float(real) if real is not None else float("nan")
        if np.isfinite(pred_f) and np.isfinite(real_f):
            delta = abs(pred_f - real_f)
            rel = delta / abs(real_f) if abs(real_f) > 0 else float("nan")
        else:
            delta = float("nan")
            rel = float("nan")
        out[f"delta_jb_log10p_{axis}"] = float(delta)
        out[f"delta_jb_stat_rel_{axis}"] = float(rel)
    return out
