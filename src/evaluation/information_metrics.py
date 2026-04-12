# -*- coding: utf-8 -*-
"""Auxiliary information-theoretic metrics for discrete input alphabets.

These metrics are intentionally auxiliary: they do not drive the twin gates.

Current scope
-------------
- Detect discrete transmitted alphabets from repeated IQ symbols.
- Estimate a shared-covariance Gaussian auxiliary channel q(y|x).
- Report symbol-metric MI / AIR and bit-metric GMI / NGMI when a valid
  Gray rectangular QAM labeling can be inferred.

If the input support looks continuous (for example the dense full-square
excitation), the functions below return ``available=False`` and fill the
numeric metrics with ``NaN``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


_LOG2 = math.log(2.0)


@dataclass
class _AlphabetSpec:
    symbols: np.ndarray
    inverse: np.ndarray
    probabilities: np.ndarray
    entropy_bits: float
    avg_repeats: float
    labels: Optional[np.ndarray]
    labeling_mode: str


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _gray_bits(index: int, n_bits: int) -> np.ndarray:
    gray = int(index) ^ (int(index) >> 1)
    return np.asarray(
        [int((gray >> shift) & 1) for shift in range(n_bits - 1, -1, -1)],
        dtype=np.int8,
    )


def _round_rows(x: np.ndarray, decimals: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) IQ array, got shape={arr.shape}")
    return np.round(arr, decimals=int(decimals))


def _build_alphabet_spec(
    x: np.ndarray,
    *,
    decimals: int = 6,
    max_alphabet: int = 4096,
    min_avg_repeats: float = 8.0,
) -> tuple[Optional[_AlphabetSpec], str]:
    xr = _round_rows(x, decimals)
    symbols, inverse, counts = np.unique(
        xr,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    m = int(len(symbols))
    if m < 2:
        return None, "alphabet_too_small"
    if m > int(max_alphabet):
        return None, "alphabet_too_large"

    avg_repeats = float(len(xr) / m)
    if avg_repeats < float(min_avg_repeats):
        return None, "alphabet_not_repeated_enough"

    probs = counts.astype(np.float64) / float(np.sum(counts))
    entropy_bits = float(-np.sum(probs * np.log2(np.clip(probs, 1e-12, None))))

    labels = _infer_gray_rectangular_labels(symbols)
    labeling_mode = "gray_rect_qam_inferred" if labels is not None else "unavailable"

    return (
        _AlphabetSpec(
            symbols=symbols.astype(np.float64),
            inverse=inverse.astype(np.int64),
            probabilities=probs.astype(np.float64),
            entropy_bits=entropy_bits,
            avg_repeats=avg_repeats,
            labels=labels,
            labeling_mode=labeling_mode,
        ),
        "ok",
    )


def _infer_gray_rectangular_labels(symbols: np.ndarray) -> Optional[np.ndarray]:
    pts = np.asarray(symbols, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None

    i_levels = np.unique(pts[:, 0])
    q_levels = np.unique(pts[:, 1])
    n_i = int(len(i_levels))
    n_q = int(len(q_levels))
    m = int(len(pts))
    if n_i * n_q != m:
        return None
    if not (_is_power_of_two(n_i) and _is_power_of_two(n_q)):
        return None

    point_to_index = {tuple(np.asarray(row, dtype=float)): idx for idx, row in enumerate(pts)}
    labels = np.zeros((m, int(round(math.log2(m)))), dtype=np.int8)
    i_bits = int(round(math.log2(n_i)))
    q_bits = int(round(math.log2(n_q)))

    for ii, i_val in enumerate(np.sort(i_levels)):
        bits_i = _gray_bits(ii, i_bits)
        for qi, q_val in enumerate(np.sort(q_levels)):
            idx = point_to_index.get((float(i_val), float(q_val)))
            if idx is None:
                return None
            bits_q = _gray_bits(qi, q_bits)
            labels[idx, :] = np.concatenate([bits_i, bits_q], axis=0)
    return labels


def _is_near_uniform_symbol_prior(
    probs: np.ndarray,
    *,
    rel_tol: float = 0.10,
) -> bool:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or len(p) == 0:
        return False
    ideal = 1.0 / float(len(p))
    max_abs = float(np.max(np.abs(p - ideal)))
    return bool(max_abs <= float(rel_tol) * ideal)


def _map_symbol_indices(
    x: np.ndarray,
    symbols: np.ndarray,
    *,
    decimals: int = 6,
) -> Optional[np.ndarray]:
    xr = _round_rows(x, decimals)
    symbol_map = {tuple(np.asarray(sym, dtype=float)): idx for idx, sym in enumerate(symbols)}
    out = np.empty(len(xr), dtype=np.int64)
    for idx, row in enumerate(xr):
        key = tuple(np.asarray(row, dtype=float))
        if key not in symbol_map:
            return None
        out[idx] = int(symbol_map[key])
    return out


def _subsample_pairs(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n = int(min(len(x_arr), len(y_arr)))
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    if n <= int(max_samples):
        return x_arr, y_arr
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(n, size=int(max_samples), replace=False))
    return x_arr[idx], y_arr[idx]


def _fit_shared_gaussian_aux(
    x: np.ndarray,
    y: np.ndarray,
    *,
    covariance_reg: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    residual = np.asarray(y, dtype=np.float64) - np.asarray(x, dtype=np.float64)
    mu = np.mean(residual, axis=0)
    centered = residual - mu
    cov = np.cov(centered.T, bias=True)
    if np.ndim(cov) == 0:
        cov = np.eye(2, dtype=np.float64) * float(cov)
    cov = np.asarray(cov, dtype=np.float64).reshape(2, 2)
    reg_scale = max(float(np.trace(cov) / 2.0), 1.0)
    cov = cov + float(covariance_reg) * reg_scale * np.eye(2, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0 or not np.isfinite(logdet):
        raise ValueError("auxiliary covariance is not positive definite")
    inv_cov = np.linalg.inv(cov)
    return mu.astype(np.float64), inv_cov.astype(np.float64), float(logdet)


def _logsumexp(arr: np.ndarray, axis: int = 1) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    amax = np.max(arr, axis=axis, keepdims=True)
    stable = arr - amax
    return np.squeeze(amax, axis=axis) + np.log(np.sum(np.exp(stable), axis=axis))


def _log_q_matrix(
    y: np.ndarray,
    symbols: np.ndarray,
    *,
    mu: np.ndarray,
    inv_cov: np.ndarray,
    logdet: float,
    chunk_size: int = 4096,
) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float64)
    means = np.asarray(symbols, dtype=np.float64) + np.asarray(mu, dtype=np.float64)[None, :]
    n = int(len(y_arr))
    m = int(len(means))
    out = np.empty((n, m), dtype=np.float64)
    const = -0.5 * (2.0 * math.log(2.0 * math.pi) + float(logdet))
    for start in range(0, n, int(chunk_size)):
        stop = min(n, start + int(chunk_size))
        diff = y_arr[start:stop, None, :] - means[None, :, :]
        maha = np.einsum("nmd,df,nmf->nm", diff, inv_cov, diff)
        out[start:stop, :] = const - 0.5 * maha
    return out


def _estimate_mi_bits(
    tx_idx: np.ndarray,
    y: np.ndarray,
    symbols: np.ndarray,
    probs: np.ndarray,
    *,
    mu: np.ndarray,
    inv_cov: np.ndarray,
    logdet: float,
) -> float:
    log_q = _log_q_matrix(y, symbols, mu=mu, inv_cov=inv_cov, logdet=logdet)
    log_p = np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-15, None))
    log_denom = _logsumexp(log_q + log_p[None, :], axis=1)
    log_num = log_q[np.arange(len(tx_idx), dtype=np.int64), np.asarray(tx_idx, dtype=np.int64)]
    value = float(np.mean((log_num - log_denom) / _LOG2))
    return value


def _estimate_gmi_bits(
    tx_idx: np.ndarray,
    y: np.ndarray,
    symbols: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    entropy_bits: float,
    *,
    mu: np.ndarray,
    inv_cov: np.ndarray,
    logdet: float,
) -> float:
    log_q = _log_q_matrix(y, symbols, mu=mu, inv_cov=inv_cov, logdet=logdet)
    log_p = np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-15, None))
    log_denom = _logsumexp(log_q + log_p[None, :], axis=1)

    labels_arr = np.asarray(labels, dtype=np.int8)
    tx_bits = labels_arr[np.asarray(tx_idx, dtype=np.int64)]
    bit_terms = np.zeros(len(tx_idx), dtype=np.float64)

    for bit_pos in range(labels_arr.shape[1]):
        mask0 = labels_arr[:, bit_pos] == 0
        mask1 = ~mask0
        log_num0 = _logsumexp(log_q[:, mask0] + log_p[None, mask0], axis=1)
        log_num1 = _logsumexp(log_q[:, mask1] + log_p[None, mask1], axis=1)
        chosen = np.where(tx_bits[:, bit_pos] == 0, log_num0, log_num1)
        bit_terms += (chosen - log_denom) / _LOG2

    return float(entropy_bits + np.mean(bit_terms))


def _nan_payload(*, status: str, available: bool = False) -> Dict[str, Any]:
    return {
        "info_metrics_available": bool(available),
        "info_metrics_status": str(status),
        "info_alphabet_size": float("nan"),
        "info_bits_per_symbol": float("nan"),
        "info_input_entropy_bits": float("nan"),
        "info_avg_symbol_repeats": float("nan"),
        "info_labeling_mode": "unavailable",
        "mi_aux_real_bits": float("nan"),
        "mi_aux_pred_bits": float("nan"),
        "mi_aux_gap_bits": float("nan"),
        "mi_aux_gap_rel": float("nan"),
        "gmi_aux_real_bits": float("nan"),
        "gmi_aux_pred_bits": float("nan"),
        "gmi_aux_gap_bits": float("nan"),
        "gmi_aux_gap_rel": float("nan"),
        "ngmi_aux_real": float("nan"),
        "ngmi_aux_pred": float("nan"),
        "ngmi_aux_gap": float("nan"),
        "air_aux_real_bits": float("nan"),
        "air_aux_pred_bits": float("nan"),
        "air_aux_gap_bits": float("nan"),
        "air_aux_gap_rel": float("nan"),
        "info_aux_channel_mode": "unavailable",
    }


def auxiliary_information_metrics(
    *,
    X_real: np.ndarray,
    Y_real: np.ndarray,
    X_pred: np.ndarray,
    Y_pred: np.ndarray,
    seed: int = 42,
    max_samples: int = 50_000,
    symbol_round_decimals: int = 6,
    max_alphabet: int = 4096,
    min_avg_repeats: float = 8.0,
    covariance_reg: float = 1e-6,
) -> Dict[str, Any]:
    """Estimate auxiliary MI/GMI/NGMI/AIR for discrete transmitted alphabets.

    Returns NaNs when the input alphabet is not discrete/repeated enough.
    """
    base = _nan_payload(status="unavailable")
    try:
        spec, status = _build_alphabet_spec(
            X_real,
            decimals=int(symbol_round_decimals),
            max_alphabet=int(max_alphabet),
            min_avg_repeats=float(min_avg_repeats),
        )
    except Exception as exc:
        out = _nan_payload(status=f"error:{type(exc).__name__}")
        out["info_metrics_error"] = str(exc)
        return out

    if spec is None:
        return _nan_payload(status=status)

    x_real_sub, y_real_sub = _subsample_pairs(
        X_real,
        Y_real,
        max_samples=int(max_samples),
        seed=int(seed),
    )
    x_pred_sub, y_pred_sub = _subsample_pairs(
        X_pred,
        Y_pred,
        max_samples=int(max_samples),
        seed=int(seed) + 1,
    )

    tx_idx_real = _map_symbol_indices(
        x_real_sub,
        spec.symbols,
        decimals=int(symbol_round_decimals),
    )
    tx_idx_pred = _map_symbol_indices(
        x_pred_sub,
        spec.symbols,
        decimals=int(symbol_round_decimals),
    )
    if tx_idx_real is None or tx_idx_pred is None:
        return _nan_payload(status="symbol_mapping_failed")

    try:
        mu_real, inv_cov_real, logdet_real = _fit_shared_gaussian_aux(
            x_real_sub,
            y_real_sub,
            covariance_reg=float(covariance_reg),
        )
        mu_pred, inv_cov_pred, logdet_pred = _fit_shared_gaussian_aux(
            x_pred_sub,
            y_pred_sub,
            covariance_reg=float(covariance_reg),
        )
    except Exception as exc:
        out = _nan_payload(status=f"error:{type(exc).__name__}")
        out["info_metrics_error"] = str(exc)
        return out

    h_bits = float(spec.entropy_bits)
    mi_real = _estimate_mi_bits(
        tx_idx_real,
        y_real_sub,
        spec.symbols,
        spec.probabilities,
        mu=mu_real,
        inv_cov=inv_cov_real,
        logdet=logdet_real,
    )
    mi_pred = _estimate_mi_bits(
        tx_idx_pred,
        y_pred_sub,
        spec.symbols,
        spec.probabilities,
        mu=mu_pred,
        inv_cov=inv_cov_pred,
        logdet=logdet_pred,
    )
    mi_real = float(np.clip(mi_real, 0.0, h_bits))
    mi_pred = float(np.clip(mi_pred, 0.0, h_bits))

    bits_per_symbol = int(round(math.log2(len(spec.symbols)))) if _is_power_of_two(len(spec.symbols)) else None
    gmi_real = float("nan")
    gmi_pred = float("nan")
    ngmi_real = float("nan")
    ngmi_pred = float("nan")
    labeling_mode = str(spec.labeling_mode)
    uniform_prior = _is_near_uniform_symbol_prior(spec.probabilities)
    if spec.labels is not None and bits_per_symbol is not None and uniform_prior:
        gmi_real = _estimate_gmi_bits(
            tx_idx_real,
            y_real_sub,
            spec.symbols,
            spec.probabilities,
            spec.labels,
            h_bits,
            mu=mu_real,
            inv_cov=inv_cov_real,
            logdet=logdet_real,
        )
        gmi_pred = _estimate_gmi_bits(
            tx_idx_pred,
            y_pred_sub,
            spec.symbols,
            spec.probabilities,
            spec.labels,
            h_bits,
            mu=mu_pred,
            inv_cov=inv_cov_pred,
            logdet=logdet_pred,
        )
        gmi_real = float(np.clip(gmi_real, 0.0, h_bits))
        gmi_pred = float(np.clip(gmi_pred, 0.0, h_bits))
        denom = float(bits_per_symbol)
        if denom > 0:
            ngmi_real = float(np.clip(1.0 - (h_bits - gmi_real) / denom, 0.0, 1.0))
            ngmi_pred = float(np.clip(1.0 - (h_bits - gmi_pred) / denom, 0.0, 1.0))
        labeling_mode = "gray_rect_qam_uniform"
    elif spec.labels is not None and bits_per_symbol is not None:
        labeling_mode = "gray_rect_qam_nonuniform_prior"

    def _gap_rel(a: float, b: float) -> float:
        if not np.isfinite(a) or not np.isfinite(b):
            return float("nan")
        den = max(abs(a), 1e-12)
        return float(abs(b - a) / den)

    out = {
        "info_metrics_available": True,
        "info_metrics_status": "ok",
        "info_alphabet_size": int(len(spec.symbols)),
        "info_bits_per_symbol": (int(bits_per_symbol) if bits_per_symbol is not None else float("nan")),
        "info_input_entropy_bits": h_bits,
        "info_avg_symbol_repeats": float(spec.avg_repeats),
        "info_labeling_mode": labeling_mode,
        "mi_aux_real_bits": mi_real,
        "mi_aux_pred_bits": mi_pred,
        "mi_aux_gap_bits": float(mi_pred - mi_real),
        "mi_aux_gap_rel": _gap_rel(mi_real, mi_pred),
        "gmi_aux_real_bits": gmi_real,
        "gmi_aux_pred_bits": gmi_pred,
        "gmi_aux_gap_bits": (
            float(gmi_pred - gmi_real) if np.isfinite(gmi_real) and np.isfinite(gmi_pred) else float("nan")
        ),
        "gmi_aux_gap_rel": _gap_rel(gmi_real, gmi_pred),
        "ngmi_aux_real": ngmi_real,
        "ngmi_aux_pred": ngmi_pred,
        "ngmi_aux_gap": (
            float(ngmi_pred - ngmi_real) if np.isfinite(ngmi_real) and np.isfinite(ngmi_pred) else float("nan")
        ),
        # AIR here is the symbol-metric auxiliary-channel AIR, i.e. the MI-style bound.
        "air_aux_real_bits": mi_real,
        "air_aux_pred_bits": mi_pred,
        "air_aux_gap_bits": float(mi_pred - mi_real),
        "air_aux_gap_rel": _gap_rel(mi_real, mi_pred),
        "info_aux_channel_mode": "gaussian_shared_covariance",
    }
    return out
