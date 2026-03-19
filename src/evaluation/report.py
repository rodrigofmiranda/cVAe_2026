# -*- coding: utf-8 -*-
"""
src.evaluation.report — Evaluation reporting / table helpers.

Reusable helpers for the canonical evaluation engine.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Global metrics assembly
# ---------------------------------------------------------------------------

def build_global_metrics(
    *,
    run_id: str,
    model_path: str,
    split_mode: str,
    N_eval: int,
    evm_real: float,
    evm_pred: float,
    snr_real: float,
    snr_pred: float,
    distm: Dict[str, float],
    det_inf: bool,
    rank_mode: str,
    mc_samples: int,
    var_mc: float,
    arch_variant: str | None = None,
    latent_prior_semantics: str | None = None,
) -> Dict[str, Any]:
    """Assemble the global-metrics dictionary (identical to monolith)."""
    distm_serialized = {}
    for key, value in distm.items():
        if isinstance(value, (bool, np.bool_)):
            distm_serialized[key] = bool(value)
        else:
            distm_serialized[key] = float(value)

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "model_path": model_path,
        "split_mode": split_mode,
        "N_eval": int(N_eval),
        "evm_real_%": float(evm_real),
        "evm_pred_%": float(evm_pred),
        "delta_evm_%": float(evm_pred - evm_real),
        "snr_real_db": float(snr_real),
        "snr_pred_db": float(snr_pred),
        "delta_snr_db": float(snr_pred - snr_real),
        **distm_serialized,
        "deterministic_inference": bool(det_inf),
        "rank_mode": str(rank_mode),
        "mc_samples": int(mc_samples),
        "var_mc_gen": (float(var_mc) if not np.isnan(var_mc) else float("nan")),
    }
    if arch_variant is not None:
        metrics["arch_variant"] = str(arch_variant)
    if latent_prior_semantics is not None:
        metrics["latent_prior_semantics"] = str(latent_prior_semantics)
    return metrics


# ---------------------------------------------------------------------------
# Latent diagnostics
# ---------------------------------------------------------------------------

def compute_latent_diagnostics(
    z_mean_q: np.ndarray,
    z_log_var_q: np.ndarray,
    z_mean_p: np.ndarray,
    z_log_var_p: np.ndarray,
    *,
    arch_variant: str = "concat",
) -> dict:
    """Return a dict with ``df_lat``, ``lat_summary``, ``z_std_p``, and KL arrays.

    Keys returned
    -------------
    df_lat : pd.DataFrame
        Per-dimension table (dim, std_mu_p, kl_q_to_p_dim_mean, kl_p_to_N0I_dim_mean).
    lat_summary : dict
        ``active_dims_std_mu_p_gt_0p05``, ``kl_q_to_p_total_mean``, ``kl_p_to_N0I_total_mean``.
    z_std_p : np.ndarray
        ``std(z_mean_p, axis=0)``
    active_dims : int
    kl_qp_dim_mean, kl_pN_dim_mean : np.ndarray  (per-dim means)
    kl_qp_total_mean, kl_pN_total_mean : float
    """
    z_std_p = np.std(z_mean_p, axis=0)
    active_dims = int(np.sum(z_std_p > 0.05))
    is_legacy_std_normal = (
        str(arch_variant or "").strip().lower() == "legacy_2025_zero_y"
    )

    vq = np.exp(np.clip(z_log_var_q, -20, 20))
    vp = np.exp(np.clip(z_log_var_p, -20, 20))

    if is_legacy_std_normal:
        kl_qp_dim_mean = np.full(z_std_p.shape, np.nan, dtype=float)
        kl_qp_total_mean = float("nan")
    else:
        kl_qp_dim = 0.5 * (
            np.log(vp + 1e-12)
            - np.log(vq + 1e-12)
            + (vq + (z_mean_q - z_mean_p) ** 2) / (vp + 1e-12)
            - 1.0
        )
        kl_qp_dim_mean = np.mean(kl_qp_dim, axis=0)
        kl_qp_total_mean = float(np.mean(np.sum(kl_qp_dim, axis=1)))

    lv_p_clip = np.clip(z_log_var_p, -20, 20)
    kl_pN_dim = 0.5 * (np.exp(lv_p_clip) + z_mean_p ** 2 - 1.0 - lv_p_clip)
    kl_pN_dim_mean = np.mean(kl_pN_dim, axis=0)
    kl_pN_total_mean = float(np.mean(np.sum(kl_pN_dim, axis=1)))

    df_lat = pd.DataFrame({
        "dim": np.arange(z_std_p.shape[0]),
        "std_mu_p": z_std_p.astype(float),
        "kl_q_to_p_dim_mean": kl_qp_dim_mean.astype(float),
        "kl_p_to_N0I_dim_mean": kl_pN_dim_mean.astype(float),
    })

    lat_summary = {
        "active_dims_std_mu_p_gt_0p05": int(active_dims),
        "kl_q_to_p_total_mean": float(kl_qp_total_mean),
        "kl_p_to_N0I_total_mean": float(kl_pN_total_mean),
        "kl_q_to_p_applicable": not is_legacy_std_normal,
        "latent_prior_semantics": (
            "std_normal_legacy_2025_zero_y"
            if is_legacy_std_normal
            else "conditional_prior"
        ),
    }

    return {
        "df_lat": df_lat,
        "lat_summary": lat_summary,
        "z_std_p": z_std_p,
        "active_dims": active_dims,
        "kl_qp_dim_mean": kl_qp_dim_mean,
        "kl_pN_dim_mean": kl_pN_dim_mean,
        "kl_qp_total_mean": kl_qp_total_mean,
        "kl_pN_total_mean": kl_pN_total_mean,
    }


# ---------------------------------------------------------------------------
# Decoder sensitivity
# ---------------------------------------------------------------------------

def decoder_sensitivity(
    prior_net,
    decoder_net,
    Xb: np.ndarray,
    Db: np.ndarray,
    Cb: np.ndarray,
    n_mc_z: int = 16,
    batch_size: int = 4096,
    arch_variant: str = "concat",
) -> Dict[str, float]:
    """Compute decoder sensitivity to z sampling.

    Returns ``{"decoder_output_variance_mean": …, "decoder_output_rms_std": …}``.
    """
    mu_p, lv_p = prior_net.predict([Xb, Db, Cb], batch_size=batch_size, verbose=0)
    lv_p = np.clip(lv_p, -10, 10)
    std_p = np.exp(0.5 * lv_p)

    cond = np.concatenate([Xb, Db, Cb], axis=1)

    outs = []
    is_delta_residual = (
        str(arch_variant or "").strip().lower() == "delta_residual"
    )
    for _ in range(int(n_mc_z)):
        eps = np.random.randn(*mu_p.shape).astype(np.float32)
        z = mu_p + std_p * eps
        out_params = decoder_net.predict([z, cond], batch_size=batch_size, verbose=0)
        y_mean = out_params[:, :2] + Xb if is_delta_residual else out_params[:, :2]
        outs.append(y_mean)

    outs = np.stack(outs, axis=0)  # [K,N,2]
    v = np.var(outs, axis=0)       # [N,2]
    return {
        "decoder_output_variance_mean": float(np.mean(v)),
        "decoder_output_rms_std": float(np.mean(np.sqrt(np.sum(v, axis=1)))),
    }


# ---------------------------------------------------------------------------
# History loader helper
# ---------------------------------------------------------------------------

def load_training_history(path: Path) -> Optional[pd.DataFrame]:
    """Load training_history.json and return a DataFrame or *None*."""
    if path is None or not Path(path).exists():
        return None
    try:
        hist = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(hist, dict) and "history" in hist and isinstance(hist["history"], dict):
            df = pd.DataFrame(hist["history"])
        elif isinstance(hist, dict) and "loss" in hist:
            df = pd.DataFrame(hist)
        elif isinstance(hist, list):
            df = pd.DataFrame(hist)
        else:
            return None
        return df if len(df) > 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Summary text builder (for summary_report.png)
# ---------------------------------------------------------------------------

def build_summary_text(
    *,
    run_id: str,
    split_mode: str,
    N_eval: int,
    evm_real: float,
    evm_pred: float,
    snr_real: float,
    snr_pred: float,
    distm: Dict[str, float],
    active_dims: int,
    kl_qp_total_mean: float,
    kl_pN_total_mean: float,
    sens_var_mean: float,
    sens_rms: float,
    arch_variant: str = "concat",
) -> str:
    """Build the summary text identical to the monolith's section 10.7."""
    is_legacy_std_normal = (
        str(arch_variant or "").strip().lower() == "legacy_2025_zero_y"
    )

    def _fmt_metric(value: float) -> str:
        try:
            return "n/a" if np.isnan(value) else f"{value:.4g}"
        except TypeError:
            return str(value)

    prior_semantics = (
        "standard-normal legacy (encoder ignores y)"
        if is_legacy_std_normal
        else "conditional prior"
    )
    return (
        f"Run: {run_id}\n"
        f"Split mode: {split_mode}\n"
        f"N_eval: {N_eval}\n"
        f"EVM real: {evm_real:.3f}% | EVM pred: {evm_pred:.3f}% | ΔEVM: {evm_pred-evm_real:+.3f} p.p.\n"
        f"SNR real: {snr_real:.3f} dB | SNR pred: {snr_pred:.3f} dB | ΔSNR: {snr_pred-snr_real:+.3f} dB\n"
        f"Δ mean L2: {distm['delta_mean_l2']:.4g} | Δ cov Fro: {distm['delta_cov_fro']:.4g} | Δ ACF L2: {distm.get('delta_acf_l2', float('nan')):.4g} | Δ PSD L2: {distm['delta_psd_l2']:.4g}\n"
        f"ρ_hetero real: {distm.get('rho_hetero_real', float('nan')):.4g} | ρ_hetero pred: {distm.get('rho_hetero_pred', float('nan')):.4g} | JSD: {distm.get('stat_jsd', float('nan')):.4g} nats\n"
        f"Latent active dims (std μ_p>0.05): {active_dims}\n"
        f"Latent prior semantics: {prior_semantics}\n"
        f"KL(q||p) total mean: {_fmt_metric(kl_qp_total_mean)} | KL(p||N) total mean: {_fmt_metric(kl_pN_total_mean)}\n"
        f"Decoder sensitivity var_mean: {sens_var_mean:.4g} | rms_std: {sens_rms:.4g}\n"
    )
