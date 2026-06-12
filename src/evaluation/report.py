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
    from src.models.losses import _mdn_expected_mean

    mu_p, lv_p = prior_net.predict([Xb, Db, Cb], batch_size=batch_size, verbose=0)
    lv_p = np.clip(lv_p, -10, 10)
    std_p = np.exp(0.5 * lv_p)
    n_decoder_inputs = len(decoder_net.inputs)
    is_seq = n_decoder_inputs == 4
    if is_seq:
        if np.asarray(Xb).ndim != 3:
            return {
                "decoder_output_variance_mean": float("nan"),
                "decoder_output_rms_std": float("nan"),
                "status": "unsupported_seq_input",
            }
        x_center = np.asarray(Xb)[:, np.asarray(Xb).shape[1] // 2, :]
        decoder_inputs = lambda z: [z, x_center, Db, Cb]
    elif n_decoder_inputs == 2:
        cond = np.concatenate([Xb, Db, Cb], axis=1)
        decoder_inputs = lambda z: [z, cond]
        x_center = np.asarray(Xb)
    else:
        return {
            "decoder_output_variance_mean": float("nan"),
            "decoder_output_rms_std": float("nan"),
            "status": "unsupported_decoder_interface",
        }

    outs = []
    is_delta_residual = (
        str(arch_variant or "").strip().lower() == "delta_residual"
    )
    for _ in range(int(n_mc_z)):
        eps = np.random.randn(*mu_p.shape).astype(np.float32)
        z = mu_p + std_p * eps
        out_params = decoder_net.predict(decoder_inputs(z), batch_size=batch_size, verbose=0)
        out_dim = int(out_params.shape[-1])
        if out_dim == 4:
            y_mean = out_params[:, :2] + x_center if is_delta_residual else out_params[:, :2]
        elif out_dim > 4 and out_dim % 5 == 0:
            k = out_dim // 5
            logits = out_params[:, :k]
            comp_mean = out_params[:, k : k + 2 * k].reshape((-1, k, 2))
            y_mean = _mdn_expected_mean(logits, comp_mean).numpy()
        else:
            return {
                "decoder_output_variance_mean": float("nan"),
                "decoder_output_rms_std": float("nan"),
                "status": "unsupported_output_params",
            }
        outs.append(y_mean)

    outs = np.stack(outs, axis=0)  # [K,N,2]
    v = np.var(outs, axis=0)       # [N,2]
    return {
        "decoder_output_variance_mean": float(np.mean(v)),
        "decoder_output_rms_std": float(np.mean(np.sqrt(np.sum(v, axis=1)))),
        "status": "ok",
    }


def mdn_decomposition_audit(
    prior_net,
    decoder_net,
    Xb: np.ndarray,
    Db: np.ndarray,
    Cb: np.ndarray,
    Yb: Optional[np.ndarray] = None,
    n_mc_z: int = 4,
    batch_size: int = 4096,
    arch_variant: str = "concat",
) -> Dict[str, Any]:
    """Audit the MDN head: mixture weights, mean separation and the residual
    variance decomposition, per axis (I, Q).

    Unlike :func:`decoder_sensitivity` (variance of the expected mean under z
    sampling), this reads the raw ``logits``/``comp_mean``/``comp_log_var``
    and decomposes the predicted residual variance as

        total ≈ E[Σ_k π_k σ_k²]  (intra-component blur)
              + E[Var_π(μ_k)]    (within-sample component separation)
              + Var(E[y|x,z]−x)  (conditional-mean structure)

    plus clamp-binding fractions against ``DECODER_LOGVAR_CLAMP_LO/HI``.
    A Gaussian head (out_dim==4) is treated as a 1-component mixture.
    """
    from src.config.defaults import (
        DECODER_LOGVAR_CLAMP_HI,
        DECODER_LOGVAR_CLAMP_LO,
    )

    mu_p, lv_p = prior_net.predict([Xb, Db, Cb], batch_size=batch_size, verbose=0)
    lv_p = np.clip(lv_p, -10, 10)
    std_p = np.exp(0.5 * lv_p)
    n_decoder_inputs = len(decoder_net.inputs)
    is_seq = n_decoder_inputs == 4
    if is_seq:
        if np.asarray(Xb).ndim != 3:
            return {"status": "unsupported_seq_input"}
        x_center = np.asarray(Xb)[:, np.asarray(Xb).shape[1] // 2, :]
        decoder_inputs = lambda z: [z, x_center, Db, Cb]
    elif n_decoder_inputs == 2:
        cond = np.concatenate([Xb, Db, Cb], axis=1)
        decoder_inputs = lambda z: [z, cond]
        x_center = np.asarray(Xb)
    else:
        return {"status": "unsupported_decoder_interface"}

    is_delta_residual = str(arch_variant or "").strip().lower() == "delta_residual"
    lo = float(DECODER_LOGVAR_CLAMP_LO)
    hi = float(DECODER_LOGVAR_CLAMP_HI)
    eps_bind = 1e-3

    intra_sum = np.zeros(2, dtype=np.float64)
    inter_sum = np.zeros(2, dtype=np.float64)
    sep_sigma_sum = np.zeros(2, dtype=np.float64)
    sigma_w_sum = np.zeros(2, dtype=np.float64)
    r_means = []
    pi_max_sum = 0.0
    pi_ent_sum = 0.0
    at_lo = 0
    at_hi = 0
    n_lv = 0
    k_out = 0

    for _ in range(int(n_mc_z)):
        eps = np.random.randn(*mu_p.shape).astype(np.float32)
        z = mu_p + std_p * eps
        out = decoder_net.predict(decoder_inputs(z), batch_size=batch_size, verbose=0)
        out_dim = int(out.shape[-1])
        if out_dim == 4:
            k = 1
            probs = np.ones((out.shape[0], 1), dtype=np.float64)
            comp_mean = out[:, :2].reshape((-1, 1, 2)).astype(np.float64)
            comp_lv = out[:, 2:].reshape((-1, 1, 2)).astype(np.float64)
        elif out_dim > 4 and out_dim % 5 == 0:
            k = out_dim // 5
            logits = out[:, :k].astype(np.float64)
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / probs.sum(axis=1, keepdims=True)
            comp_mean = out[:, k : k + 2 * k].reshape((-1, k, 2)).astype(np.float64)
            comp_lv = out[:, k + 2 * k : k + 4 * k].reshape((-1, k, 2)).astype(np.float64)
        else:
            return {"status": "unsupported_output_params"}
        k_out = k

        at_lo += int(np.sum(comp_lv <= lo + eps_bind))
        at_hi += int(np.sum(comp_lv >= hi - eps_bind))
        n_lv += int(comp_lv.size)

        var_k = np.exp(np.clip(comp_lv, lo, hi))      # (N,k,2)
        w = probs[:, :, None]                          # (N,k,1)
        intra = np.sum(w * var_k, axis=1)              # (N,2)
        cmean = np.sum(w * comp_mean, axis=1)          # (N,2)
        inter = np.maximum(
            np.sum(w * comp_mean**2, axis=1) - cmean**2, 0.0
        )

        intra_sum += intra.mean(axis=0)
        inter_sum += inter.mean(axis=0)
        sep_sigma_sum += np.mean(np.sqrt(inter) / np.sqrt(intra + 1e-12), axis=0)
        sigma_w_sum += np.sum(w * np.sqrt(var_k), axis=1).mean(axis=0)
        pi_max_sum += float(np.mean(probs.max(axis=1)))
        pi_ent_sum += float(np.mean(-np.sum(probs * np.log(probs + 1e-12), axis=1)))

        r_means.append(cmean if is_delta_residual else cmean - x_center)

    m = float(n_mc_z)
    intra_mean = intra_sum / m
    inter_mean = inter_sum / m
    struct_var = np.var(np.concatenate(r_means, axis=0), axis=0)
    total_pred = intra_mean + inter_mean + struct_var

    result: Dict[str, Any] = {
        "status": "ok",
        "mdn_components": int(k_out),
        "n_samples": int(np.asarray(Xb).shape[0]),
        "n_mc_z": int(n_mc_z),
        "clamp_lo": lo,
        "clamp_hi": hi,
        "frac_logvar_at_lo": float(at_lo / max(n_lv, 1)),
        "frac_logvar_at_hi": float(at_hi / max(n_lv, 1)),
        "pi_max_mean": float(pi_max_sum / m),
        "pi_entropy_mean": float(pi_ent_sum / m),
        "pi_eff_components": float(np.exp(pi_ent_sum / m)),
    }
    for i, ax in enumerate(("I", "Q")):
        tot = float(total_pred[i]) if total_pred[i] > 0 else float("nan")
        result[f"intra_var_{ax}"] = float(intra_mean[i])
        result[f"inter_comp_var_{ax}"] = float(inter_mean[i])
        result[f"struct_var_{ax}"] = float(struct_var[i])
        result[f"total_pred_var_{ax}"] = float(total_pred[i])
        result[f"frac_intra_{ax}"] = float(intra_mean[i] / tot)
        result[f"frac_inter_comp_{ax}"] = float(inter_mean[i] / tot)
        result[f"frac_struct_{ax}"] = float(struct_var[i] / tot)
        result[f"sigma_weighted_mean_{ax}"] = float(sigma_w_sum[i] / m)
        result[f"mean_sep_sigma_ratio_{ax}"] = float(sep_sigma_sum[i] / m)
    if Yb is not None:
        real_res = np.asarray(Yb, dtype=np.float64) - x_center
        for i, ax in enumerate(("I", "Q")):
            rv = real_res[:, i]
            zc = (rv - rv.mean()) / (rv.std() + 1e-12)
            result[f"real_res_var_{ax}"] = float(np.var(rv))
            result[f"real_res_kurt_{ax}"] = float(np.mean(zc**4) - 3.0)
    return result


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
