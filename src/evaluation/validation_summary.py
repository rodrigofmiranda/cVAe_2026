# -*- coding: utf-8 -*-
"""Canonical validation-summary builders for protocol outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from src.evaluation.stat_tests import benjamini_hochberg


TWIN_GATE_THRESHOLDS = {
    "rel_evm_error": 0.10,
    "rel_snr_error": 0.10,
    "mean_rel_sigma": 0.10,
    "cov_rel_var": 0.20,
    "delta_psd_l2": 0.25,
    "delta_skew_l2": 0.30,
    "delta_kurt_l2": 1.25,
    "delta_jb_stat_rel": 0.20,
    "stat_qval": 0.05,
}


SUMMARY_BY_REGIME_COLUMNS: List[str] = [
    "study",
    "regime_id",
    "regime_label",
    "description",
    "run_id",
    "run_dir",
    "model_run_dir",
    "model_scope",
    "train_status",
    "eval_status",
    "best_grid_tag",
    "evm_real_%",
    "evm_pred_%",
    "delta_evm_%",
    "snr_real_db",
    "snr_pred_db",
    "delta_snr_db",
    "delta_mean_l2",
    "delta_cov_fro",
    "var_real_delta",
    "var_pred_delta",
    "var_ratio_pred_real",
    "delta_skew_l2",
    "delta_kurt_l2",
    "delta_psd_l2",
    "delta_acf_l2",
    "jb_p_min",
    "jb_log10p_min",
    "reject_gaussian",
    "jb_real_p_min",
    "jb_real_log10p_min",
    "jb_real_reject_gaussian",
    "delta_jb_log10p",
    "delta_jb_stat_rel",
    "baseline_evm_pred_%",
    "baseline_snr_pred_db",
    "baseline_delta_evm_%",
    "baseline_delta_snr_db",
    "baseline_rel_evm_error",
    "baseline_rel_snr_error",
    "baseline_mean_rel_sigma",
    "baseline_cov_rel_var",
    "cvae_evm_pred_%",
    "cvae_snr_pred_db",
    "cvae_delta_evm_%",
    "cvae_delta_snr_db",
    "cvae_rel_evm_error",
    "cvae_rel_snr_error",
    "cvae_mean_rel_sigma",
    "cvae_cov_rel_var",
    "baseline_delta_mean_l2",
    "baseline_delta_cov_fro",
    "baseline_delta_skew_l2",
    "baseline_delta_kurt_l2",
    "baseline_psd_l2",
    "baseline_delta_acf_l2",
    "baseline_jb_p_min",
    "baseline_jb_log10p_min",
    "baseline_reject_gauss",
    "cvae_delta_mean_l2",
    "cvae_delta_cov_fro",
    "cvae_delta_skew_l2",
    "cvae_delta_kurt_l2",
    "cvae_psd_l2",
    "cvae_delta_acf_l2",
    "cvae_rho_hetero_real",
    "cvae_rho_hetero_pred",
    "baseline_rho_hetero_real",
    "baseline_rho_hetero_pred",
    "cvae_stat_jsd",
    "baseline_stat_jsd",
    "cvae_jb_p_min",
    "cvae_jb_log10p_min",
    "cvae_reject_gauss",
    "stat_mmd2",
    "stat_mmd_pval",
    "stat_mmd_qval",
    "stat_mmd_bandwidth",
    "stat_mmd2_normalized",
    "stat_energy",
    "stat_energy_pval",
    "stat_energy_qval",
    "stat_psd_dist",
    "stat_psd_ci_low",
    "stat_psd_ci_high",
    "stat_n_samples",
    "stat_n_perm",
    "stat_mode",
    "dist_metrics_source",
    "n_experiments_selected",
    "dist_target_m",
    "curr_target_mA",
    "better_than_baseline_mean",
    "better_than_baseline_cov",
    "better_than_baseline_skew",
    "better_than_baseline_kurt",
    "better_than_baseline_psd",
    "gate_g1",
    "gate_g2",
    "gate_g3",
    "gate_g4",
    "gate_g5",
    "gate_g6",
    "validation_status",
]

STAT_FIDELITY_COLUMNS: List[str] = [
    "study",
    "regime_id",
    "regime_label",
    "mmd2",
    "mmd_pval",
    "mmd_qval",
    "mmd_bandwidth",
    "mmd2_normalized",
    "energy",
    "energy_pval",
    "energy_qval",
    "psd_dist",
    "psd_ci_low",
    "psd_ci_high",
    "n_samples",
    "n_perm",
    "stat_mode",
]

PROTOCOL_LEADERBOARD_COLUMNS: List[str] = [
    "rank",
    "candidate_id",
    "best_grid_tag",
    "model_run_dir",
    "model_scope",
    "n_studies",
    "n_regimes",
    "n_pass",
    "n_fail",
    "n_partial",
    "all_regimes_passed",
    "gate_g1_pass",
    "gate_g2_pass",
    "gate_g3_pass",
    "gate_g4_pass",
    "gate_g5_pass",
    "gate_g6_pass",
    "gate_pass_ratio",
    "mean_cvae_rel_evm_error",
    "mean_cvae_rel_snr_error",
    "mean_cvae_mean_rel_sigma",
    "mean_cvae_cov_rel_var",
    "mean_cvae_psd_l2",
    "mean_cvae_delta_acf_l2",
    "mean_cvae_delta_skew_l2",
    "mean_cvae_delta_kurt_l2",
    "mean_delta_jb_stat_rel",
    "mean_stat_mmd_qval",
    "mean_stat_energy_qval",
    "protocol_score_v1",
]


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        if isinstance(value, str) and not value.strip():
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _safe_bool(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no"}:
            return False
    return None


def _first_finite(*values: Any) -> float:
    for value in values:
        candidate = _safe_float(value)
        if np.isfinite(candidate):
            return candidate
    return float("nan")


def _first_bool(*values: Any) -> Any:
    for value in values:
        candidate = _safe_bool(value)
        if candidate is not None:
            return candidate
    return None


def _lt(lhs: Any, rhs: Any) -> Any:
    a = _safe_float(lhs)
    b = _safe_float(rhs)
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    return bool(a < b)


def _abs_lt(value: Any, threshold: float) -> Any:
    v = _safe_float(value)
    if not np.isfinite(v):
        return None
    return bool(abs(v) < float(threshold))


def _gt(lhs: Any, rhs: Any) -> Any:
    a = _safe_float(lhs)
    b = _safe_float(rhs)
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    return bool(a > b)


def _validation_status(row: pd.Series) -> str:
    gates = [row.get(f"gate_g{i}") for i in range(1, 7)]
    gate_values = [_safe_bool(v) for v in gates]
    if any(v is False for v in gate_values):
        return "fail"
    if all(v is True for v in gate_values):
        return "pass"
    return "partial"


def _gate_all(*values: Any) -> Any:
    states = [_safe_bool(v) for v in values]
    if any(v is False for v in states):
        return False
    if all(v is True for v in states):
        return True
    return None


def _empty_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SUMMARY_BY_REGIME_COLUMNS)


def _empty_protocol_leaderboard() -> pd.DataFrame:
    return pd.DataFrame(columns=PROTOCOL_LEADERBOARD_COLUMNS)


def _build_row(result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = result.get("metrics", {}) or {}
    baseline = result.get("baseline", {}) or {}
    baseline_dist = result.get("baseline_dist", {}) or {}
    cvae_dist = result.get("cvae_dist", {}) or {}
    stat = result.get("stat_fidelity", {}) or {}

    delta_mean = _first_finite(metrics.get("delta_mean_l2"), cvae_dist.get("delta_mean_l2"))
    delta_cov = _first_finite(metrics.get("delta_cov_fro"), cvae_dist.get("delta_cov_fro"))
    delta_skew = _first_finite(metrics.get("delta_skew_l2"), cvae_dist.get("delta_skew_l2"))
    delta_kurt = _first_finite(metrics.get("delta_kurt_l2"), cvae_dist.get("delta_kurt_l2"))
    delta_psd = _first_finite(metrics.get("delta_psd_l2"), cvae_dist.get("psd_l2"))
    delta_acf = _first_finite(metrics.get("delta_acf_l2"), cvae_dist.get("delta_acf_l2"))
    jb_p_min = _first_finite(metrics.get("jb_p_min"), cvae_dist.get("jb_p_min"))
    jb_log10p_min = _first_finite(metrics.get("jb_log10p_min"), cvae_dist.get("jb_log10p_min"))
    reject_gaussian = _first_bool(metrics.get("reject_gaussian"), cvae_dist.get("reject_gaussian"))
    jb_real_p_min = _first_finite(metrics.get("jb_real_p_min"), cvae_dist.get("jb_real_p_min"))
    jb_real_log10p_min = _first_finite(
        metrics.get("jb_real_log10p_min"),
        cvae_dist.get("jb_real_log10p_min"),
    )
    jb_real_reject = _first_bool(
        metrics.get("jb_real_reject_gaussian"),
        cvae_dist.get("jb_real_reject_gaussian"),
    )

    row = {
        "study": result.get("_study", "within_regime"),
        "regime_id": result.get("regime_id", ""),
        "regime_label": result.get("regime_label", ""),
        "description": result.get("description", ""),
        "run_id": result.get("run_id", ""),
        "run_dir": result.get("run_dir", ""),
        "model_run_dir": result.get("model_run_dir", result.get("run_dir", "")),
        "model_scope": result.get("model_scope", "per_regime"),
        "train_status": result.get("train_status", ""),
        "eval_status": result.get("eval_status", ""),
        "best_grid_tag": result.get("best_grid_tag", ""),
        "evm_real_%": _safe_float(metrics.get("evm_real_%")),
        "evm_pred_%": _safe_float(metrics.get("evm_pred_%")),
        "delta_evm_%": _safe_float(metrics.get("delta_evm_%")),
        "snr_real_db": _safe_float(metrics.get("snr_real_db")),
        "snr_pred_db": _safe_float(metrics.get("snr_pred_db")),
        "delta_snr_db": _safe_float(metrics.get("delta_snr_db")),
        "delta_mean_l2": delta_mean,
        "delta_cov_fro": delta_cov,
        "var_real_delta": _first_finite(metrics.get("var_real_delta"), result.get("var_real_delta")),
        "var_pred_delta": _first_finite(metrics.get("var_pred_delta"), result.get("var_pred_delta")),
        "delta_skew_l2": delta_skew,
        "delta_kurt_l2": delta_kurt,
        "delta_psd_l2": delta_psd,
        "delta_acf_l2": delta_acf,
        "jb_p_min": jb_p_min,
        "jb_log10p_min": jb_log10p_min,
        "reject_gaussian": reject_gaussian,
        "jb_real_p_min": jb_real_p_min,
        "jb_real_log10p_min": jb_real_log10p_min,
        "jb_real_reject_gaussian": jb_real_reject,
        "baseline_evm_pred_%": _safe_float(baseline.get("evm_pred_%")),
        "baseline_snr_pred_db": _safe_float(baseline.get("snr_pred_db")),
        "baseline_delta_evm_%": _safe_float(baseline.get("delta_evm_%")),
        "baseline_delta_snr_db": _safe_float(baseline.get("delta_snr_db")),
        "cvae_evm_pred_%": _safe_float(metrics.get("evm_pred_%")),
        "cvae_snr_pred_db": _safe_float(metrics.get("snr_pred_db")),
        "cvae_delta_evm_%": _safe_float(metrics.get("delta_evm_%")),
        "cvae_delta_snr_db": _safe_float(metrics.get("delta_snr_db")),
        "baseline_delta_mean_l2": _safe_float(baseline_dist.get("delta_mean_l2")),
        "baseline_delta_cov_fro": _safe_float(baseline_dist.get("delta_cov_fro")),
        "baseline_delta_skew_l2": _safe_float(baseline_dist.get("delta_skew_l2")),
        "baseline_delta_kurt_l2": _safe_float(baseline_dist.get("delta_kurt_l2")),
        "baseline_psd_l2": _safe_float(baseline_dist.get("psd_l2")),
        "baseline_delta_acf_l2": _safe_float(baseline_dist.get("delta_acf_l2")),
        "baseline_jb_p_min": _safe_float(baseline_dist.get("jb_p_min")),
        "baseline_jb_log10p_min": _safe_float(baseline_dist.get("jb_log10p_min")),
        "baseline_reject_gauss": _safe_bool(baseline_dist.get("reject_gaussian")),
        "cvae_delta_mean_l2": _first_finite(cvae_dist.get("delta_mean_l2"), delta_mean),
        "cvae_delta_cov_fro": _first_finite(cvae_dist.get("delta_cov_fro"), delta_cov),
        "cvae_delta_skew_l2": _first_finite(cvae_dist.get("delta_skew_l2"), delta_skew),
        "cvae_delta_kurt_l2": _first_finite(cvae_dist.get("delta_kurt_l2"), delta_kurt),
        "cvae_psd_l2": _first_finite(cvae_dist.get("psd_l2"), delta_psd),
        "cvae_delta_acf_l2": _first_finite(cvae_dist.get("delta_acf_l2"), delta_acf),
        "cvae_rho_hetero_real": _first_finite(
            metrics.get("rho_hetero_real"), cvae_dist.get("rho_hetero_real")
        ),
        "cvae_rho_hetero_pred": _first_finite(
            metrics.get("rho_hetero_pred"), cvae_dist.get("rho_hetero_pred")
        ),
        "baseline_rho_hetero_real": _safe_float(baseline_dist.get("rho_hetero_real")),
        "baseline_rho_hetero_pred": _safe_float(baseline_dist.get("rho_hetero_pred")),
        "cvae_stat_jsd": _first_finite(
            metrics.get("stat_jsd"), cvae_dist.get("stat_jsd")
        ),
        "baseline_stat_jsd": _safe_float(baseline_dist.get("stat_jsd")),
        "cvae_jb_p_min": _first_finite(cvae_dist.get("jb_p_min"), jb_p_min),
        "cvae_jb_log10p_min": _first_finite(cvae_dist.get("jb_log10p_min"), jb_log10p_min),
        "cvae_reject_gauss": _first_bool(cvae_dist.get("reject_gaussian"), reject_gaussian),
        "stat_mmd2": _safe_float(stat.get("mmd2")),
        "stat_mmd_pval": _safe_float(stat.get("mmd_pval")),
        "stat_mmd_qval": float("nan"),
        "stat_mmd_bandwidth": _safe_float(stat.get("mmd_bandwidth")),
        "stat_mmd2_normalized": float("nan"),
        "stat_energy": _safe_float(stat.get("energy")),
        "stat_energy_pval": _safe_float(stat.get("energy_pval")),
        "stat_energy_qval": float("nan"),
        "stat_psd_dist": _safe_float(stat.get("psd_dist")),
        "stat_psd_ci_low": _safe_float(stat.get("psd_ci_low")),
        "stat_psd_ci_high": _safe_float(stat.get("psd_ci_high")),
        "stat_n_samples": _safe_float(stat.get("n_samples")),
        "stat_n_perm": _safe_float(stat.get("n_perm")),
        "stat_mode": stat.get("stat_mode"),
        "dist_metrics_source": result.get("dist_metrics_source"),
        "n_experiments_selected": int(len(result.get("selected_experiments", []))),
        "dist_target_m": _safe_float(result.get("selection_criteria", {}).get("distance_m")),
        "curr_target_mA": _safe_float(result.get("selection_criteria", {}).get("current_mA")),
    }
    return row


def _apply_fdr(df: pd.DataFrame) -> None:
    valid_mmd = df["stat_mmd_pval"].notna()
    valid_energy = df["stat_energy_pval"].notna()
    if not valid_mmd.any() and not valid_energy.any():
        return

    parts = []
    sizes = []
    if valid_mmd.any():
        parts.append(df.loc[valid_mmd, "stat_mmd_pval"].to_numpy(dtype=float))
        sizes.append(("mmd", valid_mmd, int(valid_mmd.sum())))
    if valid_energy.any():
        parts.append(df.loc[valid_energy, "stat_energy_pval"].to_numpy(dtype=float))
        sizes.append(("energy", valid_energy, int(valid_energy.sum())))

    all_qvals = benjamini_hochberg(np.concatenate(parts))
    cursor = 0
    for label, mask, size in sizes:
        segment = all_qvals[cursor : cursor + size]
        cursor += size
        if label == "mmd":
            df.loc[mask, "stat_mmd_qval"] = segment
        else:
            df.loc[mask, "stat_energy_qval"] = segment


def _apply_derived_metrics(df: pd.DataFrame) -> None:
    den = pd.to_numeric(df["var_real_delta"], errors="coerce")
    num = pd.to_numeric(df["var_pred_delta"], errors="coerce")
    df["var_ratio_pred_real"] = np.where(den > 0, num / den, np.nan)

    evm_real = pd.to_numeric(df["evm_real_%"], errors="coerce").abs()
    snr_real = pd.to_numeric(df["snr_real_db"], errors="coerce").abs()
    baseline_delta_evm = pd.to_numeric(df["baseline_delta_evm_%"], errors="coerce").abs()
    baseline_delta_snr = pd.to_numeric(df["baseline_delta_snr_db"], errors="coerce").abs()
    cvae_delta_evm = pd.to_numeric(df["cvae_delta_evm_%"], errors="coerce").abs()
    cvae_delta_snr = pd.to_numeric(df["cvae_delta_snr_db"], errors="coerce").abs()
    df["baseline_rel_evm_error"] = np.where(evm_real > 0, baseline_delta_evm / evm_real, np.nan)
    df["baseline_rel_snr_error"] = np.where(snr_real > 0, baseline_delta_snr / snr_real, np.nan)
    df["cvae_rel_evm_error"] = np.where(evm_real > 0, cvae_delta_evm / evm_real, np.nan)
    df["cvae_rel_snr_error"] = np.where(snr_real > 0, cvae_delta_snr / snr_real, np.nan)
    sigma_real = np.sqrt(den)
    df["baseline_mean_rel_sigma"] = np.where(
        sigma_real > 0,
        pd.to_numeric(df["baseline_delta_mean_l2"], errors="coerce") / sigma_real,
        np.nan,
    )
    df["cvae_mean_rel_sigma"] = np.where(
        sigma_real > 0,
        pd.to_numeric(df["cvae_delta_mean_l2"], errors="coerce") / sigma_real,
        np.nan,
    )
    df["baseline_cov_rel_var"] = np.where(
        den > 0,
        pd.to_numeric(df["baseline_delta_cov_fro"], errors="coerce") / den,
        np.nan,
    )
    df["cvae_cov_rel_var"] = np.where(
        den > 0,
        pd.to_numeric(df["cvae_delta_cov_fro"], errors="coerce") / den,
        np.nan,
    )

    stat_den = pd.to_numeric(df["var_real_delta"], errors="coerce")
    stat_num = pd.to_numeric(df["stat_mmd2"], errors="coerce")
    df["stat_mmd2_normalized"] = np.where(stat_den > 0, stat_num / stat_den, np.nan)

    log10p_pred = pd.to_numeric(df["cvae_jb_log10p_min"], errors="coerce")
    log10p_real = pd.to_numeric(df["jb_real_log10p_min"], errors="coerce")
    df["delta_jb_log10p"] = np.abs(log10p_pred - log10p_real)
    # Relative JB non-Gaussianity gap (N-invariant): |Δlog10p| / |log10p_real|.
    # Both numerator and denominator scale linearly with N, so the ratio is stable
    # across evaluation set sizes.  Threshold 0.20 means model reproduces the
    # channel's non-Gaussianity level within 20% of the real value.
    df["delta_jb_stat_rel"] = np.where(
        log10p_real.abs() > 0,
        df["delta_jb_log10p"] / log10p_real.abs(),
        np.nan,
    )

    df["better_than_baseline_mean"] = [
        _lt(cv, bl)
        for cv, bl in zip(df["cvae_delta_mean_l2"], df["baseline_delta_mean_l2"])
    ]
    df["better_than_baseline_cov"] = [
        _lt(cv, bl)
        for cv, bl in zip(df["cvae_delta_cov_fro"], df["baseline_delta_cov_fro"])
    ]
    df["better_than_baseline_skew"] = [
        _lt(cv, bl)
        for cv, bl in zip(df["cvae_delta_skew_l2"], df["baseline_delta_skew_l2"])
    ]
    df["better_than_baseline_kurt"] = [
        _lt(cv, bl)
        for cv, bl in zip(df["cvae_delta_kurt_l2"], df["baseline_delta_kurt_l2"])
    ]
    df["better_than_baseline_psd"] = [
        _lt(cv, bl)
        for cv, bl in zip(df["cvae_psd_l2"], df["baseline_psd_l2"])
    ]

    jb_rel_ok = [_lt(v, TWIN_GATE_THRESHOLDS["delta_jb_stat_rel"]) for v in df["delta_jb_stat_rel"]]
    mmd_ok = [_gt(v, TWIN_GATE_THRESHOLDS["stat_qval"]) for v in df["stat_mmd_qval"]]
    energy_ok = [_gt(v, TWIN_GATE_THRESHOLDS["stat_qval"]) for v in df["stat_energy_qval"]]

    # Gate ladder for a digital-twin reading:
    # G1/G2 = direct signal fidelity to the measured channel.
    # G3/G4/G5 = residual-structure fidelity to the measured channel.
    # G6 = formal distributional indistinguishability.
    #
    # Baseline columns remain in the canonical CSV as benchmark diagnostics only;
    # they do not participate in validation_status anymore.
    df["gate_g1"] = [_lt(v, TWIN_GATE_THRESHOLDS["rel_evm_error"]) for v in df["cvae_rel_evm_error"]]
    df["gate_g2"] = [_lt(v, TWIN_GATE_THRESHOLDS["rel_snr_error"]) for v in df["cvae_rel_snr_error"]]
    df["gate_g3"] = [
        _gate_all(mean_ok, cov_ok)
        for mean_ok, cov_ok in zip(
            [_lt(v, TWIN_GATE_THRESHOLDS["mean_rel_sigma"]) for v in df["cvae_mean_rel_sigma"]],
            [_lt(v, TWIN_GATE_THRESHOLDS["cov_rel_var"]) for v in df["cvae_cov_rel_var"]],
        )
    ]
    df["gate_g4"] = [_lt(v, TWIN_GATE_THRESHOLDS["delta_psd_l2"]) for v in df["cvae_psd_l2"]]
    df["gate_g5"] = [
        _gate_all(skew_ok, kurt_ok, jb_ok)
        for skew_ok, kurt_ok, jb_ok in zip(
            [_lt(v, TWIN_GATE_THRESHOLDS["delta_skew_l2"]) for v in df["cvae_delta_skew_l2"]],
            [_lt(v, TWIN_GATE_THRESHOLDS["delta_kurt_l2"]) for v in df["cvae_delta_kurt_l2"]],
            jb_rel_ok,
        )
    ]
    df["gate_g6"] = [
        _gate_all(mmd_pass, energy_pass)
        for mmd_pass, energy_pass in zip(mmd_ok, energy_ok)
    ]
    df["validation_status"] = df.apply(_validation_status, axis=1)


def _candidate_id(best_grid_tag: Any, model_run_dir: Any, model_scope: Any) -> str:
    tag = str(best_grid_tag or "").strip()
    if tag:
        return tag
    run_dir = str(model_run_dir or "").strip()
    if run_dir:
        return Path(run_dir).name
    return str(model_scope or "candidate")


def _finite_ratio(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan")
    return float(vals.mean())


def _normalized_lower_better(series: pd.Series, threshold: float) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    out = vals / float(threshold)
    return out.clip(lower=0.0, upper=100.0)


def _normalized_higher_better(series: pd.Series, threshold: float) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    safe = vals.where(vals > 1e-12, np.nan)
    out = float(threshold) / safe
    return out.clip(lower=0.0, upper=100.0)


def _gate_pass_ratio(df_group: pd.DataFrame) -> float:
    gate_cols = [f"gate_g{i}" for i in range(1, 7)]
    total = 0
    passed = 0
    for col in gate_cols:
        for raw in df_group[col].tolist():
            state = _safe_bool(raw)
            if state is None:
                continue
            total += 1
            if state:
                passed += 1
    if total == 0:
        return float("nan")
    return float(passed / total)


def _protocol_score_v1(df_group: pd.DataFrame) -> float:
    parts = [
        _normalized_lower_better(df_group["cvae_rel_evm_error"], TWIN_GATE_THRESHOLDS["rel_evm_error"]),
        _normalized_lower_better(df_group["cvae_rel_snr_error"], TWIN_GATE_THRESHOLDS["rel_snr_error"]),
        _normalized_lower_better(df_group["cvae_mean_rel_sigma"], TWIN_GATE_THRESHOLDS["mean_rel_sigma"]),
        _normalized_lower_better(df_group["cvae_cov_rel_var"], TWIN_GATE_THRESHOLDS["cov_rel_var"]),
        _normalized_lower_better(df_group["cvae_psd_l2"], TWIN_GATE_THRESHOLDS["delta_psd_l2"]),
        _normalized_lower_better(df_group["cvae_delta_skew_l2"], TWIN_GATE_THRESHOLDS["delta_skew_l2"]),
        _normalized_lower_better(df_group["cvae_delta_kurt_l2"], TWIN_GATE_THRESHOLDS["delta_kurt_l2"]),
        _normalized_lower_better(df_group["delta_jb_stat_rel"], TWIN_GATE_THRESHOLDS["delta_jb_stat_rel"]),
        _normalized_higher_better(df_group["stat_mmd_qval"], TWIN_GATE_THRESHOLDS["stat_qval"]),
        _normalized_higher_better(df_group["stat_energy_qval"], TWIN_GATE_THRESHOLDS["stat_qval"]),
    ]
    stacked = np.concatenate(
        [
            pd.to_numeric(part, errors="coerce").to_numpy(dtype=float)
            for part in parts
        ]
    )
    stacked = stacked[np.isfinite(stacked)]
    if len(stacked) == 0:
        return float("nan")
    return float(np.mean(stacked))


def _normalize_column_set(df: pd.DataFrame) -> pd.DataFrame:
    for col in SUMMARY_BY_REGIME_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df.loc[:, SUMMARY_BY_REGIME_COLUMNS]


def build_validation_summary_table(results: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Build the canonical per-regime validation summary table."""
    rows = [_build_row(result) for result in results]
    if not rows:
        return _empty_summary_frame()

    df = pd.DataFrame(rows)
    _apply_fdr(df)
    _apply_derived_metrics(df)
    return _normalize_column_set(df)


def build_protocol_leaderboard(df_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the canonical summary into a candidate leaderboard.

    This is the protocol-first ranking surface. It summarizes each evaluated
    candidate using the same per-regime gates and fidelity metrics that drive
    the scientific validation, instead of relying on training-only scores.
    """
    if df_summary is None or df_summary.empty:
        return _empty_protocol_leaderboard()

    df = df_summary.copy()
    for col in SUMMARY_BY_REGIME_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["candidate_id"] = [
        _candidate_id(tag, run_dir, scope)
        for tag, run_dir, scope in zip(
            df["best_grid_tag"],
            df["model_run_dir"],
            df["model_scope"],
        )
    ]

    rows = []
    group_cols = ["candidate_id", "best_grid_tag", "model_run_dir", "model_scope"]
    for key, grp in df.groupby(group_cols, dropna=False, sort=False):
        candidate_id, best_grid_tag, model_run_dir, model_scope = key
        n_regimes = int(len(grp))
        statuses = grp["validation_status"].astype(str)
        gate_passes = {
            f"gate_g{i}_pass": int(sum(_safe_bool(v) is True for v in grp[f"gate_g{i}"].tolist()))
            for i in range(1, 7)
        }
        rows.append(
            {
                "candidate_id": candidate_id,
                "best_grid_tag": best_grid_tag,
                "model_run_dir": model_run_dir,
                "model_scope": model_scope,
                "n_studies": int(grp["study"].astype(str).nunique()),
                "n_regimes": n_regimes,
                "n_pass": int((statuses == "pass").sum()),
                "n_fail": int((statuses == "fail").sum()),
                "n_partial": int((statuses == "partial").sum()),
                "all_regimes_passed": bool(n_regimes > 0 and (statuses == "pass").all()),
                **gate_passes,
                "gate_pass_ratio": _gate_pass_ratio(grp),
                "mean_cvae_rel_evm_error": _finite_ratio(grp["cvae_rel_evm_error"]),
                "mean_cvae_rel_snr_error": _finite_ratio(grp["cvae_rel_snr_error"]),
                "mean_cvae_mean_rel_sigma": _finite_ratio(grp["cvae_mean_rel_sigma"]),
                "mean_cvae_cov_rel_var": _finite_ratio(grp["cvae_cov_rel_var"]),
                "mean_cvae_psd_l2": _finite_ratio(grp["cvae_psd_l2"]),
                "mean_cvae_delta_acf_l2": _finite_ratio(grp["cvae_delta_acf_l2"]),
                "mean_cvae_delta_skew_l2": _finite_ratio(grp["cvae_delta_skew_l2"]),
                "mean_cvae_delta_kurt_l2": _finite_ratio(grp["cvae_delta_kurt_l2"]),
                "mean_delta_jb_stat_rel": _finite_ratio(grp["delta_jb_stat_rel"]),
                "mean_stat_mmd_qval": _finite_ratio(grp["stat_mmd_qval"]),
                "mean_stat_energy_qval": _finite_ratio(grp["stat_energy_qval"]),
                "protocol_score_v1": _protocol_score_v1(grp),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_protocol_leaderboard()

    out = out.sort_values(
        by=[
            "all_regimes_passed",
            "n_pass",
            "gate_pass_ratio",
            "protocol_score_v1",
            "mean_cvae_rel_evm_error",
            "mean_cvae_psd_l2",
        ],
        ascending=[False, False, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    for col in PROTOCOL_LEADERBOARD_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out.loc[:, PROTOCOL_LEADERBOARD_COLUMNS]


def recompute_validation_summary(df_summary: pd.DataFrame) -> pd.DataFrame:
    """Recompute derived validation columns from an existing summary table.

    This is intended for backfilling older experiment outputs after gate
    changes. It preserves raw metric columns already present in the CSV and
    refreshes only the derived helpers:

    - relative error columns
    - baseline-vs-cVAE helper flags
    - gate_g1 ... gate_g6
    - validation_status
    - FDR-corrected q-values when p-values are available
    """
    if df_summary is None or df_summary.empty:
        return _empty_summary_frame()

    df = df_summary.copy()
    for col in SUMMARY_BY_REGIME_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    _apply_fdr(df)
    _apply_derived_metrics(df)
    return _normalize_column_set(df)


def build_stat_fidelity_table(df_summary: pd.DataFrame) -> pd.DataFrame:
    """Project the canonical summary into the legacy stat-fidelity table."""
    if df_summary is None or df_summary.empty:
        return pd.DataFrame(columns=STAT_FIDELITY_COLUMNS)

    rows = []
    for _, row in df_summary.iterrows():
        mmd2 = _safe_float(row.get("stat_mmd2"))
        if not np.isfinite(mmd2):
            continue
        rows.append(
            {
                "study": row.get("study"),
                "regime_id": row.get("regime_id"),
                "regime_label": row.get("regime_label"),
                "mmd2": mmd2,
                "mmd_pval": _safe_float(row.get("stat_mmd_pval")),
                "mmd_qval": _safe_float(row.get("stat_mmd_qval")),
                "mmd_bandwidth": _safe_float(row.get("stat_mmd_bandwidth")),
                "mmd2_normalized": _safe_float(row.get("stat_mmd2_normalized")),
                "energy": _safe_float(row.get("stat_energy")),
                "energy_pval": _safe_float(row.get("stat_energy_pval")),
                "energy_qval": _safe_float(row.get("stat_energy_qval")),
                "psd_dist": _safe_float(row.get("stat_psd_dist")),
                "psd_ci_low": _safe_float(row.get("stat_psd_ci_low")),
                "psd_ci_high": _safe_float(row.get("stat_psd_ci_high")),
                "n_samples": _safe_float(row.get("stat_n_samples")),
                "n_perm": _safe_float(row.get("stat_n_perm")),
                "stat_mode": row.get("stat_mode"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=STAT_FIDELITY_COLUMNS)
    for col in STAT_FIDELITY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df.loc[:, STAT_FIDELITY_COLUMNS]


def build_stat_acceptance_summary(
    df_summary: pd.DataFrame,
    *,
    q_alpha: float = 0.05,
    psd_ratio_limit: float = 1.2,
) -> Dict[str, Any] | None:
    """Aggregate acceptance counters from the canonical summary table."""
    if df_summary is None or df_summary.empty:
        return None

    df = df_summary.copy()
    tested = df[df["stat_mmd_qval"].notna()].copy()
    if tested.empty:
        return None

    pass_mmd = int((tested["stat_mmd_qval"] > q_alpha).sum())
    pass_energy = int((tested["stat_energy_qval"] > q_alpha).sum())
    pass_both = int(
        ((tested["stat_mmd_qval"] > q_alpha) & (tested["stat_energy_qval"] > q_alpha)).sum()
    )

    psd_subset = tested.dropna(subset=["stat_psd_dist", "baseline_psd_l2"])
    psd_checked = not psd_subset.empty
    if psd_checked:
        pass_psd = int((psd_subset["stat_psd_dist"] <= psd_ratio_limit * psd_subset["baseline_psd_l2"]).sum())
        n_psd = int(len(psd_subset))
    else:
        pass_psd = int(len(tested))
        n_psd = 0

    n_tested = int(len(tested))
    return {
        "q_alpha": float(q_alpha),
        "psd_ratio_limit": float(psd_ratio_limit),
        "n_regimes_tested": n_tested,
        "pass_mmd_qval": pass_mmd,
        "pass_energy_qval": pass_energy,
        "pass_both_qval": pass_both,
        "pct_pass_mmd": round(100.0 * pass_mmd / n_tested, 1) if n_tested else 0.0,
        "pct_pass_energy": round(100.0 * pass_energy / n_tested, 1) if n_tested else 0.0,
        "pct_pass_both": round(100.0 * pass_both / n_tested, 1) if n_tested else 0.0,
        "psd_ratio_checked": psd_checked,
        "pass_psd_ratio": pass_psd,
        "n_regimes_psd_checked": n_psd,
        "pct_pass_psd_ratio": round(100.0 * pass_psd / n_psd, 1) if psd_checked and n_psd else None,
    }
