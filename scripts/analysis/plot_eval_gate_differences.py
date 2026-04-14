#!/usr/bin/env python3
"""Build global regime heatmaps from a per-regime evaluation batch."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.runtime_env import (
    ensure_required_python_modules,
    ensure_writable_mpl_config_dir,
)

ensure_writable_mpl_config_dir()
ensure_required_python_modules(
    ("numpy", "pandas", "matplotlib"),
    context="evaluation heatmap plotting",
    allow_missing=False,
)

import pandas as pd

from src.evaluation.summary_plots import (
    plot_eval_gate_difference_heatmaps,
    plot_eval_stat_screen_heatmaps,
    plot_eval_gate_threshold_heatmaps,
    plot_eval_gate_supplementary_heatmaps,
)
from src.evaluation.validation_summary import TWIN_GATE_THRESHOLDS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate global gate-difference heatmaps for an eval batch."
    )
    parser.add_argument(
        "eval_root",
        type=Path,
        help="Path to eval_16qam_all_regimes_* root.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Override output plots dir (default: <eval_root>/plots/best_model).",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="Override output tables dir (default: <eval_root>/tables).",
    )
    return parser.parse_args()


def _load_manifest(eval_root: Path) -> List[Dict[str, Any]]:
    manifest_path = eval_root / "manifest_all_regimes_eval.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected manifest schema in {manifest_path}")
    return rows


def _metric_path(eval_root: Path, row: Dict[str, Any]) -> Path:
    regime_id = str(row.get("regime_id", "")).strip()
    if regime_id:
        return eval_root / regime_id / "logs" / "metricas_globais_reanalysis.json"
    run_dir = str(row.get("run_dir", "")).strip()
    if run_dir:
        return Path(run_dir) / "logs" / "metricas_globais_reanalysis.json"
    raise ValueError(f"Missing regime_id/run_dir in manifest row: {row}")


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_ratio(num: float, den: float) -> float:
    if math.isfinite(num) and math.isfinite(den) and den > 0:
        return float(num / den)
    return float("nan")


def _build_summary_df(eval_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for row in _load_manifest(eval_root):
        if str(row.get("status", "")).lower() != "completed":
            skipped.append(str(row.get("regime_id", "unknown")))
            continue
        metric_path = _metric_path(eval_root, row)
        if not metric_path.exists():
            skipped.append(f"{row.get('regime_id', 'unknown')} (missing metrics)")
            continue

        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        delta_jb_i = abs(_coerce_float(metrics.get("delta_jb_stat_rel_I")))
        delta_jb_q = abs(_coerce_float(metrics.get("delta_jb_stat_rel_Q")))
        rho_real = _coerce_float(metrics.get("rho_hetero_real"))
        rho_pred = _coerce_float(metrics.get("rho_hetero_pred"))
        delta_evm = abs(_coerce_float(metrics.get("delta_evm_%")))
        delta_snr = abs(_coerce_float(metrics.get("delta_snr_db")))
        evm_real = abs(_coerce_float(metrics.get("evm_real_%")))
        snr_real = abs(_coerce_float(metrics.get("snr_real_db")))
        delta_mean_l2 = _coerce_float(metrics.get("delta_mean_l2"))
        delta_cov_fro = _coerce_float(metrics.get("delta_cov_fro"))
        var_real_delta = _coerce_float(metrics.get("var_real_delta"))
        sigma_real = math.sqrt(var_real_delta) if math.isfinite(var_real_delta) and var_real_delta > 0 else float("nan")
        g1_rel_error = _safe_ratio(delta_evm, evm_real)
        g2_rel_error = _safe_ratio(delta_snr, snr_real)
        g3_mean_rel_sigma = _safe_ratio(delta_mean_l2, sigma_real)
        g3_cov_rel_var = _safe_ratio(delta_cov_fro, var_real_delta)
        g4_psd = _coerce_float(metrics.get("delta_psd_l2"))
        g5_skew = _coerce_float(metrics.get("delta_skew_l2"))
        g5_kurt = _coerce_float(metrics.get("delta_kurt_l2"))
        g5_jb = max(delta_jb_i, delta_jb_q)

        rows.append(
            {
                "regime_id": row.get("regime_id"),
                "dist_target_m": _coerce_float(row.get("distance_m")),
                "curr_target_mA": _coerce_float(row.get("current_mA")),
                "delta_evm_%": _coerce_float(metrics.get("delta_evm_%")),
                "delta_snr_db": _coerce_float(metrics.get("delta_snr_db")),
                "delta_mean_l2": delta_mean_l2,
                "delta_cov_fro": delta_cov_fro,
                "delta_psd_l2": g4_psd,
                "delta_skew_l2": g5_skew,
                "delta_kurt_l2": g5_kurt,
                "delta_jb_stat_rel": g5_jb,
                "delta_acf_l2": _coerce_float(metrics.get("delta_acf_l2")),
                "delta_coverage_95": _coerce_float(metrics.get("delta_coverage_95")),
                "rho_hetero_real": rho_real,
                "rho_hetero_pred": rho_pred,
                "rho_hetero_abs_gap": abs(rho_pred - rho_real),
                "stat_jsd": _coerce_float(metrics.get("stat_jsd")),
                "evm_real_%": _coerce_float(metrics.get("evm_real_%")),
                "evm_pred_%": _coerce_float(metrics.get("evm_pred_%")),
                "snr_real_db": _coerce_float(metrics.get("snr_real_db")),
                "snr_pred_db": _coerce_float(metrics.get("snr_pred_db")),
                "coverage_95": _coerce_float(metrics.get("coverage_95")),
                "var_real_delta": var_real_delta,
                "var_pred_delta": _coerce_float(metrics.get("var_pred_delta")),
                "stat_mmd2": _coerce_float(metrics.get("stat_mmd2")),
                "stat_mmd_qval": _coerce_float(metrics.get("stat_mmd_qval")),
                "stat_energy": _coerce_float(metrics.get("stat_energy")),
                "stat_energy_qval": _coerce_float(metrics.get("stat_energy_qval")),
                "stat_psd_dist": _coerce_float(metrics.get("stat_psd_dist")),
                "mi_aux_real_bits": _coerce_float(metrics.get("mi_aux_real_bits")),
                "mi_aux_pred_bits": _coerce_float(metrics.get("mi_aux_pred_bits")),
                "mi_aux_gap_rel": _coerce_float(metrics.get("mi_aux_gap_rel")),
                "gmi_aux_real_bits": _coerce_float(metrics.get("gmi_aux_real_bits")),
                "gmi_aux_pred_bits": _coerce_float(metrics.get("gmi_aux_pred_bits")),
                "gmi_aux_gap_rel": _coerce_float(metrics.get("gmi_aux_gap_rel")),
                "ngmi_aux_real": _coerce_float(metrics.get("ngmi_aux_real")),
                "ngmi_aux_pred": _coerce_float(metrics.get("ngmi_aux_pred")),
                "ngmi_aux_gap": _coerce_float(metrics.get("ngmi_aux_gap")),
                "air_aux_real_bits": _coerce_float(metrics.get("air_aux_real_bits")),
                "air_aux_pred_bits": _coerce_float(metrics.get("air_aux_pred_bits")),
                "air_aux_gap_rel": _coerce_float(metrics.get("air_aux_gap_rel")),
                "g1_rel_error": g1_rel_error,
                "g2_rel_error": g2_rel_error,
                "g3_mean_rel_sigma": g3_mean_rel_sigma,
                "g3_cov_rel_var": g3_cov_rel_var,
                "gate_g1_ratio": _safe_ratio(g1_rel_error, TWIN_GATE_THRESHOLDS["rel_evm_error"]),
                "gate_g2_ratio": _safe_ratio(g2_rel_error, TWIN_GATE_THRESHOLDS["rel_snr_error"]),
                "gate_g3_mean_ratio": _safe_ratio(g3_mean_rel_sigma, TWIN_GATE_THRESHOLDS["mean_rel_sigma"]),
                "gate_g3_cov_ratio": _safe_ratio(g3_cov_rel_var, TWIN_GATE_THRESHOLDS["cov_rel_var"]),
                "gate_g4_ratio": _safe_ratio(g4_psd, TWIN_GATE_THRESHOLDS["delta_psd_l2"]),
                "gate_g5_skew_ratio": _safe_ratio(g5_skew, TWIN_GATE_THRESHOLDS["delta_skew_l2"]),
                "gate_g5_kurt_ratio": _safe_ratio(g5_kurt, TWIN_GATE_THRESHOLDS["delta_kurt_l2"]),
                "gate_g5_jb_ratio": _safe_ratio(g5_jb, TWIN_GATE_THRESHOLDS["delta_jb_stat_rel"]),
            }
        )

    if skipped:
        print(f"[warn] skipped_rows={len(skipped)}")
        for item in skipped[:10]:
            print(f"  - {item}")
        if len(skipped) > 10:
            print(f"  - ... and {len(skipped) - 10} more")

    if not rows:
        raise RuntimeError(f"No completed regime metrics found under {eval_root}")
    return pd.DataFrame(rows).sort_values(["dist_target_m", "curr_target_mA"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    ensure_writable_mpl_config_dir()
    ensure_required_python_modules(
        ("numpy", "pandas", "matplotlib"),
        context="evaluation heatmap plotting",
        allow_missing=False,
    )
    eval_root = args.eval_root.expanduser().resolve()
    plots_dir = (
        args.plots_dir.expanduser().resolve()
        if args.plots_dir is not None
        else (eval_root / "plots" / "best_model").resolve()
    )
    tables_dir = (
        args.tables_dir.expanduser().resolve()
        if args.tables_dir is not None
        else (eval_root / "tables").resolve()
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = _build_summary_df(eval_root)
    csv_path = tables_dir / "global_gate_differences_by_regime.csv"
    df.to_csv(csv_path, index=False)

    threshold_plot = plot_eval_gate_threshold_heatmaps(
        df,
        plots_dir,
        fname="heatmap_twin_gate_metrics_by_regime.png",
    )
    legacy_threshold_plot = plot_eval_gate_threshold_heatmaps(
        df,
        plots_dir,
        fname="heatmap_gate_metrics_by_regime.png",
    )
    main_plot = plot_eval_gate_difference_heatmaps(
        df,
        plots_dir,
        fname="heatmap_twin_gate_differences_by_regime.png",
    )
    supp_plot = plot_eval_gate_supplementary_heatmaps(
        df,
        plots_dir,
        fname="heatmap_auxiliary_analysis_by_regime.png",
    )
    stat_plot = plot_eval_stat_screen_heatmaps(
        df,
        plots_dir,
        fname="heatmap_stat_screen_by_regime.png",
    )

    print(f"[ok] rows={len(df)}")
    print(f"[ok] csv={csv_path}")
    if threshold_plot is not None:
        print(f"[ok] twin_threshold_plot={threshold_plot}")
    if legacy_threshold_plot is not None:
        print(f"[ok] legacy_threshold_plot={legacy_threshold_plot}")
    if main_plot is not None:
        print(f"[ok] twin_difference_plot={main_plot}")
    if supp_plot is not None:
        print(f"[ok] auxiliary_plot={supp_plot}")
    if stat_plot is not None:
        print(f"[ok] stat_screen_plot={stat_plot}")


if __name__ == "__main__":
    main()
