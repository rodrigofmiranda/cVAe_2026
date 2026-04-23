#!/usr/bin/env python3
"""Aggregate and compare multiple 16QAM per-regime evaluation batches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.runtime_env import (
    ensure_required_python_modules,
    ensure_writable_mpl_config_dir,
)

ensure_writable_mpl_config_dir()
ensure_required_python_modules(
    ("numpy", "pandas"),
    context="16QAM crossline comparison",
    allow_missing=False,
)

import pandas as pd


LOWER_IS_BETTER = {
    "abs_delta_evm_pct": "|ΔEVM| mean",
    "abs_delta_snr_db": "|ΔSNR| mean",
    "delta_psd_l2": "ΔPSD mean",
}

HIGHER_IS_BETTER = {
    "mi_aux_pred_bits": "MI_pred mean",
    "gmi_aux_pred_bits": "GMI_pred mean",
    "ngmi_aux_pred": "NGMI_pred mean",
    "air_aux_pred_bits": "AIR_pred mean",
}

SUMMARY_COLUMNS = [
    "candidate",
    "n_regimes",
    "mean_abs_delta_evm_pct",
    "mean_abs_delta_snr_db",
    "mean_delta_psd_l2",
    "mean_mi_aux_pred_bits",
    "mean_gmi_aux_pred_bits",
    "mean_ngmi_aux_pred",
    "mean_air_aux_pred_bits",
    "mean_stat_mmd_qval",
    "mean_stat_energy_qval",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple 16QAM eval roots candidate by candidate."
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate specification in the form label=/path/to/eval_root .",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory where comparison tables and markdown will be written.",
    )
    parser.add_argument(
        "--title",
        default="16QAM Crossline Comparison",
        help="Title used in the generated markdown summary.",
    )
    return parser.parse_args()


def _parse_candidate_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid candidate spec: {spec!r}")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip()).expanduser().resolve()
    if not label:
        raise ValueError(f"Missing candidate label in spec: {spec!r}")
    if not path.exists():
        raise FileNotFoundError(f"Candidate eval root not found: {path}")
    return label, path


def _load_manifest(eval_root: Path) -> list[dict[str, Any]]:
    manifest_path = eval_root / "manifest_all_regimes_eval.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected manifest schema in {manifest_path}")
    return rows


def _metric_path(eval_root: Path, row: dict[str, Any]) -> Path:
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


def _safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return float(numeric.mean())
    return float("nan")


def _load_candidate_metrics(label: str, eval_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest_row in _load_manifest(eval_root):
        if str(manifest_row.get("status", "")).lower() != "completed":
            continue
        metric_path = _metric_path(eval_root, manifest_row)
        if not metric_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metric_path}")
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "candidate": label,
                "eval_root": str(eval_root),
                "regime_id": manifest_row.get("regime_id"),
                "distance_m": _coerce_float(manifest_row.get("distance_m")),
                "current_mA": _coerce_float(manifest_row.get("current_mA")),
                "abs_delta_evm_pct": abs(_coerce_float(metrics.get("delta_evm_%"))),
                "abs_delta_snr_db": abs(_coerce_float(metrics.get("delta_snr_db"))),
                "delta_psd_l2": _coerce_float(metrics.get("delta_psd_l2")),
                "stat_mmd_qval": _coerce_float(metrics.get("stat_mmd_qval")),
                "stat_energy_qval": _coerce_float(metrics.get("stat_energy_qval")),
                "mi_aux_pred_bits": _coerce_float(metrics.get("mi_aux_pred_bits")),
                "gmi_aux_pred_bits": _coerce_float(metrics.get("gmi_aux_pred_bits")),
                "ngmi_aux_pred": _coerce_float(metrics.get("ngmi_aux_pred")),
                "air_aux_pred_bits": _coerce_float(metrics.get("air_aux_pred_bits")),
                "metrics_path": str(metric_path),
            }
        )
    if not rows:
        raise RuntimeError(f"No completed regime metrics found under {eval_root}")
    return pd.DataFrame(rows).sort_values(["distance_m", "current_mA"]).reset_index(drop=True)


def _build_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, grp in long_df.groupby("candidate", sort=False):
        rows.append(
            {
                "candidate": candidate,
                "n_regimes": int(grp["regime_id"].nunique()),
                "mean_abs_delta_evm_pct": _safe_mean(grp["abs_delta_evm_pct"]),
                "mean_abs_delta_snr_db": _safe_mean(grp["abs_delta_snr_db"]),
                "mean_delta_psd_l2": _safe_mean(grp["delta_psd_l2"]),
                "mean_mi_aux_pred_bits": _safe_mean(grp["mi_aux_pred_bits"]),
                "mean_gmi_aux_pred_bits": _safe_mean(grp["gmi_aux_pred_bits"]),
                "mean_ngmi_aux_pred": _safe_mean(grp["ngmi_aux_pred"]),
                "mean_air_aux_pred_bits": _safe_mean(grp["air_aux_pred_bits"]),
                "mean_stat_mmd_qval": _safe_mean(grp["stat_mmd_qval"]),
                "mean_stat_energy_qval": _safe_mean(grp["stat_energy_qval"]),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _winners_for_metric(grp: pd.DataFrame, metric: str, higher_is_better: bool) -> list[str]:
    valid = grp[pd.to_numeric(grp[metric], errors="coerce").notna()].copy()
    if valid.empty:
        return []
    best = valid[metric].max() if higher_is_better else valid[metric].min()
    tol = 1e-12
    if higher_is_better:
        mask = (valid[metric] - best).abs() <= tol
    else:
        mask = (valid[metric] - best).abs() <= tol
    return sorted(valid.loc[mask, "candidate"].astype(str).unique().tolist())


def _build_win_counts(long_df: pd.DataFrame) -> pd.DataFrame:
    counts: dict[tuple[str, str], int] = {}
    for regime_id, grp in long_df.groupby("regime_id", sort=True):
        for metric in LOWER_IS_BETTER:
            for winner in _winners_for_metric(grp, metric, higher_is_better=False):
                counts[(winner, metric)] = counts.get((winner, metric), 0) + 1
        for metric in HIGHER_IS_BETTER:
            for winner in _winners_for_metric(grp, metric, higher_is_better=True):
                counts[(winner, metric)] = counts.get((winner, metric), 0) + 1
    rows: list[dict[str, Any]] = []
    all_metrics = list(LOWER_IS_BETTER) + list(HIGHER_IS_BETTER)
    for candidate in long_df["candidate"].drop_duplicates().tolist():
        row = {"candidate": candidate}
        for metric in all_metrics:
            row[metric] = counts.get((candidate, metric), 0)
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, float_fmt: str = ".3f") -> str:
    def _escape_md(text: str) -> str:
        return text.replace("|", "\\|")

    headers = [_escape_md(str(col)) for col in df.columns.tolist()]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        cells: list[str] = []
        for col in df.columns.tolist():
            value = row[col]
            if isinstance(value, float):
                if math.isfinite(value):
                    cells.append(_escape_md(format(value, float_fmt)))
                else:
                    cells.append("nan")
            else:
                cells.append(_escape_md(str(value)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _build_markdown(
    title: str,
    candidates: list[tuple[str, Path]],
    summary_df: pd.DataFrame,
    win_df: pd.DataFrame,
) -> str:
    label_map = {col: name for col, name in {**LOWER_IS_BETTER, **HIGHER_IS_BETTER}.items()}
    renamed_summary = summary_df.rename(
        columns={
            "mean_abs_delta_evm_pct": "|ΔEVM| mean",
            "mean_abs_delta_snr_db": "|ΔSNR| mean",
            "mean_delta_psd_l2": "ΔPSD mean",
            "mean_mi_aux_pred_bits": "MI_pred mean",
            "mean_gmi_aux_pred_bits": "GMI_pred mean",
            "mean_ngmi_aux_pred": "NGMI_pred mean",
            "mean_air_aux_pred_bits": "AIR_pred mean",
            "mean_stat_mmd_qval": "MMD q mean",
            "mean_stat_energy_qval": "Energy q mean",
            "n_regimes": "regimes",
        }
    )
    renamed_wins = win_df.rename(columns=label_map)

    lines = [f"# {title}", "", "## Candidates", ""]
    for label, path in candidates:
        lines.append(f"- `{label}`: `{path}`")
    lines.extend(
        [
            "",
            "## Mean Summary",
            "",
            _markdown_table(renamed_summary),
            "",
            "## Regime-Win Counts",
            "",
            _markdown_table(renamed_wins, float_fmt=".0f"),
            "",
            "## Reading Rule",
            "",
            "- lower is better for `|ΔEVM|`, `|ΔSNR|`, and `ΔPSD`",
            "- higher is better for `MI_pred`, `GMI_pred`, `NGMI_pred`, and `AIR_pred`",
            "- `MMD q mean` and `Energy q mean` remain auxiliary statistical readings, not the main twin-acceptance criterion",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    candidates = [_parse_candidate_spec(spec) for spec in args.candidate]
    if len(candidates) < 2:
        raise ValueError("Provide at least two candidates.")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    long_parts = [_load_candidate_metrics(label, eval_root) for label, eval_root in candidates]
    long_df = pd.concat(long_parts, ignore_index=True)
    long_df = long_df.sort_values(["regime_id", "candidate"]).reset_index(drop=True)

    summary_df = _build_summary(long_df)
    win_df = _build_win_counts(long_df)

    long_csv = out_dir / "crossline_regime_metrics_long.csv"
    summary_csv = out_dir / "crossline_candidate_summary.csv"
    wins_csv = out_dir / "crossline_win_counts.csv"
    metadata_json = out_dir / "crossline_candidates.json"
    markdown_path = out_dir / "README.md"

    long_df.to_csv(long_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    win_df.to_csv(wins_csv, index=False)
    metadata_json.write_text(
        json.dumps(
            {
                "title": args.title,
                "candidates": [{"label": label, "eval_root": str(path)} for label, path in candidates],
                "n_regimes": int(long_df["regime_id"].nunique()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _build_markdown(args.title, candidates, summary_df, win_df) + "\n",
        encoding="utf-8",
    )

    print(f"[ok] out_dir={out_dir}")
    print(f"[ok] long_csv={long_csv}")
    print(f"[ok] summary_csv={summary_csv}")
    print(f"[ok] wins_csv={wins_csv}")
    print(f"[ok] metadata_json={metadata_json}")
    print(f"[ok] markdown={markdown_path}")


if __name__ == "__main__":
    main()
