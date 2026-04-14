#!/usr/bin/env python3
"""Empirical calibration helper for digital-twin gate thresholds.

This script builds an exploratory calibration table for the current twin gates.
It uses a curated positive reference pool (historically strong runs) and a
contrast negative pool (weaker/full-data regressions), then compares:

- current thresholds
- empirical envelopes of the positive pool
- candidate tightened thresholds for G1..G5

G6 is reported, but not "calibrated" the same way: alpha/q is kept as a
statistical decision level rather than an engineering threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.runtime_env import ensure_required_python_modules

ensure_required_python_modules(
    ("numpy", "pandas"),
    context="gate threshold calibration",
    allow_missing=False,
)

import numpy as np
import pandas as pd

from src.evaluation.validation_summary import TWIN_GATE_THRESHOLDS

POSITIVE_DEFAULTS = [
    "/home/rodrigo/cVAe_2026/outputs/exp_20260328_153611/tables/summary_by_regime.csv",
    "/home/rodrigo/cVAe_2026/outputs/exp_20260327_161311/tables/summary_by_regime.csv",
    "/home/rodrigo/cVAe_2026/outputs/exp_20260327_225213/tables/summary_by_regime.csv",
    "/home/rodrigo/cVAe_2026/outputs/_recovered_vlc_backup_20260402/exp_20260324_023558/tables/summary_by_regime.csv",
]

NEGATIVE_DEFAULTS = [
    "/home/rodrigo/cVAe_2026/outputs/exp_20260328_023430/tables/summary_by_regime.csv",
    "/home/rodrigo/cVAe_2026_shape/outputs/support_ablation/final_full/e2_edge_s27/exp_20260407_211428/tables/summary_by_regime.csv",
    "/home/rodrigo/cVAe_2026_shape/outputs/support_ablation/final_full_clean/e3c_covsoft_s27/exp_20260408_005942/tables/summary_by_regime.csv",
]


@dataclass(frozen=True)
class MetricSpec:
    label: str
    col: str
    direction: str  # "lower" or "higher"
    threshold_key: str
    gate: str
    calibrate: bool = True


METRICS: List[MetricSpec] = [
    MetricSpec("G1 evm", "cvae_rel_evm_error", "lower", "rel_evm_error", "G1"),
    MetricSpec("G2 snr", "cvae_rel_snr_error", "lower", "rel_snr_error", "G2"),
    MetricSpec("G3 mean_sigma", "cvae_mean_rel_sigma", "lower", "mean_rel_sigma", "G3"),
    MetricSpec("G3 cov_var", "cvae_cov_rel_var", "lower", "cov_rel_var", "G3"),
    MetricSpec("G4 psd", "cvae_psd_l2", "lower", "delta_psd_l2", "G4"),
    MetricSpec("G5 skew", "cvae_delta_skew_l2", "lower", "delta_skew_l2", "G5"),
    MetricSpec("G5 kurt", "cvae_delta_kurt_l2", "lower", "delta_kurt_l2", "G5"),
    MetricSpec("G5 jb_rel", "delta_jb_stat_rel", "lower", "delta_jb_stat_rel", "G5"),
    MetricSpec("G6 mmd_q", "stat_mmd_qval", "higher", "stat_qval", "G6", calibrate=False),
    MetricSpec("G6 energy_q", "stat_energy_qval", "higher", "stat_qval", "G6", calibrate=False),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Empirically calibrate twin gate thresholds.")
    p.add_argument("--output-dir", type=str, default=None, help="Output dir for calibration artifacts.")
    p.add_argument("--positive", nargs="*", default=POSITIVE_DEFAULTS, help="Positive reference summary_by_regime.csv files.")
    p.add_argument("--negative", nargs="*", default=NEGATIVE_DEFAULTS, help="Negative contrast summary_by_regime.csv files.")
    return p.parse_args()


def _load_pool(paths: Iterable[str], keep_pass: bool, pool_name: str) -> pd.DataFrame:
    rows = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            continue
        df = pd.read_csv(path)
        status_col = "validation_status_twin" if "validation_status_twin" in df.columns else "validation_status"
        if status_col not in df.columns:
            continue
        if keep_pass:
            df = df[df[status_col] == "pass"].copy()
        else:
            df = df[df[status_col] != "pass"].copy()
        if df.empty:
            continue
        df.insert(0, "pool_name", pool_name)
        df.insert(1, "source_csv", str(path))
        df.insert(2, "source_run", path.parents[1].name)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _round_up_sig(x: float, sig: int = 2) -> float:
    if not np.isfinite(x) or x <= 0:
        return float("nan")
    power = math.floor(math.log10(abs(x)))
    factor = 10 ** (sig - 1 - power)
    return math.ceil(x * factor) / factor


def _round_down_sig(x: float, sig: int = 2) -> float:
    if not np.isfinite(x) or x <= 0:
        return float("nan")
    power = math.floor(math.log10(abs(x)))
    factor = 10 ** (sig - 1 - power)
    return math.floor(x * factor) / factor


def _pass_rate(values: np.ndarray, threshold: float, direction: str) -> float:
    if len(values) == 0 or not np.isfinite(threshold):
        return float("nan")
    if direction == "lower":
        return float(np.mean(values < threshold))
    return float(np.mean(values > threshold))


def build_table(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for spec in METRICS:
        cur = float(TWIN_GATE_THRESHOLDS[spec.threshold_key])
        pvals = pd.to_numeric(pos.get(spec.col), errors="coerce").dropna().to_numpy(dtype=float)
        nvals = pd.to_numeric(neg.get(spec.col), errors="coerce").dropna().to_numpy(dtype=float)
        if len(pvals) == 0:
            continue

        if spec.direction == "lower":
            q90 = float(np.quantile(pvals, 0.90))
            q95 = float(np.quantile(pvals, 0.95))
            pmax = float(np.max(pvals))
            n25 = float(np.quantile(nvals, 0.25)) if len(nvals) else float("nan")
            # operational proposal: allow the full positive envelope plus 10% slack
            proposal = _round_up_sig(pmax * 1.10, sig=2) if spec.calibrate else cur
            proposal_kind = "empirical_envelope_110pct" if spec.calibrate else "keep_alpha"
        else:
            q90 = float(np.quantile(pvals, 0.10))
            q95 = float(np.quantile(pvals, 0.05))
            pmax = float(np.min(pvals))
            n25 = float(np.quantile(nvals, 0.75)) if len(nvals) else float("nan")
            proposal = _round_down_sig(pmax * 0.90, sig=2) if spec.calibrate else cur
            proposal_kind = "empirical_floor_90pct" if spec.calibrate else "keep_alpha"

        rows.append(
            {
                "gate": spec.gate,
                "metric": spec.label,
                "column": spec.col,
                "direction": spec.direction,
                "current_threshold": cur,
                "positive_n": int(len(pvals)),
                "negative_n": int(len(nvals)),
                "positive_q90_or_q10": q90,
                "positive_q95_or_q05": q95,
                "positive_max_or_min": pmax,
                "negative_q25_or_q75": n25,
                "proposal": proposal,
                "proposal_kind": proposal_kind,
                "current_positive_pass_rate": _pass_rate(pvals, cur, spec.direction),
                "proposal_positive_pass_rate": _pass_rate(pvals, proposal, spec.direction),
                "current_negative_pass_rate": _pass_rate(nvals, cur, spec.direction),
                "proposal_negative_pass_rate": _pass_rate(nvals, proposal, spec.direction),
                "calibrate": bool(spec.calibrate),
            }
        )
    return pd.DataFrame(rows)


def _markdown_report(df: pd.DataFrame, out_dir: Path, pos: pd.DataFrame, neg: pd.DataFrame, args: argparse.Namespace) -> None:
    lines: List[str] = []
    lines.append("# Gate Threshold Calibration")
    lines.append("")
    lines.append("Date: 2026-04-11")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("This is an exploratory empirical calibration based on a curated historical pool.")
    lines.append("")
    lines.append("- positive pool: pass-regime rows from historically strong reference runs")
    lines.append("- negative pool: non-pass rows from weaker or regressed full-data runs")
    lines.append("- lower-better metrics (`G1..G5`) get a proposal based on the positive envelope plus 10% slack")
    lines.append("- `G6` is reported but not re-calibrated as an engineering threshold; `q=0.05` is kept as the decision alpha")
    lines.append("")
    lines.append("Caveat:")
    lines.append("")
    lines.append("- this is a retrospective calibration, not a fully independent validation study")
    lines.append("- it should be used to tighten and justify thresholds, not as a standalone proof")
    lines.append("")
    lines.append("## Pool Sizes")
    lines.append("")
    lines.append(f"- positive rows: `{len(pos)}`")
    lines.append(f"- negative rows: `{len(neg)}`")
    lines.append("")
    lines.append("## Proposed Table")
    lines.append("")
    lines.append("| Gate | Metric | Current | Positive envelope | Proposal | Positive pass rate now | Positive pass rate proposal | Negative pass rate now | Negative pass rate proposal |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in df.iterrows():
        env = row["positive_max_or_min"]
        lines.append(
            f"| `{row['gate']}` | `{row['metric']}` | "
            f"`{row['current_threshold']:.4f}` | "
            f"`{env:.4f}` | "
            f"`{row['proposal']:.4f}` | "
            f"`{row['current_positive_pass_rate']:.3f}` | "
            f"`{row['proposal_positive_pass_rate']:.3f}` | "
            f"`{row['current_negative_pass_rate']:.3f}` | "
            f"`{row['proposal_negative_pass_rate']:.3f}` |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("- Very loose current thresholds should be tightened if historically strong twins already sit far below them.")
    lines.append("- Metrics with strong overlap between positive and negative pools should stay in a multi-gate system; they are not reliable as single-metric vetoes.")
    lines.append("- `G6` should be standardized by test budget and interpretation, not by moving alpha away from `0.05`.")
    lines.append("")
    lines.append("## Source CSVs")
    lines.append("")
    for raw in args.positive:
        lines.append(f"- positive: `{Path(raw).expanduser().resolve()}`")
    for raw in args.negative:
        lines.append(f"- negative: `{Path(raw).expanduser().resolve()}`")
    (out_dir / "gate_threshold_calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir or (ROOT / "outputs" / "analysis" / "gate_threshold_calibration_20260411")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pos = _load_pool(args.positive, keep_pass=True, pool_name="positive")
    neg = _load_pool(args.negative, keep_pass=False, pool_name="negative")
    if pos.empty:
        raise SystemExit("No positive calibration rows found.")
    if neg.empty:
        raise SystemExit("No negative contrast rows found.")

    table = build_table(pos, neg)
    table.to_csv(out_dir / "gate_threshold_calibration_table.csv", index=False)
    pos.to_csv(out_dir / "positive_pool_rows.csv", index=False)
    neg.to_csv(out_dir / "negative_pool_rows.csv", index=False)
    _markdown_report(table, out_dir, pos, neg, args)

    summary = {
        "output_dir": str(out_dir),
        "positive_rows": int(len(pos)),
        "negative_rows": int(len(neg)),
        "proposals": {
            row["column"]: {
                "gate": row["gate"],
                "current_threshold": float(row["current_threshold"]),
                "proposal": float(row["proposal"]),
                "proposal_kind": row["proposal_kind"],
                "calibrate": bool(row["calibrate"]),
            }
            for _, row in table.iterrows()
        },
    }
    (out_dir / "gate_threshold_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"output_dir: {out_dir}")
    print(f"positive_rows: {len(pos)}")
    print(f"negative_rows: {len(neg)}")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
