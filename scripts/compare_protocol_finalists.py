#!/usr/bin/env python3
"""Compare selected grid finalists across regimes inside a protocol run.

Usage:
    python scripts/compare_protocol_finalists.py outputs/exp_YYYYMMDD_HHMMSS \
      --tags TAG_A TAG_B
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare candidate grid tags across protocol regimes."
    )
    p.add_argument("run_dir", type=str, help="Protocol run dir (outputs/exp_...)")
    p.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help="Grid tags to compare. Use exactly 2 for the main pivot report.",
    )
    return p.parse_args()


def _collect_rows(run_dir: Path, tags: List[str]) -> pd.DataFrame:
    rows = []
    patterns = [
        "studies/*/regimes/*/tables/gridsearch_results.csv",  # legacy
        "eval/*/tables/gridsearch_results.csv",               # flat single-study
        "eval/*/*/tables/gridsearch_results.csv",             # flat multi-study
    ]
    csv_paths = []
    for pattern in patterns:
        csv_paths.extend(run_dir.glob(pattern))

    for csv_path in sorted({p.resolve() for p in csv_paths}):
        rel = csv_path.relative_to(run_dir)
        parts = rel.parts
        if len(parts) >= 5 and parts[0] == "studies":
            study = parts[1]
            regime_id = parts[3]
        elif len(parts) == 4 and parts[0] == "eval":
            study = "within_regime"
            regime_id = parts[1]
        elif len(parts) >= 5 and parts[0] == "eval":
            study = parts[1]
            regime_id = parts[2]
        else:
            continue
        df = pd.read_csv(csv_path)
        if "tag" not in df.columns:
            continue
        sub = df[df["tag"].isin(tags)].copy()
        if sub.empty:
            continue
        sub.insert(0, "regime_id", regime_id)
        sub.insert(0, "study", study)
        sub.insert(2, "gridsearch_csv", str(csv_path))
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    cols = [
        "study",
        "regime_id",
        "tag",
        "rank",
        "status",
        "best_epoch",
        "train_time_s",
        "score_v2",
        "delta_evm_%",
        "delta_snr_db",
        "delta_cov_fro",
        "delta_kurt_l2",
        "var_real_delta",
        "var_pred_delta",
        "active_dims",
        "kl_mean_total",
        "gridsearch_csv",
    ]
    keep = [c for c in cols if c in out.columns]
    return out[keep].sort_values(["study", "regime_id", "score_v2", "rank"])


def _build_pivot(rows: pd.DataFrame, tags: List[str]) -> pd.DataFrame:
    metrics = [
        "score_v2",
        "delta_evm_%",
        "delta_snr_db",
        "delta_cov_fro",
        "delta_kurt_l2",
        "var_pred_delta",
        "active_dims",
        "kl_mean_total",
    ]
    base = rows[["study", "regime_id"]].drop_duplicates().sort_values(["study", "regime_id"])
    for tag in tags:
        sub = rows[rows["tag"] == tag].copy()
        sub = sub[["study", "regime_id", *metrics]].rename(
            columns={m: f"{tag}::{m}" for m in metrics}
        )
        base = base.merge(sub, on=["study", "regime_id"], how="left")

    if len(tags) == 2:
        score_a = pd.to_numeric(base[f"{tags[0]}::score_v2"], errors="coerce")
        score_b = pd.to_numeric(base[f"{tags[1]}::score_v2"], errors="coerce")
        winner = []
        for a, b in zip(score_a, score_b):
            if pd.isna(a) and pd.isna(b):
                winner.append("")
            elif pd.isna(b):
                winner.append(tags[0])
            elif pd.isna(a):
                winner.append(tags[1])
            else:
                winner.append(tags[0] if a <= b else tags[1])
        base["winner_by_score_v2"] = winner
    return base


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    tags = list(args.tags)

    rows = _collect_rows(run_dir, tags)
    if rows.empty:
        print("❌ Nenhum gridsearch_results.csv compatível com as tags fornecidas.")
        return 1

    pivot = _build_pivot(rows, tags)
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows_path = tables_dir / "finalist_grid_rows_by_regime.csv"
    pivot_path = tables_dir / "finalist_grid_comparison_by_regime.csv"
    rows.to_csv(rows_path, index=False)
    pivot.to_csv(pivot_path, index=False)

    print(f"✓ Rows saved:  {rows_path}")
    print(f"✓ Pivot saved: {pivot_path}")

    if len(tags) == 2 and "winner_by_score_v2" in pivot.columns:
        counts = pivot["winner_by_score_v2"].value_counts(dropna=False)
        print("\nWins by score_v2:")
        for tag in tags:
            print(f"  {tag}: {int(counts.get(tag, 0))}")

    print("\nPreview:")
    preview_cols = [c for c in ["study", "regime_id", "winner_by_score_v2"] if c in pivot.columns]
    print(pivot[preview_cols].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
