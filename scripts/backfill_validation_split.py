#!/usr/bin/env python3
"""Backfill the twin/full/stat-screen validation split into existing V3 runs.

The canonical FULLSQUARE methodology splits the main twin status (G1..G5) from
the auxiliary statistical screen (G6). Runs produced before the split was
ported into this branch carry only a conservative `validation_status` where G6
vetoes everything. This script enriches their tables in place (originals are
backed up as *.pre_split_backup):

  tables/summary_by_regime.csv
    + stat_screen_pass, validation_status_twin, validation_status_full
    ~ validation_status rewritten to the twin semantics (G1..G5)
  tables/protocol_leaderboard.csv
    ~ n_pass/n_fail/n_partial + gate_pass_ratio recomputed under twin semantics
    + n_full_pass/n_full_fail/n_full_partial, all_regimes_full_passed,
      stat_screen_pass, gate_pass_ratio_full

Usage:
  python scripts/backfill_validation_split.py <exp_dir> [<exp_dir> ...]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "pass"):
        return True
    if s in ("false", "0", "no", "fail"):
        return False
    return None  # empty / nan / not computed


def _status(row: pd.Series, n_gates: int) -> str:
    states = [_safe_bool(row.get(f"gate_g{i}")) for i in range(1, n_gates + 1)]
    if any(s is False for s in states):
        return "fail"
    if all(s is True for s in states):
        return "pass"
    return "partial"


def _gate_pass_ratio(df: pd.DataFrame, n_gates: int) -> float:
    total = passed = 0
    for i in range(1, n_gates + 1):
        for raw in df[f"gate_g{i}"].tolist():
            s = _safe_bool(raw)
            if s is None:
                continue
            total += 1
            passed += int(s)
    return float(passed / total) if total else float("nan")


def backfill(exp_dir: Path) -> None:
    summary_path = exp_dir / "tables" / "summary_by_regime.csv"
    lb_path = exp_dir / "tables" / "protocol_leaderboard.csv"
    if not summary_path.exists():
        print(f"  SKIP (no summary): {exp_dir}")
        return

    df = pd.read_csv(summary_path)
    before = df.get("validation_status", pd.Series(dtype=str)).astype(str)

    df["stat_screen_pass"] = df.get("gate_g6", np.nan)
    df["validation_status_twin"] = df.apply(lambda r: _status(r, 5), axis=1)
    df["validation_status_full"] = df.apply(lambda r: _status(r, 6), axis=1)
    df["validation_status"] = df["validation_status_twin"]

    backup = summary_path.with_suffix(".csv.pre_split_backup")
    if not backup.exists():
        shutil.copy2(summary_path, backup)
    df.to_csv(summary_path, index=False)

    twin = df["validation_status_twin"]
    full = df["validation_status_full"]
    print(f"  {exp_dir.name}: twin {int((twin == 'pass').sum())}/{len(df)} pass"
          f" | full {int((full == 'pass').sum())}/{len(df)} pass"
          f" | screen {int(sum(_safe_bool(v) is True for v in df['stat_screen_pass']))}/{len(df)}"
          f" (era: {int((before == 'pass').sum())}/{len(before)} pass no status legado)")

    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        if len(lb) >= 1:
            lb_backup = lb_path.with_suffix(".csv.pre_split_backup")
            if not lb_backup.exists():
                shutil.copy2(lb_path, lb_backup)
            lb.loc[0, "n_pass"] = int((twin == "pass").sum())
            lb.loc[0, "n_fail"] = int((twin == "fail").sum())
            lb.loc[0, "n_partial"] = int((twin == "partial").sum())
            lb.loc[0, "all_regimes_passed"] = bool(len(df) > 0 and (twin == "pass").all())
            lb["n_full_pass"] = int((full == "pass").sum())
            lb["n_full_fail"] = int((full == "fail").sum())
            lb["n_full_partial"] = int((full == "partial").sum())
            lb["all_regimes_full_passed"] = bool(len(df) > 0 and (full == "pass").all())
            lb["stat_screen_pass"] = int(sum(_safe_bool(v) is True for v in df["stat_screen_pass"]))
            lb.loc[0, "gate_pass_ratio"] = _gate_pass_ratio(df, 5)
            lb["gate_pass_ratio_full"] = _gate_pass_ratio(df, 6)
            lb.to_csv(lb_path, index=False)


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        backfill(Path(arg).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
