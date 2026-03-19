#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""Backfill validation gates for existing experiment outputs.

Reads existing ``tables/summary_by_regime.csv`` files, recomputes the derived
validation columns using the current gate logic, and writes sidecar artifacts
without overwriting the historical originals by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.validation_summary import (
    build_stat_acceptance_summary,
    build_stat_fidelity_table,
    recompute_validation_summary,
)


def _write_table(df: pd.DataFrame, csv_path: Path, xlsx_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception:
        pass


def _recover_one(exp_dir: Path, *, overwrite: bool = False) -> Dict[str, object]:
    tables_dir = exp_dir / "tables"
    src_csv = tables_dir / "summary_by_regime.csv"
    if not src_csv.exists():
        return {"run": exp_dir.name, "status": "skip", "reason": "missing_summary"}

    original = pd.read_csv(src_csv)
    updated = recompute_validation_summary(original)

    if overwrite:
        out_csv = src_csv
        out_xlsx = tables_dir / "summary_by_regime.xlsx"
        sf_csv = tables_dir / "stat_fidelity_by_regime.csv"
        sf_xlsx = tables_dir / "stat_fidelity_by_regime.xlsx"
        meta_json = tables_dir / "validation_recompute.json"
    else:
        out_csv = tables_dir / "summary_by_regime_recomputed.csv"
        out_xlsx = tables_dir / "summary_by_regime_recomputed.xlsx"
        sf_csv = tables_dir / "stat_fidelity_by_regime_recomputed.csv"
        sf_xlsx = tables_dir / "stat_fidelity_by_regime_recomputed.xlsx"
        meta_json = tables_dir / "validation_recompute.json"

    _write_table(updated, out_csv, out_xlsx)

    df_sf = build_stat_fidelity_table(updated)
    if df_sf is not None and not df_sf.empty:
        _write_table(df_sf, sf_csv, sf_xlsx)

    acceptance = build_stat_acceptance_summary(updated)
    meta = {
        "run": exp_dir.name,
        "source_summary_csv": str(src_csv),
        "output_summary_csv": str(out_csv),
        "rows": int(len(updated)),
        "validation_status_counts": {
            str(k): int(v)
            for k, v in updated["validation_status"].fillna("partial").value_counts(dropna=False).items()
        },
        "has_stat_fidelity_table": bool(df_sf is not None and not df_sf.empty),
        "acceptance": acceptance,
    }
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"run": exp_dir.name, "status": "ok", "rows": int(len(updated)), "meta": meta}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "targets",
        nargs="*",
        help="Experiment directories or summary CSV files. Default: all outputs/exp_* runs.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite canonical summary files.")
    args = ap.parse_args()

    if args.targets:
        exp_dirs: List[Path] = []
        for raw in args.targets:
            p = Path(raw)
            if p.is_file() and p.name == "summary_by_regime.csv":
                exp_dirs.append(p.parent.parent)
            else:
                exp_dirs.append(p)
    else:
        exp_dirs = sorted(Path("/workspace/2026/outputs").glob("exp_*"))

    results = []
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            results.append({"run": str(exp_dir), "status": "skip", "reason": "missing_path"})
            continue
        results.append(_recover_one(exp_dir, overwrite=bool(args.overwrite)))

    ok = [r for r in results if r.get("status") == "ok"]
    skip = [r for r in results if r.get("status") != "ok"]
    print(f"Recovered: {len(ok)} runs")
    print(f"Skipped:   {len(skip)} runs")
    for item in ok[:20]:
        print(f" - {item['run']}: rows={item['rows']}")
    for item in skip[:20]:
        print(f" - {item['run']}: {item.get('reason', item['status'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
