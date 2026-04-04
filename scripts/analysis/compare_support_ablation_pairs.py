#!/usr/bin/env python3
"""Compare support-ablation protocol runs in a controlled, paired way.

This helper is intentionally run-centric rather than preset-centric.
It compares protocol outputs from fixed-candidate runs and warns when an input
run still contains multiple grid candidates, because those runs only expose the
stage champion during protocol evaluation.

Examples:
    python scripts/analysis/compare_support_ablation_pairs.py \
      --run e0_seq=outputs/support_ablation/e0/exp_20260405_120000 \
      --run e1_seq=outputs/support_ablation/e1/exp_20260405_140000

    python scripts/analysis/compare_support_ablation_pairs.py \
      --run e0_delta=outputs/support_ablation/e0 \
      --run e2_delta=outputs/support_ablation/e2 \
      --output_dir outputs/support_ablation/paired_reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


SUMMARY_METRICS = [
    "best_grid_tag",
    "validation_status",
    "gate_g3",
    "gate_g5",
    "gate_g6",
    "delta_jb_stat_rel",
    "cvae_delta_skew_l2",
    "cvae_delta_kurt_l2",
    "stat_mmd_qval",
    "stat_energy_qval",
]

SUPPORT_REGION_METRICS = [
    "n_samples_real",
    "n_samples_pred",
    "delta_wasserstein_I",
    "delta_wasserstein_Q",
    "delta_jb_stat_rel_I",
    "delta_jb_stat_rel_Q",
    "stat_mmd_qval",
    "stat_energy_qval",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare support-ablation protocol runs with paired, family-safe reporting."
    )
    p.add_argument(
        "--run",
        action="append",
        required=True,
        help="Label and run path in the form label=/path/to/exp_run_or_stage_dir",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write the comparison CSVs (default: ./support_ablation_pairwise)",
    )
    return p.parse_args()


def _parse_run_specs(specs: Iterable[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --run specification {spec!r}; expected label=/path/to/run."
            )
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid --run specification {spec!r}; label is empty.")
        out.append((label, Path(raw_path).expanduser().resolve()))
    return out


def _resolve_protocol_run_dir(path: Path) -> Path:
    if (path / "tables" / "summary_by_regime.csv").exists():
        return path

    exp_dirs = sorted(
        [
            p for p in path.glob("exp_*")
            if (p / "tables" / "summary_by_regime.csv").exists()
        ]
    )
    if exp_dirs:
        return exp_dirs[-1]

    raise FileNotFoundError(
        f"Could not resolve a protocol run dir from {path}. "
        "Expected a run dir with tables/summary_by_regime.csv or a stage dir containing exp_* runs."
    )


def _read_optional_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run_bundle(label: str, raw_path: Path) -> Dict[str, object]:
    run_dir = _resolve_protocol_run_dir(raw_path)
    tables_dir = run_dir / "tables"
    summary_path = tables_dir / "summary_by_regime.csv"
    support_path = tables_dir / "residual_signature_by_support_bin.csv"
    leaderboard_path = tables_dir / "protocol_leaderboard.csv"
    manifest_path = run_dir / "manifest.json"

    summary = pd.read_csv(summary_path)
    support = pd.read_csv(support_path) if support_path.exists() else pd.DataFrame()
    leaderboard = pd.read_csv(leaderboard_path) if leaderboard_path.exists() else pd.DataFrame()
    manifest = _read_optional_json(manifest_path)

    n_candidates = int(len(leaderboard)) if not leaderboard.empty else 0
    winner_best_grid_tag = (
        str(leaderboard.iloc[0]["best_grid_tag"])
        if not leaderboard.empty and "best_grid_tag" in leaderboard.columns
        else ""
    )
    base_overrides = dict(manifest.get("base_overrides", {}) or {})
    seed = base_overrides.get("seed")
    controlled_candidate_run = bool(n_candidates == 1)

    summary = summary.copy()
    summary.insert(0, "run_label", label)
    summary.insert(1, "resolved_run_dir", str(run_dir))
    summary.insert(2, "seed", seed)

    if not support.empty:
        support = support.copy()
        support.insert(0, "run_label", label)
        support.insert(1, "resolved_run_dir", str(run_dir))
        support.insert(2, "seed", seed)

    return {
        "label": label,
        "raw_path": raw_path,
        "run_dir": run_dir,
        "summary": summary,
        "support": support,
        "leaderboard": leaderboard,
        "winner_best_grid_tag": winner_best_grid_tag,
        "seed": seed,
        "n_candidates": n_candidates,
        "controlled_candidate_run": controlled_candidate_run,
    }


def _build_summary_pivot(bundles: List[Dict[str, object]]) -> pd.DataFrame:
    base = None
    for bundle in bundles:
        label = str(bundle["label"])
        summary = bundle["summary"]
        keep_cols = [
            "study",
            "regime_id",
            "regime_label",
            "dist_target_m",
            "curr_target_mA",
            *[c for c in SUMMARY_METRICS if c in summary.columns],
        ]
        sub = summary[keep_cols].copy()
        rename_map = {col: f"{label}::{col}" for col in SUMMARY_METRICS if col in sub.columns}
        sub = sub.rename(columns=rename_map)
        if base is None:
            base = sub
        else:
            base = base.merge(
                sub,
                on=["study", "regime_id", "regime_label", "dist_target_m", "curr_target_mA"],
                how="outer",
            )
    if base is None:
        return pd.DataFrame()
    return base.sort_values(["study", "dist_target_m", "curr_target_mA", "regime_id"]).reset_index(drop=True)


def _build_support_region_long(bundles: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        support = bundle["support"]
        if support.empty:
            continue
        sub = support.copy()
        if "support_region" not in sub.columns:
            continue
        if "support_axis" in sub.columns:
            sub = sub[sub["support_axis"].astype(str).str.lower() == "region"]
        sub = sub[sub["support_region"].astype(str).str.lower().isin({"edge", "corner"})]
        if sub.empty:
            continue
        keep_cols = [
            "run_label",
            "resolved_run_dir",
            "seed",
            "study",
            "regime_id",
            "regime_label",
            "dist_target_m",
            "curr_target_mA",
            "support_region",
            *[c for c in SUPPORT_REGION_METRICS if c in sub.columns],
        ]
        rows.append(sub[keep_cols])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["support_region", "dist_target_m", "curr_target_mA", "run_label"]
    )


def _build_support_region_pivot(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame()
    base_cols = [
        "study",
        "regime_id",
        "regime_label",
        "dist_target_m",
        "curr_target_mA",
        "support_region",
    ]
    base = df_long[base_cols].drop_duplicates().sort_values(base_cols)
    for label in sorted(df_long["run_label"].unique()):
        sub = df_long[df_long["run_label"] == label].copy()
        keep = base_cols + [c for c in SUPPORT_REGION_METRICS if c in sub.columns]
        sub = sub[keep]
        rename_map = {
            col: f"{label}::{col}"
            for col in SUPPORT_REGION_METRICS
            if col in sub.columns
        }
        base = base.merge(
            sub.rename(columns=rename_map),
            on=base_cols,
            how="left",
        )
    return base.reset_index(drop=True)


def _build_run_manifest(bundles: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        rows.append(
            {
                "run_label": bundle["label"],
                "input_path": str(bundle["raw_path"]),
                "resolved_run_dir": str(bundle["run_dir"]),
                "seed": bundle["seed"],
                "winner_best_grid_tag": bundle["winner_best_grid_tag"],
                "n_candidates": bundle["n_candidates"],
                "controlled_candidate_run": bool(bundle["controlled_candidate_run"]),
                "warning": (
                    ""
                    if bundle["controlled_candidate_run"]
                    else "run contains multiple grid candidates; protocol summary reflects only the champion"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    run_specs = _parse_run_specs(args.run)
    bundles = [_load_run_bundle(label, path) for label, path in run_specs]

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path.cwd() / "support_ablation_pairwise").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df_manifest = _build_run_manifest(bundles)
    df_summary_pivot = _build_summary_pivot(bundles)
    df_support_long = _build_support_region_long(bundles)
    df_support_pivot = _build_support_region_pivot(df_support_long)

    manifest_path = output_dir / "support_ablation_run_manifest.csv"
    summary_path = output_dir / "support_ablation_paired_summary_by_regime.csv"
    support_long_path = output_dir / "support_ablation_support_regions_long.csv"
    support_pivot_path = output_dir / "support_ablation_support_regions_pivot.csv"

    df_manifest.to_csv(manifest_path, index=False)
    df_summary_pivot.to_csv(summary_path, index=False)
    df_support_long.to_csv(support_long_path, index=False)
    df_support_pivot.to_csv(support_pivot_path, index=False)

    print(f"✓ Run manifest:   {manifest_path}")
    print(f"✓ Regime pivot:   {summary_path}")
    print(f"✓ Support (long): {support_long_path}")
    print(f"✓ Support pivot:  {support_pivot_path}")

    flagged = df_manifest[~df_manifest["controlled_candidate_run"]]
    if not flagged.empty:
        print("\n⚠️  Warning: some inputs are mixed-candidate sweeps.")
        for _, row in flagged.iterrows():
            print(
                f"  - {row['run_label']}: n_candidates={row['n_candidates']} | "
                f"winner_best_grid_tag={row['winner_best_grid_tag']}"
            )
        print("    Use --grid_tag <tag> --max_grids 1 for controlled paired comparisons.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
