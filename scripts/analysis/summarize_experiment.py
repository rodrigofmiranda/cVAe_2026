#!/usr/bin/env python3
"""Compact experiment summarizer for protocol-first cVAE runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.runtime_env import ensure_required_python_modules

ensure_required_python_modules(
    ("pandas",),
    context="experiment summarizer",
    allow_missing=False,
)

import pandas as pd

from src.protocol.experiment_tracking import latest_complete_protocol_experiment

def _latest_experiment_any(outputs_dir: Path) -> Path:
    candidates = sorted(
        [p for p in outputs_dir.glob("exp_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No experiment found under {outputs_dir}")
    return candidates[-1]


def _latest_experiment(outputs_dir: Path) -> Path:
    try:
        return latest_complete_protocol_experiment(outputs_dir)
    except FileNotFoundError:
        return _latest_experiment_any(outputs_dir)


def _pick_exp(path_str: Optional[str]) -> Path:
    if path_str:
        p = Path(path_str).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Experiment path not found: {p}")
        return p
    return _latest_experiment(Path("outputs").resolve())


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v) -> str:
    if pd.isna(v):
        return "nan"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _print_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _summarize_manifest(exp_dir: Path) -> None:
    manifest = _read_json(exp_dir / "manifest.json")
    if manifest is None:
        print("manifest: missing")
        return
    print(f"manifest: ok")
    print(f"git_commit: {manifest.get('git_commit', 'n/a')}")
    print(f"execution_mode: {manifest.get('execution_mode', 'n/a')}")
    print(f"n_studies: {manifest.get('n_studies', 'n/a')}")
    print(f"n_regimes: {manifest.get('n_regimes', 'n/a')}")
    if manifest.get("shared_model_run_dir"):
        print(f"shared_model_run_dir: {manifest['shared_model_run_dir']}")
    acceptance = manifest.get("stat_acceptance") or {}
    if acceptance:
        verdict = acceptance.get("verdict", "n/a")
        both = acceptance.get("both_q_gt_alpha", "n/a")
        psd = acceptance.get("psd_within_ratio", "n/a")
        print(f"stat_acceptance: verdict={verdict} both_q_gt_alpha={both} psd_within_ratio={psd}")


def _summarize_training(exp_dir: Path) -> None:
    state = _read_json(exp_dir / "train" / "state_run.json")
    grids = _read_csv(exp_dir / "train" / "tables" / "gridsearch_results.csv")
    diag = _read_csv(exp_dir / "train" / "tables" / "grid_training_diagnostics.csv")

    _print_header("Training")
    if state is None and grids is None:
        print("training: missing")
        return

    if state is not None:
        print(f"best_grid_tag: {state.get('best_grid_tag', 'n/a')}")
        print(f"best_score_v2: {_fmt(state.get('best_score_v2'))}")

    if grids is not None and len(grids):
        top = grids.sort_values(["rank", "score_v2"], ascending=[True, True]).head(5)
        print("top grids:")
        for _, row in top.iterrows():
            print(
                "  "
                f"rank={_fmt(row.get('rank'))} "
                f"tag={row.get('tag')} "
                f"arch={row.get('arch_variant')} "
                f"score_v2={_fmt(row.get('score_v2'))} "
                f"delta_psd_l2={_fmt(row.get('delta_psd_l2'))} "
                f"delta_acf_l2={_fmt(row.get('delta_acf_l2'))}"
            )

    if diag is not None and len(diag):
        top = diag.sort_values(["rank", "score_v2"], ascending=[True, True]).head(5)
        print("training diagnostics:")
        for _, row in top.iterrows():
            flags = []
            for flag_col, short in [
                ("flag_posterior_collapse", "collapse"),
                ("flag_undertrained", "under"),
                ("flag_overfit", "overfit"),
                ("flag_unstable", "unstable"),
                ("flag_lr_floor", "lr_floor"),
            ]:
                if bool(row.get(flag_col, False)):
                    flags.append(short)
            flag_text = ",".join(flags) if flags else "none"
            print(
                "  "
                f"rank={_fmt(row.get('rank'))} "
                f"tag={row.get('tag')} "
                f"best_epoch_ratio={_fmt(row.get('best_epoch_ratio'))} "
                f"active_dim_ratio={_fmt(row.get('active_dim_ratio'))} "
                f"lr_drops={_fmt(row.get('lr_drop_count'))} "
                f"flags={flag_text}"
            )


def _summarize_protocol(exp_dir: Path) -> None:
    summary = _read_csv(exp_dir / "tables" / "summary_by_regime.csv")
    leaderboard = _read_csv(exp_dir / "tables" / "protocol_leaderboard.csv")

    _print_header("Protocol")
    if summary is None:
        print("summary_by_regime.csv: missing")
        return

    print(f"summary_rows: {len(summary)}")
    if "eval_status" in summary.columns:
        counts = summary["eval_status"].fillna("nan").value_counts().to_dict()
        print(f"eval_status_counts: {counts}")

    print("twin_validation:")
    if "validation_status" in summary.columns:
        counts = summary["validation_status"].fillna("nan").value_counts().to_dict()
        print(f"  validation_status_twin_counts: {counts}")
    if "validation_status_full" in summary.columns:
        counts = summary["validation_status_full"].fillna("nan").value_counts().to_dict()
        print(f"  validation_status_full_counts: {counts}")
    if "stat_screen_pass" in summary.columns:
        ss = pd.Series(summary["stat_screen_pass"]).map(
            lambda v: "pass" if v is True else ("fail" if v is False else "partial")
        )
        print("auxiliary_analysis:")
        print(f"  stat_screen_counts: {ss.value_counts(dropna=False).to_dict()}")
        if "info_metrics_available" in summary.columns:
            info_counts = pd.Series(summary["info_metrics_available"]).map(
                lambda v: "available" if v is True else ("unavailable" if v is False else "nan")
            )
            print(f"  info_metrics_counts: {info_counts.value_counts(dropna=False).to_dict()}")
            if all(
                col in summary.columns
                for col in [
                    "mi_aux_real_bits",
                    "mi_aux_pred_bits",
                    "gmi_aux_real_bits",
                    "gmi_aux_pred_bits",
                    "ngmi_aux_real",
                    "ngmi_aux_pred",
                ]
            ):
                print(
                    "  info_means: "
                    f"MI real/pred={_fmt(pd.to_numeric(summary['mi_aux_real_bits'], errors='coerce').mean())}/"
                    f"{_fmt(pd.to_numeric(summary['mi_aux_pred_bits'], errors='coerce').mean())} "
                    f"GMI real/pred={_fmt(pd.to_numeric(summary['gmi_aux_real_bits'], errors='coerce').mean())}/"
                    f"{_fmt(pd.to_numeric(summary['gmi_aux_pred_bits'], errors='coerce').mean())} "
                    f"NGMI real/pred={_fmt(pd.to_numeric(summary['ngmi_aux_real'], errors='coerce').mean())}/"
                    f"{_fmt(pd.to_numeric(summary['ngmi_aux_pred'], errors='coerce').mean())}"
                )

    twin_gate_cols = [c for c in ["gate_g1", "gate_g2", "gate_g3", "gate_g4", "gate_g5"] if c in summary.columns]
    if twin_gate_cols:
        gate_pass = {c: int(pd.to_numeric(summary[c], errors="coerce").fillna(0).sum()) for c in twin_gate_cols}
        print(f"  twin_gate_pass_counts: {gate_pass}")
    if "gate_g6" in summary.columns:
        g6_pass = int(pd.to_numeric(summary["gate_g6"], errors="coerce").fillna(0).sum())
        print(f"  stat_screen_gate_pass_count: {g6_pass}")
    if {"stat_mmd_qval", "stat_energy_qval"}.issubset(summary.columns):
        print(
            "  stat_screen_means: "
            f"mmd_q={_fmt(pd.to_numeric(summary['stat_mmd_qval'], errors='coerce').mean())} "
            f"energy_q={_fmt(pd.to_numeric(summary['stat_energy_qval'], errors='coerce').mean())}"
        )

    if {"dist_target_m", "validation_status"}.issubset(summary.columns):
        by_dist = (
            summary.groupby(["dist_target_m", "validation_status"])
            .size()
            .reset_index(name="n")
            .sort_values(["dist_target_m", "validation_status"])
        )
        print("validation by distance:")
        for _, row in by_dist.iterrows():
            print(f"  dist={_fmt(row['dist_target_m'])} status={row['validation_status']} n={int(row['n'])}")

    if {"regime_id", "cvae_rel_evm_error", "cvae_cov_rel_var", "cvae_psd_l2"}.issubset(summary.columns):
        bad = summary.sort_values(
            ["cvae_rel_evm_error", "cvae_cov_rel_var", "cvae_psd_l2"],
            ascending=[False, False, False],
        ).head(5)
        print("hardest regimes:")
        for _, row in bad.iterrows():
            print(
                "  "
                f"{row['regime_id']} "
                f"evm_rel={_fmt(row.get('cvae_rel_evm_error'))} "
                f"cov_rel={_fmt(row.get('cvae_cov_rel_var'))} "
                f"psd={_fmt(row.get('cvae_psd_l2'))} "
                f"status_twin={row.get('validation_status', 'n/a')}"
            )

    if leaderboard is not None and len(leaderboard):
        row = leaderboard.sort_values("rank").iloc[0]
        print(
            "leaderboard_top: "
            f"tag={row.get('best_grid_tag', 'n/a')} "
            f"pass_ratio={_fmt(row.get('gate_pass_ratio'))} "
            f"n_pass_twin={_fmt(row.get('n_pass'))} "
            f"n_pass_full={_fmt(row.get('n_full_pass'))} "
            f"n_fail={_fmt(row.get('n_fail'))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a protocol experiment directory.")
    parser.add_argument("exp_dir", nargs="?", help="Path to outputs/exp_* (default: latest)")
    args = parser.parse_args()

    exp_dir = _pick_exp(args.exp_dir)
    print(f"experiment: {exp_dir}")
    _summarize_manifest(exp_dir)
    _summarize_training(exp_dir)
    _summarize_protocol(exp_dir)


if __name__ == "__main__":
    main()
