# -*- coding: utf-8 -*-
"""
Protocol runner — reproducible baseline + cVAE evaluation across regimes.

Orchestrates training and evaluation per regime defined in a protocol JSON,
then consolidates results into a summary table.

No architecture, loss, or metrics changes.  Only orchestration.

Usage
-----
    python -m src.protocol.run \\
        --dataset_root data/dataset_fullsquare_organized \\
        --output_base  outputs \\
        [--protocol configs/protocol_default.json]

    # Quick smoke-test (1 regime, 1 grid, 2 epochs):
    python -m src.protocol.run \\
        --dataset_root data/dataset_fullsquare_organized \\
        --output_base  outputs \\
        --protocol configs/protocol_default.json \\
        --max_epochs 2 --max_grids 1 --max_experiments 1

    # Dry-run (no training, just validate the protocol):
    python -m src.protocol.run \\
        --dataset_root data/dataset_fullsquare_organized \\
        --output_base  outputs \\
        --dry_run

Commit 3J.
"""

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run a reproducible protocol (train + evaluate) across VLC regimes."
    )
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--output_base", type=str, required=True)
    p.add_argument("--protocol", type=str, default=None,
                   help="Path to protocol JSON (default: configs/protocol_default.json)")
    # --- global overrides (applied to every regime; override protocol JSON) ---
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--max_grids", type=int, default=None)
    p.add_argument("--grid_group", type=str, default=None)
    p.add_argument("--grid_tag", type=str, default=None)
    p.add_argument("--max_experiments", type=int, default=None)
    p.add_argument("--max_samples_per_exp", type=int, default=None)
    p.add_argument("--val_split", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--psd_nfft", type=int, default=None)
    p.add_argument("--skip_eval", action="store_true",
                   help="Run training only, skip evaluation step")
    p.add_argument("--dry_run", action="store_true",
                   help="Validate protocol + build model summary, no training")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_protocol(path: Optional[str]) -> dict:
    """Load protocol JSON, falling back to the default bundled config."""
    if path is None:
        # try repo-relative default
        candidates = [
            Path("configs/protocol_default.json"),
            Path(__file__).resolve().parent.parent.parent / "configs" / "protocol_default.json",
        ]
        for c in candidates:
            if c.exists():
                path = str(c)
                break
        if path is None:
            raise FileNotFoundError(
                "No --protocol given and configs/protocol_default.json not found."
            )
    proto = json.loads(Path(path).read_text(encoding="utf-8"))
    if "regimes" not in proto or not proto["regimes"]:
        raise ValueError("Protocol JSON must contain a non-empty 'regimes' list.")
    return proto


def _merge_overrides(protocol_globals: dict, cli_args: argparse.Namespace) -> dict:
    """
    Build the overrides dict for a single regime by layering:
        protocol global_settings  <  CLI flags (explicit wins)
    """
    ov = {}
    pg = protocol_globals or {}

    # mapping: override_key -> (protocol_key, cli_attr, cast)
    _MAP = [
        ("seed",                "seed",                "seed",                int),
        ("val_split",           "val_split",           "val_split",           float),
        ("max_epochs",          "max_epochs",          "max_epochs",          int),
        ("max_grids",           "max_grids",           "max_grids",          int),
        ("grid_group",          "grid_group",          "grid_group",          str),
        ("grid_tag",            "grid_tag",            "grid_tag",            str),
        ("max_experiments",     "max_experiments",     "max_experiments",     int),
        ("max_samples_per_exp", "max_samples_per_exp", "max_samples_per_exp", int),
        ("psd_nfft",            "psd_nfft",            "psd_nfft",            int),
    ]

    for ov_key, pg_key, cli_attr, cast in _MAP:
        # CLI takes precedence
        cli_val = getattr(cli_args, cli_attr, None)
        if cli_val is not None:
            ov[ov_key] = cast(cli_val)
        elif pg.get(pg_key) is not None:
            ov[ov_key] = cast(pg[pg_key])

    if cli_args.dry_run:
        ov["dry_run"] = True

    return ov


def _git_commit_hash() -> str:
    """Best-effort git rev-parse HEAD."""
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _runtime_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    try:
        import tensorflow as tf
        versions["tensorflow"] = tf.__version__
    except Exception:
        pass
    try:
        import numpy as np
        versions["numpy"] = np.__version__
    except Exception:
        pass
    return versions


def _read_eval_metrics(run_dir: Path) -> dict:
    """Read evaluation metrics JSON produced by analise_cvae_reviewed."""
    p = run_dir / "logs" / "metricas_globais_reanalysis.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _read_train_state(run_dir: Path) -> dict:
    """Read state_run.json produced by the training monolith."""
    p = run_dir / "state_run.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _extract_best_grid_tag(state: dict) -> str:
    """Extract best grid tag from the training state if available."""
    try:
        res_path = state.get("artifacts", {}).get("grid_results_xlsx", "")
        if res_path and Path(res_path).exists():
            import pandas as pd
            df = pd.read_excel(res_path, sheet_name="results_sorted")
            if len(df) > 0 and "tag" in df.columns:
                return str(df.iloc[0]["tag"])
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_regime(
    regime: dict,
    dataset_root: str,
    output_base: str,
    base_overrides: dict,
    protocol_dir: Path,
    skip_eval: bool = False,
) -> dict:
    """
    Execute train + evaluate for one regime.

    Returns a result dict with run_dir, metrics, and status.
    """
    regime_id = regime["regime_id"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"protocol_{regime_id}_{ts}"

    # --- Resolve experiment filter ---
    # Option A: explicit experiment_paths
    # Option B: experiment_regex
    exp_paths = regime.get("experiment_paths", [])
    exp_regex = regime.get("experiment_regex", None)

    # Build per-regime overrides
    ov = dict(base_overrides)

    # If regime specifies experiment_paths, we filter via max_experiments = len
    # AND set the DATASET_ROOT to the parent so only those experiments are found.
    # However, the current loader discovers ALL experiments under DATASET_ROOT.
    # We handle filtering by passing experiment_paths in overrides for
    # future use, but for now we rely on max_experiments for bounding.
    if exp_paths:
        ov["_experiment_paths"] = exp_paths
    if exp_regex:
        ov["_experiment_regex"] = exp_regex

    result = {
        "regime_id": regime_id,
        "description": regime.get("description", ""),
        "run_id": run_id,
        "run_dir": None,
        "train_status": "skipped",
        "eval_status": "skipped",
        "metrics": {},
        "best_grid_tag": "",
        "error": None,
    }

    # ---- TRAINING ----
    print(f"\n{'='*70}")
    print(f"🔬 REGIME: {regime_id} — {regime.get('description', '')}")
    print(f"{'='*70}")

    os.environ["DATASET_ROOT"] = dataset_root
    os.environ["OUTPUT_BASE"] = output_base
    os.environ["RUN_ID"] = run_id

    try:
        from src.training import cvae_TRAIN_documented as train_module
        print(f"\n📦 Training regime '{regime_id}' → run_id={run_id}")
        train_module.main(overrides=ov)
        result["train_status"] = "completed"
    except Exception as e:
        result["train_status"] = "failed"
        result["error"] = f"train: {e}\n{traceback.format_exc()}"
        print(f"❌ Training failed for regime '{regime_id}': {e}")

    run_dir = Path(output_base) / run_id
    result["run_dir"] = str(run_dir)

    # Read train state
    state = _read_train_state(run_dir)
    result["best_grid_tag"] = _extract_best_grid_tag(state)

    # If dry_run was set, training exits early — skip eval
    if ov.get("dry_run", False):
        result["train_status"] = "dry_run"
        return result

    # ---- EVALUATION ----
    if skip_eval:
        print(f"⏭️  Skipping evaluation for regime '{regime_id}' (--skip_eval)")
        return result

    if result["train_status"] != "completed":
        print(f"⏭️  Skipping evaluation for regime '{regime_id}' (training failed)")
        return result

    try:
        os.environ["DATASET_ROOT"] = dataset_root
        os.environ["OUTPUT_BASE"] = output_base
        os.environ["RUN_ID"] = run_id

        eval_ov = {}
        for k in ("max_experiments", "max_samples_per_exp", "psd_nfft"):
            if k in ov:
                eval_ov[k] = ov[k]

        from src.evaluation import analise_cvae_reviewed as eval_module
        print(f"\n📊 Evaluating regime '{regime_id}' → {run_dir}")
        eval_module.main(overrides=eval_ov)
        result["eval_status"] = "completed"
    except Exception as e:
        result["eval_status"] = "failed"
        err_msg = f"eval: {e}\n{traceback.format_exc()}"
        result["error"] = (result.get("error") or "") + err_msg
        print(f"❌ Evaluation failed for regime '{regime_id}': {e}")

    # Read eval metrics
    result["metrics"] = _read_eval_metrics(run_dir)

    return result


def build_summary_table(results: List[dict]) -> "pd.DataFrame":
    """Consolidate per-regime results into a summary DataFrame."""
    import pandas as pd

    rows = []
    for r in results:
        m = r.get("metrics", {})
        row = {
            "regime_id": r["regime_id"],
            "description": r.get("description", ""),
            "run_id": r["run_id"],
            "run_dir": r.get("run_dir", ""),
            "train_status": r["train_status"],
            "eval_status": r["eval_status"],
            "best_grid_tag": r.get("best_grid_tag", ""),
            "evm_real_%": m.get("evm_real_%"),
            "evm_pred_%": m.get("evm_pred_%"),
            "delta_evm_%": m.get("delta_evm_%"),
            "snr_real_db": m.get("snr_real_db"),
            "snr_pred_db": m.get("snr_pred_db"),
            "delta_snr_db": m.get("delta_snr_db"),
            "delta_mean_l2": m.get("delta_mean_l2"),
            "delta_cov_fro": m.get("delta_cov_fro"),
            "delta_skew_l2": m.get("delta_skew_l2"),
            "delta_kurt_l2": m.get("delta_kurt_l2"),
            "delta_psd_l2": m.get("delta_psd_l2"),
            "kl_q_to_p_total": None,
            "kl_p_to_N_total": None,
            "var_mc_gen": m.get("var_mc_gen"),
        }

        # Try to enrich with latent summary from eval run
        run_dir = Path(r.get("run_dir", ""))
        lat_path = run_dir / "logs" / "latent_summary.json"
        if lat_path.exists():
            try:
                lat = json.loads(lat_path.read_text(encoding="utf-8"))
                row["kl_q_to_p_total"] = lat.get("kl_q_to_p_total_mean")
                row["kl_p_to_N_total"] = lat.get("kl_p_to_N_total_mean")
            except Exception:
                pass

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    ts_start = datetime.now()
    ts_label = ts_start.strftime("%Y%m%d_%H%M%S")

    # Load protocol
    protocol = _load_protocol(args.protocol)
    proto_globals = protocol.get("global_settings", {})
    regimes = protocol["regimes"]

    # Merge protocol globals + CLI overrides
    base_overrides = _merge_overrides(proto_globals, args)

    # Protocol output directory
    protocol_dir = Path(args.output_base) / f"protocol_{ts_label}"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    (protocol_dir / "tables").mkdir(exist_ok=True)
    (protocol_dir / "logs").mkdir(exist_ok=True)

    print(f"🚀 Protocol runner — {len(regimes)} regime(s)")
    print(f"📁 Protocol dir: {protocol_dir}")
    print(f"🔧 Base overrides: {base_overrides}")

    # Save a copy of the protocol used
    (protocol_dir / "logs" / "protocol_input.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- Run each regime ----
    results = []
    for i, regime in enumerate(regimes, 1):
        print(f"\n{'#'*70}")
        print(f"# REGIME {i}/{len(regimes)}: {regime['regime_id']}")
        print(f"{'#'*70}")
        r = run_regime(
            regime=regime,
            dataset_root=args.dataset_root,
            output_base=args.output_base,
            base_overrides=base_overrides,
            protocol_dir=protocol_dir,
            skip_eval=args.skip_eval,
        )
        results.append(r)

    # ---- Build summary table ----
    import pandas as pd
    df_summary = build_summary_table(results)

    summary_csv = protocol_dir / "tables" / "summary_by_regime.csv"
    summary_xlsx = protocol_dir / "tables" / "summary_by_regime.xlsx"
    df_summary.to_csv(summary_csv, index=False)
    df_summary.to_excel(summary_xlsx, index=False)
    print(f"\n📊 Summary table: {summary_csv}")

    # ---- Write manifest ----
    ts_end = datetime.now()
    manifest = {
        "protocol_version": protocol.get("protocol_version", "1.0"),
        "timestamp_start": ts_start.isoformat(timespec="seconds"),
        "timestamp_end": ts_end.isoformat(timespec="seconds"),
        "duration_seconds": (ts_end - ts_start).total_seconds(),
        "git_commit": _git_commit_hash(),
        "versions": _runtime_versions(),
        "args": {
            "dataset_root": args.dataset_root,
            "output_base": args.output_base,
            "protocol": args.protocol,
            "skip_eval": args.skip_eval,
            "dry_run": args.dry_run,
        },
        "base_overrides": base_overrides,
        "n_regimes": len(regimes),
        "regimes": [
            {
                "regime_id": r["regime_id"],
                "run_id": r["run_id"],
                "run_dir": r.get("run_dir", ""),
                "train_status": r["train_status"],
                "eval_status": r["eval_status"],
                "best_grid_tag": r.get("best_grid_tag", ""),
                "error": r.get("error"),
            }
            for r in results
        ],
    }
    manifest_path = protocol_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"📋 Manifest: {manifest_path}")

    # ---- Final summary to stdout ----
    print(f"\n{'='*70}")
    print(f"✅ Protocol complete — {len(results)} regime(s)")
    print(f"   Duration: {ts_end - ts_start}")
    print(f"   Output:   {protocol_dir}")
    for r in results:
        status = f"train={r['train_status']}, eval={r['eval_status']}"
        delta = ""
        m = r.get("metrics", {})
        if m.get("delta_evm_%") is not None:
            delta = f" | ΔEVM={m['delta_evm_%']:+.3f}pp ΔSNR={m.get('delta_snr_db', 0):+.3f}dB"
        print(f"   • {r['regime_id']}: {status}{delta}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
