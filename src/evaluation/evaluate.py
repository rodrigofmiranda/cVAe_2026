# -*- coding: utf-8 -*-
"""
CLI entrypoint for evaluation.

Wraps analise_cvae_reviewed.py into explicit execution.

Usage
-----
    python -m src.evaluation.evaluate \\
        --dataset_root /path/to/dataset \\
        --run_dir /path/to/outputs/run_x

    # Smoke-test (Commit 3H):
    python -m src.evaluation.evaluate \\
        --dataset_root /path/to/dataset \\
        --run_dir /path/to/outputs/run_x \\
        --max_experiments 1 --max_samples_per_exp 200000
"""

import argparse
import os

from src.config.overrides import RunOverrides


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained cVAE")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    # --- Commit 3H: optional smoke-test flags ---
    parser.add_argument("--max_experiments", type=int, default=None,
                        help="Limit number of experiments loaded (default: all)")
    parser.add_argument("--max_samples_per_exp", type=int, default=None,
                        help="Truncate samples per experiment (default: all)")
    parser.add_argument("--psd_nfft", type=int, default=None,
                        help="Override PSD NFFT size (default: use state config)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load+split+load model+build inference graph, then exit")
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["DATASET_ROOT"] = args.dataset_root

    # analise_cvae_reviewed reads RUN_ID from env; extract from run_dir path
    run_dir_name = os.path.basename(args.run_dir.rstrip("/"))
    output_base = os.path.dirname(args.run_dir.rstrip("/"))

    os.environ["OUTPUT_BASE"] = output_base
    os.environ["RUN_ID"] = run_dir_name

    # ---- backward-compat: patch state_run.json if keys are missing ----
    from pathlib import Path as _Path
    _state_path = _Path(args.run_dir) / "state_run.json"
    if _state_path.exists():
        import json as _json
        from src.config.io import ensure_state_run_compat
        _state = _json.loads(_state_path.read_text(encoding="utf-8"))
        ensure_state_run_compat(_state)  # mutates in-place, logs warnings

    # Build typed overrides from CLI flags
    overrides_obj = RunOverrides.from_namespace(args)
    overrides = overrides_obj.to_dict()

    # Import here to avoid triggering heavy TF initialization on module load
    from src.evaluation import analise_cvae_reviewed as eval_module

    eval_module.main(overrides=overrides)


if __name__ == "__main__":
    main()