# -*- coding: utf-8 -*-
"""
CLI entrypoint for evaluation.

Routes through :func:`src.evaluation.engine.evaluate_run`.

Usage
-----
    python -m src.evaluation.evaluate \\
        --dataset_root /path/to/dataset \\
        --run_dir /path/to/outputs/run_x

    # Evaluate a shared model but write artifacts elsewhere
    python -m src.evaluation.evaluate \\
        --dataset_root /path/to/dataset \\
        --run_dir /path/to/outputs/exp_x/train \\
        --output_run_dir /path/to/outputs/per_regime_eval

    # Smoke-test (Commit 3H):
    python -m src.evaluation.evaluate \\
        --dataset_root /path/to/dataset \\
        --run_dir /path/to/outputs/run_x \\
        --max_experiments 1 --max_samples_per_exp 200000
"""

import argparse
from src.config.gpu_guard import warn_if_no_gpu_and_confirm
from src.config.overrides import RunOverrides
from src.config.runtime_env import ensure_writable_mpl_config_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained cVAE")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument(
        "--output_run_dir",
        type=str,
        default=None,
        help="Optional output directory for evaluation artifacts (default: write back into --run_dir)",
    )
    # --- Commit 3H: optional smoke-test flags ---
    parser.add_argument("--max_experiments", type=int, default=None,
                        help="Limit number of experiments loaded (default: all)")
    parser.add_argument("--max_samples_per_exp", type=int, default=None,
                        help="Truncate samples per experiment (default: all)")
    parser.add_argument("--psd_nfft", type=int, default=None,
                        help="Override PSD NFFT size (default: use state config)")
    parser.add_argument("--max_dist_samples", type=int, default=None,
                        help="Cap samples used in residual-distribution metrics")
    parser.add_argument("--gauss_alpha", type=float, default=None,
                        help="Override Gaussianity significance level")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override evaluation RNG seed")
    parser.add_argument("--no_dist_metrics", action="store_true",
                        help="Skip residual-distribution metrics during evaluation")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load+split+load model+build inference graph, then exit")
    return parser.parse_args()


def main():
    args = parse_args()
    warn_if_no_gpu_and_confirm("evaluation")

    # Build typed overrides from CLI flags
    overrides_obj = RunOverrides.from_namespace(args)
    overrides = overrides_obj.to_dict()

    # Ensure MPLCONFIGDIR is writable before importing the evaluation module,
    # otherwise matplotlib may emit temp-dir warnings at import time.
    ensure_writable_mpl_config_dir()

    from src.evaluation.engine import evaluate_run

    summary = evaluate_run(
        run_dir=args.run_dir,
        dataset_root=args.dataset_root,
        overrides=overrides,
        output_run_dir=args.output_run_dir,
    )
    print(f"\n🏁 evaluate_run status: {summary.get('status', '?')}")
    print(f"📌 run_dir: {summary.get('run_dir', '')}")


if __name__ == "__main__":
    main()
