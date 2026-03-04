# -*- coding: utf-8 -*-
"""
CLI entrypoint for training.

Routes through :func:`src.training.engine.train_engine`, which
currently delegates to the monolith but provides a clean API
surface for future refactoring.

No architectural or loss changes.

Usage
-----
    python -m src.training.train \\
        --dataset_root /path/to/dataset \\
        --output_base  /path/to/outputs \\
        [--run_id test_run]

    # Smoke-test (Commit 3H):
    python -m src.training.train \\
        --dataset_root /path/to/dataset \\
        --output_base  /path/to/outputs \\
        --max_epochs 2 --max_experiments 1 --max_samples_per_exp 200000
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional prior cVAE")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_base", type=str, required=True)
    parser.add_argument("--run_id", type=str, default=None)
    # --- Commit 3H: optional smoke-test flags ---
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override max training epochs (default: use TRAINING_CONFIG)")
    parser.add_argument("--max_experiments", type=int, default=None,
                        help="Limit number of experiments loaded (default: all)")
    parser.add_argument("--max_samples_per_exp", type=int, default=None,
                        help="Truncate samples per experiment (default: all)")
    parser.add_argument("--val_split", type=float, default=None,
                        help="Override validation split ratio (default: use TRAINING_CONFIG)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed (default: use TRAINING_CONFIG)")
    parser.add_argument("--max_grids", type=int, default=None,
                        help="Limit number of grid configurations executed (default: all)")
    parser.add_argument("--grid_group", type=str, default=None,
                        help="Regex filter: keep grids whose group matches (default: all)")
    parser.add_argument("--grid_tag", type=str, default=None,
                        help="Regex filter: keep grids whose tag matches (default: all)")
    parser.add_argument("--keras_verbose", type=int, default=2, choices=[0, 1, 2],
                        help="Keras fit verbosity: 0=silent, 1=progress bar, 2=one line/epoch (default: 2)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load+split+build model+print shapes, then exit without training")
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["DATASET_ROOT"] = args.dataset_root
    os.environ["OUTPUT_BASE"] = args.output_base

    if args.run_id:
        os.environ["RUN_ID"] = args.run_id

    # Commit 3H: build overrides dict from CLI flags
    overrides = {}
    if args.max_epochs is not None:
        overrides["max_epochs"] = args.max_epochs
    if args.max_experiments is not None:
        overrides["max_experiments"] = args.max_experiments
    if args.max_samples_per_exp is not None:
        overrides["max_samples_per_exp"] = args.max_samples_per_exp
    if args.val_split is not None:
        overrides["val_split"] = args.val_split
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.max_grids is not None:
        overrides["max_grids"] = args.max_grids
    if args.grid_group is not None:
        overrides["grid_group"] = args.grid_group
    if args.grid_tag is not None:
        overrides["grid_tag"] = args.grid_tag
    if args.keras_verbose is not None:
        overrides["keras_verbose"] = args.keras_verbose
    if args.dry_run:
        overrides["dry_run"] = True

    # Route through the training engine (which currently delegates to the monolith)
    from src.training.engine import train_engine

    summary = train_engine(
        dataset_root=args.dataset_root,
        output_base=args.output_base,
        run_id=args.run_id,
        overrides=overrides,
    )
    print(f"\n🏁 train_engine status: {summary.get('status', '?')}")


if __name__ == "__main__":
    main()