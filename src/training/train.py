# -*- coding: utf-8 -*-
"""
CLI entrypoint for training.

This module wraps the existing training monolith logic
into an explicit command-line executable.

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
    if args.dry_run:
        overrides["dry_run"] = True

    # Import here to avoid triggering heavy TF initialization on module load
    from src.training import cvae_TRAIN_documented as train_module

    train_module.main(overrides=overrides)


if __name__ == "__main__":
    main()