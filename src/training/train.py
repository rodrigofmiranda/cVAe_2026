# -*- coding: utf-8 -*-
"""
CLI entrypoint for training.

Routes through :func:`src.training.engine.train_engine`, the canonical
training entrypoint.

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
from src.config.overrides import RunOverrides


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

    # Build typed overrides from CLI flags
    overrides_obj = RunOverrides.from_namespace(args)
    overrides = overrides_obj.to_dict()

    from src.training.engine import train_engine

    summary = train_engine(
        dataset_root=args.dataset_root,
        output_base=args.output_base,
        run_id=args.run_id,
        overrides=overrides,
    )
    print(f"\n🏁 train_engine status: {summary.get('status', '?')}")
    print(f"📌 run_dir: {summary.get('run_dir', '')}")


if __name__ == "__main__":
    main()
