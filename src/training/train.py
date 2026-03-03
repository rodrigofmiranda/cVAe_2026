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
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional prior cVAE")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_base", type=str, required=True)
    parser.add_argument("--run_id", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["DATASET_ROOT"] = args.dataset_root
    os.environ["OUTPUT_BASE"] = args.output_base

    if args.run_id:
        os.environ["RUN_ID"] = args.run_id

    # Import here to avoid triggering heavy TF initialization on module load
    from src.training import cvae_TRAIN_documented as train_module

    train_module.main()


if __name__ == "__main__":
    main()