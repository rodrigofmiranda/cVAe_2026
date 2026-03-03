# -*- coding: utf-8 -*-
"""
CLI entrypoint for evaluation.

Wraps analise_cvae_reviewed.py into explicit execution.

Usage
-----
    python -m src.evaluation.evaluate \\
        --dataset_root /path/to/dataset \\
        --run_dir /path/to/outputs/run_x
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained cVAE")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["DATASET_ROOT"] = args.dataset_root

    # analise_cvae_reviewed reads RUN_ID from env; extract from run_dir path
    run_dir_name = os.path.basename(args.run_dir.rstrip("/"))
    output_base = os.path.dirname(args.run_dir.rstrip("/"))

    os.environ["OUTPUT_BASE"] = output_base
    os.environ["RUN_ID"] = run_dir_name

    # Import here to avoid triggering heavy TF initialization on module load
    from src.evaluation import analise_cvae_reviewed as eval_module

    eval_module.main()


if __name__ == "__main__":
    main()