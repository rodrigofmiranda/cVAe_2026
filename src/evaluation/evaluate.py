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
from pathlib import Path

from src.config.gpu_guard import warn_if_no_gpu_and_confirm
from src.config.overrides import RunOverrides
from src.config.runtime_env import ensure_writable_mpl_config_dir
from src.data.loading import discover_experiments, parse_dist_curr_from_path, read_metadata


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
    parser.add_argument("--max_val_samples_per_exp", type=int, default=None,
                        help="Truncate validation samples per experiment after split")
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
    parser.add_argument("--distance_m", type=float, default=None,
                        help="Optional regime filter: target distance (m)")
    parser.add_argument("--current_mA", type=float, default=None,
                        help="Optional regime filter: target current (mA)")
    parser.add_argument("--dist_tol_m", type=float, default=None,
                        help="Distance tolerance for regime filtering (default: 0.05 m)")
    parser.add_argument("--curr_tol_mA", type=float, default=None,
                        help="Current tolerance for regime filtering (default: 25 mA)")
    return parser.parse_args()


def _select_experiments_by_regime(
    dataset_root: Path,
    *,
    distance_m: float | None,
    current_mA: float | None,
    dist_tol_m: float,
    curr_tol_mA: float,
) -> list[str]:
    exp_dirs = discover_experiments(dataset_root, verbose=False)
    selected: list[str] = []

    for exp_dir in exp_dirs:
        dist, curr = parse_dist_curr_from_path(exp_dir)
        if dist is None or curr is None:
            meta = read_metadata(exp_dir)
            if dist is None:
                for key in ("distance_m", "distance", "dist_m", "dist"):
                    if key in meta:
                        try:
                            dist = float(meta[key])
                            break
                        except Exception:
                            pass
            if curr is None:
                for key in ("current_mA", "current", "curr_mA", "curr"):
                    if key in meta:
                        try:
                            curr = float(meta[key])
                            break
                        except Exception:
                            pass

        if dist is None or curr is None:
            continue

        dist_ok = distance_m is None or abs(float(dist) - float(distance_m)) <= float(dist_tol_m)
        curr_ok = current_mA is None or abs(float(curr) - float(current_mA)) <= float(curr_tol_mA)
        if dist_ok and curr_ok:
            selected.append(str(exp_dir))

    selected.sort()
    if not selected:
        raise RuntimeError(
            "No experiments matched the regime filter "
            f"(distance_m={distance_m}, current_mA={current_mA}, "
            f"dist_tol_m={dist_tol_m}, curr_tol_mA={curr_tol_mA})."
        )
    return selected


def main():
    args = parse_args()
    warn_if_no_gpu_and_confirm("evaluation")

    # Build typed overrides from CLI flags
    overrides_obj = RunOverrides.from_namespace(args)
    overrides = overrides_obj.to_dict()
    if args.distance_m is not None or args.current_mA is not None:
        if not args.dataset_root:
            raise ValueError(
                "--dataset_root is required when using --distance_m/--current_mA filtering."
            )
        dist_tol = float(args.dist_tol_m) if args.dist_tol_m is not None else 0.05
        curr_tol = float(args.curr_tol_mA) if args.curr_tol_mA is not None else 25.0
        selected_exps = _select_experiments_by_regime(
            Path(args.dataset_root).resolve(),
            distance_m=args.distance_m,
            current_mA=args.current_mA,
            dist_tol_m=dist_tol,
            curr_tol_mA=curr_tol,
        )
        overrides["_selected_experiments"] = selected_exps
        overrides["dist_tol_m"] = dist_tol
        overrides["curr_tol_mA"] = curr_tol
        print(
            "🎯 Regime filter active | "
            f"distance_m={args.distance_m} | current_mA={args.current_mA} | "
            f"matches={len(selected_exps)}"
        )
        for exp_path in selected_exps:
            print(f"   • {exp_path}")

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
