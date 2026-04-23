#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate raw constellation panels for the 6 canonical .npy files per experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

FILES_ORDER = [
    "x_sent.npy",
    "y_recv.npy",
    "y_recv_sync.npy",
    "y_recv_norm.npy",
    "X.npy",
    "Y.npy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot constellation panels (2x3) for 16QAM raw .npy files."
    )
    repo_root = Path(__file__).resolve().parents[4]
    canonical_dataset_root = repo_root / "data" / "benchmarks" / "modulations" / "16qam"
    legacy_dataset_root = repo_root / "data" / "16qam"
    canonical_output_root = (
        repo_root
        / "outputs"
        / "benchmarks"
        / "modulations"
        / "16qam"
        / "crossline_summary"
        / "raw_constellations"
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=canonical_dataset_root if canonical_dataset_root.exists() else legacy_dataset_root,
        help="Root containing dist_*/curr_*/.../IQ_data directories.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=canonical_output_root,
        help="Directory where figures and manifest CSV will be written.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=80_000,
        help="Maximum points plotted per .npy array.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed used when downsampling point clouds.",
    )
    return parser.parse_args()


def ensure_iq_shape(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        arr = np.stack([arr.real, arr.imag], axis=-1)
    if arr.ndim == 2 and arr.shape[1] == 2:
        pass
    elif arr.ndim == 2 and arr.shape[0] == 2:
        arr = arr.T
    else:
        raise ValueError(f"Unexpected I/Q format: shape={arr.shape}, dtype={arr.dtype}")
    return arr.astype(np.float32, copy=False)


def sample_points(arr: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    n = len(arr)
    if n <= max_points:
        return arr
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    return arr[idx]


def symmetric_limits(sampled_arrays: List[np.ndarray]) -> tuple[float, float]:
    merged = np.concatenate(sampled_arrays, axis=0)
    lim = float(np.percentile(np.abs(merged), 99.8))
    lim = max(lim * 1.05, 1e-3)
    return -lim, lim


def discover_iq_dirs(dataset_root: Path) -> List[Path]:
    return sorted(p for p in dataset_root.rglob("IQ_data") if p.is_dir())


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    iq_dirs = discover_iq_dirs(dataset_root)
    if not iq_dirs:
        raise RuntimeError(f"No IQ_data directories found under: {dataset_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    manifest_rows = []

    # Local import so script can still do metadata-only checks without mpl.
    import matplotlib.pyplot as plt

    print(f"Dataset root: {dataset_root}")
    print(f"Output root : {output_root}")
    print(f"IQ_data dirs: {len(iq_dirs)}")

    for i, iq_dir in enumerate(iq_dirs, start=1):
        exp_dir = iq_dir.parent
        rel_exp = exp_dir.relative_to(dataset_root)
        out_dir = output_root / rel_exp
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / "constellations_6npy.png"

        arrays: Dict[str, np.ndarray] = {}
        sampled: Dict[str, np.ndarray] = {}
        missing: List[str] = []

        for fname in FILES_ORDER:
            fpath = iq_dir / fname
            if not fpath.exists():
                missing.append(fname)
                continue
            arr = ensure_iq_shape(np.load(fpath))
            arrays[fname] = arr
            sampled[fname] = sample_points(arr, args.max_points, rng)

        if not sampled:
            manifest_rows.append(
                {
                    "experiment": str(rel_exp),
                    "status": "skipped_no_arrays",
                    "missing_files": ";".join(missing),
                    "plot_path": "",
                }
            )
            print(f"[{i:02d}/{len(iq_dirs):02d}] skipped {rel_exp} (no valid arrays)")
            continue

        lim_low, lim_high = symmetric_limits(list(sampled.values()))

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(f"Raw constellations | {rel_exp}", fontsize=14)

        for ax, fname in zip(axes.ravel(), FILES_ORDER):
            arr = sampled.get(fname)
            if arr is None:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"{fname}\nmissing",
                    ha="center",
                    va="center",
                    fontsize=11,
                    family="monospace",
                )
                continue

            ax.scatter(arr[:, 0], arr[:, 1], s=1, alpha=0.30)
            ax.set_title(f"{fname} | N={len(arrays[fname]):,}", fontsize=10)
            ax.set_xlabel("I")
            ax.set_ylabel("Q")
            ax.set_xlim(lim_low, lim_high)
            ax.set_ylim(lim_low, lim_high)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)

        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
        fig.savefig(out_png, dpi=args.dpi)
        plt.close(fig)

        manifest_rows.append(
            {
                "experiment": str(rel_exp),
                "status": "ok",
                "missing_files": ";".join(missing),
                "plot_path": str(out_png),
            }
        )
        print(f"[{i:02d}/{len(iq_dirs):02d}] wrote {out_png}")

    manifest_path = output_root / "manifest_constellations_6npy.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "status", "missing_files", "plot_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_ok = sum(1 for row in manifest_rows if row["status"] == "ok")
    print("\nDone.")
    print(f"Panels generated: {n_ok}/{len(manifest_rows)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
