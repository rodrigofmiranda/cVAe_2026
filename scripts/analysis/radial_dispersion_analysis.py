#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radial dispersion analysis — Visualizes the gradient of noise dispersion
from constellation center to border.

Shows that border symbols have higher noise variance than center symbols
(non-linear LED clipping / saturation effect in VLC).
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


def load_regime_data(dataset_root, dist_str, curr_str):
    """Load X (sent) and Y (received) for a single regime."""
    dist_dir = dataset_root / dist_str
    curr_dir = dist_dir / curr_str
    exp_dirs = sorted(curr_dir.iterdir())
    if not exp_dirs:
        raise FileNotFoundError(f"No experiments in {curr_dir}")
    exp_dir = exp_dirs[0]
    iq_dir = exp_dir / "IQ_data"
    X = np.load(iq_dir / "X.npy")
    Y = np.load(iq_dir / "Y.npy")
    n = min(len(X), len(Y))
    return X[:n], Y[:n]


def radial_distance(X):
    """Euclidean distance from constellation center (0,0)."""
    return np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)


def compute_radial_bins(X, Y, n_bins=20):
    """Compute noise std in radial bins from center to border."""
    R = radial_distance(X)
    Delta = Y - X  # noise = received - sent
    noise_mag = np.sqrt(Delta[:, 0] ** 2 + Delta[:, 1] ** 2)

    bin_edges = np.linspace(R.min(), R.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_std_I = []
    bin_std_Q = []
    bin_std_mag = []
    bin_mean_mag = []
    bin_counts = []

    for i in range(n_bins):
        mask = (R >= bin_edges[i]) & (R < bin_edges[i + 1])
        if mask.sum() < 10:
            bin_std_I.append(np.nan)
            bin_std_Q.append(np.nan)
            bin_std_mag.append(np.nan)
            bin_mean_mag.append(np.nan)
            bin_counts.append(mask.sum())
            continue
        bin_std_I.append(np.std(Delta[mask, 0]))
        bin_std_Q.append(np.std(Delta[mask, 1]))
        bin_std_mag.append(np.std(noise_mag[mask]))
        bin_mean_mag.append(np.mean(noise_mag[mask]))
        bin_counts.append(mask.sum())

    return {
        "bin_centers": bin_centers,
        "std_I": np.array(bin_std_I),
        "std_Q": np.array(bin_std_Q),
        "std_mag": np.array(bin_std_mag),
        "mean_mag": np.array(bin_mean_mag),
        "counts": np.array(bin_counts),
    }


def plot_full_analysis(X, Y, regime_label, save_path, n_bins=20):
    """Generate comprehensive radial dispersion analysis plots."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(
        f"Radial Dispersion Analysis | {regime_label}\n"
        "Non-linear LED distortion: noise variance gradient center → border",
        fontsize=14, fontweight="bold",
    )

    Delta = Y - X
    R = radial_distance(X)
    noise_mag = np.sqrt(Delta[:, 0] ** 2 + Delta[:, 1] ** 2)
    bins = compute_radial_bins(X, Y, n_bins=n_bins)

    # --- Plot 1: Constellation colored by noise magnitude ---
    ax = axes[0, 0]
    n_plot = min(40000, len(X))
    idx = np.random.default_rng(42).choice(len(X), n_plot, replace=False)
    sc = ax.scatter(
        X[idx, 0], X[idx, 1], c=noise_mag[idx], s=1, alpha=0.5,
        cmap="hot", vmin=np.percentile(noise_mag, 5),
        vmax=np.percentile(noise_mag, 95),
    )
    plt.colorbar(sc, ax=ax, label="Noise magnitude |Δ|")
    ax.set_title("Sent constellation colored by noise magnitude")
    ax.set_xlabel("I"); ax.set_ylabel("Q")
    ax.set_aspect("equal")

    # --- Plot 2: Radial noise std (I and Q) ---
    ax = axes[0, 1]
    ax.plot(bins["bin_centers"], bins["std_I"], "o-", label="σ(ΔI)", color="tab:blue")
    ax.plot(bins["bin_centers"], bins["std_Q"], "s-", label="σ(ΔQ)", color="tab:orange")
    ax.set_xlabel("Radial distance from center")
    ax.set_ylabel("Noise std")
    ax.set_title("Noise std vs radial distance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Mean noise magnitude vs radius ---
    ax = axes[0, 2]
    ax.plot(bins["bin_centers"], bins["mean_mag"], "D-", color="tab:red", label="mean |Δ|")
    ax.fill_between(
        bins["bin_centers"],
        bins["mean_mag"] - bins["std_mag"],
        bins["mean_mag"] + bins["std_mag"],
        alpha=0.2, color="tab:red", label="±1σ"
    )
    ax.set_xlabel("Radial distance from center")
    ax.set_ylabel("Noise magnitude")
    ax.set_title("Mean noise magnitude vs radial distance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 4: 2D heatmap of noise variance ---
    ax = axes[1, 0]
    grid_res = 50
    x_edges = np.linspace(X[:, 0].min(), X[:, 0].max(), grid_res + 1)
    y_edges = np.linspace(X[:, 1].min(), X[:, 1].max(), grid_res + 1)
    var_map = np.full((grid_res, grid_res), np.nan)
    for i in range(grid_res):
        for j in range(grid_res):
            mask = (
                (X[:, 0] >= x_edges[i]) & (X[:, 0] < x_edges[i + 1]) &
                (X[:, 1] >= y_edges[j]) & (X[:, 1] < y_edges[j + 1])
            )
            if mask.sum() >= 5:
                var_map[j, i] = np.var(noise_mag[mask])
    im = ax.imshow(
        var_map, origin="lower", aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="inferno",
    )
    plt.colorbar(im, ax=ax, label="Var(|Δ|)")
    ax.set_title("Spatial noise variance map (X space)")
    ax.set_xlabel("I"); ax.set_ylabel("Q")

    # --- Plot 5: Overlay real vs predicted residual with radial rings ---
    ax = axes[1, 1]
    idx2 = np.random.default_rng(42).choice(len(X), min(30000, len(X)), replace=False)
    ax.scatter(Delta[idx2, 0], Delta[idx2, 1], s=1, alpha=0.2, label="Δ real", color="tab:orange")
    # Draw radial quantile rings
    for q, ls in [(0.25, ':'), (0.5, '--'), (0.75, '-'), (0.95, '-')]:
        r = np.percentile(noise_mag, q * 100)
        circle = plt.Circle((0, 0), r, fill=False, color='black', linestyle=ls, linewidth=1.2, label=f'P{int(q*100)}')
        ax.add_patch(circle)
    ax.set_title("Residual Δ with radial quantile rings")
    ax.set_xlabel("ΔI"); ax.set_ylabel("ΔQ")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", markerscale=4, fontsize=8)

    # --- Plot 6: Histogram of noise magnitude by radial zone ---
    ax = axes[1, 2]
    zones = [
        ("Center (0-25%)", 0.0, 0.25, "tab:blue"),
        ("Mid (25-50%)", 0.25, 0.50, "tab:green"),
        ("Outer (50-75%)", 0.50, 0.75, "tab:orange"),
        ("Border (75-100%)", 0.75, 1.0, "tab:red"),
    ]
    r_min, r_max = R.min(), R.max()
    for label, lo, hi, color in zones:
        r_lo = r_min + lo * (r_max - r_min)
        r_hi = r_min + hi * (r_max - r_min)
        mask = (R >= r_lo) & (R < r_hi)
        if mask.sum() > 10:
            ax.hist(noise_mag[mask], bins=60, density=True, alpha=0.45, label=label, color=color)
    ax.set_xlabel("Noise magnitude |Δ|")
    ax.set_ylabel("Density")
    ax.set_title("Noise distribution by radial zone")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {save_path}")
    return save_path


def main():
    dataset_root = Path("/workspace/2026/feat_mdn_g5_recovery_explore/data/dataset_fullsquare_organized")
    output_dir = Path("/workspace/2026/feat_mdn_g5_recovery_explore/outputs/exp_20260331_220659/plots/radial_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    regimes = [
        ("dist_0.8m", "curr_100mA"),
        ("dist_0.8m", "curr_700mA"),
        ("dist_1.0m", "curr_300mA"),
        ("dist_1.5m", "curr_100mA"),
        ("dist_1.5m", "curr_700mA"),
    ]

    for dist_str, curr_str in regimes:
        label = f"{dist_str}__{curr_str}"
        print(f"\n{'='*60}")
        print(f"Processing: {label}")
        try:
            X, Y = load_regime_data(dataset_root, dist_str, curr_str)
            print(f"  X={X.shape}, Y={Y.shape}")
            save_path = output_dir / f"radial_dispersion__{label.replace('.', 'p')}.png"
            plot_full_analysis(X, Y, label, save_path)
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
