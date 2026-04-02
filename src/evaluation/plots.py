# -*- coding: utf-8 -*-
"""
src.evaluation.plots — Reusable evaluation plot functions.

Each function creates a single matplotlib figure and saves it to disk.
No scientific changes — plot content is identical to the canonical
evaluation pipeline.

All functions accept an explicit ``save_path`` and call
``plt.savefig + plt.close`` internally, so callers never need to
manage figure lifetime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _savefig(path: Path, dpi: int = 180) -> Path:
    """Tight-layout → save → close.  Returns the written path."""
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_overlay(
    Y_real: np.ndarray,
    Y_pred: np.ndarray,
    save_path: Path,
    *,
    max_points: int = 80_000,
    title: str = "Constellation overlay: Y real vs Y pred",
) -> Path:
    """Scatter overlay of real vs predicted received IQ constellation."""
    import matplotlib.pyplot as plt

    N = min(max_points, len(Y_real))
    Yr = Y_real[:N]
    Yp = Y_pred[:N]

    plt.figure()
    plt.scatter(Yr[:, 0], Yr[:, 1], s=2, alpha=0.35, label="Y real")
    plt.scatter(Yp[:, 0], Yp[:, 1], s=2, alpha=0.35, label="Y pred")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title(title)
    plt.legend(markerscale=4)
    return _savefig(save_path)


def plot_iq_scatter(
    Y: np.ndarray,
    save_path: Path,
    *,
    max_points: int = 80_000,
    title: str = "IQ scatter",
    label: str = "Y",
) -> Path:
    """Simple IQ scatter of a single array."""
    import matplotlib.pyplot as plt

    N = min(max_points, len(Y))
    Ys = Y[:N]
    plt.figure()
    plt.scatter(Ys[:, 0], Ys[:, 1], s=2, alpha=0.35, label=label)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title(title)
    plt.legend(markerscale=4)
    return _savefig(save_path)


def plot_residual_overlay(
    X: np.ndarray,
    Y_real: np.ndarray,
    Y_pred: np.ndarray,
    save_path: Path,
    *,
    max_points: int = 80_000,
    title: str = "Residual constellation overlay (Δ)",
) -> Path:
    """Scatter overlay of real vs predicted residual Δ = Y − X."""
    import matplotlib.pyplot as plt

    N = min(max_points, len(X))
    Dr = Y_real[:N] - X[:N]
    Dp = Y_pred[:N] - X[:N]

    plt.figure()
    plt.scatter(Dr[:, 0], Dr[:, 1], s=2, alpha=0.35, label="Δ real = Y-X")
    plt.scatter(Dp[:, 0], Dp[:, 1], s=2, alpha=0.35, label="Δ pred = Ŷ-X")
    plt.xlabel("ΔI")
    plt.ylabel("ΔQ")
    plt.title(title)
    plt.legend(markerscale=4)
    return _savefig(save_path)


def plot_residual_fingerprint(
    X: np.ndarray,
    Y_real: np.ndarray,
    Y_pred: np.ndarray,
    save_path: Path,
    *,
    distm: Optional[dict] = None,
    max_points: int = 80_000,
    bins: int = 140,
    title: str = "Residual fingerprint: real vs pred",
) -> Path:
    """Compact fingerprint view for residual shape and mismatch diagnostics."""
    import matplotlib.pyplot as plt
    from src.evaluation.metrics import _skew_kurt, residual_distribution_metrics

    N = min(max_points, len(X), len(Y_real), len(Y_pred))
    Xs = np.asarray(X[:N], dtype=np.float64)
    Yr = np.asarray(Y_real[:N], dtype=np.float64)
    Yp = np.asarray(Y_pred[:N], dtype=np.float64)

    Dr = Yr - Xs
    Dp = Yp - Xs
    skew_r, kurt_r = _skew_kurt(Dr)
    skew_p, kurt_p = _skew_kurt(Dp)
    mean_r = np.mean(Dr, axis=0)
    mean_p = np.mean(Dp, axis=0)
    std_r = np.std(Dr, axis=0)
    std_p = np.std(Dp, axis=0)

    metrics = distm if isinstance(distm, dict) else residual_distribution_metrics(Xs, Yr, Yp)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(title)

    ax[0, 0].hist2d(Dr[:, 0], Dr[:, 1], bins=bins)
    ax[0, 0].set_title("Residual density (real)")
    ax[0, 0].set_xlabel("ΔI")
    ax[0, 0].set_ylabel("ΔQ")

    ax[0, 1].hist2d(Dp[:, 0], Dp[:, 1], bins=bins)
    ax[0, 1].set_title("Residual density (pred)")
    ax[0, 1].set_xlabel("ΔI")
    ax[0, 1].set_ylabel("ΔQ")

    labels = [
        "|Δmean I|", "|Δmean Q|", "|Δstd I|", "|Δstd Q|",
        "|Δskew I|", "|Δskew Q|", "|Δkurt I|", "|Δkurt Q|",
    ]
    values = [
        abs(float(metrics.get("delta_mean_I", float("nan")))),
        abs(float(metrics.get("delta_mean_Q", float("nan")))),
        abs(float(metrics.get("delta_std_I", float("nan")))),
        abs(float(metrics.get("delta_std_Q", float("nan")))),
        abs(float(metrics.get("delta_skew_I", float("nan")))),
        abs(float(metrics.get("delta_skew_Q", float("nan")))),
        abs(float(metrics.get("delta_kurt_I", float("nan")))),
        abs(float(metrics.get("delta_kurt_Q", float("nan")))),
    ]
    ax[1, 0].bar(np.arange(len(labels)), values, color="#4C78A8")
    ax[1, 0].set_xticks(np.arange(len(labels)))
    ax[1, 0].set_xticklabels(labels, rotation=30, ha="right")
    ax[1, 0].set_title("Axis fingerprint mismatch")
    ax[1, 0].grid(True, alpha=0.2)

    ax[1, 1].axis("off")
    txt = [
        f"N used: {N:,}",
        "",
        f"mean real    = ({mean_r[0]:+.3e}, {mean_r[1]:+.3e})",
        f"mean pred    = ({mean_p[0]:+.3e}, {mean_p[1]:+.3e})",
        f"std real     = ({std_r[0]:.3e}, {std_r[1]:.3e})",
        f"std pred     = ({std_p[0]:.3e}, {std_p[1]:.3e})",
        f"skew real    = ({skew_r[0]:+.3f}, {skew_r[1]:+.3f})",
        f"skew pred    = ({skew_p[0]:+.3f}, {skew_p[1]:+.3f})",
        f"kurt real    = ({kurt_r[0]:+.3f}, {kurt_r[1]:+.3f})",
        f"kurt pred    = ({kurt_p[0]:+.3f}, {kurt_p[1]:+.3f})",
        "",
        f"Δmean L2     = {float(metrics.get('delta_mean_l2', float('nan'))):.4g}",
        f"Δcov Fro     = {float(metrics.get('delta_cov_fro', float('nan'))):.4g}",
        f"ΔPSD L2      = {float(metrics.get('delta_psd_l2', float('nan'))):.4g}",
        f"ΔACF L2      = {float(metrics.get('delta_acf_l2', float('nan'))):.4g}",
        f"Δskew L2     = {float(metrics.get('delta_skew_l2', float('nan'))):.4g}",
        f"Δkurt L2     = {float(metrics.get('delta_kurt_l2', float('nan'))):.4g}",
    ]
    ax[1, 1].text(
        0.02,
        0.98,
        "\n".join(txt),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    ax[1, 1].set_title("Fingerprint summary")

    return _savefig(save_path)


def plot_comparison_panel_6(
    X: np.ndarray,
    Y_real: np.ndarray,
    Y_pred: np.ndarray,
    save_path: Path,
    *,
    max_points: int = 80_000,
    title: str = "Constellation comparison panel",
) -> Path:
    """6-panel comparison in the same visual spirit as raw dataset plots."""
    import matplotlib.pyplot as plt

    N = min(max_points, len(X), len(Y_real), len(Y_pred))
    Xs = np.asarray(X[:N], dtype=np.float64)
    Yr = np.asarray(Y_real[:N], dtype=np.float64)
    Yp = np.asarray(Y_pred[:N], dtype=np.float64)
    Dr = Yr - Xs
    Dp = Yp - Xs

    lim_y = float(np.percentile(np.abs(np.concatenate([Yr, Yp], axis=0)), 99.8)) * 1.05
    lim_d = float(np.percentile(np.abs(np.concatenate([Dr, Dp], axis=0)), 99.8)) * 1.05
    lim_y = max(lim_y, 1e-3)
    lim_d = max(lim_d, 1e-3)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(title)
    ax = axes.ravel()

    ax[0].scatter(Xs[:, 0], Xs[:, 1], s=1, alpha=0.30)
    ax[0].set_title("X sent")
    ax[1].scatter(Yr[:, 0], Yr[:, 1], s=1, alpha=0.30)
    ax[1].set_title("Y real")
    ax[2].scatter(Yp[:, 0], Yp[:, 1], s=1, alpha=0.30)
    ax[2].set_title("Y pred")

    ax[3].scatter(Yr[:, 0], Yr[:, 1], s=1, alpha=0.25, label="Y real")
    ax[3].scatter(Yp[:, 0], Yp[:, 1], s=1, alpha=0.25, label="Y pred")
    ax[3].set_title("Overlay (Y real vs Y pred)")
    ax[3].legend(markerscale=6)

    ax[4].scatter(Dr[:, 0], Dr[:, 1], s=1, alpha=0.30)
    ax[4].set_title("Residual real (Y - X)")
    ax[5].scatter(Dp[:, 0], Dp[:, 1], s=1, alpha=0.30)
    ax[5].set_title("Residual pred (Ŷ - X)")

    for idx in (0, 1, 2, 3):
        ax[idx].set_xlim(-lim_y, lim_y)
        ax[idx].set_ylim(-lim_y, lim_y)
        ax[idx].set_xlabel("I")
        ax[idx].set_ylabel("Q")
        ax[idx].set_aspect("equal", adjustable="box")
        ax[idx].grid(alpha=0.2)

    for idx in (4, 5):
        ax[idx].set_xlim(-lim_d, lim_d)
        ax[idx].set_ylim(-lim_d, lim_d)
        ax[idx].set_xlabel("ΔI")
        ax[idx].set_ylabel("ΔQ")
        ax[idx].set_aspect("equal", adjustable="box")
        ax[idx].grid(alpha=0.2)

    return _savefig(save_path)


def plot_histograms(
    Y: np.ndarray,
    save_path: Path,
    *,
    bins: int = 160,
    max_points: int = 80_000,
    title: str = "Density (hist2d)",
) -> Path:
    """2-D histogram density plot of IQ data."""
    import matplotlib.pyplot as plt

    N = min(max_points, len(Y))
    Ys = Y[:N]

    plt.figure()
    plt.hist2d(Ys[:, 0], Ys[:, 1], bins=bins)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title(title)
    return _savefig(save_path)


def plot_psd(
    X: np.ndarray,
    Y_real: np.ndarray,
    Y_pred: np.ndarray,
    save_path: Path,
    *,
    nfft: int = 2048,
    max_points: int = 80_000,
    title: str = "Residual PSD comparison",
) -> Path:
    """Log-PSD comparison of real vs predicted residual."""
    import matplotlib.pyplot as plt
    from src.evaluation.metrics import _psd_log

    N = min(max_points, len(X))
    Dr = Y_real[:N] - X[:N]
    Dp = Y_pred[:N] - X[:N]

    cr = Dr[:, 0] + 1j * Dr[:, 1]
    cp = Dp[:, 0] + 1j * Dp[:, 1]
    psd_r = _psd_log(cr, nfft=nfft)
    psd_p = _psd_log(cp, nfft=nfft)

    plt.figure()
    plt.plot(psd_r, label="Δ real")
    plt.plot(psd_p, label="Δ pred")
    plt.xlabel("freq bin")
    plt.ylabel("log10 PSD")
    plt.title(title)
    plt.legend()
    return _savefig(save_path)


def plot_latent_activity(
    std_mu_p: np.ndarray,
    save_path: Path,
    *,
    active_dims: Optional[int] = None,
    title: Optional[str] = None,
) -> Path:
    """Bar chart of per-dimension latent activity (std of μ_p)."""
    import matplotlib.pyplot as plt

    if title is None:
        ad = active_dims if active_dims is not None else int(np.sum(std_mu_p > 0.05))
        title = f"Latent activity (active dims={ad})"

    plt.figure()
    plt.bar(np.arange(len(std_mu_p)), std_mu_p)
    plt.xlabel("latent dim")
    plt.ylabel("std(μ_p)")
    plt.title(title)
    return _savefig(save_path)


def plot_latent_kl(
    dims: np.ndarray,
    kl_qp: np.ndarray,
    kl_pN: np.ndarray,
    save_path: Path,
    *,
    title: str = "Latent KL per dimension",
) -> Path:
    """Line plot of KL divergence per latent dimension."""
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(dims, kl_qp, label="KL(q||p)")
    plt.plot(dims, kl_pN, label="KL(p||N)")
    plt.xlabel("latent dim")
    plt.ylabel("KL mean")
    plt.title(title)
    plt.legend()
    return _savefig(save_path)


def plot_training_history(
    history_dict: dict,
    save_path: Path,
    *,
    keys: Sequence[str] = (
        "loss", "val_loss", "recon_loss", "val_recon_loss",
        "kl_loss", "val_kl_loss",
    ),
    title: str = "Training history",
) -> Path:
    """Plot selected loss curves from a Keras history dict."""
    import matplotlib.pyplot as plt

    plt.figure()
    for col in keys:
        vals = history_dict.get(col)
        if vals is not None:
            plt.plot(vals, label=col)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    return _savefig(save_path)


def plot_summary_report(
    text: str,
    save_path: Path,
) -> Path:
    """Save a text-only summary as a PNG plot."""
    import matplotlib.pyplot as plt

    plt.figure()
    plt.axis("off")
    plt.text(0.02, 0.98, text, va="top", family="monospace")
    return _savefig(save_path)
