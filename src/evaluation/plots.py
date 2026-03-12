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
