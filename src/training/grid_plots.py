# -*- coding: utf-8 -*-
"""Plot helpers for grid-search candidates and ranking overviews."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from src.evaluation.plots import (
    plot_histograms,
    plot_latent_activity,
    plot_overlay,
    plot_psd,
    plot_residual_overlay,
    plot_summary_report,
    plot_training_history,
)


def _savefig(path: Path, dpi: int = 180) -> Path:
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def _plot_single_series(
    values: np.ndarray,
    save_path: Path,
    *,
    title: str,
    ylabel: str,
) -> Path:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(np.arange(len(values)), values, label=ylabel)
    plt.xlabel("latent dim")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    return _savefig(save_path)


def save_candidate_plot_bundle(
    *,
    plots_dir: Path,
    Xv: np.ndarray,
    Yv: np.ndarray,
    Yp: np.ndarray,
    std_mu_p: np.ndarray,
    kl_dim_mean: np.ndarray,
    history_dict: Dict[str, Sequence[float]],
    summary_lines: Sequence[str],
    psd_nfft: int,
    title_prefix: str,
) -> List[Path]:
    """Save the standard plot bundle for one tested grid candidate."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    created: List[Path] = []
    created.append(
        plot_overlay(
            Yv,
            Yp,
            plots_dir / "overlay_constellation.png",
            title=f"{title_prefix} — constellation overlay",
        )
    )
    created.append(
        plot_residual_overlay(
            Xv,
            Yv,
            Yp,
            plots_dir / "overlay_residual_delta.png",
            title=f"{title_prefix} — residual overlay",
        )
    )
    created.append(
        plot_histograms(
            Yv,
            plots_dir / "density_y_real.png",
            title=f"{title_prefix} — density Y real",
        )
    )
    created.append(
        plot_histograms(
            Yp,
            plots_dir / "density_y_pred.png",
            title=f"{title_prefix} — density Y pred",
        )
    )
    created.append(
        plot_psd(
            Xv,
            Yv,
            Yp,
            plots_dir / "psd_residual_delta.png",
            nfft=int(psd_nfft),
            title=f"{title_prefix} — residual PSD",
        )
    )
    created.append(
        plot_latent_activity(
            std_mu_p,
            plots_dir / "latent_activity_std_mu_p.png",
            active_dims=int(np.sum(std_mu_p > 0.05)),
            title=f"{title_prefix} — latent activity",
        )
    )
    created.append(
        _plot_single_series(
            kl_dim_mean,
            plots_dir / "latent_kl_qp_per_dim.png",
            title=f"{title_prefix} — KL(q||p) per dimension",
            ylabel="KL(q||p)",
        )
    )
    if history_dict:
        created.append(
            plot_training_history(
                history_dict,
                plots_dir / "training_history.png",
                title=f"{title_prefix} — training history",
            )
        )
    created.append(
        plot_summary_report(
            "\n".join(summary_lines),
            plots_dir / "summary_report.png",
        )
    )
    return created


def plot_top_models_score(
    df_results: pd.DataFrame,
    save_path: Path,
    *,
    top_k: int = 12,
) -> Path | None:
    """Horizontal bar chart of the best `score_v2` models."""
    import matplotlib.pyplot as plt

    if df_results.empty or "score_v2" not in df_results.columns:
        return None
    sub = (
        df_results.dropna(subset=["score_v2"])
        .sort_values("score_v2", ascending=True)
        .head(int(top_k))
        .copy()
    )
    if sub.empty:
        return None
    labels = [f"#{int(r.rank)} {r.tag}" for r in sub.itertuples()]
    plt.figure(figsize=(10, max(4.5, 0.45 * len(sub))))
    plt.barh(labels[::-1], sub["score_v2"].iloc[::-1], color="#2f5d80")
    plt.xlabel("score_v2 (lower is better)")
    plt.title("Top ranked grid-search models")
    return _savefig(save_path)


def plot_cov_vs_kurt(
    df_results: pd.DataFrame,
    save_path: Path,
) -> Path | None:
    """Scatter of covariance vs kurtosis error colored by score."""
    import matplotlib.pyplot as plt

    needed = {"delta_cov_fro", "delta_kurt_l2", "score_v2"}
    if not needed.issubset(df_results.columns):
        return None
    sub = df_results.dropna(subset=list(needed))
    if sub.empty:
        return None

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(
        sub["delta_cov_fro"],
        sub["delta_kurt_l2"],
        c=sub["score_v2"],
        cmap="viridis_r",
        edgecolors="k",
        s=65,
        alpha=0.85,
    )
    best = sub.sort_values("score_v2", ascending=True).iloc[0]
    plt.scatter(
        [best["delta_cov_fro"]],
        [best["delta_kurt_l2"]],
        marker="*",
        s=220,
        c="gold",
        edgecolors="black",
        label=f"best: {best['tag']}",
    )
    plt.xlabel("delta_cov_fro")
    plt.ylabel("delta_kurt_l2")
    plt.title("Grid-search models: covariance vs kurtosis error")
    plt.legend(loc="best")
    plt.colorbar(sc, label="score_v2")
    return _savefig(save_path)


def plot_evm_vs_snr_delta(
    df_results: pd.DataFrame,
    save_path: Path,
) -> Path | None:
    """Scatter of point-metric gaps for all tested models."""
    import matplotlib.pyplot as plt

    needed = {"delta_evm_%", "delta_snr_db", "score_v2"}
    if not needed.issubset(df_results.columns):
        return None
    sub = df_results.dropna(subset=list(needed))
    if sub.empty:
        return None

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(
        sub["delta_evm_%"],
        sub["delta_snr_db"],
        c=sub["score_v2"],
        cmap="plasma_r",
        edgecolors="k",
        s=65,
        alpha=0.85,
    )
    plt.axvline(0.0, color="gray", linewidth=1.0, alpha=0.5)
    plt.axhline(0.0, color="gray", linewidth=1.0, alpha=0.5)
    plt.xlabel("delta_evm_%")
    plt.ylabel("delta_snr_db")
    plt.title("Grid-search models: point-metric gaps")
    plt.colorbar(sc, label="score_v2")
    return _savefig(save_path)


def plot_score_vs_active_dims(
    df_results: pd.DataFrame,
    save_path: Path,
) -> Path | None:
    """Scatter of final score against active latent dimensions."""
    import matplotlib.pyplot as plt

    needed = {"active_dims", "score_v2"}
    if not needed.issubset(df_results.columns):
        return None
    sub = df_results.dropna(subset=list(needed))
    if sub.empty:
        return None

    plt.figure(figsize=(7, 5))
    plt.scatter(
        sub["active_dims"],
        sub["score_v2"],
        c=sub.get("delta_cov_fro", sub["score_v2"]),
        cmap="cividis_r",
        edgecolors="k",
        s=65,
        alpha=0.85,
    )
    plt.xlabel("active_dims")
    plt.ylabel("score_v2")
    plt.title("Grid-search models: score vs active latent dimensions")
    return _savefig(save_path)


def generate_gridsearch_overview_plots(
    df_results: pd.DataFrame,
    out_dir: Path,
) -> List[Path]:
    """Generate aggregate ranking plots across all tested grid models."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created: List[Path] = []
    for plot_fn, filename in (
        (plot_top_models_score, "top_models_score_v2.png"),
        (plot_cov_vs_kurt, "scatter_cov_vs_kurt.png"),
        (plot_evm_vs_snr_delta, "scatter_delta_evm_vs_delta_snr.png"),
        (plot_score_vs_active_dims, "scatter_score_vs_active_dims.png"),
    ):
        try:
            path = plot_fn(df_results, out_dir / filename)
            if path is not None:
                created.append(path)
        except Exception:
            continue
    return created
