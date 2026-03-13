# -*- coding: utf-8 -*-
"""Summary heatmaps derived from tables/summary_by_regime."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _savefig(path: Path, dpi: int = 200) -> Path:
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def _pivot_for_heatmap(
    df: pd.DataFrame,
    value_col: str,
    row_col: str = "dist_target_m",
    col_col: str = "curr_target_mA",
) -> Optional[pd.DataFrame]:
    needed = {value_col, row_col, col_col}
    if not needed.issubset(df.columns):
        return None
    sub = df.dropna(subset=[value_col, row_col, col_col])
    if sub.empty:
        return None
    piv = sub.pivot_table(
        index=row_col, columns=col_col, values=value_col, aggfunc="mean",
    )
    piv = piv.sort_index(ascending=True)
    piv = piv[sorted(piv.columns)]
    return piv


def _draw_heatmap(ax, piv: pd.DataFrame, *, title: str, cmap: str, fmt: str,
                  cbar_label: str = "", vmin: float = None, vmax: float = None,
                  center: float = None) -> None:
    try:
        import seaborn as sns

        sns.heatmap(
            piv,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": cbar_label},
        )
    except ImportError:
        im = ax.imshow(
            piv.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax,
        )
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:.0f}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{i:.1f}" for i in piv.index])
        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                v = piv.iloc[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:{fmt.lstrip('.')}}", ha="center",
                            va="center", fontsize=8, color="white")
        ax.figure.colorbar(im, ax=ax, label=cbar_label)
    ax.set_ylabel("distance (m)")
    ax.set_xlabel("current (mA)")
    ax.set_title(title)


def plot_evm_real_vs_models(
    df_summary: pd.DataFrame,
    out_dir: Path,
    *,
    fname: str = "heatmap_evm_real_vs_models.png",
) -> Optional[Path]:
    """Render one panel comparing real, baseline and cVAE EVM across regimes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = _pivot_for_heatmap(df_summary, "evm_real_%")
    baseline = _pivot_for_heatmap(df_summary, "baseline_evm_pred_%")
    cvae_col = "cvae_evm_pred_%" if "cvae_evm_pred_%" in df_summary.columns else "evm_pred_%"
    cvae = _pivot_for_heatmap(df_summary, cvae_col)
    if real is None and baseline is None and cvae is None:
        return None

    pivots = [p for p in (real, baseline, cvae) if p is not None]
    all_vals = np.concatenate([p.to_numpy().ravel() for p in pivots])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        return None
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    entries = [
        (real, "Canal real — EVM (%)"),
        (baseline, "Baseline — EVM predito (%)"),
        (cvae, "cVAE — EVM predito (%)"),
    ]
    for ax, (piv, title) in zip(axes, entries):
        if piv is None:
            ax.axis("off")
            ax.set_title(f"{title}\n(no data)")
            continue
        _draw_heatmap(
            ax, piv,
            title=title,
            cmap="YlOrRd",
            fmt=".2f",
            cbar_label="EVM (%)",
            vmin=vmin,
            vmax=vmax,
        )

    fig.suptitle("EVM por regime — canal real vs baseline vs cVAE", fontsize=14)
    return _savefig(Path(out_dir) / fname)


def plot_delta_evm_models(
    df_summary: pd.DataFrame,
    out_dir: Path,
    *,
    fname: str = "heatmap_delta_evm_models.png",
) -> Optional[Path]:
    """Render ΔEVM heatmaps for baseline and cVAE across regimes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = _pivot_for_heatmap(df_summary, "baseline_delta_evm_%")
    cvae_col = "cvae_delta_evm_%" if "cvae_delta_evm_%" in df_summary.columns else "delta_evm_%"
    cvae = _pivot_for_heatmap(df_summary, cvae_col)
    if baseline is None and cvae is None:
        return None

    pivots = [p for p in (baseline, cvae) if p is not None]
    all_vals = np.concatenate([p.to_numpy().ravel() for p in pivots])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        return None
    vmax = float(np.max(np.abs(all_vals)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    entries = [
        (baseline, "Baseline — ΔEVM (pp)"),
        (cvae, "cVAE — ΔEVM (pp)"),
    ]
    for ax, (piv, title) in zip(axes, entries):
        if piv is None:
            ax.axis("off")
            ax.set_title(f"{title}\n(no data)")
            continue
        _draw_heatmap(
            ax, piv,
            title=title,
            cmap="RdYlGn_r",
            fmt=".2f",
            cbar_label="ΔEVM (pp)",
            vmin=-vmax,
            vmax=vmax,
            center=0.0,
        )

    fig.suptitle("ΔEVM por regime — baseline vs cVAE", fontsize=14)
    return _savefig(Path(out_dir) / fname)


def plot_abs_delta_evm_vs_real_models(
    df_summary: pd.DataFrame,
    out_dir: Path,
    *,
    fname: str = "heatmap_abs_delta_evm_vs_real_models.png",
) -> Optional[Path]:
    """Render absolute EVM error vs the real channel for baseline and cVAE.

    This is the most direct diagnostic for "which regimes are outside the
    prediction" because color encodes the magnitude of the EVM mismatch with
    the real channel, independent of sign.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "baseline_delta_evm_%" not in df_summary.columns and "cvae_delta_evm_%" not in df_summary.columns:
        return None

    df_plot = df_summary.copy()
    if "baseline_delta_evm_%" in df_plot.columns:
        df_plot["baseline_abs_delta_evm_%"] = pd.to_numeric(
            df_plot["baseline_delta_evm_%"], errors="coerce",
        ).abs()
    if "cvae_delta_evm_%" in df_plot.columns:
        df_plot["cvae_abs_delta_evm_%"] = pd.to_numeric(
            df_plot["cvae_delta_evm_%"], errors="coerce",
        ).abs()
    elif "delta_evm_%" in df_plot.columns:
        df_plot["cvae_abs_delta_evm_%"] = pd.to_numeric(
            df_plot["delta_evm_%"], errors="coerce",
        ).abs()

    baseline = _pivot_for_heatmap(df_plot, "baseline_abs_delta_evm_%")
    cvae = _pivot_for_heatmap(df_plot, "cvae_abs_delta_evm_%")
    if baseline is None and cvae is None:
        return None

    pivots = [p for p in (baseline, cvae) if p is not None]
    all_vals = np.concatenate([p.to_numpy().ravel() for p in pivots])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        return None
    vmin = 0.0
    vmax = float(np.max(all_vals))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    entries = [
        (baseline, "Baseline/AWGN — |EVM_pred - EVM_real| (pp)"),
        (cvae, "cVAE — |EVM_pred - EVM_real| (pp)"),
    ]
    for ax, (piv, title) in zip(axes, entries):
        if piv is None:
            ax.axis("off")
            ax.set_title(f"{title}\n(no data)")
            continue
        _draw_heatmap(
            ax,
            piv,
            title=title,
            cmap="YlOrRd",
            fmt=".2f",
            cbar_label="|ΔEVM| (pp)",
            vmin=vmin,
            vmax=vmax,
        )

    fig.suptitle("Erro absoluto de EVM vs canal real — baseline/AWGN vs cVAE", fontsize=14)
    return _savefig(Path(out_dir) / fname)


def generate_all(df_summary: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Generate summary heatmaps and return created paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    for fn in (
        plot_evm_real_vs_models,
        plot_delta_evm_models,
        plot_abs_delta_evm_vs_real_models,
    ):
        try:
            p = fn(df_summary, out_dir)
            if p is not None:
                created.append(p)
                print(f"   📈 {p.name}")
        except Exception as exc:
            print(f"⚠️  {fn.__name__} failed: {exc}")
    return created
