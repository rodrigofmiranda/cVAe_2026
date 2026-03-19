# -*- coding: utf-8 -*-
"""Best-model heatmaps derived from tables/summary_by_regime.

This module intentionally focuses on the selected cVAE only.
It does not render baseline-vs-cVAE comparison panels anymore.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence

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


def _draw_heatmap(
    ax,
    piv: pd.DataFrame,
    *,
    title: str,
    cmap: str,
    fmt: str,
    cbar_label: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
) -> None:
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
                    ax.text(
                        j,
                        i,
                        f"{v:{fmt.lstrip('.')}}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                    )
        ax.figure.colorbar(im, ax=ax, label=cbar_label)
    ax.set_ylabel("distance (m)")
    ax.set_xlabel("current (mA)")
    ax.set_title(title)


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


_METRIC_SPECS = [
    {
        "cols": ("cvae_delta_evm_%", "delta_evm_%"),
        "title": "cVAE vs real — ΔEVM (pp)",
        "fmt": ".2f",
        "cmap": "RdBu_r",
        "cbar": "ΔEVM (pp)",
        "signed": True,
    },
    {
        "cols": ("cvae_delta_snr_db", "delta_snr_db"),
        "title": "cVAE vs real — ΔSNR (dB)",
        "fmt": ".2f",
        "cmap": "RdBu",
        "cbar": "ΔSNR (dB)",
        "signed": True,
    },
    {
        "cols": ("cvae_delta_mean_l2", "delta_mean_l2"),
        "title": "Residual mean mismatch (L2)",
        "fmt": ".4f",
        "cmap": "YlOrRd",
        "cbar": "L2",
        "signed": False,
    },
    {
        "cols": ("cvae_delta_cov_fro", "delta_cov_fro"),
        "title": "Residual covariance mismatch (Fro)",
        "fmt": ".4f",
        "cmap": "YlOrRd",
        "cbar": "Fro",
        "signed": False,
    },
    {
        "cols": ("cvae_delta_skew_l2", "delta_skew_l2"),
        "title": "Residual skew mismatch (L2)",
        "fmt": ".4f",
        "cmap": "YlOrRd",
        "cbar": "L2",
        "signed": False,
    },
    {
        "cols": ("cvae_delta_kurt_l2", "delta_kurt_l2"),
        "title": "Residual kurtosis mismatch (L2)",
        "fmt": ".4f",
        "cmap": "YlOrRd",
        "cbar": "L2",
        "signed": False,
    },
    {
        "cols": ("cvae_psd_l2", "delta_psd_l2"),
        "title": "Residual PSD mismatch (L2)",
        "fmt": ".4f",
        "cmap": "YlOrRd",
        "cbar": "PSD L2",
        "signed": False,
    },
    {
        "cols": ("stat_mmd2",),
        "title": "MMD² (real vs cVAE)",
        "fmt": ".4f",
        "cmap": "YlOrRd",
        "cbar": "MMD²",
        "signed": False,
    },
    {
        "cols": ("stat_energy",),
        "title": "Energy distance (real vs cVAE)",
        "fmt": ".4f",
        "cmap": "YlOrRd",
        "cbar": "Energy",
        "signed": False,
    },
    {
        "cols": ("stat_psd_dist",),
        "title": "Stat PSD distance (real vs cVAE)",
        "fmt": ".4f",
        "cmap": "YlOrRd",
        "cbar": "PSD dist",
        "signed": False,
    },
]


def plot_vae_vs_real_metric_diffs(
    df_summary: pd.DataFrame,
    out_dir: Path,
    *,
    fname: str = "heatmap_vae_vs_real_metric_diffs.png",
) -> Optional[Path]:
    """Render best-model heatmaps of real-vs-cVAE metric differences."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resolved = []
    for spec in _METRIC_SPECS:
        col = _pick_column(df_summary, spec["cols"])
        if col is None:
            continue
        piv = _pivot_for_heatmap(df_summary, col)
        if piv is None:
            continue
        resolved.append((spec, piv))

    if not resolved:
        return None

    ncols = 3
    nrows = int(math.ceil(len(resolved) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (spec, piv) in zip(axes, resolved):
        vals = piv.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.axis("off")
            continue
        if spec["signed"]:
            vmax = float(np.max(np.abs(vals)))
            vmin = -vmax
            center = 0.0
        else:
            vmin = 0.0
            vmax = float(np.max(vals))
            center = None
        _draw_heatmap(
            ax,
            piv,
            title=spec["title"],
            cmap=spec["cmap"],
            fmt=spec["fmt"],
            cbar_label=spec["cbar"],
            vmin=vmin,
            vmax=vmax,
            center=center,
        )

    for ax in axes[len(resolved):]:
        ax.axis("off")

    fig.suptitle(
        "Best model only — real vs cVAE metric differences by regime",
        fontsize=15,
    )
    return _savefig(Path(out_dir) / fname)


def generate_all(df_summary: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Generate only best-model real-vs-cVAE heatmaps."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    try:
        p = plot_vae_vs_real_metric_diffs(df_summary, out_dir)
        if p is not None:
            created.append(p)
            print(f"   📈 {p.name}")
    except Exception as exc:
        print(f"⚠️  plot_vae_vs_real_metric_diffs failed: {exc}")
    return created
