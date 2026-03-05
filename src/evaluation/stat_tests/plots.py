# -*- coding: utf-8 -*-
"""
src.evaluation.stat_tests.plots — Heatmaps and scatter plots for the
Statistical Fidelity Suite.

All functions accept a **pandas DataFrame** (the FDR-corrected stat
fidelity table, optionally enriched with ``dist_m`` / ``curr_mA`` /
``baseline_evm_%`` columns) and an output directory.

Plots produced
--------------
1. ``heatmap_mmd2.png``           — MMD² by (distance, current)
2. ``heatmap_qval_mmd.png``       — −log₁₀(q_MMD) by (distance, current)
3. ``heatmap_psd_dist.png``       — PSD L2 distance by (distance, current)
4. ``scatter_mmd2_vs_evm.png``    — MMD² vs baseline EVM (optional)

Commit: refactor(etapaA3).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _savefig(path: Path, dpi: int = 200) -> Path:
    """tight_layout → savefig → close.  Returns path written."""
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def _pivot_for_heatmap(
    df: pd.DataFrame,
    value_col: str,
    row_col: str = "dist_m",
    col_col: str = "curr_mA",
) -> Optional[pd.DataFrame]:
    """
    Pivot *df* into a 2-D matrix suitable for ``sns.heatmap``.

    Returns ``None`` when the grid cannot be formed (e.g. missing columns
    or all-NaN values).
    """
    needed = {value_col, row_col, col_col}
    if not needed.issubset(df.columns):
        return None
    sub = df.dropna(subset=[value_col, row_col, col_col])
    if sub.empty:
        return None
    piv = sub.pivot_table(
        index=row_col, columns=col_col, values=value_col, aggfunc="mean",
    )
    # Sort axes numerically (distance ascending, current ascending)
    piv = piv.sort_index(ascending=True)
    piv = piv[sorted(piv.columns)]
    return piv


def _heatmap(
    piv: pd.DataFrame,
    save_path: Path,
    *,
    title: str,
    cmap: str = "viridis",
    fmt: str = ".4f",
    cbar_label: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Path:
    """Render a single annotated heatmap from a pivoted DataFrame."""
    import matplotlib.pyplot as plt
    import matplotlib

    # Use non-interactive backend
    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(piv.columns)),
                                     max(4, 0.9 * len(piv.index))))

    # Try seaborn for prettier output; fall back to pure matplotlib
    try:
        import seaborn as sns
        sns.heatmap(
            piv, annot=True, fmt=fmt, cmap=cmap,
            vmin=vmin, vmax=vmax,
            linewidths=0.5, ax=ax,
            cbar_kws={"label": cbar_label},
        )
    except ImportError:
        im = ax.imshow(piv.values, cmap=cmap, aspect="auto",
                        vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{i}" for i in piv.index])
        # Annotate cells
        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                v = piv.iloc[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:{fmt.lstrip('.')}}", ha="center",
                            va="center", fontsize=8, color="white")
        fig.colorbar(im, ax=ax, label=cbar_label)

    ax.set_ylabel("distance (m)")
    ax.set_xlabel("current (mA)")
    ax.set_title(title)
    return _savefig(save_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_heatmap_mmd2(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    fname: str = "heatmap_mmd2.png",
) -> Optional[Path]:
    """Heatmap of MMD² by (distance, current)."""
    piv = _pivot_for_heatmap(df, "mmd2")
    if piv is None:
        return None
    return _heatmap(
        piv, out_dir / fname,
        title="MMD² by regime (distance × current)",
        cmap="YlOrRd",
        fmt=".5f",
        cbar_label="MMD²",
    )


def plot_heatmap_qval_mmd(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    fname: str = "heatmap_qval_mmd.png",
) -> Optional[Path]:
    """
    Heatmap of −log₁₀(q_MMD) by (distance, current).

    Higher values → more significant departure from real data.
    Values are clipped at 10 (i.e. q < 1e-10).
    """
    if "mmd_qval" not in df.columns:
        return None
    tmp = df.copy()
    tmp["_neg_log10_q"] = -np.log10(tmp["mmd_qval"].clip(lower=1e-10))
    piv = _pivot_for_heatmap(tmp, "_neg_log10_q")
    if piv is None:
        return None
    return _heatmap(
        piv, out_dir / fname,
        title="−log₁₀(q_MMD)  [higher → more significant]",
        cmap="RdYlGn_r",
        fmt=".2f",
        cbar_label="−log₁₀(q)",
        vmin=0,
    )


def plot_heatmap_psd_dist(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    fname: str = "heatmap_psd_dist.png",
) -> Optional[Path]:
    """Heatmap of PSD L2 distance by (distance, current)."""
    piv = _pivot_for_heatmap(df, "psd_dist")
    if piv is None:
        return None
    return _heatmap(
        piv, out_dir / fname,
        title="PSD L2 distance by regime (distance × current)",
        cmap="YlOrRd",
        fmt=".4f",
        cbar_label="PSD L2",
    )


def plot_scatter_mmd2_vs_evm(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    fname: str = "scatter_mmd2_vs_evm.png",
    evm_col: str = "baseline_evm_%",
) -> Optional[Path]:
    """
    Scatter of MMD² vs baseline EVM (%).

    Only rendered when *evm_col* is present and has valid values.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if evm_col not in df.columns or "mmd2" not in df.columns:
        return None
    sub = df.dropna(subset=["mmd2", evm_col])
    if sub.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        sub[evm_col], sub["mmd2"],
        c=sub.get("psd_dist", sub["mmd2"]),
        cmap="viridis", edgecolors="k", s=60, alpha=0.85,
    )
    # Annotate each point with regime_id
    if "regime_id" in sub.columns:
        for _, row in sub.iterrows():
            ax.annotate(
                row["regime_id"], (row[evm_col], row["mmd2"]),
                fontsize=7, alpha=0.7, textcoords="offset points",
                xytext=(5, 3),
            )
    ax.set_xlabel("Baseline EVM (%)")
    ax.set_ylabel("MMD²")
    ax.set_title("MMD² vs Baseline EVM — per regime")
    fig.colorbar(sc, ax=ax, label="PSD L2" if "psd_dist" in sub.columns else "MMD²")
    return _savefig(out_dir / fname)


# ---------------------------------------------------------------------------
# Convenience: generate all plots at once
# ---------------------------------------------------------------------------

def generate_all(
    df_sf: pd.DataFrame,
    out_dir: Path,
    *,
    df_summary: Optional[pd.DataFrame] = None,
) -> List[Path]:
    """
    Generate all stat-fidelity plots and return a list of created paths.

    Parameters
    ----------
    df_sf : DataFrame
        The FDR-corrected stat fidelity table (``stat_fidelity_by_regime``).
        Must contain at least: ``regime_id``, ``mmd2``, ``psd_dist``.
        Optionally: ``mmd_qval``, ``energy_qval``.
    out_dir : Path
        Directory where PNGs are written (created if needed).
    df_summary : DataFrame, optional
        The full ``summary_by_regime`` table.  When provided, regime grid
        coordinates (``dist_target_m``, ``curr_target_mA``) and baseline
        EVM (``baseline_evm_pred_%``) are merged into *df_sf* automatically.

    Returns
    -------
    list[Path]
        Paths of all plots that were successfully written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enrich df_sf with regime grid coordinates from df_summary
    df = df_sf.copy()
    if df_summary is not None and "regime_id" in df_summary.columns:
        _col_map = {
            "dist_target_m": "dist_m",
            "curr_target_mA": "curr_mA",
            "baseline_evm_pred_%": "baseline_evm_%",
        }
        # Only merge columns that are missing from df after renaming
        _need = {alias for alias in _col_map.values() if alias not in df.columns}
        _merge_src = [src for src, alias in _col_map.items()
                      if alias in _need and src in df_summary.columns]
        if _merge_src:
            _right = df_summary[["regime_id"] + _merge_src].drop_duplicates("regime_id")
            df = df.merge(_right, on="regime_id", how="left")
            df.rename(columns={k: v for k, v in _col_map.items() if k in df.columns},
                      inplace=True)

    created: List[Path] = []

    for fn in (plot_heatmap_mmd2, plot_heatmap_qval_mmd,
               plot_heatmap_psd_dist, plot_scatter_mmd2_vs_evm):
        try:
            p = fn(df, out_dir)
            if p is not None:
                created.append(p)
                print(f"   📈 {p.name}")
        except Exception as exc:
            print(f"⚠️  {fn.__name__} failed: {exc}")

    return created
