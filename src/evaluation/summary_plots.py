# -*- coding: utf-8 -*-
"""Best-model heatmaps derived from tables/summary_by_regime.

This module intentionally focuses on the selected cVAE only.
It does not render baseline-vs-cVAE comparison panels anymore.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_CANONICAL_DISTANCES_M = (0.8, 1.0, 1.5)
_CANONICAL_CURRENTS_MA = (100.0, 300.0, 500.0, 700.0)


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
    piv = piv.reindex(
        index=_resolve_axis_order(piv.index, canonical=_CANONICAL_DISTANCES_M),
        columns=_resolve_axis_order(piv.columns, canonical=_CANONICAL_CURRENTS_MA),
    )
    return piv


def _resolve_axis_order(values, *, canonical: Sequence[float]) -> List[float]:
    observed = sorted(float(v) for v in values if pd.notna(v))
    if observed and set(observed).issubset(set(canonical)):
        return list(canonical)
    return observed


def _draw_heatmap(
    ax,
    piv: pd.DataFrame,
    *,
    title: str,
    cmap: str,
    fmt: str,
    annot_piv: Optional[pd.DataFrame] = None,
    cbar_label: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
) -> None:
    annot_data = annot_piv if annot_piv is not None else piv
    mask = piv.isna()
    try:
        import seaborn as sns

        sns.heatmap(
            piv,
            annot=annot_data,
            fmt=fmt,
            mask=mask,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            cbar_kws={"label": cbar_label},
            annot_kws={"color": "black", "fontsize": 10},
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
                display_v = annot_data.iloc[i, j]
                if pd.notna(display_v):
                    ax.text(
                        j,
                        i,
                        f"{display_v:{fmt.lstrip('.')}}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="black",
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


def _gate_heatmap_style(
    spec: dict,
    piv: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, Optional[float]]:
    annot_piv = piv.copy()
    color_piv = piv.copy()
    values = piv.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return color_piv, annot_piv, 0.0, 1.0, None

    mode = spec["mode"]
    if mode == "signed":
        color_piv = piv.abs()
        vmin = 0.0
        vmax = float(np.max(np.abs(values)))
        center = None
    elif mode == "higher_better":
        vmin = 0.0
        vmax = float(max(np.max(values), spec.get("threshold", 0.0)))
        center = float(spec.get("threshold")) if "threshold" in spec else None
    else:
        vmin = 0.0
        vmax = float(np.max(values))
        center = None

    if vmax <= vmin:
        vmax = vmin + 1e-12
    return color_piv, annot_piv, vmin, vmax, center


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
        "cols": ("cvae_delta_acf_l2", "delta_acf_l2"),
        "title": "Residual ACF mismatch (L2)",
        "fmt": ".4f",
        "cmap": "YlOrRd",
        "cbar": "ACF L2",
        "signed": False,
    },
    {
        "cols": ("cvae_rho_hetero_abs_gap",),
        "title": "Heteroscedasticity mismatch |Δρ|",
        "fmt": ".4f",
        "cmap": "YlOrRd",
        "cbar": "|Δρ|",
        "signed": False,
    },
    {
        "cols": ("cvae_stat_jsd", "stat_jsd"),
        "title": "Residual JSD (real vs cVAE)",
        "fmt": ".4f",
        "cmap": "YlOrRd",
        "cbar": "JSD (nats)",
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

_GATE_METRIC_SPECS = [
    {
        "cols": ("cvae_delta_evm_%", "delta_evm_%"),
        "title": "G1 — EVM error (pred - real) [pp]",
        "fmt": ".2f",
        "cmap": "RdYlGn_r",
        "cbar": "|ΔEVM| (pp)",
        "mode": "signed",
    },
    {
        "cols": ("cvae_delta_snr_db", "delta_snr_db"),
        "title": "G2 — SNR error (pred - real) [dB]",
        "fmt": ".2f",
        "cmap": "RdYlGn_r",
        "cbar": "|ΔSNR| (dB)",
        "mode": "signed",
    },
    {
        "cols": ("cvae_mean_rel_sigma",),
        "title": "G3 — mean error / sigma_real",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "mean_rel_sigma",
        "mode": "lower_better",
    },
    {
        "cols": ("cvae_cov_rel_var",),
        "title": "G3 — covariance error / var_real",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "cov_rel_var",
        "mode": "lower_better",
    },
    {
        "cols": ("cvae_psd_l2", "delta_psd_l2"),
        "title": "G4 — PSD mismatch",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "PSD L2",
        "mode": "lower_better",
    },
    {
        "cols": ("cvae_delta_skew_l2", "delta_skew_l2"),
        "title": "G5 — skew mismatch",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "skew L2",
        "mode": "lower_better",
    },
    {
        "cols": ("cvae_delta_kurt_l2", "delta_kurt_l2"),
        "title": "G5 — kurtosis mismatch",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "kurt L2",
        "mode": "lower_better",
    },
    {
        "cols": ("delta_jb_stat_rel",),
        "title": "G5 — JB relative mismatch",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "JB rel",
        "mode": "lower_better",
    },
    {
        "cols": ("stat_mmd_qval",),
        "title": "G6 — MMD q-value",
        "fmt": ".3f",
        "cmap": "RdYlGn",
        "cbar": "q_MMD",
        "mode": "higher_better",
        "threshold": 0.05,
    },
    {
        "cols": ("stat_energy_qval",),
        "title": "G6 — Energy q-value",
        "fmt": ".3f",
        "cmap": "RdYlGn",
        "cbar": "q_Energy",
        "mode": "higher_better",
        "threshold": 0.05,
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

    df_summary = df_summary.copy()
    if {
        "cvae_rho_hetero_real",
        "cvae_rho_hetero_pred",
    }.issubset(df_summary.columns):
        real = pd.to_numeric(df_summary["cvae_rho_hetero_real"], errors="coerce")
        pred = pd.to_numeric(df_summary["cvae_rho_hetero_pred"], errors="coerce")
        df_summary["cvae_rho_hetero_abs_gap"] = (pred - real).abs()

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


def plot_gate_metric_heatmaps(
    df_summary: pd.DataFrame,
    out_dir: Path,
    *,
    fname: str = "heatmap_gate_metrics_by_regime.png",
) -> Optional[Path]:
    """Render gate-focused regime heatmaps for the selected cVAE model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df_summary = df_summary.copy()
    resolved = []
    for spec in _GATE_METRIC_SPECS:
        col = _pick_column(df_summary, spec["cols"])
        if col is None:
            continue
        piv = _pivot_for_heatmap(df_summary, col)
        if piv is None:
            continue
        resolved.append((spec, piv))

    if not resolved:
        return None

    ncols = 2
    nrows = int(math.ceil(len(resolved) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (spec, piv) in zip(axes, resolved):
        color_piv, annot_piv, vmin, vmax, center = _gate_heatmap_style(spec, piv)
        vals = color_piv.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.axis("off")
            continue

        _draw_heatmap(
            ax,
            color_piv,
            title=spec["title"],
            cmap=spec["cmap"],
            fmt=spec["fmt"],
            annot_piv=annot_piv,
            cbar_label=spec["cbar"],
            vmin=vmin,
            vmax=vmax,
            center=center,
        )

    for ax in axes[len(resolved):]:
        ax.axis("off")

    fig.suptitle(
        "Gate metrics by regime — best model vs real",
        fontsize=15,
    )
    return _savefig(Path(out_dir) / fname)


def generate_all(df_summary: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Generate the canonical gate-focused regime heatmap bundle."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    try:
        p = plot_gate_metric_heatmaps(df_summary, out_dir)
        if p is not None:
            created.append(p)
            print(f"   📈 {p.name}")
    except Exception as exc:
        print(f"⚠️  plot_gate_metric_heatmaps failed: {exc}")
    return created
