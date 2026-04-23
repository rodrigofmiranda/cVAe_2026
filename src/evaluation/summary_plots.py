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
    norm=None,
    annot_fontsize: float = 10.0,
    tick_fontsize: float = 10.0,
    title_fontsize: float = 12.0,
) -> None:
    annot_data = annot_piv if annot_piv is not None else piv
    mask = piv.isna()
    try:
        import seaborn as sns

        kwargs = dict(
            annot=annot_data,
            fmt=fmt,
            mask=mask,
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            cbar_kws={"label": cbar_label},
            annot_kws={"color": "black", "fontsize": annot_fontsize},
        )
        if norm is not None:
            kwargs["norm"] = norm
        else:
            kwargs["vmin"] = vmin
            kwargs["vmax"] = vmax
            kwargs["center"] = center
        sns.heatmap(piv, **kwargs)
    except ImportError:
        kwargs = dict(cmap=cmap, aspect="auto")
        if norm is not None:
            kwargs["norm"] = norm
        else:
            kwargs["vmin"] = vmin
            kwargs["vmax"] = vmax
        im = ax.imshow(piv.values, **kwargs)
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
                        fontsize=annot_fontsize,
                        color="black",
                    )
        ax.figure.colorbar(im, ax=ax, label=cbar_label)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.set_ylabel("distance (m)")
    ax.set_xlabel("current (mA)")
    ax.set_title(title, fontsize=title_fontsize)


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

_EVAL_GATE_DIFF_SPECS = [
    {
        "cols": ("delta_evm_%", "cvae_delta_evm_%"),
        "title": "G1 — EVM error (pred - real) [pp]",
        "fmt": ".2f",
        "cmap": "RdYlGn_r",
        "cbar": "|ΔEVM| (pp)",
        "mode": "signed",
    },
    {
        "cols": ("delta_snr_db", "cvae_delta_snr_db"),
        "title": "G2 — SNR error (pred - real) [dB]",
        "fmt": ".2f",
        "cmap": "RdYlGn_r",
        "cbar": "|ΔSNR| (dB)",
        "mode": "signed",
    },
    {
        "cols": ("delta_mean_l2", "cvae_delta_mean_l2"),
        "title": "G3 — residual mean mismatch (L2)",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "L2",
        "mode": "lower_better",
    },
    {
        "cols": ("delta_cov_fro", "cvae_delta_cov_fro"),
        "title": "G3 — residual covariance mismatch (Fro)",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "Fro",
        "mode": "lower_better",
    },
    {
        "cols": ("delta_psd_l2", "cvae_psd_l2"),
        "title": "G4 — PSD mismatch (L2)",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "PSD L2",
        "mode": "lower_better",
    },
    {
        "cols": ("delta_skew_l2", "cvae_delta_skew_l2"),
        "title": "G5 — skew mismatch (L2)",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "skew L2",
        "mode": "lower_better",
    },
    {
        "cols": ("delta_kurt_l2", "cvae_delta_kurt_l2"),
        "title": "G5 — kurtosis mismatch (L2)",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "kurt L2",
        "mode": "lower_better",
    },
    {
        "cols": ("delta_jb_stat_rel",),
        "title": "G5 — JB relative mismatch (worst I/Q)",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "JB rel",
        "mode": "lower_better",
    },
]

_EVAL_GATE_SUPPLEMENTARY_SPECS = [
    {
        "cols": ("delta_acf_l2", "cvae_delta_acf_l2"),
        "title": "Supplementary — ACF mismatch (L2)",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "ACF L2",
        "mode": "lower_better",
    },
    {
        "cols": ("delta_coverage_95",),
        "title": "Supplementary — coverage@95% gap",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "|Δcoverage95|",
        "mode": "signed",
    },
    {
        "cols": ("rho_hetero_abs_gap", "cvae_rho_hetero_abs_gap"),
        "title": "Supplementary — heteroscedasticity gap |Δρ|",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "|Δρ|",
        "mode": "lower_better",
    },
    {
        "cols": ("stat_jsd", "cvae_stat_jsd"),
        "title": "Supplementary — residual JSD",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "JSD (nats)",
        "mode": "lower_better",
    },
]

    {
        "cols": ("stat_psd_dist",),
        "title": "Auxiliary — stat PSD distance",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "PSD dist",
        "mode": "lower_better",
    },
    {
        "cols": ("mi_aux_gap_rel",),
        "title": "Auxiliary — MI relative gap",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "rel gap",
        "mode": "lower_better",
    },
    {
        "cols": ("gmi_aux_gap_rel",),
        "title": "Auxiliary — GMI relative gap",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "rel gap",
        "mode": "lower_better",
    },
    {
        "cols": ("ngmi_aux_gap",),
        "title": "Auxiliary — NGMI gap",
        "fmt": ".3f",
        "cmap": "RdYlGn_r",
        "cbar": "ΔNGMI",
        "mode": "signed",
    },
]


def _build_twin_gate_threshold_specs() -> List[dict]:
    from src.evaluation.validation_summary import TWIN_GATE_THRESHOLDS

    return [
        {
            "ratio_col": "gate_g1_ratio",
            "title": f"G1 — rel EVM error / twin limit ({TWIN_GATE_THRESHOLDS['rel_evm_error']:.02f})",
            "fmt": "",
        },
        {
            "ratio_col": "gate_g2_ratio",
            "title": f"G2 — rel SNR error / twin limit ({TWIN_GATE_THRESHOLDS['rel_snr_error']:.02f})",
            "fmt": "",
        },
        {
            "ratio_col": "gate_g3_mean_ratio",
            "title": f"G3 — mean_rel_sigma / twin limit ({TWIN_GATE_THRESHOLDS['mean_rel_sigma']:.02f})",
            "fmt": "",
        },
        {
            "ratio_col": "gate_g3_cov_ratio",
            "title": f"G3 — cov_rel_var / twin limit ({TWIN_GATE_THRESHOLDS['cov_rel_var']:.02f})",
            "fmt": "",
        },
        {
            "ratio_col": "gate_g4_ratio",
            "title": f"G4 — PSD mismatch / twin limit ({TWIN_GATE_THRESHOLDS['delta_psd_l2']:.02f})",
            "fmt": "",
        },
        {
            "ratio_col": "gate_g5_skew_ratio",
            "title": f"G5 — skew mismatch / twin limit ({TWIN_GATE_THRESHOLDS['delta_skew_l2']:.02f})",
            "fmt": "",
        },
        {
            "ratio_col": "gate_g5_kurt_ratio",
            "title": f"G5 — kurtosis mismatch / twin limit ({TWIN_GATE_THRESHOLDS['delta_kurt_l2']:.02f})",
            "fmt": "",
        },
        {
            "ratio_col": "gate_g5_jb_ratio",
            "title": f"G5 — JB relative mismatch / twin limit ({TWIN_GATE_THRESHOLDS['delta_jb_stat_rel']:.02f})",
            "fmt": "",
        },
    ]


def _build_stat_screen_threshold_specs() -> List[dict]:
    from src.evaluation.validation_summary import TWIN_GATE_THRESHOLDS

    return [
        {
            "ratio_col": "gate_g6_mmd_ratio",
            "title": f"Stat screen — MMD q threshold ratio (q > {TWIN_GATE_THRESHOLDS['stat_qval']:.02f})",
            "fmt": "",
        },
        {
            "ratio_col": "gate_g6_energy_ratio",
            "title": f"Stat screen — Energy q threshold ratio (q > {TWIN_GATE_THRESHOLDS['stat_qval']:.02f})",
            "fmt": "",
        },
    ]


_STAT_SCREEN_PANEL_SPECS = [
    {
        "cols": ("stat_mmd_qval",),
        "title": "Stat screen — MMD q-value",
        "fmt": ".3f",
        "cmap": "RdYlGn",
        "cbar": "q_MMD",
        "mode": "higher_better",
        "threshold": 0.05,
    },
    {
        "cols": ("stat_energy_qval",),
        "title": "Stat screen — Energy q-value",
        "fmt": ".3f",
        "cmap": "RdYlGn",
        "cbar": "q_Energy",
        "mode": "higher_better",
        "threshold": 0.05,
    },
    {
        "cols": ("stat_mmd2",),
        "title": "Stat screen — MMD²",
        "fmt": ".4f",
        "cmap": "YlOrRd",
        "cbar": "MMD²",
        "mode": "lower_better",
    },
    {
        "cols": ("stat_energy",),
        "title": "Stat screen — Energy distance",
        "fmt": ".4f",
        "cmap": "YlOrRd",
        "cbar": "Energy",
        "mode": "lower_better",
    },
]


def _robust_gate_ratio_cap(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 2.0
    q95 = float(np.nanpercentile(finite, 95))
    return float(min(max(2.0, q95), 5.0))


def _format_ratio_annotations(piv: pd.DataFrame, *, cap: float) -> pd.DataFrame:
    annot = piv.copy().astype(object)
    for idx in piv.index:
        for col in piv.columns:
            value = piv.loc[idx, col]
            if pd.isna(value):
                annot.loc[idx, col] = ""
            elif float(value) > cap:
                annot.loc[idx, col] = f">{cap:.1f}x"
            else:
                annot.loc[idx, col] = f"{float(value):.2f}x"
    return annot


def _numeric_series_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _series_ratio(num: pd.Series, den: pd.Series | float) -> pd.Series:
    if np.isscalar(num):
        if np.isscalar(den):
            index = pd.RangeIndex(1)
        else:
            den_series = pd.to_numeric(den, errors="coerce")
            index = den_series.index
        num_series = pd.Series(float(num), index=index, dtype=float)
    else:
        num_series = pd.to_numeric(num, errors="coerce")

    if np.isscalar(den):
        den_series = pd.Series(float(den), index=num_series.index, dtype=float)
    else:
        den_series = pd.to_numeric(den, errors="coerce")
    return pd.Series(
        np.where(den_series > 0, num_series / den_series, np.nan),
        index=num_series.index,
        dtype=float,
    )


def _ensure_gate_ratio_columns(df_summary: pd.DataFrame) -> pd.DataFrame:
    from src.evaluation.validation_summary import TWIN_GATE_THRESHOLDS

    df = df_summary.copy()

    evm_rel = _numeric_series_or_nan(df, "cvae_rel_evm_error")
    if evm_rel.isna().all():
        evm_rel = _series_ratio(
            _numeric_series_or_nan(df, "delta_evm_%").abs(),
            _numeric_series_or_nan(df, "evm_real_%").abs(),
        )

    snr_rel = _numeric_series_or_nan(df, "cvae_rel_snr_error")
    if snr_rel.isna().all():
        snr_rel = _series_ratio(
            _numeric_series_or_nan(df, "delta_snr_db").abs(),
            _numeric_series_or_nan(df, "snr_real_db").abs(),
        )

    mean_rel_sigma = _numeric_series_or_nan(df, "cvae_mean_rel_sigma")
    if mean_rel_sigma.isna().all():
        mean_rel_sigma = _numeric_series_or_nan(df, "g3_mean_rel_sigma")
    if mean_rel_sigma.isna().all():
        sigma_real = np.sqrt(_numeric_series_or_nan(df, "var_real_delta"))
        mean_rel_sigma = _series_ratio(_numeric_series_or_nan(df, "delta_mean_l2"), sigma_real)

    cov_rel_var = _numeric_series_or_nan(df, "cvae_cov_rel_var")
    if cov_rel_var.isna().all():
        cov_rel_var = _numeric_series_or_nan(df, "g3_cov_rel_var")
    if cov_rel_var.isna().all():
        cov_rel_var = _series_ratio(
            _numeric_series_or_nan(df, "delta_cov_fro"),
            _numeric_series_or_nan(df, "var_real_delta"),
        )

    psd_metric = _numeric_series_or_nan(df, "cvae_psd_l2")
    if psd_metric.isna().all():
        psd_metric = _numeric_series_or_nan(df, "delta_psd_l2")

    skew_metric = _numeric_series_or_nan(df, "cvae_delta_skew_l2")
    if skew_metric.isna().all():
        skew_metric = _numeric_series_or_nan(df, "delta_skew_l2")

    kurt_metric = _numeric_series_or_nan(df, "cvae_delta_kurt_l2")
    if kurt_metric.isna().all():
        kurt_metric = _numeric_series_or_nan(df, "delta_kurt_l2")

    jb_metric = _numeric_series_or_nan(df, "delta_jb_stat_rel")

    mmd_q = _numeric_series_or_nan(df, "stat_mmd_qval")
    energy_q = _numeric_series_or_nan(df, "stat_energy_qval")

    derived = {
        "gate_g1_ratio": _series_ratio(evm_rel, TWIN_GATE_THRESHOLDS["rel_evm_error"]),
        "gate_g2_ratio": _series_ratio(snr_rel, TWIN_GATE_THRESHOLDS["rel_snr_error"]),
        "gate_g3_mean_ratio": _series_ratio(mean_rel_sigma, TWIN_GATE_THRESHOLDS["mean_rel_sigma"]),
        "gate_g3_cov_ratio": _series_ratio(cov_rel_var, TWIN_GATE_THRESHOLDS["cov_rel_var"]),
        "gate_g4_ratio": _series_ratio(psd_metric, TWIN_GATE_THRESHOLDS["delta_psd_l2"]),
        "gate_g5_skew_ratio": _series_ratio(skew_metric, TWIN_GATE_THRESHOLDS["delta_skew_l2"]),
        "gate_g5_kurt_ratio": _series_ratio(kurt_metric, TWIN_GATE_THRESHOLDS["delta_kurt_l2"]),
        "gate_g5_jb_ratio": _series_ratio(jb_metric, TWIN_GATE_THRESHOLDS["delta_jb_stat_rel"]),
        "gate_g6_mmd_ratio": _series_ratio(TWIN_GATE_THRESHOLDS["stat_qval"], mmd_q),
        "gate_g6_energy_ratio": _series_ratio(TWIN_GATE_THRESHOLDS["stat_qval"], energy_q),
    }
    for col, series in derived.items():
        if col not in df.columns or pd.to_numeric(df[col], errors="coerce").isna().all():
            df[col] = series
    return df


def _plot_metric_panel(
    df_summary: pd.DataFrame,
    out_dir: Path,
    *,
    specs: Sequence[dict],
    fname: str,
    panel_title: str,
    ncols: int = 2,
) -> Optional[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df_summary = df_summary.copy()
    resolved = []
    for spec in specs:
        col = _pick_column(df_summary, spec["cols"])
        if col is None:
            continue
        piv = _pivot_for_heatmap(df_summary, col)
        if piv is None:
            continue
        resolved.append((spec, piv))

    if not resolved:
        return None

    max_cols = max(len(piv.columns) for _, piv in resolved)
    max_rows = max(len(piv.index) for _, piv in resolved)
    nrows = int(math.ceil(len(resolved) / ncols))
    subplot_width = max(8.8, 0.95 * max_cols)
    subplot_height = max(4.8, 1.25 * max_rows + 1.2)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(subplot_width * ncols, subplot_height * nrows),
    )
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
            annot_fontsize=8.0,
            tick_fontsize=9.0,
            title_fontsize=12.0,
        )

    for ax in axes[len(resolved):]:
        ax.axis("off")

    fig.suptitle(panel_title, fontsize=15)
    return _savefig(Path(out_dir) / fname)


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


def plot_eval_gate_difference_heatmaps(
    df_summary: pd.DataFrame,
    out_dir: Path,
    *,
    fname: str = "heatmap_gate_differences_by_regime.png",
) -> Optional[Path]:
    """Render regime heatmaps for real-vs-model discrepancies in an eval batch."""
    return _plot_metric_panel(
        df_summary,
        out_dir,
        specs=_EVAL_GATE_DIFF_SPECS,
        fname=fname,
        panel_title="Gate-driving discrepancies by regime — real vs cVAE",
        ncols=2,
    )


def plot_eval_gate_supplementary_heatmaps(
    df_summary: pd.DataFrame,
    out_dir: Path,
    *,
    fname: str = "heatmap_auxiliary_analysis_by_regime.png",
) -> Optional[Path]:
    """Render supplementary regime heatmaps for eval batches."""
    return _plot_metric_panel(
        df_summary,
        out_dir,
        specs=_EVAL_GATE_SUPPLEMENTARY_SPECS,
        fname=fname,
        panel_title="Supplementary discrepancies by regime — real vs cVAE",
        ncols=2,
    )


def _plot_threshold_ratio_heatmaps(
    df_summary: pd.DataFrame,
    out_dir: Path,
    *,
    specs: Sequence[dict],
    fname: str,
    panel_title: str,
    legend_lines: Sequence[str],
) -> Optional[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    df_summary = _ensure_gate_ratio_columns(df_summary)
    resolved = []
    for spec in specs:
        ratio_col = spec["ratio_col"]
        if ratio_col not in df_summary.columns:
            continue
        piv = _pivot_for_heatmap(df_summary, ratio_col)
        if piv is None:
            continue
        resolved.append((spec, piv))

    if not resolved:
        return None

    max_cols = max(len(piv.columns) for _, piv in resolved)
    max_rows = max(len(piv.index) for _, piv in resolved)
    ncols = 2
    nrows = int(math.ceil(len(resolved) / ncols))
    subplot_width = max(8.8, 0.95 * max_cols)
    subplot_height = max(4.8, 1.25 * max_rows + 1.2)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(subplot_width * ncols, subplot_height * nrows),
    )
    axes = np.atleast_1d(axes).ravel()

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "gate_threshold",
        [
            (0.0, "#0b8f3a"),
            (0.55, "#b8e186"),
            (0.75, "#fee08b"),
            (0.9001, "#fdb863"),
            (1.0, "#b2182b"),
        ],
    )

    for ax, (spec, piv) in zip(axes, resolved):
        vals = piv.to_numpy(dtype=float)
        cap = _robust_gate_ratio_cap(vals)
        color_piv = piv.clip(upper=cap)
        annot_piv = _format_ratio_annotations(piv, cap=cap)
        norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=cap)
        _draw_heatmap(
            ax,
            color_piv,
            title=spec["title"],
            cmap=cmap,
            fmt=spec["fmt"],
            annot_piv=annot_piv,
            cbar_label="x gate limit",
            norm=norm,
            annot_fontsize=8.0,
            tick_fontsize=9.0,
            title_fontsize=12.0,
        )

    for ax in axes[len(resolved):]:
        ax.axis("off")

    legend_handles = [
        mpatches.Patch(color="#0b8f3a", label="within gate (< 1x)"),
        mpatches.Patch(color="#fee08b", label="near gate (~ 1x)"),
        mpatches.Patch(color="#b2182b", label="beyond gate (> 1x)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.018,
        "Color scale: green < 1x gate limit, yellow ~= 1x, red > 1x. "
        "Annotations show value/limit.\n"
        + "\n".join(str(line) for line in legend_lines if str(line).strip()),
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    fig.suptitle(panel_title, fontsize=15)
    path = Path(out_dir) / fname
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_eval_gate_threshold_heatmaps(
    df_summary: pd.DataFrame,
    out_dir: Path,
    *,
    fname: str = "heatmap_gate_metrics_by_regime.png",
) -> Optional[Path]:
    """Render threshold-aware G1..G5 heatmaps for the digital twin gates."""
    from src.evaluation.validation_summary import TWIN_GATE_THRESHOLDS

    specs = _build_twin_gate_threshold_specs()
    legend_lines = [
        (
            f"G1: rel EVM < {TWIN_GATE_THRESHOLDS['rel_evm_error']:.02f} | "
            f"G2: rel SNR < {TWIN_GATE_THRESHOLDS['rel_snr_error']:.02f} | "
            f"G3: mean < {TWIN_GATE_THRESHOLDS['mean_rel_sigma']:.02f} and "
            f"cov < {TWIN_GATE_THRESHOLDS['cov_rel_var']:.02f}"
        ),
        (
            f"G4: PSD < {TWIN_GATE_THRESHOLDS['delta_psd_l2']:.02f} | "
            f"G5: skew < {TWIN_GATE_THRESHOLDS['delta_skew_l2']:.02f}, "
            f"kurt < {TWIN_GATE_THRESHOLDS['delta_kurt_l2']:.02f}, "
            f"JB < {TWIN_GATE_THRESHOLDS['delta_jb_stat_rel']:.02f}"
        ),
        "This panel defines validation_status_twin and the digital-twin pass/fail decision.",
    ]
    return _plot_threshold_ratio_heatmaps(
        df_summary,
        out_dir,
        specs=specs,
        fname=fname,
        panel_title="Digital twin gates by regime (G1..G5 only)",
        legend_lines=legend_lines,
    )


def plot_eval_stat_screen_heatmaps(
    df_summary: pd.DataFrame,
    out_dir: Path,
    *,
    fname: str = "heatmap_stat_screen_by_regime.png",
) -> Optional[Path]:
    """Render auxiliary statistical-screen heatmaps kept outside the twin gates."""
    from src.evaluation.validation_summary import TWIN_GATE_THRESHOLDS

    ratio_plot = _plot_threshold_ratio_heatmaps(
        df_summary,
        out_dir,
        specs=_build_stat_screen_threshold_specs(),
        fname=fname,
        panel_title="Auxiliary statistical screen by regime (G6 only, not part of twin validation)",
        legend_lines=[
            (
                f"Stat screen: q_MMD > {TWIN_GATE_THRESHOLDS['stat_qval']:.02f} "
                f"and q_Energy > {TWIN_GATE_THRESHOLDS['stat_qval']:.02f}"
            ),
            "This panel defines stat_screen_pass / validation_status_full, not validation_status_twin.",
        ],
    )
    panel_plot = _plot_metric_panel(
        df_summary,
        out_dir,
        specs=_STAT_SCREEN_PANEL_SPECS,
        fname=fname.replace(".png", "_raw.png"),
        panel_title="Auxiliary statistical-screen metrics by regime (MMD / Energy)",
        ncols=2,
    )
    return ratio_plot or panel_plot


def plot_residual_signature_overview(
    df_signature: pd.DataFrame,
    out_dir: Path,
    *,
    fname: str = "residual_signature_overview.png",
) -> Optional[Path]:
    """Render a compact overview of the new residual signature by regime."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df_signature is None or df_signature.empty:
        return None

    def _rowwise_nanmax(series_list):
        stacked = pd.DataFrame(
            {
                idx: pd.to_numeric(series, errors="coerce")
                for idx, series in enumerate(series_list)
            }
        )
        return stacked.max(axis=1, skipna=True).to_numpy()

    df = df_signature.copy()
    df["var_ratio_max_abs_gap"] = _rowwise_nanmax(
        [
            np.abs(pd.to_numeric(df.get("var_ratio_I"), errors="coerce") - 1.0),
            np.abs(pd.to_numeric(df.get("var_ratio_Q"), errors="coerce") - 1.0),
        ]
    )
    df["tail_p3sigma_abs_gap_max"] = _rowwise_nanmax(
        [
            np.abs(pd.to_numeric(df.get("delta_tail_p3sigma_I"), errors="coerce")),
            np.abs(pd.to_numeric(df.get("delta_tail_p3sigma_Q"), errors="coerce")),
        ]
    )
    df["wasserstein_max"] = _rowwise_nanmax(
        [
            np.abs(pd.to_numeric(df.get("delta_wasserstein_I"), errors="coerce")),
            np.abs(pd.to_numeric(df.get("delta_wasserstein_Q"), errors="coerce")),
        ]
    )
    df["jb_rel_max"] = _rowwise_nanmax(
        [
            np.abs(pd.to_numeric(df.get("delta_jb_stat_rel_I"), errors="coerce")),
            np.abs(pd.to_numeric(df.get("delta_jb_stat_rel_Q"), errors="coerce")),
        ]
    )

    specs = [
        ("var_ratio_max_abs_gap", "Var ratio gap |pred/real - 1| (max I/Q)", "RdYlGn_r"),
        ("tail_p3sigma_abs_gap_max", "Tail p(>3σ_real) gap (max I/Q)", "RdYlGn_r"),
        ("wasserstein_max", "Wasserstein mismatch (max I/Q)", "RdYlGn_r"),
        ("jb_rel_max", "JB relative mismatch (max I/Q)", "RdYlGn_r"),
    ]
    resolved = []
    for col, title, cmap in specs:
        piv = _pivot_for_heatmap(df, col)
        if piv is None:
            continue
        annot = piv.copy().astype(object)
        for dist in piv.index:
            for curr in piv.columns:
                sub = df[
                    (pd.to_numeric(df["dist_target_m"], errors="coerce") == float(dist))
                    & (pd.to_numeric(df["curr_target_mA"], errors="coerce") == float(curr))
                ]
                if sub.empty:
                    annot.loc[dist, curr] = ""
                    continue
                row = sub.iloc[0]
                if col == "var_ratio_max_abs_gap":
                    annot.loc[dist, curr] = (
                        f"I={float(row.get('var_ratio_I', np.nan)):.2f}\n"
                        f"Q={float(row.get('var_ratio_Q', np.nan)):.2f}"
                    )
                elif col == "tail_p3sigma_abs_gap_max":
                    annot.loc[dist, curr] = (
                        f"I {float(row.get('tail_p3sigma_pred_I', np.nan)):.3f}/"
                        f"{float(row.get('tail_p3sigma_real_I', np.nan)):.3f}\n"
                        f"Q {float(row.get('tail_p3sigma_pred_Q', np.nan)):.3f}/"
                        f"{float(row.get('tail_p3sigma_real_Q', np.nan)):.3f}"
                    )
                elif col == "wasserstein_max":
                    annot.loc[dist, curr] = (
                        f"I={float(row.get('delta_wasserstein_I', np.nan)):.4f}\n"
                        f"Q={float(row.get('delta_wasserstein_Q', np.nan)):.4f}"
                    )
                else:
                    annot.loc[dist, curr] = (
                        f"I={float(row.get('delta_jb_stat_rel_I', np.nan)):.3f}\n"
                        f"Q={float(row.get('delta_jb_stat_rel_Q', np.nan)):.3f}"
                    )
        resolved.append((title, cmap, piv, annot))

    if not resolved:
        return None

    if all(
        not np.isfinite(pd.to_numeric(piv.to_numpy().ravel(), errors="coerce")).any()
        for _, _, piv, _ in resolved
    ):
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, cmap, piv, annot) in zip(axes, resolved):
        vals = piv.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        vmax = float(np.max(vals)) if vals.size else 1.0
        if vmax <= 0.0:
            vmax = 1.0
        _draw_heatmap(
            ax,
            piv,
            title=title,
            cmap=cmap,
            fmt="",
            annot_piv=annot,
            cbar_label=title,
            vmin=0.0,
            vmax=vmax,
            center=None,
        )
    for ax in axes[len(resolved):]:
        ax.axis("off")
    fig.suptitle("Residual signature overview by regime", fontsize=15)
    return _savefig(Path(out_dir) / fname)


def generate_all(df_summary: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Generate the canonical gate-focused regime heatmap bundle."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    try:
        twin_plot = plot_eval_gate_threshold_heatmaps(
            df_summary,
            out_dir,
            fname="heatmap_twin_gate_metrics_by_regime.png",
        )
        if twin_plot is None:
            twin_plot = plot_gate_metric_heatmaps(df_summary, out_dir)
        if twin_plot is not None:
            created.append(twin_plot)
            print(f"   📈 {twin_plot.name}")
            legacy_plot = plot_eval_gate_threshold_heatmaps(
                df_summary,
                out_dir,
                fname="heatmap_gate_metrics_by_regime.png",
            )
            if legacy_plot is not None:
                created.append(legacy_plot)
        stat_plot = plot_eval_stat_screen_heatmaps(df_summary, out_dir)
        if stat_plot is not None:
            created.append(stat_plot)
            print(f"   📈 {Path(stat_plot).name}")
            raw_stat_plot = out_dir / "heatmap_stat_screen_by_regime_raw.png"
            if raw_stat_plot.exists():
                created.append(raw_stat_plot)
                print(f"   📈 {raw_stat_plot.name}")
        aux_plot = plot_eval_gate_supplementary_heatmaps(df_summary, out_dir)
        if aux_plot is not None:
            created.append(aux_plot)
            print(f"   📈 {aux_plot.name}")
    except Exception as exc:
        print(f"⚠️  plot_gate_metric_heatmaps failed: {exc}")
    return created
