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
from src.training.logging import ensure_artifact_subdirs, write_artifact_manifest


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


def _corr_mean(Y_real: np.ndarray, Y_pred: np.ndarray) -> float:
    vals = []
    for axis in range(min(Y_real.shape[1], Y_pred.shape[1], 2)):
        yr = np.asarray(Y_real[:, axis], dtype=np.float64)
        yp = np.asarray(Y_pred[:, axis], dtype=np.float64)
        if np.std(yr) <= 1e-12 or np.std(yp) <= 1e-12:
            vals.append(np.nan)
            continue
        vals.append(float(np.corrcoef(yr, yp)[0, 1]))
    vals = np.asarray(vals, dtype=np.float64)
    if np.all(~np.isfinite(vals)):
        return float("nan")
    return float(np.nanmean(vals))


def _hist_jsd(Y_real: np.ndarray, Y_pred: np.ndarray, bins: int = 72) -> float:
    from scipy.spatial.distance import jensenshannon

    Yr = np.asarray(Y_real, dtype=np.float64)
    Yp = np.asarray(Y_pred, dtype=np.float64)
    low = np.minimum(Yr.min(axis=0), Yp.min(axis=0))
    high = np.maximum(Yr.max(axis=0), Yp.max(axis=0))
    if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)):
        return float("nan")
    if np.allclose(low, high):
        return 0.0
    hr, _, _ = np.histogram2d(Yr[:, 0], Yr[:, 1], bins=bins, range=[[low[0], high[0]], [low[1], high[1]]])
    hp, _, _ = np.histogram2d(Yp[:, 0], Yp[:, 1], bins=bins, range=[[low[0], high[0]], [low[1], high[1]]])
    pr = hr.ravel().astype(np.float64) + 1e-12
    pp = hp.ravel().astype(np.float64) + 1e-12
    pr /= pr.sum()
    pp /= pp.sum()
    return float(jensenshannon(pr, pp) ** 2)


def _acf_curve_complex(x: np.ndarray, max_lag: int = 128) -> np.ndarray:
    xc = np.asarray(x, dtype=np.complex128).ravel()
    xc = xc - np.mean(xc)
    if len(xc) == 0:
        return np.zeros(1, dtype=np.float64)
    denom = float(np.vdot(xc, xc).real)
    if denom <= 1e-12:
        return np.zeros(max_lag + 1, dtype=np.float64)
    out = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        if lag == 0:
            out[lag] = 1.0
        else:
            out[lag] = float(np.vdot(xc[:-lag], xc[lag:]).real / denom)
    return out


def _acf_mse(Xv: np.ndarray, Y_real: np.ndarray, Y_pred: np.ndarray, max_lag: int = 128) -> float:
    dr = (Y_real - Xv)[:, 0] + 1j * (Y_real - Xv)[:, 1]
    dp = (Y_pred - Xv)[:, 0] + 1j * (Y_pred - Xv)[:, 1]
    ar = _acf_curve_complex(dr, max_lag=max_lag)
    ap = _acf_curve_complex(dp, max_lag=max_lag)
    return float(np.mean((ar - ap) ** 2))


def _rolling_evm_curve(Xv: np.ndarray, Yv: np.ndarray, Yp: np.ndarray, n_windows: int = 80) -> tuple[np.ndarray, np.ndarray]:
    from src.evaluation.metrics import calculate_evm

    n = len(Xv)
    if n <= 0:
        return np.zeros(1), np.zeros(1)
    win = max(1, n // max(1, int(n_windows)))
    real_vals = []
    pred_vals = []
    for start in range(0, n, win):
        stop = min(n, start + win)
        if stop - start < 8:
            continue
        evm_real, _ = calculate_evm(Xv[start:stop], Yv[start:stop])
        evm_pred, _ = calculate_evm(Xv[start:stop], Yp[start:stop])
        real_vals.append(float(evm_real))
        pred_vals.append(float(evm_pred))
    if not real_vals:
        real_vals = [0.0]
        pred_vals = [0.0]
    return np.asarray(real_vals), np.asarray(pred_vals)


def _radar_norm(value: float, *, mode: str, ref_hi: float) -> float:
    if not np.isfinite(value):
        return 0.0
    ref_hi = max(float(ref_hi), 1e-9)
    if mode == "higher":
        return float(np.clip(value / ref_hi, 0.0, 1.0))
    return float(np.clip(1.0 - (value / ref_hi), 0.0, 1.0))


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
    groups = ensure_artifact_subdirs(
        plots_dir,
        groups=("reports", "core", "distribution", "latent", "training"),
    )

    created: List[Path] = []
    created.append(
        plot_overlay(
            Yv,
            Yp,
            groups["core"] / "overlay_constellation.png",
            title=f"{title_prefix} — constellation overlay",
        )
    )
    created.append(
        plot_residual_overlay(
            Xv,
            Yv,
            Yp,
            groups["core"] / "overlay_residual_delta.png",
            title=f"{title_prefix} — residual overlay",
        )
    )
    created.append(
        plot_histograms(
            Yv,
            groups["distribution"] / "density_y_real.png",
            title=f"{title_prefix} — density Y real",
        )
    )
    created.append(
        plot_histograms(
            Yp,
            groups["distribution"] / "density_y_pred.png",
            title=f"{title_prefix} — density Y pred",
        )
    )
    created.append(
        plot_psd(
            Xv,
            Yv,
            Yp,
            groups["distribution"] / "psd_residual_delta.png",
            nfft=int(psd_nfft),
            title=f"{title_prefix} — residual PSD",
        )
    )
    created.append(
        plot_latent_activity(
            std_mu_p,
            groups["latent"] / "latent_activity_std_mu_p.png",
            active_dims=int(np.sum(std_mu_p > 0.05)),
            title=f"{title_prefix} — latent activity",
        )
    )
    created.append(
        _plot_single_series(
            kl_dim_mean,
            groups["latent"] / "latent_kl_qp_per_dim.png",
            title=f"{title_prefix} — KL(q||p) per dimension",
            ylabel="KL(q||p)",
        )
    )
    if history_dict:
        created.append(
            plot_training_history(
                history_dict,
                groups["training"] / "training_history.png",
                title=f"{title_prefix} — training history",
            )
        )
    created.append(
        plot_summary_report(
            "\n".join(summary_lines),
            groups["reports"] / "summary_report.png",
        )
    )
    write_artifact_manifest(
        plots_dir,
        title="Candidate plot bundle",
        sections={
            "open_first": [
                groups["reports"] / "summary_report.png",
                groups["core"] / "overlay_constellation.png",
                groups["core"] / "overlay_residual_delta.png",
            ],
            "distribution": [
                groups["distribution"] / "density_y_real.png",
                groups["distribution"] / "density_y_pred.png",
                groups["distribution"] / "psd_residual_delta.png",
            ],
            "latent": [
                groups["latent"] / "latent_activity_std_mu_p.png",
                groups["latent"] / "latent_kl_qp_per_dim.png",
            ],
            "training": (
                [groups["training"] / "training_history.png"] if history_dict else []
            ),
        },
    )
    return created


def plot_legacy_metrics_comparison(
    *,
    Xv: np.ndarray,
    Yv: np.ndarray,
    Yp: np.ndarray,
    save_path: Path,
    model_label: str = "Champion",
) -> Path:
    """Legacy-style 2x2 comparison figure for the champion model."""
    import matplotlib.pyplot as plt
    from src.evaluation.metrics import calculate_evm, calculate_snr

    evm_real, _ = calculate_evm(Xv, Yv)
    evm_pred, _ = calculate_evm(Xv, Yp)
    snr_real = calculate_snr(Xv, Yv)
    snr_pred = calculate_snr(Xv, Yp)
    corr = _corr_mean(Yv, Yp)
    jsd = _hist_jsd(Yv, Yp)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Comparação de Métricas — Real vs {model_label}", fontsize=18)

    cats = ["Real", model_label]
    axes[0, 0].bar(cats, [evm_real, evm_pred], color=["#4C78A8", "#F58518"])
    axes[0, 0].set_title("EVM (%)")

    axes[0, 1].bar(cats, [snr_real, snr_pred], color=["#4C78A8", "#F58518"])
    axes[0, 1].set_title("SNR (dB)")

    axes[1, 0].bar(cats, [1.0, corr], color=["#4C78A8", "#F58518"])
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_title("Correlação média com Real")

    axes[1, 1].bar(cats, [0.0, jsd], color=["#4C78A8", "#F58518"])
    axes[1, 1].set_title("JSD vs Real")

    return _savefig(save_path)


def plot_legacy_radar(
    *,
    Xv: np.ndarray,
    Yv: np.ndarray,
    Yp: np.ndarray,
    save_path: Path,
    model_label: str = "Champion",
) -> Path:
    """Legacy-style radar chart for the champion model against a perfect reference."""
    import matplotlib.pyplot as plt
    from src.evaluation.metrics import calculate_evm, calculate_snr, residual_distribution_metrics

    evm_real, _ = calculate_evm(Xv, Yv)
    evm_pred, _ = calculate_evm(Xv, Yp)
    snr_real = calculate_snr(Xv, Yv)
    snr_pred = calculate_snr(Xv, Yp)
    corr = _corr_mean(Yv, Yp)
    jsd = _hist_jsd(Yv, Yp)
    acf_mse = _acf_mse(Xv, Yv, Yp)
    distm = residual_distribution_metrics(Xv, Yv, Yp)
    psd_mse = float(distm.get("delta_psd_l2", float("nan")))

    labels = [
        "EVM\n(invertido)",
        "SNR\n(norm)",
        "Correlação",
        "JSD\n(inv)",
        "ACF MSE\n(inv)",
        "PSD MSE\n(inv)",
    ]
    values = np.asarray(
        [
            _radar_norm(abs(evm_pred - evm_real), mode="lower", ref_hi=max(evm_real, 1.0)),
            _radar_norm(max(snr_pred, 0.0), mode="higher", ref_hi=max(snr_real, 1.0)),
            _radar_norm(corr, mode="higher", ref_hi=1.0),
            _radar_norm(jsd, mode="lower", ref_hi=max(jsd, 1e-3) * 2.0),
            _radar_norm(acf_mse, mode="lower", ref_hi=max(acf_mse, 1e-4) * 2.0),
            _radar_norm(psd_mse, mode="lower", ref_hi=max(psd_mse, 1e-3) * 2.0),
        ],
        dtype=np.float64,
    )
    perfect = np.ones_like(values)

    theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    theta = np.concatenate([theta, theta[:1]])
    vals = np.concatenate([values, values[:1]])
    perfect_vals = np.concatenate([perfect, perfect[:1]])

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(theta, vals, marker="o", label=model_label)
    ax.fill(theta, vals, alpha=0.25)
    ax.plot(theta, perfect_vals, "--", color="green", label="Perfeito")
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Radar — Desempenho (Quanto maior, melhor)", pad=18)
    ax.legend(loc="upper right")
    return _savefig(save_path)


def plot_legacy_analysis_dashboard(
    *,
    Xv: np.ndarray,
    Yv: np.ndarray,
    Yp: np.ndarray,
    mu_p: np.ndarray,
    std_mu_p: np.ndarray,
    kl_dim_mean: np.ndarray,
    summary_lines: Sequence[str],
    save_path: Path,
    title: str,
) -> Path:
    """Legacy-style multi-panel analysis board for the champion model."""
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats
    from src.evaluation.metrics import _psd_log, calculate_evm

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, y=0.98)
    ax = axes.ravel()

    n = min(len(Xv), len(Yv), len(Yp), 20000)
    Xs = Xv[:n]
    Yr = Yv[:n]
    Yc = Yp[:n]
    mu2 = mu_p[:n]

    ax[0].scatter(Xs[:, 0], Xs[:, 1], s=1, alpha=0.35)
    ax[0].set_title("X (sent)")
    ax[1].scatter(Yr[:, 0], Yr[:, 1], s=1, alpha=0.35)
    ax[1].set_title("Real")
    ax[2].scatter(Yc[:, 0], Yc[:, 1], s=1, alpha=0.35)
    ax[2].set_title("Champion")
    ax[3].scatter(Yr[:, 0], Yr[:, 1], s=1, alpha=0.25, label="Real")
    ax[3].scatter(Yc[:, 0], Yc[:, 1], s=1, alpha=0.25, label="Champion")
    ax[3].legend(loc="best", markerscale=4)
    ax[3].set_title("Overlay")

    if mu2.shape[1] >= 2:
        ax[4].scatter(mu2[:, 0], mu2[:, 1], s=1, alpha=0.35)
        ax[4].set_title("Latente μ_p (z0,z1)")
        ax[5].hist2d(mu2[:, 0], mu2[:, 1], bins=80)
        ax[5].set_title("Densidade μ_p (z0,z1)")
    else:
        ax[4].axis("off")
        ax[5].axis("off")

    ax[6].hist(Yr[:, 0], bins=80, density=True, alpha=0.45, label="real I")
    ax[6].hist(Yc[:, 0], bins=80, density=True, alpha=0.45, label="champ I")
    ax[6].set_title("Distribuições I")
    ax[6].legend(loc="best")
    ax[7].bar(np.arange(len(std_mu_p)), std_mu_p)
    ax[7].set_title("σ médio por dimensão")

    dr = Yr - Xs
    dp = Yc - Xs
    ax[8].hist(dr[:, 0], bins=80, density=True, alpha=0.45, label="Δ real I")
    ax[8].hist(dp[:, 0], bins=80, density=True, alpha=0.45, label="Δ pred I")
    ax[8].set_title("Erro I")
    ax[8].legend(loc="best")
    ax[9].hist(dr[:, 1], bins=80, density=True, alpha=0.45, label="Δ real Q")
    ax[9].hist(dp[:, 1], bins=80, density=True, alpha=0.45, label="Δ pred Q")
    ax[9].set_title("Erro Q")
    ax[9].legend(loc="best")
    ax[10].hist2d(dp[:, 0], dp[:, 1], bins=80)
    ax[10].set_title("Erro em I/Q")

    try:
        qq = scipy_stats.probplot(dp[:, 0], dist="norm")
        theo = qq[0][0]
        ordered = qq[0][1]
        slope, intercept = qq[1][0], qq[1][1]
        ax[11].plot(theo, ordered, "o", markersize=2)
        ax[11].plot(theo, slope * theo + intercept, "r-")
        ax[11].set_title("QQ plot (Erro I)")
    except Exception:
        ax[11].axis("off")

    evm_real_curve, evm_pred_curve = _rolling_evm_curve(Xs, Yr, Yc)
    ax[12].plot(evm_real_curve, label="Real")
    ax[12].plot(evm_pred_curve, label="Champion")
    ax[12].set_title("EVM ao longo do tempo")
    ax[12].legend(loc="best")

    cr = dr[:, 0] + 1j * dr[:, 1]
    cp = dp[:, 0] + 1j * dp[:, 1]
    ax[13].plot(_psd_log(cr, nfft=2048), label="Real")
    ax[13].plot(_psd_log(cp, nfft=2048), label="Champion")
    ax[13].set_title("PSD (Δ)")
    ax[13].legend(loc="best")

    ax[14].plot(np.arange(len(kl_dim_mean)), kl_dim_mean)
    ax[14].set_title("KL(q||p) por dimensão")

    ax[15].axis("off")
    ax[15].text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightsteelblue", alpha=0.9),
    )

    for axis in ax[:15]:
        axis.grid(True, alpha=0.2)
    return _savefig(save_path, dpi=160)


def save_legacy_champion_plots(
    *,
    plots_dir: Path,
    Xv: np.ndarray,
    Yv: np.ndarray,
    Yp: np.ndarray,
    mu_p: np.ndarray,
    std_mu_p: np.ndarray,
    kl_dim_mean: np.ndarray,
    summary_lines: Sequence[str],
    model_label: str = "Champion",
) -> List[Path]:
    """Save a legacy-style executive plot set for the champion model."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    groups = ensure_artifact_subdirs(plots_dir, groups=("legacy",))
    legacy_dir = groups["legacy"]

    created = [
        plot_overlay(
            Yv,
            Yp,
            legacy_dir / "constellation_overlay.png",
            title=f"Constellation Overlay — Real vs {model_label}",
        ),
        plot_legacy_metrics_comparison(
            Xv=Xv,
            Yv=Yv,
            Yp=Yp,
            save_path=legacy_dir / "comparacao_metricas_principais.png",
            model_label=model_label,
        ),
        plot_legacy_radar(
            Xv=Xv,
            Yv=Yv,
            Yp=Yp,
            save_path=legacy_dir / "radar_comparativo.png",
            model_label=model_label,
        ),
        plot_legacy_analysis_dashboard(
            Xv=Xv,
            Yv=Yv,
            Yp=Yp,
            mu_p=mu_p,
            std_mu_p=std_mu_p,
            kl_dim_mean=kl_dim_mean,
            summary_lines=summary_lines,
            save_path=legacy_dir / "analise_completa_vae.png",
            title=f"Análise Completa — {model_label}",
        ),
    ]
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
