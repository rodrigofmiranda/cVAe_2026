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


def _acf_curve_real(x: np.ndarray, max_lag: int = 64) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - np.mean(x)
    if len(x) == 0:
        return np.zeros(1, dtype=np.float64)
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return np.zeros(max_lag + 1, dtype=np.float64)
    out = np.empty(max_lag + 1, dtype=np.float64)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        out[lag] = float(np.dot(x[:-lag], x[lag:]) / denom)
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


def _compact_layer_sizes(value: object) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return "-".join(str(int(v)) for v in value)
    text = str(value)
    return text.replace("[", "").replace("]", "").replace(", ", "-")


def _row_flags_short(row: pd.Series) -> str:
    mapping = [
        ("flag_posterior_collapse", "collapse"),
        ("flag_undertrained", "under"),
        ("flag_overfit", "overfit"),
        ("flag_unstable", "unstable"),
        ("flag_lr_floor", "lr_floor"),
    ]
    active = [label for col, label in mapping if bool(row.get(col, False))]
    return ",".join(active) if active else "none"


def _row_recommendations_short(row: pd.Series) -> str:
    cols = [
        "recommend_lr",
        "recommend_beta_free_bits",
        "recommend_latent_dim",
        "recommend_capacity",
        "recommend_architecture",
        "recommend_epochs_patience",
    ]
    active = [
        str(row.get(col, "keep"))
        for col in cols
        if str(row.get(col, "keep")) != "keep"
    ]
    return " | ".join(active) if active else "keep"


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


def save_champion_analysis_dashboard(
    *,
    plots_dir: Path,
    Xv: np.ndarray,
    Yv: np.ndarray,
    Yp: np.ndarray,
    std_mu_p: np.ndarray,
    kl_dim_mean: np.ndarray,
    summary_lines: Sequence[str],
    model_label: str = "Champion",
    title: str = "Champion Analysis Dashboard",
) -> Path:
    """Save a single comprehensive dashboard for the winning model only."""
    import matplotlib.pyplot as plt
    from src.evaluation.metrics import (
        _psd_log,
        _skew_kurt,
        calculate_evm,
        calculate_snr,
        residual_distribution_metrics,
    )

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    groups = ensure_artifact_subdirs(plots_dir, groups=("champion",))
    out_path = groups["champion"] / "analysis_dashboard.png"

    n = min(len(Xv), len(Yv), len(Yp), 40_000)
    Xs = np.asarray(Xv[:n], dtype=np.float64)
    Yr = np.asarray(Yv[:n], dtype=np.float64)
    Yc = np.asarray(Yp[:n], dtype=np.float64)
    Dr = Yr - Xs
    Dc = Yc - Xs

    evm_real, _ = calculate_evm(Xs, Yr)
    evm_pred, _ = calculate_evm(Xs, Yc)
    snr_real = calculate_snr(Xs, Yr)
    snr_pred = calculate_snr(Xs, Yc)
    distm = residual_distribution_metrics(Xs, Yr, Yc)
    mean_r = np.mean(Dr, axis=0)
    mean_c = np.mean(Dc, axis=0)
    std_r = np.std(Dr, axis=0)
    std_c = np.std(Dc, axis=0)
    skew_r, kurt_r = _skew_kurt(Dr)
    skew_c, kurt_c = _skew_kurt(Dc)

    acf_i_real = _acf_curve_real(Dr[:, 0], max_lag=64)
    acf_i_pred = _acf_curve_real(Dc[:, 0], max_lag=64)
    acf_q_real = _acf_curve_real(Dr[:, 1], max_lag=64)
    acf_q_pred = _acf_curve_real(Dc[:, 1], max_lag=64)

    psd_i_real = _psd_log(Dr[:, 0], nfft=2048)
    psd_i_pred = _psd_log(Dc[:, 0], nfft=2048)
    psd_q_real = _psd_log(Dr[:, 1], nfft=2048)
    psd_q_pred = _psd_log(Dc[:, 1], nfft=2048)

    fig, axes = plt.subplots(6, 3, figsize=(18, 28))
    fig.suptitle(title, fontsize=18, y=0.995)
    ax = axes.ravel()

    width = 0.35
    x_main = np.arange(2)
    ax[0].bar(x_main - width / 2, [evm_real, snr_real], width=width, label="Canal real")
    ax[0].bar(x_main + width / 2, [evm_pred, snr_pred], width=width, label=model_label)
    ax[0].set_xticks(x_main)
    ax[0].set_xticklabels(["EVM (%)", "SNR (dB)"])
    ax[0].set_title("Primary Performance Metrics")
    ax[0].legend(loc="best")

    x_mean_std = np.arange(4)
    ax[1].bar(
        x_mean_std - width / 2,
        [mean_r[0], mean_r[1], std_r[0], std_r[1]],
        width=width,
        label="Canal real",
    )
    ax[1].bar(
        x_mean_std + width / 2,
        [mean_c[0], mean_c[1], std_c[0], std_c[1]],
        width=width,
        label=model_label,
    )
    ax[1].set_xticks(x_mean_std)
    ax[1].set_xticklabels(["Mean I", "Mean Q", "Std I", "Std Q"])
    ax[1].set_title("Residual Mean & Std (I & Q)")
    ax[1].legend(loc="best")

    x_mom = np.arange(4)
    ax[2].bar(
        x_mom - width / 2,
        [skew_r[0], skew_r[1], kurt_r[0], kurt_r[1]],
        width=width,
        label="Canal real",
    )
    ax[2].bar(
        x_mom + width / 2,
        [skew_c[0], skew_c[1], kurt_c[0], kurt_c[1]],
        width=width,
        label=model_label,
    )
    ax[2].set_xticks(x_mom)
    ax[2].set_xticklabels(["Skew I", "Skew Q", "Kurt I", "Kurt Q"])
    ax[2].set_title("Residual Higher-Order Moments (I & Q)")
    ax[2].legend(loc="best")

    fidelity_labels = ["Δmean", "Δcov", "PSD L2", "ACF L2", "Skew L2", "Kurt L2"]
    fidelity_vals = [
        distm["delta_mean_l2"],
        distm["delta_cov_fro"],
        distm["delta_psd_l2"],
        distm.get("delta_acf_l2", float("nan")),
        distm["delta_skew_l2"],
        distm["delta_kurt_l2"],
    ]
    ax[3].bar(np.arange(len(fidelity_labels)), fidelity_vals, color="#C98A18")
    ax[3].set_xticks(np.arange(len(fidelity_labels)))
    ax[3].set_xticklabels(fidelity_labels, rotation=25, ha="right")
    ax[3].set_title("Residual Fidelity Metrics")

    axis_mismatch_labels = ["|Δmean I|", "|Δmean Q|", "|Δstd I|", "|Δstd Q|", "W1 I", "W1 Q"]
    axis_mismatch_vals = [
        abs(float(distm.get("delta_mean_I", float("nan")))),
        abs(float(distm.get("delta_mean_Q", float("nan")))),
        abs(float(distm.get("delta_std_I", float("nan")))),
        abs(float(distm.get("delta_std_Q", float("nan")))),
        float(distm.get("delta_wasserstein_I", float("nan"))),
        float(distm.get("delta_wasserstein_Q", float("nan"))),
    ]
    ax[4].bar(np.arange(len(axis_mismatch_labels)), axis_mismatch_vals, color="#3D8C40")
    ax[4].set_xticks(np.arange(len(axis_mismatch_labels)))
    ax[4].set_xticklabels(axis_mismatch_labels, rotation=25, ha="right")
    ax[4].set_title("Axis-wise Marginal Mismatch")

    axis_shape_labels = ["|Δskew I|", "|Δskew Q|", "|Δkurt I|", "|Δkurt Q|", "ΔJB I", "ΔJB Q"]
    axis_shape_vals = [
        abs(float(distm.get("delta_skew_I", float("nan")))),
        abs(float(distm.get("delta_skew_Q", float("nan")))),
        abs(float(distm.get("delta_kurt_I", float("nan")))),
        abs(float(distm.get("delta_kurt_Q", float("nan")))),
        float(distm.get("delta_jb_log10p_I", float("nan"))),
        float(distm.get("delta_jb_log10p_Q", float("nan"))),
    ]
    ax[5].bar(np.arange(len(axis_shape_labels)), axis_shape_vals, color="#8B4E96")
    ax[5].set_xticks(np.arange(len(axis_shape_labels)))
    ax[5].set_xticklabels(axis_shape_labels, rotation=25, ha="right")
    ax[5].set_title("Axis-wise Shape / Non-Gaussianity Gap")

    ax[6].hist(Dr[:, 0], bins=80, density=True, alpha=0.45, label="Real")
    ax[6].hist(Dc[:, 0], bins=80, density=True, alpha=0.45, label=model_label)
    ax[6].set_title("Noise Distribution (I)")
    ax[6].legend(loc="best")

    ax[7].hist(Dr[:, 1], bins=80, density=True, alpha=0.45, label="Real")
    ax[7].hist(Dc[:, 1], bins=80, density=True, alpha=0.45, label=model_label)
    ax[7].set_title("Noise Distribution (Q)")
    ax[7].legend(loc="best")

    ax[8].scatter(Dr[:, 0], Dr[:, 1], s=1, alpha=0.20, label="Real")
    ax[8].scatter(Dc[:, 0], Dc[:, 1], s=1, alpha=0.20, label=model_label)
    ax[8].set_title("Residual Constellation Overlay")
    ax[8].legend(loc="best", markerscale=4)

    ax[9].plot(acf_i_real, label="Real")
    ax[9].plot(acf_i_pred, label=model_label)
    ax[9].set_title("Noise Autocorrelation (I)")
    ax[9].legend(loc="best")

    ax[10].plot(acf_q_real, label="Real")
    ax[10].plot(acf_q_pred, label=model_label)
    ax[10].set_title("Noise Autocorrelation (Q)")
    ax[10].legend(loc="best")

    ax[11].plot(psd_i_real, label="Real")
    ax[11].plot(psd_i_pred, label=model_label)
    ax[11].set_title("Power Spectral Density (I)")
    ax[11].legend(loc="best")

    ax[12].plot(psd_q_real, label="Real")
    ax[12].plot(psd_q_pred, label=model_label)
    ax[12].set_title("Power Spectral Density (Q)")
    ax[12].legend(loc="best")

    ax[13].scatter(Yr[:, 0], Yr[:, 1], s=1, alpha=0.35, color="#2F7C4F")
    ax[13].set_title("Real Channel Constellation")

    ax[14].scatter(Yc[:, 0], Yc[:, 1], s=1, alpha=0.35, color="#2C57FF")
    ax[14].set_title(f"{model_label} Constellation")

    ax[15].scatter(Yr[:, 0], Yr[:, 1], s=1, alpha=0.20, label="Real")
    ax[15].scatter(Yc[:, 0], Yc[:, 1], s=1, alpha=0.20, label=model_label)
    ax[15].set_title("Constellation Overlay")
    ax[15].legend(loc="best", markerscale=4)

    ax[16].bar(np.arange(len(std_mu_p)), std_mu_p, label="std(μ_p)")
    ax[16].plot(np.arange(len(kl_dim_mean)), kl_dim_mean, color="crimson", marker="o", label="KL dim")
    ax[16].set_title("Latent Activity & KL")
    ax[16].legend(loc="best")

    ax[17].axis("off")
    ax[17].text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightsteelblue", alpha=0.9),
    )
    ax[17].set_title("Run Summary")

    for axis in ax[:17]:
        axis.grid(True, alpha=0.20)

    dashboard_path = _savefig(out_path, dpi=170)
    write_artifact_manifest(
        plots_dir,
        title="Champion-only plot bundle",
        sections={"open_first": [dashboard_path]},
    )
    return dashboard_path


def save_training_analysis_dashboard(
    *,
    df_diag: pd.DataFrame,
    plots_dir: Path,
    top_k: int = 10,
) -> Path:
    """Save the experiment-level operational dashboard for grid convergence."""
    import matplotlib.pyplot as plt

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    groups = ensure_artifact_subdirs(plots_dir, groups=("training",))
    training_dir = groups["training"]
    out_path = training_dir / "dashboard_analysis_complete.png"

    df = df_diag.copy()
    ok = df.loc[
        (df.get("status", pd.Series(index=df.index, dtype=object)).fillna("ok") != "FAILED")
        & np.isfinite(df.get("score_v2", pd.Series(index=df.index, dtype=float))),
        :
    ].copy()
    if "arch_variant" not in ok.columns:
        ok["arch_variant"] = "unknown"
    ok["arch_variant"] = ok["arch_variant"].fillna("unknown").astype(str)

    fig, axes = plt.subplots(4, 3, figsize=(22, 18))
    fig.suptitle("Dashboard Operacional de Treinamento, Evolução e Convergência", fontsize=18, y=0.995)
    ax = axes.ravel()

    if ok.empty:
        ax[0].axis("off")
        ax[0].text(
            0.02,
            0.98,
            "Nenhum grid válido disponível para diagnóstico.",
            va="top",
            ha="left",
            fontsize=12,
        )
        for axis in ax[1:]:
            axis.axis("off")
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        write_artifact_manifest(
            training_dir,
            title="Training operational dashboard",
            sections={"open_first": [out_path]},
        )
        return out_path

    top = ok.nsmallest(min(int(top_k), len(ok)), "score_v2").copy()
    arch_stats = (
        ok.groupby("arch_variant", dropna=False)
        .agg(
            score_v2=("score_v2", "median"),
            active_dim_ratio=("active_dim_ratio", "median"),
            best_epoch_ratio=("best_epoch_ratio", "median"),
            n=("score_v2", "size"),
        )
        .reset_index()
        .sort_values("score_v2", ascending=True)
    )

    ax[0].barh(
        [f"#{int(r.rank)} {r.tag}" for r in top.iloc[::-1].itertuples()],
        top["score_v2"].iloc[::-1],
        color="#2F5D80",
    )
    ax[0].set_title("Top-K modelos por score_v2")
    ax[0].set_xlabel("score_v2 (menor = melhor)")

    ax[1].scatter(
        ok["active_dim_ratio"],
        ok["score_v2"],
        c=ok["best_epoch_ratio"],
        cmap="viridis",
        edgecolors="k",
        alpha=0.85,
        s=70,
    )
    ax[1].set_title("score_v2 vs active_dim_ratio")
    ax[1].set_xlabel("active_dim_ratio")
    ax[1].set_ylabel("score_v2")

    ax[2].scatter(
        ok["best_epoch_ratio"],
        ok["score_v2"],
        c=ok["lr_drop_count"],
        cmap="plasma",
        edgecolors="k",
        alpha=0.85,
        s=70,
    )
    ax[2].set_title("score_v2 vs best_epoch_ratio")
    ax[2].set_xlabel("best_epoch_ratio")
    ax[2].set_ylabel("score_v2")

    ax[3].scatter(
        ok["kl_mean_per_dim"],
        ok["score_v2"],
        c=ok["active_dim_ratio"],
        cmap="cividis",
        edgecolors="k",
        alpha=0.85,
        s=70,
    )
    ax[3].set_title("score_v2 vs kl_mean_per_dim")
    ax[3].set_xlabel("kl_mean_per_dim")
    ax[3].set_ylabel("score_v2")

    ax[4].scatter(
        ok["lr_drop_count"],
        ok["score_v2"],
        c=ok["late_val_std"],
        cmap="magma",
        edgecolors="k",
        alpha=0.85,
        s=70,
    )
    ax[4].set_title("score_v2 vs lr_drop_count")
    ax[4].set_xlabel("lr_drop_count")
    ax[4].set_ylabel("score_v2")

    ax[5].bar(arch_stats["arch_variant"], arch_stats["score_v2"], color="#4C78A8")
    ax[5].set_title("Comparação por arch_variant: score_v2")
    ax[5].set_ylabel("mediana")
    ax[5].tick_params(axis="x", rotation=20)

    ax[6].bar(arch_stats["arch_variant"], arch_stats["active_dim_ratio"], color="#59A14F")
    ax[6].set_title("Comparação por arch_variant: active_dim_ratio")
    ax[6].set_ylabel("mediana")
    ax[6].tick_params(axis="x", rotation=20)

    ax[7].bar(arch_stats["arch_variant"], arch_stats["best_epoch_ratio"], color="#F28E2B")
    ax[7].set_title("Comparação por arch_variant: best_epoch_ratio")
    ax[7].set_ylabel("mediana")
    ax[7].tick_params(axis="x", rotation=20)

    flag_counts = pd.Series(
        {
            "collapse": int(ok["flag_posterior_collapse"].fillna(False).sum()),
            "under": int(ok["flag_undertrained"].fillna(False).sum()),
            "overfit": int(ok["flag_overfit"].fillna(False).sum()),
            "unstable": int(ok["flag_unstable"].fillna(False).sum()),
            "lr_floor": int(ok["flag_lr_floor"].fillna(False).sum()),
        }
    )
    ax[8].bar(flag_counts.index, flag_counts.values, color="#B07AA1")
    ax[8].set_title("Contagem de flags heurísticas")
    ax[8].set_ylabel("n grids")
    ax[8].tick_params(axis="x", rotation=20)

    top_lines = []
    for row in top.itertuples():
        row_s = pd.Series(row._asdict())
        beta = row_s.get("beta", np.nan)
        free_bits = row_s.get("free_bits", np.nan)
        lr = row_s.get("lr", np.nan)
        beta_txt = f"{float(beta):.4g}" if pd.notna(beta) else "nan"
        free_bits_txt = f"{float(free_bits):.3g}" if pd.notna(free_bits) else "nan"
        lr_txt = f"{float(lr):.2g}" if pd.notna(lr) else "nan"
        top_lines.append(
            f"#{int(row.rank)} {row.tag}"
            f"\n  {row.arch_variant} | lat={int(getattr(row, 'latent_dim', 0))}"
            f" beta={beta_txt}"
            f" fb={free_bits_txt}"
            f" lr={lr_txt}"
            f" L={_compact_layer_sizes(getattr(row, 'layer_sizes', ''))}"
            f"\n  score={float(row.score_v2):.4f} best_epoch={int(getattr(row, 'best_epoch', 0))}"
            f"/{int(getattr(row, 'epochs_ran', 0))} active={float(getattr(row, 'active_dim_ratio', np.nan)):.2f}"
            f"\n  flags={_row_flags_short(row_s)}"
            f"\n  rec={_row_recommendations_short(row_s)}"
        )
    ax[9].axis("off")
    ax[9].set_title("Resumo textual do Top-K")
    ax[9].text(
        0.01,
        0.99,
        "\n\n".join(top_lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.9),
    )

    rec_cols = [
        "recommend_lr",
        "recommend_beta_free_bits",
        "recommend_latent_dim",
        "recommend_capacity",
        "recommend_architecture",
        "recommend_epochs_patience",
    ]
    rec_values = pd.concat([ok[col] for col in rec_cols if col in ok.columns], axis=0)
    rec_values = rec_values[rec_values.astype(str) != "keep"]
    rec_counts = rec_values.value_counts().head(8)
    if rec_counts.empty:
        ax[10].axis("off")
        ax[10].text(0.02, 0.98, "Nenhuma recomendação ativa além de keep.", va="top", ha="left")
    else:
        ax[10].barh(rec_counts.index[::-1], rec_counts.values[::-1], color="#E15759")
        ax[10].set_title("Recomendações mais frequentes")
        ax[10].set_xlabel("n grids")

    ax[11].scatter(
        ok["late_val_slope"],
        ok["late_val_std"],
        c=ok["score_v2"],
        cmap="viridis_r",
        edgecolors="k",
        alpha=0.85,
        s=70,
    )
    ax[11].axvline(-1e-4, color="gray", linestyle="--", linewidth=1.0)
    ax[11].set_title("Convergência tardia: slope vs std")
    ax[11].set_xlabel("late_val_slope")
    ax[11].set_ylabel("late_val_std")

    for axis in ax[:9]:
        axis.grid(True, alpha=0.20)
    ax[10].grid(True, alpha=0.20)
    ax[11].grid(True, alpha=0.20)

    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    write_artifact_manifest(
        training_dir,
        title="Training operational dashboard",
        sections={"open_first": [out_path]},
    )
    return out_path


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


def save_training_analysis_dashboard(
    *,
    df_diag: pd.DataFrame,
    plots_dir: Path,
    top_k: int = 10,
) -> Path:
    """Save the experiment-level operational dashboard for grid convergence."""
    import matplotlib.pyplot as plt

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    groups = ensure_artifact_subdirs(plots_dir, groups=("training",))
    training_dir = groups["training"]
    out_path = training_dir / "dashboard_analysis_complete.png"

    df = df_diag.copy()
    ok = df.loc[
        (df.get("status", pd.Series(index=df.index, dtype=object)).fillna("ok") != "FAILED")
        & np.isfinite(df.get("score_v2", pd.Series(index=df.index, dtype=float))),
        :
    ].copy()
    if "arch_variant" not in ok.columns:
        ok["arch_variant"] = "unknown"
    ok["arch_variant"] = ok["arch_variant"].fillna("unknown").astype(str)

    fig, axes = plt.subplots(4, 3, figsize=(22, 18))
    fig.suptitle(
        "Dashboard Operacional de Treinamento, Evolução e Convergência",
        fontsize=18,
        y=0.995,
    )
    ax = axes.ravel()

    if ok.empty:
        ax[0].axis("off")
        ax[0].text(
            0.02,
            0.98,
            "Nenhum grid válido disponível para diagnóstico.",
            va="top",
            ha="left",
            fontsize=12,
        )
        for axis in ax[1:]:
            axis.axis("off")
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        write_artifact_manifest(
            training_dir,
            title="Training operational dashboard",
            sections={"open_first": [out_path]},
        )
        return out_path

    top = ok.nsmallest(min(int(top_k), len(ok)), "score_v2").copy()
    arch_stats = (
        ok.groupby("arch_variant", dropna=False)
        .agg(
            score_v2=("score_v2", "median"),
            active_dim_ratio=("active_dim_ratio", "median"),
            best_epoch_ratio=("best_epoch_ratio", "median"),
            n=("score_v2", "size"),
        )
        .reset_index()
        .sort_values("score_v2", ascending=True)
    )

    ax[0].barh(
        [f"#{int(r.rank)} {r.tag}" for r in top.iloc[::-1].itertuples()],
        top["score_v2"].iloc[::-1],
        color="#2F5D80",
    )
    ax[0].set_title("Top-K modelos por score_v2")
    ax[0].set_xlabel("score_v2 (menor = melhor)")

    ax[1].scatter(
        ok["active_dim_ratio"],
        ok["score_v2"],
        c=ok["best_epoch_ratio"],
        cmap="viridis",
        edgecolors="k",
        alpha=0.85,
        s=70,
    )
    ax[1].set_title("score_v2 vs active_dim_ratio")
    ax[1].set_xlabel("active_dim_ratio")
    ax[1].set_ylabel("score_v2")

    ax[2].scatter(
        ok["best_epoch_ratio"],
        ok["score_v2"],
        c=ok["lr_drop_count"],
        cmap="plasma",
        edgecolors="k",
        alpha=0.85,
        s=70,
    )
    ax[2].set_title("score_v2 vs best_epoch_ratio")
    ax[2].set_xlabel("best_epoch_ratio")
    ax[2].set_ylabel("score_v2")

    ax[3].scatter(
        ok["kl_mean_per_dim"],
        ok["score_v2"],
        c=ok["active_dim_ratio"],
        cmap="cividis",
        edgecolors="k",
        alpha=0.85,
        s=70,
    )
    ax[3].set_title("score_v2 vs kl_mean_per_dim")
    ax[3].set_xlabel("kl_mean_per_dim")
    ax[3].set_ylabel("score_v2")

    ax[4].scatter(
        ok["lr_drop_count"],
        ok["score_v2"],
        c=ok["late_val_std"],
        cmap="magma",
        edgecolors="k",
        alpha=0.85,
        s=70,
    )
    ax[4].set_title("score_v2 vs lr_drop_count")
    ax[4].set_xlabel("lr_drop_count")
    ax[4].set_ylabel("score_v2")

    ax[5].bar(arch_stats["arch_variant"], arch_stats["score_v2"], color="#4C78A8")
    ax[5].set_title("Comparação por arch_variant: score_v2")
    ax[5].set_ylabel("mediana")
    ax[5].tick_params(axis="x", rotation=20)

    ax[6].bar(arch_stats["arch_variant"], arch_stats["active_dim_ratio"], color="#59A14F")
    ax[6].set_title("Comparação por arch_variant: active_dim_ratio")
    ax[6].set_ylabel("mediana")
    ax[6].tick_params(axis="x", rotation=20)

    ax[7].bar(arch_stats["arch_variant"], arch_stats["best_epoch_ratio"], color="#F28E2B")
    ax[7].set_title("Comparação por arch_variant: best_epoch_ratio")
    ax[7].set_ylabel("mediana")
    ax[7].tick_params(axis="x", rotation=20)

    flag_counts = pd.Series(
        {
            "collapse": int(ok["flag_posterior_collapse"].fillna(False).sum()),
            "under": int(ok["flag_undertrained"].fillna(False).sum()),
            "overfit": int(ok["flag_overfit"].fillna(False).sum()),
            "unstable": int(ok["flag_unstable"].fillna(False).sum()),
            "lr_floor": int(ok["flag_lr_floor"].fillna(False).sum()),
        }
    )
    ax[8].bar(flag_counts.index, flag_counts.values, color="#B07AA1")
    ax[8].set_title("Contagem de flags heurísticas")
    ax[8].set_ylabel("n grids")
    ax[8].tick_params(axis="x", rotation=20)

    top_lines = []
    for row in top.itertuples():
        row_s = pd.Series(row._asdict())
        beta = row_s.get("beta", np.nan)
        free_bits = row_s.get("free_bits", np.nan)
        lr = row_s.get("lr", np.nan)
        beta_txt = f"{float(beta):.4g}" if pd.notna(beta) else "nan"
        free_bits_txt = f"{float(free_bits):.3g}" if pd.notna(free_bits) else "nan"
        lr_txt = f"{float(lr):.2g}" if pd.notna(lr) else "nan"
        top_lines.append(
            f"#{int(row.rank)} {row.tag}"
            f"\n  {row.arch_variant} | lat={int(getattr(row, 'latent_dim', 0))}"
            f" beta={beta_txt}"
            f" fb={free_bits_txt}"
            f" lr={lr_txt}"
            f" L={_compact_layer_sizes(getattr(row, 'layer_sizes', ''))}"
            f"\n  score={float(row.score_v2):.4f} best_epoch={int(getattr(row, 'best_epoch', 0))}"
            f"/{int(getattr(row, 'epochs_ran', 0))} active={float(getattr(row, 'active_dim_ratio', np.nan)):.2f}"
            f"\n  flags={_row_flags_short(row_s)}"
            f"\n  rec={_row_recommendations_short(row_s)}"
        )
    ax[9].axis("off")
    ax[9].set_title("Resumo textual do Top-K")
    ax[9].text(
        0.01,
        0.99,
        "\n\n".join(top_lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.9),
    )

    rec_cols = [
        "recommend_lr",
        "recommend_beta_free_bits",
        "recommend_latent_dim",
        "recommend_capacity",
        "recommend_architecture",
        "recommend_epochs_patience",
    ]
    rec_values = pd.concat([ok[col] for col in rec_cols if col in ok.columns], axis=0)
    rec_values = rec_values[rec_values.astype(str) != "keep"]
    rec_counts = rec_values.value_counts().head(8)
    if rec_counts.empty:
        ax[10].axis("off")
        ax[10].text(
            0.02,
            0.98,
            "Nenhuma recomendação ativa além de keep.",
            va="top",
            ha="left",
        )
    else:
        ax[10].barh(rec_counts.index[::-1], rec_counts.values[::-1], color="#E15759")
        ax[10].set_title("Recomendações mais frequentes")
        ax[10].set_xlabel("n grids")

    ax[11].scatter(
        ok["late_val_slope"],
        ok["late_val_std"],
        c=ok["score_v2"],
        cmap="viridis_r",
        edgecolors="k",
        alpha=0.85,
        s=70,
    )
    ax[11].axvline(-1e-4, color="gray", linestyle="--", linewidth=1.0)
    ax[11].set_title("Convergência tardia: slope vs std")
    ax[11].set_xlabel("late_val_slope")
    ax[11].set_ylabel("late_val_std")

    for axis in ax[:9]:
        axis.grid(True, alpha=0.20)
    ax[10].grid(True, alpha=0.20)
    ax[11].grid(True, alpha=0.20)

    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    write_artifact_manifest(
        training_dir,
        title="Training operational dashboard",
        sections={"open_first": [out_path]},
    )
    return out_path
