# -*- coding: utf-8 -*-
"""
src.training.gridsearch — Grid-search orchestration for cVAE hyperparameters.

Shared grid-search loop for the canonical training pipeline.

No scientific or loss-function changes.
"""

from __future__ import annotations

import gc
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public helpers (moved from monolith inner scope)
# ---------------------------------------------------------------------------

def _safe_tag(s: str) -> str:
    """Sanitise a grid tag for use in directory names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:160]


def _grid_artifact_dir(models_dir: Path, gi: int, tag: str) -> Path:
    """Create and return a per-grid-item artifact directory."""
    d = models_dir / f"grid_{gi:03d}__{_safe_tag(tag)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_keras_model_compat(model: Any, path: Path) -> None:
    """Save a Keras model across TF-Keras/Keras format differences.

    Some environments reject ``include_optimizer=False`` when the target uses
    the native ``.keras`` format. We prefer excluding the optimizer state when
    supported, but transparently retry without that argument when the local
    Keras version disallows it.

    For subclassed models that cannot be serialised as HDF5/``.keras``, we
    fall back to TF SavedModel format,
    saving to a directory at the same path. ``tf.keras.models.load_model``
    auto-detects whether a path is a file or directory and loads accordingly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(path), include_optimizer=False)
        return
    except ValueError as exc:
        msg = str(exc)
        if "include_optimizer" not in msg:
            raise
        print(
            "⚠️  Keras local não aceita include_optimizer no formato .keras; "
            f"repetindo save sem esse argumento: {path}"
        )
        try:
            model.save(str(path))
            return
        except NotImplementedError:
            pass  # fall through to SavedModel fallback below
    except NotImplementedError:
        pass  # fall through to SavedModel fallback below

    # Subclassed model: save as a Keras SavedModel directory, then rename the
    # directory to the canonical artifact path. Older TF/Keras builds route
    # ``model.save(..., save_format="tf")`` through the HDF5 branch whenever
    # the filepath ends with ``.keras``. Saving first to a suffix-free temp dir
    # preserves the Keras metadata, and renaming keeps the expected path.
    print(
        f"⚠️  Modelo subclasse — salvando como SavedModel (TF format) em: {path}"
    )
    import shutil
    import tensorflow as tf
    import uuid

    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.savedmodel"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tf.keras.models.save_model(model, str(tmp_path), save_format="tf")
    tmp_path.rename(path)


def _format_regime_distance(distance_m: float) -> str:
    s = f"{float(distance_m):.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"dist_{s}m"


def _format_regime_current(current_mA: float) -> str:
    return f"curr_{int(round(float(current_mA)))}mA"


def _regime_label(distance_m: float, current_mA: float) -> str:
    return f"{_format_regime_distance(distance_m)}__{_format_regime_current(current_mA)}"


def _normalize_regime_weight_map(weights_cfg: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not weights_cfg:
        return {}
    normalized: Dict[str, float] = {}
    for key, value in dict(weights_cfg).items():
        weight = float(value)
        if weight <= 0.0:
            raise ValueError(
                "train_regime_resample_weights must contain strictly positive "
                f"multipliers; got {weight!r} for {key!r}"
            )
        key_norm = str(key).strip().lower().replace(" ", "")
        key_norm = key_norm.replace("/", "__curr_") if "/" in key_norm else key_norm
        if "__curr_" not in key_norm or not key_norm.startswith("dist_"):
            raise ValueError(
                "train_regime_resample_weights keys must use the canonical "
                "format 'dist_0p8m__curr_100mA'"
            )
        normalized[key_norm] = weight
    return normalized


def _build_regime_weight_vector(
    d_raw: np.ndarray,
    c_raw: np.ndarray,
    weights_cfg: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    weights_map = _normalize_regime_weight_map(weights_cfg)
    d_arr = np.asarray(d_raw, dtype=np.float32).reshape(-1)
    c_arr = np.asarray(c_raw, dtype=np.float32).reshape(-1)
    if len(d_arr) != len(c_arr):
        raise ValueError(
            f"regime weighting alignment mismatch: len(D)={len(d_arr)} "
            f"vs len(C)={len(c_arr)}"
        )
    labels = np.asarray(
        [_regime_label(float(d), float(c)) for d, c in zip(d_arr, c_arr)],
        dtype=object,
    )
    labels_norm = np.asarray([str(x).strip().lower() for x in labels], dtype=object)
    weights = np.ones(len(labels), dtype=np.float32)
    if not weights_map:
        return weights, labels
    for regime_key, weight in weights_map.items():
        weights[labels_norm == regime_key] = float(weight)
    return weights, labels


def _summarize_regime_counts(labels: np.ndarray) -> str:
    counts = Counter(str(x) for x in labels.tolist())
    parts = [f"{k}={counts[k]}" for k in sorted(counts)]
    return ", ".join(parts)


def _apply_regime_weighted_resampling(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    d_train_norm: np.ndarray,
    c_train_norm: np.ndarray,
    d_train_raw: Optional[np.ndarray],
    c_train_raw: Optional[np.ndarray],
    cfg: Dict[str, Any],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    weights_cfg = cfg.get("train_regime_resample_weights")
    if not weights_cfg:
        return (
            x_train,
            y_train,
            d_train_norm,
            c_train_norm,
            d_train_raw,
            c_train_raw,
            {"enabled": False},
        )
    if d_train_raw is None or c_train_raw is None:
        raise ValueError(
            "train_regime_resample_weights requires raw D/C arrays so regime "
            "labels remain in physical units."
        )

    weights, labels = _build_regime_weight_vector(d_train_raw, c_train_raw, weights_cfg)
    target_size = int(cfg.get("train_regime_resample_target_size", len(x_train)))
    if target_size <= 0:
        raise ValueError(
            "train_regime_resample_target_size must be positive when regime "
            "resampling is enabled."
        )
    if np.allclose(weights, 1.0):
        return (
            x_train,
            y_train,
            d_train_norm,
            c_train_norm,
            d_train_raw,
            c_train_raw,
            {
                "enabled": True,
                "changed": False,
                "weights_map": _normalize_regime_weight_map(weights_cfg),
                "before_counts": _summarize_regime_counts(labels),
                "after_counts": _summarize_regime_counts(labels),
                "target_size": int(target_size),
            },
        )

    rng = np.random.default_rng(int(seed))
    probs = weights.astype(np.float64)
    probs /= probs.sum()
    idx = rng.choice(len(x_train), size=target_size, replace=True, p=probs)
    labels_after = labels[idx]
    return (
        x_train[idx],
        y_train[idx],
        d_train_norm[idx],
        c_train_norm[idx],
        np.asarray(d_train_raw)[idx],
        np.asarray(c_train_raw)[idx],
        {
            "enabled": True,
            "changed": True,
            "weights_map": _normalize_regime_weight_map(weights_cfg),
            "before_counts": _summarize_regime_counts(labels),
            "after_counts": _summarize_regime_counts(labels_after),
            "target_size": int(target_size),
        },
    )


def checklist_table() -> "pd.DataFrame":
    """Return the collapse-vs-healthy checklist as a DataFrame."""
    rows = [
        ("KL no treino", "kl_loss médio > ~1 e não tende a 0",
         "kl_loss ~ 0 por muitas épocas", "colapso → decoder ignora z"),
        ("Atividade por dim", "std(z_mean_p[:,k]) > 0.05 em várias dims",
         "todas std ~0", "z não carrega info"),
        ("Sensibilidade decoder a z", "var(Y|z) muda ao perturbar z",
         "Y quase não muda ao perturbar z", "decoder ignora z"),
        ("Prior varia com (d,c)", "||μ_p||/dims muda com d ou c",
         "invariante em d,c", "condicional não aprendido"),
        ("Recon vs val", "recon e val_recon descem e estabilizam",
         "val piora muito e diverge", "overfit/instabilidade"),
        ("EVM/SNR (val)", "EVM_pred próximo do real em validação",
         "EVM_pred igual AWGN ou muito ruim", "twin fraco"),
    ]
    return pd.DataFrame(rows, columns=[
        "Item", "OK (funcionando)", "Alerta (colapsando)", "Interpretação",
    ])


# ---------------------------------------------------------------------------
# Grid report helper (moved from monolith _save_experiment_report_png)
# ---------------------------------------------------------------------------

def save_experiment_report_png(
    plot_path: Path,
    Xv: np.ndarray,
    Yv: np.ndarray,
    Yp: np.ndarray,
    std_mu_p: np.ndarray,
    kl_dim_mean: np.ndarray,
    summary_lines: List[str],
    title: str,
) -> None:
    """Save a 6-panel grid-experiment report PNG (unchanged scientific content)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(title, fontsize=18, y=0.98)

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(Yv[:, 0], Yv[:, 1], s=2)
    ax1.set_title("Real")
    ax1.set_xlabel("I"); ax1.set_ylabel("Q")
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(Yp[:, 0], Yp[:, 1], s=2)
    ax2.set_title("cVAE")
    ax2.set_xlabel("I"); ax2.set_ylabel("Q")
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(Yv[:, 0], Yv[:, 1], s=2, label="Real")
    ax3.scatter(Yp[:, 0], Yp[:, 1], s=2, label="cVAE")
    ax3.set_title("Overlay")
    ax3.set_xlabel("I"); ax3.set_ylabel("Q")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best")

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.bar(np.arange(len(std_mu_p)), std_mu_p)
    ax4.set_title("Atividade latente (std μ_p)")
    ax4.set_xlabel("dim"); ax4.set_ylabel("std")
    ax4.grid(True, alpha=0.25)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(np.arange(len(kl_dim_mean)), kl_dim_mean)
    ax5.set_title("KL(q||p) por dimensão (média)")
    ax5.set_xlabel("dim"); ax5.set_ylabel("KL_dim")
    ax5.grid(True, alpha=0.25)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    txt = "\n".join(summary_lines)
    ax6.text(0.02, 0.98, txt, va="top", ha="left",
             bbox=dict(boxstyle="round", facecolor="lightsteelblue", alpha=0.85))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_experiment_xlsx(xlsx_path: Path, row_dict: dict) -> None:
    """Save per-grid-item diagnostic Excel workbook."""
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([row_dict])
    cfg_keys = [
        "activation", "kl_anneal_epochs", "batch_size", "lr",
        "dropout", "free_bits", "layer_sizes", "latent_dim", "beta",
        "arch_variant", "lambda_mmd", "mmd_mode", "seq_gru_unroll",
    ]
    cfg_json = pd.DataFrame([{
        "cfg_json": json.dumps(
            {k: row_dict[k] for k in row_dict if k in cfg_keys},
            ensure_ascii=False,
        )
    }])
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        summary.to_excel(w, index=False, sheet_name="summary")
        cfg_json.to_excel(w, index=False, sheet_name="cfg_json")


# ---------------------------------------------------------------------------
# Scoring — verbatim from monolith (no scientific changes)
# ---------------------------------------------------------------------------

def compute_score_v2(
    *,
    evm_real: float,
    evm_pred: float,
    snr_real: float,
    snr_pred: float,
    mean_l2: float,
    cov_fro: float,
    active_dims: int,
    latent_dim: int,
    kl_mean_per_dim: float,
    var_mc: float,
    var_real: float,
    psd_l2: float,
    skew_l2: float,
    kurt_l2: float,
    w_psd: float = 0.15,
    w_skew: float = 0.05,
    w_kurt: float = 0.05,
) -> float:
    """Compute the combined ranking score (lower is better)."""
    pen_inactive = max(0, int(latent_dim // 2) - active_dims)
    pen_kl_low = max(0.0, 0.2 - kl_mean_per_dim)
    pen_kl_high = 0.001 * max(0.0, kl_mean_per_dim - 50.0)

    pen_var_mismatch = 0.0
    if not np.isnan(var_mc):
        pen_var_mismatch = float(abs(var_mc - var_real))

    return float(
        abs(evm_pred - evm_real)
        + abs(snr_pred - snr_real)
        + 0.4 * mean_l2
        + 0.2 * cov_fro
        + 2.0 * pen_inactive
        + 1.0 * pen_kl_low
        + 1.0 * pen_kl_high
        + 0.5 * pen_var_mismatch
        + w_psd * psd_l2
        + w_skew * skew_l2
        + w_kurt * kurt_l2
    )


def _ranking_scalar(value: Any, default: float = float("inf")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _candidate_ranking_key(row: Dict[str, Any], ranking_mode: str) -> Tuple[float, ...]:
    status_penalty = 0.0 if str(row.get("status", "ok")) == "ok" else 1.0
    mode = str(ranking_mode or "score_v2").strip().lower()
    if mode == "mini_protocol_v1":
        return (
            status_penalty,
            _ranking_scalar(row.get("mini_n_fail")),
            _ranking_scalar(row.get("mini_n_g6_fail")),
            _ranking_scalar(row.get("mini_mean_abs_delta_coverage_95")),
            _ranking_scalar(row.get("mini_mean_delta_jb")),
            _ranking_scalar(row.get("mini_mean_delta_psd_l2")),
            _ranking_scalar(row.get("score_v2")),
            _ranking_scalar(row.get("score_abs_delta")),
        )
    return (
        status_penalty,
        _ranking_scalar(row.get("score_v2")),
        _ranking_scalar(row.get("score_abs_delta")),
    )


def _sort_results_by_ranking(
    df_results: pd.DataFrame,
    ranking_mode: str,
) -> pd.DataFrame:
    df = df_results.copy()
    keys = df.apply(
        lambda row: _candidate_ranking_key(row.to_dict(), ranking_mode),
        axis=1,
    )
    width = max((len(key) for key in keys.tolist()), default=0)
    for i in range(width):
        df[f"_rank_{i}"] = keys.apply(lambda key, idx=i: key[idx])
    df = df.sort_values([f"_rank_{i}" for i in range(width)], ascending=[True] * width)
    return df.drop(columns=[f"_rank_{i}" for i in range(width)])


TRAINING_DIAGNOSTIC_COLUMNS: List[str] = [
    "rank",
    "grid_id",
    "group",
    "tag",
    "status",
    "arch_variant",
    "latent_dim",
    "beta",
    "free_bits",
    "lr",
    "batch_size",
    "kl_anneal_epochs",
    "layer_sizes",
    "window_size",
    "seq_hidden_size",
    "seq_gru_unroll",
    "lambda_mmd",
    "mmd_mode",
    "epochs_ran",
    "best_epoch",
    "best_epoch_ratio",
    "epochs_since_best",
    "best_val_recon_loss",
    "last_val_recon_loss",
    "val_recon_improvement",
    "worse_from_best",
    "train_recon_at_best",
    "train_recon_last",
    "final_lr",
    "lr_drop_count",
    "late_val_slope",
    "late_val_std",
    "active_dims",
    "active_dim_ratio",
    "kl_mean_total",
    "kl_mean_per_dim",
    "score_v2",
    "delta_evm_%",
    "delta_snr_db",
    "delta_mean_l2",
    "delta_cov_fro",
    "delta_psd_l2",
    "delta_acf_l2",
    "delta_skew_l2",
    "delta_kurt_l2",
    "var_real_delta",
    "var_pred_delta",
    "pen_var_mismatch",
    "flag_posterior_collapse",
    "flag_undertrained",
    "flag_overfit",
    "flag_unstable",
    "flag_lr_floor",
    "recommend_lr",
    "recommend_beta_free_bits",
    "recommend_latent_dim",
    "recommend_capacity",
    "recommend_architecture",
    "recommend_epochs_patience",
]


def _mc_point_metric_means(Xv_center: np.ndarray, Ys: np.ndarray) -> tuple[float, float]:
    """Average EVM/SNR across individual MC draws instead of the ensemble mean."""
    from src.evaluation.metrics import calculate_evm, calculate_snr

    Ys = np.asarray(Ys)
    if Ys.ndim != 3 or Ys.shape[0] == 0:
        raise ValueError("Ys must have shape (K, N, 2) with K >= 1")

    evm_vals = [calculate_evm(Xv_center, Ys[i])[0] for i in range(Ys.shape[0])]
    snr_vals = [calculate_snr(Xv_center, Ys[i]) for i in range(Ys.shape[0])]
    return float(np.mean(evm_vals)), float(np.mean(snr_vals))
def _history_series(
    history_dict: Dict[str, Sequence[float]],
    *keys: str,
) -> List[float]:
    """Return the first non-empty metric series found in ``history_dict``."""
    for key in keys:
        values = history_dict.get(key, [])
        if values:
            return [float(v) for v in values]
    return []


def _late_series_slope(values: Sequence[float], tail: int = 5) -> float:
    """Linear slope over the tail of a validation curve."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    arr = arr[-min(int(tail), arr.size):]
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=np.float64)
    return float(np.polyfit(x, arr, 1)[0])


def _late_series_std(values: Sequence[float], tail: int = 5) -> float:
    """Standard deviation over the tail of a validation curve."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    arr = arr[-min(int(tail), arr.size):]
    return float(np.std(arr))


def _count_lr_drops(values: Sequence[float]) -> int:
    """Count strict learning-rate drops across epochs."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0
    drops = 0
    prev = float(arr[0])
    for current in arr[1:]:
        curr = float(current)
        tol = max(1e-12, 1e-6 * max(abs(prev), 1.0))
        if curr < prev - tol:
            drops += 1
        prev = curr
    return int(drops)


def _build_training_diagnostics(
    *,
    history_dict: Dict[str, Sequence[float]],
    cfg: Dict[str, Any],
    max_epochs: int,
    active_dims: int,
    kl_mean_total: float,
    kl_mean_per_dim: float,
) -> Dict[str, Any]:
    """Extract compact convergence and latent diagnostics for one grid row."""
    val_hist = _history_series(history_dict, "val_recon_loss", "val_loss")
    train_hist = _history_series(history_dict, "recon_loss", "loss")
    lr_hist = _history_series(history_dict, "lr", "learning_rate")

    epochs_ran = max(
        len(val_hist),
        len(train_hist),
        len(lr_hist),
        max((len(v) for v in history_dict.values()), default=0),
    )

    if val_hist:
        best_idx = int(np.argmin(np.asarray(val_hist, dtype=np.float64)))
        best_epoch = best_idx + 1
        best_val = float(val_hist[best_idx])
        last_val = float(val_hist[-1])
        first_val = float(val_hist[0])
    else:
        best_idx = -1
        best_epoch = 0
        best_val = float("nan")
        last_val = float("nan")
        first_val = float("nan")

    train_recon_at_best = (
        float(train_hist[best_idx])
        if train_hist and best_idx >= 0 and best_idx < len(train_hist)
        else float("nan")
    )
    train_recon_last = float(train_hist[-1]) if train_hist else float("nan")
    final_lr = float(lr_hist[-1]) if lr_hist else float(cfg.get("lr", float("nan")))
    lr_drop_count = _count_lr_drops(lr_hist)
    late_val_slope = _late_series_slope(val_hist)
    late_val_std = _late_series_std(val_hist)

    latent_dim = max(int(cfg.get("latent_dim", 0)), 1)
    active_dim_ratio = float(active_dims) / float(latent_dim)
    best_epoch_ratio = (
        float(best_epoch) / float(max(int(max_epochs), 1))
        if best_epoch > 0
        else 0.0
    )
    epochs_since_best = max(int(epochs_ran) - int(best_epoch), 0)
    val_recon_improvement = (
        float(first_val - best_val)
        if np.isfinite(first_val) and np.isfinite(best_val)
        else float("nan")
    )
    worse_from_best = (
        float(last_val - best_val)
        if np.isfinite(last_val) and np.isfinite(best_val)
        else float("nan")
    )

    flag_posterior_collapse = bool(
        active_dim_ratio < 0.5 or float(kl_mean_per_dim) < 0.2
    )
    flag_undertrained = bool(
        best_epoch_ratio >= 0.85 and float(late_val_slope) < -1e-4
    )
    flag_overfit = bool(
        np.isfinite(worse_from_best)
        and np.isfinite(best_val)
        and np.isfinite(train_recon_last)
        and np.isfinite(train_recon_at_best)
        and worse_from_best > max(0.01 * abs(best_val), 1e-3)
        and train_recon_last <= train_recon_at_best
    )
    flag_unstable = bool(
        (
            np.isfinite(late_val_std)
            and np.isfinite(best_val)
            and late_val_std > max(0.02 * abs(best_val), 1e-3)
        )
        or lr_drop_count >= 4
    )
    flag_lr_floor = bool(np.isfinite(final_lr) and final_lr <= 1.1e-6)

    return {
        "epochs_ran": int(epochs_ran),
        "best_epoch": int(best_epoch),
        "best_epoch_ratio": float(best_epoch_ratio),
        "epochs_since_best": int(epochs_since_best),
        "best_val_recon_loss": float(best_val),
        "last_val_recon_loss": float(last_val),
        "val_recon_improvement": float(val_recon_improvement),
        "worse_from_best": float(worse_from_best),
        "train_recon_at_best": float(train_recon_at_best),
        "train_recon_last": float(train_recon_last),
        "final_lr": float(final_lr),
        "lr_drop_count": int(lr_drop_count),
        "late_val_slope": float(late_val_slope),
        "late_val_std": float(late_val_std),
        "active_dims": int(active_dims),
        "active_dim_ratio": float(active_dim_ratio),
        "kl_mean_total": float(kl_mean_total),
        "kl_mean_per_dim": float(kl_mean_per_dim),
        "flag_posterior_collapse": flag_posterior_collapse,
        "flag_undertrained": flag_undertrained,
        "flag_overfit": flag_overfit,
        "flag_unstable": flag_unstable,
        "flag_lr_floor": flag_lr_floor,
    }


def _apply_training_recommendations(df_results: pd.DataFrame) -> pd.DataFrame:
    """Attach heuristic tuning recommendations to the per-grid summary."""
    from src.evaluation.validation_summary import TWIN_GATE_THRESHOLDS

    df = df_results.copy()
    if df.empty:
        for col in (
            "recommend_lr",
            "recommend_beta_free_bits",
            "recommend_latent_dim",
            "recommend_capacity",
            "recommend_architecture",
            "recommend_epochs_patience",
        ):
            df[col] = pd.Series(dtype=object)
        return df

    arch_norm = (
        df.get("arch_variant", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    is_seq = arch_norm.str.contains("seq")

    seq_family_preferred = False
    seq_ok = df.loc[is_seq & np.isfinite(df["score_v2"]), :]
    pt_ok = df.loc[~is_seq & np.isfinite(df["score_v2"]), :]
    if not seq_ok.empty and not pt_ok.empty:
        best_seq = seq_ok.sort_values(["score_v2", "delta_psd_l2"], ascending=[True, True]).iloc[0]
        best_pt = pt_ok.sort_values(["score_v2", "delta_psd_l2"], ascending=[True, True]).iloc[0]
        seq_family_preferred = bool(
            float(best_seq.get("delta_psd_l2", np.inf))
            <= 0.85 * float(best_pt.get("delta_psd_l2", np.inf))
            or float(best_seq.get("delta_acf_l2", np.inf))
            <= 0.85 * float(best_pt.get("delta_acf_l2", np.inf))
        )

    def _row_recommendations(row: pd.Series) -> pd.Series:
        recommend_lr = "keep"
        recommend_beta_free_bits = "keep"
        recommend_latent_dim = "keep"
        recommend_capacity = "keep"
        recommend_architecture = "keep"
        recommend_epochs_patience = "keep"

        if bool(row.get("flag_undertrained", False)):
            recommend_epochs_patience = "increase_epochs_or_patience"

        if bool(row.get("flag_lr_floor", False)) and float(row.get("late_val_slope", 0.0)) < -1e-4:
            recommend_lr = "lower_initial_lr"

        if bool(row.get("flag_posterior_collapse", False)):
            recommend_beta_free_bits = "increase_free_bits_or_reduce_beta"
            if float(row.get("active_dim_ratio", 1.0)) < 0.25:
                recommend_latent_dim = "reduce_latent_dim"

        stable_training = not any(
            bool(row.get(flag, False))
            for flag in ("flag_undertrained", "flag_overfit", "flag_unstable")
        )
        high_structural_error = (
            float(row.get("delta_mean_l2", 0.0)) > 0.05
            or float(row.get("delta_cov_fro", 0.0)) > TWIN_GATE_THRESHOLDS["cov_rel_var"]
            or float(row.get("delta_psd_l2", 0.0)) > TWIN_GATE_THRESHOLDS["delta_psd_l2"]
        )
        if stable_training and high_structural_error:
            recommend_capacity = "increase_capacity_or_use_seq"

        proxy_g1_to_g4_good = (
            abs(float(row.get("delta_evm_%", 0.0))) <= 1.0
            and abs(float(row.get("delta_snr_db", 0.0))) <= 1.0
            and float(row.get("delta_mean_l2", np.inf)) <= 0.05
            and float(row.get("delta_cov_fro", np.inf)) <= TWIN_GATE_THRESHOLDS["cov_rel_var"]
            and float(row.get("delta_psd_l2", np.inf)) <= TWIN_GATE_THRESHOLDS["delta_psd_l2"]
        )
        proxy_g5_bad = (
            float(row.get("delta_skew_l2", 0.0)) > TWIN_GATE_THRESHOLDS["delta_skew_l2"]
            or float(row.get("delta_kurt_l2", 0.0)) > TWIN_GATE_THRESHOLDS["delta_kurt_l2"]
        )
        if proxy_g1_to_g4_good and proxy_g5_bad:
            recommend_architecture = "increase_distributional_regularization"

        if seq_family_preferred and "seq" not in str(row.get("arch_variant", "")).lower():
            recommend_architecture = "prefer_seq_family"

        return pd.Series(
            {
                "recommend_lr": recommend_lr,
                "recommend_beta_free_bits": recommend_beta_free_bits,
                "recommend_latent_dim": recommend_latent_dim,
                "recommend_capacity": recommend_capacity,
                "recommend_architecture": recommend_architecture,
                "recommend_epochs_patience": recommend_epochs_patience,
            }
        )

    recs = df.apply(_row_recommendations, axis=1)
    for col in recs.columns:
        df[col] = recs[col]
    return df


def _build_training_diagnostics_table(df_results: pd.DataFrame) -> pd.DataFrame:
    """Return the compact canonical training-diagnostics table."""
    df = df_results.copy()
    for col in TRAINING_DIAGNOSTIC_COLUMNS:
        if col not in df.columns:
            if col.startswith("flag_"):
                df[col] = False
            elif col.startswith("recommend_"):
                df[col] = "keep"
            else:
                df[col] = np.nan
    return df.loc[:, TRAINING_DIAGNOSTIC_COLUMNS]


def _stratified_val_indices_by_experiment(
    *,
    n_total: int,
    n_val_total: int,
    df_split: Optional["pd.DataFrame"],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample validation indices uniformly across validation experiments.

    If ``df_split`` is unavailable or inconsistent, falls back to global
    uniform sampling without replacement.
    """
    n_total = int(max(1, min(n_total, n_val_total)))
    idx_eval: Optional[np.ndarray] = None

    if (
        isinstance(df_split, pd.DataFrame)
        and "n_val" in df_split.columns
    ):
        n_val_list = [int(v) for v in df_split["n_val"].tolist()]
        if n_val_list and int(np.sum(n_val_list)) == int(n_val_total):
            n_exps = len(n_val_list)
            base = n_total // n_exps
            rem = n_total % n_exps

            target = np.full(n_exps, base, dtype=np.int64)
            if rem > 0:
                rem_idx = rng.permutation(n_exps)[:rem]
                target[rem_idx] += 1

            cap = np.asarray(n_val_list, dtype=np.int64)
            take = np.minimum(target, cap)
            avail = cap - take
            left = int(n_total - int(take.sum()))

            while left > 0 and int(avail.sum()) > 0:
                progressed = False
                for i in rng.permutation(n_exps):
                    if left <= 0:
                        break
                    if avail[i] > 0:
                        take[i] += 1
                        avail[i] -= 1
                        left -= 1
                        progressed = True
                if not progressed:
                    break

            idx_parts = []
            cursor = 0
            for i, n_i in enumerate(n_val_list):
                k_i = int(take[i])
                if k_i > 0:
                    if k_i < n_i:
                        local = np.sort(rng.choice(n_i, size=k_i, replace=False))
                    else:
                        local = np.arange(n_i, dtype=np.int64)
                    idx_parts.append(cursor + local)
                cursor += n_i

            if idx_parts:
                idx_eval = np.concatenate(idx_parts, axis=0)
                idx_eval.sort()

    if idx_eval is None:
        idx_eval = rng.choice(n_val_total, size=n_total, replace=False)
        idx_eval.sort()

    return idx_eval


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_gridsearch(
    *,
    grid: List[Dict[str, Any]],
    training_config: Dict[str, Any],
    analysis_quick: Dict[str, Any],
    X_train: np.ndarray,
    Y_train: np.ndarray,
    Dn_train: np.ndarray,
    Cn_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    Dn_val: np.ndarray,
    Cn_val: np.ndarray,
    D_train_raw: Optional[np.ndarray] = None,
    C_train_raw: Optional[np.ndarray] = None,
    D_val_raw: Optional[np.ndarray] = None,
    C_val_raw: Optional[np.ndarray] = None,
    run_paths: "RunPaths",
    overrides: Optional[Dict[str, Any]] = None,
    df_plan: Optional["pd.DataFrame"] = None,
    df_split: Optional["pd.DataFrame"] = None,
) -> "pd.DataFrame":
    """Execute the full grid-search loop and return sorted results.

    The scientific logic (model build, fit, score) is the canonical
    implementation used by the training engine.

    Parameters
    ----------
    grid : list of dict
        Each element has ``"group"``, ``"tag"``, ``"cfg"`` (model config dict).
    training_config : dict
        ``TRAINING_CONFIG`` from the monolith.
    analysis_quick : dict
        ``ANALYSIS_QUICK`` from the monolith.
    X_train, Y_train, Dn_train, Cn_train : ndarray
        Training arrays (IQ + normalised conditions).
    X_val, Y_val, Dn_val, Cn_val : ndarray
        Validation arrays.
    run_paths : RunPaths
        Canonical run directory bundle.
    overrides : dict, optional
        Runtime overrides (e.g. ``keras_verbose``).
    df_plan : DataFrame, optional
        Pre-built grid plan table (for inclusion in final Excel).

    Returns
    -------
    pd.DataFrame
        Sorted results table with a ``rank`` column.
    """
    if not grid:
        grid_tag = None if overrides is None else overrides.get("grid_tag")
        raise ValueError(
            "Grid search recebeu 0 configurações após o filtro. "
            "Verifique a combinação de grid_preset/grid_tag; "
            f"grid_tag atual={grid_tag!r}."
        )

    import tensorflow as tf
    from src.models.cvae import build_cvae, create_inference_model_from_full
    from src.models.callbacks import (
        MiniProtocolReanalysisCallback,
        RegimeDiagnosticsCallback,
        build_callbacks,
    )
    from src.evaluation.metrics import (
        calculate_evm, calculate_snr, residual_distribution_metrics,
    )
    from src.training.grid_plots import (
        save_champion_analysis_dashboard,
        save_training_analysis_dashboard,
    )

    _ov = overrides or {}
    MODELS_DIR = run_paths.models_dir

    results: List[Dict[str, Any]] = []
    best_score: Optional[Tuple[float, ...]] = None

    for gi, item in enumerate(grid, start=1):
        cfg = item["cfg"]
        group = item["group"]
        tag = item["tag"]
        candidate_analysis_quick = dict(analysis_quick)
        candidate_analysis_quick.update(dict(item.get("analysis_quick_overrides", {})))

        print("\n" + "=" * 92)
        print(f"🚀 GRID {gi}/{len(grid)} | group={group} | tag={tag}")
        print(f"    cfg = {cfg}")
        print("=" * 92)

        tf.keras.backend.clear_session()
        gc.collect()
        model_dir = _grid_artifact_dir(MODELS_DIR, gi, tag)

        # --- Phase 5: per-item windowing for seq_bigru_residual ---
        _arch = str(cfg.get("arch_variant", "concat")).strip().lower()
        if _arch == "seq_bigru_residual":
            from src.data.windowing import build_windows_from_split_arrays
            if df_split is None:
                raise RuntimeError(
                    "seq_bigru_residual requires df_split; "
                    "ensure run_training_pipeline passes df_split to run_gridsearch."
                )
            _ws  = int(cfg.get("window_size", 33))
            _wst = int(cfg.get("window_stride", 1))
            _wpm = str(cfg.get("window_pad_mode", "edge"))
            (X_tr_fit, Y_tr_fit, D_tr_fit, C_tr_fit,
             X_va_fit, Y_va_fit, D_va_fit, C_va_fit) = build_windows_from_split_arrays(
                X_train, Y_train, Dn_train, Cn_train,
                X_val,   Y_val,   Dn_val,   Cn_val,
                df_split=df_split,
                window_size=_ws, stride=_wst, pad_mode=_wpm,
            )
            if (
                D_train_raw is not None and C_train_raw is not None
                and D_val_raw is not None and C_val_raw is not None
            ):
                (_, _, D_tr_raw_fit, C_tr_raw_fit,
                 _, _, D_va_raw_fit, C_va_raw_fit) = build_windows_from_split_arrays(
                    X_train, Y_train, D_train_raw, C_train_raw,
                    X_val,   Y_val,   D_val_raw,   C_val_raw,
                    df_split=df_split,
                    window_size=_ws, stride=_wst, pad_mode=_wpm,
                )
            else:
                D_tr_raw_fit = D_train_raw
                C_tr_raw_fit = C_train_raw
                D_va_raw_fit = D_val_raw
                C_va_raw_fit = C_val_raw
            print(
                f"  ↳ seq windowing: window_size={_ws}, stride={_wst} | "
                f"X_tr={X_tr_fit.shape}, X_va={X_va_fit.shape}"
            )
            X_va_center_fit = X_val.copy()
        else:
            X_tr_fit, Y_tr_fit, D_tr_fit, C_tr_fit = X_train, Y_train, Dn_train, Cn_train
            X_va_fit, Y_va_fit, D_va_fit, C_va_fit = X_val,   Y_val,   Dn_val,   Cn_val
            D_tr_raw_fit = None if D_train_raw is None else np.asarray(D_train_raw).reshape(-1, 1)
            C_tr_raw_fit = None if C_train_raw is None else np.asarray(C_train_raw).reshape(-1, 1)
            D_va_raw_fit = None if D_val_raw is None else np.asarray(D_val_raw).reshape(-1, 1)
            C_va_raw_fit = None if C_val_raw is None else np.asarray(C_val_raw).reshape(-1, 1)
            X_va_center_fit = X_va_fit

        (
            X_tr_fit,
            Y_tr_fit,
            D_tr_fit,
            C_tr_fit,
            D_tr_raw_fit,
            C_tr_raw_fit,
            resample_info,
        ) = _apply_regime_weighted_resampling(
            x_train=X_tr_fit,
            y_train=Y_tr_fit,
            d_train_norm=D_tr_fit,
            c_train_norm=C_tr_fit,
            d_train_raw=D_tr_raw_fit,
            c_train_raw=C_tr_raw_fit,
            cfg=cfg,
            seed=int(training_config.get("seed", 42)),
        )
        if resample_info.get("enabled"):
            print(
                "  ↳ regime resampling: "
                f"target_size={resample_info['target_size']:,} | "
                f"changed={bool(resample_info.get('changed', False))}"
            )
            print(f"     weights={resample_info.get('weights_map', {})}")
            print(f"     before={resample_info.get('before_counts', '')}")
            print(f"     after ={resample_info.get('after_counts', '')}")

        try:
            _shuffle_train_batches = bool(
                cfg.get("shuffle_train_batches", training_config["shuffle_train_batches"])
            )
            if float(cfg.get("lambda_psd", 0.0)) > 0.0 and _shuffle_train_batches:
                raise ValueError(
                    "lambda_psd requires shuffle_train_batches=False so the batch "
                    "still preserves temporal order for the PSD term."
                )
            vae, kl_cb = build_cvae(cfg)
            regime_diag_callback = None
            if (
                bool(candidate_analysis_quick.get("train_regime_diagnostics_enabled", True))
                and D_va_raw_fit is not None
                and C_va_raw_fit is not None
            ):
                regime_diag_callback = RegimeDiagnosticsCallback(
                    logs_dir=run_paths.logs_dir,
                    x_val_input=X_va_fit,
                    x_val_center=X_va_center_fit,
                    y_val=Y_va_fit,
                    d_val_norm=D_va_fit,
                    c_val_norm=C_va_fit,
                    d_val_raw=D_va_raw_fit,
                    c_val_raw=C_va_raw_fit,
                    enabled=True,
                    every_n_epochs=int(candidate_analysis_quick.get("train_regime_diagnostics_every", 10)),
                    mc_samples=int(candidate_analysis_quick.get("train_regime_diagnostics_mc_samples", 4)),
                    max_samples_per_regime=int(
                        candidate_analysis_quick.get("train_regime_diagnostics_max_samples_per_regime", 4096)
                    ),
                    amplitude_bins=int(
                        candidate_analysis_quick.get("train_regime_diagnostics_amplitude_bins", 4)
                    ),
                    focus_only_0p8m=bool(
                        candidate_analysis_quick.get("train_regime_diagnostics_focus_only_0p8m", False)
                    ),
                    stat_seed=int(training_config.get("seed", 42)),
                )
            mini_reanalysis_callback = None
            if (
                bool(candidate_analysis_quick.get("mini_reanalysis_enabled", False))
                and D_va_raw_fit is not None
                and C_va_raw_fit is not None
            ):
                mini_reanalysis_callback = MiniProtocolReanalysisCallback(
                    artifact_dir=model_dir,
                    x_val_input=X_va_fit,
                    x_val_center=X_va_center_fit,
                    y_val=Y_va_fit,
                    d_val_norm=D_va_fit,
                    c_val_norm=C_va_fit,
                    d_val_raw=D_va_raw_fit,
                    c_val_raw=C_va_raw_fit,
                    enabled=True,
                    scope=str(candidate_analysis_quick.get("mini_reanalysis_scope", "all12")),
                    mc_samples=int(candidate_analysis_quick.get("mc_samples", 8)),
                    max_samples_per_regime=int(
                        candidate_analysis_quick.get("mini_reanalysis_max_samples_per_regime", 4096)
                    ),
                    stat_seed=int(training_config.get("seed", 42)),
                )
            callbacks = build_callbacks(
                training_config,
                cfg,
                kl_cb,
                regime_diag_callback=regime_diag_callback,
                mini_reanalysis_callback=mini_reanalysis_callback,
            )

            t0 = time.time()
            _keras_verbose = int(_ov.get("keras_verbose", 2))
            _bs_cfg = int(cfg["batch_size"])
            # Small-smoke stability: avoid single-step epochs when train << batch_size.
            if len(X_tr_fit) < _bs_cfg:
                _bs_eff = max(128, len(X_tr_fit) // 64)
            else:
                _bs_eff = _bs_cfg
            hist = vae.fit(
                [X_tr_fit, D_tr_fit, C_tr_fit, Y_tr_fit], Y_tr_fit,
                validation_data=([X_va_fit, D_va_fit, C_va_fit, Y_va_fit], Y_va_fit),
                epochs=int(training_config["epochs"]),
                batch_size=int(_bs_eff),
                callbacks=callbacks,
                verbose=_keras_verbose,
                shuffle=_shuffle_train_batches,
            )
            train_time_s = float(time.time() - t0)

            # Best epoch
            val_mon = "val_recon_loss" if "val_recon_loss" in hist.history else "val_loss"
            val_hist = hist.history.get(val_mon, [])
            if len(val_hist) > 0:
                best_epoch = int(np.argmin(val_hist) + 1)
                best_val = float(np.min(val_hist))
            else:
                best_epoch = 0
                best_val = float("nan")

            # Eval estratificado por experimento de validação.
            _n_total = int(candidate_analysis_quick["n_eval_samples"])
            _n_total = max(1, min(_n_total, len(X_va_fit)))
            _rng_eval = np.random.default_rng(int(training_config.get("seed", 42)))
            _idx = _stratified_val_indices_by_experiment(
                n_total=_n_total,
                n_val_total=len(X_va_fit),
                df_split=df_split,
                rng=_rng_eval,
            )
            Xv = X_va_fit[_idx]; Yv = Y_va_fit[_idx]
            Dv = D_va_fit[_idx]; Cv = C_va_fit[_idx]
            N = len(_idx)
            # For point metrics (EVM/SNR/dist/plots), need (N, 2) center signal.
            # For seq: Xv is (N, W, 2); Xv_center extracts the center timestep.
            # For point-wise: Xv_center == Xv (no copy, same array).
            if _arch == "seq_bigru_residual":
                Xv_center = Xv[:, Xv.shape[1] // 2, :]  # (N, 2)
            else:
                Xv_center = Xv

            rank_mode = str(candidate_analysis_quick.get("rank_mode", "mc")).lower()
            K = int(candidate_analysis_quick.get("mc_samples", 8))

            inf_det = create_inference_model_from_full(vae, deterministic=True)
            Yp_det = inf_det.predict(
                [Xv, Dv, Cv],
                batch_size=int(candidate_analysis_quick["batch_infer"]),
                verbose=0,
            )

            if rank_mode == "det" or K <= 1:
                Yp = Yp_det
                Yp_dist = Yp
                X_dist = Xv_center
                Y_dist = Yv
                var_mc = float("nan")
                Ys = None
            else:
                inf_sto = create_inference_model_from_full(vae, deterministic=False)
                Ys = []
                for _ in range(K):
                    Ys.append(inf_sto.predict(
                        [Xv, Dv, Cv],
                        batch_size=int(candidate_analysis_quick["batch_infer"]),
                        verbose=0,
                    ))
                Ys = np.stack(Ys, axis=0)
                Yp = Ys.mean(axis=0)
                # Distribution metrics must use MC concatenation (marginal predictive sample).
                Yp_dist = Ys.reshape((-1, Ys.shape[-1]))
                X_dist = np.tile(Xv_center, (K, 1))
                Y_dist = np.tile(Yv, (K, 1))
                var_mc = float(np.mean(np.var(Ys, axis=0)))

            evm_real, _ = calculate_evm(Xv_center, Yv)
            snr_real = calculate_snr(Xv_center, Yv)
            if rank_mode != "det" and K > 1:
                evm_pred, snr_pred = _mc_point_metric_means(Xv_center, Ys)
            else:
                evm_pred, _ = calculate_evm(Xv_center, Yp)
                snr_pred = calculate_snr(Xv_center, Yp)

            prior_net = vae.get_layer("prior_net")
            mu_p, logvar_p = prior_net.predict(
                [Xv, Dv, Cv],
                batch_size=int(candidate_analysis_quick["batch_infer"]),
                verbose=0,
            )

            std_mu_p = np.std(mu_p, axis=0)
            active_dims = int(np.sum(std_mu_p > 0.05))

            kl_dim = 0.5 * (np.exp(logvar_p) + mu_p ** 2 - 1.0 - logvar_p)
            kl_mean_total = float(np.mean(np.sum(kl_dim, axis=1)))
            kl_mean_per_dim = float(np.mean(np.mean(kl_dim, axis=0)))

            dist_cfg_on = bool(candidate_analysis_quick.get("dist_metrics", True))
            psd_nfft = int(candidate_analysis_quick.get("psd_nfft", 2048))
            w_psd = float(candidate_analysis_quick.get("w_psd", 0.15))
            w_skew = float(candidate_analysis_quick.get("w_skew", 0.05))
            w_kurt = float(candidate_analysis_quick.get("w_kurt", 0.05))

            if dist_cfg_on:
                distm = residual_distribution_metrics(
                    X_dist,
                    Y_dist,
                    Yp_dist,
                    psd_nfft=psd_nfft,
                    Y_samples=Ys,
                    coverage_target=Yv,
                )
                mean_l2 = float(distm["delta_mean_l2"])
                cov_fro = float(distm["delta_cov_fro"])
                var_real = float(distm["var_real_delta"])
                var_pred = float(distm["var_pred_delta"])
                skew_l2 = float(distm["delta_skew_l2"])
                kurt_l2 = float(distm["delta_kurt_l2"])
                psd_l2 = float(distm["delta_psd_l2"])
                acf_l2 = float(distm.get("delta_acf_l2", float("nan")))
            else:
                d_real = (Y_dist - X_dist)
                d_pred = (Yp_dist - X_dist)
                var_real = float(np.mean(np.var(d_real, axis=0)))
                var_pred = float(np.mean(np.var(d_pred, axis=0)))
                mean_l2 = float(np.linalg.norm(np.mean(d_pred, 0) - np.mean(d_real, 0)))
                cov_fro = float(np.linalg.norm(np.cov(d_pred.T) - np.cov(d_real.T), ord="fro"))
                skew_l2 = 0.0; kurt_l2 = 0.0; psd_l2 = 0.0; acf_l2 = float("nan")

            pen_var_mismatch = 0.0
            if not np.isnan(var_mc):
                pen_var_mismatch = float(abs(var_mc - var_real))

            history_dict = {
                k: [float(x) for x in v]
                for k, v in hist.history.items()
            }

            score_v2 = compute_score_v2(
                evm_real=evm_real, evm_pred=evm_pred,
                snr_real=snr_real, snr_pred=snr_pred,
                mean_l2=mean_l2, cov_fro=cov_fro,
                active_dims=active_dims, latent_dim=int(cfg["latent_dim"]),
                kl_mean_per_dim=kl_mean_per_dim,
                var_mc=var_mc, var_real=var_real,
                psd_l2=psd_l2, skew_l2=skew_l2, kurt_l2=kurt_l2,
                w_psd=w_psd, w_skew=w_skew, w_kurt=w_kurt,
            )
            score = abs(evm_pred - evm_real) + abs(snr_pred - snr_real)
            history_dict = {
                k: [float(x) for x in v]
                for k, v in hist.history.items()
            }

            row: Dict[str, Any] = {
                "grid_id": gi,
                "group": group,
                "tag": tag,
                **cfg,
                "status": "ok",
                "train_time_s": train_time_s,
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "evm_real_%": float(evm_real),
                "evm_pred_%": float(evm_pred),
                "delta_evm_%": float(evm_pred - evm_real),
                "snr_real_db": float(snr_real),
                "snr_pred_db": float(snr_pred),
                "delta_snr_db": float(snr_pred - snr_real),
                "score_abs_delta": float(score),
                "score_v2": float(score_v2),
                "active_dims": int(active_dims),
                "kl_mean_total": float(kl_mean_total),
                "kl_mean_per_dim": float(kl_mean_per_dim),
                "delta_mean_l2": float(mean_l2),
                "delta_cov_fro": float(cov_fro),
                "var_real_delta": float(var_real),
                "var_pred_delta": float(var_pred),
                "delta_psd_l2": float(psd_l2),
                "delta_acf_l2": float(acf_l2),
                "delta_skew_l2": float(skew_l2),
                "delta_kurt_l2": float(kurt_l2),
                "var_mc_gen": (float(var_mc) if not np.isnan(var_mc) else float("nan")),
                "pen_var_mismatch": float(pen_var_mismatch),
                "rank_mode": str(candidate_analysis_quick.get("rank_mode", "mc")).lower(),
                "mc_samples": int(candidate_analysis_quick.get("mc_samples", 8)),
            }
            ranking_mode = str(candidate_analysis_quick.get("grid_ranking_mode", "score_v2")).strip().lower()
            row["ranking_mode"] = ranking_mode
            if mini_reanalysis_callback is not None and getattr(mini_reanalysis_callback, "summary", None):
                row.update(dict(mini_reanalysis_callback.summary))
            else:
                row.update({
                    "mini_n_regimes": float("nan"),
                    "mini_n_pass": float("nan"),
                    "mini_n_partial": float("nan"),
                    "mini_n_fail": float("nan"),
                    "mini_n_g5_fail": float("nan"),
                    "mini_n_g6_fail": float("nan"),
                    "mini_mean_abs_delta_coverage_95": float("nan"),
                    "mini_mean_delta_jb": float("nan"),
                    "mini_mean_delta_psd_l2": float("nan"),
                    "mini_mean_delta_skew_l2": float("nan"),
                    "mini_mean_delta_kurt_l2": float("nan"),
                })
            row.update(
                _build_training_diagnostics(
                    history_dict=history_dict,
                    cfg=cfg,
                    max_epochs=int(training_config["epochs"]),
                    active_dims=active_dims,
                    kl_mean_total=kl_mean_total,
                    kl_mean_per_dim=kl_mean_per_dim,
                )
            )
            results.append(row)

            kl_dim_mean = np.mean(
                0.5 * (np.exp(logvar_p) + mu_p ** 2 - 1.0 - logvar_p), axis=0
            )
            summary_lines = [
                f"grid_id: {gi} | group={group} | tag={tag}",
                f"EVM real: {evm_real:.2f}% | EVM pred: {evm_pred:.2f}% | ΔEVM: {(evm_pred - evm_real):+.2f}%",
                f"SNR real: {snr_real:.2f} dB | SNR pred: {snr_pred:.2f} dB | ΔSNR: {(snr_pred - snr_real):+.2f} dB",
                f"score_abs_delta: {score:.4f}",
                f"score_v2: {score_v2:.4f}",
                f"active_dims: {active_dims}/{int(cfg['latent_dim'])} | KL_mean_total: {kl_mean_total:.3f}",
                f"epochs_ran: {row['epochs_ran']} | best_epoch: {row['best_epoch']} | "
                f"best_epoch_ratio: {row['best_epoch_ratio']:.2f}",
            ]

            results[-1]["report_png_path"] = ""
            results[-1]["report_xlsx_path"] = ""
            results[-1]["plot_bundle_dir"] = ""
            results[-1]["plot_bundle_count"] = 0

            # Save the full trainable model.
            model_path = model_dir / "model_full.keras"
            _save_keras_model_compat(vae, model_path)
            results[-1]["model_full_path"] = str(model_path)

            current_key = _candidate_ranking_key(row, ranking_mode)
            best_key = None if best_score is None else best_score
            is_best = best_key is None or current_key < best_key
            if is_best:
                best_score = current_key
                print("🏆 Novo melhor modelo do grid — salvando como 'best_model_full.keras'...")

                best_path = MODELS_DIR / "best_model_full.keras"
                _save_keras_model_compat(vae, best_path)

                _save_keras_model_compat(
                    vae.get_layer("decoder"),
                    MODELS_DIR / "best_decoder.keras",
                )
                _save_keras_model_compat(
                    vae.get_layer("prior_net"),
                    MODELS_DIR / "best_prior_net.keras",
                )

                payload = {
                    "history": history_dict,
                    "train_time_s": train_time_s,
                    "epochs_ran": int(
                        len(next(iter(hist.history.values())))
                        if hist.history else 0
                    ),
                    "grid_cfg": cfg,
                    "grid_id": gi,
                    "group": group,
                    "tag": tag,
                    "score_abs_delta": float(score),
                    "score_v2": float(score_v2),
                    "active_dims": int(active_dims),
                    "kl_mean_total": float(kl_mean_total),
                    "kl_mean_per_dim": float(kl_mean_per_dim),
                    "delta_mean_l2": float(mean_l2),
                    "delta_cov_fro": float(cov_fro),
                }
                hist_path = run_paths.write_json(
                    "logs/training_history.json", payload
                )
                print(f"✓ training_history.json salvo: {hist_path}")

                best_grid_plots_dir = run_paths.plots_dir
                try:
                    dashboard_path = save_champion_analysis_dashboard(
                        plots_dir=best_grid_plots_dir,
                        Xv=Xv_center,
                        Yv=Yv,
                        Yp=Yp,
                        std_mu_p=std_mu_p,
                        kl_dim_mean=kl_dim_mean,
                        summary_lines=summary_lines + [
                            f"ranking criterion: {ranking_mode}",
                        ],
                        model_label=f"Champion ({tag})",
                        title=f"Champion Analysis Dashboard | {tag}",
                    )
                    print(
                        f"✓ Champion analysis dashboard salvo: {dashboard_path}"
                    )
                except ModuleNotFoundError as exc:
                    if exc.name != "matplotlib":
                        raise
                    print(
                        "⚠️  matplotlib não está instalado neste ambiente; "
                        "pulando geração de plots do melhor grid sem invalidar o resultado."
                    )

        except Exception as e:
            print(f"[ERRO] Falha no grid_id={gi} tag={tag}: {repr(e)}")
            ranking_mode = str(candidate_analysis_quick.get("grid_ranking_mode", "score_v2")).strip().lower()
            results.append({
                "grid_id": gi,
                "group": group,
                "tag": tag,
                **cfg,
                "status": "FAILED",
                "ranking_mode": ranking_mode,
                "train_time_s": float("nan"),
                "best_epoch": 0,
                "best_val_loss": float("nan"),
                "evm_real_%": float("nan"),
                "evm_pred_%": float("nan"),
                "delta_evm_%": float("nan"),
                "snr_real_db": float("nan"),
                "snr_pred_db": float("nan"),
                "delta_snr_db": float("nan"),
                "score_abs_delta": float("inf"),
                "score_v2": float("inf"),
                "active_dims": 0,
                "kl_mean_total": float("nan"),
                "kl_mean_per_dim": float("nan"),
                "delta_mean_l2": float("nan"),
                "delta_cov_fro": float("nan"),
                "delta_psd_l2": float("nan"),
                "delta_acf_l2": float("nan"),
                "delta_skew_l2": float("nan"),
                "delta_kurt_l2": float("nan"),
                "var_real_delta": float("nan"),
                "var_pred_delta": float("nan"),
                "pen_var_mismatch": float("nan"),
                "mini_n_regimes": float("nan"),
                "mini_n_pass": float("nan"),
                "mini_n_partial": float("nan"),
                "mini_n_fail": float("nan"),
                "mini_n_g5_fail": float("nan"),
                "mini_n_g6_fail": float("nan"),
                "mini_mean_abs_delta_coverage_95": float("nan"),
                "mini_mean_delta_jb": float("nan"),
                "mini_mean_delta_psd_l2": float("nan"),
                "mini_mean_delta_skew_l2": float("nan"),
                "mini_mean_delta_kurt_l2": float("nan"),
                "model_full_path": "",
                "report_png_path": "",
                "report_xlsx_path": "",
            })
            continue

    # --- Build sorted results table ---
    df_results = pd.DataFrame(results)
    ranking_modes = (
        df_results.get("ranking_mode", pd.Series(dtype=object))
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )
    ranking_mode = ranking_modes[0] if len(ranking_modes) == 1 else str(
        analysis_quick.get("grid_ranking_mode", "score_v2")
    ).strip().lower()
    df_results = _sort_results_by_ranking(df_results, ranking_mode)
    df_results.insert(0, "rank", np.arange(1, len(df_results) + 1))
    df_results = _apply_training_recommendations(df_results)
    df_diag = _build_training_diagnostics_table(df_results)

    rank_rows = [
        {
            "Item": "Objetivo do ranking",
            "Descrição": (
                "Selecionar o melhor digital twin que preserva estatísticas "
                "do canal medido e evita colapso do latente."
            ),
        },
    ]
    if ranking_mode == "mini_protocol_v1":
        rank_rows.append(
            {
                "Item": "Ranking principal",
                "Descrição": (
                    "mini_protocol_v1 = ordem lexicográfica por mini_n_fail, "
                    "mini_n_g6_fail, mini_mean_abs_delta_coverage_95, "
                    "mini_mean_delta_jb, mini_mean_delta_psd_l2 e score_v2 "
                    "apenas como desempate final."
                ),
            }
        )
    else:
        rank_rows.append(
            {
                "Item": "Score principal (score_v2)",
                "Descrição": (
                    "score_v2 = |ΔEVM| + |ΔSNR| + 0.4·Δμ + 0.2·ΔΣ + "
                    "2·pen(dims_inativas) + 1·pen(KL_dim_baixo) + "
                    "termos PSD/skew/kurt/varMC. Menor é melhor."
                ),
            }
        )
    df_rank_readme = pd.DataFrame(rank_rows)

    res_path = run_paths.run_dir / "tables" / "gridsearch_results.xlsx"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(res_path, engine="openpyxl") as w:
        df_results.to_excel(w, index=False, sheet_name="results_sorted")
        if df_plan is not None:
            df_plan.to_excel(w, index=False, sheet_name="grid_plan_structured")
        checklist_table().to_excel(w, index=False, sheet_name="checklist_train_vs_collapse")
        df_rank_readme.to_excel(w, index=False, sheet_name="RANKING_README")
    print(f"📈 Grid results salvo: {res_path}")

    # Also save CSV for convenience
    run_paths.write_table("tables/gridsearch_results.csv", df_results)
    diag_csv_path = run_paths.write_table("tables/grid_training_diagnostics.csv", df_diag)
    print(f"📈 Training diagnostics salvo: {diag_csv_path}")

    try:
        training_dashboard_path = save_training_analysis_dashboard(
            df_diag=df_diag,
            plots_dir=run_paths.plots_dir,
        )
        print(f"✓ Training analysis dashboard salvo: {training_dashboard_path}")
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print(
            "⚠️  matplotlib não está instalado neste ambiente; "
            "pulando dashboard operacional de treinamento."
        )

    return df_results
