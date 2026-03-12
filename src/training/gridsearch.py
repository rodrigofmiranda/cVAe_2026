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
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    summary = pd.DataFrame([row_dict])
    cfg_keys = [
        "activation", "kl_anneal_epochs", "batch_size", "lr",
        "dropout", "free_bits", "layer_sizes", "latent_dim", "beta",
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
    import tensorflow as tf
    from src.models.cvae import build_cvae, create_inference_model_from_full
    from src.models.callbacks import build_callbacks
    from src.evaluation.metrics import (
        calculate_evm, calculate_snr, residual_distribution_metrics,
    )

    _ov = overrides or {}
    MODELS_DIR = run_paths.models_dir

    results: List[Dict[str, Any]] = []
    best_score: Optional[float] = None

    for gi, item in enumerate(grid, start=1):
        cfg = item["cfg"]
        group = item["group"]
        tag = item["tag"]

        print("\n" + "=" * 92)
        print(f"🚀 GRID {gi}/{len(grid)} | group={group} | tag={tag}")
        print(f"    cfg = {cfg}")
        print("=" * 92)

        tf.keras.backend.clear_session()
        gc.collect()

        try:
            vae, kl_cb = build_cvae(cfg)
            callbacks = build_callbacks(training_config, cfg, kl_cb)

            t0 = time.time()
            _keras_verbose = int(_ov.get("keras_verbose", 2))
            _bs_cfg = int(cfg["batch_size"])
            # Small-smoke stability: avoid single-step epochs when train << batch_size.
            if len(X_train) < _bs_cfg:
                _bs_eff = max(128, len(X_train) // 64)
            else:
                _bs_eff = _bs_cfg
            hist = vae.fit(
                [X_train, Dn_train, Cn_train, Y_train], Y_train,
                validation_data=([X_val, Dn_val, Cn_val, Y_val], Y_val),
                epochs=int(training_config["epochs"]),
                batch_size=int(_bs_eff),
                callbacks=callbacks,
                verbose=_keras_verbose,
                shuffle=bool(training_config["shuffle_train_batches"]),
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
            _n_total = int(analysis_quick["n_eval_samples"])
            _n_total = max(1, min(_n_total, len(X_val)))
            _rng_eval = np.random.default_rng(int(training_config.get("seed", 42)))
            _idx = _stratified_val_indices_by_experiment(
                n_total=_n_total,
                n_val_total=len(X_val),
                df_split=df_split,
                rng=_rng_eval,
            )
            Xv = X_val[_idx]; Yv = Y_val[_idx]
            Dv = Dn_val[_idx]; Cv = Cn_val[_idx]
            N = len(_idx)

            rank_mode = str(analysis_quick.get("rank_mode", "mc")).lower()
            K = int(analysis_quick.get("mc_samples", 8))

            inf_det = create_inference_model_from_full(vae, deterministic=True)
            Yp_det = inf_det.predict(
                [Xv, Dv, Cv],
                batch_size=int(analysis_quick["batch_infer"]),
                verbose=0,
            )

            if rank_mode == "det" or K <= 1:
                Yp = Yp_det
                Yp_dist = Yp
                X_dist = Xv
                Y_dist = Yv
                var_mc = float("nan")
            else:
                inf_sto = create_inference_model_from_full(vae, deterministic=False)
                Ys = []
                for _ in range(K):
                    Ys.append(inf_sto.predict(
                        [Xv, Dv, Cv],
                        batch_size=int(analysis_quick["batch_infer"]),
                        verbose=0,
                    ))
                Ys = np.stack(Ys, axis=0)
                # Point metrics (EVM/SNR) stay on MC mean.
                Yp = Ys.mean(axis=0)
                # Distribution metrics must use MC concatenation (marginal predictive sample).
                Yp_dist = Ys.reshape((-1, Ys.shape[-1]))
                X_dist = np.tile(Xv, (K, 1))
                Y_dist = np.tile(Yv, (K, 1))
                var_mc = float(np.mean(np.var(Ys, axis=0)))

            evm_real, _ = calculate_evm(Xv, Yv)
            evm_pred, _ = calculate_evm(Xv, Yp)
            snr_real = calculate_snr(Xv, Yv)
            snr_pred = calculate_snr(Xv, Yp)

            prior_net = vae.get_layer("prior_net")
            mu_p, logvar_p = prior_net.predict(
                [Xv, Dv, Cv],
                batch_size=int(analysis_quick["batch_infer"]),
                verbose=0,
            )

            std_mu_p = np.std(mu_p, axis=0)
            active_dims = int(np.sum(std_mu_p > 0.05))

            kl_dim = 0.5 * (np.exp(logvar_p) + mu_p ** 2 - 1.0 - logvar_p)
            kl_mean_total = float(np.mean(np.sum(kl_dim, axis=1)))
            kl_mean_per_dim = float(np.mean(np.mean(kl_dim, axis=0)))

            dist_cfg_on = bool(analysis_quick.get("dist_metrics", True))
            psd_nfft = int(analysis_quick.get("psd_nfft", 2048))
            w_psd = float(analysis_quick.get("w_psd", 0.15))
            w_skew = float(analysis_quick.get("w_skew", 0.05))
            w_kurt = float(analysis_quick.get("w_kurt", 0.05))

            if dist_cfg_on:
                distm = residual_distribution_metrics(X_dist, Y_dist, Yp_dist, psd_nfft=psd_nfft)
                mean_l2 = float(distm["delta_mean_l2"])
                cov_fro = float(distm["delta_cov_fro"])
                var_real = float(distm["var_real_delta"])
                var_pred = float(distm["var_pred_delta"])
                skew_l2 = float(distm["delta_skew_l2"])
                kurt_l2 = float(distm["delta_kurt_l2"])
                psd_l2 = float(distm["delta_psd_l2"])
            else:
                d_real = (Y_dist - X_dist)
                d_pred = (Yp_dist - X_dist)
                var_real = float(np.mean(np.var(d_real, axis=0)))
                var_pred = float(np.mean(np.var(d_pred, axis=0)))
                mean_l2 = float(np.linalg.norm(np.mean(d_pred, 0) - np.mean(d_real, 0)))
                cov_fro = float(np.linalg.norm(np.cov(d_pred.T) - np.cov(d_real.T), ord="fro"))
                skew_l2 = 0.0; kurt_l2 = 0.0; psd_l2 = 0.0

            pen_var_mismatch = 0.0
            if not np.isnan(var_mc):
                pen_var_mismatch = float(abs(var_mc - var_real))

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
                "delta_skew_l2": float(skew_l2),
                "delta_kurt_l2": float(kurt_l2),
                "var_mc_gen": (float(var_mc) if not np.isnan(var_mc) else float("nan")),
                "pen_var_mismatch": float(pen_var_mismatch),
                "rank_mode": str(analysis_quick.get("rank_mode", "mc")).lower(),
                "mc_samples": int(analysis_quick.get("mc_samples", 8)),
            }
            results.append(row)

            # Per-grid artifact directory
            model_dir = _grid_artifact_dir(MODELS_DIR, gi, tag)
            exp_plots_dir = model_dir / "plots"
            exp_tables_dir = model_dir / "tables"
            exp_plots_dir.mkdir(parents=True, exist_ok=True)
            exp_tables_dir.mkdir(parents=True, exist_ok=True)

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
            ]

            png_path = exp_plots_dir / "relatorio_completo_original_style.png"
            title = f"Relatório Consolidado — Twin + Latente | GRID {gi}/{len(grid)}"
            save_experiment_report_png(
                plot_path=png_path, Xv=Xv, Yv=Yv, Yp=Yp,
                std_mu_p=std_mu_p, kl_dim_mean=kl_dim_mean,
                summary_lines=summary_lines, title=title,
            )

            xlsx_path = exp_tables_dir / "relatorio_diagnostico_completo.xlsx"
            save_experiment_xlsx(xlsx_path=xlsx_path, row_dict=row)

            results[-1]["report_png_path"] = str(png_path)
            results[-1]["report_xlsx_path"] = str(xlsx_path)

            # Save individual model
            model_path = model_dir / "model_full.keras"
            vae.save(str(model_path), include_optimizer=False)
            results[-1]["model_full_path"] = str(model_path)

            is_best = (best_score is None) or (score_v2 < best_score)
            if is_best:
                best_score = float(score_v2)
                print("🏆 Novo melhor modelo do grid — salvando como 'best_model_full.keras'...")

                best_path = MODELS_DIR / "best_model_full.keras"
                vae.save(str(best_path), include_optimizer=False)

                vae.get_layer("decoder").save(
                    str(MODELS_DIR / "best_decoder.keras"), include_optimizer=False
                )
                vae.get_layer("prior_net").save(
                    str(MODELS_DIR / "best_prior_net.keras"), include_optimizer=False
                )

                payload = {
                    "history": {
                        k: [float(x) for x in v]
                        for k, v in hist.history.items()
                    },
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

        except Exception as e:
            print(f"[ERRO] Falha no grid_id={gi} tag={tag}: {repr(e)}")
            results.append({
                "grid_id": gi,
                "group": group,
                "tag": tag,
                **cfg,
                "status": "FAILED",
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
                "model_full_path": "",
                "report_png_path": "",
                "report_xlsx_path": "",
            })
            continue

    # --- Build sorted results table ---
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(
        ["score_v2", "score_abs_delta"], ascending=[True, True]
    )
    df_results.insert(0, "rank", np.arange(1, len(df_results) + 1))

    df_rank_readme = pd.DataFrame([
        {
            "Item": "Objetivo do ranking",
            "Descrição": (
                "Selecionar o melhor digital twin que preserva estatísticas "
                "do canal medido e evita colapso do latente."
            ),
        },
        {
            "Item": "Score principal (score_v2)",
            "Descrição": (
                "score_v2 = |ΔEVM| + |ΔSNR| + 0.4·Δμ + 0.2·ΔΣ + "
                "2·pen(dims_inativas) + 1·pen(KL_dim_baixo) + "
                "termos PSD/skew/kurt/varMC. Menor é melhor."
            ),
        },
    ])

    res_path = run_paths.run_dir / "tables" / "gridsearch_results.xlsx"
    with pd.ExcelWriter(res_path, engine="openpyxl") as w:
        df_results.to_excel(w, index=False, sheet_name="results_sorted")
        if df_plan is not None:
            df_plan.to_excel(w, index=False, sheet_name="grid_plan_structured")
        checklist_table().to_excel(w, index=False, sheet_name="checklist_train_vs_collapse")
        df_rank_readme.to_excel(w, index=False, sheet_name="RANKING_README")
    print(f"📈 Grid results salvo: {res_path}")

    # Also save CSV for convenience
    run_paths.write_table("tables/gridsearch_results.csv", df_results)

    return df_results
