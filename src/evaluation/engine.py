# -*- coding: utf-8 -*-
"""Canonical cVAE evaluation engine without env-driven orchestration."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config.runtime import build_evaluation_runtime
from src.config.runtime_env import ensure_writable_mpl_config_dir
from src.data.loading import load_experiments_as_list
from src.data.normalization import apply_condition_norm, load_normalization_from_state
from src.data.splits import cap_train_samples_per_experiment
from src.evaluation.metrics import calculate_evm, calculate_snr, residual_distribution_metrics
from src.evaluation.report import (
    build_global_metrics,
    build_summary_text,
    compute_latent_diagnostics,
    decoder_sensitivity,
)
from src.models.cvae import create_inference_model_from_full
from src.models.cvae_sequence import load_seq_model
from src.models.sampling import Sampling
from src.protocol.split_strategies import apply_split
from src.training.grid_plots import save_champion_analysis_dashboard
from src.training.logging import RunPaths


_EVAL_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_eval_cached_entry(best_model_path: Path) -> Dict[str, Any]:
    """Load and cache the full eval model plus its inference sub-graphs."""
    key = str(Path(best_model_path).resolve())
    entry = _EVAL_MODEL_CACHE.get(key)
    if entry is not None:
        return entry

    vae = load_seq_model(str(best_model_path))
    entry = {
        "vae": vae,
        "inference_det": create_inference_model_from_full(vae, deterministic=True),
        "inference_sto": create_inference_model_from_full(vae, deterministic=False),
    }
    _EVAL_MODEL_CACHE[key] = entry
    return entry


def clear_evaluation_model_cache():
    """Release cached evaluation models after the protocol finishes."""
    _EVAL_MODEL_CACHE.clear()
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass


def _autofind(path: Path, patterns):
    for pattern in patterns:
        matches = list(path.glob(pattern))
        if matches:
            return matches[0]
    return None


def _fallback_state(
    run_dir: Path,
    *,
    dataset_root: str | Path | None = None,
) -> Dict[str, Any]:
    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"
    best_model = _autofind(models_dir, ["best_model_full.keras", "*.keras"])
    if best_model is None:
        raise FileNotFoundError(f"Não encontrei modelo em {models_dir} (ex.: best_model_full.keras).")

    history_path = _autofind(logs_dir, ["training_history.json"])
    if dataset_root is None:
        raise ValueError("dataset_root must be provided when state_run.json is missing.")

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "dataset_root": str(Path(dataset_root)),
        "output_base": str(run_dir.parent),
        "training_config": {"seed": 42, "validation_split": 0.2},
        "normalization": None,
        "data_split": {
            "split_mode": "global",
            "validation_split": 0.2,
            "seed": 42,
        },
        "eval_protocol": {
            "deterministic_inference": False,
            "mc_samples": 8,
            "rank_mode": "mc",
            "n_eval_samples": 40000,
            "batch_infer": 8192,
            "eval_slice": "stratified",
        },
        "analysis_quick": {
            "dist_metrics": True,
            "psd_nfft": 2048,
            "w_psd": 0.15,
            "w_skew": 0.05,
            "w_kurt": 0.05,
        },
        "artifacts": {
            "best_model_full": str(best_model),
            "training_history_json": str(history_path) if history_path is not None else "",
        },
    }


def _first2(outputs):
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
        return outputs[0], outputs[1]
    raise ValueError(f"Saída inesperada do modelo auxiliar: {type(outputs)}")


def _stratified_val_indices_by_experiment(
    *,
    n_total: int,
    n_val_total: int,
    df_split: Optional[pd.DataFrame],
    rng: np.random.Generator,
) -> np.ndarray:
    idx_eval: Optional[np.ndarray] = None

    if isinstance(df_split, pd.DataFrame) and "n_val" in df_split.columns:
        n_val_list = [int(v) for v in df_split["n_val"].tolist()]
        if n_val_list and int(np.sum(n_val_list)) == int(n_val_total):
            n_exps = len(n_val_list)
            base = n_total // n_exps
            rem = n_total % n_exps

            target = np.full(n_exps, base, dtype=np.int64)
            if rem > 0:
                target[rng.permutation(n_exps)[:rem]] += 1

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


def evaluate_run(
    run_dir: str | Path,
    *,
    dataset_root: str | Path | None = None,
    overrides: Optional[Dict[str, Any]] = None,
    output_run_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Evaluate a trained cVAE run and persist the canonical artifacts."""
    ensure_writable_mpl_config_dir()

    model_run_dir = Path(run_dir).resolve()
    output_dir = Path(output_run_dir).resolve() if output_run_dir is not None else model_run_dir
    _incoming_ov = dict(overrides or {})
    run_paths = RunPaths.from_existing(output_dir, logs_dir=_incoming_ov.get("_logs_dir"))
    state_path = model_run_dir / "state_run.json"

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        print("✓ state_run.json carregado.")
    else:
        print("⚠ state_run.json não encontrado — usando fallback mínimo.")
        state = _fallback_state(model_run_dir, dataset_root=dataset_root)

    runtime = build_evaluation_runtime(
        run_dir=model_run_dir,
        dataset_root=dataset_root,
        state=state,
        overrides=_incoming_ov,
    )
    ov = dict(runtime.overrides)

    print(f"📁 MODEL_RUN_DIR = {model_run_dir}")
    print(f"📁 OUTPUT_RUN_DIR = {output_dir}")
    print(f"📁 DATASET_ROOT = {runtime.dataset_root}")

    artifacts = runtime.state.get("artifacts", {})
    best_model_candidate = str(artifacts.get("best_model_full", "")).strip()
    best_model_path = (
        Path(best_model_candidate)
        if best_model_candidate
        else _autofind(run_paths.models_dir, ["best_model_full.keras", "*.keras"])
    )
    if best_model_path is None:
        raise FileNotFoundError(f"best_model_full.keras não encontrado em {run_paths.models_dir}")
    if not best_model_path.exists():
        raise FileNotFoundError(f"best_model_full.keras não encontrado: {best_model_path}")

    print(f"📦 Carregando modelo: {best_model_path}")
    _runtime = _get_eval_cached_entry(best_model_path)
    vae = _runtime["vae"]

    layer_names = {layer.name for layer in vae.layers}
    print(
        "🔎 Layers-chave:",
        {
            "encoder": "encoder" in layer_names,
            "prior_net": "prior_net" in layer_names,
            "decoder": "decoder" in layer_names,
        },
    )
    if not {"prior_net", "decoder"}.issubset(layer_names):
        raise ValueError("Modelo carregado não contém prior_net/decoder.")

    encoder = vae.get_layer("encoder") if "encoder" in layer_names else None
    prior = vae.get_layer("prior_net")
    decoder = vae.get_layer("decoder")

    # Detect seq_bigru_residual via prior input rank (rank-3 → windowed sequence input)
    _is_seq = len(prior.inputs[0].shape) == 3
    if _is_seq:
        print(
            f"🔄 seq_bigru_residual detected (prior input rank=3, W={prior.inputs[0].shape[1]}) "
            "— windowed inference will be applied."
        )

    arch_variant = str(
        runtime.training_config.get(
            "arch_variant",
            runtime.state.get("training_config", {}).get("arch_variant", ""),
        )
    ).strip().lower()
    if not arch_variant:
        if vae.name == "cvae_legacy_2025_zero_y":
            arch_variant = "legacy_2025_zero_y"
        elif vae.name == "cvae_condprior_delta_residual":
            arch_variant = "delta_residual"
        elif _is_seq:
            arch_variant = "seq_bigru_residual"
        else:
            arch_variant = "concat"
    latent_prior_semantics = (
        "std_normal_legacy_2025_zero_y"
        if arch_variant == "legacy_2025_zero_y"
        else "conditional_prior"
    )

    seed0 = int(ov.get("seed", runtime.training_config.get("seed", 42)))
    np.random.seed(seed0)
    tf.random.set_seed(seed0)

    evalp = dict(runtime.eval_protocol)
    analysis_quick = dict(runtime.analysis_quick)
    det_inf = bool(evalp.get("deterministic_inference", True))
    mc_samples = int(evalp.get("mc_samples", 1))
    rank_mode = str(evalp.get("rank_mode", ("det" if det_inf else "mc"))).lower()

    inference_model = (
        _runtime["inference_det"]
        if det_inf or rank_mode == "det" or mc_samples <= 1
        else _runtime["inference_sto"]
    )
    print(
        "✅ inference_model pronto:",
        inference_model.name,
        "| deterministic =",
        det_inf,
        "| mc_samples =",
        mc_samples,
        "| rank_mode =",
        rank_mode,
    )

    if ov.get("dry_run", False):
        dry_info = {
            "dry_run": True,
            "overrides": {k: v for k, v in ov.items()},
            "inference_model": inference_model.name,
            "deterministic": det_inf,
            "model_path": str(best_model_path),
            "model_run_dir": str(model_run_dir),
            "output_run_dir": str(output_dir),
        }
        path = run_paths.write_json("logs/dry_run_eval.json", dry_info)
        print(f"✅ dry_run complete — wrote {path}")
        return {"status": "dry_run", "run_dir": str(output_dir)}

    exps, _df_info = load_experiments_as_list(runtime.dataset_root, verbose=True)

    selected_experiments = ov.get("_selected_experiments")
    if selected_experiments:
        selected = set(str(p) for p in selected_experiments)
        before = len(exps)
        exps = [(X, Y, D, C, p) for X, Y, D, C, p in exps if str(p) in selected]
        print(f"⚡ selected_experiments filter: {before} → {len(exps)} experiment(s)")

    if ov.get("max_experiments") is not None:
        max_experiments = int(ov["max_experiments"])
        exps = exps[:max_experiments]
        print(f"⚡ max_experiments aplicado: {len(exps)} experiment(s)")

    if not exps:
        raise RuntimeError("Evaluation resolved zero experiments after filtering.")

    data_split = runtime.state.get("data_split", {})
    split_mode = str(
        data_split.get(
            "split_mode",
            runtime.training_config.get("split_mode", "global"),
        )
    ).lower()
    val_split = float(
        data_split.get(
            "validation_split",
            runtime.training_config.get("validation_split", 0.2),
        )
    )
    within_shuffle = bool(
        data_split.get(
            "within_experiment_shuffle",
            runtime.training_config.get("within_experiment_shuffle", False),
        )
    )

    (
        X_train,
        Y_train,
        D_train,
        C_train,
        X_val,
        Y_val,
        D_val,
        C_val,
        df_split,
    ) = apply_split(
        exps=exps,
        strategy=split_mode,
        val_split=val_split,
        seed=seed0,
        within_exp_shuffle=within_shuffle,
    )

    if split_mode == "per_experiment" and ov.get("max_samples_per_exp") is not None:
        max_samples = int(ov["max_samples_per_exp"])
        X_train, Y_train, D_train, C_train, _ = cap_train_samples_per_experiment(
            X_train,
            Y_train,
            D_train,
            C_train,
            df_split,
            max_samples,
        )
        print(
            f"⚡ max_samples_per_exp pós-split (train cap={max_samples}/exp) | "
            f"train={len(X_train):,} | val={len(X_val):,}"
        )

    print(f"✓ Split aplicado | train={len(X_train):,} | val={len(X_val):,} | mode={split_mode}")

    # --- Seq windowing (after split+cap, before eval slicing) ---
    # Produces X_val_w (N_val, W, 2) for seq models; None for point-wise models.
    # X_val_w[i, W//2, :] == X_val[i, :] by construction (centered edge-padded windows).
    X_val_w = None
    if _is_seq:
        from src.data.windowing import build_windows_from_split_arrays
        _ds_info = runtime.state.get("data_split", {})
        _ws = int(_ds_info.get("window_size", int(prior.inputs[0].shape[1])))
        _wst = int(_ds_info.get("window_stride", 1))
        _wpm = str(_ds_info.get("window_pad_mode", "edge"))
        _, _, _, _, X_val_w, _, _, _ = build_windows_from_split_arrays(
            X_train, Y_train, D_train, C_train,
            X_val, Y_val, D_val, C_val,
            df_split=df_split,
            window_size=_ws,
            stride=_wst,
            pad_mode=_wpm,
        )
        print(f"✓ Seq windowing: X_val_w={X_val_w.shape} (W={_ws}, stride={_wst})")

    norm = load_normalization_from_state(runtime.state)
    if norm is None:
        norm = {
            "D_min": float(D_train.min()),
            "D_max": float(D_train.max()),
            "C_min": float(C_train.min()),
            "C_max": float(C_train.max()),
        }
        print(
            f"⚠ Normalização calculada no treino (fallback): "
            f"D=[{norm['D_min']:.3f},{norm['D_max']:.3f}] | "
            f"C=[{norm['C_min']:.1f},{norm['C_max']:.1f}]"
        )

    Dn_val, Cn_val = apply_condition_norm(D_val.ravel(), C_val.ravel(), norm)
    Dn_val = Dn_val.reshape(-1, 1)
    Cn_val = Cn_val.reshape(-1, 1)

    n_eval_samples = int(evalp.get("n_eval_samples", 40_000))
    batch_infer = int(evalp.get("batch_infer", 8192))
    eval_slice = str(evalp.get("eval_slice", "val_head"))
    if eval_slice == "stratified_by_regime":
        eval_slice = "stratified"
    if eval_slice not in {"val_head", "stratified", "full"}:
        print(f"⚠️  eval_slice='{eval_slice}' inválido, usando 'val_head'")
        eval_slice = "val_head"

    if eval_slice == "stratified":
        n_eval = min(n_eval_samples, len(X_val))
        rng_eval = np.random.default_rng(seed0)
        idx_eval = _stratified_val_indices_by_experiment(
            n_total=n_eval,
            n_val_total=len(X_val),
            df_split=df_split,
            rng=rng_eval,
        )
        Xv, Yv = X_val[idx_eval], Y_val[idx_eval]
        Dv, Cv = Dn_val[idx_eval], Cn_val[idx_eval]
        print(
            f"✓ eval_slice=stratified | N={len(idx_eval):,} | "
            f"n_experiments={len(df_split) if isinstance(df_split, pd.DataFrame) else 0}"
        )
    elif eval_slice == "full":
        Xv, Yv, Dv, Cv = X_val, Y_val, Dn_val, Cn_val
    else:
        n_eval = min(n_eval_samples, len(X_val))
        Xv, Yv, Dv, Cv = X_val[:n_eval], Y_val[:n_eval], Dn_val[:n_eval], Cn_val[:n_eval]

    n_eval = len(Xv)

    # --- Seq: select matching windowed slice; assign Xv_in (model input) and Xv_center (metrics) ---
    # For seq: Xv_in is (N, W, 2); Xv_center is (N, 2) — identical to Xv by construction.
    # For point-wise: Xv_in == Xv_center == Xv.
    if _is_seq:
        if eval_slice == "stratified":
            Xv_in = X_val_w[idx_eval]
        elif eval_slice == "full":
            Xv_in = X_val_w
        else:
            Xv_in = X_val_w[:n_eval]
        Xv_center = Xv  # center frame = original point-wise array
    else:
        Xv_in = Xv
        Xv_center = Xv

    if det_inf or mc_samples <= 1 or rank_mode == "det":
        Yp = inference_model.predict([Xv_in, Dv, Cv], batch_size=batch_infer, verbose=0)
        Yp_dist = Yp
        X_dist = Xv_center
        Y_dist = Yv
        var_mc = float("nan")
        Ys = None
    else:
        inf_sto = _runtime["inference_sto"]
        Ys = []
        for _ in range(int(mc_samples)):
            Ys.append(inf_sto.predict([Xv_in, Dv, Cv], batch_size=batch_infer, verbose=0))
        Ys = np.stack(Ys, axis=0)
        Yp = Ys.mean(axis=0)
        Yp_dist = Ys.reshape((-1, Ys.shape[-1]))
        X_dist = np.tile(Xv_center, (int(mc_samples), 1))
        Y_dist = np.tile(Yv, (int(mc_samples), 1))
        var_mc = float(np.mean(np.var(Ys, axis=0)))

    evm_real, _ = calculate_evm(Xv_center, Yv)
    snr_real = calculate_snr(Xv_center, Yv)
    # For stochastic inference, compute EVM/SNR as the mean over individual MC
    # draws rather than from the ensemble mean Yp.  The ensemble mean cancels
    # stochastic noise and produces artificially low EVM, making the digital
    # twin appear "cleaner" than the real channel.  Individual-draw EVM
    # reflects the per-realisation fidelity that G1/G2 intend to measure.
    if not (det_inf or mc_samples <= 1 or rank_mode == "det") and len(Ys) > 0:
        evm_mc = float(np.mean([calculate_evm(Xv_center, Ys[i])[0]
                                for i in range(len(Ys))]))
        snr_mc = float(np.mean([calculate_snr(Xv_center, Ys[i])
                                for i in range(len(Ys))]))
        evm_pred = evm_mc
        snr_pred = snr_mc
    else:
        evm_pred, _ = calculate_evm(Xv_center, Yp)
        snr_pred = calculate_snr(Xv_center, Yp)

    dist_on = bool(analysis_quick.get("dist_metrics", True)) and not bool(ov.get("no_dist_metrics", False))
    psd_nfft = int(ov.get("psd_nfft", analysis_quick.get("psd_nfft", 2048)))
    gauss_alpha = float(ov.get("gauss_alpha", 0.01))
    max_dist_samples = ov.get("max_dist_samples")

    if dist_on:
        if max_dist_samples is not None:
            max_dist_samples = max(1, min(int(max_dist_samples), len(X_dist)))
            if max_dist_samples < len(X_dist):
                rng_dist = np.random.default_rng(seed0)
                idx_dist = np.sort(
                    rng_dist.choice(len(X_dist), size=max_dist_samples, replace=False)
                )
                X_dist = X_dist[idx_dist]
                Y_dist = Y_dist[idx_dist]
                Yp_dist = Yp_dist[idx_dist]
                print(f"✓ max_dist_samples aplicado na avaliação | N={max_dist_samples:,}")
        distm = residual_distribution_metrics(
            X_dist,
            Y_dist,
            Yp_dist,
            psd_nfft=psd_nfft,
            gauss_alpha=gauss_alpha,
            Y_samples=Ys,
            coverage_target=Yv,
        )
    else:
        distm = {
            "delta_mean_l2": float("nan"),
            "delta_cov_fro": float("nan"),
            "var_real_delta": float("nan"),
            "var_pred_delta": float("nan"),
            "delta_skew_l2": float("nan"),
            "delta_kurt_l2": float("nan"),
            "delta_psd_l2": float("nan"),
            "delta_acf_l2": float("nan"),
            "rho_hetero_real": float("nan"),
            "rho_hetero_pred": float("nan"),
            "stat_jsd": float("nan"),
        }

    global_metrics = build_global_metrics(
        run_id=output_dir.name,
        model_path=str(best_model_path),
        split_mode=split_mode,
        N_eval=int(n_eval),
        evm_real=float(evm_real),
        evm_pred=float(evm_pred),
        snr_real=float(snr_real),
        snr_pred=float(snr_pred),
        distm=distm,
        det_inf=bool(det_inf),
        rank_mode=str(rank_mode),
        mc_samples=int(mc_samples),
        var_mc=float(var_mc) if not np.isnan(var_mc) else float("nan"),
        arch_variant=arch_variant,
        latent_prior_semantics=latent_prior_semantics,
    )

    metrics_json = run_paths.write_json("logs/metricas_globais_reanalysis.json", global_metrics)
    print(f"✓ metricas_globais_reanalysis.json salvo: {metrics_json}")

    pri_out = prior.predict([Xv_in, Dv, Cv], batch_size=batch_infer, verbose=0)
    z_mean_p, z_log_var_p = _first2(pri_out)

    if encoder is not None:
        enc_out = encoder.predict([Xv_in, Dv, Cv, Yv], batch_size=batch_infer, verbose=0)
        z_mean_q, z_log_var_q = _first2(enc_out)
    else:
        # Adversarial wrapper saved as inference model: encoder not available.
        # Use prior as a stand-in so latent diagnostics can still report prior stats.
        z_mean_q = z_mean_p
        z_log_var_q = z_log_var_p

    lat_diag = compute_latent_diagnostics(
        z_mean_q,
        z_log_var_q,
        z_mean_p,
        z_log_var_p,
        arch_variant=arch_variant,
    )
    df_lat = lat_diag["df_lat"]
    lat_summary = lat_diag["lat_summary"]
    z_std_p = lat_diag["z_std_p"]
    active_dims = lat_diag["active_dims"]
    kl_qp_total_mean = lat_diag["kl_qp_total_mean"]
    kl_pN_total_mean = lat_diag["kl_pN_total_mean"]

    run_paths.write_json("logs/latent_summary.json", lat_summary)

    nb = min(20000, n_eval)
    if _is_seq:
        # decoder_sensitivity uses a concatenated [X, D, C] cond input which is
        # incompatible with the seq decoder interface ([z, x_center, d, c] separately).
        # Skipped for seq_bigru_residual; a seq-specific sensitivity will be added later.
        sens = {
            "decoder_output_variance_mean": float("nan"),
            "decoder_output_rms_std": float("nan"),
        }
        print("ℹ️  decoder_sensitivity skipped for seq_bigru_residual variant.")
    else:
        sens = decoder_sensitivity(
            prior,
            decoder,
            Xv[:nb],
            Dv[:nb],
            Cv[:nb],
            n_mc_z=16,
            batch_size=batch_infer,
            arch_variant=arch_variant,
        )
    run_paths.write_json("logs/decoder_sensitivity.json", sens)

    summary_text = build_summary_text(
        run_id=output_dir.name,
        split_mode=split_mode,
        N_eval=n_eval,
        evm_real=evm_real,
        evm_pred=evm_pred,
        snr_real=snr_real,
        snr_pred=snr_pred,
        distm=distm,
        active_dims=active_dims,
        kl_qp_total_mean=kl_qp_total_mean,
        kl_pN_total_mean=kl_pN_total_mean,
        sens_var_mean=sens["decoder_output_variance_mean"],
        sens_rms=sens["decoder_output_rms_std"],
        arch_variant=arch_variant,
    )
    dashboard_path = save_champion_analysis_dashboard(
        plots_dir=run_paths.plots_dir,
        Xv=Xv_center,
        Yv=Yv,
        Yp=Yp,
        std_mu_p=z_std_p,
        kl_dim_mean=df_lat["kl_p_to_N0I_dim_mean"].values,
        summary_lines=summary_text.splitlines(),
        model_label="cVAE",
        title=f"Champion Analysis Dashboard | {output_dir.name}",
    )

    print("\n✅ Análise concluída.")
    print(f"📌 Figuras em: {run_paths.plots_dir}")
    print(f"📌 Dashboard: {dashboard_path}")
    print(f"📌 Tabelas em: {run_paths.tables_dir}")
    print(f"📌 Logs em: {run_paths.logs_dir}")

    gc.collect()

    return {
        "status": "completed",
        "run_dir": str(output_dir),
        "model_run_dir": str(model_run_dir),
        "metrics_path": str(metrics_json),
        "metrics": global_metrics,
        "n_eval": int(n_eval),
    }
