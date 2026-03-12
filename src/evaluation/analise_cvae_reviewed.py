# -*- coding: utf-8 -*-
"""
analise_cvae.py (PATCH — compatível com split POR EXPERIMENTO)

Mudanças principais vs versão anterior:
- Remove shuffle global + split global.
- Reconstrói split por experimento (head=train, tail=val) usando config do state_run.json.
- Mantém inferência via prior condicional (p(z|x,d,c)) + decoder.
- Mantém métricas, diagnósticos do latente e plots.

Requisitos:
- state_run.json gerado pelo treino (recomendado), contendo normalization e data_split.
"""

import os
import re
import json
import gc
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from src.config.runtime_env import ensure_writable_mpl_config_dir
from src.config.defaults import (
    DECODER_LOGVAR_CLAMP_HI,
    DECODER_LOGVAR_CLAMP_LO,
)
from src.data.loading import (                          # Commits 3A–3B
    ensure_iq_shape, read_metadata, parse_dist_curr_from_path,
    discover_experiments, load_experiments_as_list,
)
from src.data.splits import (                               # Commit 3D
    split_train_val_per_experiment,
    cap_train_samples_per_experiment,
)
from src.evaluation.metrics import (                    # Commit 3E
    calculate_evm, calculate_snr,
    _skew_kurt, _psd_log, residual_distribution_metrics,
)
from src.models.cvae_components import (                # Commit 3F
    Sampling, CondPriorVAELoss,
)
from src.training.logging import RunPaths                # refactor(core)
from src.evaluation.plots import (                       # refactor(step4)
    plot_overlay, plot_residual_overlay, plot_histograms,
    plot_psd, plot_latent_activity, plot_latent_kl,
    plot_training_history, plot_summary_report,
)
from src.evaluation.report import (                      # refactor(step4)
    build_global_metrics, compute_latent_diagnostics,
    decoder_sensitivity, load_training_history,
    build_summary_text,
)


def main(overrides=None):  # Commit 3H: optional CLI overrides dict
    ensure_writable_mpl_config_dir()
    _ov = overrides or {}

    # ==========================================================
    # 0) BASE PATHS + pick latest run
    # ==========================================================
    OUTPUT_BASE = Path(os.environ.get("OUTPUT_BASE", "/workspace/2026/outputs"))

    def pick_latest_run(outputs_base: Path) -> Path:
        runs = sorted([p for p in outputs_base.glob("run_*") if p.is_dir()], key=lambda p: p.name)
        if not runs:
            raise FileNotFoundError(f"Nenhum diretório run_* encontrado em {outputs_base}")
        return runs[-1]

    RUN_ID_ENV = os.environ.get("RUN_ID", "").strip()
    RUN_DIR = (OUTPUT_BASE / RUN_ID_ENV) if RUN_ID_ENV else pick_latest_run(OUTPUT_BASE)
    if RUN_ID_ENV and not RUN_DIR.exists():
        raise FileNotFoundError(f"RUN_ID={RUN_ID_ENV} não encontrado em {OUTPUT_BASE}")
    _rp = RunPaths.from_existing(RUN_DIR)
    PLOTS_DIR = _rp.plots_dir
    TABLES_DIR = _rp.tables_dir
    MODELS_DIR = _rp.models_dir
    LOGS_DIR = _rp.logs_dir

    print(f"📁 OUTPUT_BASE = {OUTPUT_BASE}")
    print(f"✅ RUN_DIR selecionado: {RUN_DIR}")

    state_path = RUN_DIR / "state_run.json"

    def _autofind(p: Path, patterns):
        for pat in patterns:
            m = list(p.glob(pat))
            if m:
                return m[0]
        return None

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        print("✓ state_run.json carregado.")
    else:
        print("⚠ state_run.json não encontrado — usando fallback (state mínimo).")

        best_model = _autofind(MODELS_DIR, ["best_model_full.keras", "*.keras"])
        if best_model is None:
            raise FileNotFoundError(f"Não encontrei modelo em {MODELS_DIR} (ex.: best_model_full.keras).")

        th = _autofind(LOGS_DIR, ["training_history.json"])

        state = {
            "run_id": RUN_DIR.name,
            "run_dir": str(RUN_DIR),
            "dataset_root": str(Path(os.environ.get("DATASET_ROOT", "/workspace/2026/dataset_fullsquare_organized"))),
            "output_base": str(OUTPUT_BASE),
            "paths": {
                "plots": str(PLOTS_DIR),
                "tables": str(TABLES_DIR),
                "models": str(MODELS_DIR),
                "logs": str(LOGS_DIR),
            },
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
            "analysis_quick": {"dist_metrics": True, "psd_nfft": 2048},
            "artifacts": {
                "best_model_full": str(best_model),
                "training_history_json": str(th) if th is not None else "",
            },
        }

    DATASET_ROOT = Path(state.get("dataset_root", os.environ.get("DATASET_ROOT", "/workspace/2026/dataset_fullsquare_organized")))
    print(f"📁 DATASET_ROOT = {DATASET_ROOT}")

    # ==========================================================
    # 1) DATA LOADER (compat)
    # ==========================================================


    # ensure_iq_shape, read_metadata, parse_dist_curr_from_path
    # -> moved to src.data.loading (Commit 3A)
    # discover_experiments, load_experiments_as_list
    # -> moved to src.data.loading (Commit 3B)

    # split_train_val_per_experiment -> moved to src.data.splits (Commit 3D)

    # calculate_evm, calculate_snr, _skew_kurt, _psd_log,
    # residual_distribution_metrics -> moved to src.evaluation.metrics (Commit 3E)

    # Sampling, CondPriorVAELoss
    # -> moved to src.models.cvae_components (Commit 3F)

    custom_objects = {"Sampling": Sampling, "CondPriorVAELoss": CondPriorVAELoss}

    # ==========================================================
    # 4) LOAD MODEL (best_model_full.keras)
    # ==========================================================
    best_model_path = Path(state["artifacts"]["best_model_full"])
    if not best_model_path.exists():
        raise FileNotFoundError(f"best_model_full.keras não encontrado: {best_model_path}")

    print(f"📦 Carregando modelo: {best_model_path}")
    vae = tf.keras.models.load_model(str(best_model_path), custom_objects=custom_objects, compile=False)

    layer_names = [l.name for l in vae.layers]
    print("🔎 Layers-chave:", {"encoder": "encoder" in layer_names, "prior_net": "prior_net" in layer_names, "decoder": "decoder" in layer_names})
    if ("prior_net" not in layer_names) or ("decoder" not in layer_names) or ("encoder" not in layer_names):
        raise ValueError("Modelo carregado não contém encoder/prior_net/decoder. Confirme o arquivo do best_model_full.")

    encoder = vae.get_layer("encoder")
    prior = vae.get_layer("prior_net")
    decoder = vae.get_layer("decoder")

    # ==========================================================
    # Helper: lidar com possíveis variações de saída do encoder/prior
    # ==========================================================
    def _first2(outputs):
        """
        Alguns modelos podem retornar (z_mean, z_log_var) ou (z_mean, z_log_var, z).
        Esta função normaliza para sempre pegar os 2 primeiros tensores.
        """
        if isinstance(outputs, (list, tuple)):
            if len(outputs) < 2:
                raise ValueError(f"Saída inesperada (len<2): {type(outputs)}")
            return outputs[0], outputs[1]
        raise ValueError(f"Saída inesperada (não é list/tuple): {type(outputs)}")


    # ==========================================================
    # 5) Inference model (prior condicional)
    # ==========================================================
    def create_inference_model_from_full(prior_net, decoder_net, deterministic: bool = True):
        """
        Cria um modelo de inferência que gera ŷ a partir de (x, d, c) usando:
          z ~ p(z|x,d,c)  (prior condicional)
          ŷ ~ p(y|x,d,c,z) (decoder heteroscedástico)

        Nota de implementação:
        - Em Keras Functional, ops TF diretas (ex.: tf.clip_by_value, tf.random.normal)
          sobre KerasTensors podem falhar dependendo da versão. Por isso usamos Lambda/Sampling.
        """
        x_in = layers.Input(shape=(2,), name="x_input")
        d_in = layers.Input(shape=(1,), name="distance_input")
        c_in = layers.Input(shape=(1,), name="current_input")

        z_mean_p, z_log_var_p = prior_net([x_in, d_in, c_in])

        # Clipping do log-var do prior (evita exp overflow/underflow)
        z_log_var_p = layers.Lambda(lambda t: tf.clip_by_value(t, -10.0, 10.0), name="clip_z_log_var_p")(z_log_var_p)

        if deterministic:
            z = z_mean_p
        else:
            # Reusa o Sampling (já registrado como custom object) para manter compatibilidade
            z = Sampling(name="sample_z")([z_mean_p, z_log_var_p])

        cond = layers.Concatenate(name="cond_concat")([x_in, d_in, c_in])
        out_params = decoder_net([z, cond])

        y_mean = layers.Lambda(lambda t: t[:, :2], name="y_mean")(out_params)
        y_log_var = layers.Lambda(lambda t: t[:, 2:], name="y_log_var_raw")(out_params)

        # Clipping do log-var do decoder (controle indireto das caudas)
        y_log_var = layers.Lambda(
            lambda t: tf.clip_by_value(
                t, DECODER_LOGVAR_CLAMP_LO, DECODER_LOGVAR_CLAMP_HI
            ),
            name="clip_y_log_var",
        )(y_log_var)

        if deterministic:
            y = y_mean
        else:
            y = Sampling(name="sample_y")([y_mean, y_log_var])

        return tf.keras.Model([x_in, d_in, c_in], y, name=("infer_det" if deterministic else "infer_mc"))

    seed0 = int(state.get("training_config", {}).get("seed", 42))
    np.random.seed(seed0)
    tf.random.set_seed(seed0)

    evalp = state.get("eval_protocol", {}) if isinstance(state.get("eval_protocol", {}), dict) else {}
    det_inf = bool(evalp.get("deterministic_inference", True))
    mc_samples = int(evalp.get("mc_samples", 1))
    rank_mode = str(evalp.get("rank_mode", ("det" if det_inf else "mc"))).lower()

    inference_model = create_inference_model_from_full(prior, decoder, deterministic=det_inf)
    print("✅ inference_model pronto:", inference_model.name, "| deterministic =", det_inf, "| mc_samples =", mc_samples, "| rank_mode =", rank_mode)
    # --- Commit 3H: dry_run --- stop after loading model + building inference graph ---
    if _ov.get("dry_run", False):
        import json as _json
        _dry_info = {
            "dry_run": True,
            "overrides": {k: v for k, v in _ov.items()},
            "inference_model": inference_model.name,
            "deterministic": det_inf,
            "model_path": str(best_model_path),
        }
        _rp.write_json("logs/dry_run_eval.json", _dry_info)
        print(f"\u2705 dry_run complete \u2014 wrote {LOGS_DIR / 'dry_run_eval.json'}")
        return
    # ==========================================================
    # 6) Load dataset + split CONSISTENTE com treino (per_experiment)
    # ==========================================================
    exps, df_info = load_experiments_as_list(DATASET_ROOT, verbose=True)

    # --- Commit 3S: filter to selected experiments from protocol runner ---
    _sel_exps = _ov.get("_selected_experiments")
    if _sel_exps:
        _sel_set = set(str(p) for p in _sel_exps)
        _before = len(exps)
        exps = [(X, Y, D, C, p) for X, Y, D, C, p in exps if str(p) in _sel_set]
        print(f"\u26a1 Commit 3S: filtered {_before} \u2192 {len(exps)} experiment(s) "
              f"(selected_experiments)")

    # --- Commit 3H: limit experiments / samples if requested ---
    if "max_experiments" in _ov:
        _me = int(_ov["max_experiments"])
        exps = exps[:_me]
        print(f"\u26a1 Commit 3H: limited to {len(exps)} experiment(s)")
    print(f"✅ Experimentos carregados: {(df_info['status']=='ok').sum()}")
    _rp.write_table("tables/dataset_inventory.xlsx", df_info)
    print(f"✓ dataset_inventory.xlsx salvo: {TABLES_DIR / 'dataset_inventory.xlsx'}")

    # decide split_mode
    data_split = state.get("data_split", {}) if isinstance(state.get("data_split", {}), dict) else {}
    split_mode = str(data_split.get("split_mode", state.get("training_config", {}).get("split_mode", "global"))).lower()

    val_split = float(data_split.get("validation_split", state.get("training_config", {}).get("validation_split", 0.2)))
    order_mode = str(data_split.get("per_experiment_split_order", state.get("training_config", {}).get("per_experiment_split_order", "head_tail")))
    within_shuffle = bool(data_split.get("within_experiment_shuffle", state.get("training_config", {}).get("within_experiment_shuffle", False)))
    df_split = None

    if split_mode == "per_experiment":
        X_train, Y_train, D_train, C_train, X_val, Y_val, D_val, C_val, df_split = split_train_val_per_experiment(
            exps, val_split=val_split, seed=seed0, order_mode=order_mode, within_exp_shuffle=within_shuffle
        )
        if "max_samples_per_exp" in _ov:
            _ms = int(_ov["max_samples_per_exp"])
            X_train, Y_train, D_train, C_train, _df_cap = cap_train_samples_per_experiment(
                X_train, Y_train, D_train, C_train, df_split, _ms
            )
            print(
                f"⚡ Commit 3H: max_samples_per_exp pós-split "
                f"(train cap={_ms}/exp) | train={len(X_train):,} | val={len(X_val):,}"
            )
        _rp.write_table("tables/split_by_experiment.xlsx", df_split)
        print(f"✓ split_by_experiment.xlsx salvo: {TABLES_DIR / 'split_by_experiment.xlsx'}")
    else:
        if "max_samples_per_exp" in _ov:
            print("⚠️  Ignorando max_samples_per_exp no split_mode='global' "
                  "(cap por experimento requer split per_experiment).")
        # fallback (não recomendado): concatena e faz split global
        X = np.concatenate([e[0] for e in exps], axis=0)
        Y = np.concatenate([e[1] for e in exps], axis=0)
        D = np.concatenate([e[2] for e in exps], axis=0)
        C = np.concatenate([e[3] for e in exps], axis=0)

        idx = np.arange(len(X))
        np.random.default_rng(seed0).shuffle(idx)
        X = X[idx]; Y = Y[idx]; D = D[idx]; C = C[idx]
        split = int((1 - val_split) * len(X))
        X_train, Y_train, D_train, C_train = X[:split], Y[:split], D[:split], C[:split]
        X_val, Y_val, D_val, C_val = X[split:], Y[split:], D[split:], C[split:]

    print(f"✓ Split aplicado | train={len(X_train):,} | val={len(X_val):,} | mode={split_mode}")

    # Normalização (preferir a do state — calculada no treino)
    norm = state.get("normalization", None)
    if isinstance(norm, dict) and all(k in norm for k in ["D_min", "D_max", "C_min", "C_max"]):
        D_min, D_max = float(norm["D_min"]), float(norm["D_max"])
        C_min, C_max = float(norm["C_min"]), float(norm["C_max"])
    else:
        # fallback: calcula no TREINO para não vazar
        D_min, D_max = float(D_train.min()), float(D_train.max())
        C_min, C_max = float(C_train.min()), float(C_train.max())
        print(f"⚠ Normalização calculada no treino (fallback): D=[{D_min:.3f},{D_max:.3f}] | C=[{C_min:.1f},{C_max:.1f}]")

    Dn_val = (D_val - D_min) / (D_max - D_min) if D_max > D_min else np.zeros_like(D_val)
    Cn_val = (C_val - C_min) / (C_max - C_min) if C_max > C_min else np.full_like(C_val, 0.5)

    # protocolo de avaliação
    N_eval = int(evalp.get("n_eval_samples", 40_000))
    bs_inf = int(evalp.get("batch_infer", 8192))
    eval_slice = str(evalp.get("eval_slice", "val_head"))
    # Backward-compat: previous runs may store "stratified_by_regime".
    if eval_slice == "stratified_by_regime":
        eval_slice = "stratified"
    # Aceita "val_head" (padrão legado), "stratified" (Fix 4) ou "full".
    _valid_slices = {"val_head", "stratified", "full"}
    if eval_slice not in _valid_slices:
        print(f"⚠️  eval_slice='{eval_slice}' inválido, usando 'val_head'")
        eval_slice = "val_head"

    if eval_slice == "stratified":
        # Estratificação explícita por experimento (quando split_by_experiment está disponível).
        # Se metadados não existirem/inconsistirem, faz fallback para amostragem uniforme global.
        N = min(N_eval, len(X_val))
        rng_eval = np.random.default_rng(seed0)
        idx_eval = None

        if (
            split_mode == "per_experiment"
            and isinstance(df_split, pd.DataFrame)
            and "n_val" in df_split.columns
        ):
            n_val_list = [int(v) for v in df_split["n_val"].tolist()]
            if n_val_list and int(np.sum(n_val_list)) == int(len(X_val)):
                n_exps = len(n_val_list)
                base = N // n_exps
                rem = N % n_exps

                target = np.full(n_exps, base, dtype=np.int64)
                if rem > 0:
                    rem_idx = rng_eval.permutation(n_exps)[:rem]
                    target[rem_idx] += 1

                cap = np.asarray(n_val_list, dtype=np.int64)
                take = np.minimum(target, cap)
                avail = cap - take
                left = int(N - int(take.sum()))

                # Redistribui sobras para experimentos com capacidade restante.
                while left > 0 and int(avail.sum()) > 0:
                    progressed = False
                    for i in rng_eval.permutation(n_exps):
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
                            local = np.sort(rng_eval.choice(n_i, size=k_i, replace=False))
                        else:
                            local = np.arange(n_i, dtype=np.int64)
                        idx_parts.append(cursor + local)
                    cursor += n_i

                if idx_parts:
                    idx_eval = np.concatenate(idx_parts, axis=0)
                    idx_eval.sort()
                    print(
                        f"✓ eval_slice=stratified (por experimento) | "
                        f"N={len(idx_eval):,} | n_experiments={n_exps}"
                    )

        if idx_eval is None:
            idx_eval = rng_eval.choice(len(X_val), size=N, replace=False)
            print("⚠️  eval_slice=stratified sem df_split válido; fallback para amostragem global uniforme.")
        idx_eval.sort()  # preserva ordem temporal dentro das amostras escolhidas
        Xv, Yv = X_val[idx_eval], Y_val[idx_eval]
        Dv, Cv = Dn_val[idx_eval], Cn_val[idx_eval]
        N = len(idx_eval)
    elif eval_slice == "full":
        N = len(X_val)
        Xv, Yv, Dv, Cv = X_val, Y_val, Dn_val, Cn_val
    else:  # val_head (padrão legado)
        N = min(N_eval, len(X_val))
        Xv, Yv, Dv, Cv = X_val[:N], Y_val[:N], Dn_val[:N], Cn_val[:N]

    # Inferência: pontual (EVM/SNR/plots) + distribuição (MC concat para Nível 2)
    if det_inf or mc_samples <= 1 or rank_mode == "det":
        Yp = inference_model.predict([Xv, Dv, Cv], batch_size=bs_inf, verbose=0)
        Yp_dist = Yp
        X_dist = Xv
        Y_dist = Yv
        var_mc = float("nan")
    else:
        inf_sto = create_inference_model_from_full(prior, decoder, deterministic=False)
        Ys = []
        for _ in range(int(mc_samples)):
            Ys.append(inf_sto.predict([Xv, Dv, Cv], batch_size=bs_inf, verbose=0))
        Ys = np.stack(Ys, axis=0)
        # Point metrics remain on MC mean.
        Yp = Ys.mean(axis=0)
        # Distribution metrics use marginal predictive samples (concatenated MC).
        Yp_dist = Ys.reshape((-1, Ys.shape[-1]))
        X_dist = np.tile(Xv, (int(mc_samples), 1))
        Y_dist = np.tile(Yv, (int(mc_samples), 1))
        var_mc = float(np.mean(np.var(Ys, axis=0)))

    # ==========================================================
    # 7) Métricas globais + salvar JSON/CSV  (refactor step 4: report module)
    # ==========================================================
    evmi_real, _ = calculate_evm(Xv, Yv)
    evmi_pred, _ = calculate_evm(Xv, Yp)
    snr_real = calculate_snr(Xv, Yv)
    snr_pred = calculate_snr(Xv, Yp)

    analysis_quick = state.get("analysis_quick", {}) if isinstance(state.get("analysis_quick", {}), dict) else {}
    dist_on = bool(analysis_quick.get("dist_metrics", True))
    psd_nfft = int(analysis_quick.get("psd_nfft", 2048))
    # --- Commit 3H: allow CLI override for psd_nfft ---
    if "psd_nfft" in _ov:
        psd_nfft = int(_ov["psd_nfft"])

    if dist_on:
        distm = residual_distribution_metrics(X_dist, Y_dist, Yp_dist, psd_nfft=psd_nfft)
    else:
        distm = {k: float("nan") for k in ["delta_mean_l2", "delta_cov_fro", "var_real_delta", "var_pred_delta",
                                           "delta_skew_l2", "delta_kurt_l2", "delta_psd_l2"]}

    global_metrics = build_global_metrics(
        run_id=state.get("run_id", RUN_DIR.name),
        model_path=str(best_model_path),
        split_mode=split_mode,
        N_eval=int(N),
        evm_real=float(evmi_real),
        evm_pred=float(evmi_pred),
        snr_real=float(snr_real),
        snr_pred=float(snr_pred),
        distm=distm,
        det_inf=bool(det_inf),
        rank_mode=str(rank_mode),
        mc_samples=int(mc_samples),
        var_mc=float(var_mc) if not np.isnan(var_mc) else float("nan"),
    )

    _rp.write_json("logs/metricas_globais_reanalysis.json", global_metrics)
    print(f"✓ metricas_globais_reanalysis.json salvo: {LOGS_DIR / 'metricas_globais_reanalysis.json'}")
    _rp.write_table("tables/metricas_globais_reanalysis.csv", pd.DataFrame([global_metrics]))

    # ==========================================================
    # 8) Diagnósticos do latente (refactor step 4: report module)
    # ==========================================================
    enc_out = encoder.predict([Xv, Dv, Cv, Yv], batch_size=bs_inf, verbose=0)
    pri_out = prior.predict([Xv, Dv, Cv], batch_size=bs_inf, verbose=0)
    z_mean_q, z_log_var_q = _first2(enc_out)
    z_mean_p, z_log_var_p = _first2(pri_out)

    lat_diag = compute_latent_diagnostics(z_mean_q, z_log_var_q, z_mean_p, z_log_var_p)
    df_lat = lat_diag["df_lat"]
    lat_summary = lat_diag["lat_summary"]
    z_std_p = lat_diag["z_std_p"]
    active_dims = lat_diag["active_dims"]
    kl_qp_total_mean = lat_diag["kl_qp_total_mean"]
    kl_pN_total_mean = lat_diag["kl_pN_total_mean"]

    _rp.write_table("tables/latent_diagnostics.xlsx", df_lat)
    print(f"✓ latent_diagnostics.xlsx salvo: {TABLES_DIR / 'latent_diagnostics.xlsx'}")
    _rp.write_json("logs/latent_summary.json", lat_summary)

    # ==========================================================
    # 9) Sensibilidade do decoder ao z (refactor step 4: report module)
    # ==========================================================
    Nb = min(20000, N)
    sens = decoder_sensitivity(prior, decoder, Xv[:Nb], Dv[:Nb], Cv[:Nb],
                               n_mc_z=16, batch_size=bs_inf)
    sens_var_mean = sens["decoder_output_variance_mean"]
    sens_rms = sens["decoder_output_rms_std"]
    _rp.write_json("logs/decoder_sensitivity.json", sens)

    # ==========================================================
    # 10) Plots (refactor step 4: plots module)
    # ==========================================================

    # 10.1 Loss curves
    train_hist_path = None
    cand = state.get("artifacts", {}).get("training_history_json", "")
    if cand:
        train_hist_path = Path(cand)
    else:
        train_hist_path = _autofind(LOGS_DIR, ["training_history.json"])

    if train_hist_path is not None and Path(train_hist_path).exists():
        try:
            dfh = load_training_history(train_hist_path)
            if dfh is not None:
                _rp.write_table("tables/training_history.xlsx", dfh)
                plot_training_history(
                    {c: dfh[c].values.tolist() for c in dfh.columns},
                    PLOTS_DIR / "training_history.png",
                )
        except Exception as e:
            print(f"⚠ Falha ao ler/plotar training_history: {e}")

    # 10.2 Constellation overlay
    plot_overlay(Yv, Yp, PLOTS_DIR / "overlay_constellation.png", max_points=80_000)

    # 10.3 Residual Δ overlay
    plot_residual_overlay(Xv, Yv, Yp, PLOTS_DIR / "overlay_residual_delta.png", max_points=80_000)

    # 10.4 Hist2D density
    plot_histograms(Yv, PLOTS_DIR / "density_y_real.png", title="Density: Y real (hist2d)")
    plot_histograms(Yp, PLOTS_DIR / "density_y_pred.png", title="Density: Y pred (hist2d)")

    # 10.5 PSD residual
    plot_psd(Xv, Yv, Yp, PLOTS_DIR / "psd_residual_delta.png", nfft=psd_nfft)

    # 10.6 Latent diagnostics plots
    plot_latent_activity(z_std_p, PLOTS_DIR / "latent_activity_std_mu_p.png",
                         active_dims=active_dims)
    plot_latent_kl(df_lat["dim"].values, df_lat["kl_q_to_p_dim_mean"].values,
                   df_lat["kl_p_to_N0I_dim_mean"].values,
                   PLOTS_DIR / "latent_kl_per_dim.png")

    # 10.7 Summary figure
    summary_text = build_summary_text(
        run_id=state.get("run_id", RUN_DIR.name),
        split_mode=split_mode, N_eval=N,
        evm_real=evmi_real, evm_pred=evmi_pred,
        snr_real=snr_real, snr_pred=snr_pred,
        distm=distm, active_dims=active_dims,
        kl_qp_total_mean=kl_qp_total_mean,
        kl_pN_total_mean=kl_pN_total_mean,
        sens_var_mean=sens_var_mean, sens_rms=sens_rms,
    )
    plot_summary_report(summary_text, PLOTS_DIR / "summary_report.png")

    print("\n✅ Análise concluída.")
    print(f"📌 Figuras em: {PLOTS_DIR}")
    print(f"📌 Tabelas em: {TABLES_DIR}")
    print(f"📌 Logs em: {LOGS_DIR}")

    # limpeza
    try:
        del exps
    except Exception:
        pass
    gc.collect()

if __name__ == "__main__":
    main()
