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
from src.data.loading import (                          # Commits 3A–3B
    ensure_iq_shape, read_metadata, parse_dist_curr_from_path,
    discover_experiments, load_experiments_as_list,
)
from src.data.splits import split_train_val_per_experiment  # Commit 3D
from src.evaluation.metrics import (                    # Commit 3E
    calculate_evm, calculate_snr,
    _skew_kurt, _psd_log, residual_distribution_metrics,
)
from src.models.cvae_components import (                # Commit 3F
    Sampling, CondPriorVAELoss,
)

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
PLOTS_DIR = RUN_DIR / "plots"
TABLES_DIR = RUN_DIR / "tables"
MODELS_DIR = RUN_DIR / "models"
LOGS_DIR = RUN_DIR / "logs"
for d in [PLOTS_DIR, TABLES_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
            "deterministic_inference": True,
            "mc_samples": 1,
            "rank_mode": "det",
            "n_eval_samples": 40000,
            "batch_infer": 8192,
            "eval_slice": "val_head",
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
    y_log_var = layers.Lambda(lambda t: tf.clip_by_value(t, -6.0, 1.0), name="clip_y_log_var")(y_log_var)

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

# ==========================================================
# 6) Load dataset + split CONSISTENTE com treino (per_experiment)
# ==========================================================
exps, df_info = load_experiments_as_list(DATASET_ROOT, verbose=True)
print(f"✅ Experimentos carregados: {(df_info['status']=='ok').sum()}")
df_info.to_excel(TABLES_DIR / "dataset_inventory.xlsx", index=False)
print(f"✓ dataset_inventory.xlsx salvo: {TABLES_DIR / 'dataset_inventory.xlsx'}")

# decide split_mode
data_split = state.get("data_split", {}) if isinstance(state.get("data_split", {}), dict) else {}
split_mode = str(data_split.get("split_mode", state.get("training_config", {}).get("split_mode", "global"))).lower()

val_split = float(data_split.get("validation_split", state.get("training_config", {}).get("validation_split", 0.2)))
order_mode = str(data_split.get("per_experiment_split_order", state.get("training_config", {}).get("per_experiment_split_order", "head_tail")))
within_shuffle = bool(data_split.get("within_experiment_shuffle", state.get("training_config", {}).get("within_experiment_shuffle", False)))

if split_mode == "per_experiment":
    X_train, Y_train, D_train, C_train, X_val, Y_val, D_val, C_val, df_split = split_train_val_per_experiment(
        exps, val_split=val_split, seed=seed0, order_mode=order_mode, within_exp_shuffle=within_shuffle
    )
    df_split.to_excel(TABLES_DIR / "split_by_experiment.xlsx", index=False)
    print(f"✓ split_by_experiment.xlsx salvo: {TABLES_DIR / 'split_by_experiment.xlsx'}")
else:
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
if eval_slice != "val_head":
    eval_slice = "val_head"

N = min(N_eval, len(X_val))
Xv = X_val[:N]
Yv = Y_val[:N]
Dv = Dn_val[:N]
Cv = Cn_val[:N]

# Inferência: determinística (média) ou MC
if det_inf or mc_samples <= 1 or rank_mode == "det":
    Yp = inference_model.predict([Xv, Dv, Cv], batch_size=bs_inf, verbose=0)
    var_mc = float("nan")
else:
    inf_sto = create_inference_model_from_full(prior, decoder, deterministic=False)
    Ys = []
    for _ in range(int(mc_samples)):
        Ys.append(inf_sto.predict([Xv, Dv, Cv], batch_size=bs_inf, verbose=0))
    Ys = np.stack(Ys, axis=0)
    Yp = Ys.mean(axis=0)
    var_mc = float(np.mean(np.var(Ys, axis=0)))

# ==========================================================
# 7) Métricas globais + salvar JSON/CSV
# ==========================================================
evmi_real, _ = calculate_evm(Xv, Yv)
evmi_pred, _ = calculate_evm(Xv, Yp)
snr_real = calculate_snr(Xv, Yv)
snr_pred = calculate_snr(Xv, Yp)

analysis_quick = state.get("analysis_quick", {}) if isinstance(state.get("analysis_quick", {}), dict) else {}
dist_on = bool(analysis_quick.get("dist_metrics", True))
psd_nfft = int(analysis_quick.get("psd_nfft", 2048))

if dist_on:
    distm = residual_distribution_metrics(Xv, Yv, Yp, psd_nfft=psd_nfft)
else:
    distm = {k: float("nan") for k in ["delta_mean_l2", "delta_cov_fro", "var_real_delta", "var_pred_delta",
                                       "delta_skew_l2", "delta_kurt_l2", "delta_psd_l2"]}

global_metrics = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "run_id": state.get("run_id", RUN_DIR.name),
    "model_path": str(best_model_path),
    "split_mode": split_mode,
    "N_eval": int(N),
    "evm_real_%": float(evmi_real),
    "evm_pred_%": float(evmi_pred),
    "delta_evm_%": float(evmi_pred - evmi_real),
    "snr_real_db": float(snr_real),
    "snr_pred_db": float(snr_pred),
    "delta_snr_db": float(snr_pred - snr_real),
    **{k: float(v) for k, v in distm.items()},
    "deterministic_inference": bool(det_inf),
    "rank_mode": str(rank_mode),
    "mc_samples": int(mc_samples),
    "var_mc_gen": (float(var_mc) if not np.isnan(var_mc) else float("nan")),
}

(LOGS_DIR / "metricas_globais_reanalysis.json").write_text(json.dumps(global_metrics, indent=2), encoding="utf-8")
print(f"✓ metricas_globais_reanalysis.json salvo: {LOGS_DIR / 'metricas_globais_reanalysis.json'}")
pd.DataFrame([global_metrics]).to_csv(TABLES_DIR / "metricas_globais_reanalysis.csv", index=False)

# ==========================================================
# 8) Diagnósticos do latente
# ==========================================================
enc_out = encoder.predict([Xv, Dv, Cv, Yv], batch_size=bs_inf, verbose=0)
pri_out = prior.predict([Xv, Dv, Cv], batch_size=bs_inf, verbose=0)
z_mean_q, z_log_var_q = _first2(enc_out)
z_mean_p, z_log_var_p = _first2(pri_out)

z_std_p = np.std(z_mean_p, axis=0)
active_dims = int(np.sum(z_std_p > 0.05))

vq = np.exp(np.clip(z_log_var_q, -20, 20))
vp = np.exp(np.clip(z_log_var_p, -20, 20))

kl_qp_dim = 0.5 * (
    np.log(vp + 1e-12)
    - np.log(vq + 1e-12)
    + (vq + (z_mean_q - z_mean_p) ** 2) / (vp + 1e-12)
    - 1.0
)
kl_qp_dim_mean = np.mean(kl_qp_dim, axis=0)
kl_qp_total_mean = float(np.mean(np.sum(kl_qp_dim, axis=1)))

lv_p_clip = np.clip(z_log_var_p, -20, 20)
kl_pN_dim = 0.5 * (np.exp(lv_p_clip) + z_mean_p ** 2 - 1.0 - lv_p_clip)
kl_pN_dim_mean = np.mean(kl_pN_dim, axis=0)
kl_pN_total_mean = float(np.mean(np.sum(kl_pN_dim, axis=1)))

df_lat = pd.DataFrame({
    "dim": np.arange(z_std_p.shape[0]),
    "std_mu_p": z_std_p.astype(float),
    "kl_q_to_p_dim_mean": kl_qp_dim_mean.astype(float),
    "kl_p_to_N0I_dim_mean": kl_pN_dim_mean.astype(float),
})
df_lat.to_excel(TABLES_DIR / "latent_diagnostics.xlsx", index=False)
print(f"✓ latent_diagnostics.xlsx salvo: {TABLES_DIR / 'latent_diagnostics.xlsx'}")

lat_summary = {
    "active_dims_std_mu_p_gt_0p05": int(active_dims),
    "kl_q_to_p_total_mean": float(kl_qp_total_mean),
    "kl_p_to_N0I_total_mean": float(kl_pN_total_mean),
}
(LOGS_DIR / "latent_summary.json").write_text(json.dumps(lat_summary, indent=2), encoding="utf-8")

# ==========================================================
# 9) Sensibilidade do decoder ao z (teste de colapso)
# ==========================================================
def decoder_sensitivity(prior_net, decoder_net, Xb, Db, Cb, n_mc_z=16):
    mu_p, lv_p = prior_net.predict([Xb, Db, Cb], batch_size=bs_inf, verbose=0)
    lv_p = np.clip(lv_p, -10, 10)
    std_p = np.exp(0.5 * lv_p)

    cond = np.concatenate([Xb, Db, Cb], axis=1)

    outs = []
    for _ in range(int(n_mc_z)):
        eps = np.random.randn(*mu_p.shape).astype(np.float32)
        z = mu_p + std_p * eps
        out_params = decoder_net.predict([z, cond], batch_size=bs_inf, verbose=0)
        y_mean = out_params[:, :2]
        outs.append(y_mean)

    outs = np.stack(outs, axis=0)  # [K,N,2]
    v = np.var(outs, axis=0)       # [N,2]
    return float(np.mean(v)), float(np.mean(np.sqrt(np.sum(v, axis=1))))

Nb = min(20000, N)
sens_var_mean, sens_rms = decoder_sensitivity(prior, decoder, Xv[:Nb], Dv[:Nb], Cv[:Nb], n_mc_z=16)

sens = {"decoder_output_variance_mean": float(sens_var_mean), "decoder_output_rms_std": float(sens_rms)}
(LOGS_DIR / "decoder_sensitivity.json").write_text(json.dumps(sens, indent=2), encoding="utf-8")

# ==========================================================
# 10) Plots
# ==========================================================
def _savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

# 10.1 Loss curves (training_history.json)
train_hist_path = None
cand = state.get("artifacts", {}).get("training_history_json", "")
if cand:
    train_hist_path = Path(cand)
else:
    train_hist_path = _autofind(LOGS_DIR, ["training_history.json"])

if train_hist_path is not None and Path(train_hist_path).exists():
    try:
        hist = json.loads(Path(train_hist_path).read_text(encoding="utf-8"))

        # aceita payload do treino (com "history") ou dict keras puro
        if isinstance(hist, dict) and "history" in hist and isinstance(hist["history"], dict):
            dfh = pd.DataFrame(hist["history"])
        elif isinstance(hist, dict) and "loss" in hist:
            dfh = pd.DataFrame(hist)
        elif isinstance(hist, list):
            dfh = pd.DataFrame(hist)
        else:
            dfh = None

        if dfh is not None and len(dfh) > 0:
            dfh.to_excel(TABLES_DIR / "training_history.xlsx", index=False)

            plt.figure()
            for col in ["loss", "val_loss", "recon_loss", "val_recon_loss", "kl_loss", "val_kl_loss"]:
                if col in dfh.columns:
                    plt.plot(dfh[col].values, label=col)
            plt.xlabel("epoch")
            plt.ylabel("value")
            plt.title("Training history")
            plt.legend()
            _savefig(PLOTS_DIR / "training_history.png")
    except Exception as e:
        print(f"⚠ Falha ao ler/plotar training_history: {e}")

# 10.2 Constellation overlay (Y real vs Y pred)
Ns = min(80000, N)
Xps = Xv[:Ns]
Yrs = Yv[:Ns]
Yps = Yp[:Ns]

plt.figure()
plt.scatter(Yrs[:, 0], Yrs[:, 1], s=2, alpha=0.35, label="Y real")
plt.scatter(Yps[:, 0], Yps[:, 1], s=2, alpha=0.35, label="Y pred")
plt.xlabel("I")
plt.ylabel("Q")
plt.title("Constellation overlay: Y real vs Y pred")
plt.legend(markerscale=4)
_savefig(PLOTS_DIR / "overlay_constellation.png")

# 10.3 Residual Δ overlay
Dr = (Yrs - Xps)
Dp = (Yps - Xps)
plt.figure()
plt.scatter(Dr[:, 0], Dr[:, 1], s=2, alpha=0.35, label="Δ real = Y-X")
plt.scatter(Dp[:, 0], Dp[:, 1], s=2, alpha=0.35, label="Δ pred = Ŷ-X")
plt.xlabel("ΔI")
plt.ylabel("ΔQ")
plt.title("Residual constellation overlay (Δ)")
plt.legend(markerscale=4)
_savefig(PLOTS_DIR / "overlay_residual_delta.png")

# 10.4 Hist2D density comparison
bins = 160
plt.figure()
plt.hist2d(Yrs[:, 0], Yrs[:, 1], bins=bins)
plt.xlabel("I")
plt.ylabel("Q")
plt.title("Density: Y real (hist2d)")
_savefig(PLOTS_DIR / "density_y_real.png")

plt.figure()
plt.hist2d(Yps[:, 0], Yps[:, 1], bins=bins)
plt.xlabel("I")
plt.ylabel("Q")
plt.title("Density: Y pred (hist2d)")
_savefig(PLOTS_DIR / "density_y_pred.png")

# 10.5 PSD residual (log)
cr = Dr[:, 0] + 1j * Dr[:, 1]
cp = Dp[:, 0] + 1j * Dp[:, 1]
psd_r = _psd_log(cr, nfft=psd_nfft)
psd_p = _psd_log(cp, nfft=psd_nfft)

plt.figure()
plt.plot(psd_r, label="Δ real")
plt.plot(psd_p, label="Δ pred")
plt.xlabel("freq bin")
plt.ylabel("log10 PSD")
plt.title("Residual PSD comparison")
plt.legend()
_savefig(PLOTS_DIR / "psd_residual_delta.png")

# 10.6 Latent diagnostics plots
plt.figure()
plt.bar(df_lat["dim"].values, df_lat["std_mu_p"].values)
plt.xlabel("latent dim")
plt.ylabel("std(μ_p)")
plt.title(f"Latent activity (active dims={active_dims})")
_savefig(PLOTS_DIR / "latent_activity_std_mu_p.png")

plt.figure()
plt.plot(df_lat["dim"].values, df_lat["kl_q_to_p_dim_mean"].values, label="KL(q||p)")
plt.plot(df_lat["dim"].values, df_lat["kl_p_to_N0I_dim_mean"].values, label="KL(p||N)")
plt.xlabel("latent dim")
plt.ylabel("KL mean")
plt.title("Latent KL per dimension")
plt.legend()
_savefig(PLOTS_DIR / "latent_kl_per_dim.png")

# 10.7 Summary figure
plt.figure()
plt.axis("off")
text = (
    f"Run: {state.get('run_id', RUN_DIR.name)}\n"
    f"Split mode: {split_mode}\n"
    f"N_eval: {N}\n"
    f"EVM real: {evmi_real:.3f}% | EVM pred: {evmi_pred:.3f}% | ΔEVM: {evmi_pred-evmi_real:+.3f} p.p.\n"
    f"SNR real: {snr_real:.3f} dB | SNR pred: {snr_pred:.3f} dB | ΔSNR: {snr_pred-snr_real:+.3f} dB\n"
    f"Δ mean L2: {distm['delta_mean_l2']:.4g} | Δ cov Fro: {distm['delta_cov_fro']:.4g} | Δ PSD L2: {distm['delta_psd_l2']:.4g}\n"
    f"Latent active dims (std μ_p>0.05): {active_dims}\n"
    f"KL(q||p) total mean: {kl_qp_total_mean:.4g} | KL(p||N) total mean: {kl_pN_total_mean:.4g}\n"
    f"Decoder sensitivity var_mean: {sens_var_mean:.4g} | rms_std: {sens_rms:.4g}\n"
)
plt.text(0.02, 0.98, text, va="top", family="monospace")
_savefig(PLOTS_DIR / "summary_report.png")

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