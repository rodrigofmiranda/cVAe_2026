# -*- coding: utf-8 -*-
"""
Eval do modelo final (exp_20260318_234955) sem retreinar.

Carrega o modelo salvo, avalia no regime 1m/300mA, roda MMD+Energy
com n_perm=2000 e gera summary_by_regime.csv com todos os gates.

Uso:
    cd /workspace/2026
    python scripts/eval_final_model.py 2>&1 | tee outputs/eval_final_model.log
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, "/workspace/2026")
os.chdir("/workspace/2026")

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
MODEL_RUN_DIR  = Path("outputs/exp_20260318_204149/global_model")
DATASET_ROOT   = Path("data/dataset_fullsquare_organized")
REGIME_DIST_M  = 1.0
REGIME_CURR_MA = 300.0
MC_SAMPLES     = 16
N_PERM         = 2000
STAT_N         = 5_000
SEED           = 42
# ---------------------------------------------------------------------------

print("Carregando deps...")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from src.models.cvae_sequence import load_seq_model
from src.models.cvae import create_inference_model_from_full
from src.data.loading import find_dataset_root, load_experiments_as_list
from src.data.normalization import apply_condition_norm
from src.protocol.split_strategies import apply_split
from src.data.windowing import build_windows_single_experiment
from src.evaluation.stat_tests.mmd import mmd_rbf
from src.evaluation.stat_tests.energy import energy_test
from src.evaluation.metrics import calculate_evm, calculate_snr

# ---------------------------------------------------------------------------
# Load model + state
# ---------------------------------------------------------------------------
state_path = MODEL_RUN_DIR / "state_run.json"
with open(state_path) as f:
    state = json.load(f)

norm        = state["normalization"]
training_cfg = state["training_config"]
model_path  = state["artifacts"]["best_model_full"]

print(f"Carregando modelo: {model_path}")
vae = load_seq_model(model_path)
print("  OK")

# ---------------------------------------------------------------------------
# Load + split regime data
# ---------------------------------------------------------------------------
print(f"\nCarregando dados {REGIME_DIST_M}m / {REGIME_CURR_MA}mA...")
resolved_root = find_dataset_root(
    marker_dirname="dataset_fullsquare_organized",
    dataset_root_hint=str(DATASET_ROOT),
    verbose=False,
)
exps, _ = load_experiments_as_list(resolved_root, verbose=False, reduction_config=None)
exps_regime = [
    (X, Y, D, C, p) for X, Y, D, C, p in exps
    if abs(float(np.unique(D)[0]) - REGIME_DIST_M) <= 0.05
    and abs(float(np.unique(C)[0]) - REGIME_CURR_MA) <= 25.0
]
print(f"  Experimentos: {len(exps_regime)}")

X_tr, Y_tr, D_tr, C_tr, X_va, Y_va, D_va, C_va, df_split = apply_split(
    exps=exps_regime,
    strategy=str(training_cfg.get("split_mode", "per_experiment")),
    val_split=float(training_cfg.get("validation_split", 0.2)),
    seed=int(training_cfg.get("seed", 42)),
    within_exp_shuffle=bool(training_cfg.get("within_experiment_shuffle", False)),
)
print(f"  val={len(X_va):,}")

Dn_va = apply_condition_norm(D_va.ravel(), C_va.ravel(), norm)[0].reshape(-1, 1)
Cn_va = apply_condition_norm(D_va.ravel(), C_va.ravel(), norm)[1].reshape(-1, 1)

# ---------------------------------------------------------------------------
# Build windows
# ---------------------------------------------------------------------------
prior = vae.get_layer("prior_net")
ws = int(prior.inputs[0].shape[1])
print(f"  Window size: {ws}")
n_val_list = [int(v) for v in df_split["n_val"].tolist()]
va_X_w, cursor = [], 0
for n_va in n_val_list:
    if n_va > 0:
        Xw, _, _, _ = build_windows_single_experiment(
            X_va[cursor:cursor + n_va],
            np.zeros((n_va, 2), dtype=np.float32),
            np.ones((n_va, 1), dtype=np.float32),
            np.ones((n_va, 1), dtype=np.float32),
            window_size=ws, stride=1, pad_mode="edge",
        )
        va_X_w.append(Xw)
    cursor += n_va
X_va_w = np.concatenate(va_X_w, axis=0)

# ---------------------------------------------------------------------------
# MC stochastic prediction
# ---------------------------------------------------------------------------
print(f"\nInferência estocástica: mc_samples={MC_SAMPLES}...")
inf_sto = create_inference_model_from_full(vae, deterministic=False)
samples = []
for i in range(MC_SAMPLES):
    tf.random.set_seed(SEED + i)
    s = inf_sto.predict([X_va_w, Dn_va, Cn_va], batch_size=4096, verbose=0)
    samples.append(s)
    print(f"  {i+1}/{MC_SAMPLES}", end="\r")
print()

# ---------------------------------------------------------------------------
# EVM / SNR from individual MC draws (avoids ensemble-mean bias)
# ---------------------------------------------------------------------------
print("Calculando EVM/SNR por MC draw...")
Ys_mc = np.stack(samples, axis=0)  # (MC_SAMPLES, N_val, 2)
evm_real_live, _ = calculate_evm(X_va, Y_va)
snr_real_live     = calculate_snr(X_va, Y_va)
evm_pred_live = float(np.mean([calculate_evm(X_va, Ys_mc[i])[0]
                                for i in range(MC_SAMPLES)]))
snr_pred_live = float(np.mean([calculate_snr(X_va, Ys_mc[i])
                                for i in range(MC_SAMPLES)]))
print(f"  evm_real={evm_real_live:.4f}%  evm_pred={evm_pred_live:.4f}%  "
      f"delta_evm={evm_pred_live - evm_real_live:+.4f}%")
print(f"  snr_real={snr_real_live:.4f}dB  snr_pred={snr_pred_live:.4f}dB  "
      f"delta_snr={snr_pred_live - snr_real_live:+.4f}dB")

Y_pred_all = np.concatenate(samples, axis=0)
X_tiled = np.tile(X_va, (MC_SAMPLES, 1))
Y_tiled = np.tile(Y_va, (MC_SAMPLES, 1))
res_real_all = Y_tiled - X_tiled
res_pred_all = Y_pred_all - X_tiled

# ---------------------------------------------------------------------------
# Sub-sample
# ---------------------------------------------------------------------------
rng = np.random.RandomState(SEED)
n = min(STAT_N, res_real_all.shape[0], res_pred_all.shape[0])
idx_r = rng.choice(res_real_all.shape[0], n, replace=False)
idx_g = rng.choice(res_pred_all.shape[0], n, replace=False)
res_real = res_real_all[idx_r]
res_pred = res_pred_all[idx_g]
print(f"  n={n:,} amostras para stat tests")

# ---------------------------------------------------------------------------
# Stat tests
# ---------------------------------------------------------------------------
print(f"\nMMD (n_perm={N_PERM})...")
mmd_res    = mmd_rbf(res_real, res_pred, n_perm=N_PERM, seed=SEED)
print(f"Energy (n_perm={N_PERM})...")
energy_res = energy_test(res_real, res_pred, n_perm=N_PERM, seed=SEED)

# ---------------------------------------------------------------------------
# Gate computation (TWIN_GATE_THRESHOLDS — linter update)
# ---------------------------------------------------------------------------
import math, csv

metrics_path = (
    MODEL_RUN_DIR.parent
    / "studies/within_regime/regimes/dist_1m__curr_300mA"
    / "tables/metricas_globais_reanalysis.csv"
)
with open(metrics_path) as f:
    m = next(csv.DictReader(f))

def flt(k): return float(m[k]) if m.get(k) and m[k] not in ('', 'nan') else float('nan')

evm_real   = flt("evm_real_%")
evm_pred   = flt("evm_pred_%")
delta_evm  = flt("delta_evm_%")
snr_real   = flt("snr_real_db")
delta_snr  = flt("delta_snr_db")
var_real   = flt("var_real_delta")
delta_mean = flt("delta_mean_l2")
delta_cov  = flt("delta_cov_fro")
delta_skew = flt("delta_skew_l2")
delta_kurt = flt("delta_kurt_l2")
delta_psd  = flt("delta_psd_l2")
jb_log10p  = flt("jb_log10p_min")
jb_real_l  = flt("jb_real_log10p_min")

# Override EVM/SNR: live computation per MC draw supersedes stale CSV values.
# CSV stores EVM from ensemble mean → cancels noise → artificially low EVM.
evm_real  = evm_real_live
evm_pred  = evm_pred_live
delta_evm = evm_pred_live - evm_real_live
snr_real  = snr_real_live
delta_snr = snr_pred_live - snr_real_live

# Baseline values (fixed for this regime — from exp_20260318_182809)
bl_delta_cov  = 0.011495
bl_delta_kurt = 0.7322
bl_delta_mean = 0.000992
bl_psd_l2     = 0.2238

sigma_real = math.sqrt(var_real) if var_real > 0 else float('nan')

# Derived
rel_evm_err  = abs(delta_evm)  / abs(evm_real)
rel_snr_err  = abs(delta_snr)  / abs(snr_real)
mean_rel_sig = delta_mean      / sigma_real
cov_rel_var  = delta_cov       / var_real
jb_rel       = abs(jb_log10p - jb_real_l) / abs(jb_real_l)

T = dict(rel_evm_error=0.10, rel_snr_error=0.10, mean_rel_sigma=0.10,
         cov_rel_var=0.20, delta_psd_l2=0.25, delta_skew_l2=0.30,
         delta_kurt_l2=1.25, delta_jb_stat_rel=0.20, stat_qval=0.05)

mmd_pval = mmd_res['pval']
energy_pval = energy_res['pval']

gate_g1 = rel_evm_err  < T["rel_evm_error"]
gate_g2 = rel_snr_err  < T["rel_snr_error"]
gate_g3 = (mean_rel_sig < T["mean_rel_sigma"]) and (cov_rel_var < T["cov_rel_var"])
gate_g4 = delta_psd    < T["delta_psd_l2"]
gate_g5 = (delta_skew  < T["delta_skew_l2"]) and (delta_kurt < T["delta_kurt_l2"]) and (jb_rel < T["delta_jb_stat_rel"])
gate_g6 = (mmd_pval    > T["stat_qval"]) and (energy_pval > T["stat_qval"])

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def yn(b): return "✅ PASS" if b else "❌ FAIL"

print()
print("=" * 66)
print(f"RESULTADO FINAL — exp_20260318_204149 (λ_mmd=1.0, β=0.001)")
print(f"Regime: {REGIME_DIST_M}m / {REGIME_CURR_MA}mA | mc={MC_SAMPLES} | n_perm={N_PERM} | n={n:,}")
print("=" * 66)
print(f"G1  rel_evm_error  = {rel_evm_err:.4f}  (thr<0.10)  {yn(gate_g1)}")
print(f"G2  rel_snr_error  = {rel_snr_err:.4f}  (thr<0.10)  {yn(gate_g2)}")
print(f"G3  mean_rel_sigma = {mean_rel_sig:.4f}  (thr<0.10)  ", end="")
print(f"cov_rel_var={cov_rel_var:.4f}  (thr<0.20)  {yn(gate_g3)}")
print(f"G4  delta_psd_l2   = {delta_psd:.4f}  (thr<0.25)  {yn(gate_g4)}")
print(f"G5  delta_skew_l2  = {delta_skew:.4f}  (thr<0.30)  ", end="")
print(f"delta_kurt_l2={delta_kurt:.4f}  (thr<1.25)  jb_rel={jb_rel:.4f}  (thr<0.20)  {yn(gate_g5)}")
print(f"G6  MMD p={mmd_pval:.4f}  Energy p={energy_pval:.4f}  (thr>0.05 ambos)  {yn(gate_g6)}")
print("=" * 66)
passed = sum([gate_g1, gate_g2, gate_g3, gate_g4, gate_g5, gate_g6])
print(f"Gates: {passed}/6 passed")
print()

# Save results
out_dir = MODEL_RUN_DIR.parent / "tables"
out_dir.mkdir(parents=True, exist_ok=True)
result = {
    "exp": "exp_20260318_204149",
    "regime": "dist_1m__curr_300mA",
    "mc_samples": MC_SAMPLES,
    "n_perm": N_PERM,
    "n_stat": n,
    "rel_evm_error": rel_evm_err,
    "rel_snr_error": rel_snr_err,
    "mean_rel_sigma": mean_rel_sig,
    "cov_rel_var": cov_rel_var,
    "delta_psd_l2": delta_psd,
    "delta_skew_l2": delta_skew,
    "delta_kurt_l2": delta_kurt,
    "delta_jb_stat_rel": jb_rel,
    "stat_mmd2": mmd_res['mmd2'],
    "stat_mmd_pval": mmd_pval,
    "stat_energy": energy_res['energy'],
    "stat_energy_pval": energy_pval,
    "gate_g1": gate_g1, "gate_g2": gate_g2, "gate_g3": gate_g3,
    "gate_g4": gate_g4, "gate_g5": gate_g5, "gate_g6": gate_g6,
    "validation_status": "pass" if passed == 6 else f"partial ({passed}/6)",
}
out_path = out_dir / "eval_final_gates.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Salvo em: {out_path}")
