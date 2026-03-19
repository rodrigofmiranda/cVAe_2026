# -*- coding: utf-8 -*-
"""
Eval do modelo final (exp_20260318_204149) sem retreinar.

Carrega o modelo salvo, avalia no regime 1m/300mA e computa TODOS os
gates G1-G6 ao vivo — sem ler métricas de CSVs gerados anteriormente.

  G1/G2  EVM/SNR          — média por MC draw individual (evita bias da média de ensemble)
  G3     mean/cov          — live sobre pool mc×N_val (cap 200 K)
  G4     PSD L2            — live, Welch 50% overlap
  G5     skew/kurt/JB      — live, mesmo pool de G3
  G6     MMD + Energy      — live, sub-amostra 5 K (evita OOM Gram matrix)

Notas de design
---------------
* cov_rel_var = ||ΔCov||_F / var_real  (ambos em signal²; threshold 0.20 calibrado
  considerando que ||Cov_real||_F ≈ √2·var_real para ruído I/Q não correlacionado).
* G6 usa p-value bruto (não BH-corrigido). Com 1 único regime o ajuste
  Benjamini-Hochberg é a identidade (q = p), portanto equivale ao pipeline
  canônico de validation_summary.py.

Uso:
    cd /workspace/2026
    nohup python -u scripts/eval_final_model.py > outputs/eval_final_model.log 2>&1 &
"""
from __future__ import annotations
import sys, os, json, math
sys.path.insert(0, "/workspace/2026")
os.chdir("/workspace/2026")

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
MODEL_RUN_DIR    = Path("outputs/exp_20260318_204149/global_model")
DATASET_ROOT     = Path("data/dataset_fullsquare_organized")
REGIME_DIST_M    = 1.0
REGIME_CURR_MA   = 300.0
MC_SAMPLES       = 16
N_PERM           = 2000
STAT_N           = 5_000     # n for MMD/Energy (Gram matrix is O(n²), keep small)
MAX_DIST_SAMPLES = 200_000   # n for G3-G5 distribution metrics
SEED             = 42
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
from src.metrics.distribution import residual_fidelity_metrics

# ---------------------------------------------------------------------------
# Load model + state
# ---------------------------------------------------------------------------
state_path = MODEL_RUN_DIR / "state_run.json"
with open(state_path) as f:
    state = json.load(f)

norm         = state["normalization"]
training_cfg = state["training_config"]
model_path   = state["artifacts"]["best_model_full"]

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
# G1/G2 — EVM/SNR: per-draw mean (avoids ensemble-mean bias)
#
# Using the ensemble mean Ys.mean(axis=0) cancels stochastic noise and
# produces artificially low EVM, making the twin appear "cleaner" than the
# real channel.  Per-draw mean gives the correct expected EVM per realisation.
# ---------------------------------------------------------------------------
print("Calculando EVM/SNR por MC draw...")
Ys_mc = np.stack(samples, axis=0)   # (MC_SAMPLES, N_val, 2)
evm_real, _  = calculate_evm(X_va, Y_va)
snr_real      = calculate_snr(X_va, Y_va)
evm_pred      = float(np.mean([calculate_evm(X_va, Ys_mc[i])[0] for i in range(MC_SAMPLES)]))
snr_pred      = float(np.mean([calculate_snr(X_va, Ys_mc[i])    for i in range(MC_SAMPLES)]))
delta_evm     = evm_pred - evm_real
delta_snr     = snr_pred - snr_real
print(f"  evm_real={evm_real:.4f}%  evm_pred={evm_pred:.4f}%  delta={delta_evm:+.4f}%")
print(f"  snr_real={snr_real:.4f}dB  snr_pred={snr_pred:.4f}dB  delta={delta_snr:+.4f}dB")

# ---------------------------------------------------------------------------
# Build residual pools for G3-G6
# ---------------------------------------------------------------------------
Y_pred_all  = np.concatenate(samples, axis=0)           # (MC*N_val, 2)
X_tiled     = np.tile(X_va, (MC_SAMPLES, 1))
Y_tiled     = np.tile(Y_va, (MC_SAMPLES, 1))
res_real_all = Y_tiled - X_tiled                        # real channel residuals
res_pred_all = Y_pred_all - X_tiled                     # predicted residuals

# ---------------------------------------------------------------------------
# G3-G5 — distribution metrics (live, same residual pool, cap MAX_DIST_SAMPLES)
#
# cov_rel_var = ||ΔCov||_F / var_real
#   Both numerator and denominator are in signal² units (dimensionless ratio).
#   For uncorrelated I/Q noise (var_I ≈ var_Q, small cross-covariance),
#   ||Cov_real||_F ≈ √2·var_real, so this formula returns values ~√2 larger
#   than the relative Frobenius error of the covariance matrix.
#   Threshold 0.20 is calibrated accordingly.
# ---------------------------------------------------------------------------
print(f"\nMétricas de distribuição (n≤{MAX_DIST_SAMPLES:,})...")
rng_dist = np.random.RandomState(SEED + 100)
n_dist   = min(MAX_DIST_SAMPLES, res_real_all.shape[0], res_pred_all.shape[0])
idx_dr   = np.sort(rng_dist.choice(res_real_all.shape[0], n_dist, replace=False))
idx_dg   = np.sort(rng_dist.choice(res_pred_all.shape[0], n_dist, replace=False))
distm = residual_fidelity_metrics(
    res_real_all[idx_dr], res_pred_all[idx_dg],
    psd_nfft=2048, gauss_alpha=0.01, max_samples=n_dist,
    X=X_tiled[idx_dr],
)
var_real   = float(np.mean(np.var(res_real_all[idx_dr], axis=0)))
delta_mean = distm["delta_mean_l2"]
delta_cov  = distm["delta_cov_fro"]
delta_skew = distm["delta_skew_l2"]
delta_kurt = distm["delta_kurt_l2"]
delta_psd  = distm["psd_l2"]
delta_acf       = distm.get("delta_acf_l2", float("nan"))
rho_het_real    = distm.get("rho_hetero_real", float("nan"))
rho_het_pred    = distm.get("rho_hetero_pred", float("nan"))
stat_jsd        = distm.get("stat_jsd", float("nan"))
jb_log10p       = distm["jb_log10p_min"]       # predicted residuals
jb_real_l       = distm["jb_real_log10p_min"]  # real residuals
print(f"  mean_l2={delta_mean:.6f}  cov_fro={delta_cov:.6f}  var_real={var_real:.6f}")
print(f"  skew_l2={delta_skew:.4f}  kurt_l2={delta_kurt:.4f}  psd_l2={delta_psd:.4f}  acf_l2={delta_acf:.4f}")
print(f"  rho_hetero_real={rho_het_real:.4f}  rho_hetero_pred={rho_het_pred:.4f}  stat_jsd={stat_jsd:.6f}")
print(f"  jb_log10p_pred={jb_log10p:.2f}  jb_log10p_real={jb_real_l:.2f}  n_dist={n_dist:,}")

# ---------------------------------------------------------------------------
# G6 — MMD + Energy (sub-sample to STAT_N to avoid O(n²) Gram matrix OOM)
#
# Note: with 1 regime the Benjamini-Hochberg correction is the identity
# (q_val = p_val), so raw p-values are equivalent to the canonical
# q-value-based gate in validation_summary.py.
# ---------------------------------------------------------------------------
rng_stat = np.random.RandomState(SEED)
n_stat   = min(STAT_N, res_real_all.shape[0], res_pred_all.shape[0])
idx_r    = rng_stat.choice(res_real_all.shape[0], n_stat, replace=False)
idx_g    = rng_stat.choice(res_pred_all.shape[0], n_stat, replace=False)
res_real = res_real_all[idx_r]
res_pred = res_pred_all[idx_g]
print(f"\n  n={n_stat:,} amostras para stat tests")

print(f"MMD (n_perm={N_PERM})...")
mmd_res    = mmd_rbf(res_real, res_pred, n_perm=N_PERM, seed=SEED)
print(f"Energy (n_perm={N_PERM})...")
energy_res = energy_test(res_real, res_pred, n_perm=N_PERM, seed=SEED)

# ---------------------------------------------------------------------------
# Gate computation
# ---------------------------------------------------------------------------
sigma_real   = math.sqrt(var_real) if var_real > 0 else float('nan')
rel_evm_err  = abs(delta_evm)  / abs(evm_real)
rel_snr_err  = abs(delta_snr)  / abs(snr_real)
mean_rel_sig = delta_mean      / sigma_real
cov_rel_var  = delta_cov       / var_real
jb_rel       = abs(jb_log10p - jb_real_l) / abs(jb_real_l)

mmd_pval    = mmd_res['pval']
energy_pval = energy_res['pval']

T = dict(rel_evm_error=0.10, rel_snr_error=0.10, mean_rel_sigma=0.10,
         cov_rel_var=0.20, delta_psd_l2=0.25, delta_skew_l2=0.30,
         delta_kurt_l2=1.25, delta_jb_stat_rel=0.20, stat_qval=0.05)

gate_g1 = rel_evm_err  < T["rel_evm_error"]
gate_g2 = rel_snr_err  < T["rel_snr_error"]
gate_g3 = (mean_rel_sig < T["mean_rel_sigma"]) and (cov_rel_var < T["cov_rel_var"])
gate_g4 = delta_psd    < T["delta_psd_l2"]
gate_g5 = (delta_skew  < T["delta_skew_l2"]) and (delta_kurt < T["delta_kurt_l2"]) and (jb_rel < T["delta_jb_stat_rel"])
gate_g6 = (mmd_pval    > T["stat_qval"]) and (energy_pval > T["stat_qval"])

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def yn(b): return "PASS" if b else "FAIL"

print()
print("=" * 70)
print(f"RESULTADO FINAL — exp_20260318_204149 (λ_mmd=1.0, β=0.001)")
print(f"Regime: {REGIME_DIST_M}m / {REGIME_CURR_MA}mA | mc={MC_SAMPLES} | n_perm={N_PERM} | n_stat={n_stat:,} | n_dist={n_dist:,}")
print("=" * 70)
print(f"G1  rel_evm_error  = {rel_evm_err:.4f}  (thr<0.10)  {yn(gate_g1)}")
print(f"G2  rel_snr_error  = {rel_snr_err:.4f}  (thr<0.10)  {yn(gate_g2)}")
print(f"G3  mean_rel_sigma = {mean_rel_sig:.4f}  (thr<0.10)  "
      f"cov_rel_var={cov_rel_var:.4f}  (thr<0.20)  {yn(gate_g3)}")
print(f"G4  delta_psd_l2   = {delta_psd:.4f}  (thr<0.25)  {yn(gate_g4)}")
print(f"G5  delta_skew_l2  = {delta_skew:.4f}  (thr<0.30)  "
      f"delta_kurt_l2={delta_kurt:.4f}  (thr<1.25)  "
      f"jb_rel={jb_rel:.4f}  (thr<0.20)  {yn(gate_g5)}")
print(f"G6  MMD p={mmd_pval:.4f}  Energy p={energy_pval:.4f}  (thr>0.05 ambos)  {yn(gate_g6)}")
print("=" * 70)
passed = sum([gate_g1, gate_g2, gate_g3, gate_g4, gate_g5, gate_g6])
print(f"Gates: {passed}/6 passed")
print()

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
out_dir = MODEL_RUN_DIR.parent / "tables"
out_dir.mkdir(parents=True, exist_ok=True)
result = {
    "exp": "exp_20260318_204149",
    "regime": "dist_1m__curr_300mA",
    "mc_samples": MC_SAMPLES,
    "n_perm": N_PERM,
    "n_stat": n_stat,
    "n_dist": n_dist,
    # G1/G2
    "evm_real_%": evm_real,
    "evm_pred_%": evm_pred,
    "delta_evm_%": delta_evm,
    "snr_real_db": snr_real,
    "snr_pred_db": snr_pred,
    "delta_snr_db": delta_snr,
    "rel_evm_error": rel_evm_err,
    "rel_snr_error": rel_snr_err,
    # G3
    "delta_mean_l2": delta_mean,
    "delta_cov_fro": delta_cov,
    "var_real": var_real,
    "mean_rel_sigma": mean_rel_sig,
    "cov_rel_var": cov_rel_var,
    # G4
    "delta_psd_l2": delta_psd,
    "delta_acf_l2": delta_acf,
    # Heteroscedasticity + JSD (reported only, no gate)
    "rho_hetero_real": rho_het_real,
    "rho_hetero_pred": rho_het_pred,
    "stat_jsd": stat_jsd,
    # G5
    "delta_skew_l2": delta_skew,
    "delta_kurt_l2": delta_kurt,
    "jb_log10p_pred": jb_log10p,
    "jb_log10p_real": jb_real_l,
    "delta_jb_stat_rel": jb_rel,
    # G6
    "stat_mmd2": mmd_res['mmd2'],
    "stat_mmd_pval": mmd_pval,
    "stat_energy": energy_res['energy'],
    "stat_energy_pval": energy_pval,
    # Gates
    "gate_g1": gate_g1, "gate_g2": gate_g2, "gate_g3": gate_g3,
    "gate_g4": gate_g4, "gate_g5": gate_g5, "gate_g6": gate_g6,
    "validation_status": "pass" if passed == 6 else f"partial ({passed}/6)",
}
out_path = out_dir / "eval_final_gates.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Salvo em: {out_path}")
