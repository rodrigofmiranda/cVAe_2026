# -*- coding: utf-8 -*-
"""
Etapa A — Diagnóstico G6 com stat_mode=full.

Carrega o melhor modelo seq salvo (exp_20260318_182809) e o regime
1.0m/300mA, e roda os testes estatísticos com n_perm=2000 e
mc_samples=32 — sem retreinar.

Uso:
    cd /workspace/2026
    python scripts/diag_g6_full.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, "/workspace/2026")
os.chdir("/workspace/2026")

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_RUN_DIR = Path(
    "outputs/exp_20260318_182809/global_model"
)
DATASET_ROOT = Path("data/dataset_fullsquare_organized")
REGIME_DIST_M = 1.0
REGIME_CURR_MA = 300.0
MC_SAMPLES = 32
N_PERM = 2000
STAT_N = 5_000
SEED = 42

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
print("Carregando dependências...")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from src.data.loading import find_dataset_root, load_experiments_as_list
from src.data.normalization import normalize_conditions
from src.protocol.split_strategies import apply_split
from src.models.cvae import create_inference_model_from_full
from src.models.cvae_sequence import load_seq_model
from src.data.windowing import build_windows_single_experiment
from src.evaluation.stat_tests.mmd import mmd_rbf
from src.evaluation.stat_tests.energy import energy_test

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"\nCarregando modelo de {MODEL_RUN_DIR}...")
import json

state_path = MODEL_RUN_DIR / "state_run.json"
with open(state_path) as f:
    state = json.load(f)

model_path = Path(state["artifacts"]["best_model_full"])
print(f"  modelo: {model_path}")
try:
    vae = load_seq_model(str(model_path))
except Exception:
    vae = tf.keras.models.load_model(str(model_path), compile=False)

norm = state["normalization"]
training_cfg = state["training_config"]

# ---------------------------------------------------------------------------
# Load regime data (1.0m / 300mA only)
# ---------------------------------------------------------------------------
print(f"\nCarregando dados do regime {REGIME_DIST_M}m / {REGIME_CURR_MA}mA...")

resolved_root = find_dataset_root(
    marker_dirname="dataset_fullsquare_organized",
    dataset_root_hint=str(DATASET_ROOT),
    verbose=False,
)

exps, _ = load_experiments_as_list(resolved_root, verbose=False, reduction_config=None)

dist_tol = 0.05
curr_tol = 25.0
exps_regime = [
    (X, Y, D, C, p) for X, Y, D, C, p in exps
    if abs(float(np.unique(D)[0]) - REGIME_DIST_M) <= dist_tol
    and abs(float(np.unique(C)[0]) - REGIME_CURR_MA) <= curr_tol
]
print(f"  Experimentos no regime: {len(exps_regime)}")

X_tr, Y_tr, D_tr, C_tr, X_va, Y_va, D_va, C_va, df_split = apply_split(
    exps=exps_regime,
    strategy=str(training_cfg.get("split_mode", "per_experiment")),
    val_split=float(training_cfg.get("validation_split", 0.2)),
    seed=int(training_cfg.get("seed", 42)),
    within_exp_shuffle=bool(training_cfg.get("within_experiment_shuffle", False)),
)
print(f"  train={len(X_tr):,}  val={len(X_va):,}")

# Apply norm from saved model
from src.data.normalization import apply_condition_norm

Dn_va = apply_condition_norm(D_va.ravel(), C_va.ravel(), norm)[0].reshape(-1, 1)
Cn_va = apply_condition_norm(D_va.ravel(), C_va.ravel(), norm)[1].reshape(-1, 1)

# ---------------------------------------------------------------------------
# Build windowed input for seq model
# ---------------------------------------------------------------------------
prior = vae.get_layer("prior_net")
_is_seq = len(prior.inputs[0].shape) == 3

if _is_seq:
    ws = int(prior.inputs[0].shape[1])
    print(f"\n  Arquitetura seq detectada: window_size={ws}")
    n_val_list = [int(v) for v in df_split["n_val"].tolist()]
    va_X_w = []
    cursor = 0
    for n_va in n_val_list:
        if n_va > 0:
            Y_dummy = np.zeros((n_va, 2), dtype=np.float32)
            D_dummy = np.ones((n_va, 1), dtype=np.float32)
            C_dummy = np.ones((n_va, 1), dtype=np.float32)
            Xw, _, _, _ = build_windows_single_experiment(
                X_va[cursor:cursor + n_va], Y_dummy, D_dummy, C_dummy,
                window_size=ws, stride=1, pad_mode="edge",
            )
            va_X_w.append(Xw)
        cursor += n_va
    X_va_w = np.concatenate(va_X_w, axis=0)
    X_in = X_va_w
else:
    X_in = X_va

# ---------------------------------------------------------------------------
# MC stochastic prediction
# ---------------------------------------------------------------------------
print(f"\nInferência estocástica: mc_samples={MC_SAMPLES}...")
batch_infer = 8192

inf_sto = create_inference_model_from_full(vae, deterministic=False)
samples = []
for i in range(MC_SAMPLES):
    tf.random.set_seed(SEED + i)
    s = inf_sto.predict([X_in, Dn_va, Cn_va], batch_size=batch_infer, verbose=0)
    samples.append(s)

Y_pred_all = np.concatenate(samples, axis=0)
X_tiled = np.tile(X_va, (MC_SAMPLES, 1))
Y_tiled = np.tile(Y_va, (MC_SAMPLES, 1))

res_real_all = Y_tiled - X_tiled
res_pred_all = Y_pred_all - X_tiled

print(f"  Pool real:  {res_real_all.shape[0]:,}")
print(f"  Pool pred:  {res_pred_all.shape[0]:,}")

# ---------------------------------------------------------------------------
# Sub-sample to STAT_N
# ---------------------------------------------------------------------------
rng = np.random.RandomState(SEED)
n = min(STAT_N, res_real_all.shape[0], res_pred_all.shape[0])
idx_real = rng.choice(res_real_all.shape[0], n, replace=False)
idx_pred = rng.choice(res_pred_all.shape[0], n, replace=False)
res_real = res_real_all[idx_real]
res_pred = res_pred_all[idx_pred]

print(f"\n  Amostras para teste estatístico: {n:,}")

# ---------------------------------------------------------------------------
# Stat tests
# ---------------------------------------------------------------------------
print(f"\nRodando MMD e Energy (n_perm={N_PERM})...")
mmd_res = mmd_rbf(res_real, res_pred, n_perm=N_PERM, seed=SEED)
energy_res = energy_test(res_real, res_pred, n_perm=N_PERM, seed=SEED)

print()
print("=" * 60)
print("RESULTADO — Etapa A (stat_mode=full)")
print("=" * 60)
print(f"  mc_samples : {MC_SAMPLES}")
print(f"  n_perm     : {N_PERM}")
print(f"  n_samples  : {n:,}")
print()
print(f"  MMD²       : {mmd_res['mmd2']:.6f}")
print(f"  MMD p-val  : {mmd_res['pval']:.4f}")
print(f"  MMD q-val  : {mmd_res.get('qval', mmd_res['pval']):.4f}")
print(f"  G6 passa?  : {'✅ SIM' if mmd_res['pval'] > 0.05 else '❌ NÃO'} (threshold q > 0.05)")
print()
print(f"  Energy     : {energy_res['energy']:.6f}")
print(f"  Energy p   : {energy_res['pval']:.4f}")
print("=" * 60)

# Compare with quick mode reference
print()
print("Referência (quick mode, n_perm=200):")
print(f"  MMD² = 0.000572  |  p = 0.00995  |  q = 0.01493  [❌]")
print()
if mmd_res['pval'] > 0.05:
    print("→ G6 PASSA com avaliação precisa. O modelo já é suficientemente bom.")
elif mmd_res['pval'] > 0.02:
    print("→ G6 ainda não passa, mas p melhorou. Avançar para Etapa B (latent_dim=8).")
else:
    print("→ G6 claramente falha. Avançar para Etapa C (loss MMD auxiliar).")
