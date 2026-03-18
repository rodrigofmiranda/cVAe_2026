# -*- coding: utf-8 -*-
"""
src.config.defaults — Centralised default values and key names.

Every default that the training monolith, protocol runner, or evaluation
pipeline may reference lives here.  Modules should import from this file
rather than re-defining magic numbers.

Commit: refactor(step1).
"""

from __future__ import annotations

# =====================================================================
# Key-name constants (avoid typos in string literals)
# =====================================================================

# --- Training ---
K_EPOCHS = "epochs"
K_PATIENCE = "patience"
K_REDUCE_LR_PATIENCE = "reduce_lr_patience"
K_VALIDATION_SPLIT = "validation_split"
K_SPLIT_MODE = "split_mode"
K_SPLIT_ORDER = "per_experiment_split_order"
K_WITHIN_EXP_SHUFFLE = "within_experiment_shuffle"
K_SHUFFLE_TRAIN_BATCHES = "shuffle_train_batches"
K_EARLY_STOP_WARMUP = "early_stop_warmup"
K_SEED = "seed"

# --- Model hyper-parameters ---
K_LAYER_SIZES = "layer_sizes"
K_LATENT_DIM = "latent_dim"
K_BETA = "beta"
K_FREE_BITS = "free_bits"
K_LR = "lr"
K_BATCH_SIZE = "batch_size"
K_KL_ANNEAL_EPOCHS = "kl_anneal_epochs"
K_DROPOUT = "dropout"
K_ACTIVATION = "activation"
K_ARCH_VARIANT = "arch_variant"

# --- Sequence model (seq_bigru_residual) ---
K_WINDOW_SIZE = "window_size"
K_WINDOW_STRIDE = "window_stride"
K_WINDOW_PAD_MODE = "window_pad_mode"
K_SEQ_HIDDEN_SIZE = "seq_hidden_size"
K_SEQ_NUM_LAYERS = "seq_num_layers"
K_SEQ_BIDIRECTIONAL = "seq_bidirectional"

# --- Data reduction ---
K_TARGET_SAMPLES = "target_samples_per_experiment"
K_MIN_SAMPLES = "min_samples_per_experiment"
K_REDUCTION_MODE = "mode"
K_BLOCK_LEN = "block_len"
K_TIME_SPREAD = "time_spread"
K_MIN_GAP_BLOCKS = "min_gap_blocks"
K_MAX_SAMPLES = "max_samples_per_experiment"

# --- Analysis ---
K_N_EVAL_SAMPLES = "n_eval_samples"
K_BATCH_INFER = "batch_infer"
K_RANK_MODE = "rank_mode"
K_MC_SAMPLES = "mc_samples"
K_DIST_METRICS = "dist_metrics"
K_PSD_NFFT = "psd_nfft"
K_W_PSD = "w_psd"
K_W_SKEW = "w_skew"
K_W_KURT = "w_kurt"

# --- State-run / paths ---
K_RUN_ID = "run_id"
K_RUN_DIR = "run_dir"
K_DATASET_ROOT = "dataset_root"
K_OUTPUT_BASE = "output_base"
K_NORMALIZATION = "normalization"
K_DATA_SPLIT = "data_split"
K_TRAINING_CONFIG = "training_config"
K_ARTIFACTS = "artifacts"

# =====================================================================
# Default values  (mirror the monolith's TRAINING_CONFIG & friends)
# =====================================================================

TRAINING_DEFAULTS: dict = {
    K_EPOCHS: 500,
    K_PATIENCE: 60,
    K_REDUCE_LR_PATIENCE: 40,
    K_VALIDATION_SPLIT: 0.2,
    K_SPLIT_MODE: "per_experiment",
    K_SPLIT_ORDER: "head_tail",
    K_WITHIN_EXP_SHUFFLE: False,
    K_SHUFFLE_TRAIN_BATCHES: True,  # True: entrelaça amostras de todos os 27 regimes por epoch (evita viés de gradiente)
    K_EARLY_STOP_WARMUP: 0,
    K_SEED: 42,
}

MODEL_DEFAULTS: dict = {
    K_LAYER_SIZES: [128, 256, 512],
    K_LATENT_DIM: 4,
    K_BETA: 0.003,
    K_FREE_BITS: 0.10,
    K_LR: 3e-4,
    K_BATCH_SIZE: 16384,
    K_KL_ANNEAL_EPOCHS: 80,
    K_DROPOUT: 0.0,
    K_ACTIVATION: "leaky_relu",
    K_ARCH_VARIANT: "concat",
    # Sequence model defaults (ignored by point-wise variants)
    K_WINDOW_SIZE: 33,
    K_WINDOW_STRIDE: 1,
    K_WINDOW_PAD_MODE: "edge",
    K_SEQ_HIDDEN_SIZE: 64,
    K_SEQ_NUM_LAYERS: 1,
    K_SEQ_BIDIRECTIONAL: True,
}

DATA_REDUCTION_DEFAULTS: dict = {
    K_TARGET_SAMPLES: 200_000,
    K_MIN_SAMPLES: 80_000,
    K_REDUCTION_MODE: "balanced_blocks",
    K_BLOCK_LEN: 4096,
    K_TIME_SPREAD: True,
    K_MIN_GAP_BLOCKS: 2,
    K_MAX_SAMPLES: 200_000,
    K_SEED: 42,
}

ANALYSIS_DEFAULTS: dict = {
    K_N_EVAL_SAMPLES: 40_000,
    K_BATCH_INFER: 8192,
    K_RANK_MODE: "mc",
    K_MC_SAMPLES: 8,
    K_DIST_METRICS: True,
    K_PSD_NFFT: 2048,
    K_W_PSD: 0.15,
    K_W_SKEW: 0.05,
    K_W_KURT: 0.05,
}

# Minimal fallback state_run for backward-compat with old runs
FALLBACK_STATE_RUN: dict = {
    K_TRAINING_CONFIG: {K_SEED: 42, K_VALIDATION_SPLIT: 0.2},
    K_NORMALIZATION: None,
    K_DATA_SPLIT: {
        K_SPLIT_MODE: "per_experiment",
        K_SPLIT_ORDER: "head_tail",
        K_WITHIN_EXP_SHUFFLE: False,
        K_VALIDATION_SPLIT: 0.2,
        K_SEED: 42,
    },
}

# =====================================================================
# Canonical scalar aliases (used across modules to avoid magic numbers)
# =====================================================================
SPLIT_MODE = TRAINING_DEFAULTS[K_SPLIT_MODE]
PER_EXPERIMENT_SPLIT_ORDER = TRAINING_DEFAULTS[K_SPLIT_ORDER]
VALIDATION_SPLIT = TRAINING_DEFAULTS[K_VALIDATION_SPLIT]
WITHIN_EXPERIMENT_SHUFFLE = TRAINING_DEFAULTS[K_WITHIN_EXP_SHUFFLE]
SHUFFLE_TRAIN_BATCHES = TRAINING_DEFAULTS[K_SHUFFLE_TRAIN_BATCHES]
REDUCTION_TARGET = DATA_REDUCTION_DEFAULTS[K_TARGET_SAMPLES]
N_EVAL_SAMPLES_STRATIFIED = ANALYSIS_DEFAULTS[K_N_EVAL_SAMPLES]
SEED = TRAINING_DEFAULTS[K_SEED]

# Decoder log-variance clamp calibrated from dataset residual statistics:
# q1%(log(var_real_delta)) - 1 nat / q99%(log(var_real_delta)) + 1 nat.
DECODER_LOGVAR_CLAMP_LO = -5.82
DECODER_LOGVAR_CLAMP_HI = -0.69
