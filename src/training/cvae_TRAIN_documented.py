# -*- coding: utf-8 -*-
"""
cvae_TRAIN.py — Digital Twin de Canal VLC via cVAE Heteroscedástico com Prior Condicional
=======================================================================================

Objetivo científico
-------------------
Implementar e avaliar um modelo gerativo condicional (cVAE) que aprenda a transformação estatística do canal VLC
a partir de dados experimentais sincronizados, aproximando:

    p(y | x, d, c)

onde:
- x ∈ R^2: amostra I/Q transmitida (baseband complexo, representada como [I,Q])
- y ∈ R^2: amostra I/Q recebida (após LED + canal óptico + fotodiodo + cadeia de RF/AFE + sincronização)
- d ∈ R: distância (m)
- c ∈ R: corrente de polarização/drive do LED (mA)

A hipótese operacional é que, para um conjunto de regimes (d,c) medidos, existe uma transformação estocástica
que pode ser aprendida diretamente por ML, sem depender de um modelo analítico detalhado (Lambertian+multipath+shot noise),
mas preservando a fidelidade estatística suficiente para:
(i) replicar o canal (gêmeo digital) e
(ii) treinar/avaliar esquemas end-to-end (por exemplo, autoencoder TX/RX) "through the twin".

Escopo experimental (cadeia real em alto nível)
-----------------------------------------------
A geração e aquisição do dataset acontecem fora deste script (GNU Radio / scripts de aquisição), mas este código
assume um fluxo típico:

1) Geração do sinal em baseband:
   - constelação (ex.: 16-QAM) e/ou excitação densa do plano IQ ("full-square") para cobrir suporte amplo.
2) Transmissão:
   - USRP (DAC) → bias-T (soma AC + bias DC) → driver/LED (não linearidades AM/AM, AM/PM, clipping).
3) Propagação:
   - canal óptico indoor/LOS+NLOS (atenuação, reflexões/multipercurso) + ruído (shot/ambiente, eletrônica).
4) Recepção:
   - fotodiodo + TIA/AFE → USRP (ADC).
5) Pós-processamento para criar pares (x[n], y[n]):
   - sincronização (atraso + fase) por correlação/correção de CFO residual (se aplicável),
   - normalizações (pico/potência), alinhamento e trimming para mesmo comprimento,
   - salvamento em .npy (float32, shape (N,2)).

Formato esperado do dataset (por experimento)
---------------------------------------------
Cada experimento representa um regime fixo (d,c), tipicamente em um diretório com padrão dist_*/curr_*:

    <exp_dir>/IQ_data/sent_data_tuple.npy
    <exp_dir>/IQ_data/received_data_tuple_sync-phase.npy   (ou variações aceitas)
    <exp_dir>/IQ_data/metadata.json  (ou <exp_dir>/metadata.json)

- X = sent_data_tuple.npy: shape (N,2), float32, colunas [I,Q]
- Y = received_data_tuple_*.npy: shape (N,2), float32, colunas [I,Q]
- d (m) e c (mA) são inferidos do path (dist_*/curr_*) ou do metadata.json.

Nota crítica: "label" aqui significa alinhamento amostra-a-amostra, não classe de símbolo.
O modelo aprende a transformação contínua do IQ.

Motivação do modelo (cVAE com prior condicional)
------------------------------------------------
Modelos determinísticos (regressão X→Y) tendem a subestimar variância e falhar em caudas/outliers.
Por isso usamos um modelo generativo:

Encoder:
    q_φ(z | x, d, c, y)

Prior condicional:
    p_ψ(z | x, d, c)

Decoder heteroscedástico (Gaussiano diagonal em I/Q):
    p_θ(y | x, d, c, z) = N( μ_θ(x,d,c,z), diag(σ^2_θ(x,d,c,z)) )

Loss (ELBO com annealing):
    L = E_q[ -log p_θ(y|x,d,c,z) ] + β · KL( q_φ(z|x,d,c,y) || p_ψ(z|x,d,c) )

Detalhes de estabilidade:
- Heteroscedasticidade: decoder prevê log σ²; clipping do log-variance evita explosões e caudas artificiais.
- β-annealing: β cresce lentamente (kl_anneal_epochs) para permitir aprendizado do mapeamento antes de forte regularização.
- Free-bits (opcional): evita posterior collapse (KL→0, z ignorado).

Split e vazamento (leakage)
---------------------------
Split padrão: "per_experiment" com "head_tail".
- Cada experimento (regime d,c) é dividido em treino/val mantendo contiguidade temporal.
- Por padrão, não embaralhamos intra-experimento para evitar leakage temporal.

Atenção: center-crop altera a distribuição temporal; use apenas para triagem.

Métricas e ranking
------------------
- Métricas físicas: EVM(%) e SNR(dB).
- Métricas do residual Δ=(Y-X): média/covariância, variância, skew/kurtosis (caudas), PSD.
- Diagnóstico latente: dims ativas (std(μ_p)), KL por dimensão, sensibilidade a z.
- score_v2 combina os itens acima para ranquear candidatos.

Reprodutibilidade
-----------------
- Cada execução cria OUTPUT_BASE/run_YYYYmmdd_HHMMSS com subpastas {models,logs,plots,tables}.
- state_run.json registra configs.
- Planilhas e plots documentam o run.

Referências (sugestões)
-----------------------
- Kingma & Welling, "Auto-Encoding Variational Bayes", 2014.
- Revisões/surveys de modelagem de canal VLC e medições OWC (2016–2025).
- Modelagem data-driven de canal VLC e canais desconhecidos via modelos generativos (2020+).

Última atualização: 2026-03-01

Digital Twin Channel Modeling Constraints (MUST READ)
-----------------------------------------------------
These invariants MUST hold for every architectural change:

1. **Decoder inputs = [x, d, c, z] only.**
   The received signal y MUST NEVER be fed to the decoder under any
   circumstances.  If y reaches the decoder, the model degenerates to
   an identity map and is useless at inference (where y is unknown).

2. **Encoder inputs = [x, d, c, y].**
   y enters ONLY the encoder (approximate posterior), which is
   discarded at inference and replaced by the conditional prior.

3. **Prior inputs = [x, d, c].**
   The prior network must predict z from observable conditions alone.

4. **Split = per-experiment head_tail.**
   No global shuffle.  Temporal order within each (d,c) regime is
   preserved to avoid leakage.

5. **Heteroscedastic decoder.**
   The decoder outputs both μ and log σ² (diagonal Gaussian).
   Log-variance is clipped to prevent numerical instability.

See also: docs/MODELING_ASSUMPTIONS.md for formal derivations.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import os
import re
import json
import time
import gc
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from src.training.logging import bootstrap_run          # Commit 2
from src.data.loading import (                          # Commits 3A–3B
    ensure_iq_shape, read_metadata, parse_dist_curr_from_path,
    discover_experiments, is_valid_dataset_root, find_dataset_root,
    reduce_experiment_xy, load_experiments_as_list,
)
from src.data.splits import split_train_val_per_experiment  # Commit 3D
from src.data.normalization import (                    # refactor(step2)
    normalize_conditions, compute_condition_norm_params,
    apply_condition_norm,
)
from src.evaluation.metrics import (                    # Commit 3E
    calculate_evm, calculate_snr,
    _skew_kurt, _psd_log, residual_distribution_metrics,
)
from src.models.cvae import (                           # refactor(step3)
    build_cvae,
    create_inference_model_from_full,
)
from src.models.callbacks import build_callbacks         # refactor(step3)
from src.models.sampling import Sampling                 # refactor(step3)
from src.models.losses import CondPriorVAELoss           # refactor(step3)
from src.training.gridsearch import (                    # refactor(step4)
    run_gridsearch, checklist_table,
)


def main(overrides=None):  # Commit 3H: optional CLI overrides dict
    _ov = overrides or {}

    # ==========================================================
    # 0) PATHS + RUN
    # ==========================================================
    DATASET_ROOT_ENV = os.environ.get("DATASET_ROOT", "/workspace/2026/dataset_fullsquare_organized")
    DATASET_ROOT = Path(DATASET_ROOT_ENV)

    OUTPUT_BASE = Path(os.environ.get("OUTPUT_BASE", "/workspace/2026/outputs"))

    _run = bootstrap_run(output_base=OUTPUT_BASE)
    RUN_ID    = _run.run_id
    RUN_DIR   = _run.run_dir
    PLOTS_DIR = _run.plots_dir
    TABLES_DIR = _run.tables_dir
    MODELS_DIR = _run.models_dir
    LOGS_DIR  = _run.logs_dir

    print(f"📁 DATASET_ROOT (env) = {DATASET_ROOT}")
    print(f"📦 OUTPUT_BASE        = {OUTPUT_BASE}")
    print(f"🏷️  RUN_ID            = {RUN_ID}")
    print(f"📌 RUN_DIR            = {RUN_DIR}")

    # ==========================================================
    # 1) CONFIGS
    # ==========================================================
    DATA_REDUCTION_CONFIG = {
        "enabled": True,

        # Quantas amostras manter por experimento (seus experimentos têm 900k)
        # 200k = redução de 4.5x; mínimo de segurança abaixo do qual não corta
        "target_samples_per_experiment": 200_000,
        "min_samples_per_experiment":     80_000,

        # "balanced_blocks" preserva spread temporal (recomendado)
        # "center_crop"     é mais rápido mas joga fora início/fim
        "mode": "balanced_blocks",

        # Tamanho de cada bloco contíguo selecionado (amostras)
        # 4096 é bom para preservar autocorrelação local do canal VLC
        "block_len": 4096,

        # Quantos blocos selecionar para atingir o target
        # é calculado automaticamente (target // block_len + 1), esse campo é ignorado
        "n_blocks": 10,          # legado — não usado na nova implementação

        # Distribui os blocos ao longo de TODO o experimento (não só no início)
        "time_spread": True,

        # Jitter máximo (em blocos) para evitar amostragem perfeitamente periódica
        "min_gap_blocks": 2,

        # Campos legados — mantidos para não quebrar nada mas sem efeito
        "bins_r":          10,
        "blocks_per_bin":  5,
        "max_samples_per_experiment": 200_000,

        "seed": 42,
    }


    TRAINING_CONFIG = {
        "epochs": 500,                 # teto maior; quem manda é o early stop
        "patience": 60,                # mais tolerante
        "reduce_lr_patience": 40,      # >> C1: aumentado para não disparar antes do warmup terminar
        "validation_split": 0.2,

        # >>> SPLIT CORRETO <
        "split_mode": "per_experiment",
        "per_experiment_split_order": "head_tail",
        "within_experiment_shuffle": False,

        # IMPORTANTES:
        "shuffle_train_batches": False,
        # >> C1: warmup=0 aqui — cada modelo usa seu próprio kl_anneal_epochs como warmup
        #        (ver callbacks no loop do grid)
        "early_stop_warmup": 0,

        "seed": 42,
    }

    # --- Commit 3H: apply CLI overrides to configs ---
    if "val_split" in _ov:
        TRAINING_CONFIG["validation_split"] = float(_ov["val_split"])
    if "seed" in _ov:
        TRAINING_CONFIG["seed"] = int(_ov["seed"])
    if "max_epochs" in _ov:
        TRAINING_CONFIG["epochs"] = int(_ov["max_epochs"])

    ANALYSIS_QUICK = {
        "n_eval_samples": 40_000,
        "batch_infer": 8192,
        "rank_mode": "mc",
        "mc_samples": 8,
        "dist_metrics": True,
        "psd_nfft": 2048,
        "w_psd": 0.15,
        "w_skew": 0.05,
        "w_kurt": 0.05,
    }
    # ==========================================================
    # 1.1) GRID (ENXUTO ~48 runs) — SUBSTITUIR NO CÓDIGO ATUAL
    # ==========================================================
    GRID = []

    def _cfg(**kwargs):
        cfg = dict(
            activation="leaky_relu",
            kl_anneal_epochs=80,   # um pouco mais longo que 60
            batch_size=16384,
            lr=3e-4,
            dropout=0.0,
            free_bits=0.10,        # leve, para evitar "morrer" KL e ficar determinístico
        )
        cfg.update(kwargs)
        return cfg

    def _tag_beta(beta: float) -> str:
        s = f"{beta:.6f}".rstrip("0").rstrip(".")
        return s.replace(".", "p")

    def _tag_lr(lr: float) -> str:
        s = f"{lr:.7f}".rstrip("0").rstrip(".")
        return s.replace(".", "p")

    def _tag_layers(ls):
        return "-".join(str(x) for x in ls)

    # --------------------------------
    # G0: Referências (2 runs)
    # --------------------------------
    GRID += [
        dict(group="G0_ref",
             tag=f"G0_lat4_b{_tag_beta(0.003)}_fb0p10_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
             cfg=_cfg(layer_sizes=[128,256,512], latent_dim=4, beta=0.003)),
        dict(group="G0_ref",
             tag=f"G0_lat4_b{_tag_beta(0.001)}_fb0p10_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
             cfg=_cfg(layer_sizes=[128,256,512], latent_dim=4, beta=0.001)),
    ]

    # --------------------------------
    # G1: Núcleo principal (24 runs)
    # - latent_dim: 4/6/8
    # - beta: 0.001/0.002/0.003
    # - dropout: 0.0/0.05
    # free_bits fixo 0.10 (bom compromisso)
    # --------------------------------
    for ld in [4, 6, 8]:
        for beta in [0.001, 0.002, 0.003]:
            for do in [0.0, 0.05]:
                GRID.append(dict(
                    group="G1_core",
                    tag=f"G1_lat{ld}_b{_tag_beta(beta)}_fb0p10_do{str(do).replace('.','p')}_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
                    cfg=_cfg(layer_sizes=[128,256,512], latent_dim=ld, beta=beta, dropout=do)
                ))

    # --------------------------------
    # G2: Free-bits sweep (12 runs)
    # - foca no "falta ruído": variar free_bits muda a pressão de KL sem mexer na estrutura
    # --------------------------------
    for ld in [4, 6]:
        for beta in [0.001, 0.002]:
            for fb in [0.0, 0.05, 0.20]:
                GRID.append(dict(
                    group="G2_freebits",
                    tag=f"G2_lat{ld}_b{_tag_beta(beta)}_fb{str(fb).replace('.','p')}_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
                    cfg=_cfg(layer_sizes=[128,256,512], latent_dim=ld, beta=beta, free_bits=fb)
                ))

    # --------------------------------
    # G3: Otimizador/batch/anneal (10 runs)
    # - lr menor + batch menor às vezes ajuda a não criar cauda estranha
    # --------------------------------
    for ld, beta, lr, bs, ae in [
        (6, 0.002, 2e-4, 16384,  80),
        (6, 0.002, 2e-4,  8192,  80),
        (6, 0.002, 3e-4,  8192,  80),
        (6, 0.001, 2e-4, 16384, 100),
        (6, 0.001, 3e-4, 16384, 100),
        (8, 0.002, 2e-4, 16384, 100),
        (8, 0.002, 3e-4,  8192, 100),
        (4, 0.002, 2e-4, 16384,  60),
        (4, 0.001, 2e-4,  8192,  60),
        (8, 0.003, 2e-4, 16384,  80),
    ]:
        GRID.append(dict(
            group="G3_opt",
            tag=f"G3_lat{ld}_b{_tag_beta(beta)}_fb0p10_lr{_tag_lr(lr)}_bs{bs}_anneal{ae}_L{_tag_layers([128,256,512])}",
            cfg=_cfg(layer_sizes=[128,256,512], latent_dim=ld, beta=beta, lr=lr, batch_size=bs, kl_anneal_epochs=ae)
        ))

    # dedup por segurança
    seen = set()
    GRID2 = []
    for g in GRID:
        if g["tag"] in seen:
            continue
        seen.add(g["tag"])
        GRID2.append(g)
    GRID = GRID2

    print(f"📊 GRID TOTAL (enxuto) = {len(GRID)} runs")

    # --- Commit 3I: grid selection controls (group/tag filter + max_grids) ---
    _grid_orig = len(GRID)
    if _ov.get("grid_group") is not None:
        GRID = [g for g in GRID if re.search(_ov["grid_group"], g.get("group", ""))]
    if _ov.get("grid_tag") is not None:
        GRID = [g for g in GRID if re.search(_ov["grid_tag"], g.get("tag", ""))]
    if _ov.get("max_grids") is not None:
        GRID = GRID[:max(1, int(_ov["max_grids"]))]
    if len(GRID) != _grid_orig:
        _tags_preview = ", ".join(g["tag"] for g in GRID[:5])
        if len(GRID) > 5:
            _tags_preview += f" … (+{len(GRID) - 5} more)"
        print(f"⚡ Commit 3I: grid {_grid_orig} → {len(GRID)} | [{_tags_preview}]")

    # ==========================================================
    # Seeds
    # ==========================================================
    np.random.seed(TRAINING_CONFIG["seed"])
    tf.random.set_seed(TRAINING_CONFIG["seed"])

    # ensure_iq_shape, read_metadata, parse_dist_curr_from_path
    # -> moved to src.data.loading (Commit 3A)
    # discover_experiments, is_valid_dataset_root, find_dataset_root,
    # reduce_experiment_xy, load_experiments_as_list
    # -> moved to src.data.loading (Commit 3B)

    # split_train_val_per_experiment -> moved to src.data.splits (Commit 3D)

    # calculate_evm, calculate_snr, _skew_kurt, _psd_log,
    # residual_distribution_metrics -> moved to src.evaluation.metrics (Commit 3E)

    # _activation_layer, Sampling, CondPriorVAELoss, KLAnnealingCallback,
    # build_mlp, build_decoder, build_condprior_cvae,
    # create_inference_model_from_full
    # -> moved to src.models.cvae_components (Commit 3F)
    # -> split into src.models.{sampling,losses,cvae,callbacks} (refactor step3)

    # EarlyStoppingAfterWarmup -> moved to src.models.callbacks (refactor step3)
    # ==========================================================
    # 5) CHECKLIST (colapso vs funcionando) — from gridsearch module
    # ==========================================================
    # checklist_table() imported from src.training.gridsearch

    # ==========================================================
    # 6) PIPELINE: carregar dataset por experimento + split correto
    # ==========================================================
    print("\n🔎 Localizando dataset...")
    dataset_root = find_dataset_root(
        marker_dirname="dataset_fullsquare_organized",
        dataset_root_hint=DATASET_ROOT,
        verbose=True,
    )

    print("\n📦 Carregando experimentos (sem redução; split por experimento)...")
    exps, df_info = load_experiments_as_list(
        dataset_root, verbose=True, reduction_config=None,
    )

    # --- Commit 3R: filter to selected experiments from protocol runner ---
    _sel_exps = _ov.get("_selected_experiments")
    if _sel_exps:
        _sel_set = set(str(p) for p in _sel_exps)
        _before = len(exps)
        exps = [(X, Y, D, C, p) for X, Y, D, C, p in exps if str(p) in _sel_set]
        print(f"⚡ Commit 3R: filtered {_before} → {len(exps)} experiment(s) "
              f"(selected_experiments={list(_sel_set)})")
        if len(exps) == 0:
            raise RuntimeError(
                f"No loaded experiments match _selected_experiments. "
                f"Available paths: {[p for *_, p in exps]}"
            )

    # --- Commit 3H: limit experiments / samples if requested ---
    if "max_experiments" in _ov:
        _me = int(_ov["max_experiments"])
        exps = exps[:_me]
        print(f"⚡ Commit 3H: limited to {len(exps)} experiment(s)")
    if "max_samples_per_exp" in _ov:
        _ms = int(_ov["max_samples_per_exp"])
        exps = [(X[:_ms], Y[:_ms], D[:_ms], C[:_ms], p) for X, Y, D, C, p in exps]
        print(f"⚡ Commit 3H: truncated to ≤{_ms} samples/exp")

    inv_path = _run.write_table("tables/dataset_inventory.xlsx", df_info, sheet_name="inventory")
    print(f"🧾 Inventário salvo: {inv_path}")

    # split por experimento (head=train, tail=val)
    X_train, Y_train, D_train, C_train, X_val, Y_val, D_val, C_val, df_split = split_train_val_per_experiment(
        exps=exps,
        val_split=float(TRAINING_CONFIG["validation_split"]),
        seed=int(TRAINING_CONFIG["seed"]),
        order_mode=str(TRAINING_CONFIG["per_experiment_split_order"]),
        within_exp_shuffle=bool(TRAINING_CONFIG["within_experiment_shuffle"]),
    )

    split_path = _run.write_table("tables/split_by_experiment.xlsx", df_split)
    print(f"✓ Split por experimento salvo: {split_path}")

    print(f"\n✓ Dados (por experimento): {len(X_train):,} treino | {len(X_val):,} validação")

    # Redução aplicada APÓS split — garante que val nunca é afetado
    _red_cfg = DATA_REDUCTION_CONFIG
    if _red_cfg.get("enabled", False):
        from src.data.loading import reduce_experiment_xy
        _rng_red = np.random.default_rng(int(TRAINING_CONFIG.get("seed", 42)))
        # Reduz train como um único bloco (já concatenado)
        X_train, Y_train = reduce_experiment_xy(
            X_train, Y_train, _red_cfg, _rng_red
        )
        D_train = D_train[:len(X_train)]
        C_train = C_train[:len(X_train)]
        print(f"✓ Data reduction pós-split: train={len(X_train):,} | val={len(X_val):,} (val intocado)")

    # --- Commit 3R: alignment assertions ---
    assert len(X_train) == len(Y_train) == len(D_train) == len(C_train), (
        f"Train alignment mismatch: X={len(X_train)} Y={len(Y_train)} "
        f"D={len(D_train)} C={len(C_train)}"
    )
    assert len(X_val) == len(Y_val) == len(D_val) == len(C_val), (
        f"Val alignment mismatch: X={len(X_val)} Y={len(Y_val)} "
        f"D={len(D_val)} C={len(C_val)}"
    )

    # Normalização (refactor step2: delegated to src.data.normalization)
    Dn_train, Cn_train, Dn_val, Cn_val, _norm_params = normalize_conditions(
        D_train, C_train, D_val, C_val,
    )
    D_min, D_max = _norm_params["D_min"], _norm_params["D_max"]
    C_min, C_max = _norm_params["C_min"], _norm_params["C_max"]

    # --- Commit 3R: log actual D/C ranges + unique values ---
    _d_unique = sorted(np.unique(D_train).tolist())
    _c_unique = sorted(np.unique(C_train).tolist())
    print(f"✓ Distância (treino): [{D_min:.3f}, {D_max:.3f}] m  unique={_d_unique}")
    print(f"✓ Corrente  (treino): [{C_min:.1f}, {C_max:.1f}] mA  unique={_c_unique}")

    # --- Commit 3R: tolerance smoke check ---
    if _sel_exps:
        _tgt_d = _ov.get("_regime_distance_m")
        _tgt_c = _ov.get("_regime_current_mA")
        _tol_d = float(_ov.get("dist_tol_m", 0.05))
        _tol_c = float(_ov.get("curr_tol_mA", 25.0))
        if _tgt_d is not None:
            for _du in _d_unique:
                if abs(_du - float(_tgt_d)) > _tol_d:
                    print(f"⚠️  D={_du:.3f}m exceeds tolerance of regime target "
                          f"{_tgt_d}m ± {_tol_d}m")
        if _tgt_c is not None:
            for _cu in _c_unique:
                if abs(_cu - float(_tgt_c)) > _tol_c:
                    print(f"⚠️  C={_cu:.1f}mA exceeds tolerance of regime target "
                          f"{_tgt_c}mA ± {_tol_c}mA")

    # ==========================================================
    # 7) GRID SEARCH
    # ==========================================================
    df_plan = pd.DataFrame([{
        "grid_id": i + 1,
        "group": g["group"],
        "tag": g["tag"],
        "cfg_json": json.dumps(g["cfg"], ensure_ascii=False),
        **g["cfg"],
    } for i, g in enumerate(GRID)])
    plan_path = _run.write_table("tables/gridsearch_plan.xlsx", df_plan)
    print(f"📌 Grid plan salvo: {plan_path}")

    # --- Commit 3H: dry_run — stop before training ---
    if _ov.get("dry_run", False):
        _first_cfg = GRID[0]["cfg"] if GRID else {}
        if _first_cfg:
            _vae, _ = build_cvae(_first_cfg)
            print(f"🔍 dry_run: model built | params={_vae.count_params():,}")
            _vae.summary(print_fn=lambda s: print("  " + s))
            del _vae
        import json as _json
        _dry_info = {
            "dry_run": True,
            "overrides": {k: v for k, v in _ov.items()},
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_grid": int(len(GRID)),
        }
        _run.write_json("logs/dry_run.json", _dry_info)
        print(f"✅ dry_run complete — wrote {LOGS_DIR / 'dry_run.json'}")
        return

    # ==========================================================
    # 7b) Delegate grid loop to gridsearch module (refactor step 4)
    # ==========================================================
    df_results = run_gridsearch(
        grid=GRID,
        training_config=TRAINING_CONFIG,
        analysis_quick=ANALYSIS_QUICK,
        X_train=X_train, Y_train=Y_train,
        Dn_train=Dn_train, Cn_train=Cn_train,
        X_val=X_val, Y_val=Y_val,
        Dn_val=Dn_val, Cn_val=Cn_val,
        run_paths=_run,
        overrides=_ov,
        df_plan=df_plan,
    )
    res_path = TABLES_DIR / "gridsearch_results.xlsx"

    state = {
        "run_id": RUN_ID,
        "run_dir": str(RUN_DIR),
        "dataset_root": str(dataset_root),
        "dataset_root_env": str(DATASET_ROOT),
        "output_base": str(OUTPUT_BASE),
        "paths": {
            "plots": str(PLOTS_DIR),
            "tables": str(TABLES_DIR),
            "models": str(MODELS_DIR),
            "logs": str(LOGS_DIR),
        },
        "training_config": TRAINING_CONFIG,
        "data_reduction_config": DATA_REDUCTION_CONFIG,
        "analysis_quick": ANALYSIS_QUICK,
        "eval_protocol": {
            "n_eval_samples": int(ANALYSIS_QUICK["n_eval_samples"]),
            "batch_infer": int(ANALYSIS_QUICK["batch_infer"]),
            "eval_slice": "val_head",
            "deterministic_inference": (str(ANALYSIS_QUICK.get("rank_mode","mc")).lower() == "det"),
            "rank_mode": str(ANALYSIS_QUICK.get("rank_mode","mc")).lower(),
            "mc_samples": int(ANALYSIS_QUICK.get("mc_samples", 8)),
        },
        "normalization": {"D_min": float(D_min), "D_max": float(D_max), "C_min": float(C_min), "C_max": float(C_max)},
        "data_split": {
            "split_mode": "per_experiment",
            "per_experiment_split_order": str(TRAINING_CONFIG["per_experiment_split_order"]),
            "within_experiment_shuffle": bool(TRAINING_CONFIG["within_experiment_shuffle"]),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "validation_split": float(TRAINING_CONFIG["validation_split"]),
            "seed": int(TRAINING_CONFIG["seed"]),
            "split_by_experiment_xlsx": str(split_path),
        },
        "grid": {
            "n_models": int(len(GRID)),
            "grid_plan_xlsx": str(plan_path),
            "grid_results_xlsx": str(res_path),
        },
        "artifacts": {
            "dataset_inventory_xlsx": str(inv_path),
            "split_by_experiment_xlsx": str(split_path),
            "grid_plan_xlsx": str(plan_path),
            "grid_results_xlsx": str(res_path),
            "best_model_full": str(MODELS_DIR / "best_model_full.keras"),
            "best_decoder": str(MODELS_DIR / "best_decoder.keras"),
            "best_prior_net": str(MODELS_DIR / "best_prior_net.keras"),
            "training_history_json": str(LOGS_DIR / "training_history.json"),
        }
    }

    state_path = _run.write_json("state_run.json", state)
    print(f"✓ state_run.json salvo: {state_path}")

if __name__ == "__main__":
    main()
