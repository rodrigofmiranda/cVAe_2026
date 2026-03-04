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
    # 5) CHECKLIST (colapso vs funcionando)
    # ==========================================================
    def checklist_table():
        rows = [
            ("KL no treino", "kl_loss médio > ~1 e não tende a 0", "kl_loss ~ 0 por muitas épocas", "colapso → decoder ignora z"),
            ("Atividade por dim", "std(z_mean_p[:,k]) > 0.05 em várias dims", "todas std ~0", "z não carrega info"),
            ("Sensibilidade decoder a z", "var(Y|z) muda ao perturbar z", "Y quase não muda ao perturbar z", "decoder ignora z"),
            ("Prior varia com (d,c)", "||μ_p||/dims muda com d ou c", "invariante em d,c", "condicional não aprendido"),
            ("Recon vs val", "recon e val_recon descem e estabilizam", "val piora muito e diverge", "overfit/instabilidade"),
            ("EVM/SNR (val)", "EVM_pred próximo do real em validação", "EVM_pred igual AWGN ou muito ruim", "twin fraco"),
        ]
        return pd.DataFrame(rows, columns=["Item", "OK (funcionando)", "Alerta (colapsando)", "Interpretação"])

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
        dataset_root, verbose=True, reduction_config=DATA_REDUCTION_CONFIG,
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

    inv_path = TABLES_DIR / "dataset_inventory.xlsx"
    with pd.ExcelWriter(inv_path, engine="openpyxl") as w:
        df_info.to_excel(w, index=False, sheet_name="inventory")
    print(f"🧾 Inventário salvo: {inv_path}")

    # split por experimento (head=train, tail=val)
    X_train, Y_train, D_train, C_train, X_val, Y_val, D_val, C_val, df_split = split_train_val_per_experiment(
        exps=exps,
        val_split=float(TRAINING_CONFIG["validation_split"]),
        seed=int(TRAINING_CONFIG["seed"]),
        order_mode=str(TRAINING_CONFIG["per_experiment_split_order"]),
        within_exp_shuffle=bool(TRAINING_CONFIG["within_experiment_shuffle"]),
    )

    split_path = TABLES_DIR / "split_by_experiment.xlsx"
    df_split.to_excel(split_path, index=False)
    print(f"✓ Split por experimento salvo: {split_path}")

    print(f"\n✓ Dados (por experimento): {len(X_train):,} treino | {len(X_val):,} validação")

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
    plan_path = TABLES_DIR / "gridsearch_plan.xlsx"
    df_plan = pd.DataFrame([{
        "grid_id": i + 1,
        "group": g["group"],
        "tag": g["tag"],
        "cfg_json": json.dumps(g["cfg"], ensure_ascii=False),
        **g["cfg"],
    } for i, g in enumerate(GRID)])
    df_plan.to_excel(plan_path, index=False)
    print(f"📌 Grid plan salvo: {plan_path}")

    results = []
    best_score = None

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
        (LOGS_DIR / "dry_run.json").write_text(_json.dumps(_dry_info, indent=2))
        print(f"✅ dry_run complete — wrote {LOGS_DIR / 'dry_run.json'}")
        return

    GRID_SAVE = {"save_each_model": True}

    def _safe_tag(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:160]

    def _grid_artifact_dir(models_dir: Path, gi: int, tag: str) -> Path:
        d = models_dir / f"grid_{gi:03d}__{_safe_tag(tag)}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_experiment_report_png(plot_path: Path, Xv, Yv, Yp, std_mu_p, kl_dim_mean, summary_lines, title):
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(title, fontsize=18, y=0.98)

        ax1 = fig.add_subplot(2, 3, 1)
        ax1.scatter(Yv[:,0], Yv[:,1], s=2)
        ax1.set_title("Real")
        ax1.set_xlabel("I"); ax1.set_ylabel("Q")
        ax1.grid(True, alpha=0.25)

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.scatter(Yp[:,0], Yp[:,1], s=2)
        ax2.set_title("cVAE")
        ax2.set_xlabel("I"); ax2.set_ylabel("Q")
        ax2.grid(True, alpha=0.25)

        ax3 = fig.add_subplot(2, 3, 3)
        ax3.scatter(Yv[:,0], Yv[:,1], s=2, label="Real")
        ax3.scatter(Yp[:,0], Yp[:,1], s=2, label="cVAE")
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

    def _save_experiment_xlsx(xlsx_path: Path, row_dict: dict):
        summary = pd.DataFrame([row_dict])
        cfg_json = pd.DataFrame([{"cfg_json": json.dumps({k: row_dict[k] for k in row_dict if k in [
            "activation","kl_anneal_epochs","batch_size","lr","dropout","free_bits","layer_sizes","latent_dim","beta"
        ]}, ensure_ascii=False)}])
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            summary.to_excel(w, index=False, sheet_name="summary")
            cfg_json.to_excel(w, index=False, sheet_name="cfg_json")

    for gi, item in enumerate(GRID, start=1):
        cfg = item["cfg"]
        group = item["group"]
        tag = item["tag"]

        print("\n" + "="*92)
        print(f"🚀 GRID {gi}/{len(GRID)} | group={group} | tag={tag}")
        print(f"    cfg = {cfg}")
        print("="*92)

        tf.keras.backend.clear_session()
        gc.collect()

        try:
            vae, kl_cb = build_cvae(cfg)
            callbacks = build_callbacks(TRAINING_CONFIG, cfg, kl_cb)

            t0 = time.time()
            _keras_verbose = int(_ov.get("keras_verbose", 2))  # Commit 3M
            hist = vae.fit(
                [X_train, Dn_train, Cn_train, Y_train], Y_train,
                validation_data=([X_val, Dn_val, Cn_val, Y_val], Y_val),
                epochs=int(TRAINING_CONFIG["epochs"]),
                batch_size=int(cfg["batch_size"]),
                callbacks=callbacks,
                verbose=_keras_verbose,
                shuffle=bool(TRAINING_CONFIG["shuffle_train_batches"]),
            )
            train_time_s = float(time.time() - t0)

            # Best epoch: consistente com o monitor do EarlyStopping (preferir val_recon_loss)
            val_mon = "val_recon_loss" if "val_recon_loss" in hist.history else "val_loss"
            val_hist = hist.history.get(val_mon, [])
            if len(val_hist) > 0:
                best_epoch = int(np.argmin(val_hist) + 1)
                best_val = float(np.min(val_hist))
            else:
                best_epoch = 0
                best_val = float("nan")

            # avaliação rápida: pega head do VAL (mas agora VAL é tail de cada experimento, não mistura com treino)
            N = min(int(ANALYSIS_QUICK["n_eval_samples"]), len(X_val))
            Xv = X_val[:N]; Yv = Y_val[:N]; Dv = Dn_val[:N]; Cv = Cn_val[:N]

            rank_mode = str(ANALYSIS_QUICK.get("rank_mode", "mc")).lower()
            K = int(ANALYSIS_QUICK.get("mc_samples", 8))

            inf_det = create_inference_model_from_full(vae, deterministic=True)
            Yp_det = inf_det.predict([Xv, Dv, Cv], batch_size=int(ANALYSIS_QUICK["batch_infer"]), verbose=0)

            if rank_mode == "det" or K <= 1:
                Yp = Yp_det
                var_mc = float("nan")
            else:
                inf_sto = create_inference_model_from_full(vae, deterministic=False)
                Ys = []
                for _ in range(K):
                    Ys.append(inf_sto.predict([Xv, Dv, Cv], batch_size=int(ANALYSIS_QUICK["batch_infer"]), verbose=0))
                Ys = np.stack(Ys, axis=0)
                Yp = Ys.mean(axis=0)
                var_mc = float(np.mean(np.var(Ys, axis=0)))

            evm_real, _ = calculate_evm(Xv, Yv)
            evm_pred, _ = calculate_evm(Xv, Yp)
            snr_real = calculate_snr(Xv, Yv)
            snr_pred = calculate_snr(Xv, Yp)

            prior_net = vae.get_layer("prior_net")
            mu_p, logvar_p = prior_net.predict([Xv, Dv, Cv], batch_size=int(ANALYSIS_QUICK["batch_infer"]), verbose=0)

            std_mu_p = np.std(mu_p, axis=0)
            active_dims = int(np.sum(std_mu_p > 0.05))

            kl_dim = 0.5 * (np.exp(logvar_p) + mu_p**2 - 1.0 - logvar_p)
            kl_mean_total = float(np.mean(np.sum(kl_dim, axis=1)))
            kl_mean_per_dim = float(np.mean(np.mean(kl_dim, axis=0)))

            dist_cfg_on = bool(ANALYSIS_QUICK.get("dist_metrics", True))
            psd_nfft = int(ANALYSIS_QUICK.get("psd_nfft", 2048))
            w_psd = float(ANALYSIS_QUICK.get("w_psd", 0.15))
            w_skew = float(ANALYSIS_QUICK.get("w_skew", 0.05))
            w_kurt = float(ANALYSIS_QUICK.get("w_kurt", 0.05))

            if dist_cfg_on:
                distm = residual_distribution_metrics(Xv, Yv, Yp, psd_nfft=psd_nfft)
                mean_l2 = float(distm["delta_mean_l2"])
                cov_fro = float(distm["delta_cov_fro"])
                var_real = float(distm["var_real_delta"])
                var_pred = float(distm["var_pred_delta"])
                skew_l2 = float(distm["delta_skew_l2"])
                kurt_l2 = float(distm["delta_kurt_l2"])
                psd_l2 = float(distm["delta_psd_l2"])
            else:
                d_real = (Yv - Xv)
                d_pred = (Yp - Xv)
                var_real = float(np.mean(np.var(d_real, axis=0)))
                var_pred = float(np.mean(np.var(d_pred, axis=0)))
                mean_l2 = float(np.linalg.norm(np.mean(d_pred,0) - np.mean(d_real,0)))
                cov_fro = float(np.linalg.norm(np.cov(d_pred.T) - np.cov(d_real.T), ord="fro"))
                skew_l2 = 0.0
                kurt_l2 = 0.0
                psd_l2 = 0.0

            pen_inactive = max(0, int(cfg["latent_dim"]//2) - active_dims)
            pen_kl_low  = max(0.0, 0.2 - kl_mean_per_dim)
            # >> C4 FIX: penalidade simétrica — KL patologicamente alto também é punido.
            #    Threshold 50/dim é conservador: para latent_dim=6 isso equivale a KL_total~300,
            #    acima do qual o prior está instável ou em variational overfit.
            pen_kl_high = 0.001 * max(0.0, kl_mean_per_dim - 50.0)

            pen_var_mismatch = 0.0
            if (not np.isnan(var_mc)):
                pen_var_mismatch = float(abs(var_mc - var_real))

            score_v2 = (
                abs(evm_pred - evm_real)
                + abs(snr_pred - snr_real)
                + 0.4 * mean_l2
                + 0.2 * cov_fro
                + 2.0 * pen_inactive
                + 1.0 * pen_kl_low
                + 1.0 * pen_kl_high   # >> C4 FIX: KL alto penalizado simetricamente
                + 0.5 * pen_var_mismatch
                + w_psd * psd_l2
                + w_skew * skew_l2
                + w_kurt * kurt_l2
            )
            score = abs(evm_pred - evm_real) + abs(snr_pred - snr_real)

            row = {
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
                "rank_mode": str(ANALYSIS_QUICK.get("rank_mode","mc")).lower(),
                "mc_samples": int(ANALYSIS_QUICK.get("mc_samples", 8)),
            }
            results.append(row)

            model_dir = _grid_artifact_dir(MODELS_DIR, gi, tag)
            exp_plots_dir = model_dir / "plots"
            exp_tables_dir = model_dir / "tables"
            exp_plots_dir.mkdir(parents=True, exist_ok=True)
            exp_tables_dir.mkdir(parents=True, exist_ok=True)

            kl_dim_mean = np.mean(0.5 * (np.exp(logvar_p) + mu_p**2 - 1.0 - logvar_p), axis=0)
            summary_lines = [
                f"grid_id: {gi} | group={group} | tag={tag}",
                f"EVM real: {evm_real:.2f}% | EVM pred: {evm_pred:.2f}% | ΔEVM: {(evm_pred-evm_real):+.2f}%",
                f"SNR real: {snr_real:.2f} dB | SNR pred: {snr_pred:.2f} dB | ΔSNR: {(snr_pred-snr_real):+.2f} dB",
                f"score_abs_delta: {score:.4f}",
                f"score_v2: {score_v2:.4f}",
                f"active_dims: {active_dims}/{int(cfg['latent_dim'])} | KL_mean_total: {kl_mean_total:.3f}",
            ]

            png_path = exp_plots_dir / "relatorio_completo_original_style.png"
            title = f"Relatório Consolidado — Twin + Latente | GRID {gi}/{len(GRID)}"
            _save_experiment_report_png(
                plot_path=png_path, Xv=Xv, Yv=Yv, Yp=Yp,
                std_mu_p=std_mu_p, kl_dim_mean=kl_dim_mean,
                summary_lines=summary_lines, title=title
            )

            xlsx_path = exp_tables_dir / "relatorio_diagnostico_completo.xlsx"
            _save_experiment_xlsx(xlsx_path=xlsx_path, row_dict=row)

            results[-1]["report_png_path"] = str(png_path)
            results[-1]["report_xlsx_path"] = str(xlsx_path)

            if GRID_SAVE["save_each_model"]:
                model_path = model_dir / "model_full.keras"
                vae.save(str(model_path), include_optimizer=False)
                results[-1]["model_full_path"] = str(model_path)

            is_best = (best_score is None) or (score_v2 < best_score)
            if is_best:
                best_score = float(score_v2)
                print("🏆 Novo melhor modelo do grid — salvando como 'best_model_full.keras'...")

                best_path = MODELS_DIR / "best_model_full.keras"
                vae.save(str(best_path), include_optimizer=False)

                vae.get_layer("decoder").save(str(MODELS_DIR / "best_decoder.keras"), include_optimizer=False)
                vae.get_layer("prior_net").save(str(MODELS_DIR / "best_prior_net.keras"), include_optimizer=False)

                hist_path = LOGS_DIR / "training_history.json"
                payload = {
                    "history": {k: [float(x) for x in v] for k, v in hist.history.items()},
                    "train_time_s": train_time_s,
                    "epochs_ran": int(len(next(iter(hist.history.values()))) if hist.history else 0),
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
                hist_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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

    res_path = TABLES_DIR / "gridsearch_results.xlsx"
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(["score_v2", "score_abs_delta"], ascending=[True, True])
    df_results.insert(0, "rank", np.arange(1, len(df_results) + 1))

    df_rank_readme = pd.DataFrame([
        {"Item": "Objetivo do ranking",
         "Descrição": "Selecionar o melhor digital twin que preserva estatísticas do canal medido e evita colapso do latente."},
        {"Item": "Score principal (score_v2)",
         "Descrição": "score_v2 = |ΔEVM| + |ΔSNR| + 0.4·Δμ + 0.2·ΔΣ + 2·pen(dims_inativas) + 1·pen(KL_dim_baixo) + termos PSD/skew/kurt/varMC. Menor é melhor."},
    ])

    with pd.ExcelWriter(res_path, engine="openpyxl") as w:
        df_results.to_excel(w, index=False, sheet_name="results_sorted")
        df_plan.to_excel(w, index=False, sheet_name="grid_plan_structured")
        checklist_table().to_excel(w, index=False, sheet_name="checklist_train_vs_collapse")
        df_rank_readme.to_excel(w, index=False, sheet_name="RANKING_README")
    print(f"📈 Grid results salvo: {res_path}")

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

    state_path = RUN_DIR / "state_run.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"✓ state_run.json salvo: {state_path}")

if __name__ == "__main__":
    main()
