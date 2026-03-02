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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

# ==========================================================
# 0) PATHS + RUN
# ==========================================================
DATASET_ROOT_ENV = os.environ.get("DATASET_ROOT", "/workspace/2026/dataset_fullsquare_organized")
DATASET_ROOT = Path(DATASET_ROOT_ENV)

OUTPUT_BASE = Path(os.environ.get("OUTPUT_BASE", "/workspace/2026/outputs"))

RUN_ID = os.environ.get("RUN_ID", datetime.now().strftime("run_%Y%m%d_%H%M%S"))
RUN_DIR = OUTPUT_BASE / RUN_ID
PLOTS_DIR = RUN_DIR / "plots"
TABLES_DIR = RUN_DIR / "tables"
MODELS_DIR = RUN_DIR / "models"
LOGS_DIR = RUN_DIR / "logs"
for p in [RUN_DIR, PLOTS_DIR, TABLES_DIR, MODELS_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

try:
    (OUTPUT_BASE / "_last_run.txt").write_text(str(RUN_DIR), encoding="utf-8")
except Exception:
    pass

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

# ==========================================================
# Seeds
# ==========================================================
np.random.seed(TRAINING_CONFIG["seed"])
tf.random.set_seed(TRAINING_CONFIG["seed"])

ALT_RECV = [
    "received_data_tuple_sync-phase.npy",
    "received_data_tuple_sync_phase.npy",
    "received_data_tuple_sync.npy",
    "received_data_tuple.npy",
]

def ensure_iq_shape(arr):
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        arr = np.stack([arr.real, arr.imag], axis=-1)
    if arr.ndim == 2 and arr.shape[1] == 2:
        pass
    elif arr.ndim == 2 and arr.shape[0] == 2:
        arr = arr.T
    else:
        raise ValueError(f"Formato inesperado I/Q: shape={arr.shape}, dtype={arr.dtype}")
    return arr.astype(np.float32, copy=False)

def read_metadata(exp_dir: Path):
    candidates = [exp_dir / "metadata.json", exp_dir / "IQ_data" / "metadata.json"]
    candidates += list(exp_dir.glob("*_meta.json"))
    for meta_path in candidates:
        if meta_path.exists():
            for enc in ["utf-8", "latin-1"]:
                try:
                    return json.loads(meta_path.read_text(encoding=enc))
                except Exception:
                    pass
    return {}

def parse_dist_curr_from_path(exp_dir: Path):
    s = str(exp_dir).replace("\\", "/")
    md = re.search(r"/dist_(\d+(?:\.\d+)?)m(?:/|$)", s)
    mc = re.search(r"/curr_(\d+)mA(?:/|$)", s)
    dist = float(md.group(1)) if md else None
    curr = int(mc.group(1)) if mc else None
    return dist, curr

# ---------------------------------------------------------------------------
# Descoberta de experimentos (regimes d,c)
# ---------------------------------------------------------------------------
def discover_experiments(dataset_root: Path, verbose=True):
    exp_dirs = set()
    for meta in dataset_root.rglob("metadata.json"):
        if meta.parent.name == "IQ_data":
            exp_dir = meta.parent.parent
            iq_dir = meta.parent
        else:
            exp_dir = meta.parent
            iq_dir = exp_dir / "IQ_data"
        sent_ok = (iq_dir / "sent_data_tuple.npy").exists()
        recv_ok = any((iq_dir / r).exists() for r in ALT_RECV)
        if sent_ok and recv_ok:
            exp_dirs.add(exp_dir)
    for iq_dir in dataset_root.rglob("IQ_data"):
        exp_dir = iq_dir.parent
        sent_ok = (iq_dir / "sent_data_tuple.npy").exists()
        recv_ok = any((iq_dir / r).exists() for r in ALT_RECV)
        if sent_ok and recv_ok:
            exp_dirs.add(exp_dir)

    exp_dirs = sorted(exp_dirs, key=lambda p: str(p))
    if verbose:
        print(f"✅ Experimentos válidos encontrados: {len(exp_dirs)}")
    if not exp_dirs:
        raise ValueError("Nenhum experimento válido encontrado (IQ_data/*.npy).")
    return exp_dirs

def is_valid_dataset_root(path: Path, verbose=False) -> bool:
    try:
        if not path.exists() or not path.is_dir():
            return False
        _ = discover_experiments(path, verbose=verbose)
        return True
    except Exception:
        return False

def find_dataset_root(marker_dirname="dataset_fullsquare_organized", verbose=True):
    if is_valid_dataset_root(DATASET_ROOT, verbose=False):
        if verbose:
            print(f"✅ Dataset root aceito do env: {DATASET_ROOT}")
        return DATASET_ROOT

    workspace = Path("/workspace")
    search_bases = [workspace / "2026", workspace / "2025", workspace]
    search_bases = [p for p in search_bases if p.exists()]

    candidates = []
    for base in search_bases:
        for p in base.rglob(marker_dirname):
            if p.is_dir():
                candidates.append(p)
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(f"Não encontrei '{marker_dirname}' em /workspace (e o DATASET_ROOT do env não é válido).")

    best_root = None
    best_count = -1
    for root in candidates:
        try:
            count = len(discover_experiments(root, verbose=False))
        except Exception:
            count = 0
        if count > best_count:
            best_count = count
            best_root = root

    if best_root is None or best_count <= 0:
        raise ValueError("Encontrei o marker, mas sem experimentos válidos.")

    if verbose:
        print(f"✅ Dataset root selecionado (auto): {best_root} ({best_count} exps)")
    return best_root

def reduce_experiment_xy(X, Y, cfg, rng):
    n = min(len(X), len(Y))
    X = X[:n]; Y = Y[:n]

    if not cfg.get("enabled", False):
        return X, Y

    target = int(cfg.get("target_samples_per_experiment", 200_000))
    minimum = int(cfg.get("min_samples_per_experiment", 80_000))

    if n <= target:
        return X, Y  # já está dentro do alvo, não corta

    target = max(target, minimum)

    mode = str(cfg.get("mode", "balanced_blocks")).lower()

    # --------------------------------------------------
    # Modo 1: center_crop — mais rápido, menos robusto
    # Pega uma janela contígua central de tamanho target.
    # Preserva continuidade temporal mas pode perder caudas.
    # --------------------------------------------------
    if mode == "center_crop":
        start = (n - target) // 2
        idx = np.arange(start, start + target, dtype=np.int64)
        return X[idx], Y[idx]

    # --------------------------------------------------
    # Modo 2: balanced_blocks (padrão)
    # Divide o experimento em blocos e amostra uniformemente
    # ao longo do tempo, preservando spread temporal.
    # --------------------------------------------------
    block_len   = int(cfg.get("block_len", 4096))
    n_blocks    = int(cfg.get("n_blocks", 10))       # blocos a selecionar por bin
    time_spread = bool(cfg.get("time_spread", True))
    min_gap     = int(cfg.get("min_gap_blocks", 2))

    # quantos blocos cabem no experimento
    n_total_blocks = n // block_len
    if n_total_blocks == 0:
        # experimento menor que um bloco: retorna tudo
        return X[:target], Y[:target]

    blocks_needed = target // block_len + 1

    if time_spread:
        # distribui os blocos escolhidos de forma espaçada
        # para cobrir toda a duração do experimento
        max_start = n_total_blocks - 1
        step = max(1, max_start // max(1, blocks_needed - 1))
        candidates = np.arange(0, n_total_blocks, step, dtype=np.int64)
        # adiciona jitter pequeno para não ser completamente periódico
        jitter = rng.integers(-min_gap, min_gap + 1, size=len(candidates))
        candidates = np.clip(candidates + jitter, 0, n_total_blocks - 1)
        candidates = np.unique(candidates)
    else:
        candidates = np.arange(n_total_blocks, dtype=np.int64)

    # seleciona blocos_needed blocos sem reposição
    n_sel = min(blocks_needed, len(candidates))
    chosen = rng.choice(candidates, size=n_sel, replace=False)
    chosen = np.sort(chosen)  # mantém ordem temporal

    idx_list = []
    for b in chosen:
        start = int(b) * block_len
        end   = start + block_len
        idx_list.append(np.arange(start, min(end, n), dtype=np.int64))

    idx = np.concatenate(idx_list)[:target]
    return X[idx], Y[idx]

# ==========================================================
# Loader: agora retorna lista por experimento (para split correto)
# ==========================================================
# ---------------------------------------------------------------------------
# Loader por experimento (evita leakage e preserva coerência física)
# ---------------------------------------------------------------------------
def load_experiments_as_list(dataset_root: Path, verbose=True):
    exp_dirs = discover_experiments(dataset_root, verbose=verbose)
    exps = []
    info = []
    rng_global = np.random.default_rng(int(DATA_REDUCTION_CONFIG.get("seed", 42)))

    for exp_dir in exp_dirs:
        meta = read_metadata(exp_dir)
        dist, curr = parse_dist_curr_from_path(exp_dir)

        if dist is None:
            for k in ["distance_m", "distance", "dist_m", "dist"]:
                if k in meta:
                    try: dist = float(meta[k]); break
                    except Exception: pass
        if curr is None:
            for k in ["current_mA", "current", "curr_mA", "curr"]:
                if k in meta:
                    try: curr = int(float(meta[k])); break
                    except Exception: pass

        iq_dir = exp_dir / "IQ_data"
        sent_path = iq_dir / "sent_data_tuple.npy"
        recv_path = None
        for r in ALT_RECV:
            p = iq_dir / r
            if p.exists():
                recv_path = p
                break

        if recv_path is None or not sent_path.exists():
            info.append({"exp_dir": str(exp_dir), "status": "missing_files"})
            continue

        try:
            X_raw = np.load(sent_path, allow_pickle=False)
            Y_raw = np.load(recv_path, allow_pickle=False)
            X = ensure_iq_shape(X_raw)
            Y = ensure_iq_shape(Y_raw)

            n0 = min(X.shape[0], Y.shape[0])
            X = X[:n0]; Y = Y[:n0]

            rng = np.random.default_rng(rng_global.integers(0, 2**32 - 1))
            X, Y = reduce_experiment_xy(X, Y, DATA_REDUCTION_CONFIG, rng)
            n = len(X)

            if dist is None or curr is None:
                raise ValueError(f"Não inferiu condições: dist={dist}, curr={curr}")

            D = np.full((n, 1), float(dist), dtype=np.float32)
            C = np.full((n, 1), float(curr), dtype=np.float32)

            exps.append((X, Y, D, C, str(exp_dir)))
            info.append({
                "exp_dir": str(exp_dir),
                "dist_m": float(dist),
                "curr_mA": int(curr),
                "n_samples": int(n),
                "status": "ok",
                "sent_path": str(sent_path),
                "recv_path": str(recv_path),
            })
        except Exception as e:
            info.append({"exp_dir": str(exp_dir), "status": "error", "error": str(e)})

    df_info = pd.DataFrame(info)
    if (df_info["status"] == "ok").sum() == 0:
        raise ValueError("Nenhum dataset carregado com sucesso.")
    if verbose:
        print(df_info["status"].value_counts())
    return exps, df_info

# ---------------------------------------------------------------------------
# Split por experimento (head_tail): treino no início, validação no fim.
# ---------------------------------------------------------------------------
def split_train_val_per_experiment(exps, val_split: float, seed: int,
                                   order_mode: str = "head_tail",
                                   within_exp_shuffle: bool = False):
    """
    Split correto:
      - por experimento (cada aquisição .npy)
      - head_tail: preserva temporalidade (head=train, tail=val)
      - sem shuffle global
    """
    rng = np.random.default_rng(seed)

    Xtr, Ytr, Dtr, Ctr = [], [], [], []
    Xva, Yva, Dva, Cva = [], [], [], []
    split_info = []

    for (X, Y, D, C, exp_path) in exps:
        n = len(X)
        n_val = int(round(val_split * n))
        n_val = max(1, n_val)
        n_train = max(1, n - n_val)

        if order_mode != "head_tail":
            order_mode = "head_tail"

        # head=train, tail=val
        idx_train = np.arange(0, n_train, dtype=np.int64)
        idx_val = np.arange(n_train, n, dtype=np.int64)

        if within_exp_shuffle:
            rng.shuffle(idx_train)
            rng.shuffle(idx_val)

        Xtr.append(X[idx_train]); Ytr.append(Y[idx_train]); Dtr.append(D[idx_train]); Ctr.append(C[idx_train])
        Xva.append(X[idx_val]);   Yva.append(Y[idx_val]);   Dva.append(D[idx_val]);   Cva.append(C[idx_val])

        split_info.append({
            "exp_dir": exp_path,
            "n_total": int(n),
            "n_train": int(len(idx_train)),
            "n_val": int(len(idx_val)),
        })

    X_train = np.concatenate(Xtr, axis=0)
    Y_train = np.concatenate(Ytr, axis=0)
    D_train = np.concatenate(Dtr, axis=0)
    C_train = np.concatenate(Ctr, axis=0)

    X_val = np.concatenate(Xva, axis=0)
    Y_val = np.concatenate(Yva, axis=0)
    D_val = np.concatenate(Dva, axis=0)
    C_val = np.concatenate(Cva, axis=0)

    df_split = pd.DataFrame(split_info)
    return X_train, Y_train, D_train, C_train, X_val, Y_val, D_val, C_val, df_split

# ==========================================================
# 3) MÉTRICAS rápidas (para grid)
# ==========================================================
def calculate_evm(ref, test):
    ref = np.asarray(ref); test = np.asarray(test)
    rc = ref[:,0] + 1j*ref[:,1]
    tc = test[:,0] + 1j*test[:,1]
    mean_power = np.mean(np.abs(rc)**2)
    if mean_power == 0:
        return float("inf"), float("-inf")
    evm = np.sqrt(np.mean(np.abs(tc-rc)**2) / mean_power)
    return float(evm*100), float(20*np.log10(max(evm, 1e-12)))

def calculate_snr(ref, test):
    ref = np.asarray(ref); test = np.asarray(test)
    rc = ref[:,0] + 1j*ref[:,1]
    tc = test[:,0] + 1j*test[:,1]
    sp = np.mean(np.abs(rc)**2)
    npow = np.mean(np.abs(rc-tc)**2)
    if npow == 0:
        return float("inf")
    return float(10*np.log10(max(sp/npow, 1e-12)))

def _skew_kurt(x: np.ndarray, eps: float = 1e-12):
    x = np.asarray(x, dtype=np.float64)
    m = np.mean(x, axis=0)
    v = np.var(x, axis=0)
    s = np.sqrt(v + eps)
    z = (x - m) / s
    skew = np.mean(z**3, axis=0)
    kurt = np.mean(z**4, axis=0) - 3.0
    return skew, kurt

def _psd_log(xc: np.ndarray, nfft: int = 2048, eps: float = 1e-12):
    xc = np.asarray(xc, dtype=np.complex128).ravel()
    n = len(xc)
    nfft = int(min(max(256, nfft), n)) if n > 0 else int(nfft)
    if nfft < 256 or n < 256:
        nfft = max(1, n)
    win = np.hanning(nfft) if nfft >= 8 else np.ones(nfft)
    nseg = 4
    hop = max(1, (n - nfft) // max(1, nseg-1)) if n > nfft else nfft
    acc = None
    cnt = 0
    for start in range(0, max(1, n - nfft + 1), hop):
        seg = xc[start:start+nfft]
        if len(seg) < nfft:
            break
        segw = seg * win
        X = np.fft.fft(segw, n=nfft)
        P = (np.abs(X)**2) / (np.sum(win**2) + eps)
        acc = P if acc is None else (acc + P)
        cnt += 1
        if cnt >= nseg:
            break
    if acc is None:
        acc = (np.abs(np.fft.fft(xc, n=nfft))**2) / max(1, nfft)
        cnt = 1
    psd = acc / max(1, cnt)
    return np.log10(psd + eps)

# ---------------------------------------------------------------------------
# Métricas do residual Δ=(Y-X): momentos e PSD ajudam a avaliar fidelidade do twin.
# ---------------------------------------------------------------------------
def residual_distribution_metrics(X: np.ndarray, Y: np.ndarray, Yp: np.ndarray, psd_nfft: int = 2048):
    d_real = np.asarray(Y) - np.asarray(X)
    d_pred = np.asarray(Yp) - np.asarray(X)

    mean_l2 = float(np.linalg.norm(np.mean(d_pred, axis=0) - np.mean(d_real, axis=0)))
    cov_fro = float(np.linalg.norm(np.cov(d_pred.T) - np.cov(d_real.T), ord="fro"))

    var_real = float(np.mean(np.var(d_real, axis=0)))
    var_pred = float(np.mean(np.var(d_pred, axis=0)))

    skew_r, kurt_r = _skew_kurt(d_real)
    skew_p, kurt_p = _skew_kurt(d_pred)
    skew_l2 = float(np.linalg.norm(skew_p - skew_r))
    kurt_l2 = float(np.linalg.norm(kurt_p - kurt_r))

    cr = d_real[:, 0] + 1j * d_real[:, 1]
    cp = d_pred[:, 0] + 1j * d_pred[:, 1]
    psd_r = _psd_log(cr, nfft=int(psd_nfft))
    psd_p = _psd_log(cp, nfft=int(psd_nfft))
    psd_l2 = float(np.linalg.norm(psd_p - psd_r) / np.sqrt(len(psd_r) if len(psd_r) else 1))

    return {
        "delta_mean_l2": mean_l2,
        "delta_cov_fro": cov_fro,
        "var_real_delta": var_real,
        "var_pred_delta": var_pred,
        "delta_skew_l2": skew_l2,
        "delta_kurt_l2": kurt_l2,
        "delta_psd_l2": psd_l2,
    }

# ==========================================================
# 4) MODELO (cVAE prior condicional)
# ==========================================================
def _activation_layer(name: str):
    name = (name or "").lower().strip()
    if name in ["leaky_relu", "lrelu", "leakyrelu"]:
        return layers.LeakyReLU(alpha=0.2)
    return layers.Activation(name)

@tf.keras.utils.register_keras_serializable(package="VLC")
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

@tf.keras.utils.register_keras_serializable(package="VLC")
# ---------------------------------------------------------------------------
# Loss heteroscedástico + KL(q||p) com β-annealing e free-bits.
# ---------------------------------------------------------------------------
class CondPriorVAELoss(layers.Layer):
    def __init__(self, beta=1.0, free_bits=0.0, **kwargs):
        super().__init__(**kwargs)
        self.beta_init = float(beta)
        self.free_bits = float(free_bits)
        self.beta = tf.Variable(self.beta_init, trainable=False, dtype=tf.float32, name="beta")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p = inputs

        y_mean = out_params[:, :2]
        y_log_var = tf.clip_by_value(out_params[:, 2:], -6.0, 1.0)
        y_var = tf.exp(y_log_var) + 1e-6
        nll = 0.5 * tf.reduce_sum(
            y_log_var + tf.square(y_true - y_mean) / y_var + tf.math.log(2.0*np.pi),
            axis=-1
        )
        recon = tf.reduce_mean(nll)

        vq = tf.exp(tf.clip_by_value(z_log_var_q, -20.0, 20.0))
        vp = tf.exp(tf.clip_by_value(z_log_var_p, -20.0, 20.0))
        kl_dim = 0.5 * (
            tf.math.log(vp + 1e-12) - tf.math.log(vq + 1e-12)
            + (vq + tf.square(z_mean_q - z_mean_p)) / (vp + 1e-12)
            - 1.0
        )
        kl_per_sample = tf.reduce_sum(kl_dim, axis=-1)

        fb = tf.cast(self.free_bits, kl_per_sample.dtype)
        kl_fb = tf.maximum(kl_per_sample - fb, 0.0)
        kl = tf.reduce_mean(kl_fb)

        total = recon + self.beta * tf.minimum(kl, 200.0)

        self.add_loss(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(tf.reduce_mean(kl_per_sample))
        return y_mean

    @property
    def metrics(self):
        return [self.recon_loss_tracker, self.kl_loss_tracker]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"beta": self.beta_init, "free_bits": self.free_bits})
        return cfg

class KLAnnealingCallback(Callback):
    def __init__(self, loss_layer, beta_start=0.0, beta_end=1.0, annealing_epochs=50):
        super().__init__()
        self.loss_layer = loss_layer
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.annealing_epochs = int(annealing_epochs)

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.annealing_epochs:
            progress = epoch / max(self.annealing_epochs, 1)
            b = self.beta_start + (self.beta_end - self.beta_start) * progress
            self.loss_layer.beta.assign(b)
        else:
            self.loss_layer.beta.assign(self.beta_end)

def build_mlp(name, in_shapes, layer_sizes, activation="leaky_relu", dropout=0.0, out_dim=32, out_name_prefix=""):
    ins = [layers.Input(shape=s, name=f"{name}_in_{i}") for i, s in enumerate(in_shapes)]
    h = layers.Concatenate(name=f"{name}_concat")(ins)
    for i, u in enumerate(layer_sizes):
        h = layers.Dense(u, kernel_initializer="glorot_uniform", name=f"{name}_dense_{i}")(h)
        h = layers.BatchNormalization(name=f"{name}_bn_{i}")(h)
        h = _activation_layer(activation)(h)
        if dropout and dropout > 0:
            h = layers.Dropout(dropout, name=f"{name}_drop_{i}")(h)
    mu = layers.Dense(out_dim, name=f"{out_name_prefix}z_mean")(h)
    lv = layers.Dense(out_dim, name=f"{out_name_prefix}z_log_var")(h)
    return models.Model(ins, [mu, lv], name=name)

def build_decoder(layer_sizes, latent_dim, activation="leaky_relu", dropout=0.0):
    z_in = layers.Input(shape=(latent_dim,), name="z_input")
    cond_in = layers.Input(shape=(4,), name="cond_input")  # x(2)+d(1)+c(1)
    h = layers.Concatenate(name="dec_concat")([z_in, cond_in])
    for i, u in enumerate(layer_sizes):
        h = layers.Dense(u, kernel_initializer="glorot_uniform", name=f"dec_dense_{i}")(h)
        h = layers.BatchNormalization(name=f"dec_bn_{i}")(h)
        h = _activation_layer(activation)(h)
        if dropout and dropout > 0:
            h = layers.Dropout(dropout, name=f"dec_drop_{i}")(h)
    out = layers.Dense(4, name="output_params")(h)  # mean_I,mean_Q,logvar_I,logvar_Q
    return models.Model([z_in, cond_in], out, name="decoder")

def build_condprior_cvae(cfg):
    layer_sizes = cfg["layer_sizes"]
    latent_dim = int(cfg["latent_dim"])
    beta = float(cfg["beta"])
    lr = float(cfg["lr"])
    dropout = float(cfg["dropout"])
    free_bits = float(cfg.get("free_bits", 0.0))
    kl_anneal_epochs = int(cfg.get("kl_anneal_epochs", 50))
    activation = cfg.get("activation", "leaky_relu")

    encoder = build_mlp(
        name="encoder",
        in_shapes=[(2,), (1,), (1,), (2,)],
        layer_sizes=layer_sizes,
        activation=activation,
        dropout=dropout,
        out_dim=latent_dim,
        out_name_prefix="q_",
    )

    prior_net = build_mlp(
        name="prior_net",
        in_shapes=[(2,), (1,), (1,)],
        layer_sizes=layer_sizes,
        activation=activation,
        dropout=dropout,
        out_dim=latent_dim,
        out_name_prefix="p_",
    )

    decoder = build_decoder(layer_sizes=layer_sizes, latent_dim=latent_dim, activation=activation, dropout=dropout)

    x_in = layers.Input(shape=(2,), name="x_input")
    d_in = layers.Input(shape=(1,), name="distance_input")
    c_in = layers.Input(shape=(1,), name="current_input")
    y_in = layers.Input(shape=(2,), name="y_true")

    z_mean_q, z_log_var_q = encoder([x_in, d_in, c_in, y_in])
    z_mean_p, z_log_var_p = prior_net([x_in, d_in, c_in])

    # >> C4.2 FIX: clip do log-var do prior também no treino, igual à inferência.
    #    Sem isso, z_log_var_p pode divergir durante o treino enquanto na inferência
    #    é clipado — viés sistemático que infla artificialmente o KL.
    z_log_var_p = layers.Lambda(
        lambda t: tf.clip_by_value(t, -10.0, 10.0), name="clip_p_logvar_train"
    )(z_log_var_p)

    z = Sampling(name="sampling")([z_mean_q, z_log_var_q])

    cond = layers.Concatenate(name="cond_concat")([x_in, d_in, c_in])  # (N,4)
    out_params = decoder([z, cond])

    beta_initial = 0.0
    loss_layer = CondPriorVAELoss(beta=beta_initial, free_bits=free_bits, name="condprior_loss")
    y_mean = loss_layer([y_in, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p])

    vae = models.Model([x_in, d_in, c_in, y_in], y_mean, name="cvae_condprior")
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    vae.compile(optimizer=opt)

    kl_cb = KLAnnealingCallback(loss_layer, beta_start=0.0, beta_end=beta, annealing_epochs=kl_anneal_epochs)
    return vae, kl_cb

# ---------------------------------------------------------------------------
# Inferência usando prior condicional: deterministic vs sampling.
# ---------------------------------------------------------------------------
def create_inference_model_from_full(full_model: tf.keras.Model, deterministic: bool = True):
    prior = full_model.get_layer("prior_net")
    dec = full_model.get_layer("decoder")

    x_in = layers.Input(shape=(2,), name="x_input")
    d_in = layers.Input(shape=(1,), name="distance_input")
    c_in = layers.Input(shape=(1,), name="current_input")

    z_mean_p, z_log_var_p = prior([x_in, d_in, c_in])
    z_log_var_p = layers.Lambda(lambda t: tf.clip_by_value(t, -10.0, 10.0), name="clip_zlogvar")(z_log_var_p)

    if deterministic:
        z = layers.Lambda(lambda t: t, name="z_det")(z_mean_p)
    else:
        eps_z = layers.Lambda(lambda t: tf.random.normal(tf.shape(t)), name="eps_z")(z_mean_p)
        z = layers.Lambda(lambda a: a[0] + tf.exp(0.5 * a[1]) * a[2], name="sample_z")([z_mean_p, z_log_var_p, eps_z])

    cond = layers.Concatenate(name="cond_concat_inf")([x_in, d_in, c_in])
    out_params = dec([z, cond])

    y_mean = layers.Lambda(lambda t: t[:, :2], name="y_mean")(out_params)
    y_log_var = layers.Lambda(lambda t: tf.clip_by_value(t[:, 2:], -6.0, 1.0), name="y_logvar")(out_params)

    if deterministic:
        y = layers.Lambda(lambda t: t, name="y_det")(y_mean)
    else:
        eps_y = layers.Lambda(lambda t: tf.random.normal(tf.shape(t)), name="eps_y")(y_mean)
        y = layers.Lambda(lambda a: a[0] + tf.exp(0.5 * a[1]) * a[2], name="sample_y")([y_mean, y_log_var, eps_y])

    return models.Model([x_in, d_in, c_in], y,
                        name=("inference_condprior_det" if deterministic else "inference_condprior"))

class EarlyStoppingAfterWarmup(tf.keras.callbacks.Callback):
    """
    EarlyStopping que só começa a contar 'patience' após um warmup de N épocas.
    Evita parar cedo demais em modelos com KL/annealing instável no início.

    >> C1 FIX: best_weights é salvo desde a época 1 (não só após o warmup).
       O warmup apenas atrasa a PARADA — o checkpoint do melhor val_recon
       é gravado continuamente, garantindo restore correto mesmo quando o
       pico ocorre antes do fim do annealing.
    """
    def __init__(self, monitor="val_recon_loss", patience=20, warmup_epochs=0,
                 min_delta=0.0, restore_best_weights=True, verbose=1):
        super().__init__()
        self.monitor = monitor
        self.patience = int(patience)
        self.warmup_epochs = int(warmup_epochs)
        self.min_delta = float(min_delta)
        self.restore_best_weights = bool(restore_best_weights)
        self.verbose = int(verbose)

        self.wait = 0
        self.best = np.inf
        self.best_weights = None
        self.best_epoch = 0  # rastreamento para log

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor, None)
        if current is None:
            return

        # >> C1 FIX: checkpoint gravado SEMPRE que há melhora, independente do warmup
        if current < (self.best - self.min_delta):
            self.best = current
            self.best_epoch = epoch + 1
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            return  # melhora: não incrementa wait, não para

        # warmup: incrementa wait mas não para
        if epoch + 1 <= self.warmup_epochs:
            self.wait += 1  # conta mas não para — impede que KL instável no início
            return          # produza parada prematura

        # pós-warmup: lógica normal de parada
        self.wait += 1
        if self.wait >= self.patience:
            if self.verbose:
                print(f"\nEarlyStoppingAfterWarmup: parando em epoch {epoch+1} "
                      f"(best {self.monitor}={self.best:.6f} @ epoch {self.best_epoch})")
            if self.restore_best_weights and (self.best_weights is not None):
                self.model.set_weights(self.best_weights)
                if self.verbose:
                    print(f"  → pesos restaurados para epoch {self.best_epoch}")
            self.model.stop_training = True
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
dataset_root = find_dataset_root(marker_dirname="dataset_fullsquare_organized", verbose=True)

print("\n📦 Carregando experimentos (sem redução; split por experimento)...")
exps, df_info = load_experiments_as_list(dataset_root, verbose=True)

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

# Normalização (recomendado: baseada no TREINO, para evitar leakage)
D_min, D_max = float(D_train.min()), float(D_train.max())
C_min, C_max = float(C_train.min()), float(C_train.max())

Dn_train = (D_train - D_min) / (D_max - D_min) if D_max > D_min else np.zeros_like(D_train)
Cn_train = (C_train - C_min) / (C_max - C_min) if C_max > C_min else np.full_like(C_train, 0.5)

Dn_val = (D_val - D_min) / (D_max - D_min) if D_max > D_min else np.zeros_like(D_val)
Cn_val = (C_val - C_min) / (C_max - C_min) if C_max > C_min else np.full_like(C_val, 0.5)

print(f"✓ Distância (treino): [{D_min:.3f}, {D_max:.3f}] m")
print(f"✓ Corrente  (treino): [{C_min:.1f}, {C_max:.1f}] mA")

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
        vae, kl_cb = build_condprior_cvae(cfg)

# >> C1 FIX: warmup por modelo = kl_anneal_epochs (razão real do warmup).
        #    ReduceLROnPlateau monitora val_loss (total) — métrica DISTINTA do
        #    EarlyStopping (val_recon_loss), eliminando o acoplamento destrutivo.
        _warmup = int(cfg.get("kl_anneal_epochs", 80))

        callbacks = [
            EarlyStoppingAfterWarmup(
                monitor="val_recon_loss",
                patience=TRAINING_CONFIG["patience"],
                warmup_epochs=_warmup,
                min_delta=1e-5,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",          # >> C1 FIX: monitor distinto do EarlyStopping
                factor=0.5,
                patience=TRAINING_CONFIG["reduce_lr_patience"],
                min_lr=1e-6,
                verbose=1,
            ),
            kl_cb,
        ]

        t0 = time.time()
        hist = vae.fit(
            [X_train, Dn_train, Cn_train, Y_train], Y_train,
            validation_data=([X_val, Dn_val, Cn_val, Y_val], Y_val),
            epochs=int(TRAINING_CONFIG["epochs"]),
            batch_size=int(cfg["batch_size"]),
            callbacks=callbacks,
            verbose=1,
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