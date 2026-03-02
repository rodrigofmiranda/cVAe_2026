# -*- coding: utf-8 -*-
"""
batch_non_gaussianity_by_regime.py

Percorre automaticamente um dataset organizado por regimes (distância/corrente),
calcula testes/estatísticas de não-Gaussianidade do DELTA REAL do canal:

    d_real = Y - X

onde:
- X = sent_data_tuple.npy (N,2)
- Y = received_data_tuple_sync-phase.npy (N,2)

Gera uma tabela comparativa por regime em:
- CSV:  tables/non_gaussianity_by_regime.csv
- XLSX: tables/non_gaussianity_by_regime.xlsx

Uso:
    python batch_non_gaussianity_by_regime.py --dataset_root /caminho/para/dataset --out_dir /caminho/saida

Se --dataset_root não for passado, tenta:
    1) env DATASET_ROOT
    2) ./dataset_fullsquare_organized (na pasta atual)
"""

from __future__ import annotations
import os
import re
import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------
# Config (subamostragem)
# ------------------------------
# Univariado: roda nos dados completos (rápido e O(N))
# Multivariado (Mardia): subamostra para evitar O(N^2) pesado
MARDIA_MAX_N = 5000   # 5k -> matriz 5k x 5k ~ 200 MB float64 (aprox) + overhead
RNG_SEED = 0

# ------------------------------
# Helpers
# ------------------------------

def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def _parse_from_path(path: Path) -> tuple[float, float]:
    """
    Tenta extrair distância/corrente do caminho:
      dist_1.0m / curr_800mA
    Retorna (distance_m, current_mA) ou (nan,nan).
    """
    s = str(path).replace("\\", "/")
    md = re.search(r"dist_([0-9]+(?:\.[0-9]+)?)m", s)
    mc = re.search(r"curr_([0-9]+)mA", s)
    dist = float(md.group(1)) if md else np.nan
    curr = float(mc.group(1)) if mc else np.nan
    return dist, curr

def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"Esperado (N,2), recebido {a.shape}")
    return a.astype(np.float64, copy=False)

def _zscore_2d(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, ddof=1, keepdims=True)
    s = np.where(s == 0.0, 1.0, s)
    return (x - mu) / s

def univariate_stats(x: np.ndarray) -> dict:
    """
    Retorna:
      - p normaltest (D'Agostino K^2)
      - p Jarque-Bera
      - skew, excess_kurtosis
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    out = {}
    if x.size < 50:
        out.update({"p_dagostino": np.nan, "p_jarque_bera": np.nan, "skew": np.nan, "excess_kurt": np.nan})
        return out

    # D'Agostino
    try:
        _, p = stats.normaltest(x)
    except Exception:
        p = np.nan
    out["p_dagostino"] = float(p) if np.isfinite(p) else np.nan

    # Jarque-Bera
    try:
        _, p = stats.jarque_bera(x)
    except Exception:
        p = np.nan
    out["p_jarque_bera"] = float(p) if np.isfinite(p) else np.nan

    # Momentos
    out["skew"] = float(stats.skew(x, bias=False))
    out["excess_kurt"] = float(stats.kurtosis(x, fisher=True, bias=False))
    return out

def mardia_test_subsample(x2: np.ndarray, max_n: int = MARDIA_MAX_N, seed: int = RNG_SEED) -> dict:
    """
    Teste de Mardia (skew/kurt) em subamostra (para evitar memória).
    Implementação com Gram matrix G = Xm @ Sinv @ Xm.T (n x n).

    Retorna p-values:
      - p_mardia_skew
      - p_mardia_kurt
    """
    X = _as_2d(x2)
    X = X[np.all(np.isfinite(X), axis=1)]
    n_full = X.shape[0]
    if n_full < 200:
        return {"n_mardia": int(n_full), "p_mardia_skew": np.nan, "p_mardia_kurt": np.nan}

    rng = np.random.default_rng(seed)
    if n_full > max_n:
        idx = rng.choice(n_full, size=max_n, replace=False)
        X = X[idx]

    X = _zscore_2d(X)  # melhora estabilidade numérica
    n, p = X.shape

    mu = X.mean(axis=0)
    S = np.cov(X, rowvar=False, ddof=1)
    Sinv = np.linalg.pinv(S)
    Xm = X - mu

    # Gram matrix
    G = Xm @ Sinv @ Xm.T  # (n,n)

    # Mardia skewness
    b1p = float(np.mean(G**3))
    df = p * (p + 1) * (p + 2) / 6.0
    chi_skew = n * b1p / 6.0
    p_skew = 1.0 - stats.chi2.cdf(chi_skew, df=df)

    # Mardia kurtosis
    di = np.einsum("ij,jk,ik->i", Xm, Sinv, Xm)  # mahalanobis^2
    b2p = float(np.mean(di**2))
    mean_b2p = p * (p + 2)
    var_b2p = (8 * p * (p + 2)) / max(n, 1)
    z_kurt = (b2p - mean_b2p) / np.sqrt(max(var_b2p, 1e-12))
    p_kurt = 2.0 * (1.0 - stats.norm.cdf(abs(z_kurt)))

    return {"n_mardia": int(n), "p_mardia_skew": float(p_skew), "p_mardia_kurt": float(p_kurt)}

def find_regime_folders(dataset_root: Path) -> list[Path]:
    """
    Um 'regime folder' é aquele que contém ambos:
      - sent_data_tuple.npy
      - received_data_tuple_sync-phase.npy

    Retorna lista de pastas contendo esses arquivos.
    """
    sent_name = "sent_data_tuple.npy"
    recv_name = "received_data_tuple_sync-phase.npy"
    out = []
    for dirpath, dirnames, filenames in os.walk(dataset_root):
        fn = set(filenames)
        if sent_name in fn and recv_name in fn:
            out.append(Path(dirpath))
    return sorted(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default=None, help="Raiz do dataset (pasta que contém dist_*/curr_*/...).")
    ap.add_argument("--out_dir", type=str, default=None, help="Diretório de saída (default: <dataset_root>/analysis_non_gaussianity).")
    ap.add_argument("--alpha", type=float, default=0.01, help="Nível de significância para flags simples.")
    ap.add_argument("--mardia_max_n", type=int, default=MARDIA_MAX_N, help="Subamostra máxima para Mardia.")
    args = ap.parse_args()

    # Resolve dataset_root
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        dataset_root = Path(os.environ.get("DATASET_ROOT", "dataset_fullsquare_organized"))
    if not dataset_root.exists():
        raise SystemExit(f"[ERRO] dataset_root não existe: {dataset_root.resolve()}")

    out_dir = Path(args.out_dir) if args.out_dir else (dataset_root / "analysis_non_gaussianity")
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    regime_dirs = find_regime_folders(dataset_root)
    if not regime_dirs:
        raise SystemExit(f"[ERRO] Nenhum regime encontrado sob {dataset_root.resolve()} (faltam os .npy esperados?)")

    rows = []
    for rd in regime_dirs:
        sent_p = rd / "sent_data_tuple.npy"
        recv_p = rd / "received_data_tuple_sync-phase.npy"

        # metadados (qualquer um dos dois formatos)
        meta1 = rd / "metadata.json"
        meta2 = next(rd.glob("*_meta.json"), None)
        meta = _load_json(meta2) if meta2 and meta2.exists() else _load_json(meta1)

        dist_from_path, curr_from_path = _parse_from_path(rd)
        dist = _safe_float(meta.get("distance_m"), dist_from_path)
        # corrente: preferir meta['psu']['i_set_A'] se existir
        curr_mA = curr_from_path
        try:
            curr_mA = 1000.0 * float(meta.get("psu", {}).get("i_set_A"))
        except Exception:
            pass

        # carrega arrays
        try:
            X = np.load(sent_p)
            Y = np.load(recv_p)
        except Exception as e:
            print(f"[WARN] falha ao carregar {rd}: {e}")
            continue

        N = min(len(X), len(Y))
        X = X[:N]
        Y = Y[:N]

        # delta real
        d_real = (Y - X)
        d_real = d_real[np.all(np.isfinite(d_real), axis=1)]
        n_used = int(d_real.shape[0])

        # univariado (dados completos)
        ui = univariate_stats(d_real[:, 0])
        uq = univariate_stats(d_real[:, 1])

        # multivariado (subamostrado)
        md = mardia_test_subsample(d_real, max_n=int(args.mardia_max_n), seed=RNG_SEED)

        # flags simples
        flag_ng = (
            (ui["p_dagostino"] is not np.nan and ui["p_dagostino"] < args.alpha) or
            (uq["p_dagostino"] is not np.nan and uq["p_dagostino"] < args.alpha) or
            (md["p_mardia_skew"] is not np.nan and md["p_mardia_skew"] < args.alpha) or
            (md["p_mardia_kurt"] is not np.nan and md["p_mardia_kurt"] < args.alpha)
        )

        rows.append({
            "regime_dir": str(rd),
            "distance_m": dist,
            "current_mA": curr_mA,
            "timestamp": meta.get("timestamp", ""),
            "N_samples": n_used,

            # EVM/SNR se existirem nos metadados gerados pelo teu pipeline
            "EVM_%": _safe_float(meta.get("EVM_percentage"), np.nan),
            "SNR_dB": _safe_float(meta.get("SNR_dB"), np.nan),

            # Univariado I/Q
            "I_p_dagostino": ui["p_dagostino"],
            "I_p_jarque_bera": ui["p_jarque_bera"],
            "I_skew": ui["skew"],
            "I_excess_kurt": ui["excess_kurt"],

            "Q_p_dagostino": uq["p_dagostino"],
            "Q_p_jarque_bera": uq["p_jarque_bera"],
            "Q_skew": uq["skew"],
            "Q_excess_kurt": uq["excess_kurt"],

            # Multivariado (Mardia)
            "Mardia_n": md["n_mardia"],
            "Mardia_p_skew": md["p_mardia_skew"],
            "Mardia_p_kurt": md["p_mardia_kurt"],

            # resumo
            "reject_gaussian_alpha": bool(flag_ng),
            "alpha": args.alpha,
        })

        print(f"[OK] {rd.name} | dist={dist:.3g}m curr={curr_mA:.0f}mA | N={n_used} | kurt(I,Q)=({ui['excess_kurt']:.3g},{uq['excess_kurt']:.3g})")

    df = pd.DataFrame(rows)

    # Ordena de forma útil
    df = df.sort_values(by=["distance_m", "current_mA", "timestamp"], na_position="last").reset_index(drop=True)

    # Salva
    csv_path = tables_dir / "non_gaussianity_by_regime.csv"
    xlsx_path = tables_dir / "non_gaussianity_by_regime.xlsx"
    df.to_csv(csv_path, index=False)

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="non_gaussianity")
    except Exception as e:
        print(f"[WARN] Falha ao salvar XLSX ({e}). CSV foi salvo em {csv_path}")

    print("\n========================")
    print("Concluído.")
    print("CSV :", csv_path)
    print("XLSX:", xlsx_path)

if __name__ == "__main__":
    main()
