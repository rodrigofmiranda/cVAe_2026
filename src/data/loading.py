# -*- coding: utf-8 -*-
"""
src.data.loading — Dataset IO, discovery, and experiment loading.

Contains:
- Low-level IO helpers (ensure_iq_shape, read_metadata, parse_dist_curr_from_path)
- Dataset discovery (discover_experiments, is_valid_dataset_root, find_dataset_root)
- Experiment loading and optional reduction (load_experiments_as_list, reduce_experiment_xy)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


# Accepted filenames for the received-signal array (tried in order).
ALT_RECV: list[str] = [
    "Y.npy",                                   # novo formato (dataset 2026)
    "received_data_tuple_sync-phase.npy",       # legado
    "received_data_tuple_sync_phase.npy",       # legado variante
    "received_data_tuple_sync.npy",             # legado
    "received_data_tuple.npy",                  # legado
]

# Accepted filenames for the sent-signal array (tried in order).
ALT_SENT: list[str] = [
    "X.npy",                   # novo formato (dataset 2026)
    "sent_data_tuple.npy",     # legado
]


def ensure_iq_shape(arr) -> np.ndarray:
    """Coerce an array to shape ``(N, 2)`` float32 I/Q format.

    Handles complex arrays (split into real/imag) and transposed layouts.

    Raises
    ------
    ValueError
        If the array cannot be coerced to ``(N, 2)``.
    """
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


def read_metadata(exp_dir: Path) -> dict:
    """Read the experiment metadata JSON, trying several candidate paths.

    Parameters
    ----------
    exp_dir : Path
        Root directory of a single experiment (e.g. ``dist_1m/curr_60mA/...``).

    Returns
    -------
    dict
        Parsed metadata, or ``{}`` if no file found / all reads fail.
    """
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


def parse_dist_curr_from_path(exp_dir: Path) -> tuple[float | None, int | None]:
    """Extract distance (m) and current (mA) from a directory path.

    Looks for patterns like ``dist_1.5m`` and ``curr_60mA`` in the
    stringified path.

    Returns
    -------
    tuple[float | None, int | None]
        ``(distance, current)``; either may be ``None`` if the pattern
        is not found.
    """
    s = str(exp_dir).replace("\\", "/")
    md = re.search(r"/dist_(\d+(?:\.\d+)?)m(?:/|$)", s)
    mc = re.search(r"/curr_(\d+)mA(?:/|$)", s)
    dist = float(md.group(1)) if md else None
    curr = int(mc.group(1)) if mc else None
    return dist, curr


# ---------------------------------------------------------------------------
# Dataset discovery (Commit 3B)
# ---------------------------------------------------------------------------
def discover_experiments(dataset_root: Path, verbose=True):
    """Scan *dataset_root* for valid experiment directories.

    An experiment is valid when its ``IQ_data/`` sub-folder contains both
    a sent-signal file (see :pydata:`ALT_SENT`) and one of the accepted
    received filenames listed in :pydata:`ALT_RECV`.

    Two discovery strategies are combined (metadata.json scan +
    IQ_data directory scan) for maximum recall.
    """
    exp_dirs = set()
    for meta in dataset_root.rglob("metadata.json"):
        if meta.parent.name == "IQ_data":
            exp_dir = meta.parent.parent
            iq_dir = meta.parent
        else:
            exp_dir = meta.parent
            iq_dir = exp_dir / "IQ_data"
        sent_ok = any((iq_dir / s).exists() for s in ALT_SENT)
        recv_ok = any((iq_dir / r).exists() for r in ALT_RECV)
        if sent_ok and recv_ok:
            exp_dirs.add(exp_dir)
    for iq_dir in dataset_root.rglob("IQ_data"):
        exp_dir = iq_dir.parent
        sent_ok = any((iq_dir / s).exists() for s in ALT_SENT)
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
    """Return ``True`` if *path* contains at least one valid experiment."""
    try:
        if not path.exists() or not path.is_dir():
            return False
        _ = discover_experiments(path, verbose=verbose)
        return True
    except Exception:
        return False


def find_dataset_root(
    marker_dirname: str = "dataset_fullsquare_organized",
    dataset_root_hint: Path | str | None = None,
    verbose: bool = True,
) -> Path:
    """Locate the dataset root directory.

    Parameters
    ----------
    marker_dirname : str
        Directory name used as a search marker.
    dataset_root_hint : Path or str, optional
        Explicit path to try first.  Falls back to ``$DATASET_ROOT``
        env-var, then filesystem search under ``/workspace``.
    verbose : bool
        Print progress messages.
    """
    if dataset_root_hint is None:
        dataset_root_hint = Path(
            os.environ.get("DATASET_ROOT", "/workspace/2026/dataset_fullsquare_organized")
        )
    else:
        dataset_root_hint = Path(dataset_root_hint)

    if is_valid_dataset_root(dataset_root_hint, verbose=False):
        if verbose:
            print(f"✅ Dataset root aceito do env: {dataset_root_hint}")
        return dataset_root_hint

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


# ---------------------------------------------------------------------------
# Data reduction (Commit 3B)
# ---------------------------------------------------------------------------
def _reduction_indices(n: int, cfg, rng) -> np.ndarray:
    """Return the sample indices kept by the configured reduction policy."""
    if not cfg.get("enabled", False):
        return np.arange(n, dtype=np.int64)

    target = int(cfg.get("target_samples_per_experiment", 200_000))
    minimum = int(cfg.get("min_samples_per_experiment", 80_000))

    if n <= target:
        return np.arange(n, dtype=np.int64)  # já está dentro do alvo, não corta

    target = max(target, minimum)

    mode = str(cfg.get("mode", "balanced_blocks")).lower()

    # --------------------------------------------------
    # Modo 1: center_crop — mais rápido, menos robusto
    # --------------------------------------------------
    if mode == "center_crop":
        start = (n - target) // 2
        return np.arange(start, start + target, dtype=np.int64)

    # --------------------------------------------------
    # Modo 2: balanced_blocks (padrão)
    # --------------------------------------------------
    block_len   = int(cfg.get("block_len", 4096))
    n_blocks    = int(cfg.get("n_blocks", 10))       # blocos a selecionar por bin
    time_spread = bool(cfg.get("time_spread", True))
    min_gap     = int(cfg.get("min_gap_blocks", 2))

    n_total_blocks = n // block_len
    if n_total_blocks == 0:
        return np.arange(min(target, n), dtype=np.int64)

    blocks_needed = target // block_len + 1

    if time_spread:
        max_start = n_total_blocks - 1
        step = max(1, max_start // max(1, blocks_needed - 1))
        candidates = np.arange(0, n_total_blocks, step, dtype=np.int64)
        jitter = rng.integers(-min_gap, min_gap + 1, size=len(candidates))
        candidates = np.clip(candidates + jitter, 0, n_total_blocks - 1)
        candidates = np.unique(candidates)
    else:
        candidates = np.arange(n_total_blocks, dtype=np.int64)

    n_sel = min(blocks_needed, len(candidates))
    chosen = rng.choice(candidates, size=n_sel, replace=False)
    chosen = np.sort(chosen)

    idx_list = []
    for b in chosen:
        start = int(b) * block_len
        end   = start + block_len
        idx_list.append(np.arange(start, min(end, n), dtype=np.int64))

    return np.concatenate(idx_list)[:target]


def reduce_aligned_arrays(*arrays, cfg, rng):
    """Reduce multiple aligned arrays with the same sampled indices.

    Parameters
    ----------
    *arrays : array-like
        Arrays sharing the same leading dimension.
    cfg : dict
        Data reduction config.
    rng : np.random.Generator
        Random generator used by the reduction policy.

    Returns
    -------
    tuple[np.ndarray, ...]
        Reduced arrays, all indexed by the exact same selected samples.
    """
    if not arrays:
        return tuple()

    n = min(len(arr) for arr in arrays)
    trimmed = [np.asarray(arr)[:n] for arr in arrays]
    idx = _reduction_indices(n, cfg, rng)
    return tuple(arr[idx] for arr in trimmed)


def reduce_experiment_xy(X, Y, cfg, rng):
    """Reduce a single experiment's X/Y arrays according to *cfg*."""
    X_red, Y_red = reduce_aligned_arrays(X, Y, cfg=cfg, rng=rng)
    return X_red, Y_red


# ---------------------------------------------------------------------------
# Experiment loading (Commit 3B)
# ---------------------------------------------------------------------------
def load_experiments_as_list(
    dataset_root: Path,
    verbose: bool = True,
    reduction_config: dict | None = None,
):
    """Load every experiment under *dataset_root* as separate arrays.

    Parameters
    ----------
    dataset_root : Path
        Root of the organised dataset.
    verbose : bool
        Print summary to stdout.
    reduction_config : dict, optional
        If provided, each experiment is reduced via
        :func:`reduce_experiment_xy` using this config dict.
        ``None`` (default) means no reduction — evaluation path.

    Returns
    -------
    tuple[list, pd.DataFrame]
        ``(exps, df_info)`` where each element of *exps* is
        ``(X, Y, D, C, exp_dir_str)``.
    """
    exp_dirs = discover_experiments(dataset_root, verbose=verbose)
    exps = []
    info = []

    if reduction_config is not None:
        rng_global = np.random.default_rng(int(reduction_config.get("seed", 42)))

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
        sent_path = None
        for s in ALT_SENT:
            p = iq_dir / s
            if p.exists():
                sent_path = p
                break
        recv_path = None
        for r in ALT_RECV:
            p = iq_dir / r
            if p.exists():
                recv_path = p
                break

        if recv_path is None or sent_path is None:
            info.append({"exp_dir": str(exp_dir), "status": "missing_files"})
            continue

        try:
            X_raw = np.load(sent_path, allow_pickle=False)
            Y_raw = np.load(recv_path, allow_pickle=False)
            X = ensure_iq_shape(X_raw)
            Y = ensure_iq_shape(Y_raw)

            n0 = min(X.shape[0], Y.shape[0])
            X = X[:n0]; Y = Y[:n0]

            if reduction_config is not None:
                rng = np.random.default_rng(rng_global.integers(0, 2**32 - 1))
                X, Y = reduce_experiment_xy(X, Y, reduction_config, rng)

            n = len(X)

            if dist is None or curr is None:
                raise ValueError(f"Não inferiu condições: dist={dist}, curr={curr}")

            D = np.full((n, 1), float(dist), dtype=np.float32)
            C = np.full((n, 1), float(curr), dtype=np.float32)

            # Leitura opcional do report.json (métricas de qualidade do canal)
            report_path = exp_dir / "report.json"
            report_data = {}
            if report_path.exists():
                try:
                    report_data = json.loads(report_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            exps.append((X, Y, D, C, str(exp_dir)))
            info.append({
                "exp_dir": str(exp_dir),
                "dist_m": float(dist),
                "curr_mA": int(curr),
                "n_samples": int(n),
                "status": "ok",
                "sent_path": str(sent_path),
                "recv_path": str(recv_path),
                "evm_pct":    float(report_data.get("evm_pct",    float("nan"))),
                "snr_dB":     float(report_data.get("snr_dB",     float("nan"))),
                "log_var_I":  float(report_data.get("log_var_I",  float("nan"))),
                "log_var_Q":  float(report_data.get("log_var_Q",  float("nan"))),
                "factor_ref": float(report_data.get("factor_ref", float("nan"))),
            })
        except Exception as e:
            info.append({"exp_dir": str(exp_dir), "status": "error", "error": str(e)})

    df_info = pd.DataFrame(info)
    if (df_info["status"] == "ok").sum() == 0:
        raise ValueError("Nenhum dataset carregado com sucesso.")
    if verbose:
        print(df_info["status"].value_counts())
    return exps, df_info
