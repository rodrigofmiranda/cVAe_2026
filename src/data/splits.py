# -*- coding: utf-8 -*-
"""
src/data/splits.py — Per-experiment temporal split (head_tail).

Canonical split utilities shared by training and evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def cap_train_samples_per_experiment(
    X_train,
    Y_train,
    D_train,
    C_train,
    df_split: pd.DataFrame,
    max_samples_per_experiment: int,
):
    """
    Cap train samples per experiment *after* temporal split.

    This enforces the required ordering:
      split (head/train, tail/val) -> cap/reduce train only -> keep val untouched.

    Parameters
    ----------
    X_train, Y_train, D_train, C_train
        Concatenated train arrays returned by :func:`split_train_val_per_experiment`.
    df_split : pd.DataFrame
        Split table with at least ``n_train`` per experiment in the same order used
        during concatenation.
    max_samples_per_experiment : int
        Maximum number of train samples kept for each experiment.

    Returns
    -------
    tuple
        ``(X_train_cap, Y_train_cap, D_train_cap, C_train_cap, df_cap)``
        where ``df_cap`` has ``n_train_before``/``n_train_after`` per experiment.
    """
    cap = int(max_samples_per_experiment)
    if cap <= 0:
        raise ValueError("max_samples_per_experiment must be > 0")
    if df_split is None or "n_train" not in df_split.columns:
        raise ValueError("df_split must contain 'n_train' column")

    n_train_per_exp = df_split["n_train"].astype(int).tolist()
    expected = int(np.sum(n_train_per_exp))
    if expected != int(len(X_train)):
        raise ValueError(
            "Train length mismatch with df_split: "
            f"len(X_train)={len(X_train)} vs sum(n_train)={expected}"
        )

    keep_idx_parts = []
    cap_rows = []
    cursor = 0

    for i, n_train in enumerate(n_train_per_exp):
        n_keep = min(n_train, cap)
        if n_keep <= 0:
            cursor += n_train
            cap_rows.append({
                "exp_dir": df_split.iloc[i].get("exp_dir", ""),
                "n_train_before": int(n_train),
                "n_train_after": int(0),
            })
            continue

        # Keep the train head deterministically to preserve temporal order.
        local_idx = np.arange(0, n_keep, dtype=np.int64)
        keep_idx_parts.append(cursor + local_idx)
        cursor += n_train

        cap_rows.append({
            "exp_dir": df_split.iloc[i].get("exp_dir", ""),
            "n_train_before": int(n_train),
            "n_train_after": int(len(local_idx)),
        })

    if keep_idx_parts:
        keep_idx = np.concatenate(keep_idx_parts, axis=0)
    else:
        keep_idx = np.empty((0,), dtype=np.int64)

    df_cap = pd.DataFrame(cap_rows)
    return (
        X_train[keep_idx],
        Y_train[keep_idx],
        D_train[keep_idx],
        C_train[keep_idx],
        df_cap,
    )


def _cap_split_samples_per_experiment(
    X_split,
    Y_split,
    D_split,
    C_split,
    df_split: pd.DataFrame,
    max_samples_per_experiment: int,
    *,
    split_col: str,
    before_label: str,
    after_label: str,
):
    """Generic deterministic cap for concatenated per-experiment split arrays."""
    cap = int(max_samples_per_experiment)
    if cap <= 0:
        raise ValueError("max_samples_per_experiment must be > 0")
    if df_split is None or split_col not in df_split.columns:
        raise ValueError(f"df_split must contain '{split_col}' column")

    n_per_exp = df_split[split_col].astype(int).tolist()
    expected = int(np.sum(n_per_exp))
    if expected != int(len(X_split)):
        raise ValueError(
            f"{before_label} length mismatch with df_split: "
            f"len(X)={len(X_split)} vs sum({split_col})={expected}"
        )

    keep_idx_parts = []
    cap_rows = []
    cursor = 0

    for i, n_local in enumerate(n_per_exp):
        n_keep = min(n_local, cap)
        if n_keep > 0:
            # Deterministic head keep to preserve temporal order inside the split.
            local_idx = np.arange(0, n_keep, dtype=np.int64)
            keep_idx_parts.append(cursor + local_idx)
        else:
            local_idx = np.empty((0,), dtype=np.int64)

        cap_rows.append({
            "exp_dir": df_split.iloc[i].get("exp_dir", ""),
            before_label: int(n_local),
            after_label: int(len(local_idx)),
        })
        cursor += n_local

    keep_idx = (
        np.concatenate(keep_idx_parts, axis=0)
        if keep_idx_parts
        else np.empty((0,), dtype=np.int64)
    )
    df_cap = pd.DataFrame(cap_rows)
    return (
        X_split[keep_idx],
        Y_split[keep_idx],
        D_split[keep_idx],
        C_split[keep_idx],
        df_cap,
    )


def cap_val_samples_per_experiment(
    X_val,
    Y_val,
    D_val,
    C_val,
    df_split: pd.DataFrame,
    max_samples_per_experiment: int,
):
    """Cap validation samples per experiment after temporal split."""
    return _cap_split_samples_per_experiment(
        X_val,
        Y_val,
        D_val,
        C_val,
        df_split,
        max_samples_per_experiment,
        split_col="n_val",
        before_label="n_val_before",
        after_label="n_val_after",
    )


def apply_caps_to_df_split(
    df_split: pd.DataFrame,
    df_train_cap: pd.DataFrame | None = None,
    df_val_cap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a split table updated to match post-cap train/val arrays.

    The sequence pipeline windows already-capped arrays per experiment. If the
    original ``df_split`` is reused after capping, the reconstruction cursor no
    longer matches the concatenated arrays and windows can cross experiment
    boundaries. This helper rewrites ``n_train``/``n_val`` to their post-cap
    counts while keeping the original metadata rows/order.
    """
    if df_split is None:
        raise ValueError("df_split must not be None")

    df_out = df_split.copy()

    def _validate_cap_rows(df_cap: pd.DataFrame, after_col: str) -> np.ndarray:
        if len(df_cap) != len(df_out):
            raise ValueError(
                "Cap table length mismatch with df_split: "
                f"len(df_cap)={len(df_cap)} vs len(df_split)={len(df_out)}"
            )
        if "exp_dir" in df_cap.columns and "exp_dir" in df_out.columns:
            lhs = df_cap["exp_dir"].astype(str).tolist()
            rhs = df_out["exp_dir"].astype(str).tolist()
            if lhs != rhs:
                raise ValueError(
                    "Cap table experiment order mismatch with df_split."
                )
        if after_col not in df_cap.columns:
            raise ValueError(f"df_cap must contain '{after_col}' column")
        return df_cap[after_col].astype(int).to_numpy(copy=True)

    if df_train_cap is not None:
        train_after = _validate_cap_rows(df_train_cap, "n_train_after")
        if "n_train_before" in df_train_cap.columns:
            df_out["n_train_before_cap"] = (
                df_train_cap["n_train_before"].astype(int).to_numpy(copy=True)
            )
        df_out["n_train"] = train_after

    if df_val_cap is not None:
        val_after = _validate_cap_rows(df_val_cap, "n_val_after")
        if "n_val_before" in df_val_cap.columns:
            df_out["n_val_before_cap"] = (
                df_val_cap["n_val_before"].astype(int).to_numpy(copy=True)
            )
        df_out["n_val"] = val_after

    return df_out
