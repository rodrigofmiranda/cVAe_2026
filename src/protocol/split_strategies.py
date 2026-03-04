# -*- coding: utf-8 -*-
"""
Split strategies — pluggable train/val splitting for the protocol engine.

Strategies
----------
per_experiment (default)
    Each experiment is split temporally (head=train, tail=val).
    Delegates to :func:`src.data.splits.split_train_val_per_experiment`.

grouped
    Experiments are bucketed by a grouping key (e.g. ``distance_m``).
    Entire groups go to train or val — no intra-experiment splitting.
    The last group(s) by sorted key value form the validation set.

Both strategies return the same 9-tuple so callers need not branch.

Commit 3W (Phase 3).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# Type alias (same as selector_engine)
Experiment = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]

# The canonical return type of every strategy
SplitResult = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,   # X_train, Y_train, D_train, C_train
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,   # X_val,   Y_val,   D_val,   C_val
    pd.DataFrame,                                       # df_split
]


# ---------------------------------------------------------------------------
# per_experiment — temporal head/tail (existing behaviour)
# ---------------------------------------------------------------------------

def split_per_experiment(
    exps: List[Experiment],
    val_split: float = 0.2,
    seed: int = 42,
    within_exp_shuffle: bool = False,
) -> SplitResult:
    """Delegate to the canonical implementation in ``src.data.splits``.

    This is a thin wrapper so that both the protocol engine and the
    training monolith share exactly the same splitting function.
    """
    from src.data.splits import split_train_val_per_experiment

    return split_train_val_per_experiment(
        exps,
        val_split=val_split,
        seed=seed,
        order_mode="head_tail",
        within_exp_shuffle=within_exp_shuffle,
    )


# ---------------------------------------------------------------------------
# grouped — entire experiments assigned to train/val by a grouping key
# ---------------------------------------------------------------------------

def _group_key_for_experiment(
    exp: Experiment,
    group_by: str,
) -> float:
    """Extract the scalar grouping value from an experiment tuple.

    Supported *group_by* values:
    - ``"distance_m"`` → ``np.mean(D)``
    - ``"current_mA"`` → ``np.mean(C)``
    """
    _X, _Y, D, C, _path = exp
    if group_by == "distance_m":
        return float(np.mean(D))
    if group_by == "current_mA":
        return float(np.mean(C))
    raise ValueError(f"Unsupported group_by key: {group_by!r}")


def split_grouped(
    exps: List[Experiment],
    val_split: float = 0.2,
    group_by: str = "distance_m",
) -> SplitResult:
    """Split experiments into train/val by group membership.

    Groups are formed by the *group_by* key value (e.g. all experiments at
    distance 0.8 m form one group).  Groups are sorted by key value; the
    last group(s) — accounting for *val_split* — become validation data.
    All samples within a group go to the **same** split (no temporal cut).

    Parameters
    ----------
    exps : list of (X, Y, D, C, path_str)
    val_split : float
        Target fraction of **groups** (not samples) for validation.
        At least 1 group is always assigned to val.
    group_by : str
        Grouping key (``"distance_m"`` or ``"current_mA"``).

    Returns
    -------
    Same 9-tuple as :func:`split_per_experiment`.
    """
    if not exps:
        raise ValueError("Cannot split an empty experiment list.")

    # Build groups: key → list of experiments
    from collections import OrderedDict
    groups: dict[float, List[Experiment]] = {}
    for exp in exps:
        k = _group_key_for_experiment(exp, group_by)
        groups.setdefault(k, []).append(exp)

    sorted_keys = sorted(groups.keys())
    n_groups = len(sorted_keys)

    # Determine how many groups go to val (at least 1)
    n_val_groups = max(1, round(val_split * n_groups))
    n_val_groups = min(n_val_groups, n_groups - 1) if n_groups > 1 else n_groups

    val_keys = set(sorted_keys[-n_val_groups:])
    train_keys = set(sorted_keys) - val_keys

    Xtr, Ytr, Dtr, Ctr = [], [], [], []
    Xva, Yva, Dva, Cva = [], [], [], []
    split_info = []

    for k in sorted_keys:
        is_val = k in val_keys
        for (X, Y, D, C, pth) in groups[k]:
            if is_val:
                Xva.append(X); Yva.append(Y); Dva.append(D); Cva.append(C)
            else:
                Xtr.append(X); Ytr.append(Y); Dtr.append(D); Ctr.append(C)
            split_info.append({
                "exp_dir": pth,
                "group_key": group_by,
                "group_value": k,
                "split": "val" if is_val else "train",
                "n_total": int(len(X)),
                "n_train": int(len(X)) if not is_val else 0,
                "n_val": int(len(X)) if is_val else 0,
            })

    def _concat_or_empty(arrs, shape1):
        if arrs:
            return np.concatenate(arrs, axis=0)
        return np.empty((0, shape1), dtype=np.float32)

    dim = exps[0][0].shape[1] if exps[0][0].ndim == 2 else 1

    X_train = _concat_or_empty(Xtr, dim)
    Y_train = _concat_or_empty(Ytr, dim)
    D_train = _concat_or_empty(Dtr, 1).ravel() if Dtr else np.empty(0, dtype=np.float32)
    C_train = _concat_or_empty(Ctr, 1).ravel() if Ctr else np.empty(0, dtype=np.float32)

    X_val = _concat_or_empty(Xva, dim)
    Y_val = _concat_or_empty(Yva, dim)
    D_val = _concat_or_empty(Dva, 1).ravel() if Dva else np.empty(0, dtype=np.float32)
    C_val = _concat_or_empty(Cva, 1).ravel() if Cva else np.empty(0, dtype=np.float32)

    df_split = pd.DataFrame(split_info)
    return X_train, Y_train, D_train, C_train, X_val, Y_val, D_val, C_val, df_split


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_split(
    exps: List[Experiment],
    strategy: str = "per_experiment",
    val_split: float = 0.2,
    seed: int = 42,
    *,
    group_by: str = "distance_m",
    within_exp_shuffle: bool = False,
) -> SplitResult:
    """Apply the named split strategy. Raises ValueError on unknown strategy.

    Parameters
    ----------
    exps : list of experiment tuples
    strategy : ``"per_experiment"`` | ``"grouped"``
    val_split, seed : passed through to the strategy
    group_by : only used by ``"grouped"``
    within_exp_shuffle : only used by ``"per_experiment"``
    """
    if strategy == "per_experiment":
        return split_per_experiment(
            exps, val_split=val_split, seed=seed,
            within_exp_shuffle=within_exp_shuffle,
        )
    if strategy == "grouped":
        return split_grouped(
            exps, val_split=val_split, group_by=group_by,
        )
    raise ValueError(
        f"Unknown split strategy {strategy!r}. "
        f"Choose from: 'per_experiment', 'grouped'."
    )
