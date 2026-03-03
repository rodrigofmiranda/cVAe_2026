# -*- coding: utf-8 -*-
"""
src/data/splits.py — Per-experiment temporal split (head_tail).

Extracted from cvae_TRAIN_documented.py and analise_cvae_reviewed.py
(Commit 3D).  Both monoliths contained identical copies of
``split_train_val_per_experiment``; this module is the single source
of truth.
"""

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
