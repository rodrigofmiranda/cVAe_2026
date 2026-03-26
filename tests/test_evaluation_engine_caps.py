# -*- coding: utf-8 -*-
"""Tests for per-experiment caps in the evaluation engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.windowing import build_windows_from_split_arrays
from src.evaluation.engine import _apply_split_caps_for_evaluation


def _make_split_arrays(n_exps: int = 3, n_per_exp: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_tr_each = int(n_per_exp * 0.8)
    n_va_each = n_per_exp - n_tr_each

    X_tr_parts, Y_tr_parts, D_tr_parts, C_tr_parts = [], [], [], []
    X_va_parts, Y_va_parts, D_va_parts, C_va_parts = [], [], [], []
    rows = []

    for i in range(n_exps):
        X_tr_parts.append(rng.normal(size=(n_tr_each, 2)).astype(np.float32))
        Y_tr_parts.append(rng.normal(size=(n_tr_each, 2)).astype(np.float32))
        D_tr_parts.append(np.full((n_tr_each, 1), float(i + 1), dtype=np.float32))
        C_tr_parts.append(np.full((n_tr_each, 1), 100.0 + i * 50, dtype=np.float32))

        X_va_parts.append(rng.normal(size=(n_va_each, 2)).astype(np.float32))
        Y_va_parts.append(rng.normal(size=(n_va_each, 2)).astype(np.float32))
        D_va_parts.append(np.full((n_va_each, 1), float(i + 1), dtype=np.float32))
        C_va_parts.append(np.full((n_va_each, 1), 100.0 + i * 50, dtype=np.float32))

        rows.append({"exp_dir": f"/fake/exp_{i}", "n_train": n_tr_each, "n_val": n_va_each})

    return (
        np.concatenate(X_tr_parts, axis=0),
        np.concatenate(Y_tr_parts, axis=0),
        np.concatenate(D_tr_parts, axis=0),
        np.concatenate(C_tr_parts, axis=0),
        np.concatenate(X_va_parts, axis=0),
        np.concatenate(Y_va_parts, axis=0),
        np.concatenate(D_va_parts, axis=0),
        np.concatenate(C_va_parts, axis=0),
        pd.DataFrame(rows),
    )


def test_apply_split_caps_for_evaluation_rewrites_df_split_for_seq_windowing():
    X_tr, Y_tr, D_tr, C_tr, X_va, Y_va, D_va, C_va, df_split = _make_split_arrays()

    out = _apply_split_caps_for_evaluation(
        X_train=X_tr,
        Y_train=Y_tr,
        D_train=D_tr,
        C_train=C_tr,
        X_val=X_va,
        Y_val=Y_va,
        D_val=D_va,
        C_val=C_va,
        df_split=df_split,
        split_mode="per_experiment",
        overrides={"max_samples_per_exp": 3, "max_val_samples_per_exp": 2},
    )
    X_tr_cap, Y_tr_cap, D_tr_cap, C_tr_cap, X_va_cap, Y_va_cap, D_va_cap, C_va_cap, df_post = out

    assert len(X_tr_cap) == 9
    assert len(X_va_cap) == 6
    assert int(df_post["n_train"].sum()) == len(X_tr_cap)
    assert int(df_post["n_val"].sum()) == len(X_va_cap)
    assert df_post["n_train"].tolist() == [3, 3, 3]
    assert df_post["n_val"].tolist() == [2, 2, 2]

    X_seq_tr, _, _, _, X_seq_va, _, _, _ = build_windows_from_split_arrays(
        X_tr_cap,
        Y_tr_cap,
        D_tr_cap,
        C_tr_cap,
        X_va_cap,
        Y_va_cap,
        D_va_cap,
        C_va_cap,
        df_split=df_post,
        window_size=7,
    )

    assert X_seq_tr.shape == (len(X_tr_cap), 7, 2)
    assert X_seq_va.shape == (len(X_va_cap), 7, 2)
