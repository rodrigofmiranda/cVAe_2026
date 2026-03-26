# -*- coding: utf-8 -*-
"""Tests for src.data.windowing.

Covers:
  1. Shape correctness
  2. Center-sample target alignment
  3. Edge padding (left and right)
  4. Stride behaviour
  5. No cross-experiment boundary leakage
  6. No cross-train/val boundary leakage
  7. One output per original sample when stride=1
  8. Input validation (even window_size, bad df_split)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splits import apply_caps_to_df_split
from src.data.windowing import (
    build_windows_single_experiment,
    build_windows_from_split_arrays,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random(N: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 2)).astype(np.float32)
    Y = rng.normal(size=(N, 2)).astype(np.float32)
    D = np.full((N, 1), 1.0, dtype=np.float32)
    C = np.full((N, 1), 300.0, dtype=np.float32)
    return X, Y, D, C


def _make_split(n_exps: int = 3, n_per_exp: int = 20, seed: int = 0):
    """Build mock concatenated split arrays for *n_exps* experiments."""
    rng = np.random.default_rng(seed)
    n_tr = int(n_per_exp * 0.8)
    n_va = n_per_exp - n_tr

    tr_X, tr_Y, tr_D, tr_C = [], [], [], []
    va_X, va_Y, va_D, va_C = [], [], [], []
    rows = []

    for i in range(n_exps):
        dist = float(i + 1)
        Xe = rng.normal(size=(n_per_exp, 2)).astype(np.float32)
        Ye = rng.normal(size=(n_per_exp, 2)).astype(np.float32)
        De = np.full(n_per_exp, dist, dtype=np.float32)
        Ce = np.full(n_per_exp, 300.0, dtype=np.float32)

        tr_X.append(Xe[:n_tr]);  tr_Y.append(Ye[:n_tr])
        tr_D.append(De[:n_tr]);  tr_C.append(Ce[:n_tr])
        va_X.append(Xe[n_tr:]);  va_Y.append(Ye[n_tr:])
        va_D.append(De[n_tr:]);  va_C.append(Ce[n_tr:])
        rows.append({"exp_dir": f"/fake/exp_{i}", "n_train": n_tr, "n_val": n_va})

    return (
        np.concatenate(tr_X), np.concatenate(tr_Y),
        np.concatenate(tr_D), np.concatenate(tr_C),
        np.concatenate(va_X), np.concatenate(va_Y),
        np.concatenate(va_D), np.concatenate(va_C),
        pd.DataFrame(rows),
        n_tr, n_va, n_exps,
    )


# ===========================================================================
# build_windows_single_experiment
# ===========================================================================

class TestBuildWindowsSingleExperiment:

    # 1. Shape correctness
    def test_output_shapes_stride1(self):
        N, W = 50, 7
        X, Y, D, C = _make_random(N)
        X_seq, Y_c, D_c, C_c = build_windows_single_experiment(X, Y, D, C, window_size=W)
        assert X_seq.shape == (N, W, 2), f"Expected ({N},{W},2), got {X_seq.shape}"
        assert Y_c.shape  == (N, 2)
        assert D_c.shape  == (N, 1)
        assert C_c.shape  == (N, 1)

    def test_output_shapes_various_sizes(self):
        for N, W in [(1, 1), (1, 3), (5, 5), (100, 33)]:
            X, Y, D, C = _make_random(N)
            X_seq, Y_c, D_c, C_c = build_windows_single_experiment(X, Y, D, C, window_size=W)
            assert X_seq.shape == (N, W, 2)
            assert Y_c.shape == (N, 2)

    # 2. Center-sample target alignment
    def test_center_matches_x(self):
        """X_seq[i, half] must equal X[i] for every i."""
        N, W = 30, 9
        half = W // 2
        X, Y, D, C = _make_random(N)
        X_seq, Y_c, _, _ = build_windows_single_experiment(X, Y, D, C, window_size=W)
        for i in range(N):
            np.testing.assert_array_equal(
                X_seq[i, half], X[i],
                err_msg=f"Center X mismatch at i={i}",
            )

    def test_center_matches_y(self):
        """Y_center[i] must equal Y[i] for every i."""
        N, W = 30, 5
        X, Y, D, C = _make_random(N)
        _, Y_c, _, _ = build_windows_single_experiment(X, Y, D, C, window_size=W)
        np.testing.assert_array_equal(Y_c, Y)

    # 3. Edge padding
    def test_left_edge_padding(self):
        """First window: positions before the sequence are padded with X[0]."""
        N, W = 10, 5
        half = W // 2
        X, Y, D, C = _make_random(N)
        X_seq, _, _, _ = build_windows_single_experiment(X, Y, D, C, window_size=W, pad_mode="edge")
        for k in range(half):
            np.testing.assert_array_equal(
                X_seq[0, k], X[0],
                err_msg=f"Left pad mismatch at position k={k}",
            )

    def test_right_edge_padding(self):
        """Last window: positions after the sequence are padded with X[-1]."""
        N, W = 10, 5
        half = W // 2
        X, Y, D, C = _make_random(N)
        X_seq, _, _, _ = build_windows_single_experiment(X, Y, D, C, window_size=W, pad_mode="edge")
        for k in range(half + 1, W):
            np.testing.assert_array_equal(
                X_seq[-1, k], X[-1],
                err_msg=f"Right pad mismatch at position k={k}",
            )

    # 4. Stride
    def test_stride_reduces_output_count(self):
        """stride > 1 reduces N_out to len(range(0, N, stride))."""
        N, W, stride = 100, 5, 4
        X, Y, D, C = _make_random(N)
        X_seq, Y_c, D_c, C_c = build_windows_single_experiment(
            X, Y, D, C, window_size=W, stride=stride
        )
        expected = len(range(0, N, stride))
        assert len(X_seq) == expected
        assert len(Y_c)   == expected
        assert X_seq.shape == (expected, W, 2)

    def test_stride_center_alignment(self):
        """Y_center[k] == Y[k * stride] for all k."""
        N, W, stride = 40, 5, 3
        X, Y, D, C = _make_random(N)
        _, Y_c, _, _ = build_windows_single_experiment(X, Y, D, C, window_size=W, stride=stride)
        for k, i in enumerate(range(0, N, stride)):
            np.testing.assert_array_equal(
                Y_c[k], Y[i],
                err_msg=f"Stride center mismatch at k={k} (i={i})",
            )

    # 6. One output per sample with stride=1
    def test_one_output_per_sample_stride1(self):
        N, W = 100, 5
        X, Y, D, C = _make_random(N)
        X_seq, Y_c, _, _ = build_windows_single_experiment(X, Y, D, C, window_size=W, stride=1)
        assert len(X_seq) == N
        assert len(Y_c)   == N

    # Input validation
    def test_even_window_size_raises(self):
        X, Y, D, C = _make_random(10)
        with pytest.raises(ValueError, match="odd"):
            build_windows_single_experiment(X, Y, D, C, window_size=4)

    def test_window_size_1_is_identity(self):
        """window_size=1 → X_seq[:, 0, :] == X."""
        N = 20
        X, Y, D, C = _make_random(N)
        X_seq, Y_c, _, _ = build_windows_single_experiment(X, Y, D, C, window_size=1)
        assert X_seq.shape == (N, 1, 2)
        np.testing.assert_array_equal(X_seq[:, 0, :], X)
        np.testing.assert_array_equal(Y_c, Y)

    def test_single_sample(self):
        """N=1: all positions in the window equal the single sample (edge pad)."""
        W = 5
        X = np.array([[1.0, 2.0]], dtype=np.float32)
        Y = np.array([[3.0, 4.0]], dtype=np.float32)
        D = np.array([[1.0]], dtype=np.float32)
        C = np.array([[300.0]], dtype=np.float32)
        X_seq, Y_c, _, _ = build_windows_single_experiment(X, Y, D, C, window_size=W)
        assert X_seq.shape == (1, W, 2)
        np.testing.assert_array_equal(X_seq[0], np.tile(X[0], (W, 1)))
        np.testing.assert_array_equal(Y_c[0], Y[0])

    def test_empty_input(self):
        """N=0: returns empty arrays with correct shapes."""
        W = 5
        X = np.empty((0, 2), dtype=np.float32)
        Y = np.empty((0, 2), dtype=np.float32)
        D = np.empty((0, 1), dtype=np.float32)
        C = np.empty((0, 1), dtype=np.float32)
        X_seq, Y_c, D_c, C_c = build_windows_single_experiment(X, Y, D, C, window_size=W)
        assert X_seq.shape == (0, W, 2)
        assert Y_c.shape   == (0, 2)
        assert D_c.shape   == (0, 1)
        assert C_c.shape   == (0, 1)


# ===========================================================================
# build_windows_from_split_arrays
# ===========================================================================

class TestBuildWindowsFromSplitArrays:

    # 1. Shape correctness
    def test_output_shapes_stride1(self):
        """stride=1 → output lengths match input lengths."""
        *args, df_split, n_tr, n_va, n_exps = _make_split()
        W = 5
        (X_seq_tr, Y_tr_out, D_tr_out, C_tr_out,
         X_seq_va, Y_va_out, D_va_out, C_va_out) = build_windows_from_split_arrays(
            *args[:8], df_split, window_size=W,
        )
        total_tr = n_tr * n_exps
        total_va = n_va * n_exps
        assert X_seq_tr.shape == (total_tr, W, 2)
        assert X_seq_va.shape == (total_va, W, 2)
        assert Y_tr_out.shape == (total_tr, 2)
        assert Y_va_out.shape == (total_va, 2)
        assert D_tr_out.shape == (total_tr, 1)
        assert C_va_out.shape == (total_va, 1)

    # 5. No cross-experiment boundary leakage
    def test_no_cross_experiment_boundary(self):
        """Windows from exp1 contain only exp1 data; same for exp2."""
        n_tr, n_va, W = 10, 4, 5

        # Exp1: X all 1.0; Exp2: X all 99.0
        X1 = np.ones((n_tr + n_va, 2), dtype=np.float32) * 1.0
        Y1 = np.ones((n_tr + n_va, 2), dtype=np.float32)
        D1 = np.ones(n_tr + n_va, dtype=np.float32)
        C1 = np.full(n_tr + n_va, 100.0, dtype=np.float32)

        X2 = np.ones((n_tr + n_va, 2), dtype=np.float32) * 99.0
        Y2 = np.ones((n_tr + n_va, 2), dtype=np.float32) * 2.0
        D2 = np.ones(n_tr + n_va, dtype=np.float32) * 2.0
        C2 = np.full(n_tr + n_va, 200.0, dtype=np.float32)

        X_tr = np.concatenate([X1[:n_tr], X2[:n_tr]])
        Y_tr = np.concatenate([Y1[:n_tr], Y2[:n_tr]])
        D_tr = np.concatenate([D1[:n_tr], D2[:n_tr]])
        C_tr = np.concatenate([C1[:n_tr], C2[:n_tr]])

        X_va = np.concatenate([X1[n_tr:], X2[n_tr:]])
        Y_va = np.concatenate([Y1[n_tr:], Y2[n_tr:]])
        D_va = np.concatenate([D1[n_tr:], D2[n_tr:]])
        C_va = np.concatenate([C1[n_tr:], C2[n_tr:]])

        df_split = pd.DataFrame([
            {"exp_dir": "/e1", "n_train": n_tr, "n_val": n_va},
            {"exp_dir": "/e2", "n_train": n_tr, "n_val": n_va},
        ])

        X_seq_tr, _, _, _, X_seq_va, _, _, _ = build_windows_from_split_arrays(
            X_tr, Y_tr, D_tr, C_tr,
            X_va, Y_va, D_va, C_va,
            df_split, window_size=W,
        )

        # First n_tr windows → exp1 train → must be all 1.0
        assert np.all(X_seq_tr[:n_tr] == 1.0), "Exp2 values leaked into exp1 train windows"
        # Last n_tr windows → exp2 train → must be all 99.0
        assert np.all(X_seq_tr[n_tr:] == 99.0), "Exp1 values leaked into exp2 train windows"
        # Validation side
        assert np.all(X_seq_va[:n_va] == 1.0), "Exp2 values leaked into exp1 val windows"
        assert np.all(X_seq_va[n_va:] == 99.0), "Exp1 values leaked into exp2 val windows"

    # No cross-train/val boundary leakage
    def test_no_train_val_boundary_leakage(self):
        """Train windows contain only train-side data; val windows only val-side data."""
        n_tr, n_va, W = 10, 5, 7

        # Train side: all 1.0; val side: all 99.0
        X_tr = np.ones((n_tr, 2), dtype=np.float32) * 1.0
        X_va = np.ones((n_va, 2), dtype=np.float32) * 99.0
        Y_tr = X_tr.copy()
        Y_va = X_va.copy()
        D_tr = np.ones(n_tr, dtype=np.float32)
        C_tr = np.full(n_tr, 100.0, dtype=np.float32)
        D_va = np.ones(n_va, dtype=np.float32)
        C_va = np.full(n_va, 100.0, dtype=np.float32)

        df_split = pd.DataFrame([{"exp_dir": "/e", "n_train": n_tr, "n_val": n_va}])

        X_seq_tr, _, _, _, X_seq_va, _, _, _ = build_windows_from_split_arrays(
            X_tr, Y_tr, D_tr, C_tr,
            X_va, Y_va, D_va, C_va,
            df_split, window_size=W,
        )

        assert np.all(X_seq_tr == 1.0),  "Val-side data (99.0) leaked into train windows"
        assert np.all(X_seq_va == 99.0), "Train-side data (1.0) leaked into val windows"

    # 6. One output per original sample with stride=1
    def test_preserve_one_output_per_sample_stride1(self):
        *args, df_split, n_tr, n_va, n_exps = _make_split(n_exps=4, n_per_exp=30)
        W = 9
        (X_seq_tr, Y_tr_out, _, _,
         X_seq_va, Y_va_out, _, _) = build_windows_from_split_arrays(
            *args[:8], df_split, window_size=W, stride=1,
        )
        X_tr = args[0]
        X_va = args[4]
        assert len(X_seq_tr) == len(X_tr), "Train output count changed with stride=1"
        assert len(X_seq_va) == len(X_va), "Val output count changed with stride=1"

    # Stride propagation
    def test_stride_propagates_to_output_count(self):
        *args, df_split, n_tr, n_va, n_exps = _make_split(n_exps=2, n_per_exp=20)
        W, stride = 5, 2
        (X_seq_tr, _, _, _,
         X_seq_va, _, _, _) = build_windows_from_split_arrays(
            *args[:8], df_split, window_size=W, stride=stride,
        )
        expected_tr = sum(len(range(0, n_tr, stride)) for _ in range(n_exps))
        expected_va = sum(len(range(0, n_va, stride)) for _ in range(n_exps))
        assert len(X_seq_tr) == expected_tr
        assert len(X_seq_va) == expected_va

    # Error handling
    def test_missing_n_train_column_raises(self):
        X = np.ones((10, 2), dtype=np.float32)
        df_bad = pd.DataFrame([{"exp_dir": "/e", "n_val": 2}])
        with pytest.raises(ValueError, match="n_train"):
            build_windows_from_split_arrays(
                X, X, X[:, :1], X[:, :1],
                X, X, X[:, :1], X[:, :1],
                df_bad, window_size=3,
            )

    def test_missing_n_val_column_raises(self):
        X = np.ones((10, 2), dtype=np.float32)
        df_bad = pd.DataFrame([{"exp_dir": "/e", "n_train": 8}])
        with pytest.raises(ValueError, match="n_val"):
            build_windows_from_split_arrays(
                X, X, X[:, :1], X[:, :1],
                X, X, X[:, :1], X[:, :1],
                df_bad, window_size=3,
            )

    def test_length_mismatch_with_df_split_raises_after_cap(self):
        X_tr = np.arange(16, dtype=np.float32).reshape(-1, 2)
        Y_tr = X_tr.copy()
        D_tr = np.ones((8, 1), dtype=np.float32)
        C_tr = np.ones((8, 1), dtype=np.float32)
        X_va = np.arange(8, dtype=np.float32).reshape(-1, 2)
        Y_va = X_va.copy()
        D_va = np.ones((4, 1), dtype=np.float32)
        C_va = np.ones((4, 1), dtype=np.float32)
        df_split = pd.DataFrame(
            [
                {"exp_dir": "/e1", "n_train": 6, "n_val": 2},
                {"exp_dir": "/e2", "n_train": 6, "n_val": 2},
            ]
        )

        # Simulate post-cap arrays, but keep stale pre-cap df_split.
        with pytest.raises(ValueError, match="post-cap counts before windowing"):
            build_windows_from_split_arrays(
                X_tr[:8], Y_tr[:8], D_tr[:8], C_tr[:8],
                X_va[:4], Y_va[:4], D_va[:4], C_va[:4],
                df_split, window_size=3,
            )

    def test_corrected_df_split_after_cap_preserves_no_boundary_leak(self):
        n_tr_pre, n_va_pre, W = 6, 3, 5
        X1 = np.ones((n_tr_pre, 2), dtype=np.float32) * 1.0
        X2 = np.ones((n_tr_pre, 2), dtype=np.float32) * 99.0
        Y1 = X1.copy()
        Y2 = X2.copy()
        D1 = np.ones((n_tr_pre, 1), dtype=np.float32) * 0.8
        D2 = np.ones((n_tr_pre, 1), dtype=np.float32) * 1.0
        C1 = np.ones((n_tr_pre, 1), dtype=np.float32) * 100.0
        C2 = np.ones((n_tr_pre, 1), dtype=np.float32) * 300.0
        Xva1 = np.ones((n_va_pre, 2), dtype=np.float32) * 1.0
        Xva2 = np.ones((n_va_pre, 2), dtype=np.float32) * 99.0
        Yva1 = Xva1.copy()
        Yva2 = Xva2.copy()
        Dva1 = np.ones((n_va_pre, 1), dtype=np.float32) * 0.8
        Dva2 = np.ones((n_va_pre, 1), dtype=np.float32) * 1.0
        Cva1 = np.ones((n_va_pre, 1), dtype=np.float32) * 100.0
        Cva2 = np.ones((n_va_pre, 1), dtype=np.float32) * 300.0

        X_tr = np.concatenate([X1[:3], X2[:3]], axis=0)
        Y_tr = np.concatenate([Y1[:3], Y2[:3]], axis=0)
        D_tr = np.concatenate([D1[:3], D2[:3]], axis=0)
        C_tr = np.concatenate([C1[:3], C2[:3]], axis=0)
        X_va = np.concatenate([Xva1[:2], Xva2[:2]], axis=0)
        Y_va = np.concatenate([Yva1[:2], Yva2[:2]], axis=0)
        D_va = np.concatenate([Dva1[:2], Dva2[:2]], axis=0)
        C_va = np.concatenate([Cva1[:2], Cva2[:2]], axis=0)

        df_split = pd.DataFrame(
            [
                {"exp_dir": "/e1", "n_train": 6, "n_val": 3},
                {"exp_dir": "/e2", "n_train": 6, "n_val": 3},
            ]
        )
        df_train_cap = pd.DataFrame(
            [
                {"exp_dir": "/e1", "n_train_before": 6, "n_train_after": 3},
                {"exp_dir": "/e2", "n_train_before": 6, "n_train_after": 3},
            ]
        )
        df_val_cap = pd.DataFrame(
            [
                {"exp_dir": "/e1", "n_val_before": 3, "n_val_after": 2},
                {"exp_dir": "/e2", "n_val_before": 3, "n_val_after": 2},
            ]
        )
        df_post = apply_caps_to_df_split(df_split, df_train_cap=df_train_cap, df_val_cap=df_val_cap)

        X_seq_tr, _, _, _, X_seq_va, _, _, _ = build_windows_from_split_arrays(
            X_tr, Y_tr, D_tr, C_tr,
            X_va, Y_va, D_va, C_va,
            df_post, window_size=W,
        )

        assert np.all(X_seq_tr[:3] == 1.0)
        assert np.all(X_seq_tr[3:] == 99.0)
        assert np.all(X_seq_va[:2] == 1.0)
        assert np.all(X_seq_va[2:] == 99.0)
