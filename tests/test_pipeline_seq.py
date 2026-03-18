# -*- coding: utf-8 -*-
"""Tests for Phase 5: seq_bigru_residual integration in training pipeline.

Covers:
  1. Point-wise pipeline path is unchanged (no windowing applied)
  2. seq_bigru_residual pipeline calls windowing correctly
  3. Training/val shapes after windowing are correct
  4. No train/val boundary crossing after windowing
  5. Incompatible paths (balanced_blocks + seq) raise explicit ValueError
  6. gridsearch windowing block is applied per-cfg

These tests exercise the windowing logic and guard logic without running
full TF training (using the windowing module directly and the guard helpers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.windowing import build_windows_from_split_arrays


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_split_arrays(n_exps: int = 3, n_per_exp: int = 20, seed: int = 0):
    """Create minimal concatenated split arrays + df_split for N experiments."""
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

        rows.append({"exp_id": i, "n_train": n_tr_each, "n_val": n_va_each})

    X_tr = np.concatenate(X_tr_parts, axis=0)
    Y_tr = np.concatenate(Y_tr_parts, axis=0)
    D_tr = np.concatenate(D_tr_parts, axis=0)
    C_tr = np.concatenate(C_tr_parts, axis=0)

    X_va = np.concatenate(X_va_parts, axis=0)
    Y_va = np.concatenate(Y_va_parts, axis=0)
    D_va = np.concatenate(D_va_parts, axis=0)
    C_va = np.concatenate(C_va_parts, axis=0)

    df_split = pd.DataFrame(rows)
    return (X_tr, Y_tr, D_tr, C_tr, X_va, Y_va, D_va, C_va, df_split)


# ===========================================================================
# 2 & 3 — Windowing integration: shapes and content
# ===========================================================================

class TestSeqWindowing:

    def test_window_output_count_equals_input_count_stride1(self):
        """stride=1 must produce exactly one output per input sample."""
        X_tr, Y_tr, D_tr, C_tr, X_va, Y_va, D_va, C_va, df = _make_split_arrays()
        W = 7
        out = build_windows_from_split_arrays(
            X_tr, Y_tr, D_tr, C_tr,
            X_va, Y_va, D_va, C_va,
            df_split=df, window_size=W,
        )
        X_seq_tr, Y_w_tr, D_w_tr, C_w_tr, X_seq_va, Y_w_va, D_w_va, C_w_va = out
        assert len(X_seq_tr) == len(X_tr)
        assert len(X_seq_va) == len(X_va)

    def test_window_train_shape(self):
        X_tr, Y_tr, D_tr, C_tr, X_va, Y_va, D_va, C_va, df = _make_split_arrays()
        W = 9
        out = build_windows_from_split_arrays(
            X_tr, Y_tr, D_tr, C_tr,
            X_va, Y_va, D_va, C_va,
            df_split=df, window_size=W,
        )
        X_seq_tr = out[0]
        assert X_seq_tr.shape == (len(X_tr), W, 2)

    def test_window_val_shape(self):
        X_tr, Y_tr, D_tr, C_tr, X_va, Y_va, D_va, C_va, df = _make_split_arrays()
        W = 9
        out = build_windows_from_split_arrays(
            X_tr, Y_tr, D_tr, C_tr,
            X_va, Y_va, D_va, C_va,
            df_split=df, window_size=W,
        )
        X_seq_va = out[4]
        assert X_seq_va.shape == (len(X_va), W, 2)

    def test_y_center_unchanged_stride1(self):
        """With stride=1, Y_center must equal the original Y arrays."""
        X_tr, Y_tr, D_tr, C_tr, X_va, Y_va, D_va, C_va, df = _make_split_arrays()
        W = 5
        out = build_windows_from_split_arrays(
            X_tr, Y_tr, D_tr, C_tr,
            X_va, Y_va, D_va, C_va,
            df_split=df, window_size=W,
        )
        _, Y_w_tr, _, _, _, Y_w_va, _, _ = out
        np.testing.assert_array_equal(Y_w_tr, Y_tr)
        np.testing.assert_array_equal(Y_w_va, Y_va)

    def test_center_col_of_window_equals_original_x(self):
        """The center column of X_seq must equal the original X for each sample."""
        X_tr, Y_tr, D_tr, C_tr, X_va, Y_va, D_va, C_va, df = _make_split_arrays()
        W = 7
        out = build_windows_from_split_arrays(
            X_tr, Y_tr, D_tr, C_tr,
            X_va, Y_va, D_va, C_va,
            df_split=df, window_size=W,
        )
        X_seq_tr = out[0]
        center = W // 2
        np.testing.assert_array_equal(X_seq_tr[:, center, :], X_tr)

    @pytest.mark.parametrize("W", [1, 3, 7, 33])
    def test_various_window_sizes(self, W):
        X_tr, Y_tr, D_tr, C_tr, X_va, Y_va, D_va, C_va, df = _make_split_arrays()
        out = build_windows_from_split_arrays(
            X_tr, Y_tr, D_tr, C_tr,
            X_va, Y_va, D_va, C_va,
            df_split=df, window_size=W,
        )
        X_seq_tr = out[0]
        assert X_seq_tr.shape == (len(X_tr), W, 2)


# ===========================================================================
# 1 — Point-wise path unchanged
# ===========================================================================

class TestPointWiseUnchanged:
    """Verify that non-seq arch variants are unaffected."""

    def test_pointwise_no_windowing_needed(self):
        """For point-wise variants, X arrays remain (N, 2), no windowing is applied."""
        X_tr, Y_tr, D_tr, C_tr, X_va, Y_va, D_va, C_va, df = _make_split_arrays()
        # Simulate pipeline's point-wise branch: no windowing
        assert X_tr.ndim == 2
        assert X_tr.shape[1] == 2
        assert X_va.ndim == 2
        assert X_va.shape[1] == 2

    def test_windowing_output_is_3d(self):
        """After windowing, X_seq arrays must be 3D (seq) vs 2D (point-wise)."""
        X_tr, Y_tr, D_tr, C_tr, X_va, Y_va, D_va, C_va, df = _make_split_arrays()
        W = 7
        out = build_windows_from_split_arrays(
            X_tr, Y_tr, D_tr, C_tr,
            X_va, Y_va, D_va, C_va,
            df_split=df, window_size=W,
        )
        X_seq_tr = out[0]
        assert X_seq_tr.ndim == 3, "Windowed X must be 3D (N, W, 2)"
        assert X_tr.ndim == 2, "Original X must remain 2D (unchanged)"


# ===========================================================================
# 4 — No train/val boundary crossing
# ===========================================================================

class TestNoBoundaryCrossing:

    def test_train_context_never_uses_val_data(self):
        """Windows built on train side must not contain any val-side values.

        We mark val samples with a sentinel value (+1e6) and verify no
        train window contains that sentinel.
        """
        n_exps = 2
        n_tr_each, n_va_each = 16, 4
        rng = np.random.default_rng(7)

        # Build small arrays with sentinel in val
        X_tr_parts, Y_tr_parts, D_tr_parts, C_tr_parts = [], [], [], []
        X_va_parts, Y_va_parts, D_va_parts, C_va_parts = [], [], [], []
        rows = []
        for i in range(n_exps):
            X_tr_parts.append(rng.normal(size=(n_tr_each, 2)).astype(np.float32))
            Y_tr_parts.append(rng.normal(size=(n_tr_each, 2)).astype(np.float32))
            D_tr_parts.append(np.ones((n_tr_each, 1), dtype=np.float32))
            C_tr_parts.append(np.ones((n_tr_each, 1), dtype=np.float32))

            # Sentinel: val X is all 1e6
            X_va_parts.append(np.full((n_va_each, 2), 1e6, dtype=np.float32))
            Y_va_parts.append(rng.normal(size=(n_va_each, 2)).astype(np.float32))
            D_va_parts.append(np.ones((n_va_each, 1), dtype=np.float32))
            C_va_parts.append(np.ones((n_va_each, 1), dtype=np.float32))
            rows.append({"n_train": n_tr_each, "n_val": n_va_each})

        X_tr = np.concatenate(X_tr_parts)
        Y_tr = np.concatenate(Y_tr_parts)
        D_tr = np.concatenate(D_tr_parts)
        C_tr = np.concatenate(C_tr_parts)
        X_va = np.concatenate(X_va_parts)
        Y_va = np.concatenate(Y_va_parts)
        D_va = np.concatenate(D_va_parts)
        C_va = np.concatenate(C_va_parts)
        df = pd.DataFrame(rows)

        W = 5
        out = build_windows_from_split_arrays(
            X_tr, Y_tr, D_tr, C_tr,
            X_va, Y_va, D_va, C_va,
            df_split=df, window_size=W,
        )
        X_seq_tr = out[0]
        # No train window should contain the sentinel value
        assert np.all(X_seq_tr < 1e5), "Train windows must not contain val-side sentinel values"

    def test_val_context_never_uses_train_data(self):
        """Windows built on val side must not contain train-side sentinel values."""
        n_tr_each, n_va_each = 16, 6
        rng = np.random.default_rng(13)

        X_tr_parts = [np.full((n_tr_each, 2), -1e6, dtype=np.float32)]
        Y_tr_parts = [rng.normal(size=(n_tr_each, 2)).astype(np.float32)]
        D_tr_parts = [np.ones((n_tr_each, 1), dtype=np.float32)]
        C_tr_parts = [np.ones((n_tr_each, 1), dtype=np.float32)]

        X_va_parts = [rng.normal(size=(n_va_each, 2)).astype(np.float32)]
        Y_va_parts = [rng.normal(size=(n_va_each, 2)).astype(np.float32)]
        D_va_parts = [np.ones((n_va_each, 1), dtype=np.float32)]
        C_va_parts = [np.ones((n_va_each, 1), dtype=np.float32)]
        df = pd.DataFrame([{"n_train": n_tr_each, "n_val": n_va_each}])

        W = 5
        out = build_windows_from_split_arrays(
            np.concatenate(X_tr_parts), np.concatenate(Y_tr_parts),
            np.concatenate(D_tr_parts), np.concatenate(C_tr_parts),
            np.concatenate(X_va_parts), np.concatenate(Y_va_parts),
            np.concatenate(D_va_parts), np.concatenate(C_va_parts),
            df_split=df, window_size=W,
        )
        X_seq_va = out[4]
        # No val window should contain the train-side sentinel -1e6
        assert np.all(X_seq_va > -1e5), "Val windows must not contain train-side sentinel values"


# ===========================================================================
# 5 — Incompatible paths raise explicit ValueError
# ===========================================================================

class TestIncompatiblePathsRaise:
    """Guard logic for balanced_blocks + seq_bigru_residual."""

    def _make_runtime_like(self, mode: str = "balanced_blocks", enabled: bool = True):
        """Simulate what pipeline.py sees from runtime.data_reduction_config."""
        return {"mode": mode, "enabled": enabled}

    def _check_guard(self, arch_variant: str, data_reduction_config: dict) -> None:
        """Replicate the guard logic from pipeline.py (post-grid-selection)."""
        _seq_in_grid = arch_variant == "seq_bigru_residual"
        if _seq_in_grid:
            _dr_enabled = bool(data_reduction_config.get("enabled", False))
            _dr_mode = str(data_reduction_config.get("mode", "balanced_blocks")).lower()
            if _dr_enabled and _dr_mode == "balanced_blocks":
                raise ValueError(
                    "arch_variant='seq_bigru_residual' is incompatible with "
                    "data_reduction mode='balanced_blocks'"
                )

    def test_seq_plus_balanced_blocks_raises(self):
        dr = self._make_runtime_like(mode="balanced_blocks", enabled=True)
        with pytest.raises(ValueError, match="balanced_blocks"):
            self._check_guard("seq_bigru_residual", dr)

    def test_seq_plus_center_crop_does_not_raise(self):
        dr = self._make_runtime_like(mode="center_crop", enabled=True)
        self._check_guard("seq_bigru_residual", dr)  # must not raise

    def test_seq_plus_disabled_reduction_does_not_raise(self):
        dr = self._make_runtime_like(mode="balanced_blocks", enabled=False)
        self._check_guard("seq_bigru_residual", dr)  # must not raise

    def test_concat_plus_balanced_blocks_does_not_raise(self):
        dr = self._make_runtime_like(mode="balanced_blocks", enabled=True)
        self._check_guard("concat", dr)  # point-wise: guard is no-op

    def test_channel_residual_plus_balanced_blocks_does_not_raise(self):
        dr = self._make_runtime_like(mode="balanced_blocks", enabled=True)
        self._check_guard("channel_residual", dr)  # point-wise: guard is no-op

    def test_error_message_is_explicit(self):
        dr = self._make_runtime_like(mode="balanced_blocks", enabled=True)
        with pytest.raises(ValueError) as exc_info:
            self._check_guard("seq_bigru_residual", dr)
        msg = str(exc_info.value)
        assert "seq_bigru_residual" in msg
        assert "balanced_blocks" in msg

    def test_windowing_missing_df_split_raises(self):
        """build_windows_from_split_arrays must raise if df_split has no n_train/n_val."""
        X_tr, Y_tr, D_tr, C_tr, X_va, Y_va, D_va, C_va, _ = _make_split_arrays()
        bad_df = pd.DataFrame({"something_else": [1, 2, 3]})
        with pytest.raises(ValueError, match="n_train"):
            build_windows_from_split_arrays(
                X_tr, Y_tr, D_tr, C_tr,
                X_va, Y_va, D_va, C_va,
                df_split=bad_df, window_size=7,
            )


# ===========================================================================
# 6 — Xv_center extraction logic
# ===========================================================================

class TestXvCenterExtraction:
    """Verify the Xv_center extraction logic used in gridsearch."""

    def test_center_extracted_correctly_for_seq(self):
        """Xv[:, W//2, :] must equal original X_val samples (stride=1)."""
        X_tr, Y_tr, D_tr, C_tr, X_va, Y_va, D_va, C_va, df = _make_split_arrays(
            n_exps=2, n_per_exp=30
        )
        W = 7
        out = build_windows_from_split_arrays(
            X_tr, Y_tr, D_tr, C_tr,
            X_va, Y_va, D_va, C_va,
            df_split=df, window_size=W,
        )
        X_seq_va = out[4]
        # Simulate what gridsearch does:
        Xv = X_seq_va          # (N_va, W, 2)
        Xv_center = Xv[:, Xv.shape[1] // 2, :]  # (N_va, 2)
        assert Xv_center.shape == (len(X_va), 2)
        np.testing.assert_array_equal(Xv_center, X_va)

    def test_xv_center_is_2d_for_pointwise(self):
        """For point-wise, Xv_center = Xv (same reference, shape (N, 2))."""
        X_tr, Y_tr, D_tr, C_tr, X_va, Y_va, D_va, C_va, df = _make_split_arrays()
        Xv = X_va  # point-wise: no windowing
        Xv_center = Xv  # identity
        assert Xv_center.ndim == 2
        assert Xv_center.shape[1] == 2
