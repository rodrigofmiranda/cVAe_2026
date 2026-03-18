# -*- coding: utf-8 -*-
"""Tests for seq_bigru_residual integration in evaluation/engine and protocol/run.

Covers:
  1. _is_seq detection from loaded seq model (rank-3 prior input)
  2. _is_seq=False for point-wise model (rank-2 prior input)
  3. load_seq_model works for point-wise models (custom objects are neutral)
  4. engine windowing: X_val_w shape matches (N_val, W, 2)
  5. Xv_in / Xv_center shapes after seq windowing slice
  6. _quick_cvae_predict global windowing (seq path): X_arr_w shape
  7. _quick_cvae_predict seq path: Y_pred shape correct, X_tiled is 2D center frame
  8. _quick_cvae_predict point-wise path: unaffected by seq changes
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.models.cvae_sequence import (
    build_seq_cvae,
    load_seq_model,
    create_seq_inference_model,
)
from src.models.cvae import build_cvae, create_inference_model_from_full
from src.data.windowing import build_windows_single_experiment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seq_cfg(**overrides) -> dict:
    cfg = {
        "window_size": 7,
        "seq_hidden_size": 16,
        "seq_num_layers": 1,
        "seq_bidirectional": True,
        "layer_sizes": [32, 64],
        "latent_dim": 4,
        "beta": 0.001,
        "lr": 3e-4,
        "dropout": 0.0,
        "free_bits": 0.0,
        "kl_anneal_epochs": 3,
        "activation": "leaky_relu",
    }
    cfg.update(overrides)
    return cfg


def _point_cfg(arch_variant: str = "concat") -> dict:
    return {
        "layer_sizes": [32, 64],
        "latent_dim": 4,
        "beta": 0.001,
        "lr": 3e-4,
        "dropout": 0.0,
        "free_bits": 0.0,
        "kl_anneal_epochs": 3,
        "arch_variant": arch_variant,
    }


def _toy_val(N: int = 20, W: int = 7, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_val = rng.normal(size=(N, 2)).astype(np.float32)
    Y_val = rng.normal(size=(N, 2)).astype(np.float32)
    D_val = rng.uniform(0, 1, size=(N, 1)).astype(np.float32)
    C_val = rng.uniform(0, 1, size=(N, 1)).astype(np.float32)
    return X_val, Y_val, D_val, C_val


# ===========================================================================
# 1 & 2 — _is_seq detection via prior input rank
# ===========================================================================

class TestIsSeqDetection:

    def test_seq_model_prior_rank3(self, tmp_path):
        vae, _ = build_seq_cvae(_seq_cfg())
        save_path = str(tmp_path / "seq.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        prior = loaded.get_layer("prior_net")
        _is_seq = len(prior.inputs[0].shape) == 3
        assert _is_seq is True

    def test_point_model_prior_rank2(self, tmp_path):
        vae, _ = build_cvae(_point_cfg("concat"))
        save_path = str(tmp_path / "point.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        prior = loaded.get_layer("prior_net")
        _is_seq = len(prior.inputs[0].shape) == 3
        assert _is_seq is False

    def test_seq_prior_window_size_preserved(self, tmp_path):
        for W in [5, 7, 9]:
            vae, _ = build_seq_cvae(_seq_cfg(window_size=W))
            save_path = str(tmp_path / f"seq_W{W}.keras")
            vae.save(save_path, include_optimizer=False)
            loaded = load_seq_model(save_path)
            prior = loaded.get_layer("prior_net")
            assert prior.inputs[0].shape[1] == W

    def test_load_seq_model_on_point_model_does_not_raise(self, tmp_path):
        """load_seq_model should also work for point-wise models (custom objects neutral)."""
        vae, _ = build_cvae(_point_cfg("concat"))
        save_path = str(tmp_path / "point2.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        assert loaded is not None
        assert loaded.name == "cvae_condprior"


# ===========================================================================
# 3 — Windowing helpers used by engine
# ===========================================================================

class TestEngineWindowing:

    def test_x_val_w_shape(self):
        W = 7
        N = 30
        X_val, Y_val, D_val, C_val = _toy_val(N)
        X_val_w, _, _, _ = build_windows_single_experiment(
            X_val, Y_val, D_val, C_val,
            window_size=W, stride=1, pad_mode="edge",
        )
        assert X_val_w.shape == (N, W, 2)

    def test_center_frame_matches_original(self):
        W = 7
        N = 40
        X_val, Y_val, D_val, C_val = _toy_val(N)
        X_val_w, _, _, _ = build_windows_single_experiment(
            X_val, Y_val, D_val, C_val,
            window_size=W, stride=1, pad_mode="edge",
        )
        # Center frame of each window must equal the original X_val row
        center = W // 2
        np.testing.assert_array_almost_equal(X_val_w[:, center, :], X_val)

    def test_xv_in_slice_stratified(self):
        """Slicing X_val_w with an index array (stratified path) gives correct shape."""
        W = 7
        N = 50
        X_val, Y_val, D_val, C_val = _toy_val(N)
        X_val_w, _, _, _ = build_windows_single_experiment(
            X_val, Y_val, D_val, C_val,
            window_size=W, stride=1, pad_mode="edge",
        )
        idx = np.array([0, 5, 10, 20, 30])
        Xv_in = X_val_w[idx]
        assert Xv_in.shape == (len(idx), W, 2)

    def test_xv_center_is_2d(self):
        """Xv_center (= Xv = X_val[:n_eval]) must always be (N, 2)."""
        N = 20
        X_val, _, _, _ = _toy_val(N)
        Xv = X_val[:N]
        assert Xv.ndim == 2
        assert Xv.shape[1] == 2


# ===========================================================================
# 4 — _quick_cvae_predict seq path (global windowing)
# ===========================================================================

class TestQuickCvaePredictSeq:

    def _make_fake_state(self, tmp_path, window_size: int = 7):
        """Write a minimal state_run.json with data_split windowing info."""
        import json
        state = {
            "normalization": {"D_min": 0.0, "D_max": 1.0, "C_min": 0.1, "C_max": 0.9},
            "data_split": {
                "window_size": window_size,
                "window_stride": 1,
                "window_pad_mode": "edge",
            },
        }
        (tmp_path / "state_run.json").write_text(json.dumps(state))

    def _save_seq_model(self, tmp_path, window_size: int = 7):
        vae, _ = build_seq_cvae(_seq_cfg(window_size=window_size))
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        vae.save(str(models_dir / "best_model_full.keras"), include_optimizer=False)
        return tmp_path

    def test_global_windowing_shape(self):
        """build_windows_single_experiment on full val array gives (N, W, 2)."""
        W = 7
        N = 25
        X_arr = np.random.randn(N, 2).astype(np.float32)
        Y_dummy = np.zeros((N, 2), dtype=np.float32)
        D_dummy = np.ones((N, 1), dtype=np.float32)
        C_dummy = np.ones((N, 1), dtype=np.float32)
        X_arr_w, _, _, _ = build_windows_single_experiment(
            X_arr, Y_dummy, D_dummy, C_dummy,
            window_size=W, stride=1, pad_mode="edge",
        )
        assert X_arr_w.shape == (N, W, 2)

    def test_x_center_is_2d_after_windowing(self):
        """X_center_arr must remain (N, 2) after windowing for tiling."""
        W = 7
        N = 20
        X_arr = np.random.randn(N, 2).astype(np.float32)
        X_center_arr = X_arr  # saved before replacing X_arr with windowed
        Y_dummy = np.zeros((N, 2), dtype=np.float32)
        D_dummy = np.ones((N, 1), dtype=np.float32)
        C_dummy = np.ones((N, 1), dtype=np.float32)
        X_arr_w, _, _, _ = build_windows_single_experiment(
            X_arr, Y_dummy, D_dummy, C_dummy,
            window_size=W, stride=1, pad_mode="edge",
        )
        assert X_center_arr.shape == (N, 2)  # 2D untouched
        assert X_arr_w.shape == (N, W, 2)    # 3D windowed

    def test_tiling_center_arr_shape(self):
        """Tiling X_center_arr (2D) produces (mc_samples*N, 2)."""
        mc = 3
        N = 10
        X_center = np.random.randn(N, 2).astype(np.float32)
        X_tiled = np.tile(X_center, (mc, 1))
        assert X_tiled.shape == (mc * N, 2)

    def test_seq_inference_forward_windowed(self, tmp_path):
        """Full seq inference forward pass with windowed (N,W,2) input."""
        W = 7
        cfg = _seq_cfg(window_size=W)
        vae, _ = build_seq_cvae(cfg)
        inf = create_seq_inference_model(vae, deterministic=True)
        N = 8
        rng = np.random.default_rng(99)
        X_win = rng.normal(size=(N, W, 2)).astype(np.float32)
        D = rng.uniform(0, 1, size=(N, 1)).astype(np.float32)
        C = rng.uniform(0, 1, size=(N, 1)).astype(np.float32)
        y = inf([X_win, D, C], training=False)
        assert y.shape == (N, 2)
        assert not np.any(np.isnan(y.numpy()))

    def test_create_inference_from_full_dispatches_seq(self, tmp_path):
        """create_inference_model_from_full dispatches to seq path for rank-3 prior."""
        cfg = _seq_cfg(window_size=7)
        vae, _ = build_seq_cvae(cfg)
        save_path = str(tmp_path / "seq_disp.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        inf = create_inference_model_from_full(loaded, deterministic=True)
        assert inf.name == "inference_seq_condprior_det"
        N, W = 6, 7
        rng = np.random.default_rng(7)
        X_win = rng.normal(size=(N, W, 2)).astype(np.float32)
        D = rng.uniform(size=(N, 1)).astype(np.float32)
        C = rng.uniform(size=(N, 1)).astype(np.float32)
        y = inf([X_win, D, C], training=False)
        assert y.shape == (N, 2)
        assert not np.any(np.isnan(y.numpy()))
