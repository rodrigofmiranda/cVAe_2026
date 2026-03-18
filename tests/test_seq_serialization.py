# -*- coding: utf-8 -*-
"""Tests for seq_bigru_residual save/load/inference robustness.

Covers:
  1. Full seq training model saves and loads cleanly (no Lambda warnings)
  2. prior_net and decoder sub-models save and reload
  3. create_seq_inference_model works from a reloaded model
  4. Forward pass after reload is correct (no NaN, correct shape)
  5. Point-wise models (concat/channel_residual) are unaffected
  6. Custom serializable layers (ExtractCenterFrame, ClipValues, SliceFeatures)
     have working get_config/from_config round-trips
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.models.cvae_sequence import (
    ClipValues,
    ExtractCenterFrame,
    SliceFeatures,
    build_seq_cvae,
    create_seq_inference_model,
    load_seq_model,
)
from src.models.cvae import build_cvae, create_inference_model_from_full


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


def _legacy_cfg(**overrides) -> dict:
    cfg = {
        "layer_sizes": [32, 64],
        "latent_dim": 4,
        "beta": 0.01,
        "lr": 3e-4,
        "dropout": 0.0,
        "free_bits": 0.0,
        "kl_anneal_epochs": 3,
        "activation": "leaky_relu",
        "arch_variant": "legacy_2025_zero_y",
    }
    cfg.update(overrides)
    return cfg


def _toy_batch(N: int = 8, W: int = 7, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_win = rng.normal(size=(N, W, 2)).astype(np.float32)
    D     = rng.uniform(0, 1, size=(N, 1)).astype(np.float32)
    C     = rng.uniform(0, 1, size=(N, 1)).astype(np.float32)
    Y     = rng.normal(size=(N, 2)).astype(np.float32)
    return X_win, D, C, Y


# ===========================================================================
# Custom layer config round-trips
# ===========================================================================

class TestCustomLayerConfigs:

    def test_extract_center_frame_get_config(self):
        layer = ExtractCenterFrame(center_idx=3, name="ecf_test")
        cfg = layer.get_config()
        assert cfg["center_idx"] == 3
        rebuilt = ExtractCenterFrame.from_config(cfg)
        assert rebuilt.center_idx == 3

    def test_clip_values_get_config(self):
        layer = ClipValues(lo=-5.0, hi=5.0, name="clip_test")
        cfg = layer.get_config()
        assert cfg["lo"] == -5.0
        assert cfg["hi"] == 5.0
        rebuilt = ClipValues.from_config(cfg)
        assert rebuilt.lo == -5.0
        assert rebuilt.hi == 5.0

    def test_slice_features_get_config(self):
        layer = SliceFeatures(start=2, end=4, name="slice_test")
        cfg = layer.get_config()
        assert cfg["start"] == 2
        assert cfg["end"] == 4
        rebuilt = SliceFeatures.from_config(cfg)
        assert rebuilt.start == 2
        assert rebuilt.end == 4

    def test_extract_center_frame_output_shape(self):
        layer = ExtractCenterFrame(center_idx=3)
        x = tf.zeros((5, 7, 2))
        out = layer(x)
        assert out.shape == (5, 2)

    def test_clip_values_clips_correctly(self):
        layer = ClipValues(lo=0.0, hi=1.0)
        x = tf.constant([-1.0, 0.5, 2.0])
        out = layer(x).numpy()
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(0.5)
        assert out[2] == pytest.approx(1.0)

    def test_slice_features_output_shape(self):
        layer = SliceFeatures(start=0, end=2)
        x = tf.zeros((5, 4))
        out = layer(x)
        assert out.shape == (5, 2)

    def test_extract_center_extracts_correct_timestep(self):
        center = 3
        layer = ExtractCenterFrame(center_idx=center)
        x = np.arange(14, dtype=np.float32).reshape(1, 7, 2)
        out = layer(tf.constant(x)).numpy()
        np.testing.assert_array_equal(out[0], x[0, center, :])


# ===========================================================================
# 1 — Full model save / load
# ===========================================================================

class TestFullModelSaveLoad:

    def test_full_model_saves(self, tmp_path):
        vae, _ = build_seq_cvae(_seq_cfg())
        save_path = str(tmp_path / "full_model.keras")
        vae.save(save_path, include_optimizer=False)
        assert (tmp_path / "full_model.keras").exists()

    def test_full_model_loads_via_helper(self, tmp_path):
        vae, _ = build_seq_cvae(_seq_cfg())
        save_path = str(tmp_path / "full_model.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        assert loaded is not None

    def test_loaded_model_name(self, tmp_path):
        vae, _ = build_seq_cvae(_seq_cfg())
        save_path = str(tmp_path / "full_model.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        assert loaded.name == "cvae_seq_condprior"

    def test_loaded_model_has_encoder(self, tmp_path):
        vae, _ = build_seq_cvae(_seq_cfg())
        save_path = str(tmp_path / "full.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        enc = loaded.get_layer("encoder")
        assert enc is not None
        assert enc.inputs[0].shape[1:] == (7, 2)

    def test_loaded_model_has_prior_net(self, tmp_path):
        vae, _ = build_seq_cvae(_seq_cfg())
        save_path = str(tmp_path / "full.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        prior = loaded.get_layer("prior_net")
        assert prior is not None
        assert prior.inputs[0].shape[1:] == (7, 2)

    def test_loaded_model_has_decoder(self, tmp_path):
        vae, _ = build_seq_cvae(_seq_cfg())
        save_path = str(tmp_path / "full.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        dec = loaded.get_layer("decoder")
        assert dec is not None
        assert dec.inputs[0].shape[1:] == (4,)  # z_input

    def test_loaded_model_prior_rank3_input(self, tmp_path):
        """prior_net.inputs[0] must be rank 3 after load (dispatch heuristic)."""
        vae, _ = build_seq_cvae(_seq_cfg())
        save_path = str(tmp_path / "full.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        prior = loaded.get_layer("prior_net")
        assert len(prior.inputs[0].shape) == 3


# ===========================================================================
# 2 — Sub-model (prior_net, decoder) save / load
# ===========================================================================

class TestSubModelSaveLoad:

    def test_decoder_saves_and_loads(self, tmp_path):
        vae, _ = build_seq_cvae(_seq_cfg())
        dec_path = str(tmp_path / "decoder.keras")
        vae.get_layer("decoder").save(dec_path, include_optimizer=False)
        dec_loaded = load_seq_model(dec_path)
        assert dec_loaded is not None
        assert dec_loaded.inputs[0].shape[1:] == (4,)  # z

    def test_prior_net_saves_and_loads(self, tmp_path):
        vae, _ = build_seq_cvae(_seq_cfg())
        prior_path = str(tmp_path / "prior_net.keras")
        vae.get_layer("prior_net").save(prior_path, include_optimizer=False)
        prior_loaded = load_seq_model(prior_path)
        assert prior_loaded is not None
        assert prior_loaded.inputs[0].shape[1:] == (7, 2)

    def test_decoder_forward_after_reload(self, tmp_path):
        cfg = _seq_cfg()
        vae, _ = build_seq_cvae(cfg)
        dec_path = str(tmp_path / "decoder.keras")
        vae.get_layer("decoder").save(dec_path, include_optimizer=False)
        dec = load_seq_model(dec_path)
        N = 4
        rng = np.random.default_rng(1)
        z      = rng.normal(size=(N, cfg["latent_dim"])).astype(np.float32)
        x_cent = rng.normal(size=(N, 2)).astype(np.float32)
        d      = rng.uniform(0, 1, size=(N, 1)).astype(np.float32)
        c      = rng.uniform(0, 1, size=(N, 1)).astype(np.float32)
        out = dec([z, x_cent, d, c], training=False)
        assert out.shape == (N, 4)
        assert not np.any(np.isnan(out.numpy()))


# ===========================================================================
# 3 & 4 — Inference model from reloaded full model + forward pass
# ===========================================================================

class TestInferenceFromLoaded:

    def test_create_seq_inference_from_loaded(self, tmp_path):
        vae, _ = build_seq_cvae(_seq_cfg())
        save_path = str(tmp_path / "full.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        inf = create_seq_inference_model(loaded, deterministic=True)
        assert inf.name == "inference_seq_condprior_det"

    def test_inference_input_shapes_from_loaded(self, tmp_path):
        W = 9
        vae, _ = build_seq_cvae(_seq_cfg(window_size=W))
        save_path = str(tmp_path / "full.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        inf = create_seq_inference_model(loaded, deterministic=True)
        assert inf.inputs[0].shape[1:] == (W, 2)
        assert inf.inputs[1].shape[1:] == (1,)
        assert inf.inputs[2].shape[1:] == (1,)

    def test_forward_pass_deterministic_after_reload(self, tmp_path):
        cfg = _seq_cfg(window_size=7)
        vae, _ = build_seq_cvae(cfg)
        save_path = str(tmp_path / "full.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        inf = create_seq_inference_model(loaded, deterministic=True)
        N, W = 8, 7
        X_win, D, C, _ = _toy_batch(N, W)
        y_pred = inf([X_win, D, C], training=False)
        assert y_pred.shape == (N, 2)
        assert not np.any(np.isnan(y_pred.numpy()))

    def test_forward_pass_stochastic_after_reload(self, tmp_path):
        cfg = _seq_cfg(window_size=7)
        vae, _ = build_seq_cvae(cfg)
        save_path = str(tmp_path / "full.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        inf = create_seq_inference_model(loaded, deterministic=False)
        N, W = 8, 7
        X_win, D, C, _ = _toy_batch(N, W, seed=42)
        y_pred = inf([X_win, D, C], training=False)
        assert y_pred.shape == (N, 2)
        assert not np.any(np.isnan(y_pred.numpy()))

    def test_create_inference_model_from_full_dispatches_correctly(self, tmp_path):
        """create_inference_model_from_full in cvae.py must dispatch to seq path after reload."""
        cfg = _seq_cfg(window_size=7)
        vae, _ = build_seq_cvae(cfg)
        save_path = str(tmp_path / "full.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        inf = create_inference_model_from_full(loaded, deterministic=True)
        assert inf.name == "inference_seq_condprior_det"
        N, W = 4, 7
        X_win, D, C, _ = _toy_batch(N, W)
        y_pred = inf([X_win, D, C], training=False)
        assert y_pred.shape == (N, 2)
        assert not np.any(np.isnan(y_pred.numpy()))

    def test_prior_net_predict_after_reload(self, tmp_path):
        """prior_net.predict from reloaded model must accept windowed input."""
        cfg = _seq_cfg(window_size=7)
        vae, _ = build_seq_cvae(cfg)
        save_path = str(tmp_path / "full.keras")
        vae.save(save_path, include_optimizer=False)
        loaded = load_seq_model(save_path)
        prior = loaded.get_layer("prior_net")
        N, W = 6, 7
        X_win, D, C, _ = _toy_batch(N, W)
        z_mean, z_lv = prior([X_win, D, C], training=False)
        assert z_mean.shape == (N, cfg["latent_dim"])
        assert not np.any(np.isnan(z_mean.numpy()))

    def test_window_size_preserved_after_reload(self, tmp_path):
        """W must be correctly inferred from reloaded prior_net.inputs[0].shape."""
        for W in [5, 7, 33]:
            vae, _ = build_seq_cvae(_seq_cfg(window_size=W))
            save_path = str(tmp_path / f"full_W{W}.keras")
            vae.save(save_path, include_optimizer=False)
            loaded = load_seq_model(save_path)
            prior = loaded.get_layer("prior_net")
            assert prior.inputs[0].shape[1] == W


# ===========================================================================
# 5 — Point-wise models unaffected
# ===========================================================================

class TestPointwiseSaveLoadUnaffected:

    def _load_pointwise(self, path):
        return load_seq_model(path)

    def test_concat_saves_and_loads(self, tmp_path):
        vae, _ = build_cvae(_point_cfg("concat"))
        path = str(tmp_path / "concat.keras")
        vae.save(path, include_optimizer=False)
        loaded = self._load_pointwise(path)
        assert loaded.name == "cvae_condprior"

    def test_concat_inference_from_loaded(self, tmp_path):
        vae, _ = build_cvae(_point_cfg("concat"))
        path = str(tmp_path / "concat.keras")
        vae.save(path, include_optimizer=False)
        loaded = self._load_pointwise(path)
        inf = create_inference_model_from_full(loaded, deterministic=True)
        # Point-wise: x_input must be (None, 2)
        assert inf.inputs[0].shape[1:] == (2,)
        rng = np.random.default_rng(0)
        N = 4
        x  = rng.normal(size=(N, 2)).astype(np.float32)
        d  = rng.uniform(size=(N, 1)).astype(np.float32)
        c  = rng.uniform(size=(N, 1)).astype(np.float32)
        y_pred = inf([x, d, c], training=False)
        assert y_pred.shape == (N, 2)

    def test_channel_residual_saves_and_loads(self, tmp_path):
        vae, _ = build_cvae(_point_cfg("channel_residual"))
        path = str(tmp_path / "ch_res.keras")
        vae.save(path, include_optimizer=False)
        loaded = self._load_pointwise(path)
        assert loaded is not None

    def test_delta_residual_saves_and_loads(self, tmp_path):
        vae, _ = build_cvae(_point_cfg("delta_residual"))
        path = str(tmp_path / "delta_res.keras")
        vae.save(path, include_optimizer=False)
        loaded = self._load_pointwise(path)
        assert loaded.name == "cvae_condprior_delta_residual"

    def test_delta_residual_inference_after_reload(self, tmp_path):
        vae, _ = build_cvae(_point_cfg("delta_residual"))
        path = str(tmp_path / "delta_res_infer.keras")
        vae.save(path, include_optimizer=False)
        loaded = self._load_pointwise(path)
        inf = create_inference_model_from_full(loaded, deterministic=False)
        rng = np.random.default_rng(13)
        N = 5
        x = rng.normal(size=(N, 2)).astype(np.float32)
        d = rng.uniform(size=(N, 1)).astype(np.float32)
        c = rng.uniform(size=(N, 1)).astype(np.float32)
        y_pred = inf([x, d, c], training=False)
        assert y_pred.shape == (N, 2)
        assert not np.any(np.isnan(y_pred.numpy()))


class TestLegacy2025SaveLoad:

    def test_legacy_2025_full_model_saves_and_loads(self, tmp_path):
        vae, _ = build_cvae(_legacy_cfg())
        path = str(tmp_path / "legacy.keras")
        vae.save(path, include_optimizer=False)

        loaded = load_seq_model(path)

        assert loaded.name == "cvae_legacy_2025_zero_y"
        assert loaded.get_layer("encoder") is not None
        assert loaded.get_layer("prior_net") is not None
        assert loaded.get_layer("decoder") is not None

    def test_legacy_2025_submodels_save_and_load(self, tmp_path):
        vae, _ = build_cvae(_legacy_cfg())
        prior_path = str(tmp_path / "legacy_prior.keras")
        dec_path = str(tmp_path / "legacy_decoder.keras")

        vae.get_layer("prior_net").save(prior_path, include_optimizer=False)
        vae.get_layer("decoder").save(dec_path, include_optimizer=False)

        prior_loaded = load_seq_model(prior_path)
        dec_loaded = load_seq_model(dec_path)

        assert prior_loaded.inputs[0].shape[1:] == (2,)
        assert dec_loaded.inputs[0].shape[1:] == (4,)

    def test_legacy_2025_inference_after_reload(self, tmp_path):
        vae, _ = build_cvae(_legacy_cfg())
        path = str(tmp_path / "legacy_full.keras")
        vae.save(path, include_optimizer=False)

        loaded = load_seq_model(path)
        inf = create_inference_model_from_full(loaded, deterministic=False)

        rng = np.random.default_rng(5)
        x = rng.normal(size=(5, 2)).astype(np.float32)
        d = rng.uniform(size=(5, 1)).astype(np.float32)
        c = rng.uniform(size=(5, 1)).astype(np.float32)
        y = inf([x, d, c], training=False)

        assert y.shape == (5, 2)
        assert not np.any(np.isnan(y.numpy()))

    def test_legacy_2025_prior_matches_encoder_zero_y_after_reload(self, tmp_path):
        vae, _ = build_cvae(_legacy_cfg())
        path = str(tmp_path / "legacy_equiv.keras")
        vae.save(path, include_optimizer=False)

        loaded = load_seq_model(path)
        encoder = loaded.get_layer("encoder")
        prior = loaded.get_layer("prior_net")

        rng = np.random.default_rng(19)
        x = rng.normal(size=(6, 2)).astype(np.float32)
        d = rng.uniform(size=(6, 1)).astype(np.float32)
        c = rng.uniform(size=(6, 1)).astype(np.float32)
        y_zero = np.zeros((6, 2), dtype=np.float32)

        mu_q, lv_q = encoder.predict([x, d, c, y_zero], verbose=0)
        mu_p, lv_p = prior.predict([x, d, c], verbose=0)

        np.testing.assert_allclose(mu_q, mu_p, atol=1e-6)
        np.testing.assert_allclose(lv_q, lv_p, atol=1e-6)
