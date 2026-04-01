# -*- coding: utf-8 -*-
"""Tests for src.models.cvae_sequence.

Covers:
  1. Build of seq cVAE with minimal config
  2. Correct input/output shapes for all sub-models and the full model
  3. Correct sub-model names (encoder, prior_net, decoder)
  4. Inference model build (deterministic and stochastic)
  5. Forward pass on small toy arrays — no NaNs, correct output shapes
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.models.cvae_sequence import (
    build_seq_cvae,
    build_seq_decoder,
    build_seq_encoder,
    build_seq_prior_net,
    create_seq_inference_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _min_cfg(**overrides) -> dict:
    """Return the smallest valid config for seq_bigru_residual."""
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
        "kl_anneal_epochs": 5,
        "activation": "leaky_relu",
    }
    cfg.update(overrides)
    return cfg


def _toy_batch(N: int = 16, W: int = 7, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_win = rng.normal(size=(N, W, 2)).astype(np.float32)
    D     = rng.uniform(0, 1, size=(N, 1)).astype(np.float32)
    C     = rng.uniform(0, 1, size=(N, 1)).astype(np.float32)
    Y     = rng.normal(size=(N, 2)).astype(np.float32)
    return X_win, D, C, Y


# ===========================================================================
# 1 & 2 — Sub-model build + shapes
# ===========================================================================

class TestSubModelBuild:

    def test_prior_net_builds(self):
        prior = build_seq_prior_net(_min_cfg())
        assert prior is not None

    def test_encoder_builds(self):
        enc = build_seq_encoder(_min_cfg())
        assert enc is not None

    def test_decoder_builds(self):
        dec = build_seq_decoder(_min_cfg())
        assert dec is not None

    def test_seq_gru_layers_are_explicitly_unrolled(self):
        prior = build_seq_prior_net(_min_cfg(seq_bidirectional=True))
        bigru = prior.get_layer("prior_net_bigru_0")
        assert bigru.forward_layer.unroll is True
        assert bigru.backward_layer.unroll is True

    def test_seq_gru_unroll_can_be_disabled_explicitly(self):
        prior = build_seq_prior_net(_min_cfg(seq_bidirectional=True, seq_gru_unroll=False))
        bigru = prior.get_layer("prior_net_bigru_0")
        assert bigru.forward_layer.unroll is False
        assert bigru.backward_layer.unroll is False

    def test_seq_gru_compat_backend_uses_rnn_wrapper(self):
        prior = build_seq_prior_net(
            _min_cfg(seq_bidirectional=True, seq_gru_unroll=False, seq_gru_backend="compat")
        )
        bigru = prior.get_layer("prior_net_bigru_0")
        assert bigru.forward_layer.__class__.__name__ == "RNN"
        assert bigru.backward_layer.__class__.__name__ == "RNN"
        assert bigru.forward_layer.cell.__class__.__name__ == "GRUCell"
        assert bigru.backward_layer.cell.__class__.__name__ == "GRUCell"

    # Input shapes — prior_net
    def test_prior_net_has_three_inputs(self):
        prior = build_seq_prior_net(_min_cfg(window_size=9))
        assert len(prior.inputs) == 3

    def test_prior_net_x_window_shape(self):
        W = 9
        prior = build_seq_prior_net(_min_cfg(window_size=W))
        assert prior.inputs[0].shape[1:] == (W, 2)

    def test_prior_net_condition_shapes(self):
        prior = build_seq_prior_net(_min_cfg())
        assert prior.inputs[1].shape[1:] == (1,)  # d
        assert prior.inputs[2].shape[1:] == (1,)  # c

    # Input shapes — encoder
    def test_encoder_has_four_inputs(self):
        enc = build_seq_encoder(_min_cfg(window_size=9))
        assert len(enc.inputs) == 4

    def test_encoder_x_window_shape(self):
        W = 9
        enc = build_seq_encoder(_min_cfg(window_size=W))
        assert enc.inputs[0].shape[1:] == (W, 2)

    def test_encoder_y_center_shape(self):
        enc = build_seq_encoder(_min_cfg())
        assert enc.inputs[3].shape[1:] == (2,)

    # Input shapes — decoder
    def test_decoder_has_four_inputs(self):
        dec = build_seq_decoder(_min_cfg(latent_dim=4))
        assert len(dec.inputs) == 4

    def test_decoder_z_shape(self):
        dec = build_seq_decoder(_min_cfg(latent_dim=6))
        assert dec.inputs[0].shape[1:] == (6,)

    def test_decoder_x_center_shape(self):
        dec = build_seq_decoder(_min_cfg())
        assert dec.inputs[1].shape[1:] == (2,)

    # Output shapes
    def test_prior_net_output_shapes(self):
        latent = 6
        prior = build_seq_prior_net(_min_cfg(latent_dim=latent))
        assert len(prior.outputs) == 2
        assert prior.outputs[0].shape[1:] == (latent,)
        assert prior.outputs[1].shape[1:] == (latent,)

    def test_encoder_output_shapes(self):
        latent = 6
        enc = build_seq_encoder(_min_cfg(latent_dim=latent))
        assert len(enc.outputs) == 2
        assert enc.outputs[0].shape[1:] == (latent,)
        assert enc.outputs[1].shape[1:] == (latent,)

    def test_decoder_output_shape(self):
        dec = build_seq_decoder(_min_cfg())
        # (mean_I, mean_Q, logvar_I, logvar_Q)
        assert dec.outputs[0].shape[1:] == (4,)

    def test_radial_feature_adds_internal_inputs_to_submodels(self):
        cfg = _min_cfg(radial_feature=True)
        prior = build_seq_prior_net(cfg)
        enc = build_seq_encoder(cfg)
        dec = build_seq_decoder(cfg)

        assert len(prior.inputs) == 4
        assert prior.inputs[3].shape[1:] == (1,)

        assert len(enc.inputs) == 5
        assert enc.inputs[4].shape[1:] == (1,)

        assert len(dec.inputs) == 5
        assert dec.inputs[4].shape[1:] == (1,)


# ===========================================================================
# 1 — Full model build
# ===========================================================================

class TestFullModelBuild:

    # 1. Build succeeds
    def test_build_returns_tuple(self):
        vae, kl_cb = build_seq_cvae(_min_cfg())
        assert vae is not None
        assert kl_cb is not None

    def test_model_name(self):
        vae, _ = build_seq_cvae(_min_cfg())
        assert vae.name == "cvae_seq_condprior"

    # 3. Sub-model names
    def test_encoder_layer_name(self):
        vae, _ = build_seq_cvae(_min_cfg())
        enc = vae.get_layer("encoder")
        assert enc is not None
        assert enc.name == "encoder"

    def test_prior_net_layer_name(self):
        vae, _ = build_seq_cvae(_min_cfg())
        prior = vae.get_layer("prior_net")
        assert prior is not None
        assert prior.name == "prior_net"

    def test_decoder_layer_name(self):
        vae, _ = build_seq_cvae(_min_cfg())
        dec = vae.get_layer("decoder")
        assert dec is not None
        assert dec.name == "decoder"

    # 2. Full model input shapes
    def test_full_model_input_count(self):
        vae, _ = build_seq_cvae(_min_cfg())
        assert len(vae.inputs) == 4

    def test_full_model_x_window_input_shape(self):
        W = 9
        vae, _ = build_seq_cvae(_min_cfg(window_size=W))
        assert vae.inputs[0].shape[1:] == (W, 2)

    def test_full_model_condition_input_shapes(self):
        vae, _ = build_seq_cvae(_min_cfg())
        assert vae.inputs[1].shape[1:] == (1,)  # d
        assert vae.inputs[2].shape[1:] == (1,)  # c
        assert vae.inputs[3].shape[1:] == (2,)  # y_true

    # Parameterized window sizes
    @pytest.mark.parametrize("W", [1, 3, 7, 33])
    def test_builds_for_various_window_sizes(self, W):
        vae, _ = build_seq_cvae(_min_cfg(window_size=W))
        assert vae.inputs[0].shape[1:] == (W, 2)

    # Parameterized latent dims
    @pytest.mark.parametrize("latent", [2, 4, 8])
    def test_builds_for_various_latent_dims(self, latent):
        vae, _ = build_seq_cvae(_min_cfg(latent_dim=latent))
        assert vae is not None

    def test_non_bidirectional_builds(self):
        vae, _ = build_seq_cvae(_min_cfg(seq_bidirectional=False))
        assert vae is not None

    def test_multi_layer_gru_builds(self):
        vae, _ = build_seq_cvae(_min_cfg(seq_num_layers=2))
        assert vae is not None

    def test_radial_feature_keeps_external_full_model_signature(self):
        vae, _ = build_seq_cvae(_min_cfg(radial_feature=True))
        assert len(vae.inputs) == 4
        assert vae.inputs[0].shape[1:] == (7, 2)
        assert vae.inputs[1].shape[1:] == (1,)
        assert vae.inputs[2].shape[1:] == (1,)
        assert vae.inputs[3].shape[1:] == (2,)


# ===========================================================================
# 5 — Forward pass (no training, shape + NaN checks)
# ===========================================================================

class TestForwardPass:

    def test_prior_net_forward_shapes(self):
        cfg = _min_cfg()
        prior = build_seq_prior_net(cfg)
        N, W = 16, cfg["window_size"]
        X_win, D, C, _ = _toy_batch(N, W)
        z_mean, z_lv = prior([X_win, D, C], training=False)
        assert z_mean.shape == (N, cfg["latent_dim"])
        assert z_lv.shape   == (N, cfg["latent_dim"])

    def test_encoder_forward_shapes(self):
        cfg = _min_cfg()
        enc = build_seq_encoder(cfg)
        N, W = 16, cfg["window_size"]
        X_win, D, C, Y = _toy_batch(N, W)
        z_mean, z_lv = enc([X_win, D, C, Y], training=False)
        assert z_mean.shape == (N, cfg["latent_dim"])
        assert z_lv.shape   == (N, cfg["latent_dim"])

    def test_decoder_forward_shapes(self):
        cfg = _min_cfg()
        dec = build_seq_decoder(cfg)
        N = 16
        rng = np.random.default_rng(1)
        z      = rng.normal(size=(N, cfg["latent_dim"])).astype(np.float32)
        x_cent = rng.normal(size=(N, 2)).astype(np.float32)
        d      = rng.uniform(0, 1, size=(N, 1)).astype(np.float32)
        c      = rng.uniform(0, 1, size=(N, 1)).astype(np.float32)
        out = dec([z, x_cent, d, c], training=False)
        assert out.shape == (N, 4)

    def test_full_model_forward_shape(self):
        cfg = _min_cfg()
        vae, _ = build_seq_cvae(cfg)
        N, W = 8, cfg["window_size"]
        X_win, D, C, Y = _toy_batch(N, W)
        out = vae([X_win, D, C, Y], training=False)
        # Loss layer returns y_mean: (N, 2)
        assert out.shape == (N, 2), f"Expected ({N},2), got {out.shape}"

    def test_full_model_no_nan(self):
        cfg = _min_cfg()
        vae, _ = build_seq_cvae(cfg)
        N, W = 8, cfg["window_size"]
        X_win, D, C, Y = _toy_batch(N, W)
        out = vae([X_win, D, C, Y], training=False)
        assert not np.any(np.isnan(out.numpy())), "NaN in full model forward pass"

    @pytest.mark.parametrize("W", [1, 5, 33])
    def test_forward_various_window_sizes(self, W):
        cfg = _min_cfg(window_size=W)
        vae, _ = build_seq_cvae(cfg)
        N = 4
        X_win, D, C, Y = _toy_batch(N, W)
        out = vae([X_win, D, C, Y], training=False)
        assert out.shape == (N, 2)
        assert not np.any(np.isnan(out.numpy()))

    def test_decoder_residual_formulation(self):
        """y_mean in decoder output must be x_center + delta, not raw prediction."""
        cfg = _min_cfg()
        dec = build_seq_decoder(cfg)
        N = 4
        rng = np.random.default_rng(42)
        z      = np.zeros((N, cfg["latent_dim"]), dtype=np.float32)
        x_cent = rng.normal(size=(N, 2)).astype(np.float32)
        d      = np.zeros((N, 1), dtype=np.float32)
        c      = np.zeros((N, 1), dtype=np.float32)
        out = dec([z, x_cent, d, c], training=False).numpy()
        y_mean   = out[:, :2]
        # y_mean should be x_cent + delta, where delta is from the MLP
        # We can verify residual structure by checking that the delta layer exists
        layer_names = {l.name for l in dec.layers}
        assert "y_mean_residual" in layer_names, "Missing 'y_mean_residual' Add layer"
        assert "delta_mean" in layer_names, "Missing 'delta_mean' Lambda layer"


# ===========================================================================
# 4 — Inference model
# ===========================================================================

class TestInferenceModel:

    def test_build_deterministic(self):
        vae, _ = build_seq_cvae(_min_cfg())
        inf = create_seq_inference_model(vae, deterministic=True)
        assert inf.name == "inference_seq_condprior_det"

    def test_build_stochastic(self):
        vae, _ = build_seq_cvae(_min_cfg())
        inf = create_seq_inference_model(vae, deterministic=False)
        assert inf.name == "inference_seq_condprior"

    def test_inference_input_count(self):
        vae, _ = build_seq_cvae(_min_cfg())
        inf = create_seq_inference_model(vae, deterministic=True)
        assert len(inf.inputs) == 3

    def test_inference_input_shapes(self):
        W = 9
        vae, _ = build_seq_cvae(_min_cfg(window_size=W))
        inf = create_seq_inference_model(vae, deterministic=True)
        assert inf.inputs[0].shape[1:] == (W, 2)  # x_window
        assert inf.inputs[1].shape[1:] == (1,)     # d
        assert inf.inputs[2].shape[1:] == (1,)     # c

    def test_inference_output_shape(self):
        cfg = _min_cfg()
        vae, _ = build_seq_cvae(cfg)
        inf = create_seq_inference_model(vae, deterministic=True)
        N, W = 16, cfg["window_size"]
        X_win, D, C, _ = _toy_batch(N, W)
        y_pred = inf([X_win, D, C], training=False)
        assert y_pred.shape == (N, 2)

    def test_inference_no_nan(self):
        cfg = _min_cfg()
        vae, _ = build_seq_cvae(cfg)
        inf = create_seq_inference_model(vae, deterministic=True)
        N, W = 16, cfg["window_size"]
        X_win, D, C, _ = _toy_batch(N, W, seed=99)
        y_pred = inf([X_win, D, C], training=False)
        assert not np.any(np.isnan(y_pred.numpy())), "NaN in inference output"

    def test_stochastic_inference_no_nan(self):
        cfg = _min_cfg()
        vae, _ = build_seq_cvae(cfg)
        inf = create_seq_inference_model(vae, deterministic=False)
        N, W = 16, cfg["window_size"]
        X_win, D, C, _ = _toy_batch(N, W, seed=7)
        y_pred = inf([X_win, D, C], training=False)
        assert not np.any(np.isnan(y_pred.numpy())), "NaN in stochastic inference"

    def test_window_size_inferred_from_prior(self):
        """create_seq_inference_model must infer W from prior.inputs[0].shape."""
        for W in [5, 9, 33]:
            vae, _ = build_seq_cvae(_min_cfg(window_size=W))
            inf = create_seq_inference_model(vae, deterministic=True)
            assert inf.inputs[0].shape[1] == W

    def test_radial_feature_keeps_external_inference_signature(self):
        vae, _ = build_seq_cvae(_min_cfg(window_size=9, radial_feature=True))
        inf = create_seq_inference_model(vae, deterministic=True)
        assert len(inf.inputs) == 3
        assert inf.inputs[0].shape[1:] == (9, 2)
        assert inf.inputs[1].shape[1:] == (1,)
        assert inf.inputs[2].shape[1:] == (1,)


# ===========================================================================
# Isolation: point-wise models unaffected
# ===========================================================================

class TestPointWiseIsolation:
    """Verify that importing cvae_sequence does not break concat/channel_residual."""

    def test_concat_still_builds(self):
        from src.models.cvae import build_cvae
        cfg = {
            "layer_sizes": [32, 64], "latent_dim": 4, "beta": 0.001,
            "lr": 3e-4, "dropout": 0.0, "free_bits": 0.0,
            "kl_anneal_epochs": 5, "arch_variant": "concat",
        }
        vae, _ = build_cvae(cfg)
        assert vae is not None

    def test_channel_residual_still_builds(self):
        from src.models.cvae import build_cvae
        cfg = {
            "layer_sizes": [32, 64], "latent_dim": 4, "beta": 0.001,
            "lr": 3e-4, "dropout": 0.0, "free_bits": 0.0,
            "kl_anneal_epochs": 5, "arch_variant": "channel_residual",
        }
        vae, _ = build_cvae(cfg)
        assert vae is not None

    def test_seq_bigru_residual_dispatches_via_build_cvae(self):
        """build_cvae must dispatch seq_bigru_residual to build_seq_cvae (Phase 4)."""
        from src.models.cvae import build_cvae
        cfg = {
            "layer_sizes": [32, 64], "latent_dim": 4, "beta": 0.001,
            "lr": 3e-4, "dropout": 0.0, "free_bits": 0.0,
            "kl_anneal_epochs": 5, "arch_variant": "seq_bigru_residual",
            "window_size": 7, "seq_hidden_size": 16, "seq_num_layers": 1,
            "seq_bidirectional": True,
        }
        vae, kl_cb = build_cvae(cfg)
        assert vae is not None
        assert vae.name == "cvae_seq_condprior"
        assert kl_cb is not None

    def test_seq_bigru_residual_build_cvae_layer_names(self):
        """build_cvae dispatch must preserve required layer names."""
        from src.models.cvae import build_cvae
        cfg = {
            "layer_sizes": [32, 64], "latent_dim": 4, "beta": 0.001,
            "lr": 3e-4, "dropout": 0.0, "free_bits": 0.0,
            "kl_anneal_epochs": 5, "arch_variant": "seq_bigru_residual",
            "window_size": 7, "seq_hidden_size": 16, "seq_num_layers": 1,
            "seq_bidirectional": True,
        }
        vae, _ = build_cvae(cfg)
        assert vae.get_layer("encoder") is not None
        assert vae.get_layer("prior_net") is not None
        assert vae.get_layer("decoder") is not None


# ===========================================================================
# Phase 4 — create_inference_model_from_full dispatch
# ===========================================================================

class TestInferenceDispatch:
    """Verify create_inference_model_from_full detects seq vs point-wise."""

    def _seq_cfg(self, **overrides):
        cfg = {
            "layer_sizes": [32, 64], "latent_dim": 4, "beta": 0.001,
            "lr": 3e-4, "dropout": 0.0, "free_bits": 0.0,
            "kl_anneal_epochs": 5, "arch_variant": "seq_bigru_residual",
            "window_size": 7, "seq_hidden_size": 16, "seq_num_layers": 1,
            "seq_bidirectional": True,
        }
        cfg.update(overrides)
        return cfg

    def _point_cfg(self, arch_variant="concat", **overrides):
        cfg = {
            "layer_sizes": [32, 64], "latent_dim": 4, "beta": 0.001,
            "lr": 3e-4, "dropout": 0.0, "free_bits": 0.0,
            "kl_anneal_epochs": 5, "arch_variant": arch_variant,
        }
        cfg.update(overrides)
        return cfg

    def test_seq_model_dispatches_to_seq_inference(self):
        from src.models.cvae import build_cvae, create_inference_model_from_full
        vae, _ = build_cvae(self._seq_cfg())
        inf = create_inference_model_from_full(vae, deterministic=True)
        assert inf.name == "inference_seq_condprior_det"

    def test_seq_model_inference_input_is_window(self):
        from src.models.cvae import build_cvae, create_inference_model_from_full
        W = 9
        vae, _ = build_cvae(self._seq_cfg(window_size=W))
        inf = create_inference_model_from_full(vae, deterministic=True)
        assert inf.inputs[0].shape[1:] == (W, 2)

    def test_seq_model_stochastic_dispatch(self):
        from src.models.cvae import build_cvae, create_inference_model_from_full
        vae, _ = build_cvae(self._seq_cfg())
        inf = create_inference_model_from_full(vae, deterministic=False)
        assert inf.name == "inference_seq_condprior"

    def test_concat_model_stays_on_pointwise_path(self):
        from src.models.cvae import build_cvae, create_inference_model_from_full
        vae, _ = build_cvae(self._point_cfg("concat"))
        inf = create_inference_model_from_full(vae, deterministic=True)
        assert inf.name == "inference_condprior_det"
        assert inf.inputs[0].shape[1:] == (2,)

    def test_channel_residual_model_stays_on_pointwise_path(self):
        from src.models.cvae import build_cvae, create_inference_model_from_full
        vae, _ = build_cvae(self._point_cfg("channel_residual"))
        inf = create_inference_model_from_full(vae, deterministic=True)
        assert inf.name == "inference_condprior_det"
        assert inf.inputs[0].shape[1:] == (2,)

    def test_seq_inference_output_shape(self):
        from src.models.cvae import build_cvae, create_inference_model_from_full
        cfg = self._seq_cfg(window_size=7)
        vae, _ = build_cvae(cfg)
        inf = create_inference_model_from_full(vae, deterministic=True)
        N, W = 8, 7
        rng = np.random.default_rng(0)
        X_win = rng.normal(size=(N, W, 2)).astype(np.float32)
        D = rng.uniform(0, 1, size=(N, 1)).astype(np.float32)
        C = rng.uniform(0, 1, size=(N, 1)).astype(np.float32)
        y_pred = inf([X_win, D, C], training=False)
        assert y_pred.shape == (N, 2)
