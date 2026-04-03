import numpy as np
import pytest

from src.evaluation.report import compute_latent_diagnostics
from src.models.cvae import (
    build_cvae,
    build_decoder,
    create_inference_model_from_full,
)
from src.models.cvae_sequence import load_seq_model
from src.training.gridsearch import _save_keras_model_compat


def _zero_dense_head(model, layer_name: str) -> None:
    layer = model.get_layer(layer_name)
    weights = layer.get_weights()
    assert len(weights) == 2
    kernel, bias = weights
    layer.set_weights([np.zeros_like(kernel), np.zeros_like(bias)])


def _legacy_cfg(**overrides):
    cfg = {
        "layer_sizes": [16, 32],
        "latent_dim": 4,
        "beta": 0.01,
        "lr": 3e-4,
        "dropout": 0.0,
        "free_bits": 0.0,
        "kl_anneal_epochs": 3,
        "batch_size": 8,
        "activation": "leaky_relu",
        "arch_variant": "legacy_2025_zero_y",
    }
    cfg.update(overrides)
    return cfg


def _point_cfg(arch_variant="concat", **overrides):
    cfg = {
        "layer_sizes": [16, 32],
        "latent_dim": 4,
        "beta": 0.01,
        "lr": 3e-4,
        "dropout": 0.0,
        "free_bits": 0.0,
        "kl_anneal_epochs": 3,
        "batch_size": 8,
        "activation": "leaky_relu",
        "arch_variant": arch_variant,
    }
    cfg.update(overrides)
    return cfg


def test_channel_residual_decoder_zero_head_returns_identity_mean():
    decoder = build_decoder(
        layer_sizes=[8],
        latent_dim=2,
        arch_variant="channel_residual",
    )
    _zero_dense_head(decoder, "output_params_raw")

    z = np.array([[0.2, -0.1]], dtype=np.float32)
    cond = np.array([[1.5, -0.25, 0.4, 0.6]], dtype=np.float32)

    out = decoder.predict([z, cond], verbose=0)

    np.testing.assert_allclose(out[:, :2], cond[:, :2], atol=1e-6)
    np.testing.assert_allclose(out[:, 2:], 0.0, atol=1e-6)


def test_delta_residual_decoder_zero_head_returns_zero_delta_params():
    decoder = build_decoder(
        layer_sizes=[8],
        latent_dim=2,
        arch_variant="delta_residual",
    )
    _zero_dense_head(decoder, "delta_output_params")

    z = np.array([[0.2, -0.1]], dtype=np.float32)
    cond = np.array([[1.5, -0.25, 0.4, 0.6]], dtype=np.float32)

    out = decoder.predict([z, cond], verbose=0)

    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_concat_decoder_zero_head_returns_zero_mean():
    decoder = build_decoder(
        layer_sizes=[8],
        latent_dim=2,
        arch_variant="concat",
    )
    _zero_dense_head(decoder, "output_params")

    z = np.array([[0.2, -0.1]], dtype=np.float32)
    cond = np.array([[1.5, -0.25, 0.4, 0.6]], dtype=np.float32)

    out = decoder.predict([z, cond], verbose=0)

    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_build_cvae_and_inference_support_channel_residual_variant():
    cfg = {
        "layer_sizes": [8],
        "latent_dim": 2,
        "beta": 0.001,
        "lr": 3e-4,
        "dropout": 0.0,
        "free_bits": 0.0,
        "kl_anneal_epochs": 2,
        "batch_size": 4,
        "activation": "leaky_relu",
        "arch_variant": "channel_residual",
    }
    vae, _ = build_cvae(cfg)

    inf_det = create_inference_model_from_full(vae, deterministic=True)
    inf_mc = create_inference_model_from_full(vae, deterministic=False)

    x = np.zeros((3, 2), dtype=np.float32)
    d = np.zeros((3, 1), dtype=np.float32)
    c = np.zeros((3, 1), dtype=np.float32)

    y_det = inf_det.predict([x, d, c], verbose=0)
    y_mc = inf_mc.predict([x, d, c], verbose=0)

    assert y_det.shape == (3, 2)
    assert y_mc.shape == (3, 2)


def test_build_cvae_and_inference_support_delta_residual_variant():
    vae, _ = build_cvae(_point_cfg("delta_residual"))

    inf_det = create_inference_model_from_full(vae, deterministic=True)
    inf_mc = create_inference_model_from_full(vae, deterministic=False)

    x = np.zeros((3, 2), dtype=np.float32)
    d = np.zeros((3, 1), dtype=np.float32)
    c = np.zeros((3, 1), dtype=np.float32)

    y_det = inf_det.predict([x, d, c], verbose=0)
    y_mc = inf_mc.predict([x, d, c], verbose=0)

    assert vae.name == "cvae_condprior_delta_residual"
    assert y_det.shape == (3, 2)
    assert y_mc.shape == (3, 2)


def test_build_cvae_and_inference_support_delta_residual_geom3_variant():
    vae, _ = build_cvae(
        _point_cfg(
            "delta_residual",
            support_feature_mode="geom3",
            support_feature_scale=1.0,
            support_weight_mode="edge_rinf_corner",
        )
    )

    inf_det = create_inference_model_from_full(vae, deterministic=True)
    inf_mc = create_inference_model_from_full(vae, deterministic=False)

    x = np.zeros((3, 2), dtype=np.float32)
    d = np.zeros((3, 1), dtype=np.float32)
    c = np.zeros((3, 1), dtype=np.float32)

    y_det = inf_det.predict([x, d, c], verbose=0)
    y_mc = inf_mc.predict([x, d, c], verbose=0)

    assert vae.get_layer("decoder").inputs[1].shape[-1] == 7
    assert y_det.shape == (3, 2)
    assert y_mc.shape == (3, 2)


def test_delta_residual_roundtrips_full_model_save_and_load(tmp_path):
    vae, _ = build_cvae(_point_cfg("delta_residual"))
    x = np.zeros((3, 2), dtype=np.float32)
    d = np.zeros((3, 1), dtype=np.float32)
    c = np.zeros((3, 1), dtype=np.float32)
    y = np.zeros((3, 2), dtype=np.float32)
    _ = vae([x, d, c, y], training=False)

    out = tmp_path / "delta_model.keras"
    _save_keras_model_compat(vae, out)

    loaded = load_seq_model(str(out))
    inf_det = create_inference_model_from_full(loaded, deterministic=True)
    inf_mc = create_inference_model_from_full(loaded, deterministic=False)

    y_det = inf_det.predict([x, d, c], verbose=0)
    y_mc = inf_mc.predict([x, d, c], verbose=0)

    assert loaded.get_layer("encoder") is not None
    assert loaded.get_layer("prior_net") is not None
    assert loaded.get_layer("decoder") is not None
    assert y_det.shape == (3, 2)
    assert y_mc.shape == (3, 2)


def test_delta_residual_inference_zero_head_returns_identity_signal():
    vae, _ = build_cvae(_point_cfg("delta_residual"))
    decoder = vae.get_layer("decoder")
    _zero_dense_head(decoder, "delta_output_params")

    inf_det = create_inference_model_from_full(vae, deterministic=True)

    x = np.array([[1.5, -0.25], [-0.4, 0.8]], dtype=np.float32)
    d = np.zeros((2, 1), dtype=np.float32)
    c = np.zeros((2, 1), dtype=np.float32)

    y_det = inf_det.predict([x, d, c], verbose=0)
    np.testing.assert_allclose(y_det, x, atol=1e-6)


def test_unknown_arch_variant_raises():
    with pytest.raises(ValueError, match="Unknown arch_variant"):
        build_decoder(layer_sizes=[8], latent_dim=2, arch_variant="nope")


def test_build_cvae_and_inference_support_legacy_2025_variant():
    vae, _ = build_cvae(_legacy_cfg())

    inf_det = create_inference_model_from_full(vae, deterministic=True)
    inf_mc = create_inference_model_from_full(vae, deterministic=False)

    x = np.zeros((3, 2), dtype=np.float32)
    d = np.zeros((3, 1), dtype=np.float32)
    c = np.zeros((3, 1), dtype=np.float32)

    y_det = inf_det.predict([x, d, c], verbose=0)
    y_mc = inf_mc.predict([x, d, c], verbose=0)

    assert vae.name == "cvae_legacy_2025_zero_y"
    assert y_det.shape == (3, 2)
    assert y_mc.shape == (3, 2)


def test_legacy_2025_encoder_ignores_y_numerically():
    vae, _ = build_cvae(_legacy_cfg())
    encoder = vae.get_layer("encoder")

    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 2)).astype(np.float32)
    d = rng.uniform(size=(5, 1)).astype(np.float32)
    c = rng.uniform(size=(5, 1)).astype(np.float32)
    y1 = rng.normal(size=(5, 2)).astype(np.float32)
    y2 = rng.normal(size=(5, 2)).astype(np.float32)

    mu1, lv1 = encoder.predict([x, d, c, y1], verbose=0)
    mu2, lv2 = encoder.predict([x, d, c, y2], verbose=0)

    np.testing.assert_allclose(mu1, mu2, atol=1e-6)
    np.testing.assert_allclose(lv1, lv2, atol=1e-6)


def test_legacy_2025_prior_matches_encoder_with_zero_y():
    vae, _ = build_cvae(_legacy_cfg())
    encoder = vae.get_layer("encoder")
    prior = vae.get_layer("prior_net")

    rng = np.random.default_rng(11)
    x = rng.normal(size=(6, 2)).astype(np.float32)
    d = rng.uniform(size=(6, 1)).astype(np.float32)
    c = rng.uniform(size=(6, 1)).astype(np.float32)
    y_zero = np.zeros((6, 2), dtype=np.float32)

    mu_q, lv_q = encoder.predict([x, d, c, y_zero], verbose=0)
    mu_p, lv_p = prior.predict([x, d, c], verbose=0)

    np.testing.assert_allclose(mu_q, mu_p, atol=1e-6)
    np.testing.assert_allclose(lv_q, lv_p, atol=1e-6)


def test_legacy_2025_nonzero_free_bits_raises():
    with pytest.raises(ValueError, match="does not support free_bits"):
        build_cvae(_legacy_cfg(free_bits=0.05))


def test_legacy_2025_latent_diagnostics_mark_kl_q_to_p_as_na():
    z_mean_q = np.zeros((4, 3), dtype=np.float32)
    z_log_var_q = np.zeros((4, 3), dtype=np.float32)
    z_mean_p = np.ones((4, 3), dtype=np.float32)
    z_log_var_p = np.zeros((4, 3), dtype=np.float32)

    diag = compute_latent_diagnostics(
        z_mean_q,
        z_log_var_q,
        z_mean_p,
        z_log_var_p,
        arch_variant="legacy_2025_zero_y",
    )

    assert np.isnan(diag["kl_qp_total_mean"])
    assert np.isnan(diag["df_lat"]["kl_q_to_p_dim_mean"]).all()
    assert diag["lat_summary"]["kl_q_to_p_applicable"] is False
    assert diag["lat_summary"]["latent_prior_semantics"] == "std_normal_legacy_2025_zero_y"
