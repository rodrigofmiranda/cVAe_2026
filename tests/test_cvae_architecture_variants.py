import numpy as np
import pytest

from src.models.cvae import (
    build_cvae,
    build_decoder,
    create_inference_model_from_full,
)


def _zero_dense_head(model, layer_name: str) -> None:
    layer = model.get_layer(layer_name)
    weights = layer.get_weights()
    assert len(weights) == 2
    kernel, bias = weights
    layer.set_weights([np.zeros_like(kernel), np.zeros_like(bias)])


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


def test_unknown_arch_variant_raises():
    with pytest.raises(ValueError, match="Unknown arch_variant"):
        build_decoder(layer_sizes=[8], latent_dim=2, arch_variant="nope")
