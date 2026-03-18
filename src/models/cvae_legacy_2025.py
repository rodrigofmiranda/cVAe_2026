# -*- coding: utf-8 -*-
"""
src/models/cvae_legacy_2025 — Legacy 2025 heteroscedastic point-wise VAE.

This module ports the effective 2025 architecture into the 2026 pipeline as
``arch_variant="legacy_2025_zero_y"``.

Behavioral contract
-------------------
- point-wise model, no temporal context
- heteroscedastic decoder output: (mean_I, mean_Q, logvar_I, logvar_Q)
- KL against N(0, I), not against a learned conditional prior
- encoder keeps the 4-input cVAE-compatible interface [x, d, c, y] but
  explicitly ignores y by replacing it with zeros internally
- prior_net mirrors that same mapping with zero-filled legacy y slots, so
  ``prior_net([x,d,c])`` matches ``encoder([x,d,c,zeros])`` exactly
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

from src.models.callbacks import KLAnnealingCallback
from src.models.losses import StdNormalHeteroscedasticVAELoss
from src.models.sampling import Sampling


def _activation_layer(name: str):
    """Return a Keras activation layer (supports ``leaky_relu``)."""
    name = (name or "").lower().strip()
    if name in ("leaky_relu", "lrelu", "leakyrelu"):
        return layers.LeakyReLU(alpha=0.2)
    return layers.Activation(name)


def build_legacy_2025_latent_core(cfg: Dict) -> tf.keras.Model:
    """Shared latent MLP that reproduces the effective 2025 topology."""
    layer_sizes = list(cfg["layer_sizes"])
    latent_dim = int(cfg["latent_dim"])
    activation = cfg.get("activation", "leaky_relu")
    dropout = float(cfg.get("dropout", 0.0))

    x_in = layers.Input(shape=(2,), name="legacy_core_x")
    d_in = layers.Input(shape=(1,), name="legacy_core_d")
    c_in = layers.Input(shape=(1,), name="legacy_core_c")
    y_legacy_in = layers.Input(shape=(2,), name="legacy_core_y_legacy")

    h = layers.Concatenate(name="legacy_core_concat")([x_in, d_in, c_in, y_legacy_in])
    h = layers.Dense(
        layer_sizes[-1],
        kernel_initializer="glorot_uniform",
        name="legacy_core_dense_in",
    )(h)
    h = layers.LayerNormalization(name="legacy_core_ln_in")(h)
    h = _activation_layer(activation)(h)

    skip_connections = []
    for i, units in enumerate(reversed(layer_sizes[:-1])):
        h = layers.Dense(
            units,
            kernel_initializer="glorot_uniform",
            name=f"legacy_core_dense_{i}",
        )(h)
        h = layers.BatchNormalization(name=f"legacy_core_bn_{i}")(h)
        h = _activation_layer(activation)(h)
        if i < len(layer_sizes) - 2 and dropout > 0.0:
            h = layers.Dropout(dropout, name=f"legacy_core_drop_{i}")(h)
        if i % 2 == 0 and i > 0:
            skip_connections.append(h)

    if skip_connections:
        h = layers.Concatenate(name="legacy_core_skip_concat")(
            [h] + skip_connections[-2:]
        )
        h = layers.Dense(
            layer_sizes[0],
            kernel_initializer="glorot_uniform",
            name="legacy_core_skip_dense",
        )(h)
        h = _activation_layer(activation)(h)
        h = layers.BatchNormalization(name="legacy_core_skip_bn")(h)

    z_mean = layers.Dense(
        latent_dim, name="legacy_core_z_mean", kernel_initializer="glorot_uniform",
    )(h)
    z_log_var = layers.Dense(
        latent_dim, name="legacy_core_z_log_var", kernel_initializer="glorot_uniform",
    )(h)
    return models.Model(
        [x_in, d_in, c_in, y_legacy_in],
        [z_mean, z_log_var],
        name="legacy_2025_latent_core",
    )


def build_legacy_2025_encoder(
    cfg: Dict,
    *,
    core: tf.keras.Model | None = None,
) -> tf.keras.Model:
    """Build encoder q(z | x, d, c, y) that ignores y by zero-filling it."""
    core = core or build_legacy_2025_latent_core(cfg)

    x_in = layers.Input(shape=(2,), name="encoder_x")
    d_in = layers.Input(shape=(1,), name="encoder_d")
    c_in = layers.Input(shape=(1,), name="encoder_c")
    y_in = layers.Input(shape=(2,), name="encoder_y")

    zero_y = layers.Subtract(name="encoder_zero_y")([y_in, y_in])
    z_mean, z_log_var = core([x_in, d_in, c_in, zero_y])
    z_mean = layers.Activation("linear", name="q_z_mean")(z_mean)
    z_log_var = layers.Activation("linear", name="q_z_log_var")(z_log_var)

    return models.Model([x_in, d_in, c_in, y_in], [z_mean, z_log_var], name="encoder")


def build_legacy_2025_prior_net(
    cfg: Dict,
    *,
    core: tf.keras.Model | None = None,
) -> tf.keras.Model:
    """Build compatibility prior_net mirroring the legacy encoder mapping."""
    core = core or build_legacy_2025_latent_core(cfg)

    x_in = layers.Input(shape=(2,), name="prior_x")
    d_in = layers.Input(shape=(1,), name="prior_d")
    c_in = layers.Input(shape=(1,), name="prior_c")

    zero_y = layers.Subtract(name="prior_zero_y")([x_in, x_in])
    z_mean, z_log_var = core([x_in, d_in, c_in, zero_y])
    z_mean = layers.Activation("linear", name="p_z_mean")(z_mean)
    z_log_var = layers.Activation("linear", name="p_z_log_var")(z_log_var)

    return models.Model([x_in, d_in, c_in], [z_mean, z_log_var], name="prior_net")


def build_legacy_2025_decoder(
    layer_sizes: Sequence[int],
    latent_dim: int,
    activation: str = "leaky_relu",
    dropout: float = 0.0,
) -> tf.keras.Model:
    """Build the legacy 2025 heteroscedastic decoder with light skip paths."""
    z_in = layers.Input(shape=(latent_dim,), name="z_input")
    cond_in = layers.Input(shape=(4,), name="cond_input")

    h = layers.Concatenate(name="legacy_dec_concat")([z_in, cond_in])
    h = layers.Dense(
        layer_sizes[0],
        kernel_initializer="glorot_uniform",
        name="legacy_dec_dense_in",
    )(h)
    h = layers.BatchNormalization(name="legacy_dec_bn_in")(h)
    h = _activation_layer(activation)(h)

    skip_connections = []
    for i, units in enumerate(layer_sizes[1:]):
        h = layers.Dense(
            units,
            kernel_initializer="glorot_uniform",
            name=f"legacy_dec_dense_{i}",
        )(h)
        h = layers.BatchNormalization(name=f"legacy_dec_bn_{i}")(h)
        h = _activation_layer(activation)(h)
        if i < len(layer_sizes) - 2 and dropout > 0.0:
            h = layers.Dropout(dropout, name=f"legacy_dec_drop_{i}")(h)
        if i % 2 == 0 and i > 0:
            skip_connections.append(h)

    if skip_connections:
        h = layers.Concatenate(name="legacy_dec_skip_concat")(
            [h] + skip_connections[-2:]
        )
        h = layers.Dense(
            layer_sizes[-1],
            kernel_initializer="glorot_uniform",
            name="legacy_dec_skip_dense",
        )(h)
        h = _activation_layer(activation)(h)
        h = layers.BatchNormalization(name="legacy_dec_skip_bn")(h)

    out = layers.Dense(
        4, name="output_params", kernel_initializer="glorot_uniform",
    )(h)
    return models.Model([z_in, cond_in], out, name="decoder")


def build_legacy_2025_cvae(
    cfg: Dict,
) -> Tuple[tf.keras.Model, "KLAnnealingCallback"]:
    """Build the full legacy-2025 point-wise VAE inside the 2026 interface."""
    layer_sizes = list(cfg["layer_sizes"])
    latent_dim = int(cfg["latent_dim"])
    beta = float(cfg["beta"])
    lr = float(cfg["lr"])
    dropout = float(cfg.get("dropout", 0.0))
    free_bits = float(cfg.get("free_bits", 0.0))
    kl_anneal_epochs = int(cfg.get("kl_anneal_epochs", 50))
    activation = cfg.get("activation", "leaky_relu")

    if abs(free_bits) > 1e-12:
        raise ValueError(
            "arch_variant='legacy_2025_zero_y' does not support free_bits; "
            f"expected 0.0, got {free_bits}."
        )

    latent_core = build_legacy_2025_latent_core(cfg)
    encoder = build_legacy_2025_encoder(cfg, core=latent_core)
    prior_net = build_legacy_2025_prior_net(cfg, core=latent_core)
    decoder = build_legacy_2025_decoder(
        layer_sizes=layer_sizes,
        latent_dim=latent_dim,
        activation=activation,
        dropout=dropout,
    )

    x_in = layers.Input(shape=(2,), name="x_input")
    d_in = layers.Input(shape=(1,), name="distance_input")
    c_in = layers.Input(shape=(1,), name="current_input")
    y_in = layers.Input(shape=(2,), name="y_true")

    z_mean_q, z_log_var_q = encoder([x_in, d_in, c_in, y_in])
    z_mean_p, z_log_var_p = prior_net([x_in, d_in, c_in])
    z = Sampling(name="sampling")([z_mean_q, z_log_var_q])

    cond = layers.Concatenate(name="cond_concat")([x_in, d_in, c_in])
    out_params = decoder([z, cond])

    loss_layer = StdNormalHeteroscedasticVAELoss(
        beta=0.0, name="stdnormal_loss",
    )
    y_mean = loss_layer(
        [y_in, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p],
    )

    vae = models.Model(
        [x_in, d_in, c_in, y_in],
        y_mean,
        name="cvae_legacy_2025_zero_y",
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    vae.compile(optimizer=opt)

    kl_cb = KLAnnealingCallback(
        loss_layer,
        beta_start=0.0,
        beta_end=beta,
        annealing_epochs=kl_anneal_epochs,
    )
    return vae, kl_cb
