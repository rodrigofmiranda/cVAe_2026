# -*- coding: utf-8 -*-
"""
src/models/cvae.py — cVAE model construction (encoder / decoder / prior / full).

Extracted from ``cvae_components.py`` (refactor step 3).
The architecture is **identical** to the monolith — no algorithmic changes.

Public API
----------
build_encoder                 Encoder MLP  q(z | x, d, c, y)
build_decoder                 Decoder MLP  p(y | z, x, d, c)
build_prior_net               Conditional prior  p(z | x, d, c)
build_cvae                    Full cVAE (compile + KL callback)
build_condprior_cvae          Alias for backward compatibility
create_inference_model_from_full   Inference graph from saved model
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config.defaults import (
    DECODER_LOGVAR_CLAMP_HI,
    DECODER_LOGVAR_CLAMP_LO,
)
from src.models.sampling import Sampling
from src.models.losses import CondPriorVAELoss
from src.models.callbacks import KLAnnealingCallback


# ======================================================================
# Activation helper
# ======================================================================
def _activation_layer(name: str):
    """Return a Keras activation layer (supports ``leaky_relu``)."""
    name = (name or "").lower().strip()
    if name in ("leaky_relu", "lrelu", "leakyrelu"):
        return layers.LeakyReLU(alpha=0.2)
    return layers.Activation(name)


# ======================================================================
# MLP helper (shared by encoder + prior)
# ======================================================================
def build_mlp(
    name: str,
    in_shapes: Sequence[tuple],
    layer_sizes: Sequence[int],
    activation: str = "leaky_relu",
    dropout: float = 0.0,
    out_dim: int = 32,
    out_name_prefix: str = "",
) -> tf.keras.Model:
    """Generic MLP that outputs ``(z_mean, z_log_var)``."""
    ins = [
        layers.Input(shape=s, name=f"{name}_in_{i}")
        for i, s in enumerate(in_shapes)
    ]
    h = layers.Concatenate(name=f"{name}_concat")(ins)
    for i, u in enumerate(layer_sizes):
        h = layers.Dense(
            u, kernel_initializer="glorot_uniform", name=f"{name}_dense_{i}",
        )(h)
        h = layers.BatchNormalization(name=f"{name}_bn_{i}")(h)
        h = _activation_layer(activation)(h)
        if dropout and dropout > 0:
            h = layers.Dropout(dropout, name=f"{name}_drop_{i}")(h)
    mu = layers.Dense(out_dim, name=f"{out_name_prefix}z_mean")(h)
    lv = layers.Dense(out_dim, name=f"{out_name_prefix}z_log_var")(h)
    return models.Model(ins, [mu, lv], name=name)


# ======================================================================
# Encoder / Prior / Decoder
# ======================================================================
def build_encoder(cfg: Dict) -> tf.keras.Model:
    """Build the encoder q(z | x, d, c, y).

    Parameters
    ----------
    cfg : dict with keys ``layer_sizes, latent_dim, activation, dropout``.

    Returns
    -------
    keras.Model  inputs=[x(2), d(1), c(1), y(2)] → [z_mean, z_log_var]
    """
    return build_mlp(
        name="encoder",
        in_shapes=[(2,), (1,), (1,), (2,)],
        layer_sizes=cfg["layer_sizes"],
        activation=cfg.get("activation", "leaky_relu"),
        dropout=float(cfg.get("dropout", 0.0)),
        out_dim=int(cfg["latent_dim"]),
        out_name_prefix="q_",
    )


def build_prior_net(cfg: Dict) -> tf.keras.Model:
    """Build the conditional prior p(z | x, d, c).

    Parameters
    ----------
    cfg : dict with keys ``layer_sizes, latent_dim, activation, dropout``.

    Returns
    -------
    keras.Model  inputs=[x(2), d(1), c(1)] → [z_mean, z_log_var]
    """
    return build_mlp(
        name="prior_net",
        in_shapes=[(2,), (1,), (1,)],
        layer_sizes=cfg["layer_sizes"],
        activation=cfg.get("activation", "leaky_relu"),
        dropout=float(cfg.get("dropout", 0.0)),
        out_dim=int(cfg["latent_dim"]),
        out_name_prefix="p_",
    )


def build_decoder(
    layer_sizes: Sequence[int],
    latent_dim: int,
    activation: str = "leaky_relu",
    dropout: float = 0.0,
) -> tf.keras.Model:
    """Build the heteroscedastic decoder p(y | z, x, d, c).

    Output has 4 units: ``(mean_I, mean_Q, logvar_I, logvar_Q)``.
    """
    z_in = layers.Input(shape=(latent_dim,), name="z_input")
    cond_in = layers.Input(shape=(4,), name="cond_input")    # x(2)+d(1)+c(1)
    h = layers.Concatenate(name="dec_concat")([z_in, cond_in])
    for i, u in enumerate(layer_sizes):
        h = layers.Dense(
            u, kernel_initializer="glorot_uniform", name=f"dec_dense_{i}",
        )(h)
        h = layers.BatchNormalization(name=f"dec_bn_{i}")(h)
        h = _activation_layer(activation)(h)
        if dropout and dropout > 0:
            h = layers.Dropout(dropout, name=f"dec_drop_{i}")(h)
    out = layers.Dense(4, name="output_params")(h)
    return models.Model([z_in, cond_in], out, name="decoder")


# ======================================================================
# Full cVAE assembly + compile
# ======================================================================
def build_cvae(cfg: Dict) -> Tuple[tf.keras.Model, "KLAnnealingCallback"]:
    """Build, compile, and return the full cVAE with KL-annealing callback.

    This is functionally identical to the former ``build_condprior_cvae``
    in ``cvae_components.py``.

    Parameters
    ----------
    cfg : dict
        Must contain ``layer_sizes, latent_dim, beta, lr, dropout``.
        Optional: ``free_bits, kl_anneal_epochs, activation``.

    Returns
    -------
    (vae, kl_cb) : (keras.Model, KLAnnealingCallback)
    """
    layer_sizes = cfg["layer_sizes"]
    latent_dim = int(cfg["latent_dim"])
    beta = float(cfg["beta"])
    lr = float(cfg["lr"])
    dropout = float(cfg["dropout"])
    free_bits = float(cfg.get("free_bits", 0.0))
    kl_anneal_epochs = int(cfg.get("kl_anneal_epochs", 50))
    activation = cfg.get("activation", "leaky_relu")

    encoder = build_encoder(cfg)
    prior_net = build_prior_net(cfg)
    decoder = build_decoder(
        layer_sizes=layer_sizes, latent_dim=latent_dim,
        activation=activation, dropout=dropout,
    )

    # --- Graph wiring ---
    x_in = layers.Input(shape=(2,), name="x_input")
    d_in = layers.Input(shape=(1,), name="distance_input")
    c_in = layers.Input(shape=(1,), name="current_input")
    y_in = layers.Input(shape=(2,), name="y_true")

    z_mean_q, z_log_var_q = encoder([x_in, d_in, c_in, y_in])
    z_mean_p, z_log_var_p = prior_net([x_in, d_in, c_in])

    # C4.2 FIX: clip prior log-var in training graph too
    z_log_var_p = layers.Lambda(
        lambda t: tf.clip_by_value(t, -10.0, 10.0),
        name="clip_p_logvar_train",
    )(z_log_var_p)

    z = Sampling(name="sampling")([z_mean_q, z_log_var_q])

    cond = layers.Concatenate(name="cond_concat")([x_in, d_in, c_in])  # (N,4)
    out_params = decoder([z, cond])

    # Loss layer (β starts at 0 for annealing)
    beta_initial = 0.0
    loss_layer = CondPriorVAELoss(
        beta=beta_initial, free_bits=free_bits, name="condprior_loss",
    )
    y_mean = loss_layer(
        [y_in, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p],
    )

    vae = models.Model([x_in, d_in, c_in, y_in], y_mean, name="cvae_condprior")
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    vae.compile(optimizer=opt)

    kl_cb = KLAnnealingCallback(
        loss_layer, beta_start=0.0, beta_end=beta,
        annealing_epochs=kl_anneal_epochs,
    )
    return vae, kl_cb


# Backward-compatible alias
build_condprior_cvae = build_cvae


# ======================================================================
# Inference model (conditional prior) from saved full model
# ======================================================================
def create_inference_model_from_full(
    full_model: tf.keras.Model,
    deterministic: bool = True,
) -> tf.keras.Model:
    """Create an inference graph from a trained full-model checkpoint.

    Parameters
    ----------
    full_model : the compiled cVAE (contains encoder, prior, decoder).
    deterministic : if *True*, z = z_mean (MAP estimate).

    Returns
    -------
    keras.Model  inputs=[x, d, c] → y_predicted
    """
    prior = full_model.get_layer("prior_net")
    dec = full_model.get_layer("decoder")

    x_in = layers.Input(shape=(2,), name="x_input")
    d_in = layers.Input(shape=(1,), name="distance_input")
    c_in = layers.Input(shape=(1,), name="current_input")

    z_mean_p, z_log_var_p = prior([x_in, d_in, c_in])
    z_log_var_p = layers.Lambda(
        lambda t: tf.clip_by_value(t, -10.0, 10.0), name="clip_zlogvar",
    )(z_log_var_p)

    if deterministic:
        z = layers.Lambda(lambda t: t, name="z_det")(z_mean_p)
    else:
        eps_z = layers.Lambda(
            lambda t: tf.random.normal(tf.shape(t)), name="eps_z",
        )(z_mean_p)
        z = layers.Lambda(
            lambda a: a[0] + tf.exp(0.5 * a[1]) * a[2], name="sample_z",
        )([z_mean_p, z_log_var_p, eps_z])

    cond = layers.Concatenate(name="cond_concat_inf")([x_in, d_in, c_in])
    out_params = dec([z, cond])

    y_mean = layers.Lambda(lambda t: t[:, :2], name="y_mean")(out_params)
    y_log_var = layers.Lambda(
        lambda t: tf.clip_by_value(
            t[:, 2:], DECODER_LOGVAR_CLAMP_LO, DECODER_LOGVAR_CLAMP_HI
        ),
        name="y_logvar",
    )(out_params)

    if deterministic:
        y = layers.Lambda(lambda t: t, name="y_det")(y_mean)
    else:
        eps_y = layers.Lambda(
            lambda t: tf.random.normal(tf.shape(t)), name="eps_y",
        )(y_mean)
        y = layers.Lambda(
            lambda a: a[0] + tf.exp(0.5 * a[1]) * a[2], name="sample_y",
        )([y_mean, y_log_var, eps_y])

    name = "inference_condprior_det" if deterministic else "inference_condprior"
    return models.Model([x_in, d_in, c_in], y, name=name)
