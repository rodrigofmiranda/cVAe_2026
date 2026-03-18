# -*- coding: utf-8 -*-
"""
src/models/losses.py — Loss functions for the heteroscedastic cVAE.

Extracted from ``cvae_components.py`` (refactor step 3).

The formulas are **identical** to the monolith — no algorithmic changes.

Public API
----------
reconstruction_loss      Heteroscedastic Gaussian NLL
kl_divergence            KL(q ‖ p) per sample
kl_to_standard_normal    KL(q ‖ N(0, I)) per sample
kl_with_freebits         Free-bits thresholded KL
compute_total_loss        recon + β · min(kl, cap)
CondPriorVAELoss         Keras layer (for training graph)
StdNormalHeteroscedasticVAELoss
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.config.defaults import (
    DECODER_LOGVAR_CLAMP_HI,
    DECODER_LOGVAR_CLAMP_LO,
)


# ======================================================================
# Standalone functional forms (TF tensors — usable in eager or graph)
# ======================================================================

def reconstruction_loss(
    y_true: tf.Tensor,
    y_mean: tf.Tensor,
    y_log_var: tf.Tensor,
) -> tf.Tensor:
    """Heteroscedastic Gaussian NLL (mean over batch).

    .. math::
        \\mathrm{NLL} = \\frac{1}{2}\\sum_d
        \\bigl(\\log\\sigma^2_d + (y_d - \\mu_d)^2 / \\sigma^2_d
        + \\log 2\\pi \\bigr)

    Parameters
    ----------
    y_true : (N, 2)
    y_mean : (N, 2)
    y_log_var : (N, 2)  — clipped externally if needed.

    Returns
    -------
    scalar Tensor — mean NLL over batch.
    """
    y_log_var = tf.clip_by_value(
        y_log_var, DECODER_LOGVAR_CLAMP_LO, DECODER_LOGVAR_CLAMP_HI
    )
    y_var = tf.exp(y_log_var) + 1e-6
    nll = 0.5 * tf.reduce_sum(
        y_log_var + tf.square(y_true - y_mean) / y_var + tf.math.log(2.0 * np.pi),
        axis=-1,
    )
    return tf.reduce_mean(nll)


def kl_divergence(
    z_mean_q: tf.Tensor,
    z_log_var_q: tf.Tensor,
    z_mean_p: tf.Tensor,
    z_log_var_p: tf.Tensor,
) -> tf.Tensor:
    """KL(q ‖ p) per sample, where q and p are diagonal Gaussians.

    Returns
    -------
    kl_per_sample : (N,)
    """
    vq = tf.exp(tf.clip_by_value(z_log_var_q, -20.0, 20.0))
    vp = tf.exp(tf.clip_by_value(z_log_var_p, -20.0, 20.0))
    kl_dim = 0.5 * (
        tf.math.log(vp + 1e-12) - tf.math.log(vq + 1e-12)
        + (vq + tf.square(z_mean_q - z_mean_p)) / (vp + 1e-12)
        - 1.0
    )
    return tf.reduce_sum(kl_dim, axis=-1)


def kl_with_freebits(
    kl_per_sample: tf.Tensor,
    free_bits: float = 0.0,
) -> tf.Tensor:
    """Apply free-bits thresholding to per-sample KL.

    Returns
    -------
    kl_fb : (N,)  — ``max(kl - free_bits, 0)``
    """
    fb = tf.cast(free_bits, kl_per_sample.dtype)
    return tf.maximum(kl_per_sample - fb, 0.0)


def kl_to_standard_normal(
    z_mean_q: tf.Tensor,
    z_log_var_q: tf.Tensor,
) -> tf.Tensor:
    """KL(q ‖ N(0, I)) per sample for a diagonal Gaussian q.

    Returns
    -------
    kl_per_sample : (N,)
    """
    z_log_var_q = tf.clip_by_value(z_log_var_q, -20.0, 20.0)
    kl_dim = 0.5 * (
        tf.exp(z_log_var_q) + tf.square(z_mean_q) - 1.0 - z_log_var_q
    )
    return tf.reduce_sum(kl_dim, axis=-1)


def compute_total_loss(
    recon: tf.Tensor,
    kl: tf.Tensor,
    beta: float | tf.Tensor,
    kl_cap: float = 200.0,
) -> tf.Tensor:
    """Total ELBO loss = recon + β · min(kl, cap).

    Parameters
    ----------
    recon : scalar — mean reconstruction NLL.
    kl    : scalar — mean KL.
    beta  : current β weight.
    kl_cap : safety clamp.

    Returns
    -------
    scalar Tensor.
    """
    return recon + beta * tf.minimum(kl, kl_cap)


# ======================================================================
# Keras layer (used inside training graph — wraps the above functions)
# ======================================================================
@tf.keras.utils.register_keras_serializable(package="VLC")
class CondPriorVAELoss(layers.Layer):
    """Heteroscedastic Gaussian NLL + KL(q‖p) with β-annealing and free-bits.

    Inputs (call):
        (y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p)

    The layer adds the total loss via ``self.add_loss`` and tracks
    ``recon_loss`` / ``kl_loss`` as Keras metrics.
    """

    def __init__(self, beta: float = 1.0, free_bits: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.beta_init = float(beta)
        self.free_bits = float(free_bits)
        self.beta = tf.Variable(
            self.beta_init, trainable=False, dtype=tf.float32, name="beta",
        )
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p = inputs

        y_mean = out_params[:, :2]
        y_log_var = out_params[:, 2:]

        recon = reconstruction_loss(y_true, y_mean, y_log_var)

        kl_per_sample = kl_divergence(
            z_mean_q, z_log_var_q, z_mean_p, z_log_var_p,
        )
        kl_fb = kl_with_freebits(kl_per_sample, self.free_bits)
        kl = tf.reduce_mean(kl_fb)

        total = compute_total_loss(recon, kl, self.beta)

        self.add_loss(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(tf.reduce_mean(kl_per_sample))
        return y_mean

    @property
    def metrics(self):
        return [self.recon_loss_tracker, self.kl_loss_tracker]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"beta": self.beta_init, "free_bits": self.free_bits})
        return cfg


@tf.keras.utils.register_keras_serializable(package="VLC")
class StdNormalHeteroscedasticVAELoss(layers.Layer):
    """Heteroscedastic Gaussian NLL + KL(q‖N(0,I)) with β-annealing.

    Inputs (call):
        (y_true, out_params, z_mean_q, z_log_var_q, *ignored)

    Any extra tensors are accepted and ignored. This lets the training graph
    keep compatibility-only submodels (for example ``prior_net`` in the
    legacy-2025 variant) connected without changing the actual loss formula.
    """

    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta_init = float(beta)
        self.beta = tf.Variable(
            self.beta_init, trainable=False, dtype=tf.float32, name="beta",
        )
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        if len(inputs) < 4:
            raise ValueError(
                "StdNormalHeteroscedasticVAELoss expects at least "
                "(y_true, out_params, z_mean_q, z_log_var_q)."
            )
        y_true, out_params, z_mean_q, z_log_var_q = inputs[:4]

        y_mean = out_params[:, :2]
        y_log_var = out_params[:, 2:]

        recon = reconstruction_loss(y_true, y_mean, y_log_var)
        kl_per_sample = kl_to_standard_normal(z_mean_q, z_log_var_q)
        kl = tf.reduce_mean(kl_per_sample)

        total = compute_total_loss(recon, kl, self.beta)

        self.add_loss(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return y_mean

    @property
    def metrics(self):
        return [self.recon_loss_tracker, self.kl_loss_tracker]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"beta": self.beta_init})
        return cfg
