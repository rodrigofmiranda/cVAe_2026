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
CondPriorDeltaVAELoss    Explicit residual-target Keras layer
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
# Mini-batch MMD² (TF ops — safe inside training graph)
# ======================================================================

def mmd2_tf(
    r_real: tf.Tensor,
    r_gen: tf.Tensor,
    n_sub: int = 512,
    bandwidth: float | None = None,
) -> tf.Tensor:
    """Unbiased mini-batch MMD² with RBF kernel, in pure TF ops.

    Parameters
    ----------
    r_real : (N, 2) — real channel residuals  (Y_real − X)
    r_gen  : (N, 2) — model residuals         (Y_pred − X)
    n_sub  : number of samples to sub-sample per call (default 512)
    bandwidth : RBF bandwidth σ². None → median heuristic computed inline.

    Returns
    -------
    scalar Tensor — unbiased MMD²
    """
    N = tf.shape(r_real)[0]
    n = tf.minimum(n_sub, N)

    # Independent sub-samples from real and generated pools
    idx_r = tf.random.shuffle(tf.range(N))[:n]
    idx_g = tf.random.shuffle(tf.range(N))[:n]
    x = tf.cast(tf.gather(r_real, idx_r), tf.float32)
    y = tf.cast(tf.gather(r_gen,  idx_g), tf.float32)

    def _sq_dists(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        aa = tf.reduce_sum(tf.square(a), axis=1, keepdims=True)
        bb = tf.reduce_sum(tf.square(b), axis=1, keepdims=True)
        ab = tf.matmul(a, b, transpose_b=True)
        return tf.maximum(aa + tf.transpose(bb) - 2.0 * ab, 0.0)

    if bandwidth is None:
        # Median heuristic on cross-set distances
        d2_xy = _sq_dists(x, y)
        flat = tf.reshape(d2_xy, [-1])
        mid = tf.cast(tf.shape(flat)[0] // 2, tf.int32)
        bw = tf.maximum(tf.sort(flat)[mid], 1e-3)
    else:
        bw = tf.constant(float(bandwidth), dtype=tf.float32)

    def _K(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.exp(-_sq_dists(a, b) / (2.0 * bw))

    Kxx = _K(x, x)
    Kyy = _K(y, y)
    Kxy = _K(x, y)

    nf = tf.cast(n, tf.float32)
    mask = 1.0 - tf.eye(n)
    term_xx = tf.reduce_sum(Kxx * mask) / (nf * (nf - 1.0))
    term_yy = tf.reduce_sum(Kyy * mask) / (nf * (nf - 1.0))
    term_xy = tf.reduce_sum(Kxy) / (nf * nf)
    return term_xx + term_yy - 2.0 * term_xy


def _resolve_mmd_mode(mode: str | None) -> str:
    """Return the canonical MMD residual-matching mode."""
    mode_norm = str(mode or "mean_residual").strip().lower()
    aliases = {
        "mean": "mean_residual",
        "mean_residual": "mean_residual",
        "sample": "sampled_residual",
        "sampled": "sampled_residual",
        "sampled_residual": "sampled_residual",
    }
    if mode_norm not in aliases:
        raise ValueError(
            "mmd_mode must be one of {'mean_residual', 'sampled_residual'}; "
            f"got {mode!r}"
        )
    return aliases[mode_norm]


def _sample_heteroscedastic(mean: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
    """Draw a differentiable sample from N(mean, diag(exp(log_var)))."""
    log_var = tf.clip_by_value(
        log_var, DECODER_LOGVAR_CLAMP_LO, DECODER_LOGVAR_CLAMP_HI
    )
    std = tf.exp(0.5 * log_var)
    eps = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
    return mean + std * eps


# ======================================================================
# Keras layer (used inside training graph — wraps the above functions)
# ======================================================================
@tf.keras.utils.register_keras_serializable(package="VLC")
class CondPriorVAELoss(layers.Layer):
    """Heteroscedastic Gaussian NLL + KL(q‖p) with β-annealing, free-bits,
    and optional auxiliary MMD² loss term.

    Inputs (call):
        (y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p)
        or, when lambda_mmd > 0:
        (y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p, x_center)

    The layer adds the total loss via ``self.add_loss`` and tracks
    ``recon_loss`` / ``kl_loss`` (and ``mmd_loss`` when active) as Keras metrics.
    """

    def __init__(
        self,
        beta: float = 1.0,
        free_bits: float = 0.0,
        lambda_mmd: float = 0.0,
        mmd_mode: str = "mean_residual",
        mmd_bandwidth: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beta_init = float(beta)
        self.free_bits = float(free_bits)
        self.lambda_mmd = float(lambda_mmd)
        self.mmd_mode = _resolve_mmd_mode(mmd_mode)
        self.mmd_bandwidth = mmd_bandwidth
        self.beta = tf.Variable(
            self.beta_init, trainable=False, dtype=tf.float32, name="beta",
        )
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        if self.lambda_mmd > 0.0:
            self.mmd_loss_tracker = tf.keras.metrics.Mean(name="mmd_loss")

    def call(self, inputs):
        if len(inputs) == 7:
            y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p, x_center = inputs
        else:
            y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p = inputs
            x_center = None

        y_mean = out_params[:, :2]
        y_log_var = out_params[:, 2:]

        recon = reconstruction_loss(y_true, y_mean, y_log_var)

        kl_per_sample = kl_divergence(
            z_mean_q, z_log_var_q, z_mean_p, z_log_var_p,
        )
        kl_fb = kl_with_freebits(kl_per_sample, self.free_bits)
        kl = tf.reduce_mean(kl_fb)

        total = compute_total_loss(recon, kl, self.beta)

        if self.lambda_mmd > 0.0 and x_center is not None:
            r_real = tf.stop_gradient(y_true - x_center)
            if self.mmd_mode == "sampled_residual":
                y_sample = _sample_heteroscedastic(y_mean, y_log_var)
                r_gen = y_sample - x_center
            else:
                r_gen = y_mean - x_center
            mmd2 = mmd2_tf(r_real, r_gen, n_sub=512, bandwidth=self.mmd_bandwidth)
            self.mmd_loss_tracker.update_state(mmd2)
            total = total + self.lambda_mmd * mmd2

        self.add_loss(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(tf.reduce_mean(kl_per_sample))
        return y_mean

    @property
    def metrics(self):
        m = [self.recon_loss_tracker, self.kl_loss_tracker]
        if self.lambda_mmd > 0.0:
            m.append(self.mmd_loss_tracker)
        return m

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "beta": self.beta_init,
            "free_bits": self.free_bits,
            "lambda_mmd": self.lambda_mmd,
            "mmd_mode": self.mmd_mode,
            "mmd_bandwidth": self.mmd_bandwidth,
        })
        return cfg


@tf.keras.utils.register_keras_serializable(package="VLC")
class CondPriorDeltaVAELoss(layers.Layer):
    """Explicit residual-target loss for ``delta_residual`` point-wise models.

    The decoder outputs residual parameters ``(Δ_mean, Δ_log_var)`` while the
    training target remains the received signal ``y_true``. This layer converts
    to the residual target ``Δ_true = y_true - x_true`` internally, optimises
    that heteroscedastic NLL, and returns ``y_mean = x_true + Δ_mean`` so the
    external model contract remains unchanged.

    Inputs (call):
        (y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p, x_true)
    """

    def __init__(
        self,
        beta: float = 1.0,
        free_bits: float = 0.0,
        lambda_mmd: float = 0.0,
        mmd_mode: str = "mean_residual",
        mmd_bandwidth: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beta_init = float(beta)
        self.free_bits = float(free_bits)
        self.lambda_mmd = float(lambda_mmd)
        self.mmd_mode = _resolve_mmd_mode(mmd_mode)
        self.mmd_bandwidth = mmd_bandwidth
        self.beta = tf.Variable(
            self.beta_init, trainable=False, dtype=tf.float32, name="beta",
        )
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        if self.lambda_mmd > 0.0:
            self.mmd_loss_tracker = tf.keras.metrics.Mean(name="mmd_loss")

    def call(self, inputs):
        if len(inputs) < 7:
            raise ValueError(
                "CondPriorDeltaVAELoss expects "
                "(y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p, x_true)."
            )

        y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p, x_true = inputs[:7]

        delta_true = y_true - x_true
        delta_mean = out_params[:, :2]
        delta_log_var = out_params[:, 2:]

        recon = reconstruction_loss(delta_true, delta_mean, delta_log_var)

        kl_per_sample = kl_divergence(
            z_mean_q, z_log_var_q, z_mean_p, z_log_var_p,
        )
        kl_fb = kl_with_freebits(kl_per_sample, self.free_bits)
        kl = tf.reduce_mean(kl_fb)

        total = compute_total_loss(recon, kl, self.beta)

        if self.lambda_mmd > 0.0:
            if self.mmd_mode == "sampled_residual":
                delta_gen = _sample_heteroscedastic(delta_mean, delta_log_var)
            else:
                delta_gen = delta_mean
            mmd2 = mmd2_tf(
                tf.stop_gradient(delta_true),
                delta_gen,
                n_sub=512,
                bandwidth=self.mmd_bandwidth,
            )
            self.mmd_loss_tracker.update_state(mmd2)
            total = total + self.lambda_mmd * mmd2

        self.add_loss(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(tf.reduce_mean(kl_per_sample))
        return x_true + delta_mean

    @property
    def metrics(self):
        m = [self.recon_loss_tracker, self.kl_loss_tracker]
        if self.lambda_mmd > 0.0:
            m.append(self.mmd_loss_tracker)
        return m

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "beta": self.beta_init,
            "free_bits": self.free_bits,
            "lambda_mmd": self.lambda_mmd,
            "mmd_mode": self.mmd_mode,
            "mmd_bandwidth": self.mmd_bandwidth,
        })
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
