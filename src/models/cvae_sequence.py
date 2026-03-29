# -*- coding: utf-8 -*-
"""
src/models/cvae_sequence.py — Sequence-aware cVAE (seq_bigru_residual).

Implements the BiGRU-based conditional VAE that models::

    p(y_t | x_{t-k:t+k}, d, I)

instead of the current point-wise::

    p(y_t | x_t, d, I)

Architecture
------------
prior_net  : RepeatVector(d,c) → concat(x_window) → BiGRU → MLP → (z_mean_p, z_log_var_p)
encoder    : same BiGRU path + y_center → MLP → (z_mean_q, z_log_var_q)
decoder    : MLP(z, x_center, d, c) → residual y_mean + logvar

Residual formulation (inherits from channel_residual)::

    delta_mean = MLP(z, x_center, d, c)[:, :2]
    y_mean     = x_center + delta_mean

The decoder takes only x_center (not x_window) because the sequence context
is already captured in z via the encoder/prior BiGRU.  This keeps the decoder
interface close to the existing point-wise decoder.

Public API
----------
build_seq_cvae                  Full seq cVAE (same signature as build_cvae).
create_seq_inference_model      Inference graph from a saved seq cVAE.
load_seq_model                  Load a saved .keras file with correct custom_objects.
build_seq_prior_net             Prior sub-model only.
build_seq_encoder               Encoder sub-model only.
build_seq_decoder               Decoder sub-model only.
ExtractCenterFrame              Serializable layer: extract center timestep.
ClipValues                      Serializable layer: clip tensor to [lo, hi].
SliceFeatures                   Serializable layer: slice last axis [start:end].

Layer-name compatibility
------------------------
The sub-models are named ``encoder``, ``prior_net``, and ``decoder`` so that
the existing evaluation engine and protocol runner can locate them via
``full_model.get_layer(name)`` without modification.
"""

from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config.defaults import (
    DECODER_LOGVAR_CLAMP_HI,
    DECODER_LOGVAR_CLAMP_LO,
)
from src.models.callbacks import KLAnnealingCallback
from src.models.losses import (
    CondPriorDiffusionVAELoss,
    CondPriorDeltaVAELoss,
    CondPriorVAELoss,
    StdNormalHeteroscedasticVAELoss,
    _diffusion_predict_x0,
    _diffusion_schedule_arrays,
    _mdn_expected_mean,
    _sample_mdn,
)
from src.models.sampling import Sampling


# ======================================================================
# Serializable custom layers
# (registered so that save/load works without manual custom_objects dicts)
# ======================================================================

@tf.keras.utils.register_keras_serializable(package="seq_cvae")
class ExtractCenterFrame(tf.keras.layers.Layer):
    """Extract the center timestep from a (batch, W, features) 3-D tensor.

    Parameters
    ----------
    center_idx : int
        Index of the center timestep (typically ``window_size // 2``).
    """

    def __init__(self, center_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.center_idx = int(center_idx)

    def call(self, x):
        return x[:, self.center_idx, :]

    def get_config(self):
        return {**super().get_config(), "center_idx": self.center_idx}


@tf.keras.utils.register_keras_serializable(package="seq_cvae")
class ClipValues(tf.keras.layers.Layer):
    """Clip tensor values element-wise to [lo, hi].

    Parameters
    ----------
    lo, hi : float  Clip bounds.
    """

    def __init__(self, lo: float, hi: float, **kwargs):
        super().__init__(**kwargs)
        self.lo = float(lo)
        self.hi = float(hi)

    def call(self, x):
        return tf.clip_by_value(x, self.lo, self.hi)

    def get_config(self):
        return {**super().get_config(), "lo": self.lo, "hi": self.hi}


@tf.keras.utils.register_keras_serializable(package="seq_cvae")
class SliceFeatures(tf.keras.layers.Layer):
    """Slice along the last axis: ``t[:, start:end]``.

    Parameters
    ----------
    start : int   Start index (inclusive).
    end   : int   End index (exclusive).
    """

    def __init__(self, start: int, end: int, **kwargs):
        super().__init__(**kwargs)
        self.start = int(start)
        self.end = int(end)

    def call(self, x):
        return x[:, self.start:self.end]

    def get_config(self):
        return {**super().get_config(), "start": self.start, "end": self.end}


@tf.keras.utils.register_keras_serializable(package="seq_cvae")
class SampleDiffusionTrainingState(tf.keras.layers.Layer):
    """Sample one DDPM training state from a clean residual target."""

    def __init__(
        self,
        diffusion_steps: int = 8,
        diffusion_beta_start: float = 1e-4,
        diffusion_beta_end: float = 2e-2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diffusion_steps = int(diffusion_steps)
        self.diffusion_beta_start = float(diffusion_beta_start)
        self.diffusion_beta_end = float(diffusion_beta_end)
        _, _, alpha_bars = _diffusion_schedule_arrays(
            self.diffusion_steps,
            self.diffusion_beta_start,
            self.diffusion_beta_end,
        )
        self._alpha_bars = tf.constant(alpha_bars, dtype=tf.float32)

    def call(self, residual_true):
        residual_true = tf.cast(residual_true, tf.float32)
        batch = tf.shape(residual_true)[0]
        t = tf.random.uniform(
            shape=(batch,),
            minval=0,
            maxval=max(self.diffusion_steps, 1),
            dtype=tf.int32,
        )
        alpha_bar = tf.gather(self._alpha_bars, t)
        sqrt_alpha_bar = tf.expand_dims(tf.sqrt(alpha_bar), axis=-1)
        sqrt_one_minus_alpha_bar = tf.expand_dims(
            tf.sqrt(tf.maximum(1.0 - alpha_bar, 1e-6)),
            axis=-1,
        )
        eps = tf.random.normal(tf.shape(residual_true), dtype=residual_true.dtype)
        residual_noisy = sqrt_alpha_bar * residual_true + sqrt_one_minus_alpha_bar * eps

        denom = tf.maximum(float(self.diffusion_steps - 1), 1.0)
        t_frac = tf.expand_dims(tf.cast(t, tf.float32) / tf.cast(denom, tf.float32), axis=-1)
        return [residual_noisy, eps, t_frac, sqrt_alpha_bar, sqrt_one_minus_alpha_bar]

    def get_config(self):
        return {
            **super().get_config(),
            "diffusion_steps": self.diffusion_steps,
            "diffusion_beta_start": self.diffusion_beta_start,
            "diffusion_beta_end": self.diffusion_beta_end,
        }


# ======================================================================
# Internal helpers
# ======================================================================

def _activation_layer(name: str):
    """Return a Keras activation layer (supports ``leaky_relu``)."""
    name = (name or "").lower().strip()
    if name in ("leaky_relu", "lrelu", "leakyrelu"):
        return layers.LeakyReLU(alpha=0.2)
    return layers.Activation(name)


def _make_gru_layer(
    *,
    seq_hidden: int,
    return_sequences: bool,
    seq_gru_unroll: bool,
    seq_gru_backend: str,
    name: str,
):
    """Build one GRU layer using either the fused or compatibility backend.

    ``seq_gru_unroll=True`` keeps the original conservative path based on
    ``layers.GRU(..., unroll=True)``.

    ``seq_gru_unroll=False`` with ``seq_gru_backend="compat"`` forces a
    backend-neutral implementation using ``layers.RNN(GRUCell(...))``. This is
    slower than the fused path, but it avoids runtime dependence on cuDNN GRU
    kernels, which is necessary on some stacks (for example the RTX 5090 line
    that raised `DoRnnForward` errors).
    """
    backend = str(seq_gru_backend or "fused").strip().lower()
    if bool(seq_gru_unroll):
        return layers.GRU(
            seq_hidden,
            return_sequences=return_sequences,
            unroll=True,
            name=name,
        )
    if backend in {"compat", "cell", "rnncell", "safe"}:
        cell = layers.GRUCell(seq_hidden, name=f"{name}_cell")
        return layers.RNN(
            cell,
            return_sequences=return_sequences,
            unroll=False,
            name=name,
        )
    return layers.GRU(
        seq_hidden,
        return_sequences=return_sequences,
        unroll=False,
        name=name,
    )


def _bigru_stack(
    h: tf.Tensor,
    seq_hidden: int,
    seq_layers: int,
    seq_bidir: bool,
    seq_gru_unroll: bool,
    seq_gru_backend: str,
    dropout: float,
    name_prefix: str,
) -> tf.Tensor:
    """Apply a stacked BiGRU (or GRU) to a sequence tensor.

    Parameters
    ----------
    h           : Input tensor, shape (batch, W, features).
    seq_hidden  : Hidden units per GRU direction.
    seq_layers  : Number of stacked GRU layers.
    seq_bidir   : Wrap each GRU with Bidirectional if True.
    dropout     : Dropout rate between intermediate sequence layers.
    name_prefix : String prefix for Keras layer names.

    Returns
    -------
    Tensor of shape (batch, seq_hidden * (2 if seq_bidir else 1)).

    Notes
    -----
    ``seq_gru_unroll=True`` preserves the conservative execution path that was
    adopted for the seq_bigru_residual family on newer GPU/cuDNN stacks.
    ``False`` no longer implies a cuDNN dependency by itself: the actual GRU
    backend is controlled by ``seq_gru_backend`` so the training loop can
    retry failed fused kernels using a compatibility backend.
    """
    for i in range(seq_layers):
        return_seqs = i < seq_layers - 1
        gru = _make_gru_layer(
            seq_hidden=seq_hidden,
            return_sequences=return_seqs,
            seq_gru_unroll=seq_gru_unroll,
            seq_gru_backend=seq_gru_backend,
            name=f"{name_prefix}_gru_{i}",
        )
        if seq_bidir:
            h = layers.Bidirectional(
                gru,
                name=f"{name_prefix}_bigru_{i}",
            )(h)
        else:
            h = gru(h)
        if dropout > 0 and return_seqs:
            h = layers.Dropout(dropout, name=f"{name_prefix}_seq_drop_{i}")(h)
    return h


def _mlp_head(
    h: tf.Tensor,
    layer_sizes,
    activation: str,
    dropout: float,
    name_prefix: str,
) -> tf.Tensor:
    """Apply a Dense-BN-Activation stack to tensor h."""
    for i, u in enumerate(layer_sizes):
        h = layers.Dense(
            u, kernel_initializer="glorot_uniform", name=f"{name_prefix}_dense_{i}",
        )(h)
        h = layers.BatchNormalization(name=f"{name_prefix}_bn_{i}")(h)
        h = _activation_layer(activation)(h)
        if dropout > 0:
            h = layers.Dropout(dropout, name=f"{name_prefix}_drop_{i}")(h)
    return h


def _diffusion_ddim_step_tf(
    residual_t: tf.Tensor,
    eps_pred: tf.Tensor,
    alpha_bars,
    step_idx: int,
) -> tf.Tensor:
    """Deterministic DDIM-like reverse step."""
    dtype = residual_t.dtype
    alpha_bar_t = tf.constant(alpha_bars[step_idx], dtype=dtype)
    sqrt_alpha_bar_t = tf.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = tf.sqrt(tf.maximum(1.0 - alpha_bar_t, 1e-6))
    residual_pred = _diffusion_predict_x0(
        residual_t,
        eps_pred,
        sqrt_alpha_bar_t,
        sqrt_one_minus_alpha_bar_t,
    )
    if step_idx == 0:
        return residual_pred
    alpha_bar_prev = tf.constant(alpha_bars[step_idx - 1], dtype=dtype)
    return (
        tf.sqrt(alpha_bar_prev) * residual_pred
        + tf.sqrt(tf.maximum(1.0 - alpha_bar_prev, 0.0)) * eps_pred
    )


def _diffusion_ddpm_step_tf(
    residual_t: tf.Tensor,
    eps_pred: tf.Tensor,
    betas,
    alpha_bars,
    step_idx: int,
) -> tf.Tensor:
    """Ancestral DDPM reverse step."""
    dtype = residual_t.dtype
    alpha_bar_t = tf.constant(alpha_bars[step_idx], dtype=dtype)
    sqrt_alpha_bar_t = tf.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = tf.sqrt(tf.maximum(1.0 - alpha_bar_t, 1e-6))
    residual_pred = _diffusion_predict_x0(
        residual_t,
        eps_pred,
        sqrt_alpha_bar_t,
        sqrt_one_minus_alpha_bar_t,
    )
    if step_idx == 0:
        return residual_pred

    beta_t = tf.constant(betas[step_idx], dtype=dtype)
    alpha_t = tf.constant(1.0 - betas[step_idx], dtype=dtype)
    alpha_bar_prev = tf.constant(alpha_bars[step_idx - 1], dtype=dtype)
    mean = (
        residual_t
        - (beta_t / tf.sqrt(tf.maximum(1.0 - alpha_bar_t, 1e-6))) * eps_pred
    ) / tf.sqrt(alpha_t)
    sigma = tf.sqrt(
        tf.maximum(
            beta_t * (1.0 - alpha_bar_prev) / tf.maximum(1.0 - alpha_bar_t, 1e-6),
            0.0,
        )
    )
    noise = tf.random.normal(tf.shape(residual_t), dtype=dtype)
    return mean + sigma * noise


# ======================================================================
# Sub-model builders
# ======================================================================

def build_seq_prior_net(cfg: Dict) -> tf.keras.Model:
    """Build the sequence-aware conditional prior p(z | x_window, d, c).

    Parameters
    ----------
    cfg : dict with keys ``window_size``, ``seq_hidden_size``, ``seq_num_layers``,
          ``seq_bidirectional``, ``layer_sizes``, ``latent_dim``,
          ``activation``, ``dropout``.

    Returns
    -------
    keras.Model  inputs=[x_window(W,2), d(1), c(1)] → [z_mean_p, z_log_var_p]
    Model name: ``"prior_net"``
    """
    W         = int(cfg["window_size"])
    hidden    = int(cfg.get("seq_hidden_size", 64))
    n_layers  = int(cfg.get("seq_num_layers", 1))
    bidir     = bool(cfg.get("seq_bidirectional", True))
    gru_unroll = bool(cfg.get("seq_gru_unroll", True))
    gru_backend = str(cfg.get("seq_gru_backend", "fused"))
    latent    = int(cfg["latent_dim"])
    act       = cfg.get("activation", "leaky_relu")
    dropout   = float(cfg.get("dropout", 0.0))
    mlp_sizes = list(cfg["layer_sizes"])

    x_win_in = layers.Input(shape=(W, 2), name="prior_net_x_window")
    d_in     = layers.Input(shape=(1,),   name="prior_net_d")
    c_in     = layers.Input(shape=(1,),   name="prior_net_c")

    # Broadcast (d, c) along the time axis and concatenate with x_window → (W, 4)
    d_rep  = layers.RepeatVector(W, name="prior_net_d_rep")(d_in)
    c_rep  = layers.RepeatVector(W, name="prior_net_c_rep")(c_in)
    seq_in = layers.Concatenate(axis=-1, name="prior_net_seq_in")(
        [x_win_in, d_rep, c_rep]
    )

    # BiGRU context → (hidden*2,) or (hidden,)
    h = _bigru_stack(
        seq_in, hidden, n_layers, bidir, gru_unroll, gru_backend, dropout, name_prefix="prior_net"
    )

    # MLP head
    h = _mlp_head(h, mlp_sizes, act, dropout, name_prefix="prior_net")

    z_mean    = layers.Dense(latent, name="p_z_mean")(h)
    z_log_var = layers.Dense(latent, name="p_z_log_var")(h)

    return models.Model(
        [x_win_in, d_in, c_in], [z_mean, z_log_var], name="prior_net",
    )


def build_seq_encoder(cfg: Dict) -> tf.keras.Model:
    """Build the sequence-aware encoder q(z | x_window, d, c, y_center).

    Parameters
    ----------
    cfg : same keys as :func:`build_seq_prior_net`.

    Returns
    -------
    keras.Model  inputs=[x_window(W,2), d(1), c(1), y_center(2)]
                 → [z_mean_q, z_log_var_q]
    Model name: ``"encoder"``
    """
    W         = int(cfg["window_size"])
    hidden    = int(cfg.get("seq_hidden_size", 64))
    n_layers  = int(cfg.get("seq_num_layers", 1))
    bidir     = bool(cfg.get("seq_bidirectional", True))
    gru_unroll = bool(cfg.get("seq_gru_unroll", True))
    gru_backend = str(cfg.get("seq_gru_backend", "fused"))
    latent    = int(cfg["latent_dim"])
    act       = cfg.get("activation", "leaky_relu")
    dropout   = float(cfg.get("dropout", 0.0))
    mlp_sizes = list(cfg["layer_sizes"])

    x_win_in  = layers.Input(shape=(W, 2), name="encoder_x_window")
    d_in      = layers.Input(shape=(1,),   name="encoder_d")
    c_in      = layers.Input(shape=(1,),   name="encoder_c")
    y_cent_in = layers.Input(shape=(2,),   name="encoder_y_center")

    # Broadcast (d, c) and concatenate with x_window → (W, 4)
    d_rep  = layers.RepeatVector(W, name="encoder_d_rep")(d_in)
    c_rep  = layers.RepeatVector(W, name="encoder_c_rep")(c_in)
    seq_in = layers.Concatenate(axis=-1, name="encoder_seq_in")(
        [x_win_in, d_rep, c_rep]
    )

    # BiGRU context → (hidden*2,) or (hidden,)
    h = _bigru_stack(
        seq_in, hidden, n_layers, bidir, gru_unroll, gru_backend, dropout, name_prefix="encoder"
    )

    # Append y_center to the context vector before the MLP head
    h = layers.Concatenate(name="encoder_ctx_y")([h, y_cent_in])

    # MLP head
    h = _mlp_head(h, mlp_sizes, act, dropout, name_prefix="encoder")

    z_mean    = layers.Dense(latent, name="q_z_mean")(h)
    z_log_var = layers.Dense(latent, name="q_z_log_var")(h)

    return models.Model(
        [x_win_in, d_in, c_in, y_cent_in], [z_mean, z_log_var], name="encoder",
    )


def build_seq_decoder(cfg: Dict) -> tf.keras.Model:
    """Build the heteroscedastic residual decoder p(y | z, x_center, d, c).

    Uses a residual formulation identical to the ``channel_residual`` variant::

        delta_mean = MLP(z, x_center, d, c)[:, :2]
        y_mean     = x_center + delta_mean

    Output is Gaussian, MDN, or diffusion depending on
    ``decoder_distribution``:
    - Gaussian: ``(mean_I, mean_Q, logvar_I, logvar_Q)``
    - MDN: flattened ``(logits[K], mean[K,2], logvar[K,2])``
    - Diffusion: predicted noise ``eps_hat`` for one DDPM step

    Parameters
    ----------
    cfg : dict with keys ``layer_sizes``, ``latent_dim``, ``activation``, ``dropout``.

    Returns
    -------
    keras.Model  inputs=[z(latent_dim), x_center(2), d(1), c(1)] → output_params(4)
    Model name: ``"decoder"``
    """
    latent    = int(cfg["latent_dim"])
    act       = cfg.get("activation", "leaky_relu")
    dropout   = float(cfg.get("dropout", 0.0))
    mlp_sizes = list(cfg["layer_sizes"])
    decoder_distribution = str(
        cfg.get("decoder_distribution", "gaussian")
    ).strip().lower()
    mdn_components = int(cfg.get("mdn_components", 1))
    cond_embed_dim = int(cfg.get("cond_embed_dim", 0))
    diffusion_steps = int(cfg.get("diffusion_steps", 8))
    diffusion_beta_start = float(cfg.get("diffusion_beta_start", 1e-4))
    diffusion_beta_end = float(cfg.get("diffusion_beta_end", 2e-2))
    diffusion_hidden_size = int(
        cfg.get("diffusion_hidden_size", mlp_sizes[-1] if mlp_sizes else 64)
    )

    z_in      = layers.Input(shape=(latent,), name="z_input")
    x_cent_in = layers.Input(shape=(2,),      name="x_center_input")
    d_in      = layers.Input(shape=(1,),      name="d_input")
    c_in      = layers.Input(shape=(1,),      name="c_input")

    if cond_embed_dim > 0:
        # Nonlinear regime embedding: (d, c) → 2-layer MLP → cond_embed_dim
        cond_raw = layers.Concatenate(name="dec_cond_raw")([d_in, c_in])
        cond_h = layers.Dense(cond_embed_dim, activation=act, name="dec_cond_emb_0")(cond_raw)
        cond_h = layers.Dense(cond_embed_dim, activation=act, name="dec_cond_emb_1")(cond_h)
        h = layers.Concatenate(name="dec_concat")([z_in, x_cent_in, cond_h])
    else:
        h = layers.Concatenate(name="dec_concat")([z_in, x_cent_in, d_in, c_in])

    h = _mlp_head(h, mlp_sizes, act, dropout, name_prefix="dec")

    if decoder_distribution == "diffusion":
        residual_noisy_in = layers.Input(
            shape=(2,), name="diffusion_residual_noisy_input"
        )
        t_frac_in = layers.Input(shape=(1,), name="diffusion_timestep_input")

        t_h = layers.Dense(
            max(diffusion_hidden_size // 2, 8),
            kernel_initializer="glorot_uniform",
            name="diffusion_t_dense_0",
        )(t_frac_in)
        t_h = _activation_layer(act)(t_h)

        diff_h = layers.Concatenate(name="diffusion_concat")(
            [h, residual_noisy_in, t_frac_in, t_h]
        )
        diff_h = _mlp_head(
            diff_h,
            [diffusion_hidden_size, diffusion_hidden_size],
            act,
            dropout,
            name_prefix="diffusion",
        )
        out = layers.Dense(2, name="output_params")(diff_h)
        model = models.Model(
            [z_in, x_cent_in, d_in, c_in, residual_noisy_in, t_frac_in],
            out,
            name="decoder",
        )
        model._diffusion_steps = diffusion_steps
        model._diffusion_beta_start = diffusion_beta_start
        model._diffusion_beta_end = diffusion_beta_end
        return model
    if decoder_distribution == "mdn":
        k = int(mdn_components)
        raw_out = layers.Dense(5 * k, name="output_params_raw")(h)
        logits = SliceFeatures(0, k, name="mixture_logits")(raw_out)
        delta_mean_flat = SliceFeatures(k, k + 2 * k, name="delta_mean_flat")(raw_out)
        delta_lv_flat = SliceFeatures(k + 2 * k, k + 4 * k, name="delta_log_var_flat")(raw_out)
        delta_mean = layers.Reshape((k, 2), name="delta_mean_components")(delta_mean_flat)
        x_center_rep = layers.RepeatVector(k, name="x_center_mixture_rep")(x_cent_in)
        y_mean = layers.Add(name="y_mean_components")([x_center_rep, delta_mean])
        y_mean_flat = layers.Reshape((2 * k,), name="y_mean_flat")(y_mean)
        out = layers.Concatenate(name="output_params")([logits, y_mean_flat, delta_lv_flat])
    else:
        # Residual: predict (delta_mean, delta_log_var), then add x_center to mean
        raw_out = layers.Dense(4, name="output_params_raw")(h)
        delta_mean = SliceFeatures(0, 2, name="delta_mean")(raw_out)
        delta_lv = SliceFeatures(2, 4, name="delta_log_var")(raw_out)
        y_mean = layers.Add(name="y_mean_residual")([x_cent_in, delta_mean])
        out = layers.Concatenate(name="output_params")([y_mean, delta_lv])

    return models.Model([z_in, x_cent_in, d_in, c_in], out, name="decoder")


# ======================================================================
# Full cVAE assembly
# ======================================================================

def build_seq_cvae(cfg: Dict) -> Tuple[tf.keras.Model, "KLAnnealingCallback"]:
    """Build, compile, and return the full sequence-aware cVAE.

    Same signature as ``build_cvae`` in ``src.models.cvae``.

    Parameters
    ----------
    cfg : dict with keys ``window_size``, ``layer_sizes``, ``latent_dim``,
          ``beta``, ``lr``, ``dropout``, ``free_bits``, ``kl_anneal_epochs``,
          ``activation``, ``seq_hidden_size``, ``seq_num_layers``,
          ``seq_bidirectional``.

    Returns
    -------
    (vae, kl_cb) : (keras.Model, KLAnnealingCallback)
        ``vae.name = "cvae_seq_condprior"``
        Sub-models accessible via ``vae.get_layer("encoder")``,
        ``vae.get_layer("prior_net")``, ``vae.get_layer("decoder")``.
    """
    W                = int(cfg["window_size"])
    latent_dim       = int(cfg["latent_dim"])
    beta             = float(cfg["beta"])
    lr               = float(cfg["lr"])
    free_bits        = float(cfg.get("free_bits", 0.0))
    kl_anneal_epochs = int(cfg.get("kl_anneal_epochs", 50))
    half             = W // 2

    encoder   = build_seq_encoder(cfg)
    prior_net = build_seq_prior_net(cfg)
    decoder   = build_seq_decoder(cfg)

    # --- Full model inputs ---
    x_win_in = layers.Input(shape=(W, 2), name="x_window_input")
    d_in     = layers.Input(shape=(1,),   name="distance_input")
    c_in     = layers.Input(shape=(1,),   name="current_input")
    y_in     = layers.Input(shape=(2,),   name="y_true")

    # Extract the center sample from the window (used by decoder and loss)
    x_center = ExtractCenterFrame(half, name="x_center_extract")(x_win_in)

    # Posterior q(z | x_window, d, c, y_center)
    z_mean_q, z_log_var_q = encoder([x_win_in, d_in, c_in, y_in])

    # Prior p(z | x_window, d, c)
    z_mean_p, z_log_var_p = prior_net([x_win_in, d_in, c_in])

    # Clip prior log-var in the training graph (mirrors cvae.py)
    z_log_var_p = ClipValues(-10.0, 10.0, name="clip_p_logvar_train")(z_log_var_p)

    # Sample z from the posterior
    z = Sampling(name="sampling")([z_mean_q, z_log_var_q])

    # Decode: p(y | z, x_center, d, c)
    decoder_distribution = str(cfg.get("decoder_distribution", "gaussian")).strip().lower()
    diffusion_steps = int(cfg.get("diffusion_steps", 8))
    diffusion_beta_start = float(cfg.get("diffusion_beta_start", 1e-4))
    diffusion_beta_end = float(cfg.get("diffusion_beta_end", 2e-2))

    if decoder_distribution == "diffusion":
        residual_true = layers.Subtract(name="residual_true")([y_in, x_center])
        (
            residual_noisy,
            noise_target,
            t_frac,
            sqrt_alpha_bar,
            sqrt_one_minus_alpha_bar,
        ) = SampleDiffusionTrainingState(
            diffusion_steps=diffusion_steps,
            diffusion_beta_start=diffusion_beta_start,
            diffusion_beta_end=diffusion_beta_end,
            name="diffusion_state",
        )(residual_true)
        out_params = decoder([z, x_center, d_in, c_in, residual_noisy, t_frac])
    else:
        out_params = decoder([z, x_center, d_in, c_in])

    # KL + reconstruction loss (beta starts at 0 for annealing)
    lambda_mmd = float(cfg.get("lambda_mmd", 0.0))
    lambda_axis = float(cfg.get("lambda_axis", 0.0))
    lambda_psd = float(cfg.get("lambda_psd", 0.0))
    lambda_coverage = float(cfg.get("lambda_coverage", 0.0))
    lambda_kurt = float(cfg.get("lambda_kurt", 0.0))
    axis_std_weight = float(cfg.get("axis_std_weight", 1.0))
    axis_skew_weight = float(cfg.get("axis_skew_weight", 0.25))
    axis_kurt_weight = float(cfg.get("axis_kurt_weight", 0.10))
    coverage_levels = tuple(float(x) for x in cfg.get("coverage_levels", [0.50, 0.80, 0.95]))
    tail_levels = tuple(float(x) for x in cfg.get("tail_levels", [0.05, 0.95]))
    coverage_temperature = float(cfg.get("coverage_temperature", 0.05))
    mmd_mode = str(cfg.get("mmd_mode", "mean_residual"))
    mdn_components = int(cfg.get("mdn_components", 1))
    if decoder_distribution == "diffusion":
        if any(v > 0.0 for v in (lambda_mmd, lambda_axis, lambda_psd, lambda_coverage, lambda_kurt)):
            raise ValueError(
                "The first diffusion route currently supports only the core DDPM objective; "
                "set lambda_mmd/lambda_axis/lambda_psd/lambda_coverage/lambda_kurt to 0."
            )
        loss_layer = CondPriorDiffusionVAELoss(
            beta=0.0,
            free_bits=free_bits,
            diffusion_steps=diffusion_steps,
            diffusion_beta_start=diffusion_beta_start,
            diffusion_beta_end=diffusion_beta_end,
            name="condprior_loss",
        )
        y_mean_out = loss_layer(
            [
                x_center,
                residual_noisy,
                noise_target,
                sqrt_alpha_bar,
                sqrt_one_minus_alpha_bar,
                out_params,
                z_mean_q,
                z_log_var_q,
                z_mean_p,
                z_log_var_p,
            ]
        )
    else:
        loss_layer = CondPriorVAELoss(
            beta=0.0, free_bits=free_bits,
            lambda_mmd=lambda_mmd,
            lambda_axis=lambda_axis,
            lambda_psd=lambda_psd,
            lambda_coverage=lambda_coverage,
            lambda_kurt=lambda_kurt,
            axis_std_weight=axis_std_weight,
            axis_skew_weight=axis_skew_weight,
            axis_kurt_weight=axis_kurt_weight,
            coverage_levels=coverage_levels,
            tail_levels=tail_levels,
            coverage_temperature=coverage_temperature,
            mmd_mode=mmd_mode,
            decoder_distribution=decoder_distribution,
            mdn_components=mdn_components,
            name="condprior_loss",
        )
        loss_inputs = [y_in, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p]
        if any(v > 0.0 for v in (lambda_mmd, lambda_axis, lambda_psd, lambda_coverage, lambda_kurt)):
            loss_inputs.append(x_center)
        y_mean_out = loss_layer(loss_inputs)

    vae = models.Model(
        [x_win_in, d_in, c_in, y_in], y_mean_out,
        name="cvae_seq_condprior",
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    vae.compile(optimizer=opt)

    kl_cb = KLAnnealingCallback(
        loss_layer, beta_start=0.0, beta_end=beta,
        annealing_epochs=kl_anneal_epochs,
    )
    return vae, kl_cb


# ======================================================================
# Inference model from saved seq cVAE
# ======================================================================

def create_seq_inference_model(
    full_model: tf.keras.Model,
    deterministic: bool = True,
) -> tf.keras.Model:
    """Create a sequence-aware inference graph from a trained seq cVAE.

    For Gaussian/MDN paths the graph stays close to the original implementation.
    The diffusion path uses a small iterative sampler during inference.

    Parameters
    ----------
    full_model    : Trained ``cvae_seq_condprior`` with sub-layers
                    ``encoder``, ``prior_net``, ``decoder``.
    deterministic : If True, z = z_mean_p (MAP estimate).
                    If False, z ~ p(z | x_window, d, c).

    Returns
    -------
    keras.Model  inputs=[x_window(W,2), d(1), c(1)] → y_prediction(2)
    """
    prior = full_model.get_layer("prior_net")
    dec   = full_model.get_layer("decoder")

    # Infer window_size from the prior's first input (x_window shape=(None,W,2))
    W    = prior.inputs[0].shape[1]
    half = W // 2

    x_win_in = layers.Input(shape=(W, 2), name="x_window_input")
    d_in     = layers.Input(shape=(1,),   name="distance_input")
    c_in     = layers.Input(shape=(1,),   name="current_input")

    z_mean_p, z_log_var_p = prior([x_win_in, d_in, c_in])
    z_log_var_p = ClipValues(-10.0, 10.0, name="clip_zlogvar")(z_log_var_p)

    if deterministic:
        z = z_mean_p  # MAP estimate — no sampling layer needed
    else:
        z = Sampling(name="z_sample")([z_mean_p, z_log_var_p])

    x_center = ExtractCenterFrame(half, name="x_center_extract")(x_win_in)
    is_diffusion = len(dec.inputs) == 6
    if is_diffusion:
        loss_layer = full_model.get_layer("condprior_loss")
        diffusion_steps = int(getattr(loss_layer, "diffusion_steps", 8))
        diffusion_beta_start = float(getattr(loss_layer, "diffusion_beta_start", 1e-4))
        diffusion_beta_end = float(getattr(loss_layer, "diffusion_beta_end", 2e-2))
        betas, _, alpha_bars = _diffusion_schedule_arrays(
            diffusion_steps,
            diffusion_beta_start,
            diffusion_beta_end,
        )

        if deterministic:
            residual = layers.Lambda(
                lambda t: tf.zeros_like(t),
                name="diffusion_residual_init",
            )(x_center)
        else:
            residual = layers.Lambda(
                lambda t: tf.random.normal(tf.shape(t), dtype=t.dtype),
                name="diffusion_residual_init",
            )(x_center)

        for step_idx in reversed(range(diffusion_steps)):
            t_value = 0.0 if diffusion_steps <= 1 else float(step_idx) / float(diffusion_steps - 1)
            t_frac = layers.Lambda(
                lambda t, v=t_value: tf.ones((tf.shape(t)[0], 1), dtype=t.dtype) * tf.cast(v, t.dtype),
                name=f"diffusion_timestep_{step_idx}",
            )(d_in)
            eps_pred = dec([z, x_center, d_in, c_in, residual, t_frac])
            if deterministic:
                residual = layers.Lambda(
                    lambda xs, s=step_idx, ab=alpha_bars: _diffusion_ddim_step_tf(xs[0], xs[1], ab, s),
                    name=f"diffusion_step_{step_idx}",
                )([residual, eps_pred])
            else:
                residual = layers.Lambda(
                    lambda xs, s=step_idx, bt=betas, ab=alpha_bars: _diffusion_ddpm_step_tf(xs[0], xs[1], bt, ab, s),
                    name=f"diffusion_step_{step_idx}",
                )([residual, eps_pred])

        y = layers.Add(name="y_diffusion")([x_center, residual])
    else:
        out_params = dec([z, x_center, d_in, c_in])
        out_dim = int(dec.output_shape[-1])
        is_mdn = out_dim > 4 and out_dim % 5 == 0

        if is_mdn:
            k = out_dim // 5
            logits = SliceFeatures(0, k, name="mixture_logits")(out_params)
            y_mean_flat = SliceFeatures(k, k + 2 * k, name="y_mean_flat")(out_params)
            y_log_var_flat = SliceFeatures(k + 2 * k, k + 4 * k, name="y_logvar_flat")(out_params)
            y_mean_components = layers.Reshape((k, 2), name="y_mean_components")(y_mean_flat)
            y_log_var_components_raw = layers.Reshape(
                (k, 2), name="y_logvar_components_raw"
            )(y_log_var_flat)
            y_log_var_components = ClipValues(
                DECODER_LOGVAR_CLAMP_LO,
                DECODER_LOGVAR_CLAMP_HI,
                name="y_logvar_components",
            )(y_log_var_components_raw)

            if deterministic:
                y = layers.Lambda(
                    lambda xs: _mdn_expected_mean(xs[0], xs[1]),
                    name="y_det",
                )([logits, y_mean_components])
            else:
                y = layers.Lambda(
                    lambda xs: _sample_mdn(xs[0], xs[1], xs[2]),
                    name="y_sample",
                )([logits, y_mean_components, y_log_var_components])
        else:
            y_mean = SliceFeatures(0, 2, name="y_mean")(out_params)
            y_log_var_raw = SliceFeatures(2, 4, name="y_logvar_slice")(out_params)
            y_log_var = ClipValues(
                DECODER_LOGVAR_CLAMP_LO, DECODER_LOGVAR_CLAMP_HI, name="y_logvar",
            )(y_log_var_raw)

            if deterministic:
                y = y_mean  # MAP — direct tensor, no identity layer needed
            else:
                y = Sampling(name="y_sample")([y_mean, y_log_var])

    name = "inference_seq_condprior_det" if deterministic else "inference_seq_condprior"
    return models.Model([x_win_in, d_in, c_in], y, name=name)


# ======================================================================
# Save / load helper
# ======================================================================

def load_seq_model(path: str) -> tf.keras.Model:
    """Load a saved cVAE model with correct custom_objects.

    Handles all active arch_variants: point-wise (functional) and
    seq_bigru_residual (functional with custom seq layers).
    Auto-detects SavedModel format (directory) vs HDF5/keras format (file).

    Parameters
    ----------
    path : str or Path
        Path to the saved .keras file or SavedModel directory.

    Returns
    -------
    keras.Model
        Loaded model with all custom layers resolved.
    """
    return tf.keras.models.load_model(
        str(path),
        custom_objects={
            "Sampling": Sampling,
            "CondPriorDeltaVAELoss": CondPriorDeltaVAELoss,
            "CondPriorDiffusionVAELoss": CondPriorDiffusionVAELoss,
            "CondPriorVAELoss": CondPriorVAELoss,
            "StdNormalHeteroscedasticVAELoss": StdNormalHeteroscedasticVAELoss,
            "ExtractCenterFrame": ExtractCenterFrame,
            "ClipValues": ClipValues,
            "SliceFeatures": SliceFeatures,
            "SampleDiffusionTrainingState": SampleDiffusionTrainingState,
        },
        compile=False,
    )
