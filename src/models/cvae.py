# -*- coding: utf-8 -*-
"""
src/models/cvae.py — cVAE model construction (encoder / decoder / prior / full).

Extracted from ``cvae_components.py`` (refactor step 3).
The default ``concat`` architecture is identical to the legacy monolith.
This module also exposes the experimental ``channel_residual`` decoder
variant, which predicts ``Δ = Y - X`` internally while preserving the
same external model interface.
It also exposes ``arch_variant="delta_residual"``, which makes that residual
target explicit in the decoder output and loss while preserving the same
external inference interface.
It also dispatches the legacy experimental
``arch_variant="legacy_2025_zero_y"`` point-wise model, which preserves the
2025 architecture under the stricter 2026 pipeline.
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

from typing import Dict, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config.defaults import (
    DECODER_LOGVAR_CLAMP_HI,
    DECODER_LOGVAR_CLAMP_LO,
)
from src.models.sampling import Sampling
from src.models.losses import CondPriorDeltaVAELoss, CondPriorVAELoss
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


def _normalize_arch_variant(arch_variant: str) -> str:
    """Validate and normalise the decoder architecture variant name."""
    variant = str(arch_variant or "concat").strip().lower()
    valid = {
        "concat",
        "channel_residual",
        "delta_residual",
        "seq_bigru_residual",
        "legacy_2025_zero_y",
    }
    if variant not in valid:
        raise ValueError(
            f"Unknown arch_variant={arch_variant!r}. "
            f"Expected one of {sorted(valid)}."
        )
    return variant


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
        Optional: ``radial_feature`` (bool) — append R=||x|| to inputs.

    Returns
    -------
    keras.Model  inputs=[x(2), d(1), c(1), y(2)] or
                        [x(2), d(1), c(1), y(2), r(1)] → [z_mean, z_log_var]
    """
    radial = bool(cfg.get("radial_feature", False))
    in_shapes = [(2,), (1,), (1,), (2,)]
    if radial:
        in_shapes.append((1,))
    return build_mlp(
        name="encoder",
        in_shapes=in_shapes,
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
        Optional: ``radial_feature`` (bool) — append R=||x|| to inputs.

    Returns
    -------
    keras.Model  inputs=[x(2), d(1), c(1)] or
                        [x(2), d(1), c(1), r(1)] → [z_mean, z_log_var]
    """
    radial = bool(cfg.get("radial_feature", False))
    in_shapes = [(2,), (1,), (1,)]
    if radial:
        in_shapes.append((1,))
    return build_mlp(
        name="prior_net",
        in_shapes=in_shapes,
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
    arch_variant: str = "concat",
    decoder_distribution: str = "gaussian",
    mdn_components: int = 1,
    radial_feature: bool = False,
) -> tf.keras.Model:
    """Build the heteroscedastic decoder p(y | z, x, d, c[, r]).

    Output has 4 units for Gaussian decoders:
    ``(mean_I, mean_Q, logvar_I, logvar_Q)``.

    ``arch_variant="concat"``
        Legacy behaviour: predict ``y`` parameters directly from
        ``concat([z, x, d, c])``.

    ``arch_variant="channel_residual"``
        Predict residual parameters ``Δ = Y - X`` internally and resolve the
        final mean as ``Y_mean = X + Δ_mean``. The external decoder output
        remains identical, so losses/inference do not need to change.

    ``arch_variant="delta_residual"``
        Predict residual parameters ``Δ = Y - X`` explicitly. The decoder
        output stays in residual space; the training loss and inference graph
        resolve the final signal as ``Y = X + Δ``.

    Parameters
    ----------
    radial_feature : bool
        If True, cond_in has shape (5,) = x(2)+d(1)+c(1)+r(1) where
        r = ||x||.  The extra feature gives the MLP a shortcut to learn
        signal-dependent (radial) noise variance.
    """
    arch_variant = _normalize_arch_variant(arch_variant)
    decoder_distribution = str(decoder_distribution or "gaussian").strip().lower()
    if decoder_distribution != "gaussian":
        raise ValueError(
            "Point-wise MDN decoder is not implemented yet; "
            "use decoder_distribution='gaussian' or the seq_bigru_residual variant."
        )
    if arch_variant == "legacy_2025_zero_y":
        from src.models.cvae_legacy_2025 import build_legacy_2025_decoder
        return build_legacy_2025_decoder(
            layer_sizes=layer_sizes,
            latent_dim=latent_dim,
            activation=activation,
            dropout=dropout,
        )

    cond_dim = 5 if radial_feature else 4
    z_in = layers.Input(shape=(latent_dim,), name="z_input")
    cond_in = layers.Input(shape=(cond_dim,), name="cond_input")    # x(2)+d(1)+c(1)[+r(1)]
    h = layers.Concatenate(name="dec_concat")([z_in, cond_in])
    for i, u in enumerate(layer_sizes):
        h = layers.Dense(
            u, kernel_initializer="glorot_uniform", name=f"dec_dense_{i}",
        )(h)
        h = layers.BatchNormalization(name=f"dec_bn_{i}")(h)
        h = _activation_layer(activation)(h)
        if dropout and dropout > 0:
            h = layers.Dropout(dropout, name=f"dec_drop_{i}")(h)
    if arch_variant == "channel_residual":
        # The first two condition features are always X=(I,Q).
        raw_out = layers.Dense(4, name="output_params_raw")(h)
        delta_mean = layers.Lambda(
            lambda t: t[:, :2], name="delta_mean",
        )(raw_out)
        delta_log_var = layers.Lambda(
            lambda t: t[:, 2:], name="delta_log_var",
        )(raw_out)
        x_passthrough = layers.Lambda(
            lambda t: t[:, :2], name="x_passthrough",
        )(cond_in)
        y_mean = layers.Add(name="y_mean_residual")(
            [x_passthrough, delta_mean]
        )
        out = layers.Concatenate(name="output_params")(
            [y_mean, delta_log_var]
        )
    elif arch_variant == "delta_residual":
        out = layers.Dense(4, name="delta_output_params")(h)
    else:
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
        Optional: ``free_bits, kl_anneal_epochs, activation, arch_variant``.

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
    arch_variant = _normalize_arch_variant(cfg.get("arch_variant", "concat"))

    if arch_variant == "seq_bigru_residual":
        from src.models.cvae_sequence import build_seq_cvae  # lazy import
        return build_seq_cvae(cfg)
    if arch_variant == "legacy_2025_zero_y":
        from src.models.cvae_legacy_2025 import build_legacy_2025_cvae
        return build_legacy_2025_cvae(cfg)

    radial = bool(cfg.get("radial_feature", False))

    encoder = build_encoder(cfg)
    prior_net = build_prior_net(cfg)
    decoder = build_decoder(
        layer_sizes=layer_sizes, latent_dim=latent_dim,
        activation=activation, dropout=dropout,
        arch_variant=arch_variant,
        decoder_distribution=str(cfg.get("decoder_distribution", "gaussian")),
        mdn_components=int(cfg.get("mdn_components", 1)),
        radial_feature=radial,
    )

    # --- Graph wiring ---
    x_in = layers.Input(shape=(2,), name="x_input")
    d_in = layers.Input(shape=(1,), name="distance_input")
    c_in = layers.Input(shape=(1,), name="current_input")
    y_in = layers.Input(shape=(2,), name="y_true")

    # Radial feature: R = ||x|| — gives the MLP an explicit shortcut to
    # learn signal-dependent noise variance (border σ >> center σ).
    if radial:
        r_feat = layers.Lambda(
            lambda x: tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + 1e-8),
            name="radial_feature",
        )(x_in)
        enc_inputs = [x_in, d_in, c_in, y_in, r_feat]
        pri_inputs = [x_in, d_in, c_in, r_feat]
    else:
        enc_inputs = [x_in, d_in, c_in, y_in]
        pri_inputs = [x_in, d_in, c_in]

    z_mean_q, z_log_var_q = encoder(enc_inputs)
    z_mean_p, z_log_var_p = prior_net(pri_inputs)

    # C4.2 FIX: clip prior log-var in training graph too
    z_log_var_p = layers.Lambda(
        lambda t: tf.clip_by_value(t, -10.0, 10.0),
        name="clip_p_logvar_train",
    )(z_log_var_p)

    z = Sampling(name="sampling")([z_mean_q, z_log_var_q])

    if radial:
        cond = layers.Concatenate(name="cond_concat")([x_in, d_in, c_in, r_feat])  # (N,5)
    else:
        cond = layers.Concatenate(name="cond_concat")([x_in, d_in, c_in])  # (N,4)
    out_params = decoder([z, cond])

    # Loss layer (β starts at 0 for annealing)
    beta_initial = 0.0
    lambda_mmd = float(cfg.get("lambda_mmd", 0.0))
    lambda_axis = float(cfg.get("lambda_axis", 0.0))
    lambda_psd = float(cfg.get("lambda_psd", 0.0))
    lambda_coverage = float(cfg.get("lambda_coverage", 0.0))
    axis_std_weight = float(cfg.get("axis_std_weight", 1.0))
    axis_skew_weight = float(cfg.get("axis_skew_weight", 0.25))
    axis_kurt_weight = float(cfg.get("axis_kurt_weight", 0.10))
    coverage_levels = tuple(float(x) for x in cfg.get("coverage_levels", [0.50, 0.80, 0.95]))
    tail_levels = tuple(float(x) for x in cfg.get("tail_levels", [0.05, 0.95]))
    coverage_temperature = float(cfg.get("coverage_temperature", 0.05))
    mmd_mode = str(cfg.get("mmd_mode", "mean_residual"))
    decoder_distribution = str(cfg.get("decoder_distribution", "gaussian"))
    mdn_components = int(cfg.get("mdn_components", 1))
    aux_needs_x = any(v > 0.0 for v in (lambda_mmd, lambda_axis, lambda_psd, lambda_coverage))
    if decoder_distribution.strip().lower() != "gaussian" and arch_variant != "seq_bigru_residual":
        raise ValueError(
            "decoder_distribution='mdn' is currently supported only for "
            "arch_variant='seq_bigru_residual'."
        )
    if arch_variant == "delta_residual":
        loss_layer = CondPriorDeltaVAELoss(
            beta=beta_initial,
            free_bits=free_bits,
            lambda_mmd=lambda_mmd,
            lambda_axis=lambda_axis,
            lambda_psd=lambda_psd,
            lambda_coverage=lambda_coverage,
            coverage_levels=coverage_levels,
            tail_levels=tail_levels,
            coverage_temperature=coverage_temperature,
            mmd_mode=mmd_mode,
            decoder_distribution=decoder_distribution,
            name="condprior_delta_loss",
        )
        loss_inputs = [y_in, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p, x_in]
    else:
        loss_layer = CondPriorVAELoss(
            beta=beta_initial, free_bits=free_bits,
            lambda_mmd=lambda_mmd,
            lambda_axis=lambda_axis,
            lambda_psd=lambda_psd,
            lambda_coverage=lambda_coverage,
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
        if aux_needs_x:
            loss_inputs.append(x_in)
    y_mean = loss_layer(loss_inputs)

    model_name = (
        "cvae_condprior_delta_residual"
        if arch_variant == "delta_residual"
        else "cvae_condprior"
    )
    vae = models.Model([x_in, d_in, c_in, y_in], y_mean, name=model_name)
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

    try:
        dec.get_layer("delta_output_params")
        is_delta_residual = True
    except Exception:
        is_delta_residual = False

    # Detect sequence variant: prior input[0] is (None, W, 2) for seq, (None, 2) for point-wise.
    if len(prior.inputs[0].shape) == 3:
        from src.models.cvae_sequence import create_seq_inference_model  # lazy import
        return create_seq_inference_model(full_model, deterministic=deterministic)

    # Detect radial feature: point-wise decoder cond_input has shape (None, 5)
    # when active. Sequence models are already handled above and do not expose
    # a `cond_input` layer.
    dec_cond_shape = dec.get_layer("cond_input").input_shape
    if isinstance(dec_cond_shape, list):
        dec_cond_shape = dec_cond_shape[0]
    has_radial = dec_cond_shape[-1] == 5

    x_in = layers.Input(shape=(2,), name="x_input")
    d_in = layers.Input(shape=(1,), name="distance_input")
    c_in = layers.Input(shape=(1,), name="current_input")

    if has_radial:
        r_feat = layers.Lambda(
            lambda x: tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + 1e-8),
            name="radial_feature",
        )(x_in)
        pri_inputs = [x_in, d_in, c_in, r_feat]
    else:
        pri_inputs = [x_in, d_in, c_in]

    z_mean_p, z_log_var_p = prior(pri_inputs)
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

    if has_radial:
        cond = layers.Concatenate(name="cond_concat_inf")([x_in, d_in, c_in, r_feat])
    else:
        cond = layers.Concatenate(name="cond_concat_inf")([x_in, d_in, c_in])
    out_params = dec([z, cond])

    if is_delta_residual:
        delta_mean = layers.Lambda(lambda t: t[:, :2], name="delta_mean")(out_params)
        delta_log_var = layers.Lambda(
            lambda t: tf.clip_by_value(
                t[:, 2:], DECODER_LOGVAR_CLAMP_LO, DECODER_LOGVAR_CLAMP_HI
            ),
            name="delta_logvar",
        )(out_params)
        if deterministic:
            y = layers.Add(name="y_det")([x_in, delta_mean])
        else:
            eps_delta = layers.Lambda(
                lambda t: tf.random.normal(tf.shape(t)), name="eps_delta",
            )(delta_mean)
            delta = layers.Lambda(
                lambda a: a[0] + tf.exp(0.5 * a[1]) * a[2], name="sample_delta",
            )([delta_mean, delta_log_var, eps_delta])
            y = layers.Add(name="sample_y")([x_in, delta])
    else:
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
