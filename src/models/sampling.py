# -*- coding: utf-8 -*-
"""
src/models/sampling.py — Reparameterisation trick and latent-space sampling.

Extracted from ``cvae_components.py`` (refactor step 3).

Public API
----------
Sampling                     Keras layer (for model graphs)
reparameterize               Functional NumPy / eager-TF form
build_prior_predict_inputs   Build caller-side inputs for prior_net.predict(...)
build_encoder_predict_inputs Build caller-side inputs for encoder.predict(...)
sample_prior                 Draw z from the conditional prior network
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ======================================================================
# Keras layer (used inside the training graph)
# ======================================================================
@tf.keras.utils.register_keras_serializable(package="VLC")
class Sampling(layers.Layer):
    """Reparameterisation trick:  z = μ + σ · ε,  ε ~ N(0, I)."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


# ======================================================================
# Functional forms (NumPy / eager)
# ======================================================================
def reparameterize(
    mu: np.ndarray,
    logvar: np.ndarray,
    *,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample z = μ + σ·ε  (NumPy, non-graph usage).

    Parameters
    ----------
    mu, logvar : ndarray, shape (N, latent_dim)
    seed : int, optional
        For reproducibility.

    Returns
    -------
    z : ndarray, same shape as *mu*
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(mu.shape).astype(mu.dtype)
    return mu + np.exp(0.5 * logvar) * eps


def build_prior_predict_inputs(
    prior_net: tf.keras.Model,
    x: np.ndarray,
    d: np.ndarray,
    c: np.ndarray,
):
    """Return the correct input list for ``prior_net.predict``.

    Supports both point-wise and sequence priors, with or without the extra
    radial feature ``r = ||x||``. The external callers continue to pass just
    ``(x, d, c)``; this helper expands the list when the model expects the
    radial conditioning input.
    """
    prior_inputs = getattr(prior_net, "inputs", None)
    if prior_inputs is None:
        return [x, d, c]

    n_inputs = len(prior_inputs)
    if n_inputs == 3:
        return [x, d, c]
    if n_inputs != 4:
        raise ValueError(
            f"Unsupported prior_net interface with {n_inputs} inputs; "
            "expected 3 (base) or 4 (radial)."
        )

    x_arr = np.asarray(x)
    if x_arr.ndim == 3:
        x_center = x_arr[:, x_arr.shape[1] // 2, :]
    elif x_arr.ndim == 2:
        x_center = x_arr
    else:
        raise ValueError(
            f"Unsupported x rank for radial prior input: {x_arr.ndim}. "
            "Expected point-wise (N,2) or sequence (N,W,2)."
        )

    r = np.sqrt(np.sum(np.square(x_center), axis=-1, keepdims=True) + 1e-8).astype(x_arr.dtype, copy=False)
    return [x, d, c, r]


def build_encoder_predict_inputs(
    encoder: tf.keras.Model,
    x: np.ndarray,
    d: np.ndarray,
    c: np.ndarray,
    y: np.ndarray,
):
    """Return the correct input list for ``encoder.predict``.

    Supports both point-wise and sequence encoders, with or without the extra
    radial feature ``r = ||x||``. External callers continue to pass just
    ``(x, d, c, y)``; this helper expands the list when the model expects the
    radial conditioning input.
    """
    encoder_inputs = getattr(encoder, "inputs", None)
    if encoder_inputs is None:
        return [x, d, c, y]

    n_inputs = len(encoder_inputs)
    if n_inputs == 4:
        return [x, d, c, y]
    if n_inputs != 5:
        raise ValueError(
            f"Unsupported encoder interface with {n_inputs} inputs; "
            "expected 4 (base) or 5 (radial)."
        )

    x_arr = np.asarray(x)
    if x_arr.ndim == 3:
        x_center = x_arr[:, x_arr.shape[1] // 2, :]
    elif x_arr.ndim == 2:
        x_center = x_arr
    else:
        raise ValueError(
            f"Unsupported x rank for radial encoder input: {x_arr.ndim}. "
            "Expected point-wise (N,2) or sequence (N,W,2)."
        )

    r = np.sqrt(np.sum(np.square(x_center), axis=-1, keepdims=True) + 1e-8).astype(x_arr.dtype, copy=False)
    return [x, d, c, y, r]


def sample_prior(
    prior_net: tf.keras.Model,
    x: np.ndarray,
    d: np.ndarray,
    c: np.ndarray,
    *,
    deterministic: bool = True,
    batch_size: int = 8192,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw z from the conditional prior p(z | x, d, c).

    Parameters
    ----------
    prior_net : keras Model
        The prior network (inputs: [x, d, c], outputs: [z_mean, z_log_var]).
    x : ndarray (N, 2)
    d, c : ndarray (N, 1)
    deterministic : bool
        If *True*, return z = z_mean (MAP).  Otherwise sample.
    batch_size : int
    seed : int, optional

    Returns
    -------
    z : ndarray (N, latent_dim)
    """
    z_mean, z_log_var = prior_net.predict(
        build_prior_predict_inputs(prior_net, x, d, c),
        batch_size=batch_size,
        verbose=0,
    )
    # Clip log-var consistently with model graph
    z_log_var = np.clip(z_log_var, -10.0, 10.0)

    if deterministic:
        return z_mean
    return reparameterize(z_mean, z_log_var, seed=seed)
