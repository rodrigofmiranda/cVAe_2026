# -*- coding: utf-8 -*-
"""
src/models/sampling.py — Reparameterisation trick and latent-space sampling.

Extracted from ``cvae_components.py`` (refactor step 3).

Public API
----------
Sampling                Keras layer (for model graphs)
reparameterize          Functional NumPy / eager-TF form
sample_prior            Draw z from the conditional prior network
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
        [x, d, c], batch_size=batch_size, verbose=0,
    )
    # Clip log-var consistently with model graph
    z_log_var = np.clip(z_log_var, -10.0, 10.0)

    if deterministic:
        return z_mean
    return reparameterize(z_mean, z_log_var, seed=seed)
