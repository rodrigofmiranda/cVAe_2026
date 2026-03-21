# -*- coding: utf-8 -*-
"""
src/models/discriminator.py — Conditional residual discriminator for delta_residual_adv.

The discriminator is a compact MLP that operates on the concatenated
conditioning context and residual: concat(x, d, c, Δ).

It is **conditional** — the same context (x, d, c) is paired with both the
real residual Δ_real = Y − X and the generated residual Δ_fake, so the
network learns whether Δ is plausible given the specific operating point,
not just whether it belongs to the marginal residual distribution.

The output is a raw scalar logit (no sigmoid) to be used with hinge loss
functions from src.models.losses.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf


def build_discriminator(
    input_dim: int,
    layer_sizes: Sequence[int] = (128, 128),
    name: str = "discriminator",
) -> tf.keras.Model:
    """Build a conditional MLP discriminator for the residual cVAE-GAN.

    Parameters
    ----------
    input_dim : int
        Total input feature size — typically x_dim + d_dim + c_dim + delta_dim.
        For the standard I/Q (2-D x, 1-D d, 1-D c, 2-D Δ) case this is 6.
    layer_sizes : sequence of int
        Hidden layer widths. Default (128, 128).
    name : str
        Keras model name. Preserved in saved-model layer hierarchy.

    Returns
    -------
    tf.keras.Model
        Functional model:
          input  — [B, input_dim] concatenated (x, d, c, delta)
          output — [B, 1]         raw logit (no activation)
    """
    inp = tf.keras.Input(shape=(input_dim,), name="disc_input")
    h = inp
    for i, sz in enumerate(layer_sizes):
        h = tf.keras.layers.Dense(
            sz,
            activation="leaky_relu",
            name=f"disc_dense_{i}",
        )(h)
    logit = tf.keras.layers.Dense(1, name="disc_logit")(h)
    return tf.keras.Model(inputs=inp, outputs=logit, name=name)
