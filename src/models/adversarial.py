# -*- coding: utf-8 -*-
"""
src/models/adversarial.py — Adversarial cVAE wrapper for delta_residual_adv.

Implements ``AdvResidualCVAEModel``: a tf.keras.Model subclass that holds
the standard cVAE sub-models (encoder, prior_net, decoder) plus a conditional
residual discriminator, and runs alternating D/G updates via a custom
``train_step``.

The generator loss is:
    L_G = L_recon + β * L_KL + λ_adv * L_adv(hinge)

The discriminator loss is hinge GAN (see ``src.models.losses``).

β is annealed via ``KLAnnealingCallback``, which updates ``wrapper.beta``
(a tf.Variable attribute).  The callback interface is identical to the
loss-layer interface used by the non-adversarial variants.

Protocol compatibility
----------------------
- ``encoder``, ``prior_net``, ``decoder`` are tracked sub-models with stable
  names, so ``create_inference_model_from_full`` can extract them via
  ``model.get_layer(name)``.
- ``train_step`` returns ``recon_loss`` and ``kl_loss`` so that the
  monitoring key ``val_recon_loss`` used by gridsearch.py is present.
- ``test_step`` mirrors the generator evaluation path (no gradient updates).
"""

from __future__ import annotations

from typing import Dict, List

import tensorflow as tf

from src.models.losses import (
    hinge_discriminator_loss,
    hinge_generator_loss,
    kl_divergence,
    kl_with_freebits,
    reconstruction_loss,
)


@tf.keras.utils.register_keras_serializable(package="VLC")
class AdvResidualCVAEModel(tf.keras.Model):
    """Conditional residual cVAE-GAN with alternating D/G train_step.

    Parameters
    ----------
    encoder : keras.Model  q(z | x, d, c, y) → (z_mean, z_log_var)
    prior_net : keras.Model  p(z | x, d, c) → (z_mean, z_log_var)
    decoder : keras.Model  p(Δ | z, cond) → delta_params [B, 4]
              The decoder must expose a ``delta_output_params`` layer
              (i.e. built with arch_variant="delta_residual").
    discriminator : keras.Model  concat(x, d, c, Δ) → logit [B, 1]
    beta_start : initial β for KL term (annealed by callback).
    lambda_adv : weight of the adversarial generator loss.
    free_bits : free-bits threshold for KL (per-dim).
    lambda_mmd : MMD weight (currently unused — reserved for future use).
    """

    def __init__(
        self,
        encoder: tf.keras.Model,
        prior_net: tf.keras.Model,
        decoder: tf.keras.Model,
        discriminator: tf.keras.Model,
        beta_start: float = 0.0,
        lambda_adv: float = 0.05,
        free_bits: float = 0.0,
        lambda_mmd: float = 0.0,
    ) -> None:
        super().__init__(name="adv_residual_cvae")
        # Stable sub-model names required for create_inference_model_from_full.
        self.encoder = encoder          # name="encoder"
        self.prior_net = prior_net      # name="prior_net"
        self.decoder = decoder          # name="decoder"
        self.discriminator = discriminator  # name="discriminator"

        # β tf.Variable — updated each epoch by KLAnnealingCallback.
        self.beta = tf.Variable(
            float(beta_start), trainable=False, dtype=tf.float32, name="beta",
        )
        self.lambda_adv = float(lambda_adv)
        self.free_bits = float(free_bits)
        self.lambda_mmd = float(lambda_mmd)  # reserved

    # ------------------------------------------------------------------
    # Keras serialization — needed for SavedModel round-trip
    # ------------------------------------------------------------------
    def get_config(self):
        return {
            "encoder_config": tf.keras.layers.serialize(self.encoder),
            "prior_net_config": tf.keras.layers.serialize(self.prior_net),
            "decoder_config": tf.keras.layers.serialize(self.decoder),
            "discriminator_config": tf.keras.layers.serialize(self.discriminator),
            "beta_start": float(self.beta.numpy()),
            "lambda_adv": self.lambda_adv,
            "free_bits": self.free_bits,
            "lambda_mmd": self.lambda_mmd,
        }

    @classmethod
    def from_config(cls, config):
        encoder = tf.keras.layers.deserialize(config["encoder_config"])
        prior_net = tf.keras.layers.deserialize(config["prior_net_config"])
        decoder = tf.keras.layers.deserialize(config["decoder_config"])
        discriminator = tf.keras.layers.deserialize(config["discriminator_config"])
        return cls(
            encoder=encoder,
            prior_net=prior_net,
            decoder=decoder,
            discriminator=discriminator,
            beta_start=config.get("beta_start", 0.0),
            lambda_adv=config.get("lambda_adv", 0.05),
            free_bits=config.get("free_bits", 0.0),
            lambda_mmd=config.get("lambda_mmd", 0.0),
        )

    # ------------------------------------------------------------------
    # compile — stores two optimizers
    # ------------------------------------------------------------------
    def compile(
        self,
        g_optimizer: tf.keras.optimizers.Optimizer,
        d_optimizer: tf.keras.optimizers.Optimizer,
        **kwargs,
    ) -> None:
        super().compile(**kwargs)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    # ------------------------------------------------------------------
    # call — inference path (prior sample → decoder → X + Δ_mean)
    # Used when the wrapper itself is called directly, e.g. wrapper.predict.
    # The protocol uses create_inference_model_from_full instead.
    # ------------------------------------------------------------------
    def call(self, inputs, training: bool = False):
        """Prior-sample inference: inputs=[x, d, c] → Ŷ = X + Δ_mean."""
        x_in, d_in, c_in = inputs[0], inputs[1], inputs[2]
        z_mean_p, z_log_var_p = self.prior_net([x_in, d_in, c_in], training=training)
        eps = tf.random.normal(tf.shape(z_mean_p))
        z = z_mean_p + tf.exp(0.5 * z_log_var_p) * eps
        cond = tf.concat([x_in, d_in, c_in], axis=-1)
        out_params = self.decoder([z, cond], training=training)
        delta_mean = out_params[:, :2]
        return x_in + delta_mean

    # ------------------------------------------------------------------
    # train_step — alternating D then G update
    # ------------------------------------------------------------------
    def train_step(self, data) -> Dict[str, tf.Tensor]:
        """One training step: 1 discriminator update then 1 generator update."""
        inputs, _ = data
        x_in, d_in, c_in, y_in = inputs[0], inputs[1], inputs[2], inputs[3]
        delta_real = y_in - x_in
        cond = tf.concat([x_in, d_in, c_in], axis=-1)  # (B, 4)

        # ── Discriminator step ──────────────────────────────────────
        with tf.GradientTape() as d_tape:
            # Sample fake residuals using the posterior encoder (not the prior)
            # so the discriminator sees plausible generation at all β values.
            z_mean_q, z_log_var_q = self.encoder(
                [x_in, d_in, c_in, y_in], training=True,
            )
            eps = tf.random.normal(tf.shape(z_mean_q))
            z_q = z_mean_q + tf.exp(0.5 * z_log_var_q) * eps
            out_params = self.decoder([z_q, cond], training=True)
            delta_fake = self._sample_delta(out_params)

            disc_in_real = tf.concat([x_in, d_in, c_in, delta_real], axis=-1)
            disc_in_fake = tf.concat([x_in, d_in, c_in, delta_fake], axis=-1)
            d_real_logits = self.discriminator(disc_in_real, training=True)
            d_fake_logits = self.discriminator(disc_in_fake, training=True)
            d_loss = hinge_discriminator_loss(d_real_logits, d_fake_logits)

        d_grads = d_tape.gradient(
            d_loss, self.discriminator.trainable_variables,
        )
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables),
        )

        # ── Generator step ──────────────────────────────────────────
        gen_vars: List[tf.Variable] = (
            self.encoder.trainable_variables
            + self.prior_net.trainable_variables
            + self.decoder.trainable_variables
        )
        with tf.GradientTape() as g_tape:
            z_mean_q, z_log_var_q = self.encoder(
                [x_in, d_in, c_in, y_in], training=True,
            )
            z_mean_p, z_log_var_p = self.prior_net(
                [x_in, d_in, c_in], training=True,
            )
            eps = tf.random.normal(tf.shape(z_mean_q))
            z_q = z_mean_q + tf.exp(0.5 * z_log_var_q) * eps
            out_params = self.decoder([z_q, cond], training=True)
            delta_fake_mean = out_params[:, :2]
            delta_fake_logvar = out_params[:, 2:]
            delta_fake = self._sample_delta(out_params)

            delta_true = y_in - x_in
            recon = tf.reduce_mean(
                reconstruction_loss(delta_true, delta_fake_mean, delta_fake_logvar),
            )
            kl_raw = kl_divergence(z_mean_q, z_log_var_q, z_mean_p, z_log_var_p)
            kl = tf.reduce_mean(kl_with_freebits(kl_raw, self.free_bits))
            disc_in_fake = tf.concat([x_in, d_in, c_in, delta_fake], axis=-1)
            # Discriminator is not updated in this step (training=False).
            adv = hinge_generator_loss(
                self.discriminator(disc_in_fake, training=False),
            )
            g_loss = recon + self.beta * kl + self.lambda_adv * adv

        g_grads = g_tape.gradient(g_loss, gen_vars)
        self.g_optimizer.apply_gradients(zip(g_grads, gen_vars))

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "recon_loss": recon,   # val_recon_loss used by gridsearch monitor
            "kl_loss": kl,
            "adv_loss": adv,
        }

    # ------------------------------------------------------------------
    # test_step — generator forward pass for validation metrics
    # ------------------------------------------------------------------
    def test_step(self, data) -> Dict[str, tf.Tensor]:
        """Validation step: generator forward pass only (no weight updates)."""
        inputs, _ = data
        x_in, d_in, c_in, y_in = inputs[0], inputs[1], inputs[2], inputs[3]
        cond = tf.concat([x_in, d_in, c_in], axis=-1)

        z_mean_q, z_log_var_q = self.encoder(
            [x_in, d_in, c_in, y_in], training=False,
        )
        z_mean_p, z_log_var_p = self.prior_net(
            [x_in, d_in, c_in], training=False,
        )
        eps = tf.random.normal(tf.shape(z_mean_q))
        z_q = z_mean_q + tf.exp(0.5 * z_log_var_q) * eps
        out_params = self.decoder([z_q, cond], training=False)
        delta_fake_mean = out_params[:, :2]
        delta_fake_logvar = out_params[:, 2:]

        delta_true = y_in - x_in
        recon = tf.reduce_mean(
            reconstruction_loss(delta_true, delta_fake_mean, delta_fake_logvar),
        )
        kl_raw = kl_divergence(z_mean_q, z_log_var_q, z_mean_p, z_log_var_p)
        kl = tf.reduce_mean(kl_with_freebits(kl_raw, self.free_bits))
        return {"recon_loss": recon, "kl_loss": kl}

    @staticmethod
    def _sample_delta(out_params: tf.Tensor) -> tf.Tensor:
        """Sample residuals from the heteroscedastic decoder head.

        The adversarial discriminator must see the full predictive
        distribution, not only the residual mean. Otherwise the GAN term
        never pushes the decoder variance head that controls stochastic
        fidelity and heteroscedasticity.
        """
        delta_mean = out_params[:, :2]
        delta_logvar = out_params[:, 2:]
        eps = tf.random.normal(tf.shape(delta_mean))
        return delta_mean + tf.exp(0.5 * delta_logvar) * eps
