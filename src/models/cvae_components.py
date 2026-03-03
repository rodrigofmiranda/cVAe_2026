# -*- coding: utf-8 -*-
"""
src/models/cvae_components.py — Structural components of the cVAE.

Extracted from cvae_TRAIN_documented.py and analise_cvae_reviewed.py
(Commit 3F).  Contains custom Keras layers, model-build helpers, and
the top-level ``build_condprior_cvae`` assembler.

Training callbacks (EarlyStoppingAfterWarmup) remain in the training
monolith because they are training-loop code, not model structure.

Contents
--------
_activation_layer        Activation selector (LeakyReLU 0.2 or named)
Sampling                 Reparametrisation trick layer
CondPriorVAELoss         Heteroscedastic Gaussian NLL + KL(q‖p) + β
KLAnnealingCallback      Linear β ramp tied to CondPriorVAELoss
build_mlp                Generic MLP → (z_mean, z_log_var)
build_decoder            Decoder MLP → output_params (4-d)
build_condprior_cvae     Full cVAE assembly + compile
create_inference_model_from_full   Inference graph from saved full model
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback


# ==========================================================================
# Activation helper
# ==========================================================================
def _activation_layer(name: str):
    name = (name or "").lower().strip()
    if name in ["leaky_relu", "lrelu", "leakyrelu"]:
        return layers.LeakyReLU(alpha=0.2)
    return layers.Activation(name)


# ==========================================================================
# Custom Keras layers
# ==========================================================================
@tf.keras.utils.register_keras_serializable(package="VLC")
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


@tf.keras.utils.register_keras_serializable(package="VLC")
# ---------------------------------------------------------------------------
# Loss heteroscedástico + KL(q||p) com β-annealing e free-bits.
# ---------------------------------------------------------------------------
class CondPriorVAELoss(layers.Layer):
    def __init__(self, beta=1.0, free_bits=0.0, **kwargs):
        super().__init__(**kwargs)
        self.beta_init = float(beta)
        self.free_bits = float(free_bits)
        self.beta = tf.Variable(self.beta_init, trainable=False, dtype=tf.float32, name="beta")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p = inputs

        y_mean = out_params[:, :2]
        y_log_var = tf.clip_by_value(out_params[:, 2:], -6.0, 1.0)
        y_var = tf.exp(y_log_var) + 1e-6
        nll = 0.5 * tf.reduce_sum(
            y_log_var + tf.square(y_true - y_mean) / y_var + tf.math.log(2.0*np.pi),
            axis=-1
        )
        recon = tf.reduce_mean(nll)

        vq = tf.exp(tf.clip_by_value(z_log_var_q, -20.0, 20.0))
        vp = tf.exp(tf.clip_by_value(z_log_var_p, -20.0, 20.0))
        kl_dim = 0.5 * (
            tf.math.log(vp + 1e-12) - tf.math.log(vq + 1e-12)
            + (vq + tf.square(z_mean_q - z_mean_p)) / (vp + 1e-12)
            - 1.0
        )
        kl_per_sample = tf.reduce_sum(kl_dim, axis=-1)

        fb = tf.cast(self.free_bits, kl_per_sample.dtype)
        kl_fb = tf.maximum(kl_per_sample - fb, 0.0)
        kl = tf.reduce_mean(kl_fb)

        total = recon + self.beta * tf.minimum(kl, 200.0)

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


# ==========================================================================
# KL annealing callback (tightly coupled to CondPriorVAELoss.beta)
# ==========================================================================
class KLAnnealingCallback(Callback):
    def __init__(self, loss_layer, beta_start=0.0, beta_end=1.0, annealing_epochs=50):
        super().__init__()
        self.loss_layer = loss_layer
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.annealing_epochs = int(annealing_epochs)

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.annealing_epochs:
            progress = epoch / max(self.annealing_epochs, 1)
            b = self.beta_start + (self.beta_end - self.beta_start) * progress
            self.loss_layer.beta.assign(b)
        else:
            self.loss_layer.beta.assign(self.beta_end)


# ==========================================================================
# Model-build helpers
# ==========================================================================
def build_mlp(name, in_shapes, layer_sizes, activation="leaky_relu", dropout=0.0, out_dim=32, out_name_prefix=""):
    ins = [layers.Input(shape=s, name=f"{name}_in_{i}") for i, s in enumerate(in_shapes)]
    h = layers.Concatenate(name=f"{name}_concat")(ins)
    for i, u in enumerate(layer_sizes):
        h = layers.Dense(u, kernel_initializer="glorot_uniform", name=f"{name}_dense_{i}")(h)
        h = layers.BatchNormalization(name=f"{name}_bn_{i}")(h)
        h = _activation_layer(activation)(h)
        if dropout and dropout > 0:
            h = layers.Dropout(dropout, name=f"{name}_drop_{i}")(h)
    mu = layers.Dense(out_dim, name=f"{out_name_prefix}z_mean")(h)
    lv = layers.Dense(out_dim, name=f"{out_name_prefix}z_log_var")(h)
    return models.Model(ins, [mu, lv], name=name)


def build_decoder(layer_sizes, latent_dim, activation="leaky_relu", dropout=0.0):
    z_in = layers.Input(shape=(latent_dim,), name="z_input")
    cond_in = layers.Input(shape=(4,), name="cond_input")  # x(2)+d(1)+c(1)
    h = layers.Concatenate(name="dec_concat")([z_in, cond_in])
    for i, u in enumerate(layer_sizes):
        h = layers.Dense(u, kernel_initializer="glorot_uniform", name=f"dec_dense_{i}")(h)
        h = layers.BatchNormalization(name=f"dec_bn_{i}")(h)
        h = _activation_layer(activation)(h)
        if dropout and dropout > 0:
            h = layers.Dropout(dropout, name=f"dec_drop_{i}")(h)
    out = layers.Dense(4, name="output_params")(h)  # mean_I,mean_Q,logvar_I,logvar_Q
    return models.Model([z_in, cond_in], out, name="decoder")


# ==========================================================================
# Full cVAE assembly + compile
# ==========================================================================
def build_condprior_cvae(cfg):
    layer_sizes = cfg["layer_sizes"]
    latent_dim = int(cfg["latent_dim"])
    beta = float(cfg["beta"])
    lr = float(cfg["lr"])
    dropout = float(cfg["dropout"])
    free_bits = float(cfg.get("free_bits", 0.0))
    kl_anneal_epochs = int(cfg.get("kl_anneal_epochs", 50))
    activation = cfg.get("activation", "leaky_relu")

    encoder = build_mlp(
        name="encoder",
        in_shapes=[(2,), (1,), (1,), (2,)],
        layer_sizes=layer_sizes,
        activation=activation,
        dropout=dropout,
        out_dim=latent_dim,
        out_name_prefix="q_",
    )

    prior_net = build_mlp(
        name="prior_net",
        in_shapes=[(2,), (1,), (1,)],
        layer_sizes=layer_sizes,
        activation=activation,
        dropout=dropout,
        out_dim=latent_dim,
        out_name_prefix="p_",
    )

    decoder = build_decoder(layer_sizes=layer_sizes, latent_dim=latent_dim, activation=activation, dropout=dropout)

    x_in = layers.Input(shape=(2,), name="x_input")
    d_in = layers.Input(shape=(1,), name="distance_input")
    c_in = layers.Input(shape=(1,), name="current_input")
    y_in = layers.Input(shape=(2,), name="y_true")

    z_mean_q, z_log_var_q = encoder([x_in, d_in, c_in, y_in])
    z_mean_p, z_log_var_p = prior_net([x_in, d_in, c_in])

    # >> C4.2 FIX: clip do log-var do prior também no treino, igual à inferência.
    #    Sem isso, z_log_var_p pode divergir durante o treino enquanto na inferência
    #    é clipado — viés sistemático que infla artificialmente o KL.
    z_log_var_p = layers.Lambda(
        lambda t: tf.clip_by_value(t, -10.0, 10.0), name="clip_p_logvar_train"
    )(z_log_var_p)

    z = Sampling(name="sampling")([z_mean_q, z_log_var_q])

    cond = layers.Concatenate(name="cond_concat")([x_in, d_in, c_in])  # (N,4)
    out_params = decoder([z, cond])

    beta_initial = 0.0
    loss_layer = CondPriorVAELoss(beta=beta_initial, free_bits=free_bits, name="condprior_loss")
    y_mean = loss_layer([y_in, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p])

    vae = models.Model([x_in, d_in, c_in, y_in], y_mean, name="cvae_condprior")
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    vae.compile(optimizer=opt)

    kl_cb = KLAnnealingCallback(loss_layer, beta_start=0.0, beta_end=beta, annealing_epochs=kl_anneal_epochs)
    return vae, kl_cb


# ==========================================================================
# Inference model (prior condicional) from saved full model
# ==========================================================================
def create_inference_model_from_full(full_model: tf.keras.Model, deterministic: bool = True):
    prior = full_model.get_layer("prior_net")
    dec = full_model.get_layer("decoder")

    x_in = layers.Input(shape=(2,), name="x_input")
    d_in = layers.Input(shape=(1,), name="distance_input")
    c_in = layers.Input(shape=(1,), name="current_input")

    z_mean_p, z_log_var_p = prior([x_in, d_in, c_in])
    z_log_var_p = layers.Lambda(lambda t: tf.clip_by_value(t, -10.0, 10.0), name="clip_zlogvar")(z_log_var_p)

    if deterministic:
        z = layers.Lambda(lambda t: t, name="z_det")(z_mean_p)
    else:
        eps_z = layers.Lambda(lambda t: tf.random.normal(tf.shape(t)), name="eps_z")(z_mean_p)
        z = layers.Lambda(lambda a: a[0] + tf.exp(0.5 * a[1]) * a[2], name="sample_z")([z_mean_p, z_log_var_p, eps_z])

    cond = layers.Concatenate(name="cond_concat_inf")([x_in, d_in, c_in])
    out_params = dec([z, cond])

    y_mean = layers.Lambda(lambda t: t[:, :2], name="y_mean")(out_params)
    y_log_var = layers.Lambda(lambda t: tf.clip_by_value(t[:, 2:], -6.0, 1.0), name="y_logvar")(out_params)

    if deterministic:
        y = layers.Lambda(lambda t: t, name="y_det")(y_mean)
    else:
        eps_y = layers.Lambda(lambda t: tf.random.normal(tf.shape(t)), name="eps_y")(y_mean)
        y = layers.Lambda(lambda a: a[0] + tf.exp(0.5 * a[1]) * a[2], name="sample_y")([y_mean, y_log_var, eps_y])

    return models.Model([x_in, d_in, c_in], y,
                        name=("inference_condprior_det" if deterministic else "inference_condprior"))
