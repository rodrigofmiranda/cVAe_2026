# -*- coding: utf-8 -*-
"""Serializable support-aware Keras layers."""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="VLC")
class SupportGeometryFeatures(tf.keras.layers.Layer):
    """Derive ``[r_l2_norm, r_inf_norm, cornerness_norm]`` from ``x=(I,Q)``."""

    def __init__(self, a_train: float, **kwargs):
        super().__init__(**kwargs)
        self.a_train = float(a_train)

    def call(self, x):
        x = tf.convert_to_tensor(x)
        scale = tf.cast(tf.maximum(self.a_train, 1e-12), x.dtype)
        abs_x = tf.abs(x)
        r_l2 = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))
        r_inf = tf.reduce_max(abs_x, axis=-1, keepdims=True)
        cornerness = (abs_x[..., :1] * abs_x[..., 1:2]) / tf.square(scale)
        return tf.concat([r_l2 / scale, r_inf / scale, cornerness], axis=-1)

    def get_config(self):
        return {**super().get_config(), "a_train": self.a_train}
