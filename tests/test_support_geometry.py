# -*- coding: utf-8 -*-

import math

import numpy as np
import tensorflow as tf

from src.data.support_geometry import (
    compute_support_geometry_stats,
    resolve_support_experiment_config,
    support_feature_dict,
    support_filter_mask,
    support_sample_weights,
)
from src.models.losses import _support_weights_from_x_tf


def test_support_geometry_stats_and_features_match_expected_values():
    x = np.array(
        [
            [1.0, 0.0],
            [0.0, -0.5],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    stats = compute_support_geometry_stats(x)
    feats = support_feature_dict(x, a_train=stats.a_train)

    assert math.isclose(stats.a_train, 1.0, rel_tol=1e-9)
    np.testing.assert_allclose(feats["r_l2_norm"], [1.0, 0.5, math.sqrt(0.5)], atol=1e-6)
    np.testing.assert_allclose(feats["r_inf_norm"], [1.0, 0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(feats["cornerness_norm"], [0.0, 0.0, 0.25], atol=1e-6)


def test_support_sample_weights_and_disk_filter_follow_train_scale():
    x = np.array(
        [
            [0.0, 0.0],
            [0.80, 0.10],
            [1.00, 1.00],
        ],
        dtype=np.float32,
    )
    weights = support_sample_weights(
        x,
        a_train=1.0,
        mode="edge_rinf_corner",
        alpha=1.5,
        tau=0.75,
        tau_corner=0.35,
        weight_max=3.0,
    )
    mask = support_filter_mask(x, a_train=1.0, mode="disk_l2")

    assert weights.shape == (3,)
    assert weights[0] == 1.0
    assert weights[1] > 1.0
    assert weights[2] > weights[1]
    assert mask.tolist() == [True, True, False]


def test_support_sample_weights_match_tensorflow_training_weights():
    x = np.array(
        [
            [0.00, 0.00],
            [0.80, 0.10],
            [0.95, 0.60],
            [1.00, 1.00],
        ],
        dtype=np.float32,
    )
    weights_np = support_sample_weights(
        x,
        a_train=1.0,
        mode="edge_rinf_corner",
        alpha=1.5,
        tau=0.75,
        tau_corner=0.35,
        weight_max=3.0,
    )
    weights_tf = _support_weights_from_x_tf(
        tf.constant(x),
        a_train=1.0,
        mode="edge_rinf_corner",
        alpha=1.5,
        tau=0.75,
        tau_corner=0.35,
        weight_max=3.0,
    )

    np.testing.assert_allclose(weights_np, weights_tf.numpy(), atol=1e-6)


def test_disk_filter_preserves_average_power_for_uniform_square_samples():
    rng = np.random.default_rng(123)
    x = rng.uniform(-1.0, 1.0, size=(100_000, 2)).astype(np.float32)
    mask = support_filter_mask(x, a_train=1.0, mode="disk_l2")
    mean_power_kept = float(np.mean(np.sum(np.square(x[mask]), axis=1)))

    assert abs(mean_power_kept - (2.0 / 3.0)) < 0.02


def test_resolve_support_experiment_config_uses_override_precedence():
    resolved = resolve_support_experiment_config(
        overrides={
            "support_weight_mode": "edge_rinf_corner",
            "support_diag_bins": 6,
            "support_feature_scale": 1.25,
        },
        state_support_cfg={
            "support_feature_mode": "geom3",
            "support_weight_mode": "edge_rinf",
            "support_filter_mode": "disk_l2",
            "support_diag_bins": 4,
            "a_train": 0.9,
        },
    )

    assert resolved["support_feature_mode"] == "geom3"
    assert resolved["support_weight_mode"] == "edge_rinf_corner"
    assert resolved["support_filter_mode"] == "disk_l2"
    assert resolved["support_diag_bins"] == 6
    assert math.isclose(resolved["a_train"], 1.25, rel_tol=1e-9)
