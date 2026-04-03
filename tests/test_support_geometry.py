# -*- coding: utf-8 -*-

import math

import numpy as np

from src.data.support_geometry import (
    compute_support_geometry_stats,
    support_feature_dict,
    support_filter_mask,
    support_sample_weights,
)


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
