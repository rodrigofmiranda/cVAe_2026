import numpy as np

from src.training.gridsearch import (
    _apply_regime_weighted_resampling,
    _build_regime_weight_vector,
    _regime_label,
)


def test_build_regime_weight_vector_targets_canonical_regimes():
    d_raw = np.array([0.8, 0.8, 1.0, 1.5], dtype=np.float32).reshape(-1, 1)
    c_raw = np.array([100.0, 300.0, 500.0, 700.0], dtype=np.float32).reshape(-1, 1)

    weights, labels = _build_regime_weight_vector(
        d_raw,
        c_raw,
        {
            "dist_0p8m__curr_100ma": 2.5,
            "dist_0p8m__curr_300ma": 3.0,
        },
    )

    assert labels.tolist() == [
        "dist_0p8m__curr_100mA",
        "dist_0p8m__curr_300mA",
        "dist_1m__curr_500mA",
        "dist_1p5m__curr_700mA",
    ]
    assert weights.tolist() == [2.5, 3.0, 1.0, 1.0]


def test_apply_regime_weighted_resampling_keeps_epoch_size_and_emphasizes_targets():
    n_per_regime = 200
    d_raw = np.concatenate(
        [
            np.full(n_per_regime, 0.8, dtype=np.float32),
            np.full(n_per_regime, 0.8, dtype=np.float32),
            np.full(n_per_regime, 1.0, dtype=np.float32),
        ]
    ).reshape(-1, 1)
    c_raw = np.concatenate(
        [
            np.full(n_per_regime, 100.0, dtype=np.float32),
            np.full(n_per_regime, 300.0, dtype=np.float32),
            np.full(n_per_regime, 500.0, dtype=np.float32),
        ]
    ).reshape(-1, 1)
    n_total = len(d_raw)
    x = np.arange(n_total * 2, dtype=np.float32).reshape(n_total, 2)
    y = x + 1.0
    d_norm = d_raw / 1.5
    c_norm = c_raw / 700.0

    out = _apply_regime_weighted_resampling(
        x_train=x,
        y_train=y,
        d_train_norm=d_norm,
        c_train_norm=c_norm,
        d_train_raw=d_raw,
        c_train_raw=c_raw,
        cfg={
            "train_regime_resample_weights": {
                "dist_0p8m__curr_100ma": 3.0,
                "dist_0p8m__curr_300ma": 3.0,
            }
        },
        seed=42,
    )

    x_out, y_out, d_out, c_out, d_raw_out, c_raw_out, info = out

    assert len(x_out) == n_total
    assert len(y_out) == n_total
    assert len(d_out) == n_total
    assert len(c_out) == n_total
    assert len(d_raw_out) == n_total
    assert len(c_raw_out) == n_total
    assert info["enabled"] is True
    assert info["changed"] is True

    out_labels = [
        _regime_label(float(d), float(c))
        for d, c in zip(d_raw_out.reshape(-1), c_raw_out.reshape(-1))
    ]
    count_100 = sum(label == "dist_0p8m__curr_100mA" for label in out_labels)
    count_300 = sum(label == "dist_0p8m__curr_300mA" for label in out_labels)
    count_1m = sum(label == "dist_1m__curr_500mA" for label in out_labels)

    assert count_100 > n_per_regime
    assert count_300 > n_per_regime
    assert count_1m < n_per_regime
