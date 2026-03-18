import numpy as np

from src.data.loading import reduce_aligned_arrays
from src.data.normalization import normalize_conditions


def test_reduce_aligned_arrays_center_crop_keeps_conditions_in_sync():
    X = np.arange(24, dtype=np.float32).reshape(12, 2)
    Y = (100 + np.arange(24, dtype=np.float32)).reshape(12, 2)
    D = np.concatenate(
        [
            np.full((6, 1), 0.8, dtype=np.float32),
            np.full((6, 1), 1.5, dtype=np.float32),
        ],
        axis=0,
    )
    C = np.concatenate(
        [
            np.full((6, 1), 100.0, dtype=np.float32),
            np.full((6, 1), 700.0, dtype=np.float32),
        ],
        axis=0,
    )

    cfg = {
        "enabled": True,
        "mode": "center_crop",
        "target_samples_per_experiment": 4,
        "min_samples_per_experiment": 4,
    }
    rng = np.random.default_rng(0)

    Xr, Yr, Dr, Cr = reduce_aligned_arrays(X, Y, D, C, cfg=cfg, rng=rng)

    expected_idx = np.arange(4, 8, dtype=np.int64)
    np.testing.assert_array_equal(Xr, X[expected_idx])
    np.testing.assert_array_equal(Yr, Y[expected_idx])
    np.testing.assert_array_equal(Dr, D[expected_idx])
    np.testing.assert_array_equal(Cr, C[expected_idx])


def test_reduce_aligned_arrays_balanced_blocks_keeps_conditions_in_sync():
    X = np.arange(32, dtype=np.float32).reshape(16, 2)
    Y = (200 + np.arange(32, dtype=np.float32)).reshape(16, 2)
    D = np.concatenate(
        [
            np.full((8, 1), 0.8, dtype=np.float32),
            np.full((8, 1), 1.5, dtype=np.float32),
        ],
        axis=0,
    )
    C = np.concatenate(
        [
            np.full((8, 1), 100.0, dtype=np.float32),
            np.full((8, 1), 700.0, dtype=np.float32),
        ],
        axis=0,
    )

    cfg = {
        "enabled": True,
        "mode": "balanced_blocks",
        "target_samples_per_experiment": 8,
        "min_samples_per_experiment": 8,
        "block_len": 2,
        "time_spread": True,
        "min_gap_blocks": 1,
    }
    rng = np.random.default_rng(42)

    Xr, Yr, Dr, Cr = reduce_aligned_arrays(X, Y, D, C, cfg=cfg, rng=rng)

    original_idx = (Xr[:, 0] / 2.0).astype(np.int64)
    np.testing.assert_array_equal(Yr, Y[original_idx])
    np.testing.assert_array_equal(Dr, D[original_idx])
    np.testing.assert_array_equal(Cr, C[original_idx])


def test_normalization_after_aligned_reduction_preserves_condition_range():
    D_train = np.array([[0.8], [0.8], [1.0], [1.5], [1.5]], dtype=np.float32)
    C_train = np.array([[100.0], [300.0], [500.0], [700.0], [700.0]], dtype=np.float32)
    D_val = np.array([[1.0], [1.5]], dtype=np.float32)
    C_val = np.array([[300.0], [700.0]], dtype=np.float32)

    Dn_train, Cn_train, Dn_val, Cn_val, params = normalize_conditions(
        D_train, C_train, D_val, C_val,
    )

    assert params["D_min"] == 0.8
    assert params["D_max"] == 1.5
    assert params["C_min"] == 100.0
    assert params["C_max"] == 700.0
    assert float(Dn_train.min()) == 0.0
    assert float(Dn_train.max()) == 1.0
    assert float(Cn_train.min()) == 0.0
    assert float(Cn_train.max()) == 1.0
    assert np.all((Dn_val >= 0.0) & (Dn_val <= 1.0))
    assert np.all((Cn_val >= 0.0) & (Cn_val <= 1.0))
