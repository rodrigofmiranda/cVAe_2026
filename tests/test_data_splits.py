import numpy as np
import pandas as pd

from src.data.splits import cap_val_samples_per_experiment


def test_cap_val_samples_per_experiment_caps_each_experiment_head():
    x_val = np.arange(10, dtype=np.float32).reshape(-1, 1)
    y_val = x_val + 100.0
    d_val = np.zeros((10, 1), dtype=np.float32)
    c_val = np.ones((10, 1), dtype=np.float32)
    df_split = pd.DataFrame(
        [
            {"exp_dir": "exp_a", "n_train": 4, "n_val": 6},
            {"exp_dir": "exp_b", "n_train": 5, "n_val": 4},
        ]
    )

    x_cap, y_cap, d_cap, c_cap, df_cap = cap_val_samples_per_experiment(
        x_val,
        y_val,
        d_val,
        c_val,
        df_split,
        max_samples_per_experiment=3,
    )

    assert x_cap.shape == (6, 1)
    assert y_cap.shape == (6, 1)
    assert d_cap.shape == (6, 1)
    assert c_cap.shape == (6, 1)

    # Deterministic head keep inside each experiment: [0,1,2] + [6,7,8]
    assert np.allclose(x_cap[:, 0], np.array([0, 1, 2, 6, 7, 8], dtype=np.float32))
    assert np.allclose(y_cap[:, 0], np.array([100, 101, 102, 106, 107, 108], dtype=np.float32))

    assert df_cap.to_dict(orient="records") == [
        {"exp_dir": "exp_a", "n_val_before": 6, "n_val_after": 3},
        {"exp_dir": "exp_b", "n_val_before": 4, "n_val_after": 3},
    ]
