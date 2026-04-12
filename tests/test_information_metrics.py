import math

import numpy as np

from src.evaluation.information_metrics import auxiliary_information_metrics


def test_auxiliary_information_metrics_reports_qam_bounds_for_discrete_input():
    levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=float)
    const = np.array([[i, q] for i in levels for q in levels], dtype=float)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(const), size=16000)
    x = const[idx]
    y_real = x + 0.20 * rng.standard_normal(x.shape)
    y_pred = x + 0.35 * rng.standard_normal(x.shape)

    out = auxiliary_information_metrics(
        X_real=x,
        Y_real=y_real,
        X_pred=x,
        Y_pred=y_pred,
        seed=42,
    )

    assert bool(out["info_metrics_available"]) is True
    assert out["info_metrics_status"] == "ok"
    assert out["info_alphabet_size"] == 16
    assert out["info_bits_per_symbol"] == 4
    assert out["info_labeling_mode"] == "gray_rect_qam_uniform"
    assert math.isfinite(float(out["mi_aux_real_bits"]))
    assert math.isfinite(float(out["mi_aux_pred_bits"]))
    assert math.isfinite(float(out["gmi_aux_real_bits"]))
    assert math.isfinite(float(out["gmi_aux_pred_bits"]))
    assert math.isfinite(float(out["ngmi_aux_real"]))
    assert math.isfinite(float(out["ngmi_aux_pred"]))
    assert math.isfinite(float(out["air_aux_real_bits"]))
    assert math.isfinite(float(out["air_aux_pred_bits"]))
    assert float(out["mi_aux_real_bits"]) >= float(out["mi_aux_pred_bits"])
    assert float(out["gmi_aux_real_bits"]) >= float(out["gmi_aux_pred_bits"])
    assert float(out["ngmi_aux_real"]) >= float(out["ngmi_aux_pred"])


def test_auxiliary_information_metrics_disables_for_continuous_input():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=(10000, 2))
    y = x + 0.1 * rng.standard_normal(x.shape)

    out = auxiliary_information_metrics(
        X_real=x,
        Y_real=y,
        X_pred=x,
        Y_pred=y,
        seed=0,
    )

    assert bool(out["info_metrics_available"]) is False
    assert out["info_metrics_status"] in {
        "alphabet_too_large",
        "alphabet_not_repeated_enough",
    }
    assert math.isnan(float(out["mi_aux_real_bits"]))
    assert math.isnan(float(out["gmi_aux_real_bits"]))
    assert math.isnan(float(out["ngmi_aux_real"]))
