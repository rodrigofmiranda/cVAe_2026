# -*- coding: utf-8 -*-

import numpy as np

from src.evaluation.metrics import (
    residual_distribution_metrics,
    residual_signature_by_amplitude_bin,
    residual_signature_by_support_bin,
)


def test_residual_distribution_metrics_exposes_signature_and_coverage_fields():
    rng = np.random.default_rng(20260324)
    n = 256
    x = rng.normal(scale=0.2, size=(n, 2))
    y_true = x + rng.normal(scale=0.10, size=(n, 2))
    y_samples = np.stack(
        [
            x + rng.normal(scale=0.12, size=(n, 2)),
            x + rng.normal(scale=0.14, size=(n, 2)),
            x + rng.normal(scale=0.16, size=(n, 2)),
            x + rng.normal(scale=0.18, size=(n, 2)),
        ],
        axis=0,
    )
    y_pred = y_samples.mean(axis=0)
    metrics = residual_distribution_metrics(
        x,
        y_true,
        y_pred,
        psd_nfft=256,
        Y_samples=y_samples,
        coverage_target=y_true,
    )

    for key in (
        "var_ratio_I",
        "var_ratio_Q",
        "iqr_real_I",
        "iqr_pred_Q",
        "q05_real_I",
        "q95_pred_Q",
        "delta_q50_I",
        "tail_p3sigma_real_I",
        "tail_p3sigma_pred_Q",
        "radial_wasserstein",
        "delta_corr_IQ",
        "ellipse_axis_ratio_real",
        "coverage_50",
        "coverage_80",
        "coverage_95",
        "delta_coverage_95",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key]) or np.isnan(metrics[key])


def test_residual_signature_by_amplitude_bin_builds_rows_and_respects_min_samples():
    rng = np.random.default_rng(7)
    n = 4096
    x_real = rng.normal(scale=0.4, size=(n, 2))
    y_real = x_real + rng.normal(scale=0.12, size=(n, 2))
    x_pred = np.tile(x_real, (2, 1))
    y_pred = x_pred + rng.normal(scale=0.14, size=x_pred.shape)

    rows = residual_signature_by_amplitude_bin(
        X_real=x_real,
        Y_real=y_real,
        X_pred=x_pred,
        Y_pred=y_pred,
        regime_id="dist_0p8m__curr_300mA",
        dist_target_m=0.8,
        curr_target_mA=300.0,
        amplitude_bins=4,
        min_samples_per_bin=512,
        stat_n_perm=32,
        stat_seed=11,
    )

    assert len(rows) == 4
    assert {row["amplitude_bin_index"] for row in rows} == {0, 1, 2, 3}
    assert all(row["n_samples_real"] >= 512 for row in rows)
    assert all("delta_wasserstein_I" in row for row in rows)
    assert all("stat_mmd_qval" in row for row in rows)
    assert all("stat_energy_qval" in row for row in rows)


def test_residual_signature_by_support_bin_builds_axis_and_region_rows():
    rng = np.random.default_rng(17)
    n = 4096
    x_real = rng.uniform(-1.0, 1.0, size=(n, 2))
    y_real = x_real + rng.normal(scale=0.10, size=(n, 2))
    x_pred = np.tile(x_real, (2, 1))
    y_pred = x_pred + rng.normal(scale=0.12, size=x_pred.shape)

    rows = residual_signature_by_support_bin(
        X_real=x_real,
        Y_real=y_real,
        X_pred=x_pred,
        Y_pred=y_pred,
        a_train=1.0,
        regime_id="dist_0p8m__curr_300mA",
        support_bins=4,
        min_samples_per_bin=512,
        stat_n_perm=16,
        stat_seed=23,
    )

    assert len(rows) == 11
    assert {row["support_axis"] for row in rows} == {"r_l2_norm", "r_inf_norm", "support_region"}
    assert {"center", "edge", "corner"} <= {row["support_region"] for row in rows}
    assert all("delta_wasserstein_I" in row for row in rows)
    assert all("stat_mmd_qval" in row for row in rows)
