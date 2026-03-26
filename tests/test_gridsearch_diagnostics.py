import math

import numpy as np
import pandas as pd

from src.training.gridsearch import (
    TRAINING_DIAGNOSTIC_COLUMNS,
    _candidate_ranking_key,
    _apply_training_recommendations,
    _build_training_diagnostics,
    _build_training_diagnostics_table,
    _mc_point_metric_means,
    _sort_results_by_ranking,
)


def test_build_training_diagnostics_flags_collapse_undertrained_and_lr_floor():
    history = {
        "recon_loss": [3.0, 2.5, 2.0, 1.8, 1.7],
        "val_recon_loss": [2.4, 2.1, 1.9, 1.7, 1.6],
        "lr": [3e-4, 1.5e-4, 7.5e-5, 3.75e-5, 1.0e-6],
    }

    out = _build_training_diagnostics(
        history_dict=history,
        cfg={"latent_dim": 8, "lr": 3e-4},
        max_epochs=5,
        active_dims=2,
        kl_mean_total=0.8,
        kl_mean_per_dim=0.1,
    )

    assert out["epochs_ran"] == 5
    assert out["best_epoch"] == 5
    assert math.isclose(out["best_epoch_ratio"], 1.0)
    assert out["flag_posterior_collapse"] is True
    assert out["flag_undertrained"] is True
    assert out["flag_unstable"] is True
    assert out["flag_lr_floor"] is True
    assert math.isclose(out["active_dim_ratio"], 0.25)


def test_build_training_diagnostics_detects_overfit():
    history = {
        "recon_loss": [1.10, 0.95, 0.60, 0.50, 0.40],
        "val_recon_loss": [1.00, 0.82, 0.70, 0.74, 0.78],
        "lr": [3e-4, 3e-4, 3e-4, 3e-4, 3e-4],
    }

    out = _build_training_diagnostics(
        history_dict=history,
        cfg={"latent_dim": 4, "lr": 3e-4},
        max_epochs=10,
        active_dims=4,
        kl_mean_total=4.0,
        kl_mean_per_dim=1.0,
    )

    assert out["flag_overfit"] is True
    assert out["flag_posterior_collapse"] is False
    assert out["flag_undertrained"] is False


def test_apply_training_recommendations_prefers_seq_family_and_adds_actions():
    df = pd.DataFrame(
        [
            {
                "status": "ok",
                "tag": "seq_best",
                "arch_variant": "seq_bigru_residual",
                "score_v2": 1.0,
                "delta_psd_l2": 0.05,
                "delta_acf_l2": 0.01,
                "delta_evm_%": 0.1,
                "delta_snr_db": -0.1,
                "delta_mean_l2": 0.02,
                "delta_cov_fro": 0.08,
                "delta_skew_l2": 0.05,
                "delta_kurt_l2": 0.20,
                "flag_posterior_collapse": False,
                "flag_undertrained": False,
                "flag_overfit": False,
                "flag_unstable": False,
                "flag_lr_floor": False,
                "active_dim_ratio": 1.0,
                "late_val_slope": 0.0,
            },
            {
                "status": "ok",
                "tag": "delta_model",
                "arch_variant": "delta_residual",
                "score_v2": 1.5,
                "delta_psd_l2": 0.20,
                "delta_acf_l2": 0.05,
                "delta_evm_%": 0.2,
                "delta_snr_db": -0.2,
                "delta_mean_l2": 0.03,
                "delta_cov_fro": 0.10,
                "delta_skew_l2": 0.05,
                "delta_kurt_l2": 0.30,
                "flag_posterior_collapse": True,
                "flag_undertrained": True,
                "flag_overfit": False,
                "flag_unstable": False,
                "flag_lr_floor": True,
                "active_dim_ratio": 0.20,
                "late_val_slope": -2e-4,
            },
        ]
    )

    out = _apply_training_recommendations(df)
    delta_row = out.loc[out["tag"] == "delta_model"].iloc[0]

    assert delta_row["recommend_architecture"] == "prefer_seq_family"
    assert delta_row["recommend_beta_free_bits"] == "increase_free_bits_or_reduce_beta"
    assert delta_row["recommend_latent_dim"] == "reduce_latent_dim"
    assert delta_row["recommend_lr"] == "lower_initial_lr"
    assert delta_row["recommend_epochs_patience"] == "increase_epochs_or_patience"


def test_build_training_diagnostics_table_exposes_canonical_columns():
    df = pd.DataFrame(
        [
            {
                "rank": 1,
                "grid_id": 1,
                "group": "g",
                "tag": "model_a",
                "status": "ok",
                "arch_variant": "delta_residual",
                "score_v2": 1.23,
            }
        ]
    )

    out = _build_training_diagnostics_table(df)

    assert list(out.columns) == TRAINING_DIAGNOSTIC_COLUMNS
    assert out.iloc[0]["tag"] == "model_a"
    assert out.iloc[0]["recommend_lr"] == "keep"
    assert bool(out.iloc[0]["flag_unstable"]) is False


def test_mc_point_metric_means_penalizes_stochastic_draws_individually():
    x = np.ones((2, 2), dtype=np.float32)
    ys = np.stack(
        [
            np.array([[1.4, 1.0], [1.4, 1.0]], dtype=np.float32),
            np.array([[0.6, 1.0], [0.6, 1.0]], dtype=np.float32),
        ],
        axis=0,
    )

    evm_mc, snr_mc = _mc_point_metric_means(x, ys)

    assert np.isfinite(evm_mc)
    assert np.isfinite(snr_mc)


def test_candidate_ranking_key_mini_protocol_prioritizes_protocol_failures():
    better_protocol = {
        "status": "ok",
        "mini_n_fail": 1,
        "mini_n_g6_fail": 0,
        "mini_mean_abs_delta_coverage_95": 0.08,
        "mini_mean_delta_jb": 1.2,
        "mini_mean_delta_psd_l2": 0.2,
        "score_v2": 50.0,
        "score_abs_delta": 10.0,
    }
    better_score_only = {
        "status": "ok",
        "mini_n_fail": 2,
        "mini_n_g6_fail": 0,
        "mini_mean_abs_delta_coverage_95": 0.01,
        "mini_mean_delta_jb": 0.5,
        "mini_mean_delta_psd_l2": 0.1,
        "score_v2": 0.5,
        "score_abs_delta": 0.1,
    }

    assert _candidate_ranking_key(
        better_protocol, "mini_protocol_v1"
    ) < _candidate_ranking_key(better_score_only, "mini_protocol_v1")


def test_sort_results_by_ranking_uses_score_v2_only_as_final_tiebreak():
    df = pd.DataFrame(
        [
            {
                "tag": "worse_score",
                "status": "ok",
                "mini_n_fail": 1,
                "mini_n_g6_fail": 0,
                "mini_mean_abs_delta_coverage_95": 0.05,
                "mini_mean_delta_jb": 0.8,
                "mini_mean_delta_psd_l2": 0.1,
                "score_v2": 2.0,
                "score_abs_delta": 1.0,
            },
            {
                "tag": "better_score",
                "status": "ok",
                "mini_n_fail": 1,
                "mini_n_g6_fail": 0,
                "mini_mean_abs_delta_coverage_95": 0.05,
                "mini_mean_delta_jb": 0.8,
                "mini_mean_delta_psd_l2": 0.1,
                "score_v2": 1.0,
                "score_abs_delta": 1.0,
            },
        ]
    )

    out = _sort_results_by_ranking(df, "mini_protocol_v1")

    assert list(out["tag"]) == ["better_score", "worse_score"]
