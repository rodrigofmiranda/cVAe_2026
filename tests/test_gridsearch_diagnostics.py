import math

import numpy as np
import pandas as pd

from src.training.gridsearch import (
    TRAINING_DIAGNOSTIC_COLUMNS,
    _apply_training_recommendations,
    _build_training_diagnostics,
    _build_training_diagnostics_table,
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
