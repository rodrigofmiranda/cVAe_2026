import math

import pandas as pd

from src.evaluation.validation_summary import (
    STAT_FIDELITY_COLUMNS,
    SUMMARY_BY_REGIME_COLUMNS,
    build_stat_acceptance_summary,
    build_stat_fidelity_table,
    build_validation_summary_table,
)


def _full_result() -> dict:
    return {
        "_study": "within_regime",
        "regime_id": "dist_1p0m__curr_300mA",
        "regime_label": "1.0m / 300mA",
        "description": "pivot regime",
        "run_id": "run_0001",
        "run_dir": "/tmp/run_0001",
        "model_run_dir": "/tmp/global_model",
        "model_scope": "shared_global",
        "train_status": "completed",
        "eval_status": "completed",
        "best_grid_tag": "G1_core",
        "metrics": {
            "evm_real_%": 10.0,
            "evm_pred_%": 10.8,
            "delta_evm_%": 0.8,
            "snr_real_db": 10.0,
            "snr_pred_db": 10.6,
            "delta_snr_db": 0.6,
            "delta_mean_l2": 0.005,
            "delta_cov_fro": 0.008,
            "var_real_delta": 0.05,
            "var_pred_delta": 0.04,
            "delta_skew_l2": 0.10,
            "delta_kurt_l2": 0.30,
            "delta_psd_l2": 0.15,
            "delta_acf_l2": 0.09,
            "jb_p_min": 1e-6,
            "jb_log10p_min": -6.0,
            "reject_gaussian": True,
            "jb_real_p_min": 5e-7,
            "jb_real_log10p_min": -6.2,
            "jb_real_reject_gaussian": True,
        },
        "baseline": {
            "evm_pred_%": 22.0,
            "snr_pred_db": 6.0,
            "delta_evm_%": 12.0,
            "delta_snr_db": -3.5,
        },
        "baseline_dist": {
            "delta_mean_l2": 0.05,
            "delta_cov_fro": 0.06,
            "delta_skew_l2": 0.09,
            "delta_kurt_l2": 0.12,
            "psd_l2": 0.14,
            "delta_acf_l2": 0.16,
            "jb_p_min": 1e-5,
            "jb_log10p_min": -5.0,
            "reject_gaussian": True,
        },
        "cvae_dist": {
            "delta_mean_l2": 0.005,
            "delta_cov_fro": 0.008,
            "delta_skew_l2": 0.10,
            "delta_kurt_l2": 0.30,
            "psd_l2": 0.15,
            "delta_acf_l2": 0.09,
            "jb_p_min": 1e-6,
            "jb_log10p_min": -6.0,
            "reject_gaussian": True,
            "jb_real_p_min": 5e-7,
            "jb_real_log10p_min": -6.2,
            "jb_real_reject_gaussian": True,
        },
        "stat_fidelity": {
            "mmd2": 0.001,
            "mmd_pval": 0.40,
            "mmd_bandwidth": 0.25,
            "energy": 0.02,
            "energy_pval": 0.60,
            "psd_dist": 0.10,
            "psd_ci_low": 0.08,
            "psd_ci_high": 0.12,
            "n_samples": 4000,
            "n_perm": 200,
            "stat_mode": "quick",
        },
        "dist_metrics_source": "eval_reanalysis",
        "selected_experiments": ["exp_0", "exp_1"],
        "selection_criteria": {"distance_m": 1.0, "current_mA": 300.0},
    }


def test_validation_summary_schema_and_derived_fields():
    df = build_validation_summary_table([_full_result()])

    assert list(df.columns) == SUMMARY_BY_REGIME_COLUMNS
    row = df.iloc[0]

    assert math.isclose(row["var_ratio_pred_real"], 0.8, rel_tol=1e-9)
    assert math.isclose(row["delta_jb_log10p"], 0.2, rel_tol=1e-9)
    assert math.isclose(row["cvae_rel_evm_error"], 0.08, rel_tol=1e-9)
    assert math.isclose(row["cvae_rel_snr_error"], 0.06, rel_tol=1e-9)
    assert math.isclose(row["cvae_mean_rel_sigma"], 0.005 / math.sqrt(0.05), rel_tol=1e-9)
    assert math.isclose(row["cvae_cov_rel_var"], 0.008 / 0.05, rel_tol=1e-9)
    assert math.isclose(row["cvae_delta_acf_l2"], 0.09, rel_tol=1e-9)
    assert row["model_run_dir"] == "/tmp/global_model"
    assert row["model_scope"] == "shared_global"
    assert bool(row["better_than_baseline_mean"]) is True
    assert bool(row["better_than_baseline_cov"]) is True
    assert bool(row["better_than_baseline_skew"]) is True
    assert bool(row["better_than_baseline_kurt"]) is True
    assert bool(row["better_than_baseline_psd"]) is True
    assert bool(row["gate_g1"]) is True
    assert bool(row["gate_g2"]) is True
    assert bool(row["gate_g3"]) is True
    assert bool(row["gate_g4"]) is True
    assert bool(row["gate_g5"]) is True
    assert bool(row["gate_g6"]) is True
    assert row["validation_status"] == "pass"


def test_signal_fidelity_gates_require_10_percent_closeness_to_real():
    result = _full_result()
    result["metrics"]["evm_pred_%"] = 12.0
    result["metrics"]["delta_evm_%"] = 2.0
    result["metrics"]["snr_pred_db"] = 8.7
    result["metrics"]["delta_snr_db"] = -1.3

    df = build_validation_summary_table([result])
    row = df.iloc[0]

    assert bool(row["gate_g1"]) is False
    assert bool(row["gate_g2"]) is False
    assert row["validation_status"] == "fail"


def test_residual_scale_gate_uses_real_residual_power_not_baseline():
    result = _full_result()
    result["metrics"]["delta_mean_l2"] = 0.020
    result["metrics"]["delta_cov_fro"] = 0.009
    result["cvae_dist"]["delta_mean_l2"] = 0.020
    result["cvae_dist"]["delta_cov_fro"] = 0.009

    df = build_validation_summary_table([result])
    row = df.iloc[0]

    assert math.isclose(row["cvae_mean_rel_sigma"], 0.020 / math.sqrt(0.05), rel_tol=1e-9)
    assert bool(row["gate_g3"]) is False
    assert row["validation_status"] == "fail"


def test_validation_summary_keeps_schema_without_optional_families():
    result = _full_result()
    result["baseline"] = {}
    result["baseline_dist"] = {}
    result["stat_fidelity"] = {}
    df = build_validation_summary_table([result])

    assert list(df.columns) == SUMMARY_BY_REGIME_COLUMNS
    row = df.iloc[0]

    assert pd.isna(row["baseline_evm_pred_%"])
    assert pd.isna(row["stat_mmd2"])
    assert pd.isna(row["stat_mmd_qval"])
    assert bool(row["gate_g1"]) is True
    assert row["gate_g6"] is None
    assert row["validation_status"] == "partial"


def test_stat_projection_and_acceptance_derive_from_summary():
    result_ok = _full_result()
    result_fail = _full_result()
    result_fail["regime_id"] = "dist_1p5m__curr_900mA"
    result_fail["run_id"] = "run_0002"
    result_fail["metrics"]["delta_evm_%"] = 18.0
    result_fail["metrics"]["delta_kurt_l2"] = 0.40
    result_fail["cvae_dist"]["delta_kurt_l2"] = 0.40
    result_fail["stat_fidelity"]["mmd_pval"] = 0.001
    result_fail["stat_fidelity"]["energy_pval"] = 0.002

    df = build_validation_summary_table([result_ok, result_fail])
    df_sf = build_stat_fidelity_table(df)
    acceptance = build_stat_acceptance_summary(df)

    assert list(df_sf.columns) == STAT_FIDELITY_COLUMNS
    assert len(df_sf) == 2
    assert "mmd2_normalized" in df_sf.columns
    assert acceptance is not None
    assert acceptance["n_regimes_tested"] == 2
    assert acceptance["pass_mmd_qval"] == 1
    assert acceptance["pass_energy_qval"] == 1
    assert df.loc[df["regime_id"] == "dist_1p5m__curr_900mA", "validation_status"].iloc[0] == "fail"
