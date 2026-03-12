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
        "train_status": "completed",
        "eval_status": "completed",
        "best_grid_tag": "G1_core",
        "metrics": {
            "evm_real_%": 10.0,
            "evm_pred_%": 12.0,
            "delta_evm_%": 2.0,
            "snr_real_db": 9.5,
            "snr_pred_db": 8.9,
            "delta_snr_db": -0.6,
            "delta_mean_l2": 0.02,
            "delta_cov_fro": 0.03,
            "var_real_delta": 0.05,
            "var_pred_delta": 0.04,
            "delta_skew_l2": 0.06,
            "delta_kurt_l2": 0.08,
            "delta_psd_l2": 0.10,
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
            "jb_p_min": 1e-5,
            "jb_log10p_min": -5.0,
            "reject_gaussian": True,
        },
        "cvae_dist": {
            "delta_mean_l2": 0.02,
            "delta_cov_fro": 0.03,
            "delta_skew_l2": 0.06,
            "delta_kurt_l2": 0.08,
            "psd_l2": 0.10,
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
    assert bool(row["better_than_baseline_cov"]) is True
    assert bool(row["better_than_baseline_kurt"]) is True
    assert bool(row["better_than_baseline_psd"]) is True
    assert bool(row["gate_g1"]) is True
    assert bool(row["gate_g2"]) is True
    assert bool(row["gate_g3"]) is True
    assert bool(row["gate_g4"]) is True
    assert bool(row["gate_g5"]) is True
    assert bool(row["gate_g6"]) is True
    assert row["validation_status"] == "pass"


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
    assert row["gate_g1"] is None
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
