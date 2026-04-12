import math

import pandas as pd

from src.evaluation.validation_summary import (
    RESIDUAL_SIGNATURE_AMPLITUDE_COLUMNS,
    RESIDUAL_SIGNATURE_COLUMNS,
    RESIDUAL_SIGNATURE_SUPPORT_COLUMNS,
    STAT_FIDELITY_COLUMNS,
    SUMMARY_BY_REGIME_COLUMNS,
    build_residual_signature_amplitude_table,
    build_residual_signature_support_table,
    build_residual_signature_table,
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
            "evm_pred_%": 10.2,
            "delta_evm_%": 0.2,
            "snr_real_db": 10.0,
            "snr_pred_db": 10.2,
            "delta_snr_db": 0.2,
            "delta_mean_l2": 0.005,
            "delta_cov_fro": 0.007,
            "delta_mean_I": 0.003,
            "delta_mean_Q": -0.004,
            "delta_std_I": -0.002,
            "delta_std_Q": 0.001,
            "var_real_delta": 0.05,
            "var_pred_delta": 0.04,
            "delta_skew_l2": 0.10,
            "delta_kurt_l2": 0.12,
            "delta_skew_I": 0.06,
            "delta_skew_Q": -0.08,
            "delta_kurt_I": 0.20,
            "delta_kurt_Q": -0.10,
            "delta_wasserstein_I": 0.015,
            "delta_wasserstein_Q": 0.022,
            "delta_psd_l2": 0.15,
            "delta_acf_l2": 0.09,
            "jb_p_min": 1e-6,
            "jb_log10p_I": -5.8,
            "jb_log10p_Q": -6.0,
            "jb_log10p_min": -6.0,
            "reject_gaussian": True,
            "jb_real_p_min": 5e-7,
            "jb_real_log10p_I": -5.9,
            "jb_real_log10p_Q": -6.2,
            "jb_real_log10p_min": -6.2,
            "jb_real_reject_gaussian": True,
            "info_metrics_available": True,
            "info_metrics_status": "ok",
            "info_alphabet_size": 16,
            "info_bits_per_symbol": 4,
            "info_input_entropy_bits": 4.0,
            "info_avg_symbol_repeats": 32.0,
            "info_labeling_mode": "gray_rect_qam_uniform",
            "info_aux_channel_mode": "gaussian_shared_covariance",
            "mi_aux_real_bits": 3.8,
            "mi_aux_pred_bits": 3.7,
            "mi_aux_gap_bits": -0.1,
            "mi_aux_gap_rel": 0.1 / 3.8,
            "gmi_aux_real_bits": 3.7,
            "gmi_aux_pred_bits": 3.6,
            "gmi_aux_gap_bits": -0.1,
            "gmi_aux_gap_rel": 0.1 / 3.7,
            "ngmi_aux_real": 0.925,
            "ngmi_aux_pred": 0.900,
            "ngmi_aux_gap": -0.025,
            "air_aux_real_bits": 3.8,
            "air_aux_pred_bits": 3.7,
            "air_aux_gap_bits": -0.1,
            "air_aux_gap_rel": 0.1 / 3.8,
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
            "delta_skew_l2": 0.12,
            "delta_kurt_l2": 0.40,
            "psd_l2": 0.18,
            "delta_acf_l2": 0.16,
            "jb_p_min": 1e-5,
            "jb_log10p_min": -5.0,
            "reject_gaussian": True,
        },
        "cvae_dist": {
            "delta_mean_l2": 0.005,
            "delta_cov_fro": 0.007,
            "delta_mean_I": 0.003,
            "delta_mean_Q": -0.004,
            "delta_std_I": -0.002,
            "delta_std_Q": 0.001,
            "delta_skew_l2": 0.10,
            "delta_kurt_l2": 0.12,
            "delta_skew_I": 0.06,
            "delta_skew_Q": -0.08,
            "delta_kurt_I": 0.20,
            "delta_kurt_Q": -0.10,
            "delta_wasserstein_I": 0.015,
            "delta_wasserstein_Q": 0.022,
            "psd_l2": 0.15,
            "delta_acf_l2": 0.09,
            "jb_p_min": 1e-6,
            "jb_log10p_I": -5.8,
            "jb_log10p_Q": -6.0,
            "jb_log10p_min": -6.0,
            "reject_gaussian": True,
            "jb_real_log10p_I": -5.9,
            "jb_real_log10p_Q": -6.2,
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
    assert math.isclose(row["delta_mean_I"], 0.003, rel_tol=1e-9)
    assert math.isclose(row["delta_mean_Q"], -0.004, rel_tol=1e-9)
    assert math.isclose(row["delta_std_I"], -0.002, rel_tol=1e-9)
    assert math.isclose(row["delta_std_Q"], 0.001, rel_tol=1e-9)
    assert math.isclose(row["delta_skew_I"], 0.06, rel_tol=1e-9)
    assert math.isclose(row["delta_skew_Q"], -0.08, rel_tol=1e-9)
    assert math.isclose(row["delta_kurt_I"], 0.20, rel_tol=1e-9)
    assert math.isclose(row["delta_kurt_Q"], -0.10, rel_tol=1e-9)
    assert math.isclose(row["delta_wasserstein_I"], 0.015, rel_tol=1e-9)
    assert math.isclose(row["delta_wasserstein_Q"], 0.022, rel_tol=1e-9)
    assert math.isclose(row["delta_jb_log10p_I"], 0.1, rel_tol=1e-9)
    assert math.isclose(row["delta_jb_log10p_Q"], 0.2, rel_tol=1e-9)
    assert math.isclose(row["delta_jb_stat_rel_I"], 0.1 / 5.9, rel_tol=1e-9)
    assert math.isclose(row["delta_jb_stat_rel_Q"], 0.2 / 6.2, rel_tol=1e-9)
    assert math.isclose(row["delta_jb_log10p"], 0.2, rel_tol=1e-9)
    assert math.isclose(row["cvae_rel_evm_error"], 0.02, rel_tol=1e-9)
    assert math.isclose(row["cvae_rel_snr_error"], 0.02, rel_tol=1e-9)
    assert math.isclose(row["cvae_mean_rel_sigma"], 0.005 / math.sqrt(0.05), rel_tol=1e-9)
    assert math.isclose(row["cvae_cov_rel_var"], 0.007 / 0.05, rel_tol=1e-9)
    assert math.isclose(row["cvae_delta_acf_l2"], 0.09, rel_tol=1e-9)
    assert bool(row["info_metrics_available"]) is True
    assert row["info_metrics_status"] == "ok"
    assert math.isclose(row["mi_aux_real_bits"], 3.8, rel_tol=1e-9)
    assert math.isclose(row["gmi_aux_pred_bits"], 3.6, rel_tol=1e-9)
    assert math.isclose(row["ngmi_aux_real"], 0.925, rel_tol=1e-9)
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
    result["metrics"]["delta_mean_l2"] = 0.030
    result["metrics"]["delta_cov_fro"] = 0.009
    result["cvae_dist"]["delta_mean_l2"] = 0.030
    result["cvae_dist"]["delta_cov_fro"] = 0.009

    df = build_validation_summary_table([result])
    row = df.iloc[0]

    assert math.isclose(row["cvae_mean_rel_sigma"], 0.030 / math.sqrt(0.05), rel_tol=1e-9)
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
    assert bool(row["info_metrics_available"]) is True
    assert bool(row["gate_g1"]) is True
    assert row["gate_g6"] is None
    assert row["validation_status"] == "pass"
    assert row["validation_status_full"] == "partial"


def test_support_signature_table_flattens_rows_with_canonical_schema():
    result = _full_result()
    result["residual_signature_support_bins"] = [
        {
            "study": "within_regime",
            "regime_id": result["regime_id"],
            "regime_label": result["regime_label"],
            "run_id": result["run_id"],
            "run_dir": result["run_dir"],
            "model_run_dir": result["model_run_dir"],
            "best_grid_tag": result["best_grid_tag"],
            "dist_target_m": 1.0,
            "curr_target_mA": 300.0,
            "support_axis": "r_inf_norm",
            "support_bin_index": 0,
            "support_bin_label": "q0-25",
            "support_lo": 0.0,
            "support_hi": 0.25,
            "support_region": "",
            "n_samples_real": 1024,
            "n_samples_pred": 2048,
            "stat_mode": "quick",
            "std_real_delta_I": 0.1,
            "std_real_delta_Q": 0.2,
            "std_pred_delta_I": 0.11,
            "std_pred_delta_Q": 0.21,
            "delta_wasserstein_I": 0.01,
            "delta_wasserstein_Q": 0.02,
            "delta_jb_stat_rel_I": 0.03,
            "delta_jb_stat_rel_Q": 0.04,
            "stat_mmd_pval": 0.5,
            "stat_mmd_qval": 0.6,
            "stat_energy_pval": 0.7,
            "stat_energy_qval": 0.8,
        }
    ]

    df = build_residual_signature_support_table([result])

    assert list(df.columns) == RESIDUAL_SIGNATURE_SUPPORT_COLUMNS
    row = df.iloc[0]
    assert row["support_axis"] == "r_inf_norm"
    assert row["support_bin_label"] == "q0-25"
    assert math.isclose(row["stat_mmd_qval"], 0.6, rel_tol=1e-9)


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


def test_residual_signature_tables_are_projected_without_polluting_summary():
    result = _full_result()
    result["metrics"].update(
        {
            "var_ratio_I": 0.82,
            "var_ratio_Q": 1.10,
            "iqr_real_I": 0.15,
            "iqr_real_Q": 0.16,
            "iqr_pred_I": 0.13,
            "iqr_pred_Q": 0.17,
            "delta_iqr_I": -0.02,
            "delta_iqr_Q": 0.01,
            "q05_real_I": -0.10,
            "q05_pred_I": -0.08,
            "q95_real_Q": 0.11,
            "q95_pred_Q": 0.14,
            "delta_q05_I": 0.02,
            "tail_p3sigma_real_I": 0.02,
            "tail_p3sigma_pred_I": 0.01,
            "delta_tail_p3sigma_I": -0.01,
            "radial_wasserstein": 0.012,
            "corr_iq_real": 0.11,
            "corr_iq_pred": 0.09,
            "delta_corr_IQ": -0.02,
            "ellipse_axis_ratio_real": 1.5,
            "ellipse_axis_ratio_pred": 1.3,
            "delta_ellipse_axis_ratio": -0.2,
            "coverage_50": 0.52,
            "coverage_80": 0.79,
            "coverage_95": 0.93,
            "delta_coverage_50": 0.02,
            "delta_coverage_80": -0.01,
            "delta_coverage_95": -0.02,
        }
    )
    result["residual_signature_bins"] = [
        {
            "study": "within_regime",
            "regime_id": result["regime_id"],
            "regime_label": result["regime_label"],
            "run_id": result["run_id"],
            "run_dir": result["run_dir"],
            "model_run_dir": result["model_run_dir"],
            "best_grid_tag": result["best_grid_tag"],
            "dist_target_m": 1.0,
            "curr_target_mA": 300.0,
            "amplitude_bin_index": 0,
            "amplitude_bin_label": "q0-25",
            "amplitude_lo": 0.0,
            "amplitude_hi": 0.2,
            "n_samples_real": 1024,
            "n_samples_pred": 1024,
            "stat_mode": "quick",
            "std_real_delta_I": 0.1,
            "std_real_delta_Q": 0.2,
            "std_pred_delta_I": 0.09,
            "std_pred_delta_Q": 0.18,
            "delta_wasserstein_I": 0.01,
            "delta_wasserstein_Q": 0.02,
            "delta_jb_stat_rel_I": 0.10,
            "delta_jb_stat_rel_Q": 0.11,
            "stat_mmd_pval": 0.30,
            "stat_mmd_qval": 0.30,
            "stat_energy_pval": 0.40,
            "stat_energy_qval": 0.40,
        }
    ]

    df_summary = build_validation_summary_table([result])
    df_sig = build_residual_signature_table([result], df_summary)
    df_amp = build_residual_signature_amplitude_table([result])

    assert list(df_summary.columns) == SUMMARY_BY_REGIME_COLUMNS
    assert list(df_sig.columns) == RESIDUAL_SIGNATURE_COLUMNS
    assert list(df_amp.columns) == RESIDUAL_SIGNATURE_AMPLITUDE_COLUMNS
    assert math.isclose(df_sig.iloc[0]["var_ratio_I"], 0.82, rel_tol=1e-9)
    assert math.isclose(df_sig.iloc[0]["coverage_95"], 0.93, rel_tol=1e-9)
    assert df_amp.iloc[0]["amplitude_bin_label"] == "q0-25"
