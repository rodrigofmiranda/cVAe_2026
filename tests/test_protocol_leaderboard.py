import math

from src.evaluation.validation_summary import (
    PROTOCOL_LEADERBOARD_COLUMNS,
    build_protocol_leaderboard,
    build_validation_summary_table,
)


def _result(tag: str, run_dir: str, regime_id: str, *, evm_delta: float, snr_delta: float, mmd_pval: float, energy_pval: float) -> dict:
    return {
        "_study": "within_regime",
        "regime_id": regime_id,
        "regime_label": regime_id,
        "description": regime_id,
        "run_id": f"run_{tag}_{regime_id}",
        "run_dir": f"/tmp/eval/{regime_id}",
        "model_run_dir": run_dir,
        "model_scope": "shared_global",
        "train_status": "completed",
        "eval_status": "completed",
        "best_grid_tag": tag,
        "metrics": {
            "evm_real_%": 10.0,
            "evm_pred_%": 10.0 + evm_delta,
            "delta_evm_%": evm_delta,
            "snr_real_db": 10.0,
            "snr_pred_db": 10.0 + snr_delta,
            "delta_snr_db": snr_delta,
            "delta_mean_l2": 0.005,
            "delta_cov_fro": 0.007,
            "var_real_delta": 0.05,
            "var_pred_delta": 0.045,
            "delta_skew_l2": 0.10,
            "delta_kurt_l2": 0.12,
            "delta_psd_l2": 0.15,
            "delta_acf_l2": 0.08,
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
            "delta_cov_fro": 0.007,
            "delta_skew_l2": 0.10,
            "delta_kurt_l2": 0.12,
            "psd_l2": 0.15,
            "delta_acf_l2": 0.08,
            "jb_p_min": 1e-6,
            "jb_log10p_min": -6.0,
            "reject_gaussian": True,
            "jb_real_p_min": 5e-7,
            "jb_real_log10p_min": -6.2,
            "jb_real_reject_gaussian": True,
        },
        "stat_fidelity": {
            "mmd2": 0.001,
            "mmd_pval": mmd_pval,
            "mmd_bandwidth": 0.25,
            "energy": 0.02,
            "energy_pval": energy_pval,
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


def test_protocol_leaderboard_ranks_candidates_from_canonical_summary():
    df_summary = build_validation_summary_table(
        [
            _result("TAG_A", "/tmp/model_a", "r1", evm_delta=0.2, snr_delta=0.2, mmd_pval=0.4, energy_pval=0.6),
            _result("TAG_A", "/tmp/model_a", "r2", evm_delta=0.2, snr_delta=0.2, mmd_pval=0.3, energy_pval=0.5),
            _result("TAG_B", "/tmp/model_b", "r1", evm_delta=2.5, snr_delta=-1.5, mmd_pval=0.001, energy_pval=0.002),
            _result("TAG_B", "/tmp/model_b", "r2", evm_delta=2.0, snr_delta=-1.2, mmd_pval=0.003, energy_pval=0.004),
        ]
    )

    df_lb = build_protocol_leaderboard(df_summary)

    assert list(df_lb.columns) == PROTOCOL_LEADERBOARD_COLUMNS
    assert list(df_lb["candidate_id"]) == ["TAG_A", "TAG_B"]

    winner = df_lb.iloc[0]
    loser = df_lb.iloc[1]

    assert winner["rank"] == 1
    assert winner["n_regimes"] == 2
    assert winner["n_pass"] == 2
    assert bool(winner["all_regimes_passed"]) is True
    assert math.isfinite(float(winner["protocol_score_v1"]))
    assert float(winner["protocol_score_v1"]) < float(loser["protocol_score_v1"])
    assert float(winner["gate_pass_ratio"]) > float(loser["gate_pass_ratio"])
