from pathlib import Path

import pandas as pd

from scripts.analysis.compare_protocol_finalists import _build_pivot, _collect_rows


def test_collect_rows_and_build_pivot(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "exp_test"
    csv_dir = run_dir / "eval" / "dist_1m__curr_300mA" / "tables"
    csv_dir.mkdir(parents=True)
    df = pd.DataFrame(
        [
            {
                "tag": "TAG_A",
                "rank": 1,
                "status": "ok",
                "best_epoch": 10,
                "train_time_s": 100.0,
                "score_v2": 1.2,
                "delta_evm_%": 0.1,
                "delta_snr_db": -0.2,
                "delta_cov_fro": 0.01,
                "delta_kurt_l2": 0.03,
                "var_real_delta": 0.1,
                "var_pred_delta": 0.11,
                "active_dims": 4,
                "kl_mean_total": 5.0,
            },
            {
                "tag": "TAG_B",
                "rank": 2,
                "status": "ok",
                "best_epoch": 11,
                "train_time_s": 101.0,
                "score_v2": 2.3,
                "delta_evm_%": 0.2,
                "delta_snr_db": -0.3,
                "delta_cov_fro": 0.02,
                "delta_kurt_l2": 0.04,
                "var_real_delta": 0.1,
                "var_pred_delta": 0.12,
                "active_dims": 4,
                "kl_mean_total": 6.0,
            },
        ]
    )
    df.to_csv(csv_dir / "gridsearch_results.csv", index=False)

    rows = _collect_rows(run_dir, ["TAG_A", "TAG_B"])
    assert len(rows) == 2
    assert set(rows["tag"]) == {"TAG_A", "TAG_B"}

    pivot = _build_pivot(rows, ["TAG_A", "TAG_B"])
    assert len(pivot) == 1
    assert pivot.iloc[0]["winner_by_score_v2"] == "TAG_A"
    assert pivot.iloc[0]["TAG_A::delta_cov_fro"] == 0.01
