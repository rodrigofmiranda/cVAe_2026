# -*- coding: utf-8 -*-

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.summary_plots import _pivot_for_heatmap, generate_all


@pytest.fixture()
def out_dir():
    d = Path(tempfile.mkdtemp(prefix="summary_plots_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_summary_df(n_dist: int = 3, n_curr: int = 4) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    rows = []
    for d in np.linspace(0.8, 1.5, n_dist):
        for c in np.linspace(100, 400, n_curr):
            rows.append({
                "regime_id": f"dist_{d:.1f}m__curr_{int(c)}mA",
                "dist_target_m": round(d, 2),
                "curr_target_mA": round(c, 0),
                "cvae_delta_evm_%": rng.uniform(-3.0, 3.0),
                "cvae_delta_snr_db": rng.uniform(-1.0, 1.0),
                "cvae_mean_rel_sigma": rng.uniform(0.001, 0.150),
                "cvae_cov_rel_var": rng.uniform(0.010, 0.250),
                "cvae_delta_mean_l2": rng.uniform(0.001, 0.050),
                "cvae_delta_cov_fro": rng.uniform(0.001, 0.050),
                "cvae_delta_skew_l2": rng.uniform(0.050, 0.500),
                "cvae_delta_kurt_l2": rng.uniform(0.050, 1.500),
                "delta_jb_stat_rel": rng.uniform(0.010, 0.500),
                "cvae_psd_l2": rng.uniform(0.050, 0.500),
                "cvae_delta_acf_l2": rng.uniform(0.050, 0.500),
                "cvae_rho_hetero_real": rng.uniform(0.200, 0.800),
                "cvae_rho_hetero_pred": rng.uniform(0.200, 0.800),
                "cvae_stat_jsd": rng.uniform(0.001, 0.200),
                "stat_mmd2": rng.uniform(0.0001, 0.0100),
                "stat_mmd_qval": rng.uniform(0.001, 0.200),
                "stat_energy": rng.uniform(0.0001, 0.0100),
                "stat_energy_qval": rng.uniform(0.001, 0.200),
                "stat_psd_dist": rng.uniform(0.050, 0.500),
            })
    return pd.DataFrame(rows)


def test_generate_all_creates_best_model_heatmap(out_dir):
    df = _make_summary_df()
    paths = generate_all(df, out_dir)
    assert len(paths) == 1
    assert paths[0].name == "heatmap_gate_metrics_by_regime.png"
    assert paths[0].exists() and paths[0].stat().st_size > 500


def test_pivot_for_heatmap_preserves_canonical_regime_grid_for_single_regime():
    df = pd.DataFrame([
        {
            "dist_target_m": 1.0,
            "curr_target_mA": 300.0,
            "cvae_delta_evm_%": 0.92,
        },
    ])

    piv = _pivot_for_heatmap(df, "cvae_delta_evm_%")

    assert piv is not None
    assert piv.shape == (3, 4)
    assert list(piv.index) == [0.8, 1.0, 1.5]
    assert list(piv.columns) == [100.0, 300.0, 500.0, 700.0]
    assert float(piv.loc[1.0, 300.0]) == pytest.approx(0.92)
    assert np.isnan(piv.loc[0.8, 100.0])
