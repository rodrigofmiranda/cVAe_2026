# -*- coding: utf-8 -*-

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.summary_plots import generate_all


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
                "cvae_delta_mean_l2": rng.uniform(0.001, 0.050),
                "cvae_delta_cov_fro": rng.uniform(0.001, 0.050),
                "cvae_delta_skew_l2": rng.uniform(0.050, 0.500),
                "cvae_delta_kurt_l2": rng.uniform(0.050, 1.500),
                "cvae_psd_l2": rng.uniform(0.050, 0.500),
                "cvae_delta_acf_l2": rng.uniform(0.050, 0.500),
                "stat_mmd2": rng.uniform(0.0001, 0.0100),
                "stat_energy": rng.uniform(0.0001, 0.0100),
                "stat_psd_dist": rng.uniform(0.050, 0.500),
            })
    return pd.DataFrame(rows)


def test_generate_all_creates_best_model_heatmap(out_dir):
    df = _make_summary_df()
    paths = generate_all(df, out_dir)
    assert len(paths) == 1
    assert paths[0].name == "heatmap_vae_vs_real_metric_diffs.png"
    assert paths[0].exists() and paths[0].stat().st_size > 500
