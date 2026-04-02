# -*- coding: utf-8 -*-

import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.summary_plots import (
    _gate_heatmap_style,
    _pivot_for_heatmap,
    generate_all,
    plot_eval_gate_difference_heatmaps,
    plot_eval_gate_supplementary_heatmaps,
    plot_eval_gate_threshold_heatmaps,
    plot_residual_signature_overview,
)


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


def test_eval_gate_difference_heatmap_creates_png(out_dir):
    df = _make_summary_df()
    df["delta_jb_stat_rel"] = np.random.RandomState(1).uniform(0.01, 1.0, len(df))
    path = plot_eval_gate_difference_heatmaps(df, out_dir)
    assert path is not None
    assert path.name == "heatmap_gate_differences_by_regime.png"
    assert path.exists() and path.stat().st_size > 500


def test_eval_gate_supplementary_heatmap_creates_png(out_dir):
    df = _make_summary_df()
    df["delta_coverage_95"] = np.random.RandomState(2).uniform(-0.3, 0.3, len(df))
    df["rho_hetero_abs_gap"] = np.random.RandomState(3).uniform(0.0, 0.2, len(df))
    path = plot_eval_gate_supplementary_heatmaps(df, out_dir)
    assert path is not None
    assert path.name == "heatmap_gate_supplementary_by_regime.png"
    assert path.exists() and path.stat().st_size > 500


def test_eval_gate_threshold_heatmap_creates_png(out_dir):
    df = _make_summary_df()
    path = plot_eval_gate_threshold_heatmaps(df, out_dir)
    assert path is not None
    assert path.name == "heatmap_gate_threshold_aware_by_regime.png"
    assert path.exists() and path.stat().st_size > 500


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


def test_gate_heatmap_style_uses_absolute_color_distance_for_signed_metrics():
    piv = pd.DataFrame(
        [[-2.0, 1.0]],
        index=[1.0],
        columns=[100.0, 300.0],
    )
    spec = {"mode": "signed", "threshold": 0.05}

    color_piv, annot_piv, vmin, vmax, center = _gate_heatmap_style(spec, piv)

    assert float(color_piv.loc[1.0, 100.0]) == pytest.approx(2.0)
    assert float(color_piv.loc[1.0, 300.0]) == pytest.approx(1.0)
    assert float(annot_piv.loc[1.0, 100.0]) == pytest.approx(-2.0)
    assert float(annot_piv.loc[1.0, 300.0]) == pytest.approx(1.0)
    assert vmin == pytest.approx(0.0)
    assert vmax == pytest.approx(2.0)
    assert center is None


def test_residual_signature_overview_skips_all_nan_inputs_without_warning(out_dir):
    df = pd.DataFrame(
        [
            {
                "dist_target_m": 0.8,
                "curr_target_mA": 100.0,
                "var_ratio_I": np.nan,
                "var_ratio_Q": np.nan,
                "delta_tail_p3sigma_I": np.nan,
                "delta_tail_p3sigma_Q": np.nan,
                "delta_wasserstein_I": np.nan,
                "delta_wasserstein_Q": np.nan,
                "delta_jb_stat_rel_I": np.nan,
                "delta_jb_stat_rel_Q": np.nan,
            }
        ]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        path = plot_residual_signature_overview(df, out_dir)

    assert path is None
    assert caught == []
