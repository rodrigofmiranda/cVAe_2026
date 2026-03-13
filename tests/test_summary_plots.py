# -*- coding: utf-8 -*-

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.summary_plots import (
    generate_all,
    plot_delta_evm_models,
    plot_evm_real_vs_models,
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
            evm_real = rng.uniform(20.0, 35.0)
            baseline = evm_real + rng.uniform(-2.0, 4.0)
            cvae = evm_real + rng.uniform(-3.0, 3.0)
            rows.append({
                "regime_id": f"dist_{d:.1f}m__curr_{int(c)}mA",
                "dist_target_m": round(d, 2),
                "curr_target_mA": round(c, 0),
                "evm_real_%": evm_real,
                "baseline_evm_pred_%": baseline,
                "baseline_delta_evm_%": baseline - evm_real,
                "cvae_evm_pred_%": cvae,
                "cvae_delta_evm_%": cvae - evm_real,
            })
    return pd.DataFrame(rows)


def test_plot_evm_real_vs_models_creates_png(out_dir):
    df = _make_summary_df()
    p = plot_evm_real_vs_models(df, out_dir)
    assert p is not None
    assert p.exists() and p.stat().st_size > 500


def test_plot_delta_evm_models_creates_png(out_dir):
    df = _make_summary_df()
    p = plot_delta_evm_models(df, out_dir)
    assert p is not None
    assert p.exists() and p.stat().st_size > 500


def test_generate_all_creates_two_files(out_dir):
    df = _make_summary_df()
    paths = generate_all(df, out_dir)
    assert len(paths) == 2
    names = {p.name for p in paths}
    assert "heatmap_evm_real_vs_models.png" in names
    assert "heatmap_delta_evm_models.png" in names

