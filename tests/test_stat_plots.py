# -*- coding: utf-8 -*-
"""
tests/test_stat_plots.py — Smoke tests for src.evaluation.stat_tests.plots.

Each test builds a small synthetic DataFrame, calls the corresponding
plot function, and asserts the output PNG exists and is non-empty.

Commit: refactor(etapaA3).
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.stat_tests.plots import (
    generate_all,
    plot_heatmap_mmd2,
    plot_heatmap_psd_dist,
    plot_heatmap_qval_mmd,
    plot_scatter_mmd2_vs_evm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def out_dir():
    d = Path(tempfile.mkdtemp(prefix="sf_plots_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_sf_df(n_dist: int = 3, n_curr: int = 4) -> pd.DataFrame:
    """Build a synthetic stat-fidelity DataFrame with grid coords."""
    rng = np.random.RandomState(0)
    rows = []
    for d in np.linspace(0.5, 2.0, n_dist):
        for c in np.linspace(10, 40, n_curr):
            rows.append({
                "regime_id": f"d{d:.1f}_c{c:.0f}",
                "regime_label": f"{d:.1f}m / {c:.0f}mA",
                "dist_m": round(d, 2),
                "curr_mA": round(c, 0),
                "mmd2": rng.uniform(0.0001, 0.05),
                "mmd_pval": rng.uniform(0.0, 1.0),
                "mmd_qval": rng.uniform(0.0, 1.0),
                "energy": rng.uniform(0.01, 1.0),
                "energy_pval": rng.uniform(0.0, 1.0),
                "energy_qval": rng.uniform(0.0, 1.0),
                "psd_dist": rng.uniform(0.01, 0.5),
                "psd_ci_low": rng.uniform(0.005, 0.04),
                "psd_ci_high": rng.uniform(0.05, 0.6),
                "baseline_evm_%": rng.uniform(1.0, 10.0),
                "n_samples": 5000,
                "n_perm": 200,
                "stat_mode": "quick",
            })
    return pd.DataFrame(rows)


def _make_summary_df(sf_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate a summary table with the columns generate_all needs."""
    return sf_df.rename(columns={
        "dist_m": "dist_target_m",
        "curr_mA": "curr_target_mA",
        "baseline_evm_%": "baseline_evm_pred_%",
    })[["regime_id", "dist_target_m", "curr_target_mA", "baseline_evm_pred_%"]]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHeatmapMMD2:
    def test_creates_png(self, out_dir):
        df = _make_sf_df()
        p = plot_heatmap_mmd2(df, out_dir)
        assert p is not None
        assert p.exists() and p.stat().st_size > 500

    def test_missing_columns_returns_none(self, out_dir):
        df = pd.DataFrame({"regime_id": ["a"]})
        assert plot_heatmap_mmd2(df, out_dir) is None


class TestHeatmapQval:
    def test_creates_png(self, out_dir):
        df = _make_sf_df()
        p = plot_heatmap_qval_mmd(df, out_dir)
        assert p is not None
        assert p.exists() and p.stat().st_size > 500

    def test_missing_qval_returns_none(self, out_dir):
        df = _make_sf_df().drop(columns=["mmd_qval"])
        assert plot_heatmap_qval_mmd(df, out_dir) is None


class TestHeatmapPSD:
    def test_creates_png(self, out_dir):
        df = _make_sf_df()
        p = plot_heatmap_psd_dist(df, out_dir)
        assert p is not None
        assert p.exists() and p.stat().st_size > 500


class TestScatterMMD2vsEVM:
    def test_creates_png(self, out_dir):
        df = _make_sf_df()
        p = plot_scatter_mmd2_vs_evm(df, out_dir, evm_col="baseline_evm_%")
        assert p is not None
        assert p.exists() and p.stat().st_size > 500

    def test_missing_evm_returns_none(self, out_dir):
        df = _make_sf_df().drop(columns=["baseline_evm_%"])
        assert plot_scatter_mmd2_vs_evm(df, out_dir) is None


class TestGenerateAll:
    def test_creates_all_four(self, out_dir):
        sf_df = _make_sf_df()
        summary_df = _make_summary_df(sf_df)
        # generate_all merges from summary → should still work
        # but sf_df already has columns, so pass summary for coverage
        paths = generate_all(sf_df, out_dir, df_summary=summary_df)
        assert len(paths) == 4
        names = {p.name for p in paths}
        assert "heatmap_mmd2.png" in names
        assert "heatmap_qval_mmd.png" in names
        assert "heatmap_psd_dist.png" in names
        assert "scatter_mmd2_vs_evm.png" in names

    def test_without_summary(self, out_dir):
        """When df_sf already has dist_m/curr_mA, no summary needed."""
        sf_df = _make_sf_df()
        paths = generate_all(sf_df, out_dir)
        # heatmaps need dist_m/curr_mA — they're in sf_df
        assert len(paths) >= 3  # scatter may also work if baseline_evm_% present

    def test_empty_df_no_crash(self, out_dir):
        df = pd.DataFrame()
        paths = generate_all(df, out_dir)
        assert paths == []

    def test_merge_from_summary(self, out_dir):
        """df_sf lacks grid cols; generate_all merges them from summary."""
        sf_df = _make_sf_df().drop(columns=["dist_m", "curr_mA", "baseline_evm_%"])
        summary_df = _make_summary_df(_make_sf_df())
        paths = generate_all(sf_df, out_dir, df_summary=summary_df)
        assert len(paths) == 4
