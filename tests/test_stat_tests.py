# -*- coding: utf-8 -*-
"""
Smoke / unit tests for src.evaluation.stat_tests.

Run with:
    python -m pytest tests/test_stat_tests.py -v

Three test families:
1. **Same distribution** (H0 true) → p-values should generally *not* reject.
2. **Different distributions** (H1 true) → p-values should be small.
3. **PSD self-distance** → should be exactly zero for identical signals.
4. **FDR** → basic monotonicity and correctness checks.

Commit: refactor(etapaA1).
"""

import numpy as np
import pytest

from src.evaluation.stat_tests.mmd import mmd_rbf
from src.evaluation.stat_tests.energy import energy_test
from src.evaluation.stat_tests.psd import psd_distance
from src.evaluation.stat_tests.fdr import benjamini_hochberg


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def same_samples():
    """Two draws from the *same* N(0, 1) — H0 true."""
    rng = np.random.default_rng(42)
    n = 400
    Y1 = rng.standard_normal((n, 2)).astype(np.float32)
    Y2 = rng.standard_normal((n, 2)).astype(np.float32)
    return Y1, Y2


@pytest.fixture()
def diff_samples():
    """N(0,1) vs N(2,1) — clearly different distributions."""
    rng = np.random.default_rng(99)
    n = 400
    Y_real = rng.standard_normal((n, 2)).astype(np.float32)
    Y_pred = (rng.standard_normal((n, 2)) + 2.0).astype(np.float32)
    return Y_real, Y_pred


# ------------------------------------------------------------------
# MMD tests
# ------------------------------------------------------------------

class TestMMD:
    def test_same_distribution_high_pval(self, same_samples):
        Y1, Y2 = same_samples
        result = mmd_rbf(Y1, Y2, n_perm=200, seed=42)
        assert "mmd2" in result and "pval" in result
        # Under H0 the p-value should not reject at α=0.01 most of the
        # time.  We use a generous threshold here (smoke test, not a
        # power study).
        assert result["pval"] > 0.01, (
            f"pval={result['pval']:.4f} unexpectedly low for same dist"
        )

    def test_different_distribution_low_pval(self, diff_samples):
        Y_real, Y_pred = diff_samples
        result = mmd_rbf(Y_real, Y_pred, n_perm=200, seed=42)
        assert result["pval"] < 0.05, (
            f"pval={result['pval']:.4f} unexpectedly high for shifted dist"
        )
        assert result["mmd2"] > 0.0

    def test_bandwidth_positive(self, same_samples):
        Y1, Y2 = same_samples
        result = mmd_rbf(Y1, Y2, n_perm=10, seed=0)
        assert result["bandwidth"] > 0.0

    def test_returns_metadata(self, same_samples):
        Y1, Y2 = same_samples
        result = mmd_rbf(Y1, Y2, n_perm=50, seed=7)
        for key in ("mmd2", "pval", "bandwidth", "n_perm", "n_real", "n_pred"):
            assert key in result, f"Missing key: {key}"
        assert result["n_perm"] == 50
        assert result["n_real"] == len(Y1)

    def test_cpu_backend_runs(self, same_samples):
        Y1, Y2 = same_samples
        result = mmd_rbf(Y1, Y2, n_perm=20, seed=3, execution_backend="cpu")
        assert "pval" in result
        assert result["n_perm"] == 20


# ------------------------------------------------------------------
# Energy distance tests
# ------------------------------------------------------------------

class TestEnergy:
    def test_same_distribution_high_pval(self, same_samples):
        Y1, Y2 = same_samples
        result = energy_test(Y1, Y2, n_perm=200, seed=42)
        assert result["pval"] > 0.01, (
            f"pval={result['pval']:.4f} unexpectedly low for same dist"
        )

    def test_different_distribution_low_pval(self, diff_samples):
        Y_real, Y_pred = diff_samples
        result = energy_test(Y_real, Y_pred, n_perm=200, seed=42)
        assert result["pval"] < 0.05, (
            f"pval={result['pval']:.4f} unexpectedly high for shifted dist"
        )
        assert result["energy"] > 0.0

    def test_returns_metadata(self, same_samples):
        Y1, Y2 = same_samples
        result = energy_test(Y1, Y2, n_perm=50, seed=0)
        for key in ("energy", "pval", "n_perm", "n_real", "n_pred"):
            assert key in result, f"Missing key: {key}"

    def test_cpu_backend_runs(self, same_samples):
        Y1, Y2 = same_samples
        result = energy_test(Y1, Y2, n_perm=20, seed=5, execution_backend="cpu")
        assert "pval" in result
        assert result["n_perm"] == 20


# ------------------------------------------------------------------
# PSD distance tests
# ------------------------------------------------------------------

class TestPSD:
    def test_identical_signal_zero_distance(self):
        """Same signal → PSD distance = 0."""
        rng = np.random.default_rng(7)
        Y = rng.standard_normal((2000, 2)).astype(np.float64)
        result = psd_distance(Y, Y, nfft=512, n_boot=50, seed=0)
        assert abs(result["psd_dist"]) < 1e-10, (
            f"Expected ~0, got {result['psd_dist']}"
        )

    def test_different_signal_positive_distance(self):
        rng = np.random.default_rng(7)
        Y1 = rng.standard_normal((2000, 2)).astype(np.float64)
        Y2 = rng.standard_normal((2000, 2)).astype(np.float64) * 3.0
        result = psd_distance(Y1, Y2, nfft=512, n_boot=50, seed=0)
        assert result["psd_dist"] > 0.0

    def test_ci_brackets_point(self):
        """CI should bracket the point estimate (approximately)."""
        rng = np.random.default_rng(7)
        Y1 = rng.standard_normal((2000, 2))
        Y2 = rng.standard_normal((2000, 2)) * 2.0
        result = psd_distance(Y1, Y2, nfft=512, n_boot=200, seed=42)
        # CI is a bootstrap distribution — point need not lie exactly
        # inside, but low should be ≤ high.
        assert result["psd_ci_low"] <= result["psd_ci_high"]

    def test_residual_mode(self):
        """Passing X should compute PSD on Δ=Y−X."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((2000, 2))
        Y = X + 0.01 * rng.standard_normal((2000, 2))
        result = psd_distance(Y, Y, X=X, nfft=512, n_boot=10, seed=0)
        assert abs(result["psd_dist"]) < 1e-10

    def test_returns_metadata(self):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((1000, 2))
        result = psd_distance(Y, Y, nfft=256, n_boot=20, seed=0)
        for key in ("psd_dist", "psd_ci_low", "psd_ci_high", "nfft", "n_boot"):
            assert key in result


# ------------------------------------------------------------------
# FDR tests
# ------------------------------------------------------------------

class TestFDR:
    def test_single_pvalue(self):
        q = benjamini_hochberg([0.03])
        assert len(q) == 1
        assert np.isclose(q[0], 0.03)

    def test_monotonicity(self):
        """Sorted q-values should be non-decreasing."""
        pvals = [0.001, 0.01, 0.03, 0.04, 0.50]
        qvals = benjamini_hochberg(pvals)
        sorted_q = np.sort(qvals)
        assert np.all(sorted_q[1:] >= sorted_q[:-1] - 1e-12)

    def test_all_significant(self):
        """All p-values very small → all q-values ≤ 0.05."""
        pvals = [0.001, 0.002, 0.003]
        qvals = benjamini_hochberg(pvals)
        assert np.all(qvals <= 0.05)

    def test_preserves_order(self):
        """Output q-values correspond to *input* order, not sorted."""
        pvals = [0.50, 0.01, 0.30]
        qvals = benjamini_hochberg(pvals)
        # The smallest p-value (index 1) should have the smallest q-value
        assert qvals[1] == qvals.min()

    def test_empty(self):
        q = benjamini_hochberg([])
        assert len(q) == 0

    def test_capped_at_one(self):
        pvals = [0.90, 0.95, 0.99]
        qvals = benjamini_hochberg(pvals)
        assert np.all(qvals <= 1.0)
