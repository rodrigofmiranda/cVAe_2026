# -*- coding: utf-8 -*-
"""
Unit test: document the impact of MC sampling on ``var_pred_delta``.

For a non-trivial stochastic regime, ``mc_samples=1`` (single stochastic draw)
must differ from ``mc_samples=16`` (MC-averaged prediction) by at least 10%.
"""

import numpy as np

from src.evaluation.metrics import residual_distribution_metrics


def _predict_from_latent(mu: np.ndarray, sigma: np.ndarray, mc_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if mc_samples <= 1:
        return mu + rng.normal(scale=sigma, size=mu.shape)
    ys = []
    for _ in range(int(mc_samples)):
        ys.append(mu + rng.normal(scale=sigma, size=mu.shape))
    ys = np.stack(ys, axis=0)
    return ys.mean(axis=0)


def test_var_pred_delta_mc1_vs_mc16_differs_nontrivial_regime():
    rng = np.random.default_rng(2026)
    n = 4096

    # Synthetic non-trivial regime: heteroscedastic residual scale over I/Q space.
    x = rng.normal(loc=0.0, scale=0.7, size=(n, 2))
    sigma = 0.12 + 0.18 * np.abs(np.tanh(1.5 * x))
    mu = x + 0.10 * np.tanh(1.7 * x)

    # Real channel samples (reference residual distribution).
    y_real = mu + rng.normal(scale=0.22 + 0.05 * np.abs(x), size=mu.shape)

    y_pred_mc1 = _predict_from_latent(mu, sigma, mc_samples=1, seed=7)
    y_pred_mc16 = _predict_from_latent(mu, sigma, mc_samples=16, seed=7)

    m1 = residual_distribution_metrics(x, y_real, y_pred_mc1, psd_nfft=512)
    m16 = residual_distribution_metrics(x, y_real, y_pred_mc16, psd_nfft=512)

    v1 = float(m1["var_pred_delta"])
    v16 = float(m16["var_pred_delta"])
    rel_diff = abs(v1 - v16) / max(abs(v16), 1e-12)

    assert rel_diff >= 0.10, (
        f"Expected >=10% impact from MC sampling choice; got {rel_diff:.4f} "
        f"(mc1={v1:.6f}, mc16={v16:.6f})"
    )
