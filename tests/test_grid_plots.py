import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.training.grid_plots import (
    generate_gridsearch_overview_plots,
    save_candidate_plot_bundle,
    save_legacy_champion_plots,
)


@pytest.fixture()
def out_dir():
    d = Path(tempfile.mkdtemp(prefix="grid_plots_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_save_candidate_plot_bundle_creates_expected_files(out_dir):
    n = 512
    rng = np.random.default_rng(0)
    Xv = rng.normal(size=(n, 2)).astype(np.float32)
    Yv = Xv + 0.1 * rng.normal(size=(n, 2)).astype(np.float32)
    Yp = Xv + 0.12 * rng.normal(size=(n, 2)).astype(np.float32)
    std_mu_p = np.abs(rng.normal(size=8)).astype(np.float32)
    kl_dim_mean = np.abs(rng.normal(size=8)).astype(np.float32)
    history = {
        "loss": [3.0, 2.0, 1.5],
        "val_loss": [3.2, 2.2, 1.7],
        "recon_loss": [2.5, 1.8, 1.4],
        "kl_loss": [0.5, 0.4, 0.3],
    }

    paths = save_candidate_plot_bundle(
        plots_dir=out_dir,
        Xv=Xv,
        Yv=Yv,
        Yp=Yp,
        std_mu_p=std_mu_p,
        kl_dim_mean=kl_dim_mean,
        history_dict=history,
        summary_lines=["grid_id: 1", "score_v2: 1.234"],
        psd_nfft=256,
        title_prefix="GRID 1 | test",
    )

    assert len(paths) == 9
    expected = {
        "overlay_constellation.png",
        "overlay_residual_delta.png",
        "density_y_real.png",
        "density_y_pred.png",
        "psd_residual_delta.png",
        "latent_activity_std_mu_p.png",
        "latent_kl_qp_per_dim.png",
        "training_history.png",
        "summary_report.png",
    }
    assert expected == {p.name for p in paths}
    assert all(p.exists() and p.stat().st_size > 500 for p in paths)


def test_generate_gridsearch_overview_plots_creates_all(out_dir):
    df = pd.DataFrame(
        [
            {
                "rank": 1,
                "tag": "g1",
                "score_v2": 1.2,
                "delta_cov_fro": 0.02,
                "delta_kurt_l2": 0.03,
                "delta_evm_%": 1.0,
                "delta_snr_db": -0.4,
                "active_dims": 6,
            },
            {
                "rank": 2,
                "tag": "g2",
                "score_v2": 1.8,
                "delta_cov_fro": 0.04,
                "delta_kurt_l2": 0.09,
                "delta_evm_%": 2.2,
                "delta_snr_db": -0.8,
                "active_dims": 4,
            },
            {
                "rank": 3,
                "tag": "g3",
                "score_v2": 2.4,
                "delta_cov_fro": 0.05,
                "delta_kurt_l2": 0.11,
                "delta_evm_%": 3.0,
                "delta_snr_db": -1.2,
                "active_dims": 3,
            },
        ]
    )

    paths = generate_gridsearch_overview_plots(df, out_dir)

    assert len(paths) == 4
    assert {
        "top_models_score_v2.png",
        "scatter_cov_vs_kurt.png",
        "scatter_delta_evm_vs_delta_snr.png",
        "scatter_score_vs_active_dims.png",
    } == {p.name for p in paths}
    assert all(p.exists() and p.stat().st_size > 500 for p in paths)


def test_save_legacy_champion_plots_creates_old_style_files(out_dir):
    n = 1024
    rng = np.random.default_rng(1)
    Xv = rng.normal(size=(n, 2)).astype(np.float32)
    Yv = Xv + 0.08 * rng.normal(size=(n, 2)).astype(np.float32)
    Yp = Xv + 0.09 * rng.normal(size=(n, 2)).astype(np.float32)
    mu_p = 0.05 * rng.normal(size=(n, 4)).astype(np.float32)
    std_mu_p = np.abs(rng.normal(size=4)).astype(np.float32)
    kl_dim_mean = np.abs(rng.normal(size=4)).astype(np.float32)

    paths = save_legacy_champion_plots(
        plots_dir=out_dir,
        Xv=Xv,
        Yv=Yv,
        Yp=Yp,
        mu_p=mu_p,
        std_mu_p=std_mu_p,
        kl_dim_mean=kl_dim_mean,
        summary_lines=["Champion summary", "score_v2: 1.23"],
        model_label="Champion",
    )

    assert len(paths) == 4
    assert {
        "analise_completa_vae.png",
        "comparacao_metricas_principais.png",
        "constellation_overlay.png",
        "radar_comparativo.png",
    } == {p.name for p in paths}
    assert all(p.exists() and p.stat().st_size > 500 for p in paths)
