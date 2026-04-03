import numpy as np
from src.models.cvae import build_cvae


def _point_cfg(arch_variant="concat", **overrides):
    cfg = {
        "layer_sizes": [16, 32],
        "latent_dim": 4,
        "beta": 0.01,
        "lr": 3e-4,
        "dropout": 0.0,
        "free_bits": 0.0,
        "kl_anneal_epochs": 3,
        "batch_size": 8,
        "activation": "leaky_relu",
        "arch_variant": arch_variant,
    }
    cfg.update(overrides)
    return cfg


def test_delta_residual_model_is_ready_for_dry_run_without_priming():
    vae, _ = build_cvae(_point_cfg("delta_residual"))
    assert vae.count_params() > 0


def test_support_geom3_model_is_ready_for_dry_run_when_scale_is_provided():
    vae, _ = build_cvae(
        _point_cfg(
            "delta_residual",
            support_feature_mode="geom3",
            support_feature_scale=1.0,
            support_weight_mode="edge_rinf_corner",
        )
    )
    assert vae.count_params() > 0
