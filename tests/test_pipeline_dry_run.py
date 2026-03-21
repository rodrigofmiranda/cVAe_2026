import numpy as np
import pytest

from src.models.cvae import build_cvae
from src.training.pipeline import _prime_model_for_dry_run


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


def test_prime_model_for_dry_run_builds_delta_residual_adv_wrapper():
    vae, _ = build_cvae(_point_cfg("delta_residual_adv"))

    with pytest.raises(ValueError, match="isn't built"):
        vae.count_params()

    X = np.zeros((2, 2), dtype=np.float32)
    Dn = np.zeros((2, 1), dtype=np.float32)
    Cn = np.zeros((2, 1), dtype=np.float32)

    _prime_model_for_dry_run(
        vae,
        {"arch_variant": "delta_residual_adv"},
        X,
        Dn,
        Cn,
    )

    assert vae.built
    assert vae.count_params() > 0
