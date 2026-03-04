# -*- coding: utf-8 -*-
"""
src/models/cvae_components.py — **Backward-compatibility shim**.

All code has been split into dedicated modules (refactor step 3):

    sampling.py   → Sampling, reparameterize, sample_prior
    losses.py     → CondPriorVAELoss, reconstruction_loss, kl_divergence, …
    cvae.py       → build_mlp, build_decoder, build_cvae (+ alias build_condprior_cvae),
                    build_encoder, build_prior_net, create_inference_model_from_full
    callbacks.py  → KLAnnealingCallback, EarlyStoppingAfterWarmup, build_callbacks

This file re-exports the original public names so that existing
``from src.models.cvae_components import …`` statements keep working.
"""

# --- re-exports (keep every name that was previously importable) ------
from src.models.sampling import Sampling                           # noqa: F401
from src.models.losses import CondPriorVAELoss                     # noqa: F401
from src.models.callbacks import KLAnnealingCallback                # noqa: F401
from src.models.cvae import (                                       # noqa: F401
    _activation_layer,
    build_mlp,
    build_decoder,
    build_condprior_cvae,
    build_cvae,
    create_inference_model_from_full,
    build_encoder,
    build_prior_net,
)

