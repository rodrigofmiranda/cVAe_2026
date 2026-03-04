# -*- coding: utf-8 -*-
"""
src.models — cVAE model package.

Submodules
----------
sampling      Reparameterisation trick and latent-space sampling
losses        Heteroscedastic NLL, KL divergence, total ELBO loss
cvae          Encoder / Decoder / Prior / full cVAE assembly
callbacks     KLAnnealing, EarlyStoppingAfterWarmup, build_callbacks
cvae_components   Backward-compatibility shim (re-exports all of the above)
"""
