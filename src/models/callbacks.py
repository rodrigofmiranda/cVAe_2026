# -*- coding: utf-8 -*-
"""
src/models/callbacks.py — Training callbacks for the cVAE.

Extracted from ``cvae_components.py`` and ``cvae_TRAIN_documented.py``
(refactor step 3).

Public API
----------
KLAnnealingCallback         Linear β ramp tied to CondPriorVAELoss.beta
EarlyStoppingAfterWarmup    Patience-based early stop with warmup guard
build_callbacks             Factory that assembles the standard callback list
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ReduceLROnPlateau,
)


# ======================================================================
# KL annealing (tightly coupled to CondPriorVAELoss.beta)
# ======================================================================
class KLAnnealingCallback(Callback):
    """Linearly ramp β from *beta_start* to *beta_end* over *annealing_epochs*."""

    def __init__(
        self,
        loss_layer,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        annealing_epochs: int = 50,
    ):
        super().__init__()
        self.loss_layer = loss_layer
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.annealing_epochs = int(annealing_epochs)

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.annealing_epochs:
            progress = epoch / max(self.annealing_epochs, 1)
            b = self.beta_start + (self.beta_end - self.beta_start) * progress
            self.loss_layer.beta.assign(b)
        else:
            self.loss_layer.beta.assign(self.beta_end)


# ======================================================================
# EarlyStopping with warmup guard
# ======================================================================
class EarlyStoppingAfterWarmup(Callback):
    """EarlyStopping that only starts counting *patience* after a warmup.

    This prevents premature stopping during KL/annealing instability in
    the first epochs.

    .. note::
        **C1 FIX** — ``best_weights`` are saved from epoch 1 (not only
        after warmup).  The warmup merely delays the *stop decision*; the
        best-val checkpoint is tracked continuously.
    """

    def __init__(
        self,
        monitor: str = "val_recon_loss",
        patience: int = 20,
        warmup_epochs: int = 0,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: int = 1,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = int(patience)
        self.warmup_epochs = int(warmup_epochs)
        self.min_delta = float(min_delta)
        self.restore_best_weights = bool(restore_best_weights)
        self.verbose = int(verbose)

        self.wait = 0
        self.best = np.inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor, None)
        if current is None:
            return

        # Checkpoint tracked ALWAYS (regardless of warmup)
        if current < (self.best - self.min_delta):
            self.best = current
            self.best_epoch = epoch + 1
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            return  # improvement — don't increment wait

        # During warmup: increment wait but do NOT stop
        if epoch + 1 <= self.warmup_epochs:
            self.wait += 1
            return

        # Post-warmup: normal stop logic
        self.wait += 1
        if self.wait >= self.patience:
            if self.verbose:
                print(
                    f"\nEarlyStoppingAfterWarmup: stopping at epoch {epoch + 1} "
                    f"(best {self.monitor}={self.best:.6f} @ epoch {self.best_epoch})"
                )
            if self.restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)
                if self.verbose:
                    print(f"  → weights restored to epoch {self.best_epoch}")
            self.model.stop_training = True


# ======================================================================
# Callback factory
# ======================================================================
def build_callbacks(
    training_config: Dict[str, Any],
    model_config: Dict[str, Any],
    kl_cb: KLAnnealingCallback,
) -> List[Callback]:
    """Assemble the standard training callback list.

    Parameters
    ----------
    training_config : dict
        Must contain ``patience``, ``reduce_lr_patience``.
        Optional: ``early_stop_warmup`` (default: 0).
    model_config : dict
        The grid-search model config (``kl_anneal_epochs`` used as warmup).
    kl_cb : KLAnnealingCallback
        Created by ``build_cvae`` / ``build_condprior_cvae``.

    Returns
    -------
    list[Callback]
    """
    warmup = int(model_config.get("kl_anneal_epochs", 80))

    return [
        EarlyStoppingAfterWarmup(
            monitor="val_recon_loss",
            patience=int(training_config["patience"]),
            warmup_epochs=warmup,
            min_delta=1e-5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=int(training_config["reduce_lr_patience"]),
            min_lr=1e-6,
            verbose=1,
        ),
        kl_cb,
    ]
