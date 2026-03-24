# -*- coding: utf-8 -*-
"""
src/models/callbacks.py — Training callbacks for the cVAE.

Shared callbacks for the canonical cVAE training pipeline.

Public API
----------
KLAnnealingCallback         Linear β ramp tied to CondPriorVAELoss.beta
EarlyStoppingAfterWarmup    Patience-based early stop with warmup guard
build_callbacks             Factory that assembles the standard callback list
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
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


class RegimeDiagnosticsCallback(Callback):
    """Periodic per-regime validation diagnostics during training."""

    def __init__(
        self,
        *,
        logs_dir: Path,
        x_val_input: np.ndarray,
        x_val_center: np.ndarray,
        y_val: np.ndarray,
        d_val_norm: np.ndarray,
        c_val_norm: np.ndarray,
        d_val_raw: np.ndarray,
        c_val_raw: np.ndarray,
        enabled: bool = True,
        every_n_epochs: int = 10,
        mc_samples: int = 4,
        max_samples_per_regime: int = 4096,
        amplitude_bins: int = 4,
        focus_only_0p8m: bool = False,
        stat_n_perm: int = 100,
        stat_seed: int = 42,
    ):
        super().__init__()
        self.logs_dir = Path(logs_dir)
        self.train_logs_dir = self.logs_dir / "train"
        self.history_path = self.train_logs_dir / "regime_diagnostics_history.csv"
        self.latest_path = self.train_logs_dir / "regime_diagnostics_latest.json"
        self.enabled = bool(enabled)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.mc_samples = max(1, int(mc_samples))
        self.max_samples_per_regime = max(128, int(max_samples_per_regime))
        self.amplitude_bins = max(2, int(amplitude_bins))
        self.focus_only_0p8m = bool(focus_only_0p8m)
        self.stat_n_perm = max(32, int(stat_n_perm))
        self.stat_seed = int(stat_seed)
        self.x_val_input = np.asarray(x_val_input)
        self.x_val_center = np.asarray(x_val_center)
        self.y_val = np.asarray(y_val)
        self.d_val_norm = np.asarray(d_val_norm).reshape(-1, 1)
        self.c_val_norm = np.asarray(c_val_norm).reshape(-1, 1)
        self.d_val_raw = np.asarray(d_val_raw).reshape(-1, 1)
        self.c_val_raw = np.asarray(c_val_raw).reshape(-1, 1)
        self._history: List[Dict[str, Any]] = []
        self.train_logs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _format_regime_id(distance_m: float, current_mA: float) -> str:
        d = ("%g" % float(distance_m)).replace(".", "p")
        c = int(round(float(current_mA)))
        return f"dist_{d}m__curr_{c}mA"

    @staticmethod
    def _validation_status_partial(*gates: Any) -> str:
        vals = [g for g in gates if g is not None]
        if not vals:
            return "partial"
        if any(v is False for v in vals):
            return "fail"
        if all(v is True for v in vals):
            return "pass"
        return "partial"

    def _gate_g3(self, distm: Dict[str, float]) -> Optional[bool]:
        from src.evaluation.validation_summary import TWIN_GATE_THRESHOLDS

        var_real = float(distm.get("var_real_delta", float("nan")))
        if not np.isfinite(var_real) or var_real <= 0.0:
            return None
        mean_rel_sigma = float(distm.get("delta_mean_l2", float("nan"))) / np.sqrt(var_real)
        cov_rel_var = float(distm.get("delta_cov_fro", float("nan"))) / var_real
        if not (np.isfinite(mean_rel_sigma) and np.isfinite(cov_rel_var)):
            return None
        return bool(
            mean_rel_sigma < float(TWIN_GATE_THRESHOLDS["mean_rel_sigma"])
            and cov_rel_var < float(TWIN_GATE_THRESHOLDS["cov_rel_var"])
        )

    def _gate_g5(self, distm: Dict[str, float]) -> Optional[bool]:
        from src.evaluation.validation_summary import TWIN_GATE_THRESHOLDS

        vals = [
            float(distm.get("delta_skew_l2", float("nan"))),
            float(distm.get("delta_kurt_l2", float("nan"))),
            float(distm.get("delta_jb_stat_rel", float("nan"))),
        ]
        if not all(np.isfinite(v) for v in vals):
            return None
        return bool(
            vals[0] < float(TWIN_GATE_THRESHOLDS["delta_skew_l2"])
            and vals[1] < float(TWIN_GATE_THRESHOLDS["delta_kurt_l2"])
            and vals[2] < float(TWIN_GATE_THRESHOLDS["delta_jb_stat_rel"])
        )

    def _gate_g6(self, mmd_q: float, energy_q: float) -> Optional[bool]:
        from src.evaluation.validation_summary import TWIN_GATE_THRESHOLDS

        if not (np.isfinite(mmd_q) and np.isfinite(energy_q)):
            return None
        thr = float(TWIN_GATE_THRESHOLDS["stat_qval"])
        return bool(mmd_q > thr and energy_q > thr)

    def on_epoch_end(self, epoch, logs=None):
        if not self.enabled or ((epoch + 1) % self.every_n_epochs) != 0:
            return

        from src.evaluation.metrics import residual_distribution_metrics
        from src.evaluation.stat_tests import benjamini_hochberg, energy_test, mmd_rbf
        from src.models.cvae import create_inference_model_from_full

        logs = logs or {}
        rng = np.random.default_rng(self.stat_seed + epoch + 1)

        inf_det = create_inference_model_from_full(self.model, deterministic=True)
        y_det = inf_det.predict(
            [self.x_val_input, self.d_val_norm, self.c_val_norm],
            batch_size=4096,
            verbose=0,
        )

        y_samples = None
        if self.mc_samples > 1:
            inf_sto = create_inference_model_from_full(self.model, deterministic=False)
            ys = []
            for _ in range(self.mc_samples):
                ys.append(
                    inf_sto.predict(
                        [self.x_val_input, self.d_val_norm, self.c_val_norm],
                        batch_size=4096,
                        verbose=0,
                    )
                )
            y_samples = np.stack(ys, axis=0)
            y_dist = y_samples.reshape((-1, y_samples.shape[-1]))
            x_dist = np.tile(self.x_val_center, (self.mc_samples, 1))
            y_real_dist = np.tile(self.y_val, (self.mc_samples, 1))
        else:
            y_dist = y_det
            x_dist = self.x_val_center
            y_real_dist = self.y_val

        regimes = sorted(
            {
                (float(d), float(c))
                for d, c in zip(self.d_val_raw.ravel().tolist(), self.c_val_raw.ravel().tolist())
            }
        )
        if self.focus_only_0p8m:
            regimes = [rc for rc in regimes if abs(rc[0] - 0.8) <= 1e-9]

        rows: List[Dict[str, Any]] = []
        for distance_m, current_mA in regimes:
            mask = (
                np.isclose(self.d_val_raw.ravel(), distance_m)
                & np.isclose(self.c_val_raw.ravel(), current_mA)
            )
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                continue
            if len(idx) > self.max_samples_per_regime:
                idx = np.sort(rng.choice(idx, size=self.max_samples_per_regime, replace=False))

            x_center = self.x_val_center[idx]
            y_true = self.y_val[idx]
            y_det_reg = y_det[idx]
            y_stack_reg = None if y_samples is None else y_samples[:, idx, :]
            y_dist_reg = y_det_reg if y_stack_reg is None else y_stack_reg.reshape((-1, y_stack_reg.shape[-1]))
            x_dist_reg = x_center if y_stack_reg is None else np.tile(x_center, (self.mc_samples, 1))
            y_real_reg = y_true if y_stack_reg is None else np.tile(y_true, (self.mc_samples, 1))

            distm = residual_distribution_metrics(
                x_dist_reg,
                y_real_reg,
                y_dist_reg,
                psd_nfft=512,
                Y_samples=y_stack_reg,
                coverage_target=y_true,
            )

            rr = y_true - x_center
            rp = y_dist_reg - x_dist_reg
            n_cmp = min(len(rr), len(rp))
            if n_cmp < len(rr):
                rr = rr[rng.choice(len(rr), n_cmp, replace=False)]
            if n_cmp < len(rp):
                rp = rp[rng.choice(len(rp), n_cmp, replace=False)]
            sf_mmd = mmd_rbf(rr, rp, n_perm=self.stat_n_perm, seed=self.stat_seed + epoch + int(current_mA))
            sf_energy = energy_test(rr, rp, n_perm=self.stat_n_perm, seed=self.stat_seed + epoch + int(current_mA) + 1)

            rows.append(
                {
                    "epoch": int(epoch + 1),
                    "regime_id": self._format_regime_id(distance_m, current_mA),
                    "dist_target_m": float(distance_m),
                    "curr_target_mA": float(current_mA),
                    "n_samples": int(len(idx)),
                    "val_recon_loss_proxy": float(np.mean(np.sum((y_det_reg - y_true) ** 2, axis=1))),
                    "delta_wasserstein_I": float(distm.get("delta_wasserstein_I", float("nan"))),
                    "delta_wasserstein_Q": float(distm.get("delta_wasserstein_Q", float("nan"))),
                    "delta_jb_stat_rel_I": float(distm.get("delta_jb_stat_rel_I", float("nan"))),
                    "delta_jb_stat_rel_Q": float(distm.get("delta_jb_stat_rel_Q", float("nan"))),
                    "var_ratio_I": float(distm.get("var_ratio_I", float("nan"))),
                    "var_ratio_Q": float(distm.get("var_ratio_Q", float("nan"))),
                    "stat_mmd_pval": float(sf_mmd["pval"]),
                    "stat_energy_pval": float(sf_energy["pval"]),
                    "stat_mmd_qval": float("nan"),
                    "stat_energy_qval": float("nan"),
                    "coverage_50": float(distm.get("coverage_50", float("nan"))),
                    "coverage_80": float(distm.get("coverage_80", float("nan"))),
                    "coverage_95": float(distm.get("coverage_95", float("nan"))),
                    "delta_psd_l2": float(distm.get("delta_psd_l2", float("nan"))),
                    "delta_skew_l2": float(distm.get("delta_skew_l2", float("nan"))),
                    "delta_kurt_l2": float(distm.get("delta_kurt_l2", float("nan"))),
                    "delta_jb_stat_rel": float(distm.get("delta_jb_stat_rel", float("nan"))),
                }
            )

        if not rows:
            return

        valid_mmd = [i for i, row in enumerate(rows) if np.isfinite(row["stat_mmd_pval"])]
        valid_energy = [i for i, row in enumerate(rows) if np.isfinite(row["stat_energy_pval"])]
        if valid_mmd:
            qvals = benjamini_hochberg(np.asarray([rows[i]["stat_mmd_pval"] for i in valid_mmd], dtype=float))
            for i, qv in zip(valid_mmd, qvals):
                rows[i]["stat_mmd_qval"] = float(qv)
        if valid_energy:
            qvals = benjamini_hochberg(np.asarray([rows[i]["stat_energy_pval"] for i in valid_energy], dtype=float))
            for i, qv in zip(valid_energy, qvals):
                rows[i]["stat_energy_qval"] = float(qv)

        for row in rows:
            row["gate_g3"] = self._gate_g3(row)
            row["gate_g5"] = self._gate_g5(row)
            row["gate_g6"] = self._gate_g6(row["stat_mmd_qval"], row["stat_energy_qval"])
            row["validation_status_partial"] = self._validation_status_partial(
                row["gate_g3"], row["gate_g5"], row["gate_g6"]
            )
            row["val_recon_loss"] = float(logs.get("val_recon_loss", np.nan))
            row["val_loss"] = float(logs.get("val_loss", np.nan))
            self._history.append(row)

        df_hist = pd.DataFrame(self._history)
        df_hist.to_csv(self.history_path, index=False)
        latest = {"epoch": int(epoch + 1), "rows": rows}
        self.latest_path.write_text(json.dumps(latest, indent=2), encoding="utf-8")


# ======================================================================
# Callback factory
# ======================================================================
def build_callbacks(
    training_config: Dict[str, Any],
    model_config: Dict[str, Any],
    kl_cb: KLAnnealingCallback,
    regime_diag_callback: Optional[Callback] = None,
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

    callbacks: List[Callback] = [
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
    if regime_diag_callback is not None:
        callbacks.append(regime_diag_callback)
    return callbacks
