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


def _validation_status_partial(*gates: Any) -> str:
    vals = [g for g in gates if g is not None]
    if not vals:
        return "partial"
    if any(v is False for v in vals):
        return "fail"
    if all(v is True for v in vals):
        return "pass"
    return "partial"


def _effective_jb_rel(distm: Dict[str, float]) -> float:
    total = float(distm.get("delta_jb_stat_rel", float("nan")))
    if np.isfinite(total):
        return total
    vals = [
        float(distm.get("delta_jb_stat_rel_I", float("nan"))),
        float(distm.get("delta_jb_stat_rel_Q", float("nan"))),
    ]
    vals = [abs(v) for v in vals if np.isfinite(v)]
    return float(max(vals)) if vals else float("nan")


def _effective_abs_delta_coverage_95(distm: Dict[str, float]) -> float:
    vals = [
        float(distm.get("delta_coverage_95_I", float("nan"))),
        float(distm.get("delta_coverage_95_Q", float("nan"))),
    ]
    vals = [abs(v) for v in vals if np.isfinite(v)]
    if vals:
        return float(np.mean(vals))
    total = float(distm.get("delta_coverage_95", float("nan")))
    return abs(total) if np.isfinite(total) else float("nan")


def _gate_g3_from_distm(distm: Dict[str, float]) -> Optional[bool]:
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


def _gate_g5_from_distm(distm: Dict[str, float]) -> Optional[bool]:
    from src.evaluation.validation_summary import TWIN_GATE_THRESHOLDS

    skew_l2 = float(distm.get("delta_skew_l2", float("nan")))
    kurt_l2 = float(distm.get("delta_kurt_l2", float("nan")))
    jb_rel = _effective_jb_rel(distm)
    if not all(np.isfinite(v) for v in (skew_l2, kurt_l2, jb_rel)):
        return None
    return bool(
        skew_l2 < float(TWIN_GATE_THRESHOLDS["delta_skew_l2"])
        and kurt_l2 < float(TWIN_GATE_THRESHOLDS["delta_kurt_l2"])
        and jb_rel < float(TWIN_GATE_THRESHOLDS["delta_jb_stat_rel"])
    )


def _gate_g6_from_qvals(mmd_q: float, energy_q: float) -> Optional[bool]:
    from src.evaluation.validation_summary import TWIN_GATE_THRESHOLDS

    if not (np.isfinite(mmd_q) and np.isfinite(energy_q)):
        return None
    thr = float(TWIN_GATE_THRESHOLDS["stat_qval"])
    return bool(mmd_q > thr and energy_q > thr)


def _collect_regime_diagnostic_rows(
    *,
    model,
    x_val_input: np.ndarray,
    x_val_center: np.ndarray,
    y_val: np.ndarray,
    d_val_norm: np.ndarray,
    c_val_norm: np.ndarray,
    d_val_raw: np.ndarray,
    c_val_raw: np.ndarray,
    mc_samples: int,
    max_samples_per_regime: int,
    focus_only_0p8m: bool,
    stat_n_perm: int,
    stat_seed: int,
    seed_offset: int,
    logs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    from src.evaluation.metrics import residual_distribution_metrics
    from src.evaluation.stat_tests import benjamini_hochberg, energy_test, mmd_rbf
    from src.models.cvae import create_inference_model_from_full

    logs = logs or {}
    rng = np.random.default_rng(int(stat_seed) + int(seed_offset))

    inf_det = create_inference_model_from_full(model, deterministic=True)
    y_det = inf_det.predict(
        [x_val_input, d_val_norm, c_val_norm],
        batch_size=4096,
        verbose=0,
    )

    y_samples = None
    if mc_samples > 1:
        inf_sto = create_inference_model_from_full(model, deterministic=False)
        ys = []
        for _ in range(int(mc_samples)):
            ys.append(
                inf_sto.predict(
                    [x_val_input, d_val_norm, c_val_norm],
                    batch_size=4096,
                    verbose=0,
                )
            )
        y_samples = np.stack(ys, axis=0)

    regimes = sorted(
        {
            (float(d), float(c))
            for d, c in zip(np.asarray(d_val_raw).ravel().tolist(), np.asarray(c_val_raw).ravel().tolist())
        }
    )
    if focus_only_0p8m:
        regimes = [rc for rc in regimes if abs(rc[0] - 0.8) <= 1e-9]

    rows: List[Dict[str, Any]] = []
    for distance_m, current_mA in regimes:
        mask = (
            np.isclose(np.asarray(d_val_raw).ravel(), distance_m)
            & np.isclose(np.asarray(c_val_raw).ravel(), current_mA)
        )
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        if len(idx) > int(max_samples_per_regime):
            idx = np.sort(
                rng.choice(idx, size=int(max_samples_per_regime), replace=False)
            )

        x_center = np.asarray(x_val_center)[idx]
        y_true = np.asarray(y_val)[idx]
        y_det_reg = y_det[idx]
        y_stack_reg = None if y_samples is None else y_samples[:, idx, :]
        y_dist_reg = y_det_reg if y_stack_reg is None else y_stack_reg.reshape((-1, y_stack_reg.shape[-1]))
        x_dist_reg = x_center if y_stack_reg is None else np.tile(x_center, (int(mc_samples), 1))
        y_real_reg = y_true if y_stack_reg is None else np.tile(y_true, (int(mc_samples), 1))

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
        sf_mmd = mmd_rbf(rr, rp, n_perm=int(stat_n_perm), seed=int(stat_seed) + int(seed_offset) + int(current_mA))
        sf_energy = energy_test(rr, rp, n_perm=int(stat_n_perm), seed=int(stat_seed) + int(seed_offset) + int(current_mA) + 1)

        row: Dict[str, Any] = {
            "regime_id": RegimeDiagnosticsCallback._format_regime_id(distance_m, current_mA),
            "dist_target_m": float(distance_m),
            "curr_target_mA": float(current_mA),
            "n_samples": int(len(idx)),
            "val_recon_loss_proxy": float(np.mean(np.sum((y_det_reg - y_true) ** 2, axis=1))),
            "delta_wasserstein_I": float(distm.get("delta_wasserstein_I", float("nan"))),
            "delta_wasserstein_Q": float(distm.get("delta_wasserstein_Q", float("nan"))),
            "delta_jb_stat_rel_I": float(distm.get("delta_jb_stat_rel_I", float("nan"))),
            "delta_jb_stat_rel_Q": float(distm.get("delta_jb_stat_rel_Q", float("nan"))),
            "delta_jb_stat_rel": float(distm.get("delta_jb_stat_rel", float("nan"))),
            "var_ratio_I": float(distm.get("var_ratio_I", float("nan"))),
            "var_ratio_Q": float(distm.get("var_ratio_Q", float("nan"))),
            "stat_mmd_pval": float(sf_mmd["pval"]),
            "stat_energy_pval": float(sf_energy["pval"]),
            "stat_mmd_qval": float("nan"),
            "stat_energy_qval": float("nan"),
            "coverage_50": float(distm.get("coverage_50", float("nan"))),
            "coverage_80": float(distm.get("coverage_80", float("nan"))),
            "coverage_95": float(distm.get("coverage_95", float("nan"))),
            "coverage_95_I": float(distm.get("coverage_95_I", float("nan"))),
            "coverage_95_Q": float(distm.get("coverage_95_Q", float("nan"))),
            "delta_coverage_95": float(distm.get("delta_coverage_95", float("nan"))),
            "delta_coverage_95_I": float(distm.get("delta_coverage_95_I", float("nan"))),
            "delta_coverage_95_Q": float(distm.get("delta_coverage_95_Q", float("nan"))),
            "delta_psd_l2": float(distm.get("delta_psd_l2", float("nan"))),
            "delta_skew_l2": float(distm.get("delta_skew_l2", float("nan"))),
            "delta_kurt_l2": float(distm.get("delta_kurt_l2", float("nan"))),
            "var_real_delta": float(distm.get("var_real_delta", float("nan"))),
            "delta_mean_l2": float(distm.get("delta_mean_l2", float("nan"))),
            "delta_cov_fro": float(distm.get("delta_cov_fro", float("nan"))),
        }
        row["effective_delta_jb_stat_rel"] = _effective_jb_rel(row)
        row["effective_abs_delta_coverage_95"] = _effective_abs_delta_coverage_95(row)
        rows.append(row)

    if not rows:
        return rows

    valid_mmd = [i for i, row in enumerate(rows) if np.isfinite(row["stat_mmd_pval"])]
    valid_energy = [i for i, row in enumerate(rows) if np.isfinite(row["stat_energy_pval"])]
    if valid_mmd:
        qvals = benjamini_hochberg(
            np.asarray([rows[i]["stat_mmd_pval"] for i in valid_mmd], dtype=float)
        )
        for i, qv in zip(valid_mmd, qvals):
            rows[i]["stat_mmd_qval"] = float(qv)
    if valid_energy:
        qvals = benjamini_hochberg(
            np.asarray([rows[i]["stat_energy_pval"] for i in valid_energy], dtype=float)
        )
        for i, qv in zip(valid_energy, qvals):
            rows[i]["stat_energy_qval"] = float(qv)

    for row in rows:
        row["gate_g3"] = _gate_g3_from_distm(row)
        row["gate_g5"] = _gate_g5_from_distm(row)
        row["gate_g6"] = _gate_g6_from_qvals(row["stat_mmd_qval"], row["stat_energy_qval"])
        row["validation_status_partial"] = _validation_status_partial(
            row["gate_g3"], row["gate_g5"], row["gate_g6"]
        )
        row["val_recon_loss"] = float(logs.get("val_recon_loss", np.nan))
        row["val_loss"] = float(logs.get("val_loss", np.nan))

    return rows


def _summarize_mini_protocol_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _mean(key: str) -> float:
        vals = [float(row.get(key, float("nan"))) for row in rows]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    statuses = [str(row.get("validation_status_partial", "")) for row in rows]
    failed_g5 = [row for row in rows if row.get("gate_g5") is False]
    failed_g6 = [row for row in rows if row.get("gate_g6") is False]
    failed_0p8m = [row for row in rows if str(row.get("regime_id", "")).startswith("dist_0p8m__")]
    failed_0p8m_any = [row for row in failed_0p8m if row.get("validation_status_partial") == "fail"]
    failed_0p8m_g5 = [row for row in failed_0p8m if row.get("gate_g5") is False]
    failed_0p8m_g6 = [row for row in failed_0p8m if row.get("gate_g6") is False]
    failed = [row for row in rows if row.get("validation_status_partial") == "fail"]
    partial = [row for row in rows if row.get("validation_status_partial") == "partial"]
    passed = [row for row in rows if row.get("validation_status_partial") == "pass"]
    return {
        "ranking_mode": "mini_protocol_v1",
        "mini_n_regimes": int(len(rows)),
        "mini_n_pass": int(len(passed)),
        "mini_n_partial": int(len(partial)),
        "mini_n_fail": int(len(failed)),
        "mini_n_fail_0p8m": int(len(failed_0p8m_any)),
        "mini_n_g5_fail": int(len(failed_g5)),
        "mini_n_g6_fail": int(len(failed_g6)),
        "mini_n_g5_fail_0p8m": int(len(failed_0p8m_g5)),
        "mini_n_g6_fail_0p8m": int(len(failed_0p8m_g6)),
        "mini_mean_abs_delta_coverage_95": _mean("effective_abs_delta_coverage_95"),
        "mini_mean_delta_jb": _mean("effective_delta_jb_stat_rel"),
        "mini_mean_delta_psd_l2": _mean("delta_psd_l2"),
        "mini_mean_delta_skew_l2": _mean("delta_skew_l2"),
        "mini_mean_delta_kurt_l2": _mean("delta_kurt_l2"),
        "mini_failed_regimes": [row["regime_id"] for row in failed],
        "mini_failed_g6_regimes": [row["regime_id"] for row in failed_g6],
        "mini_failed_g5_regimes": [row["regime_id"] for row in failed_g5],
        "status_counts": {
            str(k): int(v)
            for k, v in pd.Series(statuses).value_counts().sort_index().items()
        },
    }


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

    def on_epoch_end(self, epoch, logs=None):
        if not self.enabled or ((epoch + 1) % self.every_n_epochs) != 0:
            return
        rows = _collect_regime_diagnostic_rows(
            model=self.model,
            x_val_input=self.x_val_input,
            x_val_center=self.x_val_center,
            y_val=self.y_val,
            d_val_norm=self.d_val_norm,
            c_val_norm=self.c_val_norm,
            d_val_raw=self.d_val_raw,
            c_val_raw=self.c_val_raw,
            mc_samples=self.mc_samples,
            max_samples_per_regime=self.max_samples_per_regime,
            focus_only_0p8m=self.focus_only_0p8m,
            stat_n_perm=self.stat_n_perm,
            stat_seed=self.stat_seed,
            seed_offset=int(epoch + 1),
            logs=logs,
        )
        if not rows:
            return
        for row in rows:
            row["epoch"] = int(epoch + 1)
            self._history.append(row)

        pd.DataFrame(self._history).to_csv(self.history_path, index=False)
        latest = {"epoch": int(epoch + 1), "rows": rows}
        self.latest_path.write_text(json.dumps(latest, indent=2), encoding="utf-8")


class MiniProtocolReanalysisCallback(Callback):
    """Run a protocol-like validation sweep once at the end of training."""

    def __init__(
        self,
        *,
        artifact_dir: Path,
        x_val_input: np.ndarray,
        x_val_center: np.ndarray,
        y_val: np.ndarray,
        d_val_norm: np.ndarray,
        c_val_norm: np.ndarray,
        d_val_raw: np.ndarray,
        c_val_raw: np.ndarray,
        enabled: bool = False,
        scope: str = "all12",
        mc_samples: int = 4,
        max_samples_per_regime: int = 4096,
        stat_n_perm: int = 100,
        stat_seed: int = 42,
    ):
        super().__init__()
        self.artifact_dir = Path(artifact_dir)
        self.logs_dir = self.artifact_dir / "logs" / "train"
        self.tables_dir = self.artifact_dir / "tables"
        self.summary_path = self.logs_dir / "mini_protocol_summary.json"
        self.table_path = self.tables_dir / "mini_protocol_by_regime.csv"
        self.enabled = bool(enabled)
        self.scope = str(scope or "all12").strip().lower()
        self.mc_samples = max(1, int(mc_samples))
        self.max_samples_per_regime = max(128, int(max_samples_per_regime))
        self.stat_n_perm = max(32, int(stat_n_perm))
        self.stat_seed = int(stat_seed)
        self.x_val_input = np.asarray(x_val_input)
        self.x_val_center = np.asarray(x_val_center)
        self.y_val = np.asarray(y_val)
        self.d_val_norm = np.asarray(d_val_norm).reshape(-1, 1)
        self.c_val_norm = np.asarray(c_val_norm).reshape(-1, 1)
        self.d_val_raw = np.asarray(d_val_raw).reshape(-1, 1)
        self.c_val_raw = np.asarray(c_val_raw).reshape(-1, 1)
        self.rows: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def on_train_end(self, logs=None):
        if not self.enabled:
            return
        if self.scope != "all12":
            raise ValueError(
                f"Unsupported mini_reanalysis_scope={self.scope!r}; expected 'all12'."
            )
        rows = _collect_regime_diagnostic_rows(
            model=self.model,
            x_val_input=self.x_val_input,
            x_val_center=self.x_val_center,
            y_val=self.y_val,
            d_val_norm=self.d_val_norm,
            c_val_norm=self.c_val_norm,
            d_val_raw=self.d_val_raw,
            c_val_raw=self.c_val_raw,
            mc_samples=self.mc_samples,
            max_samples_per_regime=self.max_samples_per_regime,
            focus_only_0p8m=False,
            stat_n_perm=self.stat_n_perm,
            stat_seed=self.stat_seed,
            seed_offset=int(self.params.get("epochs", 0)),
            logs=logs,
        )
        self.rows = rows
        self.summary = _summarize_mini_protocol_rows(rows) if rows else {
            "ranking_mode": "mini_protocol_v1",
            "mini_n_regimes": 0,
            "mini_n_pass": 0,
            "mini_n_partial": 0,
            "mini_n_fail": 0,
            "mini_n_g5_fail": 0,
            "mini_n_g6_fail": 0,
            "mini_mean_abs_delta_coverage_95": float("nan"),
            "mini_mean_delta_jb": float("nan"),
            "mini_mean_delta_psd_l2": float("nan"),
        }
        self.summary["scope"] = self.scope
        self.summary["artifact_dir"] = str(self.artifact_dir)
        self.summary_path.write_text(
            json.dumps(self.summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        pd.DataFrame(rows).to_csv(self.table_path, index=False)


# ======================================================================
# Callback factory
# ======================================================================
def build_callbacks(
    training_config: Dict[str, Any],
    model_config: Dict[str, Any],
    kl_cb: KLAnnealingCallback,
    regime_diag_callback: Optional[Callback] = None,
    mini_reanalysis_callback: Optional[Callback] = None,
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
    if mini_reanalysis_callback is not None:
        callbacks.append(mini_reanalysis_callback)
    return callbacks
