# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.callbacks import MiniProtocolReanalysisCallback, RegimeDiagnosticsCallback


class _FakeInferenceModel:
    def __init__(self, deterministic: bool):
        self.deterministic = deterministic

    def predict(self, inputs, batch_size=4096, verbose=0):
        x = np.asarray(inputs[0], dtype=np.float32)
        if x.ndim == 3:
            x_center = x[:, x.shape[1] // 2, :]
        else:
            x_center = x
        if self.deterministic:
            return x_center + 0.05
        return x_center + np.random.normal(scale=0.08, size=x_center.shape).astype(np.float32)


def test_regime_diagnostics_callback_respects_epoch_cadence_and_writes_files(monkeypatch, tmp_path):
    from src import models as _models_pkg  # noqa: F401
    from src.models import cvae as cvae_mod
    from src.evaluation import stat_tests as stat_tests_mod

    monkeypatch.setattr(
        cvae_mod,
        "create_inference_model_from_full",
        lambda model, deterministic=True: _FakeInferenceModel(deterministic),
    )
    monkeypatch.setattr(
        stat_tests_mod,
        "mmd_rbf",
        lambda a, b, n_perm=100, seed=42: {"mmd2": 0.01, "pval": 0.20, "bandwidth": 0.5},
    )
    monkeypatch.setattr(
        stat_tests_mod,
        "energy_test",
        lambda a, b, n_perm=100, seed=42: {"energy": 0.01, "pval": 0.30},
    )

    n = 16
    x = np.linspace(-0.2, 0.2, n * 2, dtype=np.float32).reshape(n, 2)
    y = x + 0.02
    d_raw = np.asarray([[0.8]] * 8 + [[1.0]] * 8, dtype=np.float32)
    c_raw = np.asarray([[100.0]] * 8 + [[300.0]] * 8, dtype=np.float32)
    d_norm = np.asarray([[0.0]] * 8 + [[1.0]] * 8, dtype=np.float32)
    c_norm = np.asarray([[0.0]] * 8 + [[1.0]] * 8, dtype=np.float32)

    cb = RegimeDiagnosticsCallback(
        logs_dir=tmp_path / "logs",
        x_val_input=x,
        x_val_center=x,
        y_val=y,
        d_val_norm=d_norm,
        c_val_norm=c_norm,
        d_val_raw=d_raw,
        c_val_raw=c_raw,
        every_n_epochs=2,
        mc_samples=2,
        max_samples_per_regime=8,
        stat_n_perm=32,
    )
    cb.set_model(object())

    cb.on_epoch_end(0, logs={"val_loss": 1.0, "val_recon_loss": 0.8})
    assert not cb.history_path.exists()

    cb.on_epoch_end(1, logs={"val_loss": 0.9, "val_recon_loss": 0.7})
    assert cb.history_path.exists()
    assert cb.latest_path.exists()

    df = pd.read_csv(cb.history_path)
    latest = json.loads(Path(cb.latest_path).read_text(encoding="utf-8"))
    assert set(df["regime_id"]) == {"dist_0p8m__curr_100mA", "dist_1m__curr_300mA"}
    assert set(df["epoch"]) == {2}
    assert "gate_g3" in df.columns
    assert "gate_g5" in df.columns
    assert "gate_g6" in df.columns
    assert latest["epoch"] == 2
    assert len(latest["rows"]) == 2


def test_mini_protocol_reanalysis_callback_writes_summary_and_table(monkeypatch, tmp_path):
    import src.models.callbacks as callbacks_mod

    rows = [
        {
            "regime_id": "dist_0p8m__curr_100mA",
            "validation_status_partial": "fail",
            "gate_g5": False,
            "gate_g6": False,
            "effective_abs_delta_coverage_95": 0.12,
            "effective_delta_jb_stat_rel": 8.0,
            "delta_psd_l2": 0.2,
            "delta_skew_l2": 0.3,
            "delta_kurt_l2": 0.4,
        },
        {
            "regime_id": "dist_1m__curr_300mA",
            "validation_status_partial": "pass",
            "gate_g5": True,
            "gate_g6": True,
            "effective_abs_delta_coverage_95": 0.02,
            "effective_delta_jb_stat_rel": 1.0,
            "delta_psd_l2": 0.1,
            "delta_skew_l2": 0.1,
            "delta_kurt_l2": 0.2,
        },
    ]

    monkeypatch.setattr(callbacks_mod, "_collect_regime_diagnostic_rows", lambda **kwargs: rows)

    n = 4
    x = np.zeros((n, 2), dtype=np.float32)
    y = np.zeros((n, 2), dtype=np.float32)
    d = np.zeros((n, 1), dtype=np.float32)
    c = np.zeros((n, 1), dtype=np.float32)

    cb = MiniProtocolReanalysisCallback(
        artifact_dir=tmp_path / "artifact",
        x_val_input=x,
        x_val_center=x,
        y_val=y,
        d_val_norm=d,
        c_val_norm=c,
        d_val_raw=d,
        c_val_raw=c,
        enabled=True,
        scope="all12",
    )
    cb.set_model(object())
    cb.set_params({"epochs": 5})
    cb.on_train_end(logs={"val_loss": 0.2})

    assert cb.summary_path.exists()
    assert cb.table_path.exists()

    summary = json.loads(cb.summary_path.read_text(encoding="utf-8"))
    df = pd.read_csv(cb.table_path)

    assert summary["ranking_mode"] == "mini_protocol_v1"
    assert summary["mini_n_regimes"] == 2
    assert summary["mini_n_fail"] == 1
    assert summary["mini_n_fail_0p8m"] == 1
    assert summary["mini_n_g5_fail"] == 1
    assert summary["mini_n_g6_fail"] == 1
    assert summary["mini_n_g5_fail_0p8m"] == 1
    assert summary["mini_n_g6_fail_0p8m"] == 1
    assert summary["scope"] == "all12"
    assert list(df["regime_id"]) == ["dist_0p8m__curr_100mA", "dist_1m__curr_300mA"]
