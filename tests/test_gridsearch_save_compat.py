from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.training.gridsearch import (
    _is_retryable_seq_gru_runtime_error,
    _save_keras_model_compat,
    run_gridsearch,
)


class _FakeModel:
    def __init__(self, fail_with_include_optimizer: bool = True):
        self.fail_with_include_optimizer = fail_with_include_optimizer
        self.calls = []

    def save(self, path, **kwargs):
        self.calls.append((path, dict(kwargs)))
        if self.fail_with_include_optimizer and "include_optimizer" in kwargs:
            raise ValueError(
                "The following argument(s) are not supported with the native TF-Keras format: ['include_optimizer']"
            )


def test_save_keras_model_compat_retries_without_include_optimizer(tmp_path: Path):
    model = _FakeModel(fail_with_include_optimizer=True)
    out = tmp_path / "model.keras"

    _save_keras_model_compat(model, out)

    assert len(model.calls) == 2
    assert model.calls[0][0] == str(out)
    assert model.calls[0][1] == {"include_optimizer": False}
    assert model.calls[1][0] == str(out)
    assert model.calls[1][1] == {}


def test_save_keras_model_compat_keeps_other_value_errors(tmp_path: Path):
    class _ExplodingModel:
        def save(self, path, **kwargs):
            raise ValueError("another save failure")

    out = tmp_path / "model.keras"
    model = _ExplodingModel()

    try:
        _save_keras_model_compat(model, out)
    except ValueError as exc:
        assert str(exc) == "another save failure"
    else:
        raise AssertionError("expected ValueError to propagate")


def test_run_gridsearch_rejects_empty_grid_with_clear_message(tmp_path: Path):
    arrays_2 = np.zeros((0, 2), dtype=np.float32)
    arrays_1 = np.zeros((0, 1), dtype=np.float32)
    run_paths = SimpleNamespace(models_dir=tmp_path / "models", run_dir=tmp_path)

    try:
        run_gridsearch(
            grid=[],
            training_config={},
            analysis_quick={},
            X_train=arrays_2,
            Y_train=arrays_2,
            Dn_train=arrays_1,
            Cn_train=arrays_1,
            X_val=arrays_2,
            Y_val=arrays_2,
            Dn_val=arrays_1,
            Cn_val=arrays_1,
            run_paths=run_paths,
            overrides={"grid_tag": "NON_EXISTENT"},
        )
    except ValueError as exc:
        msg = str(exc)
        assert "0 configura" in msg
        assert "grid_preset/grid_tag" in msg
        assert "NON_EXISTENT" in msg
    else:
        raise AssertionError("expected ValueError for empty filtered grid")


def test_retryable_seq_gru_runtime_error_detects_cudnn_failure_signature():
    cfg = {"arch_variant": "seq_bigru_residual", "seq_gru_unroll": False}
    exc = RuntimeError("Sequence lengths for RNN are required from CUDNN 9.0+")

    assert _is_retryable_seq_gru_runtime_error(exc, cfg) is True


def test_retryable_seq_gru_runtime_error_ignores_non_seq_or_compat_cfg():
    exc = RuntimeError("Failed to call DoRnnForward with model config")

    assert _is_retryable_seq_gru_runtime_error(
        exc, {"arch_variant": "concat", "seq_gru_unroll": False}
    ) is False
    assert _is_retryable_seq_gru_runtime_error(
        exc, {"arch_variant": "seq_bigru_residual", "seq_gru_unroll": True}
    ) is False
    assert _is_retryable_seq_gru_runtime_error(
        exc,
        {
            "arch_variant": "seq_bigru_residual",
            "seq_gru_unroll": False,
            "seq_gru_backend": "compat",
        },
    ) is False
