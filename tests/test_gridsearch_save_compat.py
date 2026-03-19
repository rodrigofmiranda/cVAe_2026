from pathlib import Path

from src.training.gridsearch import _save_keras_model_compat


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
