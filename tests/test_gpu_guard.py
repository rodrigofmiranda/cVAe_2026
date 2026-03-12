import sys
from types import SimpleNamespace

import pytest

from src.config import gpu_guard


def test_check_tensorflow_gpu_reports_visible_gpu(monkeypatch):
    fake_tf = SimpleNamespace(
        config=SimpleNamespace(
            list_physical_devices=lambda kind: ["GPU:0"] if kind == "GPU" else [],
            experimental=SimpleNamespace(set_memory_growth=lambda *_args, **_kwargs: None),
        )
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    gpus = gpu_guard.check_tensorflow_gpu()

    assert gpus == ["GPU:0"]


def test_warn_if_no_gpu_and_confirm_aborts_on_negative_answer(monkeypatch, capsys):
    fake_tf = SimpleNamespace(
        config=SimpleNamespace(
            list_physical_devices=lambda kind: [],
            experimental=SimpleNamespace(set_memory_growth=lambda *_args, **_kwargs: None),
        )
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    with pytest.raises(RuntimeError) as exc:
        gpu_guard.warn_if_no_gpu_and_confirm("training")

    assert "Aborted by user" in str(exc.value)
    out = capsys.readouterr().out
    assert "INICIANDO SEM GPU, CONTINUAR?" in out


def test_warn_if_no_gpu_and_confirm_allows_positive_answer(monkeypatch):
    fake_tf = SimpleNamespace(
        config=SimpleNamespace(
            list_physical_devices=lambda kind: [],
            experimental=SimpleNamespace(set_memory_growth=lambda *_args, **_kwargs: None),
        )
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    gpus = gpu_guard.warn_if_no_gpu_and_confirm("training")

    assert gpus == []
