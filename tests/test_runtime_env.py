from __future__ import annotations

import subprocess

import pytest

from src.config import runtime_env


def test_ensure_required_python_modules_bootstraps_known_lightweight_modules(monkeypatch, tmp_path):
    state = {"installed": False}

    monkeypatch.setenv("CVAE_AUTO_BOOTSTRAP_PYDEPS", "1")
    monkeypatch.setattr(runtime_env, "default_pydeps_dir", lambda: tmp_path / ".pydeps")
    monkeypatch.setattr(runtime_env, "ensure_repo_pydeps_on_sys_path", lambda: [])

    real_import_module = runtime_env.importlib.import_module

    def _fake_import(name: str):
        if name == "matplotlib" and not state["installed"]:
            raise ModuleNotFoundError(name)
        return object() if name == "matplotlib" else real_import_module(name)

    def _fake_run(cmd, capture_output, text):
        assert "matplotlib<3.9" in cmd
        state["installed"] = True
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(runtime_env.importlib, "import_module", _fake_import)
    monkeypatch.setattr(runtime_env.subprocess, "run", _fake_run)

    missing = runtime_env.ensure_required_python_modules(
        ("matplotlib",),
        context="runtime-env test",
        allow_missing=False,
    )

    assert missing == []
    assert state["installed"] is True


def test_ensure_required_python_modules_does_not_auto_install_unknown_modules(monkeypatch):
    monkeypatch.setenv("CVAE_AUTO_BOOTSTRAP_PYDEPS", "1")

    def _fake_import(name: str):
        if name == "tensorflow":
            raise ModuleNotFoundError(name)
        return object()

    monkeypatch.setattr(runtime_env, "ensure_repo_pydeps_on_sys_path", lambda: [])
    monkeypatch.setattr(runtime_env.importlib, "import_module", _fake_import)

    with pytest.raises(RuntimeError):
        runtime_env.ensure_required_python_modules(
            ("tensorflow",),
            context="runtime-env test",
            allow_missing=False,
        )
