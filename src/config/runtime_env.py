# -*- coding: utf-8 -*-
"""
src.config.runtime_env — runtime environment hardening helpers.

Keeps optional runtime side effects (for example Matplotlib cache setup)
in one place so entrypoints can opt in without duplicating logic.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


_AUTO_BOOTSTRAP_PACKAGE_SPECS = {
    "numpy": "numpy<2",
    "pandas": "pandas<3",
    "openpyxl": "openpyxl<4",
    "matplotlib": "matplotlib<3.9",
    "pytest": "pytest<10",
}


def ensure_writable_mpl_config_dir() -> str:
    """Point Matplotlib to a writable cache/config directory.

    Some execution environments expose a non-writable ``$HOME`` which makes
    Matplotlib fall back to ephemeral temp directories and emit noisy warnings.
    If ``MPLCONFIGDIR`` is already defined, this function preserves it.

    Returns
    -------
    str
        Effective ``MPLCONFIGDIR`` path.
    """
    existing = os.environ.get("MPLCONFIGDIR", "").strip()
    if existing:
        return existing

    user = os.environ.get("USER") or os.environ.get("USERNAME") or "codex"
    target = Path("/tmp") / f"matplotlib-{user}"
    target.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(target)
    return str(target)


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def candidate_pydeps_dirs() -> List[Path]:
    """Return likely persistent dependency dirs mounted in container runs."""
    repo_root = Path(__file__).resolve().parents[2]
    cwd = Path.cwd()

    env_pydeps = os.environ.get("CVAE_PYDEPS_DIR", "").strip()
    env_workdir = os.environ.get("CVAE_TF25_WORKDIR", "").strip()

    candidates: List[Path] = []
    if env_pydeps:
        candidates.append(Path(env_pydeps))
    if env_workdir:
        candidates.append(Path(env_workdir) / ".pydeps")

    candidates.extend(
        [
            cwd / ".pydeps",
            repo_root / ".pydeps",
        ]
    )
    return _unique_paths(candidates)


def default_pydeps_dir() -> Path:
    """Return the preferred persistent ``.pydeps`` install target."""
    candidates = candidate_pydeps_dirs()
    if candidates:
        return candidates[0]
    return Path.cwd() / ".pydeps"


def ensure_repo_pydeps_on_sys_path() -> List[str]:
    """Prepend discovered ``.pydeps`` directories to ``sys.path``.

    This makes direct ``python -m src.protocol.run ...`` invocations behave
    like the bootstrap wrapper, even if the shell did not source it.
    """
    added: List[str] = []
    for d in candidate_pydeps_dirs():
        if not d.is_dir():
            continue
        d_str = str(d.resolve())
        if d_str in sys.path:
            continue
        sys.path.insert(0, d_str)
        added.append(d_str)
    return added


def _install_known_modules_into_pydeps(
    modules: Sequence[str],
    *,
    context: str,
) -> List[str]:
    """Install known lightweight Python modules into the repo ``.pydeps`` dir.

    Returns the subset of modules that still remains missing after the attempt.
    Unknown/heavy modules (for example ``tensorflow``) are not installed here.
    """
    target = default_pydeps_dir().resolve()
    target.mkdir(parents=True, exist_ok=True)

    install_specs: List[str] = []
    for mod in modules:
        spec = _AUTO_BOOTSTRAP_PACKAGE_SPECS.get(mod)
        if spec and spec not in install_specs:
            install_specs.append(spec)

    if not install_specs:
        return list(modules)

    print(
        "[bootstrap] missing lightweight python deps for "
        f"{context}; installing into {target}: {', '.join(install_specs)}"
    )
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--target",
        str(target),
        *install_specs,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(
            "[bootstrap][warn] pip install failed while fixing runtime deps "
            f"for {context}.\n{proc.stdout}\n{proc.stderr}"
        )
        return list(modules)

    ensure_repo_pydeps_on_sys_path()
    still_missing: List[str] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            still_missing.append(mod)
    return still_missing


def ensure_required_python_modules(
    modules: Sequence[str],
    *,
    context: str,
    allow_missing: bool = False,
) -> List[str]:
    """Validate required python modules and fail early when missing.

    Parameters
    ----------
    modules:
        Module names to import-test.
    context:
        Human-readable label for error messages.
    allow_missing:
        When True, only prints a warning and returns the missing list.

    Returns
    -------
    list[str]
        Missing module names (possibly empty).
    """
    ensure_repo_pydeps_on_sys_path()

    missing: List[str] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            missing.append(mod)

    auto_bootstrap = (
        os.environ.get("CVAE_AUTO_BOOTSTRAP_PYDEPS", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    if missing and auto_bootstrap:
        lightweight = [mod for mod in missing if mod in _AUTO_BOOTSTRAP_PACKAGE_SPECS]
        if lightweight:
            remaining = _install_known_modules_into_pydeps(
                lightweight,
                context=context,
            )
            missing = [mod for mod in missing if mod not in lightweight] + remaining

    if not missing:
        return []

    hint = (
        "Dependências ausentes para "
        f"{context}: {', '.join(missing)}. "
        "Use `source scripts/ops/container_bootstrap_python.sh` "
        "ou execute via `scripts/ops/train.sh` / `scripts/ops/eval.sh`."
    )
    if allow_missing:
        print(f"⚠️  {hint}")
        return missing
    raise RuntimeError(hint)
