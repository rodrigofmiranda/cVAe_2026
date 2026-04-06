# -*- coding: utf-8 -*-
"""
src.config.runtime_env — runtime environment hardening helpers.

Keeps optional runtime side effects (for example Matplotlib cache setup)
in one place so entrypoints can opt in without duplicating logic.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


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
