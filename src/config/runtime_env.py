# -*- coding: utf-8 -*-
"""
src.config.runtime_env — runtime environment hardening helpers.

Keeps optional runtime side effects (for example Matplotlib cache setup)
in one place so entrypoints can opt in without duplicating logic.
"""

from __future__ import annotations

import os
from pathlib import Path


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
