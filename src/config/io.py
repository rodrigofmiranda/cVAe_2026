# -*- coding: utf-8 -*-
"""
src.config.io — Load / save / merge configuration files.

Supports JSON and YAML (if PyYAML is installed).

Also provides :func:`ensure_state_run_compat` for backward-compatible
loading of ``state_run.json`` from older runs that may be missing keys.

Commit: refactor(step1).
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.config.defaults import FALLBACK_STATE_RUN, TRAINING_DEFAULTS

logger = logging.getLogger(__name__)

# =====================================================================
# YAML availability
# =====================================================================

try:
    import yaml as _yaml

    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _yaml = None  # type: ignore[assignment]
    _HAS_YAML = False


# =====================================================================
# Public helpers
# =====================================================================

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON or YAML configuration file and return a plain dict.

    The format is determined by the file extension (``.json`` vs
    ``.yaml`` / ``.yml``).  Raises ``ValueError`` for unsupported
    extensions or if PyYAML is missing when a YAML file is requested.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        return _load_json(path)
    if suffix in (".yaml", ".yml"):
        return _load_yaml(path)
    raise ValueError(
        f"Unsupported config format {suffix!r} for {path}. "
        "Use .json or .yaml/.yml."
    )


def save_json(path: Union[str, Path], obj: Any, *, indent: int = 2) -> Path:
    """Write *obj* as pretty-printed JSON.  Creates parent dirs.

    Returns the resolved ``Path`` that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, default=str), encoding="utf-8")
    return path


def merge_overrides(
    base: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a **shallow copy** of *base* updated with *overrides*.

    - ``None`` values in overrides are skipped (don't override with None).
    - Original *base* is never mutated.
    """
    merged = copy.copy(base)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                merged[k] = v
    return merged


def ensure_state_run_compat(state: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all expected top-level keys exist in a ``state_run.json`` dict.

    Missing keys are filled from :data:`src.config.defaults.FALLBACK_STATE_RUN`
    and a warning is logged for each patched key.  The original dict is
    **mutated in-place** and also returned for convenience.

    This allows evaluation code to safely load state files from older runs
    that were created before certain keys existed.
    """
    fallback = FALLBACK_STATE_RUN

    for key, default_val in fallback.items():
        if key not in state:
            state[key] = copy.deepcopy(default_val)
            logger.warning(
                "state_run compat: missing key %r — filled with default.", key
            )

    # Ensure nested training_config has at minimum seed + val_split
    tc = state.get("training_config")
    if isinstance(tc, dict):
        for k, v in fallback.get("training_config", {}).items():
            if k not in tc:
                tc[k] = v
                logger.warning(
                    "state_run compat: training_config.%s missing — filled.", k
                )
    # Ensure data_split has split_mode
    ds = state.get("data_split")
    if isinstance(ds, dict):
        for k, v in fallback.get("data_split", {}).items():
            if k not in ds:
                ds[k] = v
                logger.warning(
                    "state_run compat: data_split.%s missing — filled.", k
                )

    return state


# =====================================================================
# Internals
# =====================================================================

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict:
    if not _HAS_YAML:
        raise ImportError(
            "PyYAML is required to load .yaml/.yml configs. "
            "Install with: pip install pyyaml"
        )
    with open(path, "r", encoding="utf-8") as f:
        return _yaml.safe_load(f) or {}
