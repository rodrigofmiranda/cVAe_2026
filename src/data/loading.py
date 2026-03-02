# -*- coding: utf-8 -*-
"""
src.data.loading — Low-level dataset IO helpers.

Pure functions for reading and validating VLC experiment arrays.
No dataset discovery, splitting, or normalization here — those
will be added in later commits.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


# Accepted filenames for the received-signal array (tried in order).
ALT_RECV: list[str] = [
    "received_data_tuple_sync-phase.npy",
    "received_data_tuple_sync_phase.npy",
    "received_data_tuple_sync.npy",
    "received_data_tuple.npy",
]


def ensure_iq_shape(arr) -> np.ndarray:
    """Coerce an array to shape ``(N, 2)`` float32 I/Q format.

    Handles complex arrays (split into real/imag) and transposed layouts.

    Raises
    ------
    ValueError
        If the array cannot be coerced to ``(N, 2)``.
    """
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        arr = np.stack([arr.real, arr.imag], axis=-1)
    if arr.ndim == 2 and arr.shape[1] == 2:
        pass
    elif arr.ndim == 2 and arr.shape[0] == 2:
        arr = arr.T
    else:
        raise ValueError(f"Formato inesperado I/Q: shape={arr.shape}, dtype={arr.dtype}")
    return arr.astype(np.float32, copy=False)


def read_metadata(exp_dir: Path) -> dict:
    """Read the experiment metadata JSON, trying several candidate paths.

    Parameters
    ----------
    exp_dir : Path
        Root directory of a single experiment (e.g. ``dist_1m/curr_60mA/...``).

    Returns
    -------
    dict
        Parsed metadata, or ``{}`` if no file found / all reads fail.
    """
    candidates = [exp_dir / "metadata.json", exp_dir / "IQ_data" / "metadata.json"]
    candidates += list(exp_dir.glob("*_meta.json"))
    for meta_path in candidates:
        if meta_path.exists():
            for enc in ["utf-8", "latin-1"]:
                try:
                    return json.loads(meta_path.read_text(encoding=enc))
                except Exception:
                    pass
    return {}


def parse_dist_curr_from_path(exp_dir: Path) -> tuple[float | None, int | None]:
    """Extract distance (m) and current (mA) from a directory path.

    Looks for patterns like ``dist_1.5m`` and ``curr_60mA`` in the
    stringified path.

    Returns
    -------
    tuple[float | None, int | None]
        ``(distance, current)``; either may be ``None`` if the pattern
        is not found.
    """
    s = str(exp_dir).replace("\\", "/")
    md = re.search(r"/dist_(\d+(?:\.\d+)?)m(?:/|$)", s)
    mc = re.search(r"/curr_(\d+)mA(?:/|$)", s)
    dist = float(md.group(1)) if md else None
    curr = int(mc.group(1)) if mc else None
    return dist, curr
