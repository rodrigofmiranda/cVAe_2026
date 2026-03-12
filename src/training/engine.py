# -*- coding: utf-8 -*-
"""Canonical cVAE training engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def train_engine(
    dataset_root: str | Path,
    output_base: str | Path,
    *,
    run_id: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a full cVAE training pipeline (data → train → save).

    This is the **canonical entry-point** for training.  It:

    1. Bootstraps the run directory (``outputs/run_YYYYMMDD_HHMMSS``).
    2. Loads and splits the dataset.
    3. Normalises conditions (D, C → [0, 1]).
    4. Runs the grid-search training loop.
    5. Saves models, logs, plots, tables, and ``state_run.json``.

    Parameters
    ----------
    dataset_root : path
        Root of the organised dataset
        (e.g. ``data/dataset_fullsquare_organized``).
    output_base : path
        Root for all run directories (e.g. ``outputs/``).
    run_id : str, optional
        Explicit run identifier.  If *None*, a timestamped id is
        generated automatically.
    overrides : dict, optional
        CLI / protocol overrides forwarded to the training logic
        (``max_epochs``, ``max_grids``, ``dry_run``, etc.).

    Returns
    -------
    dict
        Summary with at least ``{"run_id", "run_dir", "status"}``.
        On success also contains ``"state_run_path"``.
    """
    dataset_root = str(Path(dataset_root).resolve())
    output_base = str(Path(output_base).resolve())
    overrides = dict(overrides or {})

    from src.training.pipeline import run_training_pipeline

    return run_training_pipeline(
        dataset_root=dataset_root,
        output_base=output_base,
        run_id=run_id,
        overrides=overrides,
    )
