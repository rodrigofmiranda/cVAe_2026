# -*- coding: utf-8 -*-
"""
src.training.logging — Run bootstrap utilities and state_run writer.

Responsible for:
- Resolving OUTPUT_BASE from environment.
- Generating or reading RUN_ID.
- Creating RUN_DIR and its canonical subdirectories
  (plots, tables, models, logs).
- Writing ``_last_run.txt`` pointer.
- Writing the official ``state_run.json`` (see :func:`write_state_run`).

Commit: refactor(step1).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Union


class RunPaths(NamedTuple):
    """Immutable bundle of canonical run directory paths."""
    run_id: str
    run_dir: Path
    plots_dir: Path
    tables_dir: Path
    models_dir: Path
    logs_dir: Path


def bootstrap_run(
    output_base: Path | str | None = None,
    run_id: str | None = None,
) -> RunPaths:
    """Create (or reuse) a timestamped run directory with standard subdirs.

    Parameters
    ----------
    output_base : Path or str, optional
        Root for all runs.  Falls back to ``$OUTPUT_BASE`` env-var,
        then ``/workspace/2026/outputs``.
    run_id : str, optional
        Explicit run identifier (e.g. ``run_20260302_143000``).
        Falls back to ``$RUN_ID`` env-var, then auto-generates
        ``run_YYYYMMDD_HHMMSS`` from current wall-clock time.

    Returns
    -------
    RunPaths
        Named tuple with *run_id*, *run_dir*, *plots_dir*,
        *tables_dir*, *models_dir*, *logs_dir*.
    """
    if output_base is None:
        output_base = Path(os.environ.get("OUTPUT_BASE", "/workspace/2026/outputs"))
    else:
        output_base = Path(output_base)

    if run_id is None:
        run_id = os.environ.get("RUN_ID", "").strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_dir = output_base / run_id
    plots_dir = run_dir / "plots"
    tables_dir = run_dir / "tables"
    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"

    for p in [run_dir, plots_dir, tables_dir, models_dir, logs_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # Pointer file so that evaluation can find the latest run.
    try:
        (output_base / "_last_run.txt").write_text(str(run_dir), encoding="utf-8")
    except Exception:
        pass

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        plots_dir=plots_dir,
        tables_dir=tables_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
    )


# ------------------------------------------------------------------
# state_run.json writer
# ------------------------------------------------------------------

def write_state_run(
    run_dir: Union[str, Path],
    *,
    run_id: str = "",
    dataset_root: str = "",
    output_base: str = "",
    training_config: Optional[Dict[str, Any]] = None,
    data_reduction_config: Optional[Dict[str, Any]] = None,
    analysis_quick: Optional[Dict[str, Any]] = None,
    normalization: Optional[Dict[str, float]] = None,
    data_split: Optional[Dict[str, Any]] = None,
    eval_protocol: Optional[Dict[str, Any]] = None,
    grid: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write (or overwrite) the canonical ``state_run.json``.

    This is the *single authoritative writer* for the file.  Both the
    training monolith and future refactored code should call this
    instead of hand-building the JSON.

    Parameters
    ----------
    run_dir : path
        Run directory (will contain the resulting ``state_run.json``).
    run_id, dataset_root, output_base : str
        Top-level scalar metadata.
    training_config, data_reduction_config, analysis_quick,
    normalization, data_split, eval_protocol, grid, artifacts : dict
        Sub-sections of the state.  ``None`` → empty dict / None.
    extra : dict, optional
        Any additional top-level keys to merge (e.g. ``dataset_root_env``).

    Returns
    -------
    Path
        The written ``state_run.json`` path.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "plots": str(run_dir / "plots"),
        "tables": str(run_dir / "tables"),
        "models": str(run_dir / "models"),
        "logs": str(run_dir / "logs"),
    }

    state: Dict[str, Any] = {
        "run_id": run_id or run_dir.name,
        "run_dir": str(run_dir),
        "dataset_root": dataset_root,
        "output_base": output_base,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "paths": paths,
        "training_config": training_config or {},
        "data_reduction_config": data_reduction_config or {},
        "analysis_quick": analysis_quick or {},
        "normalization": normalization,
        "data_split": data_split or {},
        "eval_protocol": eval_protocol or {},
        "grid": grid or {},
        "artifacts": artifacts or {},
    }

    if extra:
        state.update(extra)

    state_path = run_dir / "state_run.json"
    state_path.write_text(
        json.dumps(state, indent=2, default=str), encoding="utf-8"
    )
    return state_path
