# -*- coding: utf-8 -*-
"""
src.training.logging — Run bootstrap utilities.

Responsible for:
- Resolving OUTPUT_BASE from environment.
- Generating or reading RUN_ID.
- Creating RUN_DIR and its canonical subdirectories
  (plots, tables, models, logs).
- Writing ``_last_run.txt`` pointer.

All other state-run / metrics logging will be added in later commits.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import NamedTuple


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
