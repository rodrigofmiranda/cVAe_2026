# -*- coding: utf-8 -*-
"""
src.training.engine — Training orchestrator.

Provides :func:`train_engine`, the single entry-point that future code
(CLI, protocol runner, notebooks) should call to launch a cVAE training
run.

Current implementation delegates to the training monolith
(``cvae_TRAIN_documented.main``).  The monolith itself now uses the
modular model API (refactor step 3):

    src.models.cvae       → build_cvae, create_inference_model_from_full
    src.models.losses     → CondPriorVAELoss, compute_total_loss
    src.models.sampling   → Sampling, reparameterize
    src.models.callbacks  → build_callbacks

Commits: refactor(step2), refactor(step3).
"""

from __future__ import annotations

import os
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

    # --- environment setup (monolith reads these) ---
    os.environ["DATASET_ROOT"] = dataset_root
    os.environ["OUTPUT_BASE"] = output_base
    if run_id:
        os.environ["RUN_ID"] = run_id

    # --- delegate to monolith (temporary) ---
    # Import lazily to avoid heavy TF init at import time.
    from src.training import cvae_TRAIN_documented as _monolith

    summary: Dict[str, Any] = {
        "run_id": run_id or os.environ.get("RUN_ID", ""),
        "run_dir": "",
        "status": "unknown",
    }

    try:
        _monolith.main(overrides=overrides)
        summary["status"] = "completed"
    except SystemExit:
        # dry_run may call sys.exit in some paths
        summary["status"] = "dry_run"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        raise

    # Resolve actual run_dir (monolith may have generated the id)
    _rid = os.environ.get("RUN_ID", summary["run_id"])
    _rdir = Path(output_base) / _rid
    summary["run_id"] = _rid
    summary["run_dir"] = str(_rdir)

    state_path = _rdir / "state_run.json"
    if state_path.exists():
        summary["state_run_path"] = str(state_path)

    return summary
