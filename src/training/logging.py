# -*- coding: utf-8 -*-
"""
src.training.logging — RunPaths and artifact-writing utilities.

Responsible for:
- Resolving OUTPUT_BASE from environment.
- Generating or reading RUN_ID.
- Creating canonical subdirectories (plots, tables, models, logs).
- Writing ``_last_run.txt`` pointer.
- Artifact-writer helpers (:meth:`RunPaths.write_json`,
  :meth:`RunPaths.write_table`, :meth:`RunPaths.write_text`).
- Writing the official ``state_run.json`` (see :func:`write_state_run`).

Commit: refactor(step1) → refactor(core): RunPaths class.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from src.config.defaults import (
    DECODER_LOGVAR_CLAMP_HI,
    DECODER_LOGVAR_CLAMP_LO,
)


# =====================================================================
# RunPaths — single source of truth for run-directory layout
# =====================================================================

class RunPaths:
    """Canonical run directory paths with artifact-writing helpers.

    Attributes
    ----------
    run_id : str
    run_dir : Path
    plots_dir : Path
    tables_dir : Path
    models_dir : Path
    logs_dir : Path

    The five subdirectories are created automatically on construction
    (unless ``_mkdir=False``).
    """

    __slots__ = ("run_id", "run_dir", "plots_dir", "tables_dir",
                 "models_dir", "logs_dir")

    def __init__(
        self,
        run_id: str,
        run_dir: Union[str, Path],
        *,
        _mkdir: bool = True,
    ) -> None:
        self.run_id: str = run_id
        self.run_dir: Path = Path(run_dir)
        self.plots_dir: Path = self.run_dir / "plots"
        self.tables_dir: Path = self.run_dir / "tables"
        self.models_dir: Path = self.run_dir / "models"
        self.logs_dir: Path = self.run_dir / "logs"
        if _mkdir:
            for p in (self.run_dir, self.plots_dir, self.tables_dir,
                      self.models_dir, self.logs_dir):
                p.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------

    def _resolve(self, filename: Union[str, Path]) -> Path:
        """Resolve *filename* relative to *run_dir*.  Creates parents."""
        p = Path(filename)
        if not p.is_absolute():
            p = self.run_dir / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ----------------------------------------------------------
    # Artifact writers
    # ----------------------------------------------------------

    def write_json(
        self,
        filename: Union[str, Path],
        obj: Any,
        *,
        indent: int = 2,
    ) -> Path:
        """Pretty-print *obj* as JSON inside the run directory.

        Parameters
        ----------
        filename : str or Path
            Relative to *run_dir* (e.g. ``"logs/dry_run.json"``).
        obj : any JSON-serialisable object
        indent : int
            JSON indentation (default 2).

        Returns
        -------
        Path
            The written file path.
        """
        p = self._resolve(filename)
        p.write_text(
            json.dumps(obj, indent=indent, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        return p

    def write_table(
        self,
        filename: Union[str, Path],
        df: "pd.DataFrame",
        **kwargs: Any,
    ) -> Path:
        """Write a DataFrame as CSV or single-sheet Excel.

        The format is inferred from the file extension:
        ``.csv`` → ``to_csv``, ``.xlsx`` / ``.xls`` → ``to_excel``.

        Parameters
        ----------
        filename : str or Path
            Relative to *run_dir* (e.g. ``"tables/inventory.xlsx"``).
        df : pandas.DataFrame
        **kwargs
            Forwarded to the underlying pandas writer.
            ``index`` defaults to ``False`` if not specified.

        Returns
        -------
        Path
            The written file path.
        """
        p = self._resolve(filename)
        kwargs.setdefault("index", False)
        suffix = p.suffix.lower()
        if suffix == ".csv":
            df.to_csv(p, **kwargs)
        elif suffix in (".xlsx", ".xls"):
            df.to_excel(p, **kwargs)
        else:
            df.to_csv(p, **kwargs)
        return p

    def write_text(
        self,
        filename: Union[str, Path],
        text: str,
        *,
        encoding: str = "utf-8",
    ) -> Path:
        """Write raw text inside the run directory.

        Parameters
        ----------
        filename : str or Path
            Relative to *run_dir*.
        text : str

        Returns
        -------
        Path
            The written file path.
        """
        p = self._resolve(filename)
        p.write_text(text, encoding=encoding)
        return p

    # ----------------------------------------------------------
    # Convenience constructors
    # ----------------------------------------------------------

    @classmethod
    def from_existing(cls, run_dir: Union[str, Path]) -> "RunPaths":
        """Wrap an existing run directory (run_id = dir name)."""
        run_dir = Path(run_dir)
        return cls(run_id=run_dir.name, run_dir=run_dir)

    def __repr__(self) -> str:
        return f"RunPaths(run_id={self.run_id!r}, run_dir={self.run_dir})"


# =====================================================================
# bootstrap_run — create a new timestamped run
# =====================================================================

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
        Instance with *run_id*, *run_dir*, *plots_dir*,
        *tables_dir*, *models_dir*, *logs_dir* — directories
        already created on disk.
    """
    if output_base is None:
        output_base = Path(os.environ.get("OUTPUT_BASE", "/workspace/2026/outputs"))
    else:
        output_base = Path(output_base)

    if run_id is None:
        run_id = os.environ.get("RUN_ID", "").strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    rp = RunPaths(run_id=run_id, run_dir=output_base / run_id)

    # Pointer file so that evaluation can find the latest run.
    try:
        (output_base / "_last_run.txt").write_text(str(rp.run_dir), encoding="utf-8")
    except Exception:
        pass

    return rp


def ensure_artifact_subdirs(
    base_dir: Union[str, Path],
    groups: Sequence[str],
) -> Dict[str, Path]:
    """Create and return named subdirectories under *base_dir*.

    This is used to keep large plot bundles grouped by purpose rather than
    writing every PNG into a single flat directory.
    """
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for group in groups:
        p = root / str(group)
        p.mkdir(parents=True, exist_ok=True)
        out[str(group)] = p
    return out


def write_artifact_manifest(
    base_dir: Union[str, Path],
    *,
    title: str,
    sections: Mapping[str, Sequence[Union[str, Path]]],
) -> Path:
    """Write a lightweight text index for grouped artifact bundles."""
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)

    lines = [title, ""]
    for section, paths in sections.items():
        rels = []
        for item in paths:
            p = Path(item)
            rels.append(str(p.relative_to(root)) if p.is_absolute() else str(p))
        if not rels:
            continue
        lines.append(f"[{section}]")
        for rel in rels:
            lines.append(f"- {rel}")
        lines.append("")

    manifest_path = root / "README.txt"
    manifest_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return manifest_path


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
        "model_constraints": {
            "decoder_logvar_clamp_lo": DECODER_LOGVAR_CLAMP_LO,
            "decoder_logvar_clamp_hi": DECODER_LOGVAR_CLAMP_HI,
            "clamp_origin": "empirical q1pct-1nat / q99pct+1nat, 27 VLC regimes",
        },
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
