# -*- coding: utf-8 -*-
"""
Protocol runner — reproducible baseline + cVAE evaluation across regimes.

Orchestrates training and evaluation per regime defined in a protocol JSON,
then consolidates results into a summary table.

No architecture, loss, or metrics changes.  Only orchestration.

Usage
-----
    python -m src.protocol.run \\
        --dataset_root data/dataset_fullsquare_organized \\
        --output_base  outputs \\
        [--protocol configs/all_regimes_sel4curr.json]

    # Quick reduced smoke-test (12 regimes, 1 grid, 2 epochs):
    python -m src.protocol.run \\
        --dataset_root data/dataset_fullsquare_organized \\
        --output_base  outputs \\
        --protocol configs/all_regimes_sel4curr.json \\
        --train_once_eval_all \\
        --max_epochs 2 --max_grids 1 --max_experiments 1

    # Dry-run (no training, just validate the protocol):
    python -m src.protocol.run \\
        --dataset_root data/dataset_fullsquare_organized \\
        --output_base  outputs \\
        --dry_run

Commit 3J.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from src.config.overrides import RunOverrides
from src.config.gpu_guard import warn_if_no_gpu_and_confirm
from src.config.runtime_env import ensure_writable_mpl_config_dir
from src.evaluation.validation_summary import (
    build_protocol_leaderboard,
    build_residual_signature_amplitude_table,
    build_residual_signature_table,
    build_stat_acceptance_summary,
    build_stat_fidelity_table,
    build_validation_summary_table,
)
from src.protocol.experiment_tracking import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    RUN_TYPE_PROTOCOL,
    write_latest_completed_experiment_record,
)
from src.training.logging import RunPaths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run a reproducible protocol (train + evaluate) across VLC regimes."
    )
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--output_base", type=str, required=True)
    p.add_argument("--protocol", type=str, default=None,
                   help="Path to protocol JSON (default: bundled reduced 12-regime protocol)")
    p.add_argument("--protocol_config", type=str, default=None,
                   help="Path to protocol YAML config (takes precedence over --protocol)")
    # --- global overrides (applied to every regime; override protocol JSON) ---
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--max_grids", type=int, default=None)
    p.add_argument("--max_regimes", type=int, default=None,
                   help="Limit the number of regimes executed after protocol resolution")
    p.add_argument("--grid_group", type=str, default=None)
    p.add_argument("--grid_tag", type=str, default=None)
    p.add_argument("--grid_preset", type=str, default=None,
                   help="Named grid subset, e.g. exploratory_small or residual_small (default: all)")
    p.add_argument("--max_experiments", type=int, default=None)
    p.add_argument("--max_samples_per_exp", type=int, default=None)
    p.add_argument("--val_split", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--patience", type=int, default=None,
                   help="Early stopping patience after KL warmup (default: use TRAINING_CONFIG)")
    p.add_argument("--reduce_lr_patience", type=int, default=None,
                   help="ReduceLROnPlateau patience (default: use TRAINING_CONFIG)")
    p.add_argument("--psd_nfft", type=int, default=None)
    p.add_argument("--train_regime_diagnostics_enabled", type=int, choices=[0, 1], default=None,
                   help="Enable periodic per-regime train diagnostics callback (default: 1)")
    p.add_argument("--train_regime_diagnostics_every", type=int, default=None,
                   help="Epoch cadence for periodic per-regime train diagnostics (default: 10)")
    p.add_argument("--train_regime_diagnostics_mc_samples", type=int, default=None,
                   help="MC samples for periodic per-regime train diagnostics (default: 4)")
    p.add_argument("--train_regime_diagnostics_max_samples_per_regime", type=int, default=None,
                   help="Max validation samples per regime for periodic train diagnostics (default: 4096)")
    p.add_argument("--train_regime_diagnostics_amplitude_bins", type=int, default=None,
                   help="Amplitude bins for residual signature diagnostics (default: 4)")
    p.add_argument("--train_regime_diagnostics_focus_only_0p8m", type=int, choices=[0, 1], default=None,
                   help="Restrict periodic train diagnostics to 0.8m regimes only (default: 0)")
    p.add_argument("--keras_verbose", type=int, default=2, choices=[0, 1, 2],
                   help="Keras fit verbosity: 0=silent, 1=progress bar, 2=one line/epoch (default: 2)")
    p.add_argument("--max_dist_samples", type=int, default=None,
                   help="Max validation samples for distribution metrics (default: 200000)")
    p.add_argument("--gauss_alpha", type=float, default=None,
                   help="Significance level for Gaussianity test (default: 0.01)")
    p.add_argument("--dist_tol_m", type=float, default=None,
                   help="Distance tolerance in metres for regime experiment filtering (default: 0.05)")
    p.add_argument("--curr_tol_mA", type=float, default=None,
                   help="Current tolerance in mA for regime experiment filtering (default: 25)")
    p.add_argument("--no_baseline", action="store_true",
                   help="Skip deterministic baseline (default: run baseline before cVAE)")
    p.add_argument("--no_cvae", action="store_true",
                   help="Skip cVAE training/evaluation and run only the baseline path")
    p.add_argument("--baseline_only", action="store_true",
                   help="Alias for --no_cvae")
    p.add_argument("--no_dist_metrics", action="store_true",
                   help="Skip distribution-fidelity metrics (moments, PSD, Gaussianity)")
    p.add_argument("--no_data_reduction", action="store_true",
                   help="Disable data reduction (balanced_blocks); use all train samples after split")
    p.add_argument("--skip_eval", action="store_true",
                   help="Run training only, skip evaluation step")
    p.add_argument("--dry_run", action="store_true",
                   help="Validate protocol + build model summary, no training")
    p.add_argument(
        "--train_once_eval_all",
        action="store_true",
        help=(
            "Train one global cVAE on all selected data, then evaluate that same model "
            "across all regimes without per-regime retraining"
        ),
    )
    p.add_argument(
        "--reuse_model_run_dir",
        type=str,
        default=None,
        help=(
            "Reuse an existing shared model run directory (must contain "
            "models/best_model_full.keras) and skip global retraining. "
            "Intended for protocol re-evaluation of a previously trained winner."
        ),
    )
    # --- Statistical Fidelity Suite (Etapa A2) ---
    p.add_argument("--stat_tests", action="store_true",
                   help="Run two-sample statistical tests (MMD, Energy, PSD) per regime")
    p.add_argument("--stat_mode", type=str, default="quick", choices=["quick", "full"],
                   help="quick = fewer permutations (200); full = 2000 (default: quick)")
    p.add_argument("--stat_n_perm", type=int, default=None,
                   help="Explicit number of permutations (overrides --stat_mode default)")
    p.add_argument("--stat_seed", type=int, default=42,
                   help="RNG seed for stat tests (default: 42)")
    p.add_argument("--stat_max_n", type=int, default=None,
                   help="Max validation samples for stat tests (default: 5000 in quick, 50000 in full)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_regimes(dataset_root: str) -> List[dict]:
    """Scan *dataset_root* for ``dist_<X>m/curr_<Y>mA/`` directories.

    Returns a sorted list of regime dicts (by distance then current),
    each containing ``regime_id``, ``description``, ``distance_m``,
    ``current_mA``, and study tags ready for the protocol runner.

    The naming convention ``dist_<X>m/curr_<Y>mA`` is the only
    assumption — every pair found becomes a regime.  This makes the
    protocol fully dataset-agnostic.
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_root not found: {root}")

    _dist_re = re.compile(r"^dist_([\d.]+)m$")
    _curr_re = re.compile(r"^curr_(\d+)mA$")

    points: List[tuple] = []  # (distance_m, current_mA)
    for dist_dir in sorted(root.iterdir()):
        if not dist_dir.is_dir():
            continue
        dm = _dist_re.match(dist_dir.name)
        if dm is None:
            continue
        dist_val = float(dm.group(1))
        for curr_dir in sorted(dist_dir.iterdir()):
            if not curr_dir.is_dir():
                continue
            cm = _curr_re.match(curr_dir.name)
            if cm is None:
                continue
            curr_val = float(cm.group(1))
            points.append((dist_val, curr_val))

    # Deduplicate & sort deterministically
    points = sorted(set(points))

    if not points:
        raise RuntimeError(
            f"No dist_<X>m/curr_<Y>mA directories found under {root}. "
            "Check the dataset layout."
        )

    regimes: List[dict] = []
    for dist_val, curr_val in points:
        regimes.append({
            "regime_id": make_regime_id(dist_val, curr_val),
            "description": f"{dist_val} m / {int(curr_val)} mA",
            "distance_m": dist_val,
            "current_mA": curr_val,
            "_study": "within_regime",
            "_split_strategy": "per_experiment",
        })

    return regimes


def _build_discovered_protocol(dataset_root: str) -> dict:
    """Build a complete protocol dict from filesystem discovery.

    Equivalent to loading a config file, but no config needed — regimes
    are enumerated by scanning ``dataset_root`` for the standard
    ``dist_<X>m/curr_<Y>mA/`` layout.
    """
    regimes = discover_regimes(dataset_root)
    regime_ids = [r["regime_id"] for r in regimes]

    # Quality-gate: log first 3 discovered regimes
    print(f"🔍 Auto-discovered {len(regimes)} regime(s) from dataset:")
    for r in regimes[:3]:
        print(f"   • {r['regime_id']}  ({r['distance_m']} m, {r['current_mA']} mA)")
    if len(regimes) > 3:
        print(f"   … and {len(regimes) - 3} more")

    return {
        "protocol_version": "1.0",
        "description": "Auto-discovered from dataset directory structure.",
        "global_settings": {},
        "regimes": regimes,
        "_studies": [{
            "name": "within_regime",
            "split_strategy": "per_experiment",
            "regime_ids": regime_ids,
        }],
    }


def _load_protocol(path: Optional[str]) -> dict:
    """Load protocol JSON, falling back to the bundled default when requested."""
    if path is None:
        # try repo-relative default
        candidates = [
            Path("configs/protocol_default.json"),
            Path(__file__).resolve().parent.parent.parent / "configs" / "protocol_default.json",
        ]
        for c in candidates:
            if c.exists():
                path = str(c)
                break
        if path is None:
            raise FileNotFoundError(
                "No --protocol given and configs/protocol_default.json not found."
            )
    proto = json.loads(Path(path).read_text(encoding="utf-8"))

    # Legacy/default format: explicit regimes list.
    if "regimes" in proto and proto["regimes"]:
        # Tag regimes with implicit within_regime study (Commit 3X)
        return _ensure_studies(proto)

    # Alternative JSON format (aligned with YAML): studies + regime_ids/selectors.
    if "studies" in proto and proto["studies"]:
        return _protocol_from_studies_json(proto)

    raise ValueError(
        "Protocol JSON must contain a non-empty 'regimes' list "
        "or a non-empty 'studies' list."
    )


def _limit_protocol_regimes(protocol: dict, max_regimes: Optional[int]) -> dict:
    """Return a copy of *protocol* restricted to the first *max_regimes* regimes."""
    if max_regimes is None:
        return protocol

    n_keep = int(max_regimes)
    if n_keep <= 0:
        raise ValueError("--max_regimes must be > 0")

    regimes = list(protocol.get("regimes", []))
    if len(regimes) <= n_keep:
        return protocol

    kept_regimes = regimes[:n_keep]
    kept_ids = {r["regime_id"] for r in kept_regimes}

    limited = dict(protocol)
    limited["regimes"] = kept_regimes

    studies = []
    for study in protocol.get("_studies", []):
        study_ids = [rid for rid in study.get("regime_ids", []) if rid in kept_ids]
        if study_ids:
            study_copy = dict(study)
            study_copy["regime_ids"] = study_ids
            studies.append(study_copy)
    limited["_studies"] = studies
    return limited


def _should_run_cvae(*, no_cvae: bool = False, baseline_only: bool = False) -> bool:
    """Return ``True`` when the cVAE path should execute for a regime."""
    return not (bool(no_cvae) or bool(baseline_only))


def _effective_stat_max_n(stat_mode: str, stat_max_n: Optional[int]) -> int:
    """Resolve the effective sample cap for statistical fidelity tests."""
    if stat_max_n is not None:
        n_eff = int(stat_max_n)
        if n_eff <= 0:
            raise ValueError("--stat_max_n must be > 0")
        return n_eff
    return 5_000 if str(stat_mode).strip().lower() == "quick" else 50_000


_BASELINE_DEFAULTS = {
    "model": "deterministic_mlp",
    "hidden": [128, 64],
    "dropout": 0.0,
    "epochs": 50,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "verbose": 0,
    "loss": "mse",
}

_DIST_METRICS_DEFAULTS = {
    "psd_nfft": 2048,
    "gauss_alpha": 0.01,
    "max_dist_samples": 200_000,
}


def _override_dict(
    overrides: Optional[Union[RunOverrides, Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Return a plain dict regardless of the override representation."""
    if overrides is None:
        return {}
    if isinstance(overrides, RunOverrides):
        return overrides.to_dict()
    return dict(overrides)


def _effective_baseline_config(
    overrides: Optional[Union[RunOverrides, Mapping[str, Any]]],
    *,
    enabled: bool,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """Single source of truth for baseline runtime + manifest config."""
    ov = _override_dict(overrides)
    cfg = dict(_BASELINE_DEFAULTS)
    if ov.get("max_epochs") is not None:
        cfg["epochs"] = int(ov["max_epochs"])
    if ov.get("keras_verbose") is not None:
        cfg["verbose"] = int(ov["keras_verbose"])
    cfg["enabled"] = bool(enabled)
    cfg["return_predictions"] = bool(return_predictions)
    return cfg


def _effective_cvae_config(
    overrides: Optional[Union[RunOverrides, Mapping[str, Any]]],
    *,
    enabled: bool,
    execution_mode: str = "per_regime_retrain",
) -> Dict[str, Any]:
    """Expose the cVAE-relevant effective overrides in the manifest."""
    ov = _override_dict(overrides)
    cfg: Dict[str, Any] = {
        "enabled": bool(enabled),
        "execution_mode": str(execution_mode),
    }
    for key in (
        "max_epochs",
        "max_grids",
        "grid_group",
        "grid_tag",
        "grid_preset",
        "val_split",
        "seed",
        "patience",
        "reduce_lr_patience",
        "max_experiments",
        "max_samples_per_exp",
        "keras_verbose",
        "no_data_reduction",
    ):
        if ov.get(key) is not None:
            cfg[key] = ov[key]
    return cfg


def _protocol_execution_mode(*, train_once_eval_all: bool) -> str:
    return "train_once_eval_all" if bool(train_once_eval_all) else "per_regime_retrain"


def _effective_dist_metrics_config(
    overrides: Optional[Union[RunOverrides, Mapping[str, Any]]],
    *,
    enabled: bool,
) -> Dict[str, Any]:
    """Single source of truth for distribution-metric knobs."""
    ov = _override_dict(overrides)
    cfg = {
        "enabled": bool(enabled),
        "psd_nfft": _DIST_METRICS_DEFAULTS["psd_nfft"],
        "gauss_alpha": _DIST_METRICS_DEFAULTS["gauss_alpha"],
        "max_dist_samples": _DIST_METRICS_DEFAULTS["max_dist_samples"],
    }
    if ov.get("psd_nfft") is not None:
        cfg["psd_nfft"] = int(ov["psd_nfft"])
    if ov.get("gauss_alpha") is not None:
        cfg["gauss_alpha"] = float(ov["gauss_alpha"])
    if ov.get("max_dist_samples") is not None:
        cfg["max_dist_samples"] = int(ov["max_dist_samples"])
    return cfg


def _parse_regime_id_physical(rid: str) -> Tuple[float, float]:
    """Parse ``dist_...m__curr_...mA`` into ``(distance_m, current_mA)``."""
    m = re.match(r"^dist_([0-9p.]+)m__curr_([0-9]+)mA$", str(rid).strip())
    if m is None:
        raise ValueError(
            f"Invalid regime_id '{rid}'. Expected format dist_<D>m__curr_<C>mA"
        )
    dist = float(m.group(1).replace("p", "."))
    curr = float(m.group(2))
    return dist, curr


def _protocol_from_studies_json(raw: dict) -> dict:
    """Build canonical protocol dict from JSON ``studies`` format.

    Accepted per-study entries:
    - ``regime_ids`` (list of ``dist_...m__curr_...mA`` ids)
    - ``selectors``  (list with ``distance_m``/``current_mA``)
    """
    studies = list(raw.get("studies", []))
    regimes: List[dict] = []
    resolved_studies: List[dict] = []
    seen_ids: set = set()
    multi_study = len(studies) > 1

    for study in studies:
        sname = str(study.get("name", "within_regime"))
        split_strat = str(study.get("split_strategy", "per_experiment"))
        selectors = list(study.get("selectors", []) or [])
        regime_ids = list(study.get("regime_ids", []) or [])

        if not selectors and not regime_ids:
            raise ValueError(
                f"Study '{sname}' has no selectors nor regime_ids."
            )

        study_regime_ids: List[str] = []

        # Explicit selectors with physical values.
        for entry in selectors:
            dist = float(entry["distance_m"])
            curr = float(entry["current_mA"])
            rid = entry.get("regime_id") or make_regime_id(dist, curr)
            rid = make_regime_id(*_parse_regime_id_physical(rid)) if "distance_m" not in entry else rid
            full_rid = f"{sname}/{rid}" if multi_study else rid
            if full_rid in seen_ids:
                raise ValueError(f"Duplicate regime_id '{full_rid}' in studies JSON.")
            seen_ids.add(full_rid)

            desc = entry.get("description", f"{dist} m / {int(curr)} mA")
            regimes.append({
                "regime_id": full_rid,
                "regime_label": entry.get("regime_id", rid),
                "description": desc,
                "distance_m": dist,
                "current_mA": curr,
                "_study": sname,
                "_split_strategy": split_strat,
            })
            study_regime_ids.append(full_rid)

        # Compact form with only regime_ids (used by smoke tests).
        for rid_in in regime_ids:
            dist, curr = _parse_regime_id_physical(str(rid_in))
            rid = make_regime_id(dist, curr)
            full_rid = f"{sname}/{rid}" if multi_study else rid
            if full_rid in seen_ids:
                continue
            seen_ids.add(full_rid)
            regimes.append({
                "regime_id": full_rid,
                "regime_label": str(rid_in),
                "description": f"{dist} m / {int(curr)} mA",
                "distance_m": dist,
                "current_mA": curr,
                "_study": sname,
                "_split_strategy": split_strat,
            })
            study_regime_ids.append(full_rid)

        resolved_studies.append({
            "name": sname,
            "split_strategy": split_strat,
            "regime_ids": study_regime_ids,
        })

    return {
        "protocol_version": str(raw.get("protocol_version", "1.0")),
        "description": raw.get("description", ""),
        "global_settings": raw.get("global_settings", {}),
        "regimes": regimes,
        "_studies": resolved_studies,
    }


def _ensure_studies(proto: dict) -> dict:
    """Ensure protocol dict has ``_studies`` metadata.

    When the protocol was loaded from JSON (no ``_studies`` key),
    wraps all regimes into a single ``within_regime`` study with
    ``per_experiment`` split.

    Also overrides each regime's ``regime_id`` with the canonical
    physical ID (``dist_…m__curr_…mA``) so that folder names are
    always dataset-agnostic.  The original human-friendly id is
    kept as ``regime_label``.
    """
    if "_studies" in proto:
        return proto
    regimes = proto["regimes"]
    for r in regimes:
        # Derive physical ID; keep old name as label
        if r.get("distance_m") is not None and r.get("current_mA") is not None:
            old_id = r["regime_id"]
            r["regime_id"] = make_regime_id(r["distance_m"], r["current_mA"])
            r.setdefault("regime_label", old_id)
        r.setdefault("_study", "within_regime")
        r.setdefault("_split_strategy", "per_experiment")
    proto["_studies"] = [{
        "name": "within_regime",
        "split_strategy": "per_experiment",
        "regime_ids": [r["regime_id"] for r in regimes],
    }]
    return proto


def _fmt_number(x: float, nd: int) -> str:
    """Format *x* with *nd* decimal places, strip trailing fractional zeros, replace '.' with 'p'."""
    s = f"{x:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace(".", "p") if s else "0"


def make_regime_id(distance_m: float, current_mA: float) -> str:
    """Deterministic, filesystem-safe regime ID from physical operating point.

    Format: ``dist_{D}m__curr_{C}mA``

    Examples:
        >>> make_regime_id(0.8, 200)
        'dist_0p8m__curr_200mA'
        >>> make_regime_id(1.5, 800)
        'dist_1p5m__curr_800mA'
        >>> make_regime_id(1.0, 100.0)
        'dist_1m__curr_100mA'
    """
    d = _fmt_number(distance_m, 2)
    c = _fmt_number(current_mA, 0)
    return f"dist_{d}m__curr_{c}mA"


def _load_protocol_yaml(path: str) -> dict:
    """Load protocol config from YAML and return the same dict structure as _load_protocol.

    Supports two YAML layouts:

    1. **studies** (new — Commit 3X):
       Top-level ``studies`` list.  Each study has ``name``,
       ``split_strategy``, and ``selectors``.  Selectors are expanded
       into regime dicts with ``_study`` and ``_split_strategy`` tags.

    2. **regimes** (legacy — Commit 3U):
       Top-level ``regimes`` list, auto-wrapped into a single
       ``within_regime`` study.

    Returns the canonical protocol dict consumed by ``main()``.
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Protocol YAML '{path}' must be a mapping.")

    # --- Normalise to studies list ---
    raw_studies = raw.get("studies")
    if raw_studies:
        studies = list(raw_studies)
    elif raw.get("regimes"):
        # Legacy: bare regimes → wrap into a single study
        studies = [{
            "name": "within_regime",
            "split_strategy": "per_experiment",
            "selectors": list(raw["regimes"]),
        }]
    else:
        raise ValueError(
            f"Protocol YAML '{path}' must contain 'studies' or 'regimes'."
        )

    # --- Expand studies into flat regimes list (tagged) ---
    regimes: List[dict] = []
    seen_ids: set = set()
    resolved_studies: List[dict] = []

    for study in studies:
        sname = str(study.get("name", "default"))
        split_strat = str(study.get("split_strategy", "per_experiment"))
        selectors = study.get("selectors", [])
        if not selectors:
            raise ValueError(f"Study '{sname}' has no selectors.")

        study_regime_ids: List[str] = []
        for entry in selectors:
            dist = float(entry["distance_m"])
            curr = float(entry["current_mA"])

            rid = entry.get("regime_id") or make_regime_id(dist, curr)
            # Prefix with study name to avoid cross-study collisions
            full_rid = f"{sname}/{rid}" if len(studies) > 1 else rid
            if full_rid in seen_ids:
                raise ValueError(f"Duplicate regime_id '{full_rid}' in YAML config.")
            seen_ids.add(full_rid)

            desc = entry.get("description", f"{dist} m / {int(curr)} mA")

            regime: Dict[str, object] = {
                "regime_id": full_rid,
                "description": desc,
                "distance_m": dist,
                "current_mA": curr,
                "_study": sname,
                "_split_strategy": split_strat,
            }
            # Forward any extra per-selector keys
            for k, v in entry.items():
                if k not in regime:
                    regime[k] = v
            regimes.append(regime)
            study_regime_ids.append(full_rid)

        resolved_studies.append({
            "name": sname,
            "split_strategy": split_strat,
            "regime_ids": study_regime_ids,
        })

    proto = {
        "protocol_version": str(raw.get("protocol_version", "1.0")),
        "description": raw.get("description", ""),
        "global_settings": raw.get("global_settings", {}),
        "regimes": regimes,
        "_studies": resolved_studies,
    }
    return proto


def _merge_overrides(protocol_globals: dict, cli_args: argparse.Namespace) -> RunOverrides:
    """
    Build the overrides for a single regime by layering:
        protocol global_settings  <  CLI flags (explicit wins)

    Returns a :class:`RunOverrides` dataclass (call ``.to_dict()``
    when a plain dict is needed by legacy consumers).
    """
    cli_ov = RunOverrides.from_namespace(cli_args)
    return RunOverrides.merge(protocol_globals, cli_ov)


def _git_commit_hash() -> str:
    """Best-effort git rev-parse HEAD."""
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _runtime_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    try:
        import tensorflow as tf
        versions["tensorflow"] = tf.__version__
    except Exception:
        pass
    try:
        import numpy as np
        versions["numpy"] = np.__version__
    except Exception:
        pass
    return versions


def _read_eval_metrics(run_dir: Path) -> dict:
    """Read evaluation metrics JSON produced by the canonical eval engine."""
    p = run_dir / "logs" / "metricas_globais_reanalysis.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _extract_cvae_dist_from_eval_metrics(metrics: dict) -> dict:
    """Map eval global metrics to the cVAE dist-metrics schema used by protocol."""
    if not isinstance(metrics, dict) or not metrics:
        return {}

    def _f(v):
        try:
            if v is None:
                return None
            return float(v)
        except Exception:
            return None

    out = {
        "delta_mean_l2": _f(metrics.get("delta_mean_l2")),
        "delta_cov_fro": _f(metrics.get("delta_cov_fro")),
        "mean_real_delta_I": _f(metrics.get("mean_real_delta_I")),
        "mean_real_delta_Q": _f(metrics.get("mean_real_delta_Q")),
        "mean_pred_delta_I": _f(metrics.get("mean_pred_delta_I")),
        "mean_pred_delta_Q": _f(metrics.get("mean_pred_delta_Q")),
        "std_real_delta_I": _f(metrics.get("std_real_delta_I")),
        "std_real_delta_Q": _f(metrics.get("std_real_delta_Q")),
        "std_pred_delta_I": _f(metrics.get("std_pred_delta_I")),
        "std_pred_delta_Q": _f(metrics.get("std_pred_delta_Q")),
        "delta_mean_I": _f(metrics.get("delta_mean_I")),
        "delta_mean_Q": _f(metrics.get("delta_mean_Q")),
        "delta_std_I": _f(metrics.get("delta_std_I")),
        "delta_std_Q": _f(metrics.get("delta_std_Q")),
        "var_real_delta": _f(metrics.get("var_real_delta")),
        "var_pred_delta": _f(metrics.get("var_pred_delta")),
        "delta_skew_l2": _f(metrics.get("delta_skew_l2")),
        "delta_kurt_l2": _f(metrics.get("delta_kurt_l2")),
        "delta_skew_I": _f(metrics.get("delta_skew_I")),
        "delta_skew_Q": _f(metrics.get("delta_skew_Q")),
        "delta_kurt_I": _f(metrics.get("delta_kurt_I")),
        "delta_kurt_Q": _f(metrics.get("delta_kurt_Q")),
        "delta_wasserstein_I": _f(metrics.get("delta_wasserstein_I")),
        "delta_wasserstein_Q": _f(metrics.get("delta_wasserstein_Q")),
        # eval JSON uses delta_psd_l2 naming; protocol table expects psd_l2.
        "psd_l2": _f(metrics.get("delta_psd_l2", metrics.get("psd_l2"))),
        "delta_acf_l2": _f(metrics.get("delta_acf_l2")),
        # Optional JB fields (present after CORREÇÃO 3 propagation in eval metrics).
        "jb_stat_I": _f(metrics.get("jb_stat_I")),
        "jb_stat_Q": _f(metrics.get("jb_stat_Q")),
        "jb_p_I": _f(metrics.get("jb_p_I")),
        "jb_p_Q": _f(metrics.get("jb_p_Q")),
        "jb_p_min": _f(metrics.get("jb_p_min")),
        "jb_log10p_I": _f(metrics.get("jb_log10p_I")),
        "jb_log10p_Q": _f(metrics.get("jb_log10p_Q")),
        "jb_log10p_min": _f(metrics.get("jb_log10p_min")),
        "delta_jb_log10p_I": _f(metrics.get("delta_jb_log10p_I")),
        "delta_jb_log10p_Q": _f(metrics.get("delta_jb_log10p_Q")),
        "delta_jb_stat_rel_I": _f(metrics.get("delta_jb_stat_rel_I")),
        "delta_jb_stat_rel_Q": _f(metrics.get("delta_jb_stat_rel_Q")),
        "jb_real_p_min": _f(metrics.get("jb_real_p_min")),
        "jb_real_log10p_I": _f(metrics.get("jb_real_log10p_I")),
        "jb_real_log10p_Q": _f(metrics.get("jb_real_log10p_Q")),
        "jb_real_log10p_min": _f(metrics.get("jb_real_log10p_min")),
        "reject_gaussian": (bool(metrics.get("reject_gaussian"))
                            if metrics.get("reject_gaussian") is not None else None),
        "jb_real_reject_gaussian": (
            bool(metrics.get("jb_real_reject_gaussian"))
            if metrics.get("jb_real_reject_gaussian") is not None
            else None
        ),
    }
    # Require at least one core distance metric to consider mapping valid.
    _core = ("delta_mean_l2", "delta_cov_fro", "delta_skew_l2", "delta_kurt_l2", "psd_l2", "delta_acf_l2")
    if all(out.get(k) is None for k in _core):
        return {}
    return out


def _read_train_state(run_dir: Path) -> dict:
    """Read state_run.json produced by the training monolith."""
    p = run_dir / "state_run.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _resolve_reuse_model_run_dir(path_str: Optional[str]) -> Optional[Path]:
    """Validate and resolve a reusable shared-model run directory."""
    if path_str is None or str(path_str).strip() == "":
        return None

    run_dir = Path(path_str).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"--reuse_model_run_dir not found: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"--reuse_model_run_dir is not a directory: {run_dir}")

    model_path = run_dir / "models" / "best_model_full.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"--reuse_model_run_dir must contain models/best_model_full.keras: {model_path}"
        )
    return run_dir


def _extract_best_grid_tag(state: dict) -> str:
    """Extract best grid tag from the training state if available."""
    try:
        import pandas as pd

        artifacts = state.get("artifacts", {})
        csv_path = artifacts.get("grid_results_csv", "")
        if csv_path and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            if len(df) > 0 and "tag" in df.columns:
                return str(df.iloc[0]["tag"])

        xlsx_path = artifacts.get("grid_results_xlsx", "")
        if xlsx_path and Path(xlsx_path).exists():
            df = pd.read_excel(xlsx_path, sheet_name="results_sorted")
            if len(df) > 0 and "tag" in df.columns:
                return str(df.iloc[0]["tag"])
    except Exception:
        pass
    return ""


def _extract_training_operational_artifacts(run_dir: Optional[Path]) -> Dict[str, Optional[str]]:
    """Resolve compact training-operational artifacts from a training run dir."""
    if run_dir is None:
        return {
            "grid_training_diagnostics_csv": None,
            "training_dashboard_png": None,
        }

    state = _read_train_state(run_dir)
    artifacts = state.get("artifacts", {}) if isinstance(state, dict) else {}

    diag_path = artifacts.get("grid_training_diagnostics_csv")
    dash_path = artifacts.get("training_dashboard_png")

    if not diag_path:
        candidate = run_dir / "tables" / "grid_training_diagnostics.csv"
        diag_path = str(candidate) if candidate.exists() else None
    if not dash_path:
        candidate = run_dir / "plots" / "training" / "dashboard_analysis_complete.png"
        dash_path = str(candidate) if candidate.exists() else None

    return {
        "grid_training_diagnostics_csv": diag_path,
        "training_dashboard_png": dash_path,
    }


def _protocol_manifest_args(
    args: argparse.Namespace,
    *,
    proto_config_path: Optional[str],
    reused_model_run_dir: Optional[Path],
) -> Dict[str, Any]:
    return {
        "dataset_root": args.dataset_root,
        "output_base": args.output_base,
        "protocol": args.protocol,
        "protocol_config": proto_config_path,
        "max_regimes": args.max_regimes,
        "skip_eval": args.skip_eval,
        "no_baseline": args.no_baseline,
        "no_cvae": args.no_cvae,
        "baseline_only": args.baseline_only,
        "no_dist_metrics": args.no_dist_metrics,
        "dry_run": args.dry_run,
        "stat_tests": args.stat_tests,
        "train_once_eval_all": args.train_once_eval_all,
        "reuse_model_run_dir": (
            str(reused_model_run_dir) if reused_model_run_dir is not None else None
        ),
    }


def _write_protocol_running_manifest(
    exp_paths: RunPaths,
    *,
    ts_start: datetime,
    git_commit: Optional[str],
    versions: Dict[str, Any],
    args_payload: Dict[str, Any],
    execution_mode: str,
    studies_meta: List[dict],
    regimes: List[dict],
) -> Path:
    payload = {
        "run_type": RUN_TYPE_PROTOCOL,
        "run_status": RUN_STATUS_RUNNING,
        "timestamp_start": ts_start.isoformat(timespec="seconds"),
        "timestamp_end": None,
        "duration_seconds": None,
        "git_commit": git_commit,
        "versions": versions,
        "args": args_payload,
        "execution_mode": execution_mode,
        "n_studies": len(studies_meta),
        "studies": [
            {
                "name": s["name"],
                "split_strategy": s.get("split_strategy", "per_experiment"),
                "n_regimes": len(s["regime_ids"]),
                "regime_ids": s["regime_ids"],
            }
            for s in studies_meta
        ],
        "n_regimes": len(regimes),
    }
    return exp_paths.write_json("manifest.json", payload)


def _filter_experiments_for_regime(
    exps: list,
    regime: dict,
    dist_tol_m: float = 0.05,
    curr_tol_mA: float = 25.0,
) -> list:
    """
    Filter loaded experiments to those matching the regime's distance/current.

    Thin wrapper around :func:`selector_engine.select_experiments` so that
    existing call-sites in *run_regime* keep working unchanged.

    Commit 3V: delegates to selector_engine.
    """
    from src.protocol.selector_engine import select_experiments

    selector = {
        "distance_m": regime.get("distance_m"),
        "current_mA": regime.get("current_mA"),
    }
    return select_experiments(
        inventory=exps,
        selector=selector,
        dist_tol=dist_tol_m,
        curr_tol=curr_tol_mA,
        label=regime.get("regime_id", "?"),
    )


def _quick_cvae_predict(
    run_dir: Path,
    X_va,
    D_va,
    C_va,
    batch_size: int = 4096,
    mc_samples: int = 16,
    seed: int = 42,
    mode: str = "mc_concat",
    df_split=None,
):
    """
    Load best_model_full.keras from *run_dir*/models and generate cVAE predictions.

    ``mode="mc_concat"`` is the default for distribution-oriented metrics
    (MMD², Energy, Δcov, Δkurt), which must use samples from the marginal
    predictive distribution:

        y ~ p(y | z, x, d, c),  z ~ p(z | x, d, c)

    Therefore this helper returns the concatenation of *mc_samples* stochastic
    draws (not their average), along with tiled conditioning arrays so callers
    can compute residuals on matched shapes.

    ``mode="det"`` is reserved for point metrics (for example EVM/SNR), using
    MAP-like inference ``z = mu_prior`` and ``y = mu_decoder``.

    Returns
    -------
    tuple or None
        ``mode="mc_concat"`` -> ``(Y_pred_concat, X_tiled, D_tiled, C_tiled)``
        ``mode="det"``       -> ``(Y_pred, X_va, D_va, C_va)``
        ``None`` on failure.
    """
    import numpy as _np

    model_path = run_dir / "models" / "best_model_full.keras"
    if not model_path.exists():
        print(f"⚠️  best_model_full.keras not found at {model_path}")
        return None
    print(f"   📂 Loading model: {model_path}")

    # --- Shape guard ---
    for name, arr in [("X_va", X_va), ("D_va", D_va), ("C_va", C_va)]:
        assert arr is not None, f"_quick_cvae_predict: {name} is None"
    n = X_va.shape[0]
    assert D_va.shape[0] == n and C_va.shape[0] == n, (
        f"Shape mismatch: X_va={X_va.shape}, D_va={D_va.shape}, C_va={C_va.shape}"
    )
    mc_samples = max(1, int(mc_samples))
    mode = str(mode).strip().lower()
    if mode not in {"mc_concat", "det"}:
        raise ValueError(f"Unsupported _quick_cvae_predict mode: {mode}")

    try:
        import tensorflow as tf
        from src.models.cvae import create_inference_model_from_full
        from src.models.losses import CondPriorVAELoss
        from src.models.sampling import Sampling

        from src.models.cvae_sequence import load_seq_model
        vae = load_seq_model(str(model_path))

        # Detect seq_bigru_residual via prior input rank (rank-3 → windowed input)
        _is_seq = len(vae.get_layer("prior_net").inputs[0].shape) == 3

        # --- Normalizar D e C antes de alimentar o modelo ---
        # (modelo foi treinado com D,C em [0,1])
        # Usa apply_condition_norm para reproduzir a lógica exata do treino,
        # incluindo o fallback D_max==D_min → 0.0, C_max==C_min → 0.5.
        from src.data.normalization import apply_condition_norm
        D_arr = _np.asarray(D_va)
        C_arr = _np.asarray(C_va)
        try:
            _state = json.loads((run_dir / "state_run.json").read_text())
            _norm = _state.get("normalization", {})
            _norm_params = {
                "D_min": float(_norm.get("D_min", D_arr.min())),
                "D_max": float(_norm.get("D_max", D_arr.max())),
                "C_min": float(_norm.get("C_min", C_arr.min())),
                "C_max": float(_norm.get("C_max", C_arr.max())),
            }
        except Exception:
            # fallback: normalizar pelo próprio batch (menos preciso)
            _norm_params = {
                "D_min": float(D_arr.min()), "D_max": float(D_arr.max()),
                "C_min": float(C_arr.min()), "C_max": float(C_arr.max()),
            }

        _D_norm, _C_norm = apply_condition_norm(D_arr.ravel(), C_arr.ravel(), _norm_params)
        _D_norm = _D_norm.reshape(-1, 1)
        _C_norm = _C_norm.reshape(-1, 1)

        X_arr = _np.asarray(X_va)
        X_center_arr = X_arr  # (N, 2) — center frame for residuals and tiling

        # --- Seq windowing ---
        # Uses per-experiment boundaries when df_split is available (no boundary leakage).
        # Falls back to global windowing only when df_split is absent (emergency fallback).
        if _is_seq:
            from src.data.windowing import build_windows_single_experiment
            _ws_shape = int(vae.get_layer("prior_net").inputs[0].shape[1])
            try:
                _state_qs = json.loads((run_dir / "state_run.json").read_text())
                _ds_qs = _state_qs.get("data_split", {})
                _ws = int(_ds_qs.get("window_size", _ws_shape))
                _wst = int(_ds_qs.get("window_stride", 1))
                _wpm = str(_ds_qs.get("window_pad_mode", "edge"))
            except Exception:
                _ws, _wst, _wpm = _ws_shape, 1, "edge"

            _df = df_split
            if _df is not None and "n_val" in _df.columns:
                # Per-experiment windowing — no cross-experiment boundary leakage
                _n_val_list = [int(v) for v in _df["n_val"].tolist()]
                _va_X = []
                _cursor = 0
                for _n_va in _n_val_list:
                    if _n_va > 0:
                        _Y_d = _np.zeros((_n_va, 2), dtype=_np.float32)
                        _D_d = _np.ones((_n_va, 1), dtype=_np.float32)
                        _C_d = _np.ones((_n_va, 1), dtype=_np.float32)
                        _X_w, _, _, _ = build_windows_single_experiment(
                            X_arr[_cursor:_cursor + _n_va],
                            _Y_d, _D_d, _C_d,
                            window_size=_ws, stride=_wst, pad_mode=_wpm,
                        )
                        _va_X.append(_X_w)
                    _cursor += _n_va
                X_arr_w = _np.concatenate(_va_X, axis=0) if _va_X else _np.empty((0, _ws, 2), dtype=_np.float32)
                print(f"   🔄 seq windowing (per-experiment): X_arr_w={X_arr_w.shape}")
            else:
                # Global windowing — windows at experiment boundaries may include context
                # from adjacent experiments. Only reached when df_split is unavailable.
                print("   ⚠️  seq windowing: df_split unavailable — using global windowing "
                      "(windows may cross experiment boundaries).")
                Y_dummy = _np.zeros((X_arr.shape[0], 2), dtype=_np.float32)
                D_dummy = _np.ones((X_arr.shape[0], 1), dtype=_np.float32)
                C_dummy = _np.ones((X_arr.shape[0], 1), dtype=_np.float32)
                X_arr_w, _, _, _ = build_windows_single_experiment(
                    X_arr, Y_dummy, D_dummy, C_dummy,
                    window_size=_ws, stride=_wst, pad_mode=_wpm,
                )
                print(f"   🔄 seq windowing (global): X_arr_w={X_arr_w.shape}")

            X_center_arr = X_arr  # keep 2D for tiling
            X_arr = X_arr_w        # switch to windowed for inference

        if mode == "det":
            inference_model = create_inference_model_from_full(vae, deterministic=True)
            Y_pred = inference_model.predict(
                [X_arr, _D_norm, _C_norm], batch_size=batch_size, verbose=0
            )
            del inference_model, vae
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass
            return Y_pred, X_center_arr, D_arr, C_arr

        samples = []
        for i in range(mc_samples):
            # Rebuild the stochastic inference graph each draw so each sample is
            # an independent realization of p(y | x, d, c).
            tf.random.set_seed(int(seed) + i)
            inference_model = create_inference_model_from_full(vae, deterministic=False)
            s = inference_model.predict(
                [X_arr, _D_norm, _C_norm], batch_size=batch_size, verbose=0
            )
            samples.append(s)
            del inference_model
        Y_pred_concat = _np.concatenate(samples, axis=0)

        def _tile_like_input(a):
            a = _np.asarray(a)
            if a.ndim == 1:
                return _np.tile(a, mc_samples)
            reps = (mc_samples,) + (1,) * (a.ndim - 1)
            return _np.tile(a, reps)

        X_tiled = _tile_like_input(X_center_arr)  # tile 2D center frame for residual calcs
        D_tiled = _tile_like_input(D_arr)
        C_tiled = _tile_like_input(C_arr)

        del vae, samples
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        return Y_pred_concat, X_tiled, D_tiled, C_tiled
    except Exception as e:
        print(f"⚠️  _quick_cvae_predict failed (mode={mode}): {e}")
        return None


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_regime(
    regime: dict,
    dataset_root: str,
    base_overrides: Union[RunOverrides, Mapping[str, Any]],
    protocol_dir: Path,
    logs_root: Optional[Path] = None,
    shared_model_run_dir: Optional[Path] = None,
    run_cvae: bool = True,
    skip_eval: bool = False,
    run_baseline: bool = True,
    run_dist_metrics: bool = True,
    run_stat_fidelity: bool = False,
    stat_mode: str = "quick",
    stat_n_perm: Optional[int] = None,
    stat_seed: int = 42,
    stat_max_n: int = 50_000,
) -> dict:
    """
    Execute train + evaluate for one regime.

    Regime artefacts are written to ``protocol_dir / regime_id``.
    When ``shared_model_run_dir`` is provided, the cVAE is not retrained for this
    regime; evaluation uses the shared model and writes outputs to the regime dir.
    Returns a result dict with run_dir, metrics, and status.
    """
    regime_id = regime["regime_id"]
    run_id = str(regime_id).split("/", 1)[-1]   # lives under protocol_dir/<run_id>/
    model_run_dir = Path(shared_model_run_dir).resolve() if shared_model_run_dir is not None else None

    # --- Resolve experiment filter ---
    # Option A: explicit experiment_paths
    # Option B: experiment_regex
    exp_paths = regime.get("experiment_paths", [])
    exp_regex = regime.get("experiment_regex", None)

    # Build per-regime overrides
    ov = _override_dict(base_overrides)
    if logs_root is not None:
        ov["_logs_dir"] = str((Path(logs_root) / run_id).resolve())

    # If regime specifies experiment_paths, we filter via max_experiments = len
    # AND set the DATASET_ROOT to the parent so only those experiments are found.
    # However, the current loader discovers ALL experiments under DATASET_ROOT.
    # We handle filtering by passing experiment_paths in overrides for
    # future use, but for now we rely on max_experiments for bounding.
    if exp_paths:
        ov["_experiment_paths"] = exp_paths
    if exp_regex:
        ov["_experiment_regex"] = exp_regex

    _split_strategy = str(regime.get("_split_strategy", "per_experiment")).strip() or "per_experiment"
    ov["_split_strategy"] = _split_strategy

    result = {
        "regime_id": regime_id,
        "regime_label": regime.get("regime_label", ""),
        "description": regime.get("description", ""),
        "split_strategy": _split_strategy,
        "run_id": run_id,
        "run_dir": None,
        "model_run_dir": str(model_run_dir) if model_run_dir is not None else None,
        "model_scope": "shared_global" if model_run_dir is not None else "per_regime",
        "train_status": "skipped",
        "eval_status": "skipped",
        "metrics": {},
        "best_grid_tag": "",
        "baseline": {},
        "baseline_time_s": 0.0,
        "cvae_time_s": 0.0,
        "baseline_dist": {},
        "cvae_dist": {},
        "residual_signature": {},
        "residual_signature_bins": [],
        "dist_metrics_source": None,      # "eval_reanalysis" | "quick" | "quick_fallback" | None
        "stat_fidelity": {},               # Etapa A2: stat tests per regime
        "selected_experiments": [],        # Commit 3Q: paths of selected exps
        "selection_criteria": {},          # Commit 3Q: filter params used
        "error": None,
    }

    # ---- Commit 3R: compute selected experiments early ----
    # Needed by BOTH shared data loading AND training code.
    _sel_paths = []         # populated below
    _sel_criteria = {}      # populated below
    try:
        from src.data.loading import load_experiments_as_list as _leal
        _ds = Path(dataset_root)
        _dist_tol = float(ov.get("dist_tol_m", 0.05))
        _curr_tol = float(ov.get("curr_tol_mA", 25.0))
        _max_exp = ov.get("max_experiments")

        print(f"\n📦 Loading data from {_ds}")
        _all_exps, _ = _leal(_ds, verbose=False, reduction_config=None)
        _filt_exps = _filter_experiments_for_regime(
            _all_exps, regime, dist_tol_m=_dist_tol, curr_tol_mA=_curr_tol,
        )
        print(f"   🎯 After regime filter: {len(_filt_exps)} experiment(s) "
              f"(dist_tol={_dist_tol}, curr_tol={_curr_tol})")
        if _max_exp is not None:
            _filt_exps = _filt_exps[:int(_max_exp)]

        _sel_paths = [str(t[4]) for t in _filt_exps]
        _sel_criteria = {
            "distance_m": regime.get("distance_m"),
            "current_mA": regime.get("current_mA"),
            "dist_tol_m": _dist_tol,
            "curr_tol_mA": _curr_tol,
            "max_experiments": int(_max_exp) if _max_exp is not None else None,
        }
        print(f"   ✅ Selected experiments ({len(_sel_paths)}): {_sel_paths}")
        del _all_exps  # free memory; _filt_exps kept for shared loading
    except Exception as e:
        print(f"⚠️  Experiment selection failed for regime '{regime_id}': {e}")
        _filt_exps = []

    result["selected_experiments"] = _sel_paths
    result["selection_criteria"] = _sel_criteria
    # Inject into overrides so training code uses the same selection (Commit 3R)
    ov["_selected_experiments"] = _sel_paths
    ov["_regime_distance_m"] = regime.get("distance_m")
    ov["_regime_current_mA"] = regime.get("current_mA")

    # ---- SHARED DATA LOADING (Commit 3O) ----
    # Loaded once, shared by baseline + dist-metrics + quick cVAE inference.
    _val_data = None        # (X_va, Y_va, D_va, C_va) or None
    _val_df_split = None    # df_split from apply_split; kept for per-experiment seq windowing
    _X_tr = _Y_tr = None
    _need_data = (run_baseline or run_dist_metrics or run_stat_fidelity) and not ov.get("dry_run", False)

    if _need_data:
        try:
            from src.protocol.split_strategies import apply_split
            from src.data.splits import cap_train_samples_per_experiment

            _val_split = float(ov.get("val_split", 0.2))
            _seed = int(ov.get("seed", 42))
            _max_spe = ov.get("max_samples_per_exp")

            # Use already-filtered experiments from above
            exps = list(_filt_exps)
            X_tr, Y_tr, _D_tr, _C_tr, X_va, Y_va, D_va, C_va, _df_split = \
                apply_split(
                    exps,
                    strategy=_split_strategy,
                    val_split=_val_split,
                    seed=_seed,
                    within_exp_shuffle=bool(ov.get("within_experiment_shuffle", False)),
                )

            # Enforce split -> reduce(train only) ordering for max_samples_per_exp.
            if _max_spe is not None:
                _ms = int(_max_spe)
                X_tr, Y_tr, _D_tr, _C_tr, _df_cap = cap_train_samples_per_experiment(
                    X_tr, Y_tr, _D_tr, _C_tr, _df_split, _ms
                )
                print(
                    f"   ⚡ max_samples_per_exp pós-split: train={len(X_tr):,} "
                    f"(cap={_ms}/exp) | val={len(X_va):,} (val intocado)"
                )

            # --- Commit 3P: shape guard ---
            assert X_va.shape[0] == Y_va.shape[0] == D_va.shape[0] == C_va.shape[0], (
                f"Shape mismatch after split: X_va={X_va.shape}, Y_va={Y_va.shape}, "
                f"D_va={D_va.shape}, C_va={C_va.shape}"
            )

            _val_data = (X_va.copy(), Y_va.copy(), D_va.copy(), C_va.copy())  # Commit 3P: isolate
            _val_df_split = _df_split  # kept for per-experiment windowing in _quick_cvae_predict
            _X_tr, _Y_tr = X_tr, Y_tr
            print(f"   split={_split_strategy} | train={len(X_tr):,}  val={len(X_va):,}")

            # Commit 3P: data fingerprint for debugging
            import numpy as _np_fp
            print(f"   🔑 val fingerprint: X mean={_np_fp.mean(X_va):.6f} "
                  f"std={_np_fp.std(X_va):.6f} D_unique={_np_fp.unique(D_va).tolist()} "
                  f"C_unique={_np_fp.unique(C_va).tolist()}")
            del _D_tr, _C_tr, exps, X_tr, Y_tr, X_va, Y_va, D_va, C_va, _df_split
        except Exception as e:
            print(f"⚠️  Data loading failed for regime '{regime_id}': {e}")

    del _filt_exps  # Commit 3R: free filtered arrays

    # ---- BASELINE (Commit 3N) + dist-metrics (Commit 3O) ----
    if run_baseline and _val_data is not None:
        try:
            import time as _time
            from src.baselines.deterministic import run_deterministic_baseline

            X_va, Y_va, D_va, C_va = _val_data

            _bl_cfg = _effective_baseline_config(
                ov,
                enabled=run_baseline,
                return_predictions=run_dist_metrics,
            )

            _bl_t0 = _time.time()
            bl_metrics = run_deterministic_baseline(
                _X_tr, _Y_tr, X_va, Y_va,
                config=_bl_cfg,
            )
            result["baseline_time_s"] = round(_time.time() - _bl_t0, 2)

            # Pop large prediction array before storing metrics
            Y_pred_bl = bl_metrics.pop("_Y_pred", None)
            result["baseline"] = bl_metrics
            print(f"   ✅ Baseline: EVM={bl_metrics['evm_pred_%']:.3f}%  "
                  f"SNR={bl_metrics['snr_pred_db']:.2f}dB  "
                  f"({bl_metrics['train_time_s']:.1f}s)")

            # Distribution-fidelity metrics for baseline
            if run_dist_metrics and Y_pred_bl is not None:
                from src.metrics.distribution import residual_fidelity_metrics
                _psd_nfft = int(ov.get("psd_nfft", 2048))
                _max_ds = int(ov.get("max_dist_samples", 200_000))
                _g_alpha = float(ov.get("gauss_alpha", 0.01))
                res_real = Y_va - X_va
                res_pred_bl = Y_pred_bl - X_va
                result["baseline_dist"] = residual_fidelity_metrics(
                    res_real, res_pred_bl,
                    psd_nfft=_psd_nfft, max_samples=_max_ds, gauss_alpha=_g_alpha,
                    X=X_va,
                )
                print(f"   📐 Baseline dist: Δmean_l2={result['baseline_dist']['delta_mean_l2']:.4f}  "
                      f"Δacf_l2={result['baseline_dist'].get('delta_acf_l2', float('nan')):.4f}  "
                      f"psd_l2={result['baseline_dist']['psd_l2']:.4f}  "
                      f"reject_gauss={result['baseline_dist']['reject_gaussian']}")
                del Y_pred_bl, res_pred_bl
        except Exception as e:
            result["baseline"] = result.get("baseline") or {"error": str(e)}
            print(f"⚠️  Baseline failed for regime '{regime_id}': {e}")

    # Free training arrays (val data kept for cVAE dist metrics)
    _X_tr = _Y_tr = None  # Commit 3P: explicit None for safety
    import gc; gc.collect()

    if not run_cvae:
        result["train_status"] = "not_requested"
        result["eval_status"] = "not_requested"
        run_dir = protocol_dir / run_id
        result["run_dir"] = str(run_dir)
        print(f"⏭️  Skipping cVAE path for regime '{regime_id}' (--no_cvae/--baseline_only)")
        _val_data = None
        gc.collect()
        return result

    # ---- TRAINING / MODEL RESOLUTION ----
    print(f"\n{'='*70}")
    print(f"🔬 REGIME: {regime_id} — {regime.get('description', '')}")
    print(f"📁 DATASET_ROOT (effective) = {dataset_root}")
    print(f"{'='*70}")

    _ds_root = str(Path(dataset_root).resolve())
    run_dir = (protocol_dir / run_id).resolve()

    if model_run_dir is None:
        _cvae_t0 = time.time()
        try:
            from src.training.engine import train_engine
            print(f"\n📦 Training regime '{regime_id}' → run_id={run_id}")
            _train_summary = train_engine(
                dataset_root=_ds_root,
                output_base=str(protocol_dir.resolve()),
                run_id=run_id,
                overrides=ov,
            )
            result["train_status"] = _train_summary.get("status", "completed")
            run_dir = Path(_train_summary.get("run_dir", protocol_dir / run_id)).resolve()
            model_run_dir = run_dir
            result["model_run_dir"] = str(model_run_dir)
        except Exception as e:
            result["train_status"] = "failed"
            result["error"] = f"train: {e}\n{traceback.format_exc()}"
            print(f"❌ Training failed for regime '{regime_id}': {e}")

        result["cvae_time_s"] = round(time.time() - _cvae_t0, 2)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        result["train_status"] = "shared_model"
        print(
            f"\n🔁 Reusing shared global model for regime '{regime_id}'\n"
            f"   model_run_dir={model_run_dir}\n"
            f"   eval_output_dir={run_dir}"
        )

    result["run_dir"] = str(run_dir)

    # If dry_run was set, do not require a materialized model under the regime dir.
    if ov.get("dry_run", False):
        result["train_status"] = "dry_run"
        return result

    if model_run_dir is None:
        return result

    # Read train state from the model source
    state = _read_train_state(model_run_dir)
    result["best_grid_tag"] = _extract_best_grid_tag(state)

    # ---- EVALUATION ----
    _eval_ran = False
    if skip_eval:
        print(f"⏭️  Skipping evaluation for regime '{regime_id}' (--skip_eval)")
    elif result["train_status"] not in {"completed", "shared_model"}:
        print(f"⏭️  Skipping evaluation for regime '{regime_id}' (training failed)")
    else:
        try:
            # Evaluation consumes the same effective override dict used by
            # baseline/training so no knob silently diverges.
            eval_ov = dict(ov)

            from src.evaluation.engine import evaluate_run
            print(f"\n📊 Evaluating regime '{regime_id}' → {run_dir}")
            _eval_summary = evaluate_run(
                run_dir=model_run_dir,
                dataset_root=_ds_root,
                overrides=eval_ov,
                output_run_dir=run_dir,
            )
            result["eval_status"] = _eval_summary.get("status", "completed")
            result["metrics"] = dict(_eval_summary.get("metrics", {}))
            _eval_ran = True
        except Exception as e:
            result["eval_status"] = "failed"
            err_msg = f"eval: {e}\n{traceback.format_exc()}"
            result["error"] = (result.get("error") or "") + err_msg
            print(f"❌ Evaluation failed for regime '{regime_id}': {e}")

    # Read eval metrics
    if not result["metrics"]:
        result["metrics"] = _read_eval_metrics(run_dir)
    if result["metrics"]:
        result["residual_signature"] = dict(result["metrics"])

    # ---- cVAE DISTRIBUTION-FIDELITY METRICS (single source of truth) ----
    # Priority order:
    # 1) Evaluation metrics JSON (same slice/MC/calc as the canonical eval engine).
    # 2) Quick fallback from shared val split when eval is unavailable.
    if run_dist_metrics and result["train_status"] in {"completed", "shared_model"}:
        _dm_source = None  # will be set below
        try:
            _eval_dm = _extract_cvae_dist_from_eval_metrics(result.get("metrics", {}))
            if _eval_ran and result["eval_status"] == "completed" and _eval_dm:
                _dm_source = "eval_reanalysis"
                result["cvae_dist"] = _eval_dm
                result["dist_metrics_source"] = _dm_source
                print(f"\n🔍 cVAE dist-metrics for regime '{regime_id}' "
                      f"(source={_dm_source}, N_eval={result.get('metrics', {}).get('N_eval', 'n/a')})")
                _dm_mean = _eval_dm.get("delta_mean_l2")
                _dm_acf = _eval_dm.get("delta_acf_l2")
                _dm_psd = _eval_dm.get("psd_l2")
                print(f"   📐 cVAE dist ({_dm_source}): "
                      f"Δmean_l2={(float(_dm_mean) if _dm_mean is not None else float('nan')):.4f}  "
                      f"Δacf_l2={(float(_dm_acf) if _dm_acf is not None else float('nan')):.4f}  "
                      f"psd_l2={(float(_dm_psd) if _dm_psd is not None else float('nan')):.4f}  "
                      f"reject_gauss={_eval_dm.get('reject_gaussian')}")
            else:
                import numpy as _np_dm
                _psd_nfft = int(ov.get("psd_nfft", 2048))
                _max_ds = int(ov.get("max_dist_samples", 200_000))
                _g_alpha = float(ov.get("gauss_alpha", 0.01))
                _mc_dm = max(1, int(result.get("metrics", {}).get("mc_samples", 8)))

                if _val_data is None:
                    raise RuntimeError("Shared validation data not available")

                _X_va, _Y_va, _D_va, _C_va = _val_data
                model_check = model_run_dir / "models" / "best_model_full.keras"
                if not model_check.exists():
                    raise FileNotFoundError(f"Model not found at {model_check}")

                if _eval_ran and result["eval_status"] != "completed":
                    _dm_source = "quick_fallback"
                    print(f"⚠️  Eval status={result['eval_status']} para '{regime_id}' "
                          f"— usando fallback quick.")
                else:
                    _dm_source = "quick"

                print(f"\n🔍 cVAE dist-metrics for regime '{regime_id}' "
                      f"({len(_X_va):,} val pts, source={_dm_source}, "
                      f"mc_samples={_mc_dm}, model={model_check})")
                _pred_pack = _quick_cvae_predict(
                    model_run_dir, _X_va, _D_va, _C_va,
                    mc_samples=_mc_dm,
                    seed=int(ov.get("seed", 42)),
                    mode="mc_concat",
                    df_split=_val_df_split,
                )
                if _pred_pack is not None:
                    from src.metrics.distribution import residual_fidelity_metrics

                    Y_pred_cvae, X_tiled, _D_tiled, _C_tiled = _pred_pack
                    res_real_all = _Y_va - _X_va
                    res_pred_all = Y_pred_cvae - X_tiled

                    _n_cmp = min(_max_ds, res_real_all.shape[0], res_pred_all.shape[0])
                    _rng_dm = _np_dm.random.default_rng(int(ov.get("seed", 42)))
                    idx_real = (_rng_dm.choice(res_real_all.shape[0], _n_cmp, replace=False)
                                if _n_cmp < res_real_all.shape[0]
                                else _np_dm.arange(res_real_all.shape[0]))
                    idx_pred = (_rng_dm.choice(res_pred_all.shape[0], _n_cmp, replace=False)
                                if _n_cmp < res_pred_all.shape[0]
                                else _np_dm.arange(res_pred_all.shape[0]))
                    res_real = res_real_all[idx_real]
                    res_pred_cvae = res_pred_all[idx_pred]
                    cvae_dm = residual_fidelity_metrics(
                        res_real, res_pred_cvae,
                        psd_nfft=_psd_nfft, max_samples=_n_cmp, gauss_alpha=_g_alpha,
                        X=_X_va[idx_real],
                        X_pred=X_tiled[idx_pred],
                    )
                    result["cvae_dist"] = cvae_dm
                    result["dist_metrics_source"] = _dm_source
                    print(f"   📐 cVAE dist ({_dm_source}): "
                          f"Δmean_l2={cvae_dm['delta_mean_l2']:.4f}  "
                          f"Δacf_l2={cvae_dm.get('delta_acf_l2', float('nan')):.4f}  "
                          f"psd_l2={cvae_dm['psd_l2']:.4f}  "
                          f"reject_gauss={cvae_dm['reject_gaussian']}")
                    del _pred_pack, Y_pred_cvae, X_tiled, _D_tiled, _C_tiled
                    del res_real_all, res_pred_all, res_real, res_pred_cvae, cvae_dm
                else:
                    print(f"⚠️  cVAE quick_predict returned None for regime '{regime_id}'")
                del _X_va, _Y_va, _D_va, _C_va
        except Exception as e:
            result["cvae_dist"] = {"error": str(e)}
            if _dm_source is not None:
                result["dist_metrics_source"] = _dm_source
            print(f"⚠️  cVAE dist metrics failed for regime '{regime_id}': {e}")

    if run_dist_metrics and result["train_status"] in {"completed", "shared_model"} and _val_data is not None:
        try:
            from src.evaluation.metrics import residual_signature_by_amplitude_bin

            _X_va, _Y_va, _D_va, _C_va = _val_data
            _mc_bins = max(1, int(result.get("metrics", {}).get("mc_samples", 8)))
            _pred_pack_sig = _quick_cvae_predict(
                model_run_dir,
                _X_va,
                _D_va,
                _C_va,
                mc_samples=_mc_bins,
                seed=int(ov.get("seed", 42)),
                mode="mc_concat",
                df_split=_val_df_split,
            )
            if _pred_pack_sig is not None:
                Y_pred_sig, X_tiled_sig, _D_tiled_sig, _C_tiled_sig = _pred_pack_sig
                _stat_n_perm_bins = 200 if stat_n_perm is None else int(stat_n_perm)
                result["residual_signature_bins"] = residual_signature_by_amplitude_bin(
                    X_real=_X_va,
                    Y_real=_Y_va,
                    X_pred=X_tiled_sig,
                    Y_pred=Y_pred_sig,
                    regime_id=regime_id,
                    regime_label=result.get("regime_label", ""),
                    study=result.get("_study", ""),
                    run_id=run_id,
                    run_dir=str(run_dir),
                    model_run_dir=str(model_run_dir),
                    best_grid_tag=result.get("best_grid_tag", ""),
                    dist_target_m=float(result.get("selection_criteria", {}).get("distance_m", float("nan"))),
                    curr_target_mA=float(result.get("selection_criteria", {}).get("current_mA", float("nan"))),
                    amplitude_bins=int(ov.get("train_regime_diagnostics_amplitude_bins", 4)),
                    min_samples_per_bin=512,
                    stat_mode=str(stat_mode),
                    stat_n_perm=_stat_n_perm_bins,
                    stat_seed=int(stat_seed),
                )
                del _pred_pack_sig, Y_pred_sig, X_tiled_sig, _D_tiled_sig, _C_tiled_sig
            del _X_va, _Y_va, _D_va, _C_va
        except Exception as e:
            print(f"⚠️  Residual signature by amplitude bin failed for regime '{regime_id}': {e}")

    # ---- STATISTICAL FIDELITY TESTS (Etapa A2) ----
    # Runs MMD², Energy distance, and PSD L2 on residuals (Y - X) for cVAE.
    # p-values are stored per regime; global FDR correction is applied in main().
    if run_stat_fidelity and result["train_status"] in {"completed", "shared_model"}:
        try:
            import numpy as _np_sf
            from src.evaluation.stat_tests import mmd_rbf, energy_test, psd_distance

            if _val_data is None:
                raise RuntimeError("Shared validation data not available for stat tests")

            _X_va, _Y_va, _D_va, _C_va = _val_data

            model_check = model_run_dir / "models" / "best_model_full.keras"
            if not model_check.exists():
                raise FileNotFoundError(f"Model not found at {model_check}")

            # Reuse MC predictions from quick predict
            _mc_sf = max(1, int(result.get("metrics", {}).get("mc_samples", 8)))
            _pred_pack = _quick_cvae_predict(
                model_run_dir, _X_va, _D_va, _C_va,
                mc_samples=_mc_sf,
                seed=stat_seed,
                mode="mc_concat",
                df_split=_val_df_split,
            )
            if _pred_pack is None:
                raise RuntimeError("cVAE quick_predict returned None")
            Y_pred_sf, X_tiled_sf, _D_tiled_sf, _C_tiled_sf = _pred_pack

            # Compute residual pools
            res_real_all = _Y_va - _X_va
            res_pred_all = Y_pred_sf - X_tiled_sf

            # Sub-sample both pools independently to the same size.
            _n_sf = min(stat_max_n, res_real_all.shape[0], res_pred_all.shape[0])
            rng = _np_sf.random.RandomState(stat_seed)
            idx_real = (rng.choice(res_real_all.shape[0], _n_sf, replace=False)
                        if _n_sf < res_real_all.shape[0]
                        else _np_sf.arange(res_real_all.shape[0]))
            idx_pred = (rng.choice(res_pred_all.shape[0], _n_sf, replace=False)
                        if _n_sf < res_pred_all.shape[0]
                        else _np_sf.arange(res_pred_all.shape[0]))
            res_real = res_real_all[idx_real]
            res_pred = res_pred_all[idx_pred]

            _n_perm = stat_n_perm if stat_n_perm is not None else (200 if stat_mode == "quick" else 2000)
            _psd_nfft_sf = int(ov.get("psd_nfft", 2048))

            print(f"\n🧪 Stat fidelity for regime '{regime_id}' "
                  f"(n={_n_sf:,}, n_perm={_n_perm}, mode={stat_mode})")

            sf_mmd = mmd_rbf(res_real, res_pred, n_perm=_n_perm, seed=stat_seed)
            sf_energy = energy_test(res_real, res_pred, n_perm=_n_perm, seed=stat_seed)
            sf_psd = psd_distance(res_real, res_pred, nfft=_psd_nfft_sf, seed=stat_seed)

            result["stat_fidelity"] = {
                "mmd2": sf_mmd["mmd2"],
                "mmd_pval": sf_mmd["pval"],
                "mmd_bandwidth": sf_mmd["bandwidth"],
                "energy": sf_energy["energy"],
                "energy_pval": sf_energy["pval"],
                "psd_dist": sf_psd["psd_dist"],
                "psd_ci_low": sf_psd["psd_ci_low"],
                "psd_ci_high": sf_psd["psd_ci_high"],
                "n_samples": _n_sf,
                "n_perm": _n_perm,
                "stat_mode": stat_mode,
                "stat_seed": stat_seed,
            }
            print(f"   📐 MMD²={sf_mmd['mmd2']:.6f} (p={sf_mmd['pval']:.4f})  "
                  f"Energy={sf_energy['energy']:.6f} (p={sf_energy['pval']:.4f})  "
                  f"PSD_L2={sf_psd['psd_dist']:.4f} "
                  f"[{sf_psd['psd_ci_low']:.4f}, {sf_psd['psd_ci_high']:.4f}]")
            del _pred_pack, Y_pred_sf, X_tiled_sf, _D_tiled_sf, _C_tiled_sf
            del res_real_all, res_pred_all, res_real, res_pred, sf_mmd, sf_energy, sf_psd
        except Exception as e:
            result["stat_fidelity"] = {"error": str(e)}
            print(f"⚠️  Stat fidelity failed for regime '{regime_id}': {e}")

    # Commit 3P: aggressively free val data; prevent cross-regime leakage
    _val_data = None
    gc.collect()

    return result


def build_summary_table(results: List[dict]) -> "pd.DataFrame":
    """Compatibility wrapper around the canonical validation-summary builder."""
    return build_validation_summary_table(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_writable_mpl_config_dir()
    args = parse_args()
    warn_if_no_gpu_and_confirm("protocol run")
    ts_start = datetime.now()
    ts_label = ts_start.strftime("%Y%m%d_%H%M%S")
    git_commit = _git_commit_hash()
    versions = _runtime_versions()

    # --- Commit 3M: resolve dataset_root to absolute path ---
    args.dataset_root = str(Path(args.dataset_root).resolve())
    args.output_base = str(Path(args.output_base).resolve())

    # Load protocol (YAML > JSON > bundled reduced default)
    _proto_config_path: Optional[str] = None
    if args.protocol_config is not None:
        _proto_config_path = str(Path(args.protocol_config).resolve())
        protocol = _load_protocol_yaml(_proto_config_path)
        print(f"📄 Loaded protocol YAML: {_proto_config_path}")
    elif args.protocol is not None:
        protocol = _load_protocol(args.protocol)
    else:
        protocol = _load_protocol(None)
        print("📄 No --protocol / --protocol_config given — using bundled reduced default protocol")
    protocol = _limit_protocol_regimes(protocol, args.max_regimes)
    proto_globals = protocol.get("global_settings", {})
    regimes = protocol["regimes"]
    run_cvae = _should_run_cvae(
        no_cvae=args.no_cvae,
        baseline_only=args.baseline_only,
    )
    execution_mode = _protocol_execution_mode(
        train_once_eval_all=bool(args.train_once_eval_all) and bool(run_cvae),
    )
    args.stat_max_n = _effective_stat_max_n(args.stat_mode, args.stat_max_n)
    reused_model_run_dir = _resolve_reuse_model_run_dir(args.reuse_model_run_dir)

    # Merge protocol globals + CLI overrides → typed RunOverrides
    base_overrides = _merge_overrides(proto_globals, args)
    base_overrides_dict = base_overrides.to_dict()          # legacy compat
    # Pass through private keys (e.g. _selected_experiments) from global_settings
    # that RunOverrides does not model as typed fields.
    for _k, _v in proto_globals.items():
        if _k.startswith("_") and _k not in base_overrides_dict:
            base_overrides_dict[_k] = _v

    # ---- Etapa A5: guard incompatible flag combos ----
    if args.dry_run and args.stat_tests:
        print("⚠️  --stat_tests requires a trained model for Y_pred — "
              "incompatible with --dry_run.  Disabling --stat_tests.")
        args.stat_tests = False
    if not run_cvae and args.skip_eval:
        print("⚠️  --skip_eval is redundant with --no_cvae/--baseline_only.")
    if not run_cvae and args.stat_tests:
        print("⚠️  --stat_tests requires cVAE predictions — disabling because --no_cvae/--baseline_only was set.")
        args.stat_tests = False
    if args.train_once_eval_all and not run_cvae:
        print("⚠️  --train_once_eval_all has no effect when cVAE execution is disabled.")
        execution_mode = "per_regime_retrain"
    if reused_model_run_dir is not None:
        if not run_cvae:
            raise ValueError("--reuse_model_run_dir requires the cVAE path to be enabled.")
        if execution_mode != "train_once_eval_all":
            raise ValueError("--reuse_model_run_dir requires --train_once_eval_all.")

    # Experiment output directory (single folder per protocol run)
    exp_paths = RunPaths(
        run_id=f"exp_{ts_label}",
        run_dir=Path(args.output_base) / f"exp_{ts_label}",
        logs_dir=Path(args.output_base) / f"exp_{ts_label}" / "logs",
    )
    exp_dir = exp_paths.run_dir                         # backward compat

    studies_meta = protocol.get("_studies", [])
    n_studies = len(studies_meta)
    manifest_args = _protocol_manifest_args(
        args,
        proto_config_path=_proto_config_path,
        reused_model_run_dir=reused_model_run_dir,
    )

    print(f"🚀 Protocol runner — {len(studies_meta)} study(ies), {len(regimes)} regime(s)")
    print(f"🧭 Execution mode: {execution_mode}")
    print(f"📁 Experiment dir: {exp_dir}")
    print(f"🔧 Base overrides: {base_overrides}")

    _write_protocol_running_manifest(
        exp_paths,
        ts_start=ts_start,
        git_commit=git_commit,
        versions=versions,
        args_payload=manifest_args,
        execution_mode=execution_mode,
        studies_meta=studies_meta,
        regimes=regimes,
    )

    # Save a copy of the protocol used (always the resolved dict as JSON)
    exp_paths.write_json("logs/protocol_input.json", protocol)
    # Also save original YAML when applicable
    if _proto_config_path is not None:
        import shutil
        shutil.copy2(_proto_config_path, exp_paths.logs_dir / "protocol_input.yaml")

    shared_model_run_dir: Optional[Path] = None
    training_operational_artifacts = {
        "grid_training_diagnostics_csv": None,
        "training_dashboard_png": None,
        "source_run_dir": None,
        "reused": False,
    }
    if execution_mode == "train_once_eval_all":
        print(f"\n{'='*70}")
        if reused_model_run_dir is not None:
            shared_model_run_dir = reused_model_run_dir
            _reused_state = _read_train_state(shared_model_run_dir)
            _reused_best_tag = _extract_best_grid_tag(_reused_state)
            _reused_training_artifacts = _extract_training_operational_artifacts(shared_model_run_dir)
            training_operational_artifacts = {
                **_reused_training_artifacts,
                "source_run_dir": str(shared_model_run_dir),
                "reused": True,
            }
            exp_paths.write_json(
                "logs/train/reused_model.json",
                {
                    "shared_model_run_dir": str(shared_model_run_dir),
                    "best_grid_tag": _reused_best_tag,
                    "state_run_present": bool(_reused_state),
                    "grid_training_diagnostics_csv": _reused_training_artifacts["grid_training_diagnostics_csv"],
                    "training_dashboard_png": _reused_training_artifacts["training_dashboard_png"],
                },
            )
            print("🌐 GLOBAL MODEL REUSE (skip training, evaluate all regimes)")
            print(f"📁 Reused model dir: {shared_model_run_dir}")
            if _reused_best_tag:
                print(f"🏷️  Reused best_grid_tag: {_reused_best_tag}")
            print(f"{'='*70}")
        else:
            shared_model_run_dir = exp_dir / "train"
            shared_train_overrides = dict(base_overrides_dict)
            shared_train_overrides["_logs_dir"] = str((exp_dir / "logs" / "train").resolve())
            print("🌐 GLOBAL MODEL TRAINING (train once, evaluate all regimes)")
            print(f"📁 Shared model dir: {shared_model_run_dir}")
            print(f"{'='*70}")
            _shared_t0 = time.time()
            try:
                from src.training.engine import train_engine

                _global_summary = train_engine(
                    dataset_root=args.dataset_root,
                    output_base=str(exp_dir),
                    run_id="train",
                    overrides=shared_train_overrides,
                )
                shared_model_run_dir = Path(
                    _global_summary.get("run_dir", shared_model_run_dir)
                ).resolve()
                training_operational_artifacts = {
                    **_extract_training_operational_artifacts(shared_model_run_dir),
                    "source_run_dir": str(shared_model_run_dir),
                    "reused": False,
                }
                print(
                    f"✅ Shared global model status={_global_summary.get('status', 'completed')} "
                    f"| run_dir={shared_model_run_dir} "
                    f"| time={time.time() - _shared_t0:.1f}s"
                )
            except Exception as e:
                err = f"global_train: {e}\n{traceback.format_exc()}"
                manifest = {
                    "run_type": RUN_TYPE_PROTOCOL,
                    "run_status": RUN_STATUS_FAILED,
                    "protocol_version": protocol.get("protocol_version", "1.0"),
                    "timestamp_start": ts_start.isoformat(timespec="seconds"),
                    "timestamp_end": datetime.now().isoformat(timespec="seconds"),
                    "duration_seconds": (datetime.now() - ts_start).total_seconds(),
                    "git_commit": git_commit,
                    "versions": versions,
                    "args": manifest_args,
                    "execution_mode": execution_mode,
                    "training_operational_artifacts": training_operational_artifacts,
                    "error": err,
                }
                exp_paths.write_json("manifest.json", manifest)
                raise RuntimeError(f"Shared global training failed: {e}") from e

    # ---- Run studies → regimes ----
    results = []
    for si, study_info in enumerate(studies_meta, 1):
        sname = study_info["name"]
        study_rids = set(study_info["regime_ids"])
        study_regimes = [r for r in regimes if r["regime_id"] in study_rids]

        # Regime output directory: exp_dir/eval[/<study>]/
        regimes_dir = exp_dir / "eval"
        study_logs_root = exp_dir / "logs" / "eval"
        if n_studies > 1:
            regimes_dir = regimes_dir / sname
            study_logs_root = study_logs_root / sname
        regimes_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"= STUDY {si}/{len(studies_meta)}: {sname}")
        print(f"= Split strategy: {study_info.get('split_strategy', 'per_experiment')}")
        print(f"= Regimes dir:    {regimes_dir}")
        print(f"{'='*70}")

        for ri, regime in enumerate(study_regimes, 1):
            print(f"\n{'#'*70}")
            print(f"# REGIME {ri}/{len(study_regimes)}: {regime['regime_id']}")
            print(f"# Study: {sname}")
            print(f"{'#'*70}")
            r = run_regime(
                regime=regime,
                dataset_root=args.dataset_root,
                base_overrides=base_overrides,
                protocol_dir=regimes_dir,
                logs_root=study_logs_root,
                shared_model_run_dir=shared_model_run_dir,
                run_cvae=run_cvae,
                skip_eval=args.skip_eval,
                run_baseline=not args.no_baseline,
                run_dist_metrics=not args.no_dist_metrics,
                run_stat_fidelity=args.stat_tests,
                stat_mode=args.stat_mode,
                stat_n_perm=args.stat_n_perm,
                stat_seed=args.stat_seed,
                stat_max_n=args.stat_max_n,
            )
            r["_study"] = sname
            for _row in r.get("residual_signature_bins", []) or []:
                _row["study"] = sname
            results.append(r)

    df_summary = build_summary_table(results)

    summary_csv = exp_paths.write_table("tables/summary_by_regime.csv", df_summary)
    print(f"\n📊 Summary table: {summary_csv}")

    try:
        df_signature = build_residual_signature_table(results, df_summary)
        if not df_signature.empty:
            sig_csv = exp_paths.write_table("tables/residual_signature_by_regime.csv", df_signature)
            print(f"🧬 Residual signature table: {sig_csv}")
        else:
            df_signature = None
    except Exception as e:
        df_signature = None
        print(f"⚠️  Residual signature table failed: {e}")

    try:
        df_signature_amp = build_residual_signature_amplitude_table(results)
        if not df_signature_amp.empty:
            sig_amp_csv = exp_paths.write_table(
                "tables/residual_signature_by_amplitude_bin.csv",
                df_signature_amp,
            )
            print(f"📶 Residual signature by amplitude bin: {sig_amp_csv}")
        else:
            df_signature_amp = None
    except Exception as e:
        df_signature_amp = None
        print(f"⚠️  Residual signature by amplitude bin failed: {e}")

    try:
        df_leaderboard = build_protocol_leaderboard(df_summary)
        leaderboard_csv = exp_paths.write_table("tables/protocol_leaderboard.csv", df_leaderboard)
        print(f"🏆 Protocol leaderboard: {leaderboard_csv}")
    except Exception as e:
        df_leaderboard = None
        print(f"⚠️  Protocol leaderboard failed: {e}")

    # ---- Best-model heatmaps (derived from canonical summary table) ----
    try:
        from src.evaluation.summary_plots import generate_all as _summary_plots

        _plot_dir = exp_dir / "plots" / "best_model"
        _summary_created = _summary_plots(df_summary, _plot_dir)
        if _summary_created:
            print(f"📈 Best-model heatmaps ({len(_summary_created)}): {_plot_dir}")
    except Exception as _se:
        print(f"⚠️  Best-model heatmaps failed: {_se}")

    try:
        if df_signature is not None and not df_signature.empty:
            from src.evaluation.summary_plots import plot_residual_signature_overview

            _plot_dir = exp_dir / "plots" / "best_model"
            _plot_dir.mkdir(parents=True, exist_ok=True)
            _sig_plot = plot_residual_signature_overview(df_signature, _plot_dir)
            if _sig_plot is not None:
                print(f"📈 Residual signature overview: {_sig_plot}")
    except Exception as _se:
        print(f"⚠️  Residual signature overview failed: {_se}")

    try:
        _diag_src = Path(shared_model_run_dir) / "logs" / "train" / "regime_diagnostics_history.csv" if shared_model_run_dir is not None else None
        if _diag_src is not None and _diag_src.exists():
            _diag_dst = exp_dir / "tables" / "train_regime_diagnostics_history.csv"
            _diag_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_diag_src, _diag_dst)
            print(f"📒 Copied train regime diagnostics history: {_diag_dst}")
    except Exception as _se:
        print(f"⚠️  Copy train regime diagnostics history failed: {_se}")

    # ---- Etapa A2: Stat fidelity projection (derived from canonical summary) ----
    if args.stat_tests:
        try:
            df_sf = build_stat_fidelity_table(df_summary)
            if not df_sf.empty:
                sf_csv = exp_paths.write_table("tables/stat_fidelity_by_regime.csv", df_sf)
                print(f"📊 Stat fidelity table (derived from summary): {sf_csv}")

                _stat_acceptance = build_stat_acceptance_summary(df_summary)
            else:
                _stat_acceptance = None
                print("⚠️  No valid stat fidelity results to aggregate")
        except Exception as e:
            _stat_acceptance = None
            print(f"⚠️  FDR aggregation failed: {e}")
    else:
        _stat_acceptance = None

    # ---- Write manifest ----
    ts_end = datetime.now()
    _baseline_cfg = _effective_baseline_config(
        base_overrides,
        enabled=not args.no_baseline,
        return_predictions=False,
    )
    _baseline_cfg.pop("return_predictions", None)
    _dist_cfg = _effective_dist_metrics_config(
        base_overrides,
        enabled=not args.no_dist_metrics,
    )
    manifest = {
        "run_type": RUN_TYPE_PROTOCOL,
        "run_status": RUN_STATUS_COMPLETED,
        "protocol_version": protocol.get("protocol_version", "1.0"),
        "timestamp_start": ts_start.isoformat(timespec="seconds"),
        "timestamp_end": ts_end.isoformat(timespec="seconds"),
        "duration_seconds": (ts_end - ts_start).total_seconds(),
        "git_commit": git_commit,
        "versions": versions,
        "args": manifest_args,
        "execution_mode": execution_mode,
        "baseline_config": _baseline_cfg,
        "cvae_config": _effective_cvae_config(
            base_overrides,
            enabled=run_cvae,
            execution_mode=execution_mode,
        ),
        "dist_metrics_config": _dist_cfg,
        "stat_fidelity_config": {
            "enabled": args.stat_tests,
            "stat_mode": args.stat_mode,
            "stat_n_perm": args.stat_n_perm,
            "stat_seed": args.stat_seed,
            "stat_max_n": args.stat_max_n,
        },
        "shared_model_run_dir": str(shared_model_run_dir) if shared_model_run_dir is not None else None,
        "training_operational_artifacts": training_operational_artifacts,
        "stat_acceptance": _stat_acceptance,
        "protocol_leaderboard": (
            {
                "path": str(exp_dir / "tables" / "protocol_leaderboard.csv"),
                "n_candidates": int(len(df_leaderboard)) if df_leaderboard is not None else 0,
                "winner_candidate_id": (
                    str(df_leaderboard.iloc[0]["candidate_id"])
                    if df_leaderboard is not None and not df_leaderboard.empty
                    else None
                ),
                "winner_best_grid_tag": (
                    str(df_leaderboard.iloc[0]["best_grid_tag"])
                    if df_leaderboard is not None and not df_leaderboard.empty
                    else None
                ),
            }
            if df_leaderboard is not None
            else None
        ),
        "base_overrides": base_overrides_dict,
        "n_studies": len(studies_meta),
        "studies": [
            {
                "name": s["name"],
                "split_strategy": s.get("split_strategy", "per_experiment"),
                "n_regimes": len(s["regime_ids"]),
                "regime_ids": s["regime_ids"],
            }
            for s in studies_meta
        ],
        "n_regimes": len(regimes),
        "regimes": [
            {
                "regime_id": r["regime_id"],
                "regime_label": r.get("regime_label", ""),
                "study": r.get("_study", "within_regime"),
                "run_id": r["run_id"],
                "run_dir": r.get("run_dir", ""),
                "model_run_dir": r.get("model_run_dir"),
                "model_scope": r.get("model_scope"),
                "train_status": r["train_status"],
                "eval_status": r["eval_status"],
                "best_grid_tag": r.get("best_grid_tag", ""),
                "baseline_time_s": r.get("baseline_time_s", 0.0),
                "cvae_time_s": r.get("cvae_time_s", 0.0),
                "baseline_dist": r.get("baseline_dist", {}),
                "cvae_dist": r.get("cvae_dist", {}),
                "dist_metrics_source": r.get("dist_metrics_source"),
                "stat_fidelity": r.get("stat_fidelity", {}),
                "selected_experiments": r.get("selected_experiments", []),
                "selection_criteria": r.get("selection_criteria", {}),
                "error": r.get("error"),
            }
            for r in results
        ],
    }
    manifest_path = exp_paths.write_json("manifest.json", manifest)
    write_latest_completed_experiment_record(args.output_base, exp_dir)
    print(f"📋 Manifest: {manifest_path}")

    # ---- Final summary to stdout ----
    print(f"\n{'='*70}")
    print(f"✅ Protocol complete — {len(studies_meta)} study(ies), {len(results)} regime(s)")
    print(f"   Duration: {ts_end - ts_start}")
    print(f"   Output:   {exp_dir}")
    for r in results:
        slab = f"[{r.get('_study', '?')}] "
        status = f"train={r['train_status']}, eval={r['eval_status']}"
        delta = ""
        m = r.get("metrics", {})
        bl = r.get("baseline", {})
        bd = r.get("baseline_dist", {})
        cd = r.get("cvae_dist", {})
        if m.get("delta_evm_%") is not None:
            delta = f" | ΔEVM={m['delta_evm_%']:+.3f}pp ΔSNR={m.get('delta_snr_db', 0):+.3f}dB"
        if bl.get("evm_pred_%") is not None:
            delta += f" | baseline EVM={bl['evm_pred_%']:.3f}%"
        if bd.get("delta_mean_l2") is not None:
            delta += f" | bl_Δmean={bd['delta_mean_l2']:.4f}"
        if bd.get("delta_acf_l2") is not None:
            delta += f" | bl_Δacf={bd['delta_acf_l2']:.4f}"
        if cd.get("delta_mean_l2") is not None:
            delta += f" | cv_Δmean={cd['delta_mean_l2']:.4f}"
        if cd.get("delta_acf_l2") is not None:
            delta += f" | cv_Δacf={cd['delta_acf_l2']:.4f}"
        sf = r.get("stat_fidelity", {})
        if sf.get("mmd2") is not None:
            delta += f" | MMD²={sf['mmd2']:.6f}(p={sf['mmd_pval']:.3f})"
        print(f"   • {slab}{r['regime_id']}: {status}{delta}")

    # ---- Etapa A4: acceptance summary ----
    if _stat_acceptance is not None:
        sa = _stat_acceptance
        print(f"\n{'─'*70}")
        print(f"🧪 STAT FIDELITY ACCEPTANCE  (q_α={sa['q_alpha']}, "
              f"PSD ratio ≤ {sa['psd_ratio_limit']}×baseline)")
        print(f"   Regimes tested:  {sa['n_regimes_tested']}")
        print(f"   q_MMD  > α:      {sa['pass_mmd_qval']}/{sa['n_regimes_tested']} "
              f"({sa['pct_pass_mmd']:.1f}%)")
        print(f"   q_Energy > α:    {sa['pass_energy_qval']}/{sa['n_regimes_tested']} "
              f"({sa['pct_pass_energy']:.1f}%)")
        print(f"   Both > α:        {sa['pass_both_qval']}/{sa['n_regimes_tested']} "
              f"({sa['pct_pass_both']:.1f}%)")
        if sa["psd_ratio_checked"]:
            print(f"   PSD ≤ {sa['psd_ratio_limit']}×bl:   "
                  f"{sa['pass_psd_ratio']}/{sa['n_regimes_psd_checked']} "
                  f"({sa['pct_pass_psd_ratio']:.1f}%)")
        else:
            print(f"   PSD ratio:       not checked (baseline PSD unavailable)")
        _verdict = "PASS ✅" if sa["pct_pass_both"] == 100.0 else "PARTIAL ⚠️"
        print(f"   Verdict:         {_verdict}")
        print(f"{'─'*70}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
