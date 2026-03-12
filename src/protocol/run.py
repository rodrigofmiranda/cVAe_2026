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
        [--protocol configs/protocol_default.json]

    # Quick smoke-test (1 regime, 1 grid, 2 epochs):
    python -m src.protocol.run \\
        --dataset_root data/dataset_fullsquare_organized \\
        --output_base  outputs \\
        --protocol configs/one_regime_1p0m_300mA.json \\
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
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from src.config.overrides import RunOverrides
from src.config.runtime_env import ensure_writable_mpl_config_dir
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
                   help="Path to protocol JSON (default: auto-discover when omitted)")
    p.add_argument("--protocol_config", type=str, default=None,
                   help="Path to protocol YAML config (takes precedence over --protocol)")
    # --- global overrides (applied to every regime; override protocol JSON) ---
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--max_grids", type=int, default=None)
    p.add_argument("--max_regimes", type=int, default=None,
                   help="Limit the number of regimes executed after protocol resolution")
    p.add_argument("--grid_group", type=str, default=None)
    p.add_argument("--grid_tag", type=str, default=None)
    p.add_argument("--max_experiments", type=int, default=None)
    p.add_argument("--max_samples_per_exp", type=int, default=None)
    p.add_argument("--val_split", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--psd_nfft", type=int, default=None)
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
    p.add_argument("--skip_eval", action="store_true",
                   help="Run training only, skip evaluation step")
    p.add_argument("--dry_run", action="store_true",
                   help="Validate protocol + build model summary, no training")
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
) -> Dict[str, Any]:
    """Expose the cVAE-relevant effective overrides in the manifest."""
    ov = _override_dict(overrides)
    cfg: Dict[str, Any] = {"enabled": bool(enabled)}
    for key in (
        "max_epochs",
        "max_grids",
        "grid_group",
        "grid_tag",
        "val_split",
        "seed",
        "max_experiments",
        "max_samples_per_exp",
        "keras_verbose",
    ):
        if ov.get(key) is not None:
            cfg[key] = ov[key]
    return cfg


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
    """Read evaluation metrics JSON produced by analise_cvae_reviewed."""
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
        "var_real_delta": _f(metrics.get("var_real_delta")),
        "var_pred_delta": _f(metrics.get("var_pred_delta")),
        "delta_skew_l2": _f(metrics.get("delta_skew_l2")),
        "delta_kurt_l2": _f(metrics.get("delta_kurt_l2")),
        # eval JSON uses delta_psd_l2 naming; protocol table expects psd_l2.
        "psd_l2": _f(metrics.get("delta_psd_l2", metrics.get("psd_l2"))),
        # Optional JB fields (present after CORREÇÃO 3 propagation in eval metrics).
        "jb_stat_I": _f(metrics.get("jb_stat_I")),
        "jb_stat_Q": _f(metrics.get("jb_stat_Q")),
        "jb_p_I": _f(metrics.get("jb_p_I")),
        "jb_p_Q": _f(metrics.get("jb_p_Q")),
        "jb_p_min": _f(metrics.get("jb_p_min")),
        "jb_log10p_I": _f(metrics.get("jb_log10p_I")),
        "jb_log10p_Q": _f(metrics.get("jb_log10p_Q")),
        "jb_log10p_min": _f(metrics.get("jb_log10p_min")),
        "reject_gaussian": (bool(metrics.get("reject_gaussian"))
                            if metrics.get("reject_gaussian") is not None else None),
    }
    # Require at least one core distance metric to consider mapping valid.
    _core = ("delta_mean_l2", "delta_cov_fro", "delta_skew_l2", "delta_kurt_l2", "psd_l2")
    if all(out.get(k) is None for k in _core):
        return {}
    return out


def _read_train_state(run_dir: Path) -> dict:
    """Read state_run.json produced by the training monolith."""
    p = run_dir / "state_run.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _extract_best_grid_tag(state: dict) -> str:
    """Extract best grid tag from the training state if available."""
    try:
        res_path = state.get("artifacts", {}).get("grid_results_xlsx", "")
        if res_path and Path(res_path).exists():
            import pandas as pd
            df = pd.read_excel(res_path, sheet_name="results_sorted")
            if len(df) > 0 and "tag" in df.columns:
                return str(df.iloc[0]["tag"])
    except Exception:
        pass
    return ""


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
        from src.models.cvae_components import Sampling, CondPriorVAELoss
        from src.models.cvae import create_inference_model_from_full

        custom_objects = {"Sampling": Sampling, "CondPriorVAELoss": CondPriorVAELoss}
        vae = tf.keras.models.load_model(
            str(model_path), custom_objects=custom_objects, compile=False
        )

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
            return Y_pred, X_arr, D_arr, C_arr

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

        X_tiled = _tile_like_input(X_arr)
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
    Returns a result dict with run_dir, metrics, and status.
    """
    regime_id = regime["regime_id"]
    run_id = regime_id   # lives under protocol_dir/<regime_id>/

    # --- Resolve experiment filter ---
    # Option A: explicit experiment_paths
    # Option B: experiment_regex
    exp_paths = regime.get("experiment_paths", [])
    exp_regex = regime.get("experiment_regex", None)

    # Build per-regime overrides
    ov = _override_dict(base_overrides)

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
        "train_status": "skipped",
        "eval_status": "skipped",
        "metrics": {},
        "best_grid_tag": "",
        "baseline": {},
        "baseline_time_s": 0.0,
        "cvae_time_s": 0.0,
        "baseline_dist": {},
        "cvae_dist": {},
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
    _val_data = None   # (X_va, Y_va, D_va, C_va) or None
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
                )
                print(f"   📐 Baseline dist: Δmean_l2={result['baseline_dist']['delta_mean_l2']:.4f}  "
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

    # ---- TRAINING ----
    print(f"\n{'='*70}")
    print(f"🔬 REGIME: {regime_id} — {regime.get('description', '')}")
    print(f"📁 DATASET_ROOT (effective) = {dataset_root}")
    print(f"{'='*70}")

    _ds_root = str(Path(dataset_root).resolve())

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
        result["run_dir"] = str(_train_summary.get("run_dir", protocol_dir / run_id))
    except Exception as e:
        result["train_status"] = "failed"
        result["error"] = f"train: {e}\n{traceback.format_exc()}"
        print(f"❌ Training failed for regime '{regime_id}': {e}")

    result["cvae_time_s"] = round(time.time() - _cvae_t0, 2)
    run_dir = Path(result["run_dir"] or (protocol_dir / run_id))
    result["run_dir"] = str(run_dir)

    # Read train state
    state = _read_train_state(run_dir)
    result["best_grid_tag"] = _extract_best_grid_tag(state)

    # If dry_run was set, training exits early — skip eval
    if ov.get("dry_run", False):
        result["train_status"] = "dry_run"
        return result

    # ---- EVALUATION ----
    _eval_ran = False
    if skip_eval:
        print(f"⏭️  Skipping evaluation for regime '{regime_id}' (--skip_eval)")
    elif result["train_status"] != "completed":
        print(f"⏭️  Skipping evaluation for regime '{regime_id}' (training failed)")
    else:
        try:
            os.environ["DATASET_ROOT"] = _ds_root
            os.environ["OUTPUT_BASE"] = str(protocol_dir.resolve())
            os.environ["RUN_ID"] = run_id

            # Evaluation consumes the same effective override dict used by
            # baseline/training so no knob silently diverges.
            eval_ov = dict(ov)

            from src.evaluation import analise_cvae_reviewed as eval_module
            print(f"\n📊 Evaluating regime '{regime_id}' → {run_dir}")
            eval_module.main(overrides=eval_ov)
            result["eval_status"] = "completed"
            _eval_ran = True
        except Exception as e:
            result["eval_status"] = "failed"
            err_msg = f"eval: {e}\n{traceback.format_exc()}"
            result["error"] = (result.get("error") or "") + err_msg
            print(f"❌ Evaluation failed for regime '{regime_id}': {e}")

    # Read eval metrics
    result["metrics"] = _read_eval_metrics(run_dir)

    # ---- cVAE DISTRIBUTION-FIDELITY METRICS (single source of truth) ----
    # Priority order:
    # 1) Evaluation metrics JSON (same slice/MC/calc as analise_cvae_reviewed).
    # 2) Quick fallback from shared val split when eval is unavailable.
    if run_dist_metrics and result["train_status"] == "completed":
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
                _dm_psd = _eval_dm.get("psd_l2")
                print(f"   📐 cVAE dist ({_dm_source}): "
                      f"Δmean_l2={(float(_dm_mean) if _dm_mean is not None else float('nan')):.4f}  "
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
                model_check = run_dir / "models" / "best_model_full.keras"
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
                    run_dir, _X_va, _D_va, _C_va,
                    mc_samples=_mc_dm,
                    seed=int(ov.get("seed", 42)),
                    mode="mc_concat",
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
                    )
                    result["cvae_dist"] = cvae_dm
                    result["dist_metrics_source"] = _dm_source
                    print(f"   📐 cVAE dist ({_dm_source}): "
                          f"Δmean_l2={cvae_dm['delta_mean_l2']:.4f}  "
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

    # ---- STATISTICAL FIDELITY TESTS (Etapa A2) ----
    # Runs MMD², Energy distance, and PSD L2 on residuals (Y - X) for cVAE.
    # p-values are stored per regime; global FDR correction is applied in main().
    if run_stat_fidelity and result["train_status"] == "completed":
        try:
            import numpy as _np_sf
            from src.evaluation.stat_tests import mmd_rbf, energy_test, psd_distance

            if _val_data is None:
                raise RuntimeError("Shared validation data not available for stat tests")

            _X_va, _Y_va, _D_va, _C_va = _val_data

            model_check = run_dir / "models" / "best_model_full.keras"
            if not model_check.exists():
                raise FileNotFoundError(f"Model not found at {model_check}")

            # Reuse MC predictions from quick predict
            _mc_sf = max(1, int(result.get("metrics", {}).get("mc_samples", 8)))
            _pred_pack = _quick_cvae_predict(
                run_dir, _X_va, _D_va, _C_va,
                mc_samples=_mc_sf,
                seed=stat_seed,
                mode="mc_concat",
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
    """Consolidate per-regime results into a summary DataFrame."""
    import pandas as pd

    rows = []
    for r in results:
        m = r.get("metrics", {})
        bl = r.get("baseline", {})
        bd = r.get("baseline_dist", {})
        cd = r.get("cvae_dist", {})
        sf = r.get("stat_fidelity", {})
        row = {
            "study": r.get("_study", "within_regime"),
            "regime_id": r["regime_id"],
            "regime_label": r.get("regime_label", ""),
            "description": r.get("description", ""),
            "run_id": r["run_id"],
            "run_dir": r.get("run_dir", ""),
            "train_status": r["train_status"],
            "eval_status": r["eval_status"],
            "best_grid_tag": r.get("best_grid_tag", ""),
            "evm_real_%": m.get("evm_real_%"),
            "evm_pred_%": m.get("evm_pred_%"),
            "delta_evm_%": m.get("delta_evm_%"),
            "snr_real_db": m.get("snr_real_db"),
            "snr_pred_db": m.get("snr_pred_db"),
            "delta_snr_db": m.get("delta_snr_db"),
            "delta_mean_l2": m.get("delta_mean_l2"),
            "delta_cov_fro": m.get("delta_cov_fro"),
            # Prefer eval metrics JSON; fallback to top-level keys when available.
            "var_real_delta": m.get("var_real_delta", r.get("var_real_delta")),
            "var_pred_delta": m.get("var_pred_delta", r.get("var_pred_delta")),
            "delta_skew_l2": m.get("delta_skew_l2"),
            "delta_kurt_l2": m.get("delta_kurt_l2"),
            "delta_psd_l2": m.get("delta_psd_l2"),
            "kl_q_to_p_total": None,
            "kl_p_to_N_total": None,
            "var_mc_gen": m.get("var_mc_gen"),
            # Commit 3N: baseline signal-quality
            "baseline_evm_pred_%": bl.get("evm_pred_%"),
            "baseline_snr_pred_db": bl.get("snr_pred_db"),
            "baseline_delta_evm_%": bl.get("delta_evm_%"),
            "baseline_delta_snr_db": bl.get("delta_snr_db"),
            "baseline_time_s": r.get("baseline_time_s", 0.0),
            "cvae_time_s": r.get("cvae_time_s", 0.0),
            # Commit 3O: distribution-fidelity — baseline
            "baseline_delta_mean_l2": bd.get("delta_mean_l2"),
            "baseline_delta_cov_fro": bd.get("delta_cov_fro"),
            "baseline_delta_skew_l2": bd.get("delta_skew_l2"),
            "baseline_delta_kurt_l2": bd.get("delta_kurt_l2"),
            "baseline_psd_l2": bd.get("psd_l2"),
            "baseline_jb_p_min": bd.get("jb_p_min"),
            "baseline_jb_log10p_min": bd.get("jb_log10p_min"),
            "baseline_reject_gauss": bd.get("reject_gaussian"),
            # Commit 3O: distribution-fidelity — cVAE
            "cvae_delta_mean_l2": cd.get("delta_mean_l2"),
            "cvae_delta_cov_fro": cd.get("delta_cov_fro"),
            "cvae_delta_skew_l2": cd.get("delta_skew_l2"),
            "cvae_delta_kurt_l2": cd.get("delta_kurt_l2"),
            "cvae_psd_l2": cd.get("psd_l2"),
            "cvae_jb_p_min": cd.get("jb_p_min"),
            "cvae_jb_log10p_min": cd.get("jb_log10p_min"),
            "cvae_reject_gauss": cd.get("reject_gaussian"),
            "dist_metrics_source": r.get("dist_metrics_source"),
            # Commit 3Q: regime-aware experiment selection
            "n_experiments_selected": len(r.get("selected_experiments", [])),
            "dist_target_m": r.get("selection_criteria", {}).get("distance_m"),
            "curr_target_mA": r.get("selection_criteria", {}).get("current_mA"),
            # Etapa A2: statistical fidelity tests
            "stat_mmd2": sf.get("mmd2"),
            "stat_mmd_pval": sf.get("mmd_pval"),
            "stat_energy": sf.get("energy"),
            "stat_energy_pval": sf.get("energy_pval"),
            "stat_psd_dist": sf.get("psd_dist"),
            "stat_psd_ci_low": sf.get("psd_ci_low"),
            "stat_psd_ci_high": sf.get("psd_ci_high"),
            "stat_n_samples": sf.get("n_samples"),
            "stat_mode": sf.get("stat_mode"),
        }

        # Commit 3P: backfill legacy delta_* columns from cvae_dist when eval
        # was skipped (backward compatibility)
        if row["delta_mean_l2"] is None and cd.get("delta_mean_l2") is not None:
            row["delta_mean_l2"] = cd["delta_mean_l2"]
            row["delta_cov_fro"] = cd.get("delta_cov_fro")
            row["delta_skew_l2"] = cd.get("delta_skew_l2")
            row["delta_kurt_l2"] = cd.get("delta_kurt_l2")
            row["delta_psd_l2"] = cd.get("psd_l2")

        # Try to enrich with latent summary from eval run
        run_dir = Path(r.get("run_dir", ""))
        lat_path = run_dir / "logs" / "latent_summary.json"
        if lat_path.exists():
            try:
                lat = json.loads(lat_path.read_text(encoding="utf-8"))
                row["kl_q_to_p_total"] = lat.get("kl_q_to_p_total_mean")
                row["kl_p_to_N_total"] = lat.get("kl_p_to_N_total_mean")
            except Exception:
                pass

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_writable_mpl_config_dir()
    args = parse_args()
    ts_start = datetime.now()
    ts_label = ts_start.strftime("%Y%m%d_%H%M%S")

    # --- Commit 3M: resolve dataset_root to absolute path ---
    args.dataset_root = str(Path(args.dataset_root).resolve())
    args.output_base = str(Path(args.output_base).resolve())

    # Load protocol (YAML > JSON > auto-discovery from dataset)
    _proto_config_path: Optional[str] = None
    if args.protocol_config is not None:
        _proto_config_path = str(Path(args.protocol_config).resolve())
        protocol = _load_protocol_yaml(_proto_config_path)
        print(f"📄 Loaded protocol YAML: {_proto_config_path}")
    elif args.protocol is not None:
        protocol = _load_protocol(args.protocol)
    else:
        # No config given → auto-discover regimes from dataset layout
        protocol = _build_discovered_protocol(args.dataset_root)
        print("📄 No --protocol / --protocol_config given — using auto-discovery")
    protocol = _limit_protocol_regimes(protocol, args.max_regimes)
    proto_globals = protocol.get("global_settings", {})
    regimes = protocol["regimes"]
    run_cvae = _should_run_cvae(
        no_cvae=args.no_cvae,
        baseline_only=args.baseline_only,
    )
    args.stat_max_n = _effective_stat_max_n(args.stat_mode, args.stat_max_n)

    # Merge protocol globals + CLI overrides → typed RunOverrides
    base_overrides = _merge_overrides(proto_globals, args)
    base_overrides_dict = base_overrides.to_dict()          # legacy compat

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

    # Experiment output directory (single folder per protocol run)
    exp_paths = RunPaths(run_id=f"exp_{ts_label}",
                         run_dir=Path(args.output_base) / f"exp_{ts_label}")
    exp_dir = exp_paths.run_dir                         # backward compat

    studies_meta = protocol.get("_studies", [])

    print(f"🚀 Protocol runner — {len(studies_meta)} study(ies), {len(regimes)} regime(s)")
    print(f"📁 Experiment dir: {exp_dir}")
    print(f"🔧 Base overrides: {base_overrides}")

    # Save a copy of the protocol used (always the resolved dict as JSON)
    exp_paths.write_json("logs/protocol_input.json", protocol)
    # Also save original YAML when applicable
    if _proto_config_path is not None:
        import shutil
        shutil.copy2(_proto_config_path, exp_dir / "logs" / "protocol_input.yaml")

    # ---- Run studies → regimes ----
    results = []
    for si, study_info in enumerate(studies_meta, 1):
        sname = study_info["name"]
        study_rids = set(study_info["regime_ids"])
        study_regimes = [r for r in regimes if r["regime_id"] in study_rids]

        # Regime output directory: exp_dir/studies/<study>/regimes/
        regimes_dir = exp_dir / "studies" / sname / "regimes"
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
            results.append(r)

    # ---- Build summary table ----
    import pandas as pd
    df_summary = build_summary_table(results)

    summary_csv = exp_paths.write_table("tables/summary_by_regime.csv", df_summary)
    exp_paths.write_table("tables/summary_by_regime.xlsx", df_summary)
    print(f"\n📊 Summary table: {summary_csv}")

    # ---- Etapa A2: Global FDR-corrected stat fidelity table ----
    if args.stat_tests:
        try:
            from src.evaluation.stat_tests import benjamini_hochberg
            import numpy as _np_fdr

            sf_rows = []
            for r in results:
                sf = r.get("stat_fidelity", {})
                if sf and "error" not in sf and sf.get("mmd_pval") is not None:
                    sf_rows.append({
                        "study": r.get("_study", "within_regime"),
                        "regime_id": r["regime_id"],
                        "regime_label": r.get("regime_label", ""),
                        "mmd2": sf["mmd2"],
                        "mmd_pval": sf["mmd_pval"],
                        "mmd_bandwidth": sf.get("mmd_bandwidth"),
                        "energy": sf["energy"],
                        "energy_pval": sf["energy_pval"],
                        "psd_dist": sf["psd_dist"],
                        "psd_ci_low": sf["psd_ci_low"],
                        "psd_ci_high": sf["psd_ci_high"],
                        "n_samples": sf["n_samples"],
                        "n_perm": sf["n_perm"],
                        "stat_mode": sf["stat_mode"],
                    })

            if sf_rows:
                df_sf = pd.DataFrame(sf_rows)
                # Collect all p-values for FDR (MMD + Energy = 2 per regime)
                pvals_mmd = df_sf["mmd_pval"].values
                pvals_energy = df_sf["energy_pval"].values
                all_pvals = _np_fdr.concatenate([pvals_mmd, pvals_energy])
                all_qvals = benjamini_hochberg(all_pvals)
                n_reg = len(pvals_mmd)
                df_sf["mmd_qval"] = all_qvals[:n_reg]
                df_sf["energy_qval"] = all_qvals[n_reg:]

                # Add mmd2_normalized = mmd2 / var_real_delta (dimensionless).
                if "regime_id" in df_summary.columns and "var_real_delta" in df_summary.columns:
                    _var_map = df_summary[["regime_id", "var_real_delta"]].drop_duplicates("regime_id")
                    df_sf = df_sf.merge(_var_map, on="regime_id", how="left")
                    _den = pd.to_numeric(df_sf["var_real_delta"], errors="coerce")
                    df_sf["mmd2_normalized"] = _np_fdr.where(
                        _den > 0, df_sf["mmd2"] / _den, _np_fdr.nan
                    )
                else:
                    print("⚠️  Could not compute mmd2_normalized: "
                          "missing 'regime_id' or 'var_real_delta' in summary_by_regime.")

                sf_csv = exp_paths.write_table("tables/stat_fidelity_by_regime.csv", df_sf)
                exp_paths.write_table("tables/stat_fidelity_by_regime.xlsx", df_sf)
                print(f"📊 Stat fidelity table (FDR-corrected): {sf_csv}")

                # ---- Etapa A3: stat fidelity plots ----
                try:
                    from src.evaluation.stat_tests.plots import generate_all as _sf_plots
                    _plot_dir = exp_dir / "plots" / "stat_tests"
                    _sf_created = _sf_plots(
                        df_sf, _plot_dir, df_summary=df_summary,
                    )
                    if _sf_created:
                        print(f"📈 Stat fidelity plots ({len(_sf_created)}): {_plot_dir}")
                except Exception as _pe:
                    print(f"⚠️  Stat fidelity plots failed: {_pe}")

                # ---- Etapa A4: acceptance summary ("strong check") ----
                _n_sf = len(df_sf)
                _q_alpha = 0.05
                _psd_ratio_limit = 1.2
                _pass_mmd = (df_sf["mmd_qval"] > _q_alpha).sum()
                _pass_energy = (df_sf["energy_qval"] > _q_alpha).sum()
                _pass_both = ((df_sf["mmd_qval"] > _q_alpha) &
                              (df_sf["energy_qval"] > _q_alpha)).sum()

                # PSD ratio check: cVAE stat psd_dist <= _psd_ratio_limit × baseline_psd_l2
                _psd_check_df = df_sf[["regime_id", "psd_dist"]].copy()
                _pass_psd = _n_sf  # default: all pass if baseline unavailable
                _psd_checked = False
                if "baseline_psd_l2" in df_summary.columns:
                    _bl_psd = df_summary[["regime_id", "baseline_psd_l2"]].drop_duplicates("regime_id")
                    _psd_check_df = _psd_check_df.merge(_bl_psd, on="regime_id", how="left")
                    _has_both = _psd_check_df.dropna(subset=["psd_dist", "baseline_psd_l2"])
                    if not _has_both.empty:
                        _psd_checked = True
                        _pass_psd = int((_has_both["psd_dist"] <=
                                         _psd_ratio_limit * _has_both["baseline_psd_l2"]).sum())
                        _n_sf_psd = len(_has_both)
                    else:
                        _n_sf_psd = _n_sf
                else:
                    _n_sf_psd = _n_sf

                _stat_acceptance = {
                    "q_alpha": _q_alpha,
                    "psd_ratio_limit": _psd_ratio_limit,
                    "n_regimes_tested": _n_sf,
                    "pass_mmd_qval": int(_pass_mmd),
                    "pass_energy_qval": int(_pass_energy),
                    "pass_both_qval": int(_pass_both),
                    "pct_pass_mmd": round(100 * _pass_mmd / _n_sf, 1) if _n_sf else 0,
                    "pct_pass_energy": round(100 * _pass_energy / _n_sf, 1) if _n_sf else 0,
                    "pct_pass_both": round(100 * _pass_both / _n_sf, 1) if _n_sf else 0,
                    "psd_ratio_checked": _psd_checked,
                    "pass_psd_ratio": int(_pass_psd),
                    "n_regimes_psd_checked": int(_n_sf_psd) if _psd_checked else 0,
                    "pct_pass_psd_ratio": round(100 * _pass_psd / _n_sf_psd, 1) if _psd_checked and _n_sf_psd else None,
                }
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
        "protocol_version": protocol.get("protocol_version", "1.0"),
        "timestamp_start": ts_start.isoformat(timespec="seconds"),
        "timestamp_end": ts_end.isoformat(timespec="seconds"),
        "duration_seconds": (ts_end - ts_start).total_seconds(),
        "git_commit": _git_commit_hash(),
        "versions": _runtime_versions(),
        "args": {
            "dataset_root": args.dataset_root,
            "output_base": args.output_base,
            "protocol": args.protocol,
            "protocol_config": _proto_config_path,
            "max_regimes": args.max_regimes,
            "skip_eval": args.skip_eval,
            "no_baseline": args.no_baseline,
            "no_cvae": args.no_cvae,
            "baseline_only": args.baseline_only,
            "no_dist_metrics": args.no_dist_metrics,
            "dry_run": args.dry_run,
            "stat_tests": args.stat_tests,
        },
        "baseline_config": _baseline_cfg,
        "cvae_config": _effective_cvae_config(base_overrides, enabled=run_cvae),
        "dist_metrics_config": _dist_cfg,
        "stat_fidelity_config": {
            "enabled": args.stat_tests,
            "stat_mode": args.stat_mode,
            "stat_n_perm": args.stat_n_perm,
            "stat_seed": args.stat_seed,
            "stat_max_n": args.stat_max_n,
        },
        "stat_acceptance": _stat_acceptance,
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
        if cd.get("delta_mean_l2") is not None:
            delta += f" | cv_Δmean={cd['delta_mean_l2']:.4f}"
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
