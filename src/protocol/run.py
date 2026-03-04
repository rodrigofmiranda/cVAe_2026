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
        --protocol configs/protocol_default.json \\
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
from typing import Dict, List, Optional


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
                   help="Path to protocol JSON (default: configs/protocol_default.json)")
    p.add_argument("--protocol_config", type=str, default=None,
                   help="Path to protocol YAML config (takes precedence over --protocol)")
    # --- global overrides (applied to every regime; override protocol JSON) ---
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--max_grids", type=int, default=None)
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
    p.add_argument("--no_dist_metrics", action="store_true",
                   help="Skip distribution-fidelity metrics (moments, PSD, Gaussianity)")
    p.add_argument("--skip_eval", action="store_true",
                   help="Run training only, skip evaluation step")
    p.add_argument("--dry_run", action="store_true",
                   help="Validate protocol + build model summary, no training")
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
    """Load protocol JSON, falling back to the default bundled config."""
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
    if "regimes" not in proto or not proto["regimes"]:
        raise ValueError("Protocol JSON must contain a non-empty 'regimes' list.")
    # Tag regimes with implicit within_regime study (Commit 3X)
    proto = _ensure_studies(proto)
    return proto


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


def _merge_overrides(protocol_globals: dict, cli_args: argparse.Namespace) -> dict:
    """
    Build the overrides dict for a single regime by layering:
        protocol global_settings  <  CLI flags (explicit wins)
    """
    ov = {}
    pg = protocol_globals or {}

    # mapping: override_key -> (protocol_key, cli_attr, cast)
    _MAP = [
        ("seed",                "seed",                "seed",                int),
        ("val_split",           "val_split",           "val_split",           float),
        ("max_epochs",          "max_epochs",          "max_epochs",          int),
        ("max_grids",           "max_grids",           "max_grids",          int),
        ("grid_group",          "grid_group",          "grid_group",          str),
        ("grid_tag",            "grid_tag",            "grid_tag",            str),
        ("max_experiments",     "max_experiments",     "max_experiments",     int),
        ("max_samples_per_exp", "max_samples_per_exp", "max_samples_per_exp", int),
        ("psd_nfft",            "psd_nfft",            "psd_nfft",            int),
        ("max_dist_samples",    "max_dist_samples",    "max_dist_samples",    int),
        ("gauss_alpha",         "gauss_alpha",         "gauss_alpha",         float),
        ("dist_tol_m",          "dist_tol_m",          "dist_tol_m",          float),
        ("curr_tol_mA",         "curr_tol_mA",         "curr_tol_mA",         float),
        ("keras_verbose",       "keras_verbose",       "keras_verbose",       int),
    ]

    for ov_key, pg_key, cli_attr, cast in _MAP:
        # CLI takes precedence
        cli_val = getattr(cli_args, cli_attr, None)
        if cli_val is not None:
            ov[ov_key] = cast(cli_val)
        elif pg.get(pg_key) is not None:
            ov[ov_key] = cast(pg[pg_key])

    if cli_args.dry_run:
        ov["dry_run"] = True

    return ov


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


def _quick_cvae_predict(run_dir: Path, X_va, D_va, C_va, batch_size: int = 4096):
    """
    Load best_model_full.keras **strictly** from *run_dir*/models/,
    build a deterministic inference model (prior → decoder → y_mean),
    and predict on the given validation arrays.

    Commit 3P: removed fallback to state_run.json to avoid cross-regime
    model leakage.  Fails loudly if model is missing.

    Returns Y_pred (ndarray) or None on failure.
    """
    import numpy as _np

    model_path = run_dir / "models" / "best_model_full.keras"
    if not model_path.exists():
        print(f"⚠️  best_model_full.keras not found at {model_path}")
        return None
    print(f"   📂 Loading model: {model_path}")

    # --- Shape guard (Commit 3P) ---
    for name, arr in [("X_va", X_va), ("D_va", D_va), ("C_va", C_va)]:
        assert arr is not None, f"_quick_cvae_predict: {name} is None"
    n = X_va.shape[0]
    assert D_va.shape[0] == n and C_va.shape[0] == n, (
        f"Shape mismatch: X_va={X_va.shape}, D_va={D_va.shape}, C_va={C_va.shape}"
    )

    try:
        import tensorflow as tf
        from src.models.cvae_components import Sampling, CondPriorVAELoss

        custom_objects = {"Sampling": Sampling, "CondPriorVAELoss": CondPriorVAELoss}
        vae = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False)

        prior = vae.get_layer("prior_net")
        decoder = vae.get_layer("decoder")

        layers = tf.keras.layers
        x_in = layers.Input(shape=(2,), name="x_input")
        d_in = layers.Input(shape=(1,), name="distance_input")
        c_in = layers.Input(shape=(1,), name="current_input")

        prior_out = prior([x_in, d_in, c_in])
        z_mean_p = prior_out[0] if isinstance(prior_out, (list, tuple)) else prior_out
        z = z_mean_p  # deterministic

        cond = layers.Concatenate()([x_in, d_in, c_in])
        out_params = decoder([z, cond])
        y_mean = layers.Lambda(lambda t: t[:, :2], name="y_mean_quick")(out_params)

        inference_model = tf.keras.Model([x_in, d_in, c_in], y_mean, name="quick_infer_det")
        Y_pred = inference_model.predict([X_va, D_va, C_va], batch_size=batch_size, verbose=0)

        del vae, inference_model, prior, decoder
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        return Y_pred
    except Exception as e:
        print(f"⚠️  Quick cVAE inference failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_regime(
    regime: dict,
    dataset_root: str,
    base_overrides: dict,
    protocol_dir: Path,
    skip_eval: bool = False,
    run_baseline: bool = True,
    run_dist_metrics: bool = True,
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
    ov = dict(base_overrides)

    # If regime specifies experiment_paths, we filter via max_experiments = len
    # AND set the DATASET_ROOT to the parent so only those experiments are found.
    # However, the current loader discovers ALL experiments under DATASET_ROOT.
    # We handle filtering by passing experiment_paths in overrides for
    # future use, but for now we rely on max_experiments for bounding.
    if exp_paths:
        ov["_experiment_paths"] = exp_paths
    if exp_regex:
        ov["_experiment_regex"] = exp_regex

    result = {
        "regime_id": regime_id,
        "regime_label": regime.get("regime_label", ""),
        "description": regime.get("description", ""),
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
        "dist_metrics_source": None,      # Commit 3S: "eval" | "quick" | "quick_fallback" | None
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
    _need_data = (run_baseline or run_dist_metrics) and not ov.get("dry_run", False)

    if _need_data:
        try:
            from src.data.loading import load_experiments_as_list
            from src.protocol.split_strategies import apply_split

            _val_split = float(ov.get("val_split", 0.2))
            _seed = int(ov.get("seed", 42))
            _max_spe = ov.get("max_samples_per_exp")

            # Use already-filtered experiments from above
            exps = list(_filt_exps)
            if _max_spe is not None:
                _ms = int(_max_spe)
                exps = [(X[:_ms], Y[:_ms], D[:_ms], C[:_ms], p) for X, Y, D, C, p in exps]

            X_tr, Y_tr, _D_tr, _C_tr, X_va, Y_va, D_va, C_va, _ = \
                apply_split(exps, strategy="per_experiment",
                            val_split=_val_split, seed=_seed)

            # --- Commit 3P: shape guard ---
            assert X_va.shape[0] == Y_va.shape[0] == D_va.shape[0] == C_va.shape[0], (
                f"Shape mismatch after split: X_va={X_va.shape}, Y_va={Y_va.shape}, "
                f"D_va={D_va.shape}, C_va={C_va.shape}"
            )

            _val_data = (X_va.copy(), Y_va.copy(), D_va.copy(), C_va.copy())  # Commit 3P: isolate
            _X_tr, _Y_tr = X_tr, Y_tr
            print(f"   train={len(X_tr):,}  val={len(X_va):,}")

            # Commit 3P: data fingerprint for debugging
            import numpy as _np_fp
            print(f"   🔑 val fingerprint: X mean={_np_fp.mean(X_va):.6f} "
                  f"std={_np_fp.std(X_va):.6f} D_unique={_np_fp.unique(D_va).tolist()} "
                  f"C_unique={_np_fp.unique(C_va).tolist()}")
            del _D_tr, _C_tr, exps, X_tr, Y_tr, X_va, Y_va, D_va, C_va
        except Exception as e:
            print(f"⚠️  Data loading failed for regime '{regime_id}': {e}")

    del _filt_exps  # Commit 3R: free filtered arrays

    # ---- BASELINE (Commit 3N) + dist-metrics (Commit 3O) ----
    if run_baseline and _val_data is not None:
        try:
            import time as _time
            from src.baselines.deterministic import run_deterministic_baseline

            _keras_v = int(ov.get("keras_verbose", 0))
            X_va, Y_va, D_va, C_va = _val_data

            _bl_t0 = _time.time()
            bl_metrics = run_deterministic_baseline(
                _X_tr, _Y_tr, X_va, Y_va,
                config={"verbose": _keras_v, "return_predictions": run_dist_metrics},
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

    # ---- TRAINING ----
    print(f"\n{'='*70}")
    print(f"🔬 REGIME: {regime_id} — {regime.get('description', '')}")
    print(f"📁 DATASET_ROOT (effective) = {dataset_root}")
    print(f"{'='*70}")

    # --- Commit 3M: always use absolute path for DATASET_ROOT ---
    _ds_root = str(Path(dataset_root).resolve())
    os.environ["DATASET_ROOT"] = _ds_root
    os.environ["OUTPUT_BASE"] = str(protocol_dir.resolve())
    os.environ["RUN_ID"] = run_id

    _cvae_t0 = time.time()
    try:
        from src.training import cvae_TRAIN_documented as train_module
        print(f"\n📦 Training regime '{regime_id}' → run_id={run_id}")
        train_module.main(overrides=ov)
        result["train_status"] = "completed"
    except Exception as e:
        result["train_status"] = "failed"
        result["error"] = f"train: {e}\n{traceback.format_exc()}"
        print(f"❌ Training failed for regime '{regime_id}': {e}")

    result["cvae_time_s"] = round(time.time() - _cvae_t0, 2)
    run_dir = protocol_dir / run_id
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

            eval_ov = {}
            for k in ("max_experiments", "max_samples_per_exp", "psd_nfft",
                      "_selected_experiments"):
                if k in ov:
                    eval_ov[k] = ov[k]

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

    # ---- cVAE DISTRIBUTION-FIDELITY METRICS (Commit 3O, 3P, 3S) ----
    # Commit 3S: always use shared _val_data + _quick_cvae_predict so
    # baseline and cVAE are evaluated on the SAME validation split.
    # When eval ran successfully → dist_metrics_source="eval".
    # When eval skipped/failed  → dist_metrics_source="quick".
    # Guardrail: if eval ran but model is missing → "quick_fallback".
    if run_dist_metrics and result["train_status"] == "completed":
        _dm_source = None  # will be set below
        try:
            import numpy as _np_dm
            _psd_nfft = int(ov.get("psd_nfft", 2048))
            _max_ds = int(ov.get("max_dist_samples", 200_000))
            _g_alpha = float(ov.get("gauss_alpha", 0.01))

            if _val_data is None:
                raise RuntimeError("Shared validation data not available")

            _X_va, _Y_va, _D_va, _C_va = _val_data
            model_check = run_dir / "models" / "best_model_full.keras"

            if not model_check.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_check}")

            # Decide dist_metrics_source label
            if _eval_ran and result["eval_status"] == "completed":
                _dm_source = "eval"
            elif skip_eval:
                _dm_source = "quick"
            else:
                # eval was enabled but failed
                _dm_source = "quick_fallback"
                print(f"⚠️  Eval enabled but status={result['eval_status']} "
                      f"for regime '{regime_id}' — falling back to quick dist-metrics")

            print(f"\n🔍 cVAE dist-metrics for regime '{regime_id}' "
                  f"({len(_X_va):,} val pts, source={_dm_source}, model={model_check})")
            Y_pred_cvae = _quick_cvae_predict(run_dir, _X_va, _D_va, _C_va)
            if Y_pred_cvae is not None:
                from src.metrics.distribution import residual_fidelity_metrics
                res_real = _Y_va - _X_va
                res_pred_cvae = Y_pred_cvae - _X_va
                cvae_dm = residual_fidelity_metrics(
                    res_real, res_pred_cvae,
                    psd_nfft=_psd_nfft, max_samples=_max_ds, gauss_alpha=_g_alpha,
                )
                # --- Commit 3P: finiteness guard ---
                _finite_keys = ["delta_mean_l2", "delta_cov_fro", "delta_skew_l2",
                                "delta_kurt_l2", "psd_l2"]
                for _fk in _finite_keys:
                    v = cvae_dm.get(_fk)
                    if v is not None and not _np_dm.isfinite(v):
                        print(f"⚠️  Non-finite dist metric {_fk}={v} for regime '{regime_id}'")
                result["cvae_dist"] = cvae_dm
                result["dist_metrics_source"] = _dm_source
                print(f"   📐 cVAE dist ({_dm_source}): "
                      f"Δmean_l2={cvae_dm['delta_mean_l2']:.4f}  "
                      f"psd_l2={cvae_dm['psd_l2']:.4f}  "
                      f"reject_gauss={cvae_dm['reject_gaussian']}")
                del Y_pred_cvae, res_pred_cvae, cvae_dm
            else:
                print(f"⚠️  cVAE quick_predict returned None for regime '{regime_id}'")
            del _X_va, _Y_va, _D_va, _C_va
        except Exception as e:
            result["cvae_dist"] = {"error": str(e)}
            if _dm_source is not None:
                result["dist_metrics_source"] = _dm_source
            print(f"⚠️  cVAE dist metrics failed for regime '{regime_id}': {e}")

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
            "baseline_reject_gauss": bd.get("reject_gaussian"),
            # Commit 3O: distribution-fidelity — cVAE
            "cvae_delta_mean_l2": cd.get("delta_mean_l2"),
            "cvae_delta_cov_fro": cd.get("delta_cov_fro"),
            "cvae_delta_skew_l2": cd.get("delta_skew_l2"),
            "cvae_delta_kurt_l2": cd.get("delta_kurt_l2"),
            "cvae_psd_l2": cd.get("psd_l2"),
            "cvae_jb_p_min": cd.get("jb_p_min"),
            "cvae_reject_gauss": cd.get("reject_gaussian"),
            # Commit 3Q: regime-aware experiment selection
            "n_experiments_selected": len(r.get("selected_experiments", [])),
            "dist_target_m": r.get("selection_criteria", {}).get("distance_m"),
            "curr_target_mA": r.get("selection_criteria", {}).get("current_mA"),
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
    proto_globals = protocol.get("global_settings", {})
    regimes = protocol["regimes"]

    # Merge protocol globals + CLI overrides
    base_overrides = _merge_overrides(proto_globals, args)

    # Experiment output directory (single folder per protocol run)
    exp_dir = Path(args.output_base) / f"exp_{ts_label}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "tables").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)

    studies_meta = protocol.get("_studies", [])

    print(f"🚀 Protocol runner — {len(studies_meta)} study(ies), {len(regimes)} regime(s)")
    print(f"📁 Experiment dir: {exp_dir}")
    print(f"🔧 Base overrides: {base_overrides}")

    # Save a copy of the protocol used (always the resolved dict as JSON)
    (exp_dir / "logs" / "protocol_input.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
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
                skip_eval=args.skip_eval,
                run_baseline=not args.no_baseline,
                run_dist_metrics=not args.no_dist_metrics,
            )
            r["_study"] = sname
            results.append(r)

    # ---- Build summary table ----
    import pandas as pd
    df_summary = build_summary_table(results)

    summary_csv = exp_dir / "tables" / "summary_by_regime.csv"
    summary_xlsx = exp_dir / "tables" / "summary_by_regime.xlsx"
    df_summary.to_csv(summary_csv, index=False)
    df_summary.to_excel(summary_xlsx, index=False)
    print(f"\n📊 Summary table: {summary_csv}")

    # ---- Write manifest ----
    ts_end = datetime.now()
    _psd_nfft_eff = int(base_overrides.get("psd_nfft", 2048))
    _ga_eff = float(base_overrides.get("gauss_alpha", 0.01))
    _mds_eff = int(base_overrides.get("max_dist_samples", 200_000))
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
            "skip_eval": args.skip_eval,
            "no_baseline": args.no_baseline,
            "no_dist_metrics": args.no_dist_metrics,
            "dry_run": args.dry_run,
        },
        "baseline_config": {
            "model": "deterministic_mlp",
            "hidden": [128, 64],
            "epochs": 50,
            "loss": "mse",
            "enabled": not args.no_baseline,
        },
        "dist_metrics_config": {
            "enabled": not args.no_dist_metrics,
            "psd_nfft": _psd_nfft_eff,
            "gauss_alpha": _ga_eff,
            "max_dist_samples": _mds_eff,
        },
        "base_overrides": base_overrides,
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
                "selected_experiments": r.get("selected_experiments", []),
                "selection_criteria": r.get("selection_criteria", {}),
                "error": r.get("error"),
            }
            for r in results
        ],
    }
    manifest_path = exp_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
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
        print(f"   • {slab}{r['regime_id']}: {status}{delta}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
