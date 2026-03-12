from __future__ import annotations

"""
Verify the mandatory pipeline fixes on a protocol run.

Usage:
    python scripts/verify_pipeline_fixes.py
    python scripts/verify_pipeline_fixes.py outputs/exp_YYYYMMDD_HHMMSS

If no run is provided, the most recent ``outputs/exp_*`` run is used.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _latest_exp_dir() -> Path:
    candidates = sorted(ROOT.glob("outputs/exp_*/manifest.json"))
    if not candidates:
        raise FileNotFoundError("No run found under outputs/exp_*/manifest.json")
    return candidates[-1].parent


def _resolve_exp_dir(raw: str | None) -> Path:
    if raw is None:
        return _latest_exp_dir()
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if path.is_file() and path.name == "manifest.json":
        return path.parent
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_run_dir(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _shuffle_train_batches_status(manifest: dict) -> tuple[bool | None, str]:
    base_overrides = manifest.get("base_overrides", {})
    if "shuffle_train_batches" in base_overrides:
        return bool(base_overrides["shuffle_train_batches"]), "manifest.base_overrides"

    values = []
    for regime in manifest.get("regimes", []):
        run_dir = _resolve_run_dir(regime.get("run_dir"))
        if run_dir is None:
            continue
        state_path = run_dir / "state_run.json"
        if not state_path.exists():
            continue
        state = _read_json(state_path)
        values.append(state.get("training_config", {}).get("shuffle_train_batches"))

    if not values:
        return None, "unavailable"
    return all(v is True for v in values), "state_run.training_config"


def _print_fix7_sources(manifest: dict, errors: list[str]) -> None:
    valid_sources = {"quick", "quick_fallback", "eval", "eval_reanalysis"}
    for regime in manifest.get("regimes", []):
        regime_id = str(regime.get("regime_id", "?"))
        source = str(regime.get("dist_metrics_source", "") or "").strip()
        has_dist = bool(regime.get("cvae_dist")) or bool(regime.get("stat_fidelity"))
        status = "OK" if source in valid_sources else "WARN"
        print(f"Fix 7 - dist_metrics_source [{regime_id}]: {source or '<missing>'} [{status}]")
        if has_dist and source not in valid_sources:
            errors.append(
                f"Fix7: dist_metrics_source missing/invalid for regime {regime_id}: {source!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify mandatory pipeline fixes on a run.")
    parser.add_argument(
        "exp_dir",
        nargs="?",
        help="Run directory or manifest.json path. Defaults to the latest outputs/exp_* run.",
    )
    args = parser.parse_args()

    exp_dir = _resolve_exp_dir(args.exp_dir)
    manifest_path = exp_dir / "manifest.json"
    summary_path = exp_dir / "tables" / "summary_by_regime.csv"
    sf_path = exp_dir / "tables" / "stat_fidelity_by_regime.csv"

    if not manifest_path.exists():
        print(f"FAIL: manifest.json not found in {exp_dir}")
        sys.exit(1)

    manifest = _read_json(manifest_path)
    print(f"=== Verifying: {exp_dir.name} ===")
    print()

    errors: list[str] = []

    if summary_path.exists():
        df = pd.read_csv(summary_path)
        required = {"var_real_delta", "var_pred_delta"}
        if required.issubset(df.columns):
            var_real = pd.to_numeric(df["var_real_delta"], errors="coerce")
            var_pred = pd.to_numeric(df["var_pred_delta"], errors="coerce")
            mask = np.isfinite(var_real) & np.isfinite(var_pred) & (var_real > 0)
            ratio = (var_pred[mask] / var_real[mask]).dropna()
            if len(ratio) == 0:
                errors.append("Fix1: no finite var_pred_delta/var_real_delta ratios available")
            else:
                max_ratio = float(ratio.max())
                status = "OK" if max_ratio < 5.0 else "FAIL"
                print(f"Fix 1 - var_pred/var_real max: {max_ratio:.3f} [{status}]")
                if max_ratio >= 5.0:
                    errors.append(
                        f"Fix1: var_pred_delta/var_real_delta max={max_ratio:.3f} >= 5.0"
                    )
        else:
            errors.append("Fix1: var_real_delta/var_pred_delta columns missing in summary_by_regime.csv")
    else:
        errors.append("Fix1: summary_by_regime.csv not found")

    shuffle_ok, shuffle_source = _shuffle_train_batches_status(manifest)
    shuffle_status = "OK" if shuffle_ok else "FAIL"
    print(
        f"Fix 2 - shuffle_train_batches ({shuffle_source}): "
        f"{shuffle_ok if shuffle_ok is not None else '<missing>'} [{shuffle_status}]"
    )
    if shuffle_ok is not True:
        errors.append("Fix2: shuffle_train_batches is not True in the available run metadata")

    if sf_path.exists():
        sf = pd.read_csv(sf_path)
        if "mmd2_normalized" in sf.columns:
            n_nan = int(sf["mmd2_normalized"].isna().sum())
            status = "OK" if n_nan < len(sf) else "FAIL"
            print(f"Fix 9 - mmd2_normalized: present, NaNs={n_nan}/{len(sf)} [{status}]")
            if n_nan == len(sf):
                errors.append("Fix9: mmd2_normalized is all-NaN (summary/stat merge likely failed)")
        else:
            errors.append("Fix9: mmd2_normalized column missing in stat_fidelity_by_regime.csv")
    else:
        print("Fix 9 - stat_fidelity_by_regime.csv missing [WARN: run may have been executed without --stat_tests]")

    _print_fix7_sources(manifest, errors)

    print()
    print("=" * 50)
    if errors:
        print("FAIL:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("OK: all requested pipeline fixes were verified successfully.")


if __name__ == "__main__":
    main()
