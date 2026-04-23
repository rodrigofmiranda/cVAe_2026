from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


RUN_TYPE_PROTOCOL = "protocol_experiment"
RUN_STATUS_RUNNING = "running"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED = "failed"
RUN_STATUS_INTERRUPTED = "interrupted"
RUN_STATUS_INCOMPLETE = "incomplete"
LATEST_COMPLETED_EXPERIMENT_JSON = "_latest_completed_experiment.json"


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _to_utc_naive(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_iso_dt(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        return _to_utc_naive(datetime.fromisoformat(text))
    except ValueError:
        return None


def _protocol_required_artifacts(exp_dir: Path, manifest: Optional[dict]) -> List[str]:
    raw_status = ""
    if isinstance(manifest, dict):
        raw_status = str(manifest.get("run_status", "")).strip().lower()

    # Legacy runs did not carry explicit lifecycle status and often predate the
    # canonical protocol leaderboard/training-dashboard bundle. For those runs,
    # treat manifest + summary as the completion contract so they remain
    # queryable and are never mistaken for disposable incomplete runs.
    if not raw_status:
        return [
            "manifest.json",
            "tables/summary_by_regime.csv",
        ]

    required = [
        "manifest.json",
        "tables/summary_by_regime.csv",
        "tables/protocol_leaderboard.csv",
    ]
    args = manifest.get("args", {}) if isinstance(manifest, dict) else {}
    if bool(args.get("stat_tests")):
        required.append("tables/stat_fidelity_by_regime.csv")

    if (exp_dir / "train").exists():
        required.extend(
            [
                "train/state_run.json",
                "train/tables/gridsearch_results.csv",
                "train/tables/grid_training_diagnostics.csv",
                "train/models/best_model_full.keras",
            ]
        )

    return sorted(set(required))


def inspect_protocol_experiment(exp_dir: str | Path) -> Dict[str, Any]:
    exp_dir = Path(exp_dir).resolve()
    manifest_path = exp_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    required = _protocol_required_artifacts(exp_dir, manifest)
    missing = [rel for rel in required if not (exp_dir / rel).exists()]

    raw_status = ""
    if isinstance(manifest, dict):
        raw_status = str(manifest.get("run_status", "")).strip().lower()

    status = RUN_STATUS_INCOMPLETE
    if manifest is None:
        status = RUN_STATUS_INCOMPLETE
    elif raw_status == RUN_STATUS_COMPLETED:
        status = RUN_STATUS_COMPLETED if not missing else RUN_STATUS_INCOMPLETE
    elif raw_status == RUN_STATUS_FAILED:
        status = RUN_STATUS_FAILED
    elif raw_status == RUN_STATUS_INTERRUPTED:
        status = RUN_STATUS_INTERRUPTED
    elif raw_status == RUN_STATUS_RUNNING:
        status = RUN_STATUS_RUNNING
    elif manifest.get("error"):
        status = RUN_STATUS_FAILED
    elif not missing and manifest.get("timestamp_end"):
        status = RUN_STATUS_COMPLETED

    started_at = None
    ended_at = None
    if isinstance(manifest, dict):
        started_at = _parse_iso_dt(manifest.get("timestamp_start"))
        ended_at = _parse_iso_dt(manifest.get("timestamp_end"))
    stat = exp_dir.stat()
    mtime = datetime.utcfromtimestamp(stat.st_mtime)
    sort_time = ended_at or started_at or mtime

    return {
        "exp_dir": exp_dir,
        "run_id": exp_dir.name,
        "manifest": manifest,
        "raw_status": raw_status or None,
        "status": status,
        "missing_artifacts": missing,
        "required_artifacts": required,
        "timestamp_start": started_at,
        "timestamp_end": ended_at,
        "mtime": mtime,
        "sort_time": sort_time,
        "is_complete": status == RUN_STATUS_COMPLETED,
    }


def latest_complete_protocol_experiment(outputs_dir: str | Path) -> Path:
    outputs_dir = Path(outputs_dir).resolve()
    candidates = []
    for exp_dir in outputs_dir.glob("exp_*"):
        if not exp_dir.is_dir():
            continue
        info = inspect_protocol_experiment(exp_dir)
        if info["is_complete"]:
            candidates.append(info)
    if not candidates:
        raise FileNotFoundError(f"No completed experiment found under {outputs_dir}")
    candidates.sort(key=lambda info: (info["sort_time"], info["run_id"]))
    return Path(candidates[-1]["exp_dir"])


def write_latest_completed_experiment_record(
    outputs_dir: str | Path,
    exp_dir: str | Path,
) -> Path:
    outputs_dir = Path(outputs_dir).resolve()
    info = inspect_protocol_experiment(exp_dir)
    manifest = info.get("manifest") if isinstance(info.get("manifest"), dict) else {}
    payload = {
        "run_type": RUN_TYPE_PROTOCOL,
        "run_status": info["status"],
        "run_id": info["run_id"],
        "path": str(info["exp_dir"]),
        "timestamp_start": (
            info["timestamp_start"].isoformat(timespec="seconds")
            if info["timestamp_start"] is not None
            else None
        ),
        "timestamp_end": (
            info["timestamp_end"].isoformat(timespec="seconds")
            if info["timestamp_end"] is not None
            else None
        ),
        "missing_artifacts": list(info["missing_artifacts"]),
        "required_artifacts": list(info["required_artifacts"]),
        "git_commit": manifest.get("git_commit"),
        "git_branch": manifest.get("git_branch"),
    }
    return _write_json(outputs_dir / LATEST_COMPLETED_EXPERIMENT_JSON, payload)


def prune_stale_incomplete_protocol_experiments(
    outputs_dir: str | Path,
    *,
    older_than_hours: float = 24.0,
    dry_run: bool = True,
    remove_failed: bool = False,
) -> List[Dict[str, Any]]:
    outputs_dir = Path(outputs_dir).resolve()
    cutoff = datetime.utcnow() - timedelta(hours=float(older_than_hours))
    removable = {RUN_STATUS_RUNNING, RUN_STATUS_INCOMPLETE}
    if remove_failed:
        removable.add(RUN_STATUS_FAILED)

    actions: List[Dict[str, Any]] = []
    for exp_dir in sorted(outputs_dir.glob("exp_*")):
        if not exp_dir.is_dir():
            continue
        info = inspect_protocol_experiment(exp_dir)
        # Only auto-prune runs that are clearly modern/instrumented (explicit
        # lifecycle status) or runs with no manifest at all. Legacy manifests
        # are intentionally preserved unless removed manually.
        if info["manifest"] is not None and not info["raw_status"]:
            continue
        if info["status"] not in removable:
            continue
        age_ref = info["timestamp_start"] or info["mtime"]
        if age_ref > cutoff:
            continue

        action = {
            "run_id": info["run_id"],
            "path": str(info["exp_dir"]),
            "status": info["status"],
            "timestamp_start": (
                info["timestamp_start"].isoformat(timespec="seconds")
                if info["timestamp_start"] is not None
                else None
            ),
            "missing_artifacts": list(info["missing_artifacts"]),
            "deleted": False,
        }
        if not dry_run:
            shutil.rmtree(info["exp_dir"], ignore_errors=False)
            action["deleted"] = True
        actions.append(action)

    return actions
