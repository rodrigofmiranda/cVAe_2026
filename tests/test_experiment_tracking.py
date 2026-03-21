from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from src.protocol.experiment_tracking import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_RUNNING,
    inspect_protocol_experiment,
    latest_complete_protocol_experiment,
    prune_stale_incomplete_protocol_experiments,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_complete_protocol_run(base: Path, run_id: str, *, timestamp_end: str) -> Path:
    exp_dir = base / run_id
    (exp_dir / "tables").mkdir(parents=True, exist_ok=True)
    (exp_dir / "train" / "tables").mkdir(parents=True, exist_ok=True)
    (exp_dir / "train" / "models").mkdir(parents=True, exist_ok=True)
    (exp_dir / "tables" / "summary_by_regime.csv").write_text("regime_id\nr0\n", encoding="utf-8")
    (exp_dir / "tables" / "protocol_leaderboard.csv").write_text("rank\n1\n", encoding="utf-8")
    (exp_dir / "train" / "state_run.json").write_text("{}", encoding="utf-8")
    (exp_dir / "train" / "tables" / "gridsearch_results.csv").write_text("rank,tag\n1,best\n", encoding="utf-8")
    (exp_dir / "train" / "tables" / "grid_training_diagnostics.csv").write_text("rank,tag\n1,best\n", encoding="utf-8")
    (exp_dir / "train" / "models" / "best_model_full.keras").write_text("stub", encoding="utf-8")
    _write_json(
        exp_dir / "manifest.json",
        {
            "run_type": "protocol_experiment",
            "run_status": RUN_STATUS_COMPLETED,
            "timestamp_start": "2026-03-21T10:00:00",
            "timestamp_end": timestamp_end,
            "execution_mode": "train_once_eval_all",
            "args": {"stat_tests": False},
        },
    )
    return exp_dir


def test_latest_complete_protocol_experiment_ignores_running_runs(tmp_path: Path):
    outputs = tmp_path / "outputs"
    _make_complete_protocol_run(outputs, "exp_20260321_100000", timestamp_end="2026-03-21T10:10:00")
    newer = outputs / "exp_20260321_120000"
    _write_json(
        newer / "manifest.json",
        {
            "run_type": "protocol_experiment",
            "run_status": RUN_STATUS_RUNNING,
            "timestamp_start": "2026-03-21T12:00:00",
            "timestamp_end": None,
            "args": {"stat_tests": False},
        },
    )

    out = latest_complete_protocol_experiment(outputs)

    assert out.name == "exp_20260321_100000"


def test_inspect_protocol_experiment_marks_completed_only_when_artifacts_exist(tmp_path: Path):
    outputs = tmp_path / "outputs"
    exp_dir = outputs / "exp_20260321_130000"
    _write_json(
        exp_dir / "manifest.json",
        {
            "run_type": "protocol_experiment",
            "run_status": RUN_STATUS_COMPLETED,
            "timestamp_start": "2026-03-21T13:00:00",
            "timestamp_end": "2026-03-21T13:05:00",
            "args": {"stat_tests": False},
        },
    )

    info = inspect_protocol_experiment(exp_dir)

    assert info["status"] != RUN_STATUS_COMPLETED
    assert "tables/summary_by_regime.csv" in info["missing_artifacts"]


def test_prune_stale_incomplete_protocol_experiments_removes_old_running_runs(tmp_path: Path):
    outputs = tmp_path / "outputs"
    stale = outputs / "exp_20260320_010000"
    stale.mkdir(parents=True, exist_ok=True)
    _write_json(
        stale / "manifest.json",
        {
            "run_type": "protocol_experiment",
            "run_status": RUN_STATUS_RUNNING,
            "timestamp_start": (datetime.utcnow() - timedelta(hours=48)).isoformat(timespec="seconds"),
            "timestamp_end": None,
            "args": {"stat_tests": False},
        },
    )

    actions = prune_stale_incomplete_protocol_experiments(
        outputs,
        older_than_hours=24,
        dry_run=False,
    )

    assert len(actions) == 1
    assert actions[0]["deleted"] is True
    assert not stale.exists()
