import json
from pathlib import Path

from src.training.logging import RunPaths, write_state_run


def test_runpaths_redirects_logs_to_shared_root(tmp_path: Path):
    run_dir = tmp_path / "exp_001" / "global_model"
    shared_logs = tmp_path / "exp_001" / "logs" / "global_model"

    rp = RunPaths(run_id="global_model", run_dir=run_dir, logs_dir=shared_logs)
    out = rp.write_json("logs/training_history.json", {"ok": True})

    assert out == shared_logs / "training_history.json"
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == {"ok": True}
    assert not (run_dir / "logs").exists()


def test_write_state_run_uses_explicit_logs_dir(tmp_path: Path):
    run_dir = tmp_path / "exp_001" / "global_model"
    shared_logs = tmp_path / "exp_001" / "logs" / "global_model"
    run_dir.mkdir(parents=True, exist_ok=True)

    state_path = write_state_run(
        run_dir,
        run_id="global_model",
        dataset_root="/tmp/data",
        output_base=str(tmp_path),
        logs_dir=shared_logs,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["paths"]["logs"] == str(shared_logs.resolve())
    assert state["paths"]["models"] == str(run_dir / "models")
