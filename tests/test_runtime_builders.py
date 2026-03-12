import json

from src.config.runtime import build_evaluation_runtime, build_training_runtime
from src.config.schema import AnalysisConfig, EvalProtocolConfig
from src.training.engine import train_engine
from src.training.logging import write_state_run


def test_build_training_runtime_applies_overrides_and_defaults():
    runtime = build_training_runtime(
        "data/dataset_fullsquare_organized",
        "outputs",
        run_id="run_test",
        overrides={
            "max_epochs": 7,
            "val_split": 0.3,
            "seed": 123,
            "_split_strategy": "grouped",
        },
    )

    assert runtime.run_id == "run_test"
    assert runtime.training_config["epochs"] == 7
    assert runtime.training_config["validation_split"] == 0.3
    assert runtime.training_config["seed"] == 123
    assert runtime.training_config["split_mode"] == "grouped"
    assert runtime.analysis_quick == AnalysisConfig.from_dict({}).to_dict()
    assert runtime.eval_protocol == EvalProtocolConfig.from_dict({}).to_dict()
    assert runtime.data_reduction_config["enabled"] is True


def test_build_evaluation_runtime_roundtrip_uses_state_dataset_root(tmp_path, monkeypatch):
    monkeypatch.delenv("DATASET_ROOT", raising=False)
    monkeypatch.delenv("OUTPUT_BASE", raising=False)
    monkeypatch.delenv("RUN_ID", raising=False)

    run_dir = tmp_path / "run_0001"
    state_path = write_state_run(
        run_dir,
        run_id="run_0001",
        dataset_root="/tmp/dataset_root",
        output_base="/tmp/outputs",
        training_config={"seed": 7, "validation_split": 0.2},
        data_reduction_config={"enabled": True},
        analysis_quick=AnalysisConfig.from_dict({}).to_dict(),
        normalization={"D_min": 1.0, "D_max": 1.5, "C_min": 300.0, "C_max": 900.0},
        data_split={"split_mode": "per_experiment", "validation_split": 0.2, "seed": 7},
        eval_protocol=EvalProtocolConfig.from_dict({}).to_dict(),
        grid={"n_models": 1},
        artifacts={"best_model_full": str(run_dir / "models" / "best_model_full.keras")},
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))

    runtime = build_evaluation_runtime(run_dir=run_dir, state=state)

    assert str(runtime.dataset_root) == "/tmp/dataset_root"
    assert runtime.eval_protocol["eval_slice"] == "stratified"
    assert runtime.training_config["seed"] == 7


def test_train_engine_does_not_require_env_vars(monkeypatch):
    monkeypatch.delenv("DATASET_ROOT", raising=False)
    monkeypatch.delenv("OUTPUT_BASE", raising=False)
    monkeypatch.delenv("RUN_ID", raising=False)

    captured = {}

    def _fake_pipeline(*, dataset_root, output_base, run_id=None, overrides=None):
        captured["dataset_root"] = dataset_root
        captured["output_base"] = output_base
        captured["run_id"] = run_id
        captured["overrides"] = dict(overrides or {})
        return {"status": "dry_run", "run_id": run_id or "run_x", "run_dir": "/tmp/run_x"}

    monkeypatch.setattr("src.training.pipeline.run_training_pipeline", _fake_pipeline)

    summary = train_engine(
        dataset_root="data/dataset_fullsquare_organized",
        output_base="outputs",
        run_id="run_explicit",
        overrides={"max_epochs": 2},
    )

    assert summary["status"] == "dry_run"
    assert captured["run_id"] == "run_explicit"
    assert captured["overrides"] == {"max_epochs": 2}
