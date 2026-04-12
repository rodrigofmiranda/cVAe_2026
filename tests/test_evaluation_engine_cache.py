from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

from src.evaluation.engine import (
    _get_eval_cached_entry,
    clear_evaluation_model_cache,
)


def test_evaluation_engine_caches_model_and_inference(monkeypatch, tmp_path: Path):
    import src.evaluation.engine as engine_mod

    calls = {"load": 0, "infer": 0}

    class _DummyVae:
        pass

    def _fake_load(_path):
        calls["load"] += 1
        return _DummyVae()

    def _fake_infer(_vae, deterministic):
        calls["infer"] += 1
        return f"infer-{deterministic}"

    monkeypatch.setattr(engine_mod, "load_seq_model", _fake_load)
    monkeypatch.setattr(engine_mod, "create_inference_model_from_full", _fake_infer)

    model_path = tmp_path / "best_model_full.keras"
    model_path.write_text("placeholder", encoding="utf-8")

    clear_evaluation_model_cache()
    e1 = _get_eval_cached_entry(model_path)
    e2 = _get_eval_cached_entry(model_path)
    clear_evaluation_model_cache()

    assert e1 is e2
    assert e1["inference_det"] == "infer-True"
    assert e1["inference_sto"] == "infer-False"
    assert calls == {"load": 1, "infer": 2}
