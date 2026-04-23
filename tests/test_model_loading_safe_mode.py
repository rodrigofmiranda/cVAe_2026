from pathlib import Path

from src.models import cvae_sequence as seq_mod


def test_load_seq_model_retries_with_safe_mode_disabled_for_lambda_block(monkeypatch, tmp_path: Path):
    calls = []

    def _fake_load_model(path, **kwargs):
        calls.append((path, kwargs))
        if len(calls) == 1:
            raise ValueError(
                "Requested the deserialization of a Lambda layer with a Python "
                "lambda inside it. This carries a potential risk of arbitrary "
                "code execution and thus it is disallowed by default. If you "
                "trust the source of the saved model, you can pass "
                "safe_mode=False to the loading function in order to allow "
                "Lambda layer loading."
            )
        return "loaded-ok"

    monkeypatch.setattr(seq_mod.tf.keras.models, "load_model", _fake_load_model)

    model_path = tmp_path / "best_model_full.keras"
    model_path.write_text("placeholder", encoding="utf-8")

    out = seq_mod.load_seq_model(str(model_path))

    assert out == "loaded-ok"
    assert len(calls) == 2
    assert calls[0][0] == str(model_path)
    assert calls[0][1]["compile"] is False
    assert "safe_mode" not in calls[0][1]
    assert calls[1][1]["safe_mode"] is False


def test_load_seq_model_does_not_retry_for_unrelated_errors(monkeypatch, tmp_path: Path):
    def _fake_load_model(_path, **_kwargs):
        raise OSError("corrupted file")

    monkeypatch.setattr(seq_mod.tf.keras.models, "load_model", _fake_load_model)

    model_path = tmp_path / "best_model_full.keras"
    model_path.write_text("placeholder", encoding="utf-8")

    try:
        seq_mod.load_seq_model(str(model_path))
    except OSError as exc:
        assert "corrupted file" in str(exc)
    else:
        raise AssertionError("Expected unrelated load failure to be re-raised")
