from __future__ import annotations

import pytest

from src.training.train import _migration_message, main


def test_migration_message_points_to_protocol_run() -> None:
    msg = _migration_message(
        [
            "python",
            "--dataset_root",
            "data/dataset_fullsquare_organized",
            "--output_base",
            "outputs",
            "--grid_preset",
            "delta_residual_small",
        ]
    )
    assert "src.training.train was removed as a public entrypoint." in msg
    assert "python -m src.protocol.run" in msg
    assert "--train_once_eval_all" in msg


def test_migration_message_does_not_duplicate_protocol_flag() -> None:
    msg = _migration_message(
        [
            "python",
            "--dataset_root",
            "data/dataset_fullsquare_organized",
            "--train_once_eval_all",
        ]
    )
    assert msg.count("--train_once_eval_all") == 1


def test_main_exits_with_migration_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["python", "--dataset_root", "data/dataset_fullsquare_organized"],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert "src.training.train was removed as a public entrypoint." in str(exc.value)
