from src.protocol.run import (
    _effective_baseline_config,
    _effective_cvae_config,
    _effective_dist_metrics_config,
    _effective_stat_max_n,
    _limit_protocol_regimes,
    _protocol_execution_mode,
    _should_run_cvae,
)


def test_limit_protocol_regimes_preserves_order_and_studies():
    proto = {
        "regimes": [
            {"regime_id": "r0"},
            {"regime_id": "r1"},
            {"regime_id": "r2"},
        ],
        "_studies": [
            {"name": "s0", "regime_ids": ["r0", "r1"]},
            {"name": "s1", "regime_ids": ["r2"]},
        ],
    }

    out = _limit_protocol_regimes(proto, 2)

    assert [r["regime_id"] for r in out["regimes"]] == ["r0", "r1"]
    assert out["_studies"] == [{"name": "s0", "regime_ids": ["r0", "r1"]}]


def test_limit_protocol_regimes_rejects_nonpositive_values():
    proto = {"regimes": [{"regime_id": "r0"}], "_studies": []}

    try:
        _limit_protocol_regimes(proto, 0)
    except ValueError as exc:
        assert "max_regimes" in str(exc)
    else:
        raise AssertionError("Expected ValueError for max_regimes <= 0")


def test_should_run_cvae_false_for_baseline_only_modes():
    assert _should_run_cvae(no_cvae=True) is False
    assert _should_run_cvae(baseline_only=True) is False
    assert _should_run_cvae(no_cvae=False, baseline_only=False) is True


def test_effective_stat_max_n_depends_on_mode():
    assert _effective_stat_max_n("quick", None) == 5000
    assert _effective_stat_max_n("full", None) == 50000
    assert _effective_stat_max_n("quick", 1234) == 1234


def test_effective_baseline_config_uses_runtime_overrides():
    cfg = _effective_baseline_config(
        {"max_epochs": 7, "keras_verbose": 1},
        enabled=True,
        return_predictions=True,
    )

    assert cfg["epochs"] == 7
    assert cfg["verbose"] == 1
    assert cfg["enabled"] is True
    assert cfg["return_predictions"] is True
    assert cfg["loss"] == "mse"


def test_effective_cvae_config_only_exposes_relevant_override_keys():
    cfg = _effective_cvae_config(
        {"max_epochs": 5, "max_grids": 2, "seed": 123, "gauss_alpha": 0.1},
        enabled=True,
    )

    assert cfg == {
        "enabled": True,
        "execution_mode": "per_regime_retrain",
        "max_epochs": 5,
        "max_grids": 2,
        "seed": 123,
    }


def test_protocol_execution_mode_switches_with_flag():
    assert _protocol_execution_mode(train_once_eval_all=False) == "per_regime_retrain"
    assert _protocol_execution_mode(train_once_eval_all=True) == "train_once_eval_all"


def test_effective_dist_metrics_config_uses_overrides_and_defaults():
    cfg = _effective_dist_metrics_config(
        {"psd_nfft": 4096, "gauss_alpha": 0.05, "max_dist_samples": 1234},
        enabled=False,
    )

    assert cfg == {
        "enabled": False,
        "psd_nfft": 4096,
        "gauss_alpha": 0.05,
        "max_dist_samples": 1234,
    }
