from src.protocol.run import (
    _effective_stat_max_n,
    _limit_protocol_regimes,
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
