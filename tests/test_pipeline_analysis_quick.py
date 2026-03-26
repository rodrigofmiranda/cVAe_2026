from src.training.pipeline import (
    _resolve_best_candidate_analysis_quick,
    _resolve_grid_analysis_quick,
)


def test_resolve_grid_analysis_quick_merges_identical_overrides():
    base = {"batch_infer": 1024, "mini_reanalysis_enabled": False}
    grid = [
        {"tag": "A", "analysis_quick_overrides": {"batch_infer": 2048}},
        {"tag": "B", "analysis_quick_overrides": {"batch_infer": 2048}},
    ]

    resolved = _resolve_grid_analysis_quick(base, grid)

    assert resolved == {"batch_infer": 2048, "mini_reanalysis_enabled": False}


def test_resolve_grid_analysis_quick_keeps_base_when_overrides_differ():
    base = {"batch_infer": 1024, "mini_reanalysis_enabled": False}
    grid = [
        {"tag": "A", "analysis_quick_overrides": {"batch_infer": 2048}},
        {"tag": "B", "analysis_quick_overrides": {"batch_infer": 4096}},
    ]

    resolved = _resolve_grid_analysis_quick(base, grid)

    assert resolved == base


def test_resolve_best_candidate_analysis_quick_uses_winner_overrides():
    base = {"batch_infer": 1024, "mini_reanalysis_enabled": False}
    grid = [
        {"tag": "A", "analysis_quick_overrides": {"batch_infer": 2048}},
        {
            "tag": "B",
            "analysis_quick_overrides": {
                "batch_infer": 4096,
                "mini_reanalysis_enabled": True,
            },
        },
    ]

    resolved = _resolve_best_candidate_analysis_quick(base, grid, "B")

    assert resolved == {"batch_infer": 4096, "mini_reanalysis_enabled": True}
