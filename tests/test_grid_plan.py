from src.training.grid_plan import select_grid


def test_select_grid_exploratory_small_returns_compact_subset():
    grid = select_grid({"grid_preset": "exploratory_small"})

    assert len(grid) == 8
    assert grid[0]["tag"] == "G0_lat4_b0p003_fb0p10_lr0p0003_L128-256-512"
    assert grid[-1]["group"] == "G3_opt"


def test_select_grid_exploratory_small_still_obeys_max_grids():
    grid = select_grid({"grid_preset": "exploratory_small", "max_grids": 3})

    assert len(grid) == 3

