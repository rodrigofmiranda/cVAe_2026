from src.training.grid_plan import select_grid


def test_select_grid_exploratory_small_returns_compact_subset():
    grid = select_grid({"grid_preset": "exploratory_small"})

    assert len(grid) == 8
    assert grid[0]["tag"] == "G0_lat4_b0p003_fb0p10_lr0p0003_L128-256-512"
    assert grid[-1]["group"] == "G3_opt"


def test_select_grid_exploratory_small_still_obeys_max_grids():
    grid = select_grid({"grid_preset": "exploratory_small", "max_grids": 3})

    assert len(grid) == 3


def test_select_grid_residual_small_returns_residual_variants_only():
    grid = select_grid({"grid_preset": "residual_small"})

    assert len(grid) == 4
    assert all(item["cfg"]["arch_variant"] == "channel_residual" for item in grid)
    assert grid[0]["tag"] == "R0res_lat4_b0p003_fb0p10_lr0p0003_L128-256-512"


def test_select_grid_legacy2025_smoke_returns_single_legacy_variant():
    grid = select_grid({"grid_preset": "legacy2025_smoke"})

    assert len(grid) == 1
    assert grid[0]["cfg"]["arch_variant"] == "legacy_2025_zero_y"
    assert grid[0]["cfg"]["free_bits"] == 0.0
    assert grid[0]["tag"] == "L0legacy_lat4_b0p01_fb0p0_lr0p0003_bs1024_anneal3_L32-64"


def test_select_grid_legacy2025_ref_matches_expected_reference_cfg():
    grid = select_grid({"grid_preset": "legacy2025_ref"})

    assert len(grid) == 1
    cfg = grid[0]["cfg"]
    assert cfg["arch_variant"] == "legacy_2025_zero_y"
    assert cfg["layer_sizes"] == [32, 64, 128, 256]
    assert cfg["latent_dim"] == 16
    assert cfg["beta"] == 0.1
    assert cfg["lr"] == 1e-4
    assert cfg["batch_size"] == 4096
    assert cfg["kl_anneal_epochs"] == 50
    assert cfg["free_bits"] == 0.0


def test_select_grid_legacy2025_batch_sweep_varies_only_batch_size():
    grid = select_grid({"grid_preset": "legacy2025_batch_sweep"})

    assert [item["cfg"]["batch_size"] for item in grid] == [4096, 8192, 16384, 32768, 65536]
    assert all(item["cfg"]["arch_variant"] == "legacy_2025_zero_y" for item in grid)
    assert all(item["cfg"]["layer_sizes"] == [32, 64, 128, 256] for item in grid)
    assert all(item["cfg"]["latent_dim"] == 16 for item in grid)
    assert all(item["cfg"]["beta"] == 0.1 for item in grid)
    assert all(item["cfg"]["lr"] == 1e-4 for item in grid)
    assert all(item["cfg"]["kl_anneal_epochs"] == 50 for item in grid)
    assert all(item["cfg"]["free_bits"] == 0.0 for item in grid)


def test_select_grid_legacy2025_large_builds_reduced_data_large_search():
    grid = select_grid({"grid_preset": "legacy2025_large"})

    assert len(grid) == 12
    assert all(item["cfg"]["arch_variant"] == "legacy_2025_zero_y" for item in grid)
    assert {tuple(item["cfg"]["layer_sizes"]) for item in grid} == {
        (32, 64, 128, 256),
        (64, 128, 256, 512),
    }
    assert {item["cfg"]["latent_dim"] for item in grid} == {8, 16, 24}
    assert {item["cfg"]["beta"] for item in grid} == {0.03, 0.1}
    assert all(item["cfg"]["batch_size"] == 8192 for item in grid)
    assert all(item["cfg"]["lr"] == 1e-4 for item in grid)
    assert all(item["cfg"]["kl_anneal_epochs"] == 50 for item in grid)
