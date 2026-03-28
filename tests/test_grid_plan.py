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


def test_select_grid_delta_residual_smoke_returns_single_delta_variant():
    grid = select_grid({"grid_preset": "delta_residual_smoke"})

    assert len(grid) == 1
    assert grid[0]["cfg"]["arch_variant"] == "delta_residual"
    assert grid[0]["cfg"]["free_bits"] == 0.10
    assert grid[0]["tag"] == "D0delta_lat4_b0p001_fb0p10_lr0p0003_L128-256-512"


def test_select_grid_delta_residual_small_returns_expected_candidates():
    grid = select_grid({"grid_preset": "delta_residual_small"})

    assert len(grid) == 4
    assert all(item["cfg"]["arch_variant"] == "delta_residual" for item in grid)
    assert {item["cfg"]["beta"] for item in grid} == {0.001, 0.002, 0.003}
    assert {item["cfg"]["free_bits"] for item in grid} == {0.0, 0.10}
    assert all(item["cfg"]["layer_sizes"] == [128, 256, 512] for item in grid)


def test_select_grid_delta_residual_refine_focuses_around_current_winner():
    grid = select_grid({"grid_preset": "delta_residual_refine"})

    assert len(grid) == 6
    assert all(item["cfg"]["arch_variant"] == "delta_residual" for item in grid)
    assert {item["cfg"]["latent_dim"] for item in grid} == {4, 6, 8}
    assert {item["cfg"]["beta"] for item in grid} == {0.0005, 0.001}
    assert all(item["cfg"]["free_bits"] == 0.0 for item in grid)
    assert all(item["cfg"]["batch_size"] == 16384 for item in grid)
    assert all(item["cfg"]["layer_sizes"] == [128, 256, 512] for item in grid)


def test_select_grid_delta_residual_local_varies_only_latent_and_batch():
    grid = select_grid({"grid_preset": "delta_residual_local"})

    assert len(grid) == 8
    assert all(item["cfg"]["arch_variant"] == "delta_residual" for item in grid)
    assert {item["cfg"]["latent_dim"] for item in grid} == {3, 4, 5, 6}
    assert {item["cfg"]["batch_size"] for item in grid} == {8192, 16384}
    assert all(item["cfg"]["beta"] == 0.001 for item in grid)
    assert all(item["cfg"]["free_bits"] == 0.0 for item in grid)
    assert all(item["cfg"]["layer_sizes"] == [128, 256, 512] for item in grid)


def test_select_grid_delta_residual_frontier_opens_new_non_repeated_band():
    grid = select_grid({"grid_preset": "delta_residual_frontier"})

    assert len(grid) == 27
    assert all(item["cfg"]["arch_variant"] == "delta_residual" for item in grid)
    assert {item["cfg"]["latent_dim"] for item in grid} == {4, 5, 6}
    assert {item["cfg"]["beta"] for item in grid} == {0.0007, 0.00085, 0.00115}
    assert {item["cfg"]["free_bits"] for item in grid} == {0.0, 0.02, 0.05}
    assert all(item["cfg"]["batch_size"] == 16384 for item in grid)
    assert all(item["cfg"]["lr"] == 3e-4 for item in grid)
    assert all(item["cfg"]["kl_anneal_epochs"] == 80 for item in grid)
    assert all(item["cfg"]["layer_sizes"] == [128, 256, 512] for item in grid)
    # The frontier preset intentionally avoids the already repeated center:
    assert all(item["cfg"]["beta"] != 0.001 for item in grid)


def test_select_grid_best_compare_large_mixes_delta_and_seq_anchors():
    grid = select_grid({"grid_preset": "best_compare_large"})

    assert len(grid) == 12
    assert {item["group"] for item in grid} == {"C5_best_compare"}

    delta = [item for item in grid if item["cfg"]["arch_variant"] == "delta_residual"]
    seq = [item for item in grid if item["cfg"]["arch_variant"] == "seq_bigru_residual"]

    assert len(delta) == 4
    assert len(seq) == 8

    assert {item["cfg"]["free_bits"] for item in delta} == {0.0}
    assert {item["cfg"]["batch_size"] for item in delta} == {16384}
    assert {item["cfg"]["batch_size"] for item in seq} == {8192}
    assert {item["cfg"]["seq_hidden_size"] for item in seq} == {64}
    assert {item["cfg"]["window_size"] for item in seq} == {7}
    assert {item["cfg"]["lambda_mmd"] for item in seq} == {0.0, 0.1, 0.5, 1.0}

    tags = {item["tag"] for item in grid}
    assert "D3delta_lat5_b0p001_fb0p0_lr0p0003_bs16384_anneal80_L128-256-512" in tags
    assert "COPT_lat6_b0p001_fb0p0_lr0p0001_bs16384_anneal120_L64-128-256" in tags
    assert "S2seq_W7_h64_lat4_b0p001_lmmd1p0_fb0p10_lr0p0003_L128-256-512" in tags


def test_select_grid_seq_residual_nightly_builds_large_seq_only_sweep():
    grid = select_grid({"grid_preset": "seq_residual_nightly"})

    assert len(grid) == 24
    assert {item["group"] for item in grid} == {"S3_seq_nightly"}
    assert all(item["cfg"]["arch_variant"] == "seq_bigru_residual" for item in grid)
    assert {item["cfg"]["seq_hidden_size"] for item in grid} == {64, 96, 128}
    assert {item["cfg"]["beta"] for item in grid} == {0.001, 0.003}
    assert {item["cfg"]["lambda_mmd"] for item in grid} == {0.25, 0.5, 0.75, 1.0}
    assert {item["cfg"]["latent_dim"] for item in grid} == {4}
    assert {item["cfg"]["free_bits"] for item in grid} == {0.10}
    assert {item["cfg"]["batch_size"] for item in grid} == {8192}
    assert {item["cfg"]["window_size"] for item in grid} == {7}


def test_select_grid_seq_investigation_large_focuses_on_context_and_mmd():
    grid = select_grid({"grid_preset": "seq_investigation_large"})

    assert len(grid) == 17
    assert {item["group"] for item in grid} == {"S4_seq_investigation"}

    seq = [item for item in grid if item["cfg"]["arch_variant"] == "seq_bigru_residual"]
    delta = [item for item in grid if item["cfg"]["arch_variant"] == "delta_residual"]

    assert len(seq) == 16
    assert len(delta) == 1

    assert {item["cfg"]["window_size"] for item in seq} == {7, 9, 11}
    assert {item["cfg"]["seq_hidden_size"] for item in seq} == {64, 96}
    assert {item["cfg"]["lambda_mmd"] for item in seq} == {1.0, 1.25}
    assert {item["cfg"]["beta"] for item in seq} == {0.002, 0.003, 0.004}
    assert {item["cfg"]["latent_dim"] for item in seq} == {4}
    assert {item["cfg"]["free_bits"] for item in seq} == {0.10}
    assert {item["cfg"]["batch_size"] for item in seq} == {8192}

    anchor = delta[0]
    assert anchor["tag"] == "COPT_lat6_b0p001_fb0p0_lr0p0001_bs16384_anneal120_L64-128-256"
    assert anchor["cfg"]["layer_sizes"] == [64, 128, 256]
    assert anchor["cfg"]["latent_dim"] == 6
    assert anchor["cfg"]["beta"] == 0.001
    assert anchor["cfg"]["free_bits"] == 0.0


def test_select_grid_seq_sampled_mmd_compare_builds_causal_three_run_set():
    grid = select_grid({"grid_preset": "seq_sampled_mmd_compare"})

    assert len(grid) == 3
    assert {item["group"] for item in grid} == {"S9_seq_sampled_mmd"}
    assert all(item["cfg"]["arch_variant"] == "seq_bigru_residual" for item in grid)
    assert {item["cfg"]["window_size"] for item in grid} == {7}
    assert {item["cfg"]["seq_hidden_size"] for item in grid} == {64}
    assert {item["cfg"]["lambda_mmd"] for item in grid} == {1.75}
    assert {item["cfg"]["mmd_mode"] for item in grid} == {"mean_residual", "sampled_residual"}
    assert {item["cfg"]["lr"] for item in grid} == {3e-4, 2e-4}


def test_select_grid_seq_hybrid_loss_smoke_disables_shuffle_for_psd_term():
    grid = select_grid({"grid_preset": "seq_hybrid_loss_smoke"})

    assert len(grid) == 1
    cfg = grid[0]["cfg"]
    assert cfg["arch_variant"] == "seq_bigru_residual"
    assert cfg["lambda_axis"] == 0.10
    assert cfg["lambda_psd"] == 0.02
    assert cfg["decoder_distribution"] == "gaussian"
    assert cfg["shuffle_train_batches"] is False


def test_select_grid_seq_mdn_smoke_builds_single_mdn_candidate():
    grid = select_grid({"grid_preset": "seq_mdn_smoke"})

    assert len(grid) == 1
    cfg = grid[0]["cfg"]
    assert cfg["arch_variant"] == "seq_bigru_residual"
    assert cfg["decoder_distribution"] == "mdn"
    assert cfg["mdn_components"] == 3
    assert cfg["lambda_axis"] == 0.10
    assert cfg["lambda_psd"] == 0.02
    assert cfg["shuffle_train_batches"] is False


def test_select_grid_seq_mdn_proof_builds_two_component_sweep():
    grid = select_grid({"grid_preset": "seq_mdn_proof"})

    assert len(grid) == 2
    assert {item["cfg"]["mdn_components"] for item in grid} == {3, 5}
    assert all(item["cfg"]["decoder_distribution"] == "mdn" for item in grid)
    assert all(item["cfg"]["lambda_axis"] == 0.10 for item in grid)
    assert all(item["cfg"]["lambda_psd"] == 0.02 for item in grid)
    assert all(item["cfg"]["shuffle_train_batches"] is False for item in grid)


def test_select_grid_seq_mdn_regime_weight_quick_builds_three_run_compare():
    grid = select_grid({"grid_preset": "seq_mdn_regime_weight_quick"})

    assert len(grid) == 3
    assert {item["group"] for item in grid} == {"S17_seq_mdn_regime_weight"}
    assert all(item["cfg"]["arch_variant"] == "seq_bigru_residual" for item in grid)
    assert all(item["cfg"]["decoder_distribution"] == "mdn" for item in grid)
    assert all(item["cfg"]["mdn_components"] == 3 for item in grid)
    assert all(item["cfg"]["beta"] == 0.002 for item in grid)
    assert all(item["cfg"]["lambda_mmd"] == 0.25 for item in grid)
    assert all(item["cfg"]["lambda_axis"] == 0.01 for item in grid)

    control = grid[0]["cfg"]
    weighted = [item["cfg"] for item in grid[1:]]
    assert "train_regime_resample_weights" not in control
    assert all("train_regime_resample_weights" in cfg for cfg in weighted)
    assert all(
        set(cfg["train_regime_resample_weights"]) == {
            "dist_0p8m__curr_100ma",
            "dist_0p8m__curr_300ma",
            "dist_0p8m__curr_500ma",
        }
        for cfg in weighted
    )


def test_select_grid_seq_mdn_v2_quick_enables_mini_protocol_ranking_and_coverage():
    grid = select_grid({"grid_preset": "seq_mdn_v2_quick"})

    assert len(grid) == 4
    assert {item["group"] for item in grid} == {"S20_seq_mdn_v2"}
    assert all(item["cfg"]["arch_variant"] == "seq_bigru_residual" for item in grid)
    assert all(item["cfg"]["decoder_distribution"] == "mdn" for item in grid)
    assert all(item["cfg"]["mdn_components"] == 3 for item in grid)
    assert all(item["cfg"]["beta"] == 0.002 for item in grid)
    assert all(item["cfg"]["lambda_mmd"] == 0.25 for item in grid)
    assert all(item["cfg"]["lambda_axis"] == 0.01 for item in grid)
    assert {item["cfg"]["lambda_coverage"] for item in grid} == {0.0, 0.02, 0.05}

    analysis_overrides = [item["analysis_quick_overrides"] for item in grid]
    assert all(ov == analysis_overrides[0] for ov in analysis_overrides)
    assert analysis_overrides[0]["mini_reanalysis_enabled"] is True
    assert analysis_overrides[0]["mini_reanalysis_scope"] == "all12"
    assert analysis_overrides[0]["grid_ranking_mode"] == "mini_protocol_v1"


def test_select_grid_seq_imdd_graybox_smoke_builds_single_gaussian_candidate():
    grid = select_grid({"grid_preset": "seq_imdd_graybox_smoke"})

    assert len(grid) == 1
    cfg = grid[0]["cfg"]
    assert cfg["arch_variant"] == "seq_imdd_graybox"
    assert cfg["decoder_distribution"] == "gaussian"
    assert cfg["window_size"] == 7
    assert cfg["seq_hidden_size"] == 16
    assert cfg["batch_size"] == 8192
    assert cfg["imdd_poly_orders"] == [1, 3, 5]
    assert grid[0]["analysis_quick_overrides"]["batch_infer"] == 16384


def test_select_grid_seq_imdd_graybox_mdn_smoke_builds_single_mdn_candidate():
    grid = select_grid({"grid_preset": "seq_imdd_graybox_mdn_smoke"})

    assert len(grid) == 1
    cfg = grid[0]["cfg"]
    assert cfg["arch_variant"] == "seq_imdd_graybox"
    assert cfg["decoder_distribution"] == "mdn"
    assert cfg["mdn_components"] == 3
    assert cfg["window_size"] == 7
    assert cfg["seq_hidden_size"] == 32
    assert cfg["latent_dim"] == 6
    assert cfg["batch_size"] == 6144
    assert cfg["lambda_mmd"] == 0.25
    assert cfg["lambda_axis"] == 0.01
    assert cfg["lambda_coverage"] == 0.04
    assert grid[0]["analysis_quick_overrides"]["batch_infer"] == 16384


def test_select_grid_seq_imdd_graybox_capacity_quick_builds_local_capacity_sweep():
    grid = select_grid({"grid_preset": "seq_imdd_graybox_capacity_quick"})

    assert len(grid) == 4
    assert {item["group"] for item in grid} == {"SGB1_seq_imdd_graybox_capacity"}
    assert all(item["cfg"]["arch_variant"] == "seq_imdd_graybox" for item in grid)
    assert all(item["cfg"]["decoder_distribution"] == "gaussian" for item in grid)
    assert all(item["cfg"]["window_size"] == 7 for item in grid)
    assert all(item["cfg"]["batch_size"] == 8192 for item in grid)
    assert [item["cfg"]["seq_hidden_size"] for item in grid] == [16, 32, 32, 32]
    assert [item["cfg"]["latent_dim"] for item in grid] == [4, 4, 6, 6]
    assert [item["cfg"]["layer_sizes"] for item in grid] == [[64, 128], [64, 128], [64, 128], [128, 256]]
    assert all(item["analysis_quick_overrides"]["batch_infer"] == 32768 for item in grid)


def test_select_grid_seq_imdd_graybox_mdn_guided_quick_builds_local_mdn_sweep():
    grid = select_grid({"grid_preset": "seq_imdd_graybox_mdn_guided_quick"})

    assert len(grid) == 4
    assert {item["group"] for item in grid} == {"SGBM1_seq_imdd_graybox_mdn_guided"}
    assert all(item["cfg"]["arch_variant"] == "seq_imdd_graybox" for item in grid)
    assert all(item["cfg"]["decoder_distribution"] == "mdn" for item in grid)
    assert all(item["cfg"]["mdn_components"] == 3 for item in grid)
    assert all(item["cfg"]["seq_hidden_size"] == 32 for item in grid)
    assert all(item["cfg"]["latent_dim"] == 6 for item in grid)
    assert {item["cfg"]["window_size"] for item in grid} == {7, 11}
    assert {item["cfg"]["lr"] for item in grid} == {2e-4, 3e-4}
    assert {item["cfg"]["lambda_coverage"] for item in grid} == {0.04, 0.06}

    analysis_overrides = [item["analysis_quick_overrides"] for item in grid]
    assert all(ov == analysis_overrides[0] for ov in analysis_overrides)
    assert analysis_overrides[0]["mini_reanalysis_enabled"] is True
    assert analysis_overrides[0]["mini_reanalysis_scope"] == "all12"
    assert analysis_overrides[0]["grid_ranking_mode"] == "mini_protocol_v1"
    assert analysis_overrides[0]["batch_infer"] == 16384


def test_select_grid_seq_mdn_v2_perf_compare_quick_builds_control_and_fast_variants():
    grid = select_grid({"grid_preset": "seq_mdn_v2_perf_compare_quick"})

    assert len(grid) == 3
    assert {item["group"] for item in grid} == {"S21_seq_mdn_v2_perf"}
    assert all(item["cfg"]["arch_variant"] == "seq_bigru_residual" for item in grid)
    assert all(item["cfg"]["decoder_distribution"] == "mdn" for item in grid)
    assert all(item["cfg"]["mdn_components"] == 3 for item in grid)
    assert all(item["cfg"]["lambda_coverage"] == 0.0 for item in grid)
    assert [item["cfg"]["batch_size"] for item in grid] == [4096, 8192, 8192]
    assert [item["cfg"]["seq_gru_unroll"] for item in grid] == [True, True, False]

    analysis_overrides = [item["analysis_quick_overrides"] for item in grid]
    assert analysis_overrides[0]["batch_infer"] == 8192
    assert analysis_overrides[1]["batch_infer"] == 16384
    assert analysis_overrides[2]["batch_infer"] == 16384
    assert all(ov["mini_reanalysis_enabled"] is True for ov in analysis_overrides)
    assert all(ov["grid_ranking_mode"] == "mini_protocol_v1" for ov in analysis_overrides)


def test_select_grid_seq_mdn_v2_fastbase_quick_uses_fast_operational_baseline():
    grid = select_grid({"grid_preset": "seq_mdn_v2_fastbase_quick"})

    assert len(grid) == 4
    assert {item["group"] for item in grid} == {"S22_seq_mdn_v2_fast"}
    assert all(item["cfg"]["arch_variant"] == "seq_bigru_residual" for item in grid)
    assert all(item["cfg"]["decoder_distribution"] == "mdn" for item in grid)
    assert all(item["cfg"]["mdn_components"] == 3 for item in grid)
    assert all(item["cfg"]["batch_size"] == 8192 for item in grid)
    assert all(item["cfg"]["seq_gru_unroll"] is False for item in grid)
    assert {item["cfg"]["lambda_coverage"] for item in grid} == {0.0, 0.02, 0.05}
    assert {item["cfg"]["coverage_temperature"] for item in grid} == {0.05, 0.03}

    analysis_overrides = [item["analysis_quick_overrides"] for item in grid]
    assert all(ov == analysis_overrides[0] for ov in analysis_overrides)
    assert analysis_overrides[0]["batch_infer"] == 16384
    assert analysis_overrides[0]["mini_reanalysis_enabled"] is True
    assert analysis_overrides[0]["grid_ranking_mode"] == "mini_protocol_v1"


def test_select_grid_seq_mdn_v2_g5_followup_quick_stays_local_to_fastbase_champion():
    grid = select_grid({"grid_preset": "seq_mdn_v2_g5_followup_quick"})

    assert len(grid) == 5
    assert {item["group"] for item in grid} == {"S23_seq_mdn_v2_g5"}
    assert all(item["cfg"]["arch_variant"] == "seq_bigru_residual" for item in grid)
    assert all(item["cfg"]["decoder_distribution"] == "mdn" for item in grid)
    assert all(item["cfg"]["mdn_components"] == 3 for item in grid)
    assert all(item["cfg"]["batch_size"] == 8192 for item in grid)
    assert all(item["cfg"]["seq_gru_unroll"] is False for item in grid)
    assert {item["cfg"]["coverage_temperature"] for item in grid} == {0.03, 0.04}
    assert {item["cfg"]["lambda_coverage"] for item in grid} == {0.04, 0.05, 0.06}
    assert {item["cfg"]["lambda_axis"] for item in grid} == {0.01, 0.0125}

    analysis_overrides = [item["analysis_quick_overrides"] for item in grid]
    assert all(ov == analysis_overrides[0] for ov in analysis_overrides)
    assert analysis_overrides[0]["batch_infer"] == 16384
    assert analysis_overrides[0]["mini_reanalysis_enabled"] is True
    assert analysis_overrides[0]["grid_ranking_mode"] == "mini_protocol_v1"


def test_select_grid_seq_mdn_v2_overnight_decision_quick_combines_local_and_exploratory_candidates():
    grid = select_grid({"grid_preset": "seq_mdn_v2_overnight_decision_quick"})

    assert len(grid) == 9
    assert {item["group"] for item in grid} == {"S24_seq_mdn_v2_overnight"}
    assert all(item["cfg"]["arch_variant"] == "seq_bigru_residual" for item in grid)
    assert all(item["cfg"]["decoder_distribution"] == "mdn" for item in grid)
    assert all(item["cfg"]["mdn_components"] == 3 for item in grid)
    assert all(item["cfg"]["seq_gru_unroll"] is False for item in grid)
    assert {item["cfg"]["lambda_coverage"] for item in grid} == {0.06, 0.07}
    assert {item["cfg"]["coverage_temperature"] for item in grid} == {0.025, 0.03, 0.035}
    assert {item["cfg"]["latent_dim"] for item in grid} == {4, 6}
    assert {item["cfg"]["seq_hidden_size"] for item in grid} == {64, 96}
    assert {item["cfg"]["window_size"] for item in grid} == {7, 11}
    assert {item["cfg"]["lambda_mmd"] for item in grid} == {0.25, 0.30}
    assert {item["cfg"]["batch_size"] for item in grid} == {6144, 8192}

    analysis_overrides = [item["analysis_quick_overrides"] for item in grid]
    assert all(ov == analysis_overrides[0] for ov in analysis_overrides)
    assert analysis_overrides[0]["batch_infer"] == 16384
    assert analysis_overrides[0]["mini_reanalysis_enabled"] is True
    assert analysis_overrides[0]["grid_ranking_mode"] == "mini_protocol_v1"


def test_select_grid_seq_mdn_v2_overnight_5090safe_quick_keeps_fast_path_local_only():
    grid = select_grid({"grid_preset": "seq_mdn_v2_overnight_5090safe_quick"})

    assert len(grid) == 9
    assert {item["group"] for item in grid} == {"S25_seq_mdn_v2_overnight_5090safe"}
    assert all(item["cfg"]["arch_variant"] == "seq_bigru_residual" for item in grid)
    assert all(item["cfg"]["decoder_distribution"] == "mdn" for item in grid)
    assert all(item["cfg"]["mdn_components"] == 3 for item in grid)
    assert {item["cfg"]["lambda_coverage"] for item in grid} == {0.06, 0.07}
    assert {item["cfg"]["coverage_temperature"] for item in grid} == {0.025, 0.03, 0.035}
    assert {item["cfg"]["latent_dim"] for item in grid} == {4, 6}
    assert {item["cfg"]["seq_hidden_size"] for item in grid} == {64, 96}
    assert {item["cfg"]["window_size"] for item in grid} == {7, 11}
    assert {item["cfg"]["lambda_mmd"] for item in grid} == {0.25, 0.30}
    assert {item["cfg"]["batch_size"] for item in grid} == {6144, 8192}

    local_fast = [
        item for item in grid
        if item["cfg"]["window_size"] == 7
        and item["cfg"]["seq_hidden_size"] == 64
        and item["cfg"]["latent_dim"] == 4
    ]
    structural = [item for item in grid if item not in local_fast]

    assert len(local_fast) == 5
    assert len(structural) == 4
    assert all(item["cfg"]["seq_gru_unroll"] is False for item in local_fast)
    assert all(item["cfg"]["seq_gru_unroll"] is True for item in structural)

    analysis_overrides = [item["analysis_quick_overrides"] for item in grid]
    assert all(ov == analysis_overrides[0] for ov in analysis_overrides)
    assert analysis_overrides[0]["batch_infer"] == 16384
    assert analysis_overrides[0]["mini_reanalysis_enabled"] is True
    assert analysis_overrides[0]["grid_ranking_mode"] == "mini_protocol_v1"


def test_select_grid_seq_mdn_v2_a600_tail_explore_quick_opens_tail_specific_branch():
    grid = select_grid({"grid_preset": "seq_mdn_v2_a600_tail_explore_quick"})

    assert len(grid) == 7
    assert {item["group"] for item in grid} == {"S26_seq_mdn_v2_a600_tail"}
    assert all(item["cfg"]["arch_variant"] == "seq_bigru_residual" for item in grid)
    assert all(item["cfg"]["decoder_distribution"] == "mdn" for item in grid)
    assert all(item["cfg"]["mdn_components"] == 3 for item in grid)
    assert all(item["cfg"]["seq_gru_unroll"] is False for item in grid)
    assert {tuple(item["cfg"]["tail_levels"]) for item in grid} == {
        (0.05, 0.95),
        (0.02, 0.98),
        (0.01, 0.99),
    }
    assert {item["cfg"]["lambda_coverage"] for item in grid} == {0.06, 0.07}
    assert {item["cfg"]["lambda_axis"] for item in grid} == {0.01, 0.0125}
    assert {item["cfg"]["latent_dim"] for item in grid} == {4, 6}
    assert {item["cfg"]["seq_hidden_size"] for item in grid} == {64, 96}
    assert {item["cfg"]["batch_size"] for item in grid} == {6144, 8192}

    analysis_overrides = [item["analysis_quick_overrides"] for item in grid]
    assert all(ov == analysis_overrides[0] for ov in analysis_overrides)
    assert analysis_overrides[0]["batch_infer"] == 16384
    assert analysis_overrides[0]["mini_reanalysis_enabled"] is True
    assert analysis_overrides[0]["grid_ranking_mode"] == "mini_protocol_v1"


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
