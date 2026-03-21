from src.training.pipeline import _filter_selected_experiments


def test_filter_selected_experiments_accepts_portable_dataset_relative_suffixes():
    exps = [
        (None, None, None, None, "/mnt/clone_a/cVAe_2026/data/dataset_fullsquare_organized/dist_1.0m/curr_300mA/full_square_1m_300mA_001_20260210_154725"),
        (None, None, None, None, "/mnt/clone_a/cVAe_2026/data/dataset_fullsquare_organized/dist_1.5m/curr_500mA/full_square_1.5m_500mA_001_20260210_183335"),
    ]

    selected = [
        "/workspace/2026/feat_delta_residual_adv/data/dataset_fullsquare_organized/dist_1.0m/curr_300mA",
    ]

    filtered = _filter_selected_experiments(exps, selected)

    assert len(filtered) == 1
    assert str(filtered[0][4]).endswith(
        "dataset_fullsquare_organized/dist_1.0m/curr_300mA/full_square_1m_300mA_001_20260210_154725"
    )
