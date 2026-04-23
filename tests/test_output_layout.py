from pathlib import Path

from src.evaluation.output_layout import (
    build_architecture_16qam_root,
    build_crossline_16qam_root,
    compact_16qam_run_tag,
    relocate_manifest_payload,
    relocate_manifest_rows,
)


def test_compact_16qam_run_tag_strips_legacy_prefixes():
    assert compact_16qam_run_tag("eval_16qam_crossline_20260420_clean") == "crossline_20260420_clean"
    assert compact_16qam_run_tag("eval_all_regimes") == "all_regimes"
    assert compact_16qam_run_tag(None) == "all_regimes"



def test_build_architecture_16qam_root_uses_compact_layout(tmp_path: Path):
    out = build_architecture_16qam_root(
        repo_root=tmp_path,
        architecture_family="clean_baseline",
        candidate_name="S27cov_fc_clean_lc0p25_t0p03_lat10",
        run_tag="eval_16qam_crossline_20260420_clean",
    )
    assert out == (
        tmp_path
        / "outputs"
        / "architectures"
        / "clean_baseline"
        / "S27cov_fc_clean_lc0p25_t0p03_lat10"
        / "16qam"
        / "crossline_20260420_clean"
    )



def test_build_crossline_16qam_root_uses_compact_layout(tmp_path: Path):
    out = build_crossline_16qam_root(tmp_path, "eval_16qam_crossline_20260422_plus_soft_radial")
    assert out == (
        tmp_path
        / "outputs"
        / "architectures"
        / "_crossline"
        / "16qam"
        / "crossline_20260422_plus_soft_radial"
    )



def test_relocate_manifest_payload_rewrites_regime_artifact_paths(tmp_path: Path):
    new_root = tmp_path / "outputs" / "architectures" / "clean_baseline" / "cand" / "16qam" / "crossline_20260420_clean"
    payload = {
        "results": [
            {
                "regime_id": "dist_1p0m__curr_100mA",
                "run_dir": "/workspace/legacy/eval/full_circle_clean_lat10/dist_1p0m__curr_100mA",
                "metrics_path": "/workspace/legacy/eval/full_circle_clean_lat10/dist_1p0m__curr_100mA/logs/metricas_globais_reanalysis.json",
                "dashboard_path": "/workspace/legacy/eval/full_circle_clean_lat10/dist_1p0m__curr_100mA/plots/champion/analysis_dashboard.png",
            }
        ]
    }

    relocated = relocate_manifest_payload(payload, new_root)
    row = relocated["results"][0]

    assert row["run_dir"] == str(new_root / "dist_1p0m__curr_100mA")
    assert row["metrics_path"] == str(new_root / "dist_1p0m__curr_100mA" / "logs" / "metricas_globais_reanalysis.json")
    assert row["dashboard_path"] == str(new_root / "dist_1p0m__curr_100mA" / "plots" / "champion" / "analysis_dashboard.png")



def test_relocate_manifest_rows_rewrites_csv_like_rows(tmp_path: Path):
    new_root = tmp_path / "outputs" / "architectures" / "disk_geom3" / "cand" / "16qam" / "crossline_20260420_clean"
    rows = [
        {
            "regime_id": "dist_0p8m__curr_300mA",
            "run_dir": "/workspace/legacy/dist_0p8m__curr_300mA",
            "panel6_path": "/workspace/legacy/dist_0p8m__curr_300mA/plots/champion/comparison_panel_6.png",
            "overlay_path": "",
            "fingerprint_path": "",
            "metrics_path": "",
            "dashboard_path": "",
        }
    ]

    relocated = relocate_manifest_rows(rows, new_root)
    row = relocated[0]

    assert row["run_dir"] == str(new_root / "dist_0p8m__curr_300mA")
    assert row["panel6_path"] == str(new_root / "dist_0p8m__curr_300mA" / "plots" / "champion" / "comparison_panel_6.png")
