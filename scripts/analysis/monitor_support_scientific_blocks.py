#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    print(f"[{utc_now()}] {msg}", flush=True)


def log_has_completion(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return False
    return "Protocol complete" in text


def latest_experiment(base_dir: Path) -> Optional[Path]:
    if not base_dir.exists():
        return None
    exps = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("exp_")])
    return exps[-1] if exps else None


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def first_row(path: Path) -> Optional[Dict[str, str]]:
    rows = read_csv_rows(path)
    return rows[0] if rows else None


def find_row(rows: List[Dict[str, str]], tag: str) -> Optional[Dict[str, str]]:
    for row in rows:
        if row.get("candidate_id") == tag or row.get("best_grid_tag") == tag or row.get("tag") == tag:
            return row
    return None


def as_float(row: Optional[Dict[str, str]], key: str) -> Optional[float]:
    if not row:
        return None
    value = row.get(key, "")
    if value in {"", "nan", "NaN", "None", None}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass
class BlockConfig:
    name: str
    log_path: Path
    base_dir: Path
    control_tag: str
    family_tags: List[str]


def summarize_block(block: BlockConfig) -> Dict[str, object]:
    exp_dir = latest_experiment(block.base_dir)
    if exp_dir is None:
        raise FileNotFoundError(f"No exp_* found under {block.base_dir}")

    manifest_path = exp_dir / "manifest.json"
    leaderboard_path = exp_dir / "tables" / "protocol_leaderboard.csv"
    leaderboard = read_csv_rows(leaderboard_path)
    champion = first_row(leaderboard_path)
    control = find_row(leaderboard, block.control_tag)
    family_rows = [r for r in leaderboard if (r.get("candidate_id") or r.get("best_grid_tag") or r.get("tag")) in block.family_tags]

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    return {
        "exp_dir": str(exp_dir),
        "manifest": manifest,
        "leaderboard_path": str(leaderboard_path),
        "champion": champion,
        "control": control,
        "family_rows": family_rows,
    }


def render_block_md(name: str, summary: Dict[str, object]) -> str:
    champion = summary["champion"] or {}
    control = summary["control"] or {}
    family_rows = summary["family_rows"] or []

    lines = []
    lines.append(f"## Block {name}")
    lines.append("")
    lines.append(f"- Experiment: `{summary['exp_dir']}`")
    lines.append(f"- Leaderboard: `{summary['leaderboard_path']}`")
    lines.append(
        f"- Champion: `{champion.get('candidate_id', champion.get('best_grid_tag', 'unknown'))}`"
    )
    if champion:
        lines.append(
            f"- Champion result: `twin_pass={champion.get('n_pass', '?')}` "
            f"`full_pass={champion.get('n_full_pass', '?')}` "
            f"`g5={champion.get('gate_g5_pass', '?')}` "
            f"`g6={champion.get('gate_g6_pass', '?')}`"
        )
    if control:
        lines.append(
            f"- Control result: `twin_pass={control.get('n_pass', '?')}` "
            f"`full_pass={control.get('n_full_pass', '?')}` "
            f"`g5={control.get('gate_g5_pass', '?')}` "
            f"`g6={control.get('gate_g6_pass', '?')}`"
        )
    lines.append("")
    lines.append("| Candidate | twin_pass | full_pass | G5 | G6 | rel_evm | cov_rel_var | psd_l2 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in family_rows:
        tag = row.get("candidate_id", row.get("best_grid_tag", row.get("tag", "?")))
        lines.append(
            f"| `{tag}` | {row.get('n_pass', '?')} | {row.get('n_full_pass', '?')} | {row.get('gate_g5_pass', '?')} | "
            f"{row.get('gate_g6_pass', '?')} | {row.get('mean_cvae_rel_evm_error', '?')} | "
            f"{row.get('mean_cvae_cov_rel_var', '?')} | {row.get('mean_cvae_psd_l2', '?')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--poll-seconds", type=int, default=180)
    p.add_argument("--report-dir", type=Path, required=True)
    args = p.parse_args()

    control_tag = "S27cov_sciv1_ctrl_lc0p25_t0p03_a1p50_tau0p75_tc0p35_wmax3p0"
    block_a = BlockConfig(
        name="A",
        log_path=Path("/home/rodrigo/cVAe_2026_shape/outputs/_launch_logs/run_support_scientific_block_a_clean_20260409_010200.log"),
        base_dir=Path("/home/rodrigo/cVAe_2026_shape/outputs/support_ablation/final_grid/e2_scientific_screen_v1/block_a_clean"),
        control_tag=control_tag,
        family_tags=[
            control_tag,
            "S27cov_sciv1_edgebroad_a1p25_tau0p70_tc0p30_wmax2p8",
            "S27cov_sciv1_edgelocal_a1p50_tau0p80_tc0p40_wmax3p0",
            "S27cov_sciv1_cornerhard_a1p75_tau0p82_tc0p45_wmax3p2",
        ],
    )
    block_b = BlockConfig(
        name="B",
        log_path=Path("/home/rodrigo/cVAe_2026_shape/outputs/_launch_logs/run_support_scientific_block_b_clean_20260409_010200.log"),
        base_dir=Path("/home/rodrigo/cVAe_2026_shape/outputs/support_ablation/final_grid/e2_scientific_screen_v1/block_b_clean"),
        control_tag=control_tag,
        family_tags=[
            control_tag,
            "S27cov_sciv1_covsoft_lc0p20_t0p035",
            "S27cov_sciv1_covhard_lc0p30_t0p025",
            "S27cov_sciv1_tail98_lc0p25_t0p03",
            "S27cov_sciv1_covdense_cg50-75-90-95_lc0p25_t0p03",
        ],
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_md = args.report_dir / "support_scientific_blocks_ab_clean_report.md"
    report_json = args.report_dir / "support_scientific_blocks_ab_clean_report.json"

    log("monitor started")
    while True:
        done_a = log_has_completion(block_a.log_path)
        done_b = log_has_completion(block_b.log_path)
        log(f"status A={done_a} B={done_b}")
        if done_a and done_b:
            break
        time.sleep(args.poll_seconds)

    summary_a = summarize_block(block_a)
    summary_b = summarize_block(block_b)

    md = [
        "# Support Scientific Blocks A/B Clean Report",
        "",
        f"Generated: `{utc_now()}`",
        "",
        render_block_md("A", summary_a),
        render_block_md("B", summary_b),
    ]
    report_md.write_text("\n".join(md))
    report_json.write_text(
        json.dumps({"generated_at": utc_now(), "A": summary_a, "B": summary_b}, indent=2)
    )
    log(f"report written: {report_md}")


if __name__ == "__main__":
    main()
