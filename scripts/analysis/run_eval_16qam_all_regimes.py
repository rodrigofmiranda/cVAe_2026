#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path

from src.data.loading import discover_experiments, parse_dist_curr_from_path, read_metadata
from src.evaluation.engine import clear_evaluation_model_cache, evaluate_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run per-regime 16QAM evaluation for a selected trained model."
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path("/workspace/2026/feat_seq_bigru_residual_cvae"),
        help="Repository root inside the container.",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=None,
        help="Dataset root to evaluate (default: <repo_root>/data/16qam).",
    )
    parser.add_argument(
        "--model_run_dir",
        type=Path,
        required=True,
        help="Train run directory that contains state_run.json and models/.",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=None,
        help="Output root for per-regime evaluation artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else (repo / 'data' / '16qam').resolve()
    )
    model_run_dir = args.model_run_dir.resolve()
    out_root = (
        args.out_root.resolve()
        if args.out_root is not None
        else (model_run_dir.parent / 'eval_16qam_all_regimes').resolve()
    )
    out_root.mkdir(parents=True, exist_ok=True)

    print(f'[batch] dataset_root={dataset_root}')
    print(f'[batch] model_run_dir={model_run_dir}')
    print(f'[batch] out_root={out_root}')

    exp_dirs = discover_experiments(dataset_root, verbose=False)
    regime_map: dict[tuple[float, int], list[str]] = {}

    for exp in exp_dirs:
        dist, curr = parse_dist_curr_from_path(exp)
        if dist is None or curr is None:
            meta = read_metadata(exp)
            if dist is None:
                for key in ('distance_m', 'distance', 'dist_m', 'dist'):
                    if key in meta:
                        try:
                            dist = float(meta[key])
                            break
                        except Exception:
                            pass
            if curr is None:
                for key in ('current_mA', 'current', 'curr_mA', 'curr'):
                    if key in meta:
                        try:
                            curr = int(float(meta[key]))
                            break
                        except Exception:
                            pass

        if dist is None or curr is None:
            print(f'[batch][warn] skipping experiment without regime parse: {exp}')
            continue

        key = (float(dist), int(curr))
        regime_map.setdefault(key, []).append(str(exp))

    regimes = sorted(regime_map.keys(), key=lambda t: (t[0], t[1]))
    print(f'[batch] regimes_found={len(regimes)}')
    for d, c in regimes:
        print(f'  - dist={d:.1f}m curr={c}mA exps={len(regime_map[(d, c)])}')

    rows: list[dict] = []

    try:
        for idx, (dist, curr) in enumerate(regimes, start=1):
            rid = f"dist_{str(dist).replace('.', 'p')}m__curr_{curr}mA"
            out_dir = out_root / rid
            print(f"\n[batch] ({idx}/{len(regimes)}) {rid}")

            overrides = {
                '_selected_experiments': sorted(regime_map[(dist, curr)]),
                'dist_tol_m': 0.01,
                'curr_tol_mA': 1.0,
            }

            try:
                summary = evaluate_run(
                    run_dir=model_run_dir,
                    dataset_root=dataset_root,
                    overrides=overrides,
                    output_run_dir=out_dir,
                )
                rows.append(
                    {
                        'regime_id': rid,
                        'distance_m': dist,
                        'current_mA': curr,
                        'status': summary.get('status', 'unknown'),
                        'run_dir': summary.get('run_dir', str(out_dir)),
                        'panel6_path': summary.get('panel6_path', ''),
                        'overlay_path': summary.get('overlay_path', ''),
                        'fingerprint_path': summary.get('fingerprint_path', ''),
                        'dashboard_path': summary.get('dashboard_path', ''),
                    }
                )
                print(f"[batch][ok] {rid} -> {summary.get('run_dir', str(out_dir))}")
            except Exception as exc:
                rows.append(
                    {
                        'regime_id': rid,
                        'distance_m': dist,
                        'current_mA': curr,
                        'status': f'error: {exc}',
                        'run_dir': str(out_dir),
                        'panel6_path': '',
                        'overlay_path': '',
                        'fingerprint_path': '',
                        'dashboard_path': '',
                    }
                )
                print(f"[batch][error] {rid}: {exc}")
                traceback.print_exc()
    finally:
        clear_evaluation_model_cache()

    manifest_json = out_root / 'manifest_all_regimes_eval.json'
    manifest_csv = out_root / 'manifest_all_regimes_eval.csv'

    payload = {
        'dataset_root': str(dataset_root),
        'model_run_dir': str(model_run_dir),
        'n_regimes': len(regimes),
        'results': rows,
    }
    manifest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    with manifest_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'regime_id',
                'distance_m',
                'current_mA',
                'status',
                'run_dir',
                'panel6_path',
                'overlay_path',
                'fingerprint_path',
                'dashboard_path',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum(1 for r in rows if str(r.get('status', '')).lower() == 'completed')
    print(f"\n[batch] done: ok={n_ok}/{len(rows)}")
    print(f"[batch] manifest_json={manifest_json}")
    print(f"[batch] manifest_csv={manifest_csv}")


if __name__ == '__main__':
    main()
