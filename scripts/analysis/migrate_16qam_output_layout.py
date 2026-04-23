#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.output_layout import (  # noqa: E402
    build_crossline_16qam_root,
    compact_16qam_run_tag,
    relocate_manifest_payload,
    relocate_manifest_rows,
)

_TEXT_SUFFIXES = {".md", ".json", ".csv", ".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate verbose 16QAM output trees to the compact canonical layout.",
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=ROOT,
        help="Repository root that owns outputs/architectures.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show the moves without changing files.",
    )
    return parser.parse_args()



def _cleanup_empty_parents(path: Path, stop_at: Path) -> None:
    current = path.resolve()
    sentinel = stop_at.resolve()
    while current != sentinel and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent



def _rewrite_manifest_files(eval_root: Path) -> None:
    manifest_json = eval_root / "manifest_all_regimes_eval.json"
    manifest_csv = eval_root / "manifest_all_regimes_eval.csv"

    if manifest_json.exists():
        payload = json.loads(manifest_json.read_text(encoding="utf-8"))
        payload = relocate_manifest_payload(payload, eval_root)
        manifest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if manifest_csv.exists():
        with manifest_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        rows = relocate_manifest_rows(rows, eval_root)
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with manifest_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)



def _rewrite_text_paths(tree_root: Path, replacements: dict[str, str]) -> None:
    if not replacements:
        return
    for path in tree_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except Exception:
            continue
        updated = original
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != original:
            path.write_text(updated, encoding="utf-8")



def _move_tree(src: Path, dst: Path, *, dry_run: bool) -> None:
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    print(f"[move] {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))



def migrate_architecture_runs(repo_root: Path, *, dry_run: bool) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for old_root in sorted((repo_root / "outputs" / "architectures").glob("**/benchmarks/16qam/*")):
        if not old_root.is_dir():
            continue
        run_parent = old_root.parents[2]
        new_root = run_parent / "16qam" / compact_16qam_run_tag(old_root.name)
        replacements[str(old_root.resolve())] = str(new_root.resolve())
        _move_tree(old_root, new_root, dry_run=dry_run)
        if not dry_run:
            _rewrite_manifest_files(new_root)
            _rewrite_text_paths(new_root, replacements)
            _cleanup_empty_parents(old_root.parent, run_parent)
    return replacements



def migrate_crossline_runs(repo_root: Path, *, dry_run: bool, replacements: dict[str, str]) -> int:
    moved = 0
    crossline_root = repo_root / "outputs" / "architectures" / "_crossline" / "16qam"
    if crossline_root.exists():
        for old_root in sorted(crossline_root.iterdir()):
            if not old_root.is_dir():
                continue
            new_root = build_crossline_16qam_root(repo_root, old_root.name)
            if compact_16qam_run_tag(old_root.name) == old_root.name:
                if not dry_run:
                    _rewrite_text_paths(old_root, replacements)
                continue
            replacements[str(old_root.resolve())] = str(new_root.resolve())
            _move_tree(old_root, new_root, dry_run=dry_run)
            if not dry_run:
                _rewrite_text_paths(new_root, replacements)
                moved += 1

    legacy_analysis_root = repo_root / "outputs" / "analysis"
    if legacy_analysis_root.exists():
        for old_root in sorted(legacy_analysis_root.glob("eval_16qam_crossline_*")):
            if not old_root.is_dir():
                continue
            new_root = build_crossline_16qam_root(repo_root, old_root.name)
            replacements[str(old_root.resolve())] = str(new_root.resolve())
            _move_tree(old_root, new_root, dry_run=dry_run)
            if not dry_run:
                _rewrite_text_paths(new_root, replacements)
                _cleanup_empty_parents(old_root.parent, repo_root / "outputs")
                moved += 1
    return moved



def refresh_compact_runs(repo_root: Path, replacements: dict[str, str]) -> None:
    for eval_root in sorted((repo_root / "outputs" / "architectures").glob("**/16qam/*")):
        if not eval_root.is_dir():
            continue
        _rewrite_manifest_files(eval_root)
        _rewrite_text_paths(eval_root, replacements)



def cleanup_legacy_dirs(repo_root: Path) -> None:
    for path in sorted((repo_root / "outputs" / "architectures").glob("**/benchmarks/16qam"), reverse=True):
        if path.is_dir():
            _cleanup_empty_parents(path, path.parents[2])
    analysis_root = repo_root / "outputs" / "analysis"
    if analysis_root.exists() and not any(analysis_root.iterdir()):
        analysis_root.rmdir()



def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    arch_replacements = migrate_architecture_runs(repo_root, dry_run=args.dry_run)
    moved_crossline = migrate_crossline_runs(
        repo_root,
        dry_run=args.dry_run,
        replacements=dict(arch_replacements),
    )

    if not args.dry_run:
        refresh_compact_runs(repo_root, dict(arch_replacements))
        cleanup_legacy_dirs(repo_root)

    print(
        "[done] "
        f"architecture_runs={len(arch_replacements)} crossline_runs={moved_crossline} "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
