#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocol.experiment_tracking import (
    prune_stale_incomplete_protocol_experiments,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove stale incomplete protocol experiments under outputs/."
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Root outputs directory (default: outputs)",
    )
    parser.add_argument(
        "--older-than-hours",
        type=float,
        default=24.0,
        help="Only prune runs older than this age in hours (default: 24)",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also prune runs explicitly marked as failed.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete directories. Without this flag, only prints what would be removed.",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).resolve()
    actions = prune_stale_incomplete_protocol_experiments(
        outputs_dir,
        older_than_hours=float(args.older_than_hours),
        dry_run=not bool(args.apply),
        remove_failed=bool(args.include_failed),
    )

    if not actions:
        print("No stale incomplete experiments found.")
        return

    for action in actions:
        verb = "deleted" if action["deleted"] else "would_delete"
        print(
            f"{verb}: {action['run_id']} "
            f"status={action['status']} "
            f"path={action['path']} "
            f"missing={action['missing_artifacts']}"
        )


if __name__ == "__main__":
    main()
