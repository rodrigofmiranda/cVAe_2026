#!/usr/bin/env python3

"""
CLI entrypoint (stub) for future modular training pipeline.

Commit 1: scaffolding only.
This CLI must not change current behavior; training still runs via
src/training/cvae_TRAIN_documented.py (called by scripts/train.sh).
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.training.train",
        description="Training CLI (stub). Not wired in Commit 1."
    )
    p.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config (future).")
    p.add_argument("--run_id", type=str, default=None, help="Override RUN_ID (future).")
    return p


def main(argv: list[str] | None = None) -> int:
    _ = build_parser().parse_args(argv)
    print("train.py is a stub (Commit 1). Use: python -u src/training/cvae_TRAIN_documented.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())