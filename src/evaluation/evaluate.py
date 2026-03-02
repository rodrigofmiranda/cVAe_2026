#!/usr/bin/env python3
"""
CLI entrypoint (stub) for future modular evaluation pipeline.

Commit 1: scaffolding only.
Evaluation still runs via src/evaluation/analise_cvae_reviewed.py (called by scripts/eval.sh).
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.evaluation.evaluate",
        description="Evaluation CLI (stub). Not wired in Commit 1."
    )
    p.add_argument("--run_dir", type=str, required=False, help="Path to outputs/run_... (future).")
    return p


def main(argv: list[str] | None = None) -> int:
    _ = build_parser().parse_args(argv)
    print("evaluate.py is a stub (Commit 1). Use: python -u src/evaluation/analise_cvae_reviewed.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())