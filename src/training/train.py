# -*- coding: utf-8 -*-
"""
Deprecated training CLI shim.

`src.training.train` is intentionally no longer a valid experiment entrypoint.
All serious runs must go through `src.protocol.run` so exploration and
scientific validation use the same orchestration path and output contract.
"""

from __future__ import annotations

import sys


def _migration_message(argv: list[str]) -> str:
    forwarded = " ".join(argv[1:]).strip()
    has_protocol_flag = "--train_once_eval_all" in argv[1:]
    extra = f" {forwarded}" if forwarded else ""
    protocol_suffix = "" if has_protocol_flag else " --train_once_eval_all"
    return (
        "src.training.train was removed as a public entrypoint.\n\n"
        "Use src.protocol.run instead.\n\n"
        "Equivalent starting point:\n"
        "  python -m src.protocol.run"
        f"{extra}{protocol_suffix}\n\n"
        "Rationale:\n"
        "- there is now a single experiment path\n"
        "- exploratory grids and final validation must use the same protocol\n"
        "- output layout and ranking now live under protocol.run only\n"
    )


def main() -> None:
    raise SystemExit(_migration_message(sys.argv))


if __name__ == "__main__":
    main()
