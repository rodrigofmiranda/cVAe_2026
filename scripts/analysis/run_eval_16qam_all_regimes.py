#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

TARGET = (
    Path(__file__).resolve().parents[1]
    / 'benchmarks'
    / 'modulations'
    / '16qam'
    / 'run_eval_16qam_all_regimes.py'
)

if __name__ == '__main__':
    runpy.run_path(str(TARGET), run_name='__main__')
