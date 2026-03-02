#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_BASE=${OUTPUT_BASE:-/workspace/2026/outputs}

python -u src/evaluation/analise_cvae_reviewed.py
