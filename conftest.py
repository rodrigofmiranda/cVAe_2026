# conftest.py — project-root pytest configuration.
# Ensures ``import src.*`` works without setting PYTHONPATH manually.

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
