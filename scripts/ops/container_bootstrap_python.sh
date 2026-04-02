#!/usr/bin/env bash
set -euo pipefail

# Persistent Python deps for the mounted repo (survives container restarts).
WORKDIR="${CVAE_TF25_WORKDIR:-$PWD}"
PYDEPS_DIR="${CVAE_PYDEPS_DIR:-${WORKDIR}/.pydeps}"
MPL_DIR="${CVAE_MPLCONFIGDIR:-${WORKDIR}/.mplconfig}"
PIP_CACHE="${CVAE_PIP_CACHE_DIR:-${WORKDIR}/.cache/pip}"
BOOTSTRAP_PLOT_DEPS="${CVAE_BOOTSTRAP_PLOT_DEPS:-1}"

export HOME="${WORKDIR}"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPL_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE}"

mkdir -p "${PYDEPS_DIR}" "${MPL_DIR}" "${PIP_CACHE}"

if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="${WORKDIR}:${PYDEPS_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${WORKDIR}:${PYDEPS_DIR}"
fi

if [ "${BOOTSTRAP_PLOT_DEPS}" = "1" ]; then
  if ! python3 -c "import matplotlib" >/dev/null 2>&1; then
    echo "[bootstrap] matplotlib not found; installing persistent plot deps in ${PYDEPS_DIR}"
    if ! python3 -m pip install --no-cache-dir --target "${PYDEPS_DIR}" "numpy<2" "matplotlib<3.9"; then
      echo "[bootstrap][warn] plot deps install failed; dashboards may be skipped in this session."
    fi
  fi
fi

# Optional quick visibility line for troubleshooting.
python3 - <<'PY' || true
import os
print("[bootstrap] PYTHONNOUSERSITE=", os.environ.get("PYTHONNOUSERSITE", ""))
print("[bootstrap] PYTHONPATH=", os.environ.get("PYTHONPATH", ""))
print("[bootstrap] MPLCONFIGDIR=", os.environ.get("MPLCONFIGDIR", ""))
PY
