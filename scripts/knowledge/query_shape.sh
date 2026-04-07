#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${KNOWLEDGE_PYTHON:-$REPO_ROOT/.venvs/knowledge/bin/python}"
QUERY_SCRIPT="$REPO_ROOT/scripts/knowledge/query_knowledge.py"

usage() {
  cat <<'EOF'
Usage:
  scripts/knowledge/query_shape.sh <preset|query> [extra query_knowledge args...]

Presets:
  shape           probabilistic shaping + VLC + nonlinearity
  twin            digital twin + channel modeling + VLC
  e2e             end-to-end learned constellations under power/nonlinearity constraints
  edge            edge/tail/support mismatch in learned channel models
  askari          focus on Askari/Lampe and related shaping work
  shu             focus on Shu 2020 and capacity-oriented shaping
  project         broad project shortlist spanning shape + twin + e2e
  custom          use the remaining positional args as the raw query text

Examples:
  scripts/knowledge/query_shape.sh shape
  scripts/knowledge/query_shape.sh twin --top-k 10
  scripts/knowledge/query_shape.sh custom "visible light communication digital twin nonlinearity"
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

MODE="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
shift || true

DEFAULT_ARGS=(
  --source both
  --dedupe-paper
  --show-paths
  --top-k 8
)

case "$MODE" in
  shape)
    QUERY="probabilistic shaping nonlinearity tolerance visible light communication"
    ;;
  twin)
    QUERY="visible light communication digital twin channel modeling nonlinearity generative model"
    ;;
  e2e)
    QUERY="end-to-end visible light communication constellation shaping average power nonlinearity"
    ;;
  edge)
    QUERY="tail coverage edge support corner mismatch generative channel model constellation"
    ;;
  askari)
    QUERY="probabilistic shaping nonlinearity tolerance"
    DEFAULT_ARGS+=(--paper-pattern "askari|lampe|nonlinearity-tolerant|temporal probabilistic shaping")
    ;;
  shu)
    QUERY="probabilistic shaping shannon limit channel capacity optical communications"
    DEFAULT_ARGS+=(--paper-pattern "shu|shannon|capacity|optical")
    ;;
  project)
    QUERY="probabilistic shaping digital twin visible light communication end-to-end nonlinearity average power"
    ;;
  custom)
    if [[ $# -lt 1 ]]; then
      echo "custom mode requires a query string." >&2
      exit 1
    fi
    QUERY="$1"
    shift || true
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    QUERY="$MODE"
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Knowledge Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$QUERY_SCRIPT" "$QUERY" "${DEFAULT_ARGS[@]}" "$@"
