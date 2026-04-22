#!/usr/bin/env bash
set -euo pipefail

declare -Ar SESSION_NAMES=(
  [algo1]="algo1"
  [algo2]="algo2"
)

declare -Ar CONTAINER_NAMES=(
  [algo1]="cvae_algo1"
  [algo2]="cvae_algo2"
)

declare -Ar HOST_REPOS=(
  [algo1]="/home/rodrigo/cVAe_2026_shape_fullcircle"
  [algo2]="/home/rodrigo/cVAe_2026_shape_fullcircle"
)

declare -Ar CONTAINER_REPOS=(
  [algo1]="/workspace/2026/feat_seq_bigru_residual_cvae"
  [algo2]="/workspace/2026/feat_seq_bigru_residual_cvae"
)

usage() {
  cat <<'EOF'
Canonical two-slot helper for the always-on parallel sessions.

Usage:
  scripts/ops/canonical_dual_run.sh status
  scripts/ops/canonical_dual_run.sh enter <algo1|algo2>
  scripts/ops/canonical_dual_run.sh send <algo1|algo2> -- <command...>
  scripts/ops/canonical_dual_run.sh run <algo1|algo2> <run_tag> -- <command...>

Commands:
  status
    Show the canonical slots, their default directories, and whether tmux and
    Docker are up.

  enter <slot>
    Attach to the canonical tmux session for the slot.

  send <slot> -- <command...>
    Send a command to the canonical session after changing to the canonical
    container workdir.

  run <slot> <run_tag> -- <command...>
    Same as "send", but also injects:
      RUN_STAMP    UTC timestamp: YYYYMMDD_HHMMSS
      RUN_SLOT     algo1 | algo2
      RUN_TAG      user-provided semantic tag
      RUN_LOG_PATH outputs/_launch_logs/<RUN_STAMP>_<RUN_SLOT>_<RUN_TAG>.log
    and tees stdout/stderr to RUN_LOG_PATH.

Examples:
  scripts/ops/canonical_dual_run.sh status
  scripts/ops/canonical_dual_run.sh enter algo1
  scripts/ops/canonical_dual_run.sh send algo2 -- pwd
  scripts/ops/canonical_dual_run.sh run algo1 e2_full -- scripts/ops/train_support_final_full.sh e2
  scripts/ops/canonical_dual_run.sh run algo2 eval_16qam -- python3 scripts/analysis/run_eval_16qam_all_regimes.py --help
EOF
}

die() {
  echo "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

require_slot() {
  local slot="${1:-}"
  [[ -n "${SESSION_NAMES[$slot]+x}" ]] || die "Invalid slot '${slot}'. Use 'algo1' or 'algo2'."
}

require_tmux_session() {
  local slot="$1"
  local session="${SESSION_NAMES[$slot]}"
  tmux has-session -t "${session}" 2>/dev/null || die "tmux session '${session}' is not running."
}

sanitize_tag() {
  local tag="$1"
  [[ "${tag}" =~ ^[A-Za-z0-9._-]+$ ]] || die "Invalid run_tag '${tag}'. Use only letters, numbers, dot, underscore, or hyphen."
}

container_running() {
  local slot="$1"
  docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAMES[$slot]}"
}

session_running() {
  local slot="$1"
  tmux has-session -t "${SESSION_NAMES[$slot]}" 2>/dev/null
}

print_status() {
  local slot session_state container_state

  printf '%-8s %-8s %-12s %-42s %s\n' "slot" "tmux" "container" "host_repo" "container_repo"
  printf '%-8s %-8s %-12s %-42s %s\n' "--------" "--------" "------------" "------------------------------------------" "------------------------------"

  for slot in algo1 algo2; do
    if session_running "${slot}"; then
      session_state="up"
    else
      session_state="down"
    fi

    if container_running "${slot}"; then
      container_state="up"
    else
      container_state="down"
    fi

    printf '%-8s %-8s %-12s %-42s %s\n' \
      "${slot}" \
      "${session_state}" \
      "${container_state}" \
      "${HOST_REPOS[$slot]}" \
      "${CONTAINER_REPOS[$slot]}"
  done
}

send_payload() {
  local slot="$1"
  local payload="$2"
  local session="${SESSION_NAMES[$slot]}"

  tmux send-keys -t "${session}" "${payload}" C-m
}

join_quoted() {
  local arg
  for arg in "$@"; do
    printf '%q ' "${arg}"
  done
}

cmd_enter() {
  local slot="$1"
  local session="${SESSION_NAMES[$slot]}"
  require_tmux_session "${slot}"

  if [[ -n "${TMUX:-}" ]]; then
    exec tmux switch-client -t "${session}"
  fi
  exec tmux attach -t "${session}"
}

cmd_send() {
  local slot="$1"
  shift
  [[ "${1:-}" == "--" ]] && shift
  [[ $# -gt 0 ]] || die "send requires a command after '--'."

  require_tmux_session "${slot}"

  local workdir="${CONTAINER_REPOS[$slot]}"
  local user_cmd
  user_cmd="$(join_quoted "$@")"

  send_payload "${slot}" "cd $(printf '%q' "${workdir}") && ${user_cmd}"
  printf 'sent slot=%s workdir=%s\n' "${slot}" "${workdir}"
}

cmd_run() {
  local slot="$1"
  local run_tag="$2"
  shift 2
  [[ "${1:-}" == "--" ]] && shift
  [[ $# -gt 0 ]] || die "run requires a command after '--'."

  require_tmux_session "${slot}"
  sanitize_tag "${run_tag}"

  local workdir="${CONTAINER_REPOS[$slot]}"
  local user_cmd
  user_cmd="$(join_quoted "$@")"

  local payload
  payload="cd $(printf '%q' "${workdir}")"
  payload+=" && mkdir -p outputs/_launch_logs"
  payload+=" && RUN_STAMP=\$(date -u +%Y%m%d_%H%M%S)"
  payload+=" && RUN_SLOT=${slot}"
  payload+=" && RUN_TAG=${run_tag}"
  payload+=" && RUN_LOG_PATH=\"outputs/_launch_logs/\${RUN_STAMP}_${slot}_${run_tag}.log\""
  payload+=" && export RUN_STAMP RUN_SLOT RUN_TAG RUN_LOG_PATH"
  payload+=" && echo \"[canonical-run] slot=${slot} tag=${run_tag} workdir=${workdir}\""
  payload+=" && echo \"[canonical-run] RUN_STAMP=\${RUN_STAMP}\""
  payload+=" && echo \"[canonical-run] RUN_LOG_PATH=\${RUN_LOG_PATH}\""
  payload+=" && ${user_cmd} 2>&1 | tee \"\${RUN_LOG_PATH}\""

  send_payload "${slot}" "${payload}"
  printf 'launched slot=%s tag=%s workdir=%s\n' "${slot}" "${run_tag}" "${workdir}"
}

main() {
  require_cmd tmux
  require_cmd docker

  local subcmd="${1:-}"
  case "${subcmd}" in
    status)
      print_status
      ;;
    enter)
      shift
      require_slot "${1:-}"
      cmd_enter "$1"
      ;;
    send)
      shift
      require_slot "${1:-}"
      local slot_send="$1"
      shift
      cmd_send "${slot_send}" "$@"
      ;;
    run)
      shift
      require_slot "${1:-}"
      local slot_run="$1"
      local run_tag="${2:-}"
      [[ -n "${run_tag}" ]] || die "run requires <run_tag>."
      shift 2
      cmd_run "${slot_run}" "${run_tag}" "$@"
      ;;
    -h|--help|help|"")
      usage
      ;;
    *)
      die "Unknown subcommand '${subcmd}'. Use --help for usage."
      ;;
  esac
}

main "$@"
