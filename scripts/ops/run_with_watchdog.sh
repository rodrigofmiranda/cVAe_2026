#!/usr/bin/env bash
# run_with_watchdog.sh — lança um protocolo e aborta cedo se todos os modelos
# do grid falharem na primeira época (indicativo de OOM / conflito de GPU).
#
# Uso:
#   source scripts/ops/container_bootstrap_python.sh
#   bash scripts/ops/run_with_watchdog.sh <stdout_log> <stderr_log> <args...>
#
# Exemplo:
#   bash scripts/ops/run_with_watchdog.sh \
#     outputs/s25seq_stdout.log outputs/s25seq_stderr.log \
#     --dataset_root data/dataset_fullsquare_organized \
#     --output_base outputs \
#     --protocol configs/all_regimes_full_dataset.json \
#     --train_once_eval_all \
#     --grid_preset seq_mdn_v2_overnight_5090safe_quick \
#     --no_baseline --stat_tests --stat_mode quick --no_data_reduction

set -euo pipefail

STDOUT_LOG="$1"; shift
STDERR_LOG="$1"; shift

# Tempo de espera antes de checar (segundos) — tempo suficiente para a epoch 1
WATCHDOG_WAIT="${WATCHDOG_WAIT:-90}"
# Número de falhas consecutivas para abortar
FAIL_THRESHOLD="${FAIL_THRESHOLD:-2}"

echo "[watchdog] lançando: python -m src.protocol.run $*"
echo "[watchdog] stdout -> $STDOUT_LOG"
echo "[watchdog] stderr -> $STDERR_LOG"

python -m src.protocol.run "$@" >"$STDOUT_LOG" 2>"$STDERR_LOG" &
PID=$!
echo "[watchdog] PID=$PID"

# Aguarda a primeira época e verifica erros
sleep "$WATCHDOG_WAIT"

if ! kill -0 "$PID" 2>/dev/null; then
    echo "[watchdog] ❌ PROCESSO JÁ MORREU antes do watchdog checar."
    exit 1
fi

ERROS=$(grep -c "\[ERRO\]" "$STDOUT_LOG" 2>/dev/null || echo 0)
TOTAL_GRIDS=$(grep -c "🚀 GRID" "$STDOUT_LOG" 2>/dev/null || echo 0)

echo "[watchdog] grids iniciados=$TOTAL_GRIDS  falhas=[ERRO]=$ERROS"

if [ "$ERROS" -ge "$FAIL_THRESHOLD" ] && [ "$TOTAL_GRIDS" -le "$((ERROS + 1))" ]; then
    echo ""
    echo "[watchdog] ⛔ ABORTAR: $ERROS/$TOTAL_GRIDS modelos falharam já na epoch 1."
    echo "[watchdog] Causa provável: VRAM insuficiente (outro treino ocupando a GPU)."
    echo "[watchdog] Verifique: nvidia-smi"
    echo ""
    kill "$PID" 2>/dev/null || true
    exit 2
fi

echo "[watchdog] ✅ Treino parece saudável ($ERROS falhas em $TOTAL_GRIDS grids). Continuando."
echo "[watchdog] Para acompanhar: tail -f $STDOUT_LOG"

wait "$PID"
EXIT_CODE=$?
echo "[watchdog] processo terminou com exit=$EXIT_CODE"
exit $EXIT_CODE
