#!/usr/bin/env bash
# Detached watcher: notifies ntfy.sh/projeto_vlc_ia when the V3 deterministic
# run finishes (leaderboard written) or the container exits unexpectedly.
set -u
TOPIC="projeto_vlc_ia"
NAME="cvae_v3_s39b_det"
BASE="/home/rodrigo/cVAe_2026_full_square_v3det/outputs/v3_fullsquare_s39b_det_20260610"
LOG="$BASE/run.log"

send() { curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1; }

while true; do
  CSV=$(find "$BASE" -path '*tables/protocol_leaderboard.csv' 2>/dev/null | head -1)
  if [ -n "$CSV" ]; then
    EP=$(grep -c "^Epoch " "$LOG" 2>/dev/null)
    PASS=$(grep -oE "twin: [0-9]+/12 pass|[0-9]+ pass, [0-9]+ partial, [0-9]+ fail" "$LOG" 2>/dev/null | tail -1)
    BEST=$(grep -oE "val_recon_loss: [-0-9.e+]+" "$LOG" 2>/dev/null | awk '{print $2+0}' | sort -g | head -1)
    send "✅ V3 S39B det — concluído" "Run determinístico V3 terminou. epochs=$EP, melhor val_recon=$BEST. Gates: ${PASS:-ver leaderboard}. CSV: $CSV"
    break
  fi
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$NAME"; then
    EP=$(grep -c "^Epoch " "$LOG" 2>/dev/null)
    send "⚠️ V3 S39B det — container saiu sem leaderboard" "Container $NAME saiu no epoch ~$EP sem gerar leaderboard. Checar $LOG"
    break
  fi
  sleep 120
done
