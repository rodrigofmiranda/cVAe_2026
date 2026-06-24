#!/usr/bin/env bash
# Guard de prioridade: o E1 (cvae_v3fc_e1gauss) manda. Quando o braço HEAVY do E1
# começar (GRID 2/ ou tag _heavy_ no log), para a réplica seed7 (descartável) p/
# liberar a GPU ao experimento de controle. Se a réplica terminar antes, só avisa.
# Detached (setsid nohup) p/ sobreviver a cortes do harness. NÃO toca no eduardo.
set -u
TOPIC=projeto_vlc_ia
E1_LOG=/home/rodrigo/cVAe_2026_full_square_v3det/outputs/v3fc_e1gauss_20260617/run_train.log
REP=cvae_v3fc_e1gauss_s7
send(){ curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1; }

while true; do
  # réplica já saiu (terminou sozinha)?
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${REP}$"; then
    send "🔁 Réplica seed7 encerrada" "cvae_v3fc_e1gauss_s7 saiu (terminou ou foi parado). Guard finalizado."
    exit 0
  fi
  # braço heavy do E1 começou?
  if grep -qE "GRID 2/|_heavy_lmmd05" "$E1_LOG" 2>/dev/null; then
    docker stop "$REP" >/dev/null 2>&1
    send "🛑 Réplica parada p/ proteger E1" "braço HEAVY do E1 começou; parei a réplica seed7 e liberei a GPU ao controle. Veja a trajetória parcial de val_recon p/ comparar bacias."
    exit 0
  fi
  sleep 120
done
