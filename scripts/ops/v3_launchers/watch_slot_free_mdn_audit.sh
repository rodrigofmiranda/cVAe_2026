#!/usr/bin/env bash
# Waits for one of the 2 hot slots to free (2-slot doctrine, INFRA_GUIDE §2.1),
# then launches the MDN audit container in the freed slot.
set -u
send() { curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/projeto_vlc_ia" >/dev/null 2>&1; }

DEADLINE=$(( $(date +%s) + 6*3600 ))
while true; do
  N=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c '^cvae_v3_')
  [ "$N" -lt 2 ] && break
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    send "⚠️ MDN audit não disparado" "2 slots ocupados por mais de 6h; disparar manualmente."
    exit 1
  fi
  sleep 120
done
/home/rodrigo/cVAe_2026_full_square_v3det/scripts/analysis/run_mdn_audit_eval_reuse.sh
send "🔬 MDN audit disparado" "Slot liberou; auditoria MDN do S35C (eval-only reuse) rodando em cvae_v3_mdn_audit."
