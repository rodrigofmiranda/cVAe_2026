#!/usr/bin/env bash
# Painel de resultados da reprodução. Rode: bash /home/rodrigo/repro_c622/results.sh
PARSE=/home/rodrigo/repro_c622/parse_min.py
ED=/home/eduardo/cVAe_2026/outputs/exp_20260427_141943

echo "============================================================"
echo " REPRODUÇÃO cVAE 141943 — PAINEL ($(date -u +%FT%TZ))"
echo "============================================================"

echo; echo "── VEREDITOS (status.log) ──────────────────────────────────"
for S in /home/rodrigo/repro_c622/native_retry/status.log /home/rodrigo/repro_c622/retry/status.log; do
  [ -f "$S" ] && { echo "[$S]"; grep -E "RETRY_START|ATTEMPT_RESULT|EARLY_KILL|RETRY_SUCCESS|RETRY_GAVEUP" "$S"; }
done

echo; echo "── TENTATIVA EM ANDAMENTO ──────────────────────────────────"
C=$(docker ps --filter name=cvae_rodrigo_native --filter name=cvae_rodrigo_retry --format '{{.Names}} ({{.Status}})' 2>/dev/null)
echo "container: ${C:-nenhum rodando}"
for B in /home/rodrigo/repro_c622/native_retry /home/rodrigo/repro_c622/retry; do
  L=$(ls -dt "$B"/attempt_* 2>/dev/null | head -1)
  [ -n "$L" ] && [ -f "$L/train.log" ] && python3 "$PARSE" "$L/train.log" 2>/dev/null | awk -v p="$L" '{print "  "p": épocas="$1" min_val_recon="$2}'
done

echo; echo "── RUNS BOAS COMPLETAS (val_recon <= -4.6) ─────────────────"
found=0
for TH in $(ls /home/rodrigo/cvae_repro_141943/outputs/retry_native/att_*/exp_*/logs/train/training_history.json \
               /home/rodrigo/repro_c622/outputs/retry/attempt_*/exp_*/logs/train/training_history.json 2>/dev/null); do
  m=$(python3 -c "import json;v=[float(x) for x in json.load(open('$TH'))['history']['val_recon_loss']];print(round(min(v),4))" 2>/dev/null)
  ok=$(python3 -c "print(1 if $m<=-4.6 else 0)" 2>/dev/null)
  if [ "$ok" = "1" ]; then
    found=1
    EXP=$(dirname "$(dirname "$(dirname "$TH")")")
    echo "  ✅ $EXP  (min val_recon=$m)"
    LB="$EXP/tables/protocol_leaderboard.csv"
    [ -f "$LB" ] && python3 - "$LB" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1])))[0]
print("     n_pass=%s n_fail=%s | gates G1..G6 = %s/%s/%s/%s/%s/%s"%(
 r.get("n_pass"),r.get("n_fail"),r.get("gate_g1_pass"),r.get("gate_g2_pass"),
 r.get("gate_g3_pass"),r.get("gate_g4_pass"),r.get("gate_g5_pass"),r.get("gate_g6_pass")))
PY
  fi
done
[ "$found" = "0" ] && echo "  (nenhuma run boa completa ainda)"

echo; echo "── REFERÊNCIA EDUARDO (alvo) ───────────────────────────────"
LB="$ED/tables/protocol_leaderboard.csv"
[ -f "$LB" ] && python3 - "$LB" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1])))[0]
print("  eduardo: n_pass=%s n_fail=%s | gates G1..G6 = %s/%s/%s/%s/%s/%s | min val_recon=-4.978"%(
 r.get("n_pass"),r.get("n_fail"),r.get("gate_g1_pass"),r.get("gate_g2_pass"),
 r.get("gate_g3_pass"),r.get("gate_g4_pass"),r.get("gate_g5_pass"),r.get("gate_g6_pass")))
PY
echo "============================================================"
