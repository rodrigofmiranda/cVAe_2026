#!/usr/bin/env bash
# One FULL-data deterministic baseline (twin_base = S39B) run to completion, to
# clear the doubt: does deterministic + FULL data reach the good basin?
# Seed-retry with flatness early-kill; the first good-basin draw runs to the end.
set -uo pipefail

THRESH=-4.6; MAX_TRIES=6
BASE=/home/rodrigo/repro_c622/twin_search
INNER=/home/rodrigo/repro_c622/inner_variant.sh
PARSE2=/home/rodrigo/repro_c622/parse_min2.py
STATUS="$BASE/fullbase_status.log"
CLONE=/home/rodrigo/cvae_repro_141943
PSRC=/home/rodrigo/cvae_det/src
mkdir -p "$BASE"
log(){ echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; echo "$*"; }

log "FULLBASE_START twin_base FULL-data deterministic thresh=$THRESH"
for ((t=0; t<MAX_TRIES; t++)); do
  SEED=$((33+t)); SUB="fullbase_s$SEED"
  OUT_HOST="$BASE/$SUB"; rm -rf "$OUT_HOST"; mkdir -p "$OUT_HOST"
  TL="$OUT_HOST/train.log"; cname="cvae_rodrigo_fullbase_$SEED"
  docker rm -f "$cname" >/dev/null 2>&1 || true
  rm -rf "$CLONE/outputs/twin_search/$SUB"
  log "FULLBASE_TRY seed=$SEED"
  docker run --rm --name "$cname" \
    --runtime=nvidia --security-opt apparmor=unconfined \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e CVAE_DETERMINISTIC=1 -e CVAE_DET_ENV=1 -e CVAE_DET_OPDET=0 -e CVAE_DET_SETSEED=0 \
    -e REPRO_MAX_SAMPLES=0 -e REPRO_PRESET=twin_sweep -e REPRO_TAG=twin_base \
    -e REPRO_SEED="$SEED" -e REPRO_SUB="$SUB" \
    -e HOME=/workspace/2026/feat_seq_bigru_residual_cvae \
    -u "$(id -u):$(id -g)" \
    -v "$CLONE":/workspace/2026/feat_seq_bigru_residual_cvae \
    -v "$PSRC/protocol/run.py":/workspace/2026/feat_seq_bigru_residual_cvae/src/protocol/run.py:ro \
    -v "$PSRC/training/pipeline.py":/workspace/2026/feat_seq_bigru_residual_cvae/src/training/pipeline.py:ro \
    -v "$PSRC/training/grid_plan.py":/workspace/2026/feat_seq_bigru_residual_cvae/src/training/grid_plan.py:ro \
    -w /workspace/2026/feat_seq_bigru_residual_cvae \
    --entrypoint bash vlc/tf25-gpu-ready:1 -lc "$(cat "$INNER")" > "$TL" 2>&1 &
  runpid=$!; killed=0
  while kill -0 "$runpid" 2>/dev/null; do
    sleep 45
    [ -f "$TL" ] || continue
    read -r ep mn mp < <(python3 "$PARSE2" "$TL" 2>/dev/null) || true
    [ -z "${ep:-}" ] && continue; [ "$ep" = "0" ] && continue
    if [ "$ep" -ge 50 ]; then
      stuck=$(python3 -c "print(1 if ($mn > -4.0 and ($mn-$mp) > -0.05) else 0)" 2>/dev/null || echo 0)
      if [ "$stuck" = "1" ]; then log "  EARLY_KILL seed=$SEED epoch=$ep min=$mn prev=$mp (flat bad plateau)"; docker kill "$cname">/dev/null 2>&1||true; killed=1; break; fi
    fi
  done
  wait "$runpid" 2>/dev/null || true
  TH=$(ls "$CLONE"/outputs/twin_search/$SUB/exp_*/logs/train/training_history.json 2>/dev/null | head -1)
  if [ -n "${TH:-}" ]; then
    MIN=$(python3 -c "import json;v=[float(x) for x in json.load(open('$TH'))['history']['val_recon_loss']];print(round(min(v),4))" 2>/dev/null)
    EP=$(python3 -c "import json;print(json.load(open('$TH')).get('epochs_ran'))" 2>/dev/null)
  else
    read -r EP MIN _ < <(python3 "$PARSE2" "$TL" 2>/dev/null) || true
  fi
  if [ -z "${MIN:-}" ] || [ "$MIN" = "nan" ]; then log "FULLBASE_RESULT seed=$SEED verdict=ERROR killed=$killed"; continue; fi
  GOOD=$(python3 -c "print(1 if $MIN <= $THRESH else 0)" 2>/dev/null)
  if [ "$GOOD" = "1" ]; then
    EXP=$(dirname "$(dirname "$(dirname "$TH")")"); LB="$EXP/tables/protocol_leaderboard.csv"
    log "FULLBASE_RESULT seed=$SEED verdict=GOOD min=$MIN epochs=$EP"
    [ -f "$LB" ] && log "FULLBASE_GATES $(python3 -c "import csv;r=list(csv.DictReader(open('$LB')))[0];print('n_pass='+str(r.get('n_pass')),'G='+'/'.join(str(r.get('gate_g%d_pass'%i)) for i in range(1,7)))" 2>/dev/null)"
    log "FULLBASE_DONE good seed=$SEED exp=$EXP"
    exit 0
  else
    log "FULLBASE_RESULT seed=$SEED verdict=BAD min=$MIN epochs=$EP killed=$killed"
  fi
done
log "FULLBASE_GAVEUP no good basin in $MAX_TRIES full-data deterministic seeds (-> determinism suspect)"
exit 1
