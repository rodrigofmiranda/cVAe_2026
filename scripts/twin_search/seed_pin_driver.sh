#!/usr/bin/env bash
# PHASE 0 of digital-twin search: find a DETERMINISTIC seed that lands the S39B
# baseline in the good basin (full 12-regime run). Native cvae_repro env + overlay
# of the 2 determinism-patched files. Early-kill bad seeds at epoch 40; the first
# good seed runs to completion and becomes the pinned baseline.
set -uo pipefail

THRESH=-4.6; EARLY_EPOCH=40; EARLY_THRESH=-4.3; MAX=12
BASE=/home/rodrigo/repro_c622/twin_search
INNER=/home/rodrigo/repro_c622/inner_seed.sh
PARSE=/home/rodrigo/repro_c622/parse_min.py
STATUS="$BASE/status.log"
CLONE=/home/rodrigo/cvae_repro_141943
PSRC=/home/rodrigo/cvae_det/src
mkdir -p "$BASE"
log(){ echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; echo "$*"; }

log "TWIN_SEED_SEARCH start deterministic full-12 thresh=$THRESH early=${EARLY_EPOCH}/${EARLY_THRESH}"

# Primary seed 33 (42 is overused); skip 42 entirely.
SEEDS=(33 34 35 36 37 38 39 40 41 43 44 45)
for SEED in "${SEEDS[@]}"; do
  SUB="seed_$SEED"
  OUT_HOST="$BASE/$SUB"; rm -rf "$OUT_HOST"; mkdir -p "$OUT_HOST"
  TL="$OUT_HOST/train.log"; cname="cvae_rodrigo_twin_$SEED"
  docker rm -f "$cname" >/dev/null 2>&1 || true
  rm -rf "$CLONE/outputs/twin_search/$SUB"
  log "SEED_BEGIN seed=$SEED"

  docker run --rm --name "$cname" \
    --runtime=nvidia --security-opt apparmor=unconfined \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e CVAE_DETERMINISTIC=1 -e CVAE_DET_ENV=1 -e CVAE_DET_OPDET=0 -e CVAE_DET_SETSEED=0 \
    -e REPRO_SEED="$SEED" -e REPRO_SUB="$SUB" \
    -e HOME=/workspace/2026/feat_seq_bigru_residual_cvae \
    -u "$(id -u):$(id -g)" \
    -v "$CLONE":/workspace/2026/feat_seq_bigru_residual_cvae \
    -v "$PSRC/protocol/run.py":/workspace/2026/feat_seq_bigru_residual_cvae/src/protocol/run.py:ro \
    -v "$PSRC/training/pipeline.py":/workspace/2026/feat_seq_bigru_residual_cvae/src/training/pipeline.py:ro \
    -w /workspace/2026/feat_seq_bigru_residual_cvae \
    --entrypoint bash vlc/tf25-gpu-ready:1 -lc "$(cat "$INNER")" > "$TL" 2>&1 &
  runpid=$!

  killed=0
  while kill -0 "$runpid" 2>/dev/null; do
    sleep 45
    [ -f "$TL" ] || continue
    read -r ep mn < <(python3 "$PARSE" "$TL" 2>/dev/null) || true
    [ -z "${ep:-}" ] && continue; [ "$ep" = "0" ] && continue
    if [ "$ep" -ge "$EARLY_EPOCH" ]; then
      bad=$(python3 -c "print(1 if $mn > $EARLY_THRESH else 0)" 2>/dev/null || echo 0)
      if [ "$bad" = "1" ]; then
        log "EARLY_KILL seed=$SEED epoch=$ep min=$mn"
        docker kill "$cname" >/dev/null 2>&1 || true; killed=1; break
      fi
    fi
  done
  wait "$runpid" 2>/dev/null || true

  TH=$(ls "$CLONE"/outputs/twin_search/$SUB/exp_*/logs/train/training_history.json 2>/dev/null | head -1)
  if [ -n "${TH:-}" ]; then
    MIN=$(python3 -c "import json;v=[float(x) for x in json.load(open('$TH'))['history']['val_recon_loss']];print(round(min(v),4))" 2>/dev/null)
    EP=$(python3 -c "import json;print(json.load(open('$TH')).get('epochs_ran'))" 2>/dev/null)
  else
    read -r EP MIN < <(python3 "$PARSE" "$TL" 2>/dev/null) || true
  fi
  if [ -z "${MIN:-}" ] || [ "$MIN" = "nan" ]; then log "SEED_RESULT seed=$SEED verdict=ERROR killed=$killed"; continue; fi
  GOOD=$(python3 -c "print(1 if $MIN <= $THRESH else 0)" 2>/dev/null)
  if [ "$GOOD" = "1" ]; then
    EXP=$(dirname "$(dirname "$(dirname "$TH")")")
    log "SEED_RESULT seed=$SEED verdict=GOOD min=$MIN epochs=$EP"
    echo "$SEED" > "$BASE/PINNED_SEED.txt"; echo "$EXP" > "$BASE/BASELINE_EXP.txt"
    log "TWIN_SEED_PINNED seed=$SEED exp=$EXP"
    exit 0
  else
    log "SEED_RESULT seed=$SEED verdict=BAD min=$MIN epochs=$EP killed=$killed"
  fi
done
log "TWIN_SEED_GAVEUP"
exit 1
