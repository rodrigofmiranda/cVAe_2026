#!/usr/bin/env bash
# PHASE 1 (targeted): run tail-fix variants for the 0.8m G5 kurtosis overshoot.
# Each variant = one full 12-regime deterministic run. Uses the pinned good seed;
# if a variant draws the bad basin, retries up to MAX_TRIES seeds. First good-basin
# run completes and its gates are recorded.
set -uo pipefail

THRESH=-4.3; MAX_TRIES=4   # good if final running-min <= -4.3 (bad basin never below -3.98)
BASE=/home/rodrigo/repro_c622/twin_search
INNER=/home/rodrigo/repro_c622/inner_variant.sh
PARSE=/home/rodrigo/repro_c622/parse_min.py
PARSE2=/home/rodrigo/repro_c622/parse_min2.py
STATUS="$BASE/variants_status.log"
CLONE=/home/rodrigo/cvae_repro_141943
PSRC=/home/rodrigo/cvae_det/src
mkdir -p "$BASE"
log(){ echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; echo "$*"; }

PINNED=$(cat "$BASE/PINNED_SEED.txt" 2>/dev/null || echo 33)
log "VARIANTS_START pinned_seed=$PINNED thresh=$THRESH"

# name | grid_preset | grid_tag | clamp_hi(env, empty=default)
# Ordered: baseline + primary tail-fix + alt base first, then broad exploration.
VARIANTS=(
  "base|twin_sweep|twin_base|"
  "kurt05|twin_sweep|twin_kurt05|"
  "kurt10|twin_sweep|twin_kurt10|"
  "mdn2|twin_sweep|twin_mdn2|"
  "mdn5|twin_sweep|twin_mdn5|"
  "clampHI12|twin_sweep|twin_base|-1.2"
  "clampHI10|twin_sweep|twin_base|-1.0"
  "s38d|seq_cond_embed_fast_stage4_edgegap|S38D_smplmmd_cov30_t02_tail02-98_e96_emb3_resid_w08x18|"
  "mmd15|twin_sweep|twin_mmd15|"
  "mmd40|twin_sweep|twin_mmd40|"
  "mmdmean|twin_sweep|twin_mmdmean|"
  "cov15|twin_sweep|twin_cov15|"
  "cov45|twin_sweep|twin_cov45|"
  "axis02|twin_sweep|twin_axis02|"
  "beta001|twin_sweep|twin_beta001|"
  "beta003|twin_sweep|twin_beta003|"
  "fb05|twin_sweep|twin_fb05|"
  "fb15|twin_sweep|twin_fb15|"
  "kl120|twin_sweep|twin_kl120|"
  "lat6|twin_sweep|twin_lat6|"
  "lat10|twin_sweep|twin_lat10|"
  "h192|twin_sweep|twin_h192|"
  "emb128|twin_sweep|twin_emb128|"
  "w7|twin_sweep|twin_w7|"
  "w11|twin_sweep|twin_w11|"
  "l3|twin_sweep|twin_l3|"
)

run_attempt() { # name preset tag clamp seed -> sets RESULT_MIN/RESULT_EP/RESULT_EXP/RESULT_KILLED
  local name=$1 preset=$2 tag=$3 clamp=$4 seed=$5
  local SUB="var_${name}_s${seed}"
  local OUT_HOST="$BASE/$SUB"; rm -rf "$OUT_HOST"; mkdir -p "$OUT_HOST"
  local TL="$OUT_HOST/train.log"; local cname="cvae_rodrigo_var_${name}_${seed}"
  docker rm -f "$cname" >/dev/null 2>&1 || true
  rm -rf "$CLONE/outputs/twin_search/$SUB"
  local clampenv=(); [ -n "$clamp" ] && clampenv=(-e "CVAE_DECODER_LOGVAR_CLAMP_HI=$clamp")
  docker run --rm --name "$cname" \
    --runtime=nvidia --security-opt apparmor=unconfined \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e CVAE_DETERMINISTIC=1 -e CVAE_DET_ENV=1 -e CVAE_DET_OPDET=0 -e CVAE_DET_SETSEED=0 \
    "${clampenv[@]}" \
    -e REPRO_PRESET="$preset" -e REPRO_TAG="$tag" -e REPRO_SEED="$seed" -e REPRO_SUB="$SUB" \
    -e HOME=/workspace/2026/feat_seq_bigru_residual_cvae \
    -u "$(id -u):$(id -g)" \
    -v "$CLONE":/workspace/2026/feat_seq_bigru_residual_cvae \
    -v "$PSRC/protocol/run.py":/workspace/2026/feat_seq_bigru_residual_cvae/src/protocol/run.py:ro \
    -v "$PSRC/training/pipeline.py":/workspace/2026/feat_seq_bigru_residual_cvae/src/training/pipeline.py:ro \
    -v "$PSRC/training/grid_plan.py":/workspace/2026/feat_seq_bigru_residual_cvae/src/training/grid_plan.py:ro \
    -w /workspace/2026/feat_seq_bigru_residual_cvae \
    --entrypoint bash vlc/tf25-gpu-ready:1 -lc "$(cat "$INNER")" > "$TL" 2>&1 &
  local runpid=$!; RESULT_KILLED=0
  while kill -0 "$runpid" 2>/dev/null; do
    sleep 45
    [ -f "$TL" ] || continue
    read -r ep mn mp < <(python3 "$PARSE2" "$TL" 2>/dev/null) || true
    [ -z "${ep:-}" ] && continue; [ "$ep" = "0" ] && continue
    # Kill ONLY if stuck in the flat ~-3.9 bad plateau (rmin>-4.0 AND no real
    # improvement over the last 30 epochs). This spares slow/late-blooming good
    # runs that are still descending. Safety hard-cap at epoch 200.
    if [ "$ep" -ge 50 ]; then
      stuck=$(python3 -c "print(1 if ($mn > -4.0 and ($mn-$mp) > -0.05) else 0)" 2>/dev/null || echo 0)
      hardcap=$(python3 -c "print(1 if ($ep >= 200 and $mn > -4.3) else 0)" 2>/dev/null || echo 0)
      if [ "$stuck" = "1" ] || [ "$hardcap" = "1" ]; then
        log "  EARLY_KILL $name seed=$seed epoch=$ep min=$mn prev=$mp (flat bad plateau)"
        docker kill "$cname" >/dev/null 2>&1 || true; RESULT_KILLED=1; break
      fi
    fi
  done
  wait "$runpid" 2>/dev/null || true
  local TH=$(ls "$CLONE"/outputs/twin_search/$SUB/exp_*/logs/train/training_history.json 2>/dev/null | head -1)
  RESULT_EXP=""; RESULT_MIN=""; RESULT_EP=""
  if [ -n "${TH:-}" ]; then
    RESULT_MIN=$(python3 -c "import json;v=[float(x) for x in json.load(open('$TH'))['history']['val_recon_loss']];print(round(min(v),4))" 2>/dev/null)
    RESULT_EP=$(python3 -c "import json;print(json.load(open('$TH')).get('epochs_ran'))" 2>/dev/null)
    RESULT_EXP=$(dirname "$(dirname "$(dirname "$TH")")")
  else
    read -r RESULT_EP RESULT_MIN < <(python3 "$PARSE" "$TL" 2>/dev/null) || true
  fi
}

report_gates() { # exp -> logs n_pass + 0.8m/300 G5
  local exp=$1; local lb="$exp/tables/protocol_leaderboard.csv"; local sr="$exp/tables/summary_by_regime.csv"
  python3 - "$lb" "$sr" <<'PY' 2>/dev/null
import csv,sys
lb,sr=sys.argv[1],sys.argv[2]
try:
    r=list(csv.DictReader(open(lb)))[0]
    g="/".join(str(r.get("gate_g%d_pass"%i)) for i in range(1,7))
    base="n_pass=%s n_fail=%s G1..6=%s"%(r.get("n_pass"),r.get("n_fail"),g)
except Exception as e: base="leaderboard? %s"%e
try:
    rows={x["regime_id"]:x for x in csv.DictReader(open(sr))}
    r3=rows.get("dist_0p8m__curr_300mA",{})
    extra=" | 0.8m/300: G5=%s delta_jb_rel=%s cvae_jb=%s"%(r3.get("gate_g5"),r3.get("delta_jb_stat_rel"),r3.get("cvae_jb_log10p_min"))
except Exception as e: extra=" | summary? %s"%e
print(base+extra)
PY
}

# Continuous: keep sweeping until the STOP sentinel appears (user says "chega")
# or every config has found a good basin. Configs that found a good basin are
# marked DONE_<name> and skipped on later passes; not-yet-good configs are
# retried with fresh seeds each pass.
PASS=0
while [ ! -f "$BASE/STOP" ]; do
  any_pending=0
  for entry in "${VARIANTS[@]}"; do
    [ -f "$BASE/STOP" ] && break
    IFS='|' read -r name preset tag clamp <<< "$entry"
    [ -f "$BASE/DONE_$name" ] && continue
    any_pending=1
    log "VARIANT_BEGIN $name preset=$preset tag=$tag clamp_hi=${clamp:-default} pass=$PASS"
    good=0
    for ((t=0; t<MAX_TRIES; t++)); do
      [ -f "$BASE/STOP" ] && break
      seed=$((PINNED + PASS*MAX_TRIES + t))
      log "  TRY $name seed=$seed"
      run_attempt "$name" "$preset" "$tag" "$clamp" "$seed"
      if [ -z "${RESULT_MIN:-}" ] || [ "$RESULT_MIN" = "nan" ]; then log "  RESULT $name seed=$seed ERROR"; continue; fi
      ok=$(python3 -c "print(1 if $RESULT_MIN <= $THRESH else 0)" 2>/dev/null)
      if [ "$ok" = "1" ]; then
        log "VARIANT_RESULT $name seed=$seed verdict=GOOD min=$RESULT_MIN epochs=$RESULT_EP"
        log "  GATES $name -> $(report_gates "$RESULT_EXP")"
        log "  EXP $name -> $RESULT_EXP"
        touch "$BASE/DONE_$name"; good=1; break
      else
        log "  RESULT $name seed=$seed BAD min=$RESULT_MIN (basin) killed=$RESULT_KILLED"
      fi
    done
    [ "$good" = "0" ] && log "VARIANT_RESULT $name pass=$PASS verdict=NO_GOOD_BASIN (retry next pass)"
  done
  [ "$any_pending" = "0" ] && { log "ALL_VARIANTS_GOOD (every config reached a good basin)"; break; }
  PASS=$((PASS+1))
done
log "VARIANTS_STOPPED pass=$PASS"
