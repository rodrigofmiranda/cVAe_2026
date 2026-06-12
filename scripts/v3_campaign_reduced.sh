#!/usr/bin/env bash
# V3 baseline campaign (REDUCED data): train the best candidates so far on the
# V3 FULLSQUARE 12-regime protocol under an identical reduced training budget
# (200k contiguous samples/exp = 2.4M windows; val untouched) to elect the best
# digital-twin candidate. Sequential on the single GPU; ntfy per leg + final
# ranking. Run this script DETACHED (setsid nohup ... & disown).
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
BUCKET=outputs/v3_campaign_reduced_20260611
TOPIC="projeto_vlc_ia"
WORKDIR=/workspace/2026/feat_seq_bigru_residual_cvae

send() { curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1; }

# name|grid_preset|grid_tag
CANDIDATES=(
  "s39b|seq_edgegap_targeted_short|S39B_edgegap_lowlr_all08_w18_p120"
  "s38d|seq_cond_embed_fast_stage4_edgegap|S38D_smplmmd_cov30_t02_tail02-98_e96_emb3_resid_w08x18"
  "s35c|seq_cond_embed_fast_stage1|S35C_fast_e64_base"
  "v3g6a|v3_g6_aligned|V3G6A_s39b_multibw_energy_lmmd05_le05"
)

leg_result() { # <leg_dir> -> "twin X/N | full Y/N | screen Z/N" from newest leaderboard
  python3 - "$1" <<'PY'
import csv, glob, sys
paths = sorted(glob.glob(sys.argv[1] + "/exp_*/tables/protocol_leaderboard.csv"))
if not paths:
    print("sem leaderboard"); raise SystemExit
r = list(csv.DictReader(open(paths[-1])))[0]
n = r.get("n_regimes", "?")
print(f"twin {r.get('n_pass','?')}/{n} | full {r.get('n_full_pass','?')}/{n} | screen {r.get('stat_screen_pass','?')}/{n}")
PY
}

send "🏁 Campanha V3 (dados reduzidos) — iniciada" \
  "4 candidatos sequenciais: S39B, S38D, S35C, V3G6A. 12 regimes, 200k/exp, seed 33, det ON, stat tests ON."

SUMMARY=""
for SPEC in "${CANDIDATES[@]}"; do
  NAME="${SPEC%%|*}"; REST="${SPEC#*|}"; PRESET="${REST%%|*}"; TAG="${REST#*|}"
  LEGDIR="$REPO/$BUCKET/$NAME"
  mkdir -p "$LEGDIR"
  CNAME="cvae_v3_cmp_${NAME}"
  docker rm -f "$CNAME" >/dev/null 2>&1 || true

  docker run -d --rm --name "$CNAME" \
    --runtime=nvidia --security-opt apparmor=unconfined \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e TF_CPP_MIN_LOG_LEVEL=2 -e CVAE_BOOTSTRAP_PLOT_DEPS=0 \
    -e CVAE_DETERMINISTIC=1 -e CVAE_DET_OPDET=0 -e CVAE_DET_SETSEED=0 \
    -u "$(id -u):$(id -g)" \
    -e HOME="$WORKDIR" \
    -e CVAE_DECODER_LOGVAR_CLAMP_LO=-6.61 -e CVAE_DECODER_LOGVAR_CLAMP_HI=-0.54 \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -v "$REPO":"$WORKDIR" \
    -v /home/rodrigo/cVAe_2026_full_square/.pydeps:"$WORKDIR"/.pydeps \
    -v /home/rodrigo/cVAe_2026_full_square/.git:/home/rodrigo/cVAe_2026_full_square/.git:ro \
    -v /home/rodrigo/1-Data:/data:ro \
    -w "$WORKDIR" \
    --entrypoint bash vlc/tf25-gpu-ready:1 -lc "
      source scripts/ops/container_bootstrap_python.sh 2>/dev/null || true
      export PYTHONPATH=\$PWD
      python -u -m src.protocol.run \
        --dataset_root /data/Dataset/V3/FULLSQUARE_2026_V3_ORGANIZED \
        --output_base $BUCKET/$NAME \
        --protocol configs/protocol_v3_fullsquare.json \
        --train_once_eval_all \
        --grid_preset $PRESET --grid_tag $TAG \
        --seed 33 --no_data_reduction --max_samples_per_exp 200000 \
        --stat_tests --stat_mode quick --stat_seed 42 \
        --train_regime_diagnostics_focus_only_0p8m 0 \
        > $BUCKET/$NAME/run.log 2>&1
    "

  # wait for the leg to finish
  while docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CNAME"; do sleep 120; done

  RES=$(leg_result "$LEGDIR")
  EP=$(grep -c "^Epoch " "$LEGDIR/run.log" 2>/dev/null)
  SUMMARY="${SUMMARY}${NAME}: ${RES} (ep≈${EP})
"
  send "✅ Campanha V3 — ${NAME} concluído" "${TAG}: ${RES} (epochs≈${EP})"
done

send "🏆 Campanha V3 (reduzida) — RANKING FINAL" "$SUMMARY"
echo "$SUMMARY" > "$REPO/$BUCKET/CAMPAIGN_SUMMARY.txt"
