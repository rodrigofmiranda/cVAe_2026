#!/usr/bin/env bash
# Seed-sweep por geometria (C4): hibrido S35CG6A em FS e FC, seeds 34/35/36,
# budget reduzido (cap 100k), UM POR VEZ, container --memory limitado, ntfy/run.
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
WORKDIR=/workspace/2026/feat_seq_bigru_residual_cvae
TOPIC=projeto_vlc_ia
send(){ curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1; }

declare -A DSROOT=( [fs]=/data/Dataset/V3/FULLSQUARE_2026_V3_ORGANIZED [fc]=/data/Dataset/V3/FULL_CIRCLE_2026_V3_ORGANIZED )
declare -A PROT=( [fs]=configs/protocol_v3fs_cross_train.json [fc]=configs/protocol_v3fc_cross_train.json )

for SEED in 34 35 36; do
  for G in fs fc; do
    NAME=v3_seedsweep_${G}_s${SEED}
    CNAME=cvae_ss_${G}_s${SEED}
    docker rm -f "$CNAME" >/dev/null 2>&1 || true
    mkdir -p "$REPO/outputs/seedsweep_geom/$NAME"
    docker run --rm --name "$CNAME" \
      --runtime=nvidia --security-opt apparmor=unconfined \
      -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
      -e TF_CPP_MIN_LOG_LEVEL=2 -e CVAE_BOOTSTRAP_PLOT_DEPS=0 \
      -e CVAE_DETERMINISTIC=1 -e CVAE_DET_OPDET=0 -e CVAE_DET_SETSEED=0 \
      -u "$(id -u):$(id -g)" -e HOME="$WORKDIR" \
      -e CVAE_DECODER_LOGVAR_CLAMP_LO=-6.61 -e CVAE_DECODER_LOGVAR_CLAMP_HI=-0.54 \
      --memory=24g --memory-swap=24g \
      -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
      -v "$REPO":"$WORKDIR" \
      -v /home/rodrigo/cVAe_2026_full_square/.pydeps:"$WORKDIR"/.pydeps \
      -v /home/rodrigo/cVAe_2026_full_square/.git:/home/rodrigo/cVAe_2026_full_square/.git:ro \
      -v /home/rodrigo/1-Data:/data:ro \
      -w "$WORKDIR" \
      --entrypoint bash vlc/tf25-gpu-ready:1 -lc "
        source scripts/ops/container_bootstrap_python.sh 2>/dev/null || true
        export PYTHONPATH=\$PWD:\$PWD/.pydeps
        python -u -m src.protocol.run \
          --dataset_root ${DSROOT[$G]} \
          --output_base outputs/seedsweep_geom/$NAME \
          --protocol ${PROT[$G]} \
          --train_once_eval_all \
          --grid_preset v3_g6_aligned_s35c --grid_tag S35CG6A_multibw_energy_lmmd05_le05 \
          --seed $SEED --no_data_reduction --max_samples_per_exp 100000 \
          --stat_tests --stat_mode quick --stat_seed 42 \
          --train_regime_diagnostics_focus_only_0p8m 0 \
          > outputs/seedsweep_geom/$NAME/run.log 2>&1
      "
    R=$(python3 - "$REPO/outputs/seedsweep_geom/$NAME" <<'PY'
import csv,glob,sys
p=sorted(glob.glob(sys.argv[1]+"/exp_*/tables/protocol_leaderboard.csv"))
if p:
    r=list(csv.DictReader(open(p[-1])))[0]; print(f"twin {r.get('n_pass','?')}/{r.get('n_regimes','?')}")
else: print("sem leaderboard")
PY
)
    send "seed-sweep ${G^^} s${SEED}" "$R"
  done
done
send "🏁 seed-sweep geometria concluido" "FS/FC x seeds 34/35/36 prontos em outputs/seedsweep_geom/"
