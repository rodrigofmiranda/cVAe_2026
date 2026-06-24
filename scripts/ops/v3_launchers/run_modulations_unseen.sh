#!/usr/bin/env bash
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
WORKDIR=/workspace/2026/feat_seq_bigru_residual_cvae
DEST=/home/rodrigo/comparison_v3/modulations_unseen
send(){ curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/projeto_vlc_ia" >/dev/null 2>&1; }
mkdir -p "$DEST"
FS_MODEL=outputs/v3fs_crossdist_20260613/exp_20260613_205805/train
FC_MODEL=outputs/v3fc_crossdist_20260613/exp_20260613_211826/train
docker run --rm --name cvae_modulations_unseen \
  --runtime=nvidia --security-opt apparmor=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e TF_CPP_MIN_LOG_LEVEL=3 -e CVAE_BOOTSTRAP_PLOT_DEPS=1 \
  -u "$(id -u):$(id -g)" -e HOME="$WORKDIR" -e CVAE_REPO="$WORKDIR" \
  -e CVAE_DECODER_LOGVAR_CLAMP_LO=-6.61 -e CVAE_DECODER_LOGVAR_CLAMP_HI=-0.54 \
  --memory=18g --memory-swap=18g \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v "$REPO":"$WORKDIR" -v /home/rodrigo/cVAe_2026_full_square/.pydeps:"$WORKDIR"/.pydeps \
  -v /home/rodrigo/1-Data:/data:ro -w "$WORKDIR" \
  --entrypoint bash vlc/tf25-gpu-ready:1 -lc '
    source scripts/ops/container_bootstrap_python.sh 2>/dev/null || true
    for spec in "FS:'"$FS_MODEL"'" "FC:'"$FC_MODEL"'"; do
      LBL=${spec%%:*}; MODEL=${spec#*:}
      for MOD in 4QAM 16QAM 64QAM; do
        python -u scripts/analysis/modulation_check.py --model "$MODEL" \
          --modulation ${MOD}_2026_V3_ORGANIZED --dists 0.9,1.16,1.25 \
          --currents 100,300,500,700,900 --n-cap 50000 --label $LBL \
          --out outputs/v3_modulations_unseen/${LBL}_${MOD} 2>&1 \
          | grep -E "BER real|escrito|Error" | grep -vE "ptx85"
      done
    done

    echo "===== Plotting BER Comparison ====="
    python3 scripts/analysis/plot_ber_comparison.py \
      --dir outputs/v3_modulations_unseen \
      --out outputs/v3_modulations_unseen/ber_comparison_unseen.png \
      --title "Comparação de BER (Real vs cVAE vs AWGN) — Distâncias NÃO-Vistas"

    echo "===== Plotting BER Fidelity ====="
    python3 scripts/analysis/plot_ber_fidelity.py \
      --dir outputs/v3_modulations_unseen \
      --out outputs/v3_modulations_unseen/ber_fidelity_unseen.png \
      --title "Fidelidade de BER (Discrepância Absoluta contra o Real) — Distâncias NÃO-Vistas"
  ' > /home/rodrigo/run_modulations_unseen.log 2>&1
cp -r "$REPO/outputs/v3_modulations_unseen/." "$DEST/" 2>/dev/null
send "📊 Modulações UNSEEN (0.9/1.16/1.25) prontas" "comparison_v3/modulations_unseen/ — tabela de BER e gráfico ber_comparison_unseen.png gerado."
