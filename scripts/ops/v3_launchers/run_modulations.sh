#!/usr/bin/env bash
# Verificação das modulações (UNIFICADA): FS e FC × 4/16/64-QAM em TODAS as 7
# distâncias × 9 correntes, num passo só. O modelo NUNCA treina em modulação
# (o treino é só FC/FS no canal) — modulação é 100% inferência, então não há
# distinção "treinada/não-vista" para a modulação; a distância é só uma variável.
# Gera UMA figura por tipo (ber_comparison.png / ber_fidelity.png, + _adaptive).
# Escreve em comparison_v3/modulations/ (scratch coletado pelo macro) e avisa no ntfy.
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
WORKDIR=/workspace/2026/feat_seq_bigru_residual_cvae
DEST=/home/rodrigo/comparison_v3/modulations
send(){ curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/projeto_vlc_ia" >/dev/null 2>&1; }
mkdir -p "$DEST"

FS_MODEL=outputs/v3fs_crossdist_20260613/exp_20260613_205805/train
FC_MODEL=outputs/v3fc_crossdist_20260613/exp_20260613_211826/train

docker run --rm --name cvae_modulations \
  --runtime=nvidia --security-opt apparmor=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e TF_CPP_MIN_LOG_LEVEL=3 -e CVAE_BOOTSTRAP_PLOT_DEPS=1 \
  -u "$(id -u):$(id -g)" -e HOME="$WORKDIR" -e CVAE_REPO="$WORKDIR" \
  -e CVAE_DECODER_LOGVAR_CLAMP_LO=-6.61 -e CVAE_DECODER_LOGVAR_CLAMP_HI=-0.54 \
  --memory=22g --memory-swap=22g \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v "$REPO":"$WORKDIR" \
  -v /home/rodrigo/cVAe_2026_full_square/.pydeps:"$WORKDIR"/.pydeps \
  -v /home/rodrigo/1-Data:/data:ro \
  -w "$WORKDIR" \
  --entrypoint bash vlc/tf25-gpu-ready:1 -lc '
    source scripts/ops/container_bootstrap_python.sh 2>/dev/null || true
    for spec in "FS:'"$FS_MODEL"'" "FC:'"$FC_MODEL"'"; do
      LBL=${spec%%:*}; MODEL=${spec#*:}
      for MOD in 4QAM 16QAM 64QAM; do
        echo "===== $LBL × $MOD ====="
        python -u scripts/analysis/modulation_check.py \
          --model "$MODEL" --modulation ${MOD}_2026_V3_ORGANIZED \
          --dists 0.75,0.9,1.0,1.16,1.25,1.35,1.5 \
          --currents 100,200,300,400,500,600,700,800,900 \
          --n-cap 50000 --label $LBL \
          --out outputs/v3_modulations/${LBL}_${MOD} 2>&1 \
          | grep -E "modelo|BER real|escrito|sem dados|Error|Traceback" | grep -vE "ptx85"
      done
    done

    echo "===== Plotting BER Comparison (unificado) ====="
    python3 scripts/analysis/plot_ber_comparison.py \
      --dir outputs/v3_modulations \
      --out outputs/v3_modulations/ber_comparison.png \
      --title "Comparação de BER (Real vs cVAE vs AWGN) — todas as distâncias (inferência)"

    echo "===== Plotting BER Fidelity (unificado) ====="
    python3 scripts/analysis/plot_ber_fidelity.py \
      --dir outputs/v3_modulations \
      --out outputs/v3_modulations/ber_fidelity.png \
      --title "Fidelidade de BER (Discrepância Absoluta contra o Real) — todas as distâncias (inferência)"
  ' > /home/rodrigo/run_modulations.log 2>&1

cp -r "$REPO/outputs/v3_modulations/." "$DEST/" 2>/dev/null
N=$(find "$DEST" -name "ber_table_*.csv" 2>/dev/null | wc -l)
send "📊 Modulações FS/FC × QAM (7 distâncias) prontas" "comparison_v3/modulations/ — ${N} tabelas BER, ber_comparison.png unificado."
