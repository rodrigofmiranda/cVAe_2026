#!/usr/bin/env bash
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
WORKDIR=/workspace/2026/feat_seq_bigru_residual_cvae
OUTDIR=cross_correlation/../comparison_crossdist
send(){ curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/projeto_vlc_ia" >/dev/null 2>&1; }
# esperar os 2 containers de cross-dist saírem
while docker ps --format '{{.Names}}' 2>/dev/null | grep -qE 'cvae_v3f[sc]_crossdist'; do sleep 180; done
mkdir -p "$REPO/comparison_crossdist"
docker run --rm --security-opt apparmor=unconfined \
  -u "$(id -u):$(id -g)" -e HOME="$WORKDIR" -e CUDA_VISIBLE_DEVICES=-1 -e TF_CPP_MIN_LOG_LEVEL=3 \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v "$REPO":"$WORKDIR" \
  -v /home/rodrigo/cVAe_2026_full_square/.pydeps:"$WORKDIR"/.pydeps \
  -w "$WORKDIR" --entrypoint bash vlc/tf25-gpu-ready:1 -lc "
    source scripts/ops/container_bootstrap_python.sh 2>/dev/null || true
    python scripts/analysis/make_gates_side_by_side.py \
      --left outputs/v3fs_crossdist_20260613 --left-label FS \
      --right outputs/v3fc_crossdist_20260613 --right-label FC \
      --out comparison_crossdist/gates_summary_FS_vs_FC_crossdist \
      --unseen 0.9,1.16,1.25 \
      --title 'Gate pass/fail por regime — FS vs FC (cross-distance, 7 distancias)'
  " >> /home/rodrigo/trigger_crossdist_table.log 2>&1
if [ -f "$REPO/comparison_crossdist/gates_summary_FS_vs_FC_crossdist.png" ]; then
  send "📊 Tabela FS vs FC (cross-dist) pronta" "comparison_crossdist/gates_summary_FS_vs_FC_crossdist.{csv,png} — unseen 0.9/1.16/1.25 destacadas."
else
  send "⚠️ Tabela FS vs FC falhou" "ver trigger_crossdist_table.log"
fi
