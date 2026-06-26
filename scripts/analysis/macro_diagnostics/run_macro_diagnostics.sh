#!/usr/bin/env bash
# Macro post-modeling diagnostic pipeline — orchestrator (self-contained runs).
#
# Cada execução produz UM diretório auto-contido em macro_diagnostics/runs/<stamp>__<label>/
# com a estrutura:
#   0_census/  1_xcorr/  2_awgn/  3_gates/  4_crossdist/  5_modulations/  (flat: inferência, todas as distâncias)
#   REPORT.md (com seção "Figuras" linkando tudo) · summary_master.csv · manifest.json
#
# Modos:
#   --fast (default)  Coleta os artefatos já existentes (CPU, sem docker) + agrega.
#   --full            Re-roda a censura model-free + plots (docker CPU) e agrega.
#   --regen-upstream  Além de --full, re-dispara os geradores upstream (GPU) via run_*.sh.
#   --label <nome>    Rótulo do run (default: fc_fs_base).
set -u

HERE=/home/rodrigo/comparison_v3/macro_diagnostics
ROOT=/home/rodrigo/comparison_v3
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
WORKDIR=/workspace/2026/feat_seq_bigru_residual_cvae
STAMP=$(date -u +%Y%m%d_%H%M%S)
MODE=fast
REGEN_UPSTREAM=0
LABEL=fc_fs_base
send(){ curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/projeto_vlc_ia" >/dev/null 2>&1; }

args=("$@")
i=0
while [ $i -lt ${#args[@]} ]; do
  case "${args[$i]}" in
    --fast) MODE=fast ;;
    --full) MODE=full ;;
    --regen-upstream) MODE=full; REGEN_UPSTREAM=1 ;;
    --label) i=$((i+1)); LABEL="${args[$i]:-fc_fs_base}" ;;
    *) echo "arg desconhecido: ${args[$i]}"; exit 2 ;;
  esac
  i=$((i+1))
done

OUT="$HERE/runs/${STAMP}__${LABEL}"
mkdir -p "$OUT"/{0_census,1_xcorr,2_awgn,3_gates,4_crossdist,5_modulations}
echo "=== Macro Diagnostics ($MODE, label=$LABEL) -> $OUT ==="

# ----- opcional: re-gerar camadas upstream dependentes do modelo (GPU) ---------
if [ "$REGEN_UPSTREAM" = "1" ]; then
  echo "--- regen upstream (GPU, pesado) ---"
  bash "$ROOT/run_compare_awgn_vs_cvae_v3.sh" || echo "WARN: awgn upstream falhou"
  bash "$ROOT/run_noise_var_vs_amp_v3.sh"     || echo "WARN: shot-noise upstream falhou"
  # modulações UNIFICADAS (7 distâncias, figura única — inferência, sem split trained/unseen)
  bash "$REPO/scripts/ops/v3_launchers/run_modulations.sh" || echo "WARN: modulations upstream falhou"
fi

# ----- Camada 0: censura model-free (numpy, docker CPU) -> 0_census/ -----------
if [ "$MODE" = "full" ]; then
  echo "--- Camada 0: regime_census + plots (docker, CPU) -> 0_census/ ---"
  docker run --rm --name macro_census_$STAMP \
    --security-opt apparmor=unconfined \
    -e TF_CPP_MIN_LOG_LEVEL=3 -e CVAE_BOOTSTRAP_PLOT_DEPS=1 \
    -u "$(id -u):$(id -g)" -e HOME="$WORKDIR" -e CVAE_REPO="$WORKDIR" \
    --memory=16g --memory-swap=16g \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -v /home/rodrigo:/home/rodrigo \
    -v "$REPO":"$WORKDIR" \
    -v /home/rodrigo/cVAe_2026_full_square/.pydeps:"$WORKDIR"/.pydeps \
    -v /home/rodrigo/1-Data:/data:ro \
    -w "$WORKDIR" \
    --entrypoint bash vlc/tf25-gpu-ready:1 -lc '
      source scripts/ops/container_bootstrap_python.sh 2>/dev/null || true
      python3 '"$HERE"'/regime_census.py --out '"$OUT"'/0_census --n-cap 60000
      python3 '"$HERE"'/lib/plot_led_synthesis_v3.py --census '"$OUT"'/0_census/regime_census.csv \
        --out '"$OUT"'/0_census/led_synthesis.png || echo "WARN led_synthesis"
      python3 '"$HERE"'/lib/plot_residual_tail_v3.py --census '"$OUT"'/0_census/regime_census.csv \
        --out '"$OUT"'/0_census/residual_tail_grid.png || echo "WARN residual_tail"
      python3 '"$HERE"'/lib/plot_residual_decomposition_v3.py --census '"$OUT"'/0_census/regime_census.csv \
        --out '"$OUT"'/0_census/residual_decomposition.png || echo "WARN residual_decomposition"
    ' 2>&1 | tee "$OUT/census.log" | grep -E "dataset_root|kurt=|escrito|WARN|Error|Traceback"
else
  # --fast: reaproveita o regime_census.csv mais recente já existente
  LAST_CENSUS=$(ls -t "$HERE"/runs/*/0_census/regime_census.csv "$HERE"/runs/*/regime_census.csv "$ROOT"/macro_diagnostics/regime_census.csv 2>/dev/null | grep -v "$OUT" | head -1)
  [ -n "${LAST_CENSUS:-}" ] && cp "$LAST_CENSUS" "$OUT/0_census/regime_census.csv" 2>/dev/null && echo "  census reaproveitado: $LAST_CENSUS"
  # plots de censo ficam ao lado do csv (layout novo: .../0_census/; antigo: .../<run>/)
  CENSUS_DIR=$(dirname "${LAST_CENSUS:-/x/y}")
  for png in led_synthesis residual_tail_grid residual_decomposition; do
    cp "$CENSUS_DIR/$png.png" "$CENSUS_DIR/0_census/$png.png" "$OUT/0_census/" 2>/dev/null || true
  done
fi

# ----- COLETA: pastas fixas (scratch) -> subpastas auto-contidas do run --------
echo "--- coletando figuras/tabelas para o run auto-contido ---"
# 1_xcorr
cp -f "$ROOT"/cross_correlation/* "$OUT/1_xcorr/" 2>/dev/null || true
# 2_awgn (inclui subpastas fc/ fs/)
cp -rf "$ROOT"/awgn/. "$OUT/2_awgn/" 2>/dev/null || true
# 3_gates (CSVs + figura trained) e 4_crossdist (figuras unseen/crossdist)
cp -f "$ROOT"/fs_vs_fc_crossdist/*.csv "$OUT/3_gates/" 2>/dev/null || true
cp -f "$ROOT"/fs_vs_fc_crossdist/*trained*.png "$OUT/3_gates/" 2>/dev/null || true
cp -f "$ROOT"/fs_vs_fc_crossdist/*unseen*.png "$ROOT"/fs_vs_fc_crossdist/*crossdist*.png "$OUT/4_crossdist/" 2>/dev/null || true
# 5_modulations — FLAT (modulação é 100% inferência; gerada UNIFICADA nas 7 distâncias).
cp -rf "$ROOT"/modulations/. "$OUT/5_modulations/" 2>/dev/null || true

# ----- Camada 7: síntese / agregação (pure-stdlib, host, lê o layout N_*/) -----
echo "--- Camada 7: build_macro_report (--in-dir, lê o run auto-contido) ---"
python3 "$HERE/build_macro_report.py" --in-dir "$OUT" 2>&1 | tee "$OUT/aggregate.log"

echo "=== pronto: $OUT ==="
send "📐 Macro diagnostics ($MODE) pronto" "run auto-contido em macro_diagnostics/runs/${STAMP}__${LABEL}/ (REPORT.md com Figuras)."
echo "Veja: $OUT/REPORT.md"
