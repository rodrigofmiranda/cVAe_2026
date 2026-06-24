#!/usr/bin/env bash
# Macro post-modeling diagnostic pipeline — orchestrator.
#
#   --fast   (default)  Agrega APENAS os artefatos já existentes (CPU, sem docker).
#                       Roda build_macro_report.py -> summary_master.csv + REPORT.md.
#   --full              Re-roda a censura model-free + plots de cauda/LED em docker
#                       (numpy, CPU) e então agrega. Não precisa de GPU.
#   --regen-upstream    Além de --full, re-dispara os geradores upstream que
#                       dependem do modelo (GPU) via os run_*.sh existentes.
#                       Pesado e opt-in (NÃO roda por padrão).
#
# Os artefatos por execução vão para macro_diagnostics/runs/<stamp>/.
set -u

HERE=/home/rodrigo/comparison_v3/macro_diagnostics
ROOT=/home/rodrigo/comparison_v3
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
WORKDIR=/workspace/2026/feat_seq_bigru_residual_cvae
STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT="$HERE/runs/$STAMP"
MODE=fast
REGEN_UPSTREAM=0
send(){ curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/projeto_vlc_ia" >/dev/null 2>&1; }

for arg in "$@"; do
  case "$arg" in
    --fast) MODE=fast ;;
    --full) MODE=full ;;
    --regen-upstream) MODE=full; REGEN_UPSTREAM=1 ;;
    *) echo "arg desconhecido: $arg"; exit 2 ;;
  esac
done

mkdir -p "$OUT"
echo "=== Macro Diagnostics ($MODE) -> $OUT ==="

# ----- opcional: re-gerar camadas upstream dependentes do modelo (GPU) ---------
if [ "$REGEN_UPSTREAM" = "1" ]; then
  echo "--- regen upstream (GPU, pesado) ---"
  bash "$ROOT/run_compare_awgn_vs_cvae_v3.sh" || echo "WARN: awgn upstream falhou"
  bash "$ROOT/run_noise_var_vs_amp_v3.sh"     || echo "WARN: shot-noise upstream falhou"
  bash /home/rodrigo/run_modulations_all.sh    || echo "WARN: modulations upstream falhou"
  bash /home/rodrigo/run_modulations_unseen.sh || echo "WARN: modulations unseen upstream falhou"
fi

# ----- Camada 0: censura model-free (numpy, docker CPU) ------------------------
if [ "$MODE" = "full" ]; then
  echo "--- Camada 0: regime_census + plots de cauda/LED (docker, CPU) ---"
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
      python3 '"$HERE"'/regime_census.py --out '"$OUT"' --n-cap 60000
      python3 '"$HERE"'/lib/plot_led_synthesis_v3.py --census '"$OUT"'/regime_census.csv \
        --out '"$OUT"'/led_synthesis.png || echo "WARN led_synthesis"
      python3 '"$HERE"'/lib/plot_residual_tail_v3.py --census '"$OUT"'/regime_census.csv \
        --out '"$OUT"'/residual_tail_grid.png || echo "WARN residual_tail"
      python3 '"$HERE"'/lib/plot_residual_decomposition_v3.py --census '"$OUT"'/regime_census.csv \
        --out '"$OUT"'/residual_decomposition.png || echo "WARN residual_decomposition"
    ' 2>&1 | tee "$OUT/census.log" | grep -E "dataset_root|kurt=|escrito|WARN|Error|Traceback"
fi

# ----- Camada 7: síntese / agregação (pure-stdlib, host) ----------------------
echo "--- Camada 7: build_macro_report (agregação) ---"
python3 "$HERE/build_macro_report.py" --root "$ROOT" --out-dir "$OUT" 2>&1 | tee -a "$OUT/aggregate.log"

NPASS=$(grep -c "PASS" "$OUT/summary_master.csv" 2>/dev/null || echo "?")
echo "=== pronto: $OUT ==="
send "📐 Macro diagnostics ($MODE) pronto" "REPORT.md + summary_master.csv em macro_diagnostics/runs/$STAMP/"
echo "Veja: $OUT/REPORT.md"
