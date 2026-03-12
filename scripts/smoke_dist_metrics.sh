#!/usr/bin/env bash
# scripts/smoke_dist_metrics.sh — Smoke-test for distribution-fidelity metrics.
#
# Runs the protocol runner with eval ON (fast: 1 epoch, 1 grid, 1 experiment,
# 2000 samples) and verifies that:
#   1) dist_metrics_source == "eval_reanalysis" in manifest for all regimes
#   2) cvae_delta_mean_l2 is populated in the summary CSV
#   3) delta_mean_l2 is populated (from eval or backfill)
#   4) selected_experiments appears in the manifest per regime (Commit 3Q)
#   5) n_experiments_selected is populated in the CSV (Commit 3Q)
#   6) baseline_delta_mean_l2 is populated (same val split, Commit 3S)
#   7) regime subdirectories exist under studies/<study>/regimes/ (Phase 5)
#
# Usage:
#   bash scripts/smoke_dist_metrics.sh
#
# Phase 5.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Smoke: dist-metrics (Phase 5 — exp_<ts> / studies layout) ==="

OUT_BASE="outputs"

python -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base  "$OUT_BASE" \
    --protocol     configs/protocol_default.json \
    --max_epochs 1 --max_grids 1 --max_experiments 1 \
    --max_samples_per_exp 2000 \
    --keras_verbose 0 \
    --dist_tol_m 0.01 --curr_tol_mA 1

# Find latest experiment dir
EXP_DIR=$(ls -td "$OUT_BASE"/exp_2* | head -1)
echo ""
echo "Experiment dir: $EXP_DIR"

FAIL=0

# --- Check 1: dist_metrics_source == "eval_reanalysis" in manifest ---
echo ""
echo "--- Check 1: dist_metrics_source == 'eval_reanalysis' for all regimes ---"
N_EVAL=$(python -c "
import json
m = json.load(open('$EXP_DIR/manifest.json'))
vals = [r.get('dist_metrics_source') for r in m['regimes']]
print(sum(1 for v in vals if v == 'eval_reanalysis'))
")
N_REG=$(python -c "
import json; m = json.load(open('$EXP_DIR/manifest.json')); print(len(m['regimes']))
")
if [ "$N_EVAL" -eq "$N_REG" ]; then
    echo "  ✓ All $N_REG regimes have dist_metrics_source='eval_reanalysis'"
else
    echo "  ✗ Only $N_EVAL / $N_REG regimes have 'eval_reanalysis' source"
    python -c "
import json; m = json.load(open('$EXP_DIR/manifest.json'))
for r in m['regimes']:
    print(f\"    {r['regime_id']}: {r.get('dist_metrics_source')}\")"
    FAIL=1
fi

# --- Check 2: cvae_delta_mean_l2 in CSV ---
echo ""
echo "--- Check 2: cvae_delta_mean_l2 in CSV ---"
HEADER=$(head -1 "$EXP_DIR/tables/summary_by_regime.csv")
if echo "$HEADER" | grep -q "cvae_delta_mean_l2"; then
    echo "  ✓ cvae_delta_mean_l2 column present"
    VAL=$(python -c "
import csv
with open('$EXP_DIR/tables/summary_by_regime.csv') as f:
    r = next(csv.DictReader(f))
    v = r.get('cvae_delta_mean_l2', '')
    print(v if v else 'EMPTY')
")
    if [ "$VAL" != "EMPTY" ]; then
        echo "  ✓ cvae_delta_mean_l2 has value: $VAL"
    else
        echo "  ✗ cvae_delta_mean_l2 is EMPTY"
        FAIL=1
    fi
else
    echo "  ✗ cvae_delta_mean_l2 column NOT in CSV"
    FAIL=1
fi

# --- Check 3: delta_mean_l2 populated (eval or backfill) ---
echo ""
echo "--- Check 3: delta_mean_l2 populated ---"
LEGACY=$(python -c "
import csv
with open('$EXP_DIR/tables/summary_by_regime.csv') as f:
    r = next(csv.DictReader(f))
    v = r.get('delta_mean_l2', '')
    print(v if v else 'EMPTY')
")
if [ "$LEGACY" != "EMPTY" ]; then
    echo "  ✓ delta_mean_l2 = $LEGACY"
else
    echo "  ✗ delta_mean_l2 is EMPTY"
    FAIL=1
fi

# --- Check 4: selected_experiments in manifest (Commit 3Q) ---
echo ""
echo "--- Check 4: selected_experiments in manifest ---"
if grep -q '"selected_experiments"' "$EXP_DIR/manifest.json"; then
    echo "  ✓ selected_experiments found in manifest"
    N_SEL=$(python -c "
import json
m = json.load(open('$EXP_DIR/manifest.json'))
for r in m['regimes']:
    n = len(r.get('selected_experiments', []))
    if n > 0:
        print(n); break
else:
    print(0)
")
    if [ "$N_SEL" -gt 0 ]; then
        echo "  ✓ At least one regime has $N_SEL selected experiment(s)"
    else
        echo "  ✗ selected_experiments is empty for all regimes"
        FAIL=1
    fi
else
    echo "  ✗ selected_experiments NOT found in manifest"
    FAIL=1
fi

# --- Check 5: n_experiments_selected in CSV (Commit 3Q) ---
echo ""
echo "--- Check 5: n_experiments_selected in CSV ---"
HEADER=$(head -1 "$EXP_DIR/tables/summary_by_regime.csv")
if echo "$HEADER" | grep -q "n_experiments_selected"; then
    N_EXP=$(python -c "
import csv
with open('$EXP_DIR/tables/summary_by_regime.csv') as f:
    r = next(csv.DictReader(f))
    v = r.get('n_experiments_selected', '')
    print(v if v else 'EMPTY')
")
    if [ "$N_EXP" != "EMPTY" ] && [ "$N_EXP" != "0" ]; then
        echo "  ✓ n_experiments_selected = $N_EXP"
    else
        echo "  ✗ n_experiments_selected is EMPTY or 0"
        FAIL=1
    fi
else
    echo "  ✗ n_experiments_selected column NOT in CSV"
    FAIL=1
fi

# --- Check 6: baseline_delta_mean_l2 populated (Commit 3S) ---
echo ""
echo "--- Check 6: baseline_delta_mean_l2 populated ---"
BL_DM=$(python -c "
import csv
with open('$EXP_DIR/tables/summary_by_regime.csv') as f:
    r = next(csv.DictReader(f))
    v = r.get('baseline_delta_mean_l2', '')
    print(v if v else 'EMPTY')
")
if [ "$BL_DM" != "EMPTY" ]; then
    echo "  ✓ baseline_delta_mean_l2 = $BL_DM"
else
    echo "  ✗ baseline_delta_mean_l2 is EMPTY"
    FAIL=1
fi

# --- Check 7: regime subdirectories under studies/<study>/regimes/ (Phase 5) ---
echo ""
echo "--- Check 7: regime dirs nested under studies/<study>/regimes/ ---"
REGIME_INFO=$(python -c "
import json
m = json.load(open('$EXP_DIR/manifest.json'))
for r in m['regimes']:
    print(r['study'] + '/regimes/' + r['regime_id'])
")
NEST_OK=1
while IFS= read -r entry; do
    REGIME_SUBDIR="$EXP_DIR/studies/$entry"
    rid="${entry##*/}"
    if [ -d "$REGIME_SUBDIR/models" ] && [ -d "$REGIME_SUBDIR/logs" ]; then
        echo "  ✓ studies/$entry/ exists with models/ and logs/"
    else
        echo "  ✗ studies/$entry/ missing or incomplete (expected $REGIME_SUBDIR)"
        NEST_OK=0
    fi
done <<< "$REGIME_INFO"
if [ "$NEST_OK" -eq 0 ]; then
    FAIL=1
fi

# --- Cleanup ---
echo ""
echo "--- Cleanup ---"
rm -rf "$EXP_DIR"
echo "  Removed $EXP_DIR"

# --- Result ---
echo ""
if [ $FAIL -eq 0 ]; then
    echo "✅ All smoke checks passed."
else
    echo "❌ Some checks FAILED."
    exit 1
fi
