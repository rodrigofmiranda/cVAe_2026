#!/usr/bin/env bash
# scripts/smoke_dist_metrics.sh — Smoke-test for distribution-fidelity metrics.
#
# Runs the protocol runner with minimal training (1 epoch, 1 grid, 1 experiment,
# 2000 samples, skip_eval) and verifies that:
#   1) dist_metrics_source appears in the manifest per regime
#   2) cvae_delta_mean_l2 is populated in the summary CSV
#   3) Legacy delta_mean_l2 is backfilled when eval is skipped
#   4) selected_experiments appears in the manifest per regime (Commit 3Q)
#   5) n_experiments_selected is populated in the CSV (Commit 3Q)
#
# Usage:
#   bash scripts/smoke_dist_metrics.sh
#
# Commit 3P + 3Q updates.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Smoke: dist-metrics (Commit 3P) ==="

OUT_BASE="outputs"

python -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base  "$OUT_BASE" \
    --protocol     configs/protocol_default.json \
    --max_epochs 1 --max_grids 1 --max_experiments 1 \
    --max_samples_per_exp 2000 \
    --keras_verbose 0 \
    --dist_tol_m 0.01 --curr_tol_mA 1 \
    --skip_eval

# Find latest protocol dir
PROTO_DIR=$(ls -td "$OUT_BASE"/protocol_2* | head -1)
echo ""
echo "Protocol dir: $PROTO_DIR"

FAIL=0

# --- Check 1: dist_metrics_source in manifest ---
echo ""
echo "--- Check 1: dist_metrics_source in manifest ---"
if grep -q '"dist_metrics_source"' "$PROTO_DIR/manifest.json"; then
    echo "  ✓ dist_metrics_source found in manifest"
    grep '"dist_metrics_source"' "$PROTO_DIR/manifest.json" | head -3
else
    echo "  ✗ dist_metrics_source NOT found in manifest"
    FAIL=1
fi

# --- Check 2: cvae_delta_mean_l2 in CSV ---
echo ""
echo "--- Check 2: cvae_delta_mean_l2 in CSV ---"
HEADER=$(head -1 "$PROTO_DIR/tables/summary_by_regime.csv")
if echo "$HEADER" | grep -q "cvae_delta_mean_l2"; then
    echo "  ✓ cvae_delta_mean_l2 column present"
    # Verify it's populated (not empty) for first row
    VAL=$(python -c "
import csv
with open('$PROTO_DIR/tables/summary_by_regime.csv') as f:
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

# --- Check 3: legacy backfill (delta_mean_l2) ---
echo ""
echo "--- Check 3: legacy delta_mean_l2 backfill (eval skipped) ---"
LEGACY=$(python -c "
import csv
with open('$PROTO_DIR/tables/summary_by_regime.csv') as f:
    r = next(csv.DictReader(f))
    v = r.get('delta_mean_l2', '')
    print(v if v else 'EMPTY')
")
if [ "$LEGACY" != "EMPTY" ]; then
    echo "  ✓ delta_mean_l2 backfilled: $LEGACY"
else
    echo "  ✗ delta_mean_l2 NOT backfilled"
    FAIL=1
fi

# --- Check 4: selected_experiments in manifest (Commit 3Q) ---
echo ""
echo "--- Check 4: selected_experiments in manifest ---"
if grep -q '"selected_experiments"' "$PROTO_DIR/manifest.json"; then
    echo "  ✓ selected_experiments found in manifest"
    # Verify it's a non-empty list for at least one regime
    N_SEL=$(python -c "
import json
m = json.load(open('$PROTO_DIR/manifest.json'))
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
HEADER=$(head -1 "$PROTO_DIR/tables/summary_by_regime.csv")
if echo "$HEADER" | grep -q "n_experiments_selected"; then
    N_EXP=$(python -c "
import csv
with open('$PROTO_DIR/tables/summary_by_regime.csv') as f:
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

# --- Cleanup ---
echo ""
echo "--- Cleanup ---"
rm -rf "$PROTO_DIR"
echo "  Removed $PROTO_DIR"

# --- Result ---
echo ""
if [ $FAIL -eq 0 ]; then
    echo "✅ All smoke checks passed."
else
    echo "❌ Some checks FAILED."
    exit 1
fi
