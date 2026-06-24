#!/usr/bin/env bash
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
send() { curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/projeto_vlc_ia" >/dev/null 2>&1; }

leg_result() {
  python3 - "$1" <<'PY'
import csv, glob, sys
paths = sorted(glob.glob(sys.argv[1] + "/exp_*/tables/protocol_leaderboard.csv"))
if not paths:
    print("sem leaderboard — verificar log"); raise SystemExit
r = list(csv.DictReader(open(paths[-1])))[0]
n = r.get("n_regimes", "?")
print(f"twin {r.get('n_pass','?')}/{n} | full {r.get('n_full_pass','?')}/{n} | screen {r.get('stat_screen_pass','?')}/{n}")
PY
}

declare -A DIRS=(
  [cvae_v3_s35cg6a_qsw]="$REPO/outputs/v3_campaign_reduced_20260611/s35c_g6a_qsw"
  [cvae_v3_holdout1m]="$REPO/outputs/v3_holdout_1m_s35cg6a_20260612"
)
declare -A LABELS=(
  [cvae_v3_s35cg6a_qsw]="híbrido+quantile/SW loss (S35CG6A_qsw05)"
  [cvae_v3_holdout1m]="held-out 1.0m (treino 8 regimes + eval 12)"
)
PENDING="cvae_v3_s35cg6a_qsw cvae_v3_holdout1m"

while [ -n "$PENDING" ]; do
  sleep 180
  STILL=""
  for C in $PENDING; do
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$C"; then
      STILL="$STILL $C"
    else
      RES=$(leg_result "${DIRS[$C]}")
      send "✅ V3 — ${LABELS[$C]} concluído" "$RES"
    fi
  done
  PENDING="${STILL# }"
done
send "🏁 V3 — os 2 longos concluídos" "quantile-loss e held-out 1.0m terminaram."
