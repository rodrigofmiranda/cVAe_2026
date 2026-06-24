#!/usr/bin/env bash
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
BUCKET=outputs/v3_campaign_reduced_20260611
send() { curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/projeto_vlc_ia" >/dev/null 2>&1; }
leg_result() {
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
declare -A LEGS=( [cvae_v3_s35cg6a_s35]=s35c_g6a_s35 [cvae_v3_s35cg6a_s36]=s35c_g6a_s36 )
PENDING="cvae_v3_s35cg6a_s35 cvae_v3_s35cg6a_s36"
while [ -n "$PENDING" ]; do
  sleep 180
  STILL=""
  for C in $PENDING; do
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$C"; then
      STILL="$STILL $C"
    else
      NAME="${LEGS[$C]}"
      RES=$(leg_result "$REPO/$BUCKET/$NAME")
      send "✅ V3 seed-sweep — ${NAME} concluído" "${NAME}: ${RES} (campeão seed33: twin 9/12)"
    fi
  done
  PENDING="${STILL# }"
done
send "🏁 V3 seed-sweep — 35 e 36 concluídos" "distribuição-sobre-seeds do campeão híbrido completa (33/34/35/36)."
