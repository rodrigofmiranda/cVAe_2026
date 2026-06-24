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
declare -A LEGS=( [cvae_v3_abl_mmdonly]=s35c_g6a_mmdonly [cvae_v3_abl_energyonly]=s35c_g6a_energyonly )
PENDING="cvae_v3_abl_mmdonly cvae_v3_abl_energyonly"
while [ -n "$PENDING" ]; do
  sleep 180
  STILL=""
  for C in $PENDING; do
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$C"; then
      STILL="$STILL $C"
    else
      NAME="${LEGS[$C]}"
      RES=$(leg_result "$REPO/$BUCKET/$NAME")
      send "✅ V3 ablação — ${NAME} concluído" "${NAME}: ${RES} (referência híbrido: twin 9/12 screen 7/12)"
    fi
  done
  PENDING="${STILL# }"
done
send "🏁 V3 ablação — ambas concluídas" "multibw-only e energy-only terminaram; comparar com híbrido 9/7/7."
