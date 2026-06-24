#!/usr/bin/env bash
# Detached watcher: ntfy when each of the 2 hybrid follow-up legs finishes.
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
BUCKET=outputs/v3_campaign_reduced_20260611
TOPIC=projeto_vlc_ia

send() { curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1; }

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

declare -A LEGS=( [cvae_v3_s35cg6a_lo9]=s35c_g6a_lo9 [cvae_v3_s35cg6a_s34]=s35c_g6a_s34 )
PENDING="cvae_v3_s35cg6a_lo9 cvae_v3_s35cg6a_s34"

while [ -n "$PENDING" ]; do
  sleep 120
  STILL=""
  for C in $PENDING; do
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$C"; then
      STILL="$STILL $C"
    else
      NAME="${LEGS[$C]}"
      RES=$(leg_result "$REPO/$BUCKET/$NAME")
      send "✅ V3 follow-up — ${NAME} concluído" "${NAME}: ${RES}"
    fi
  done
  PENDING="${STILL# }"
done
send "🏁 V3 follow-up — ambos concluídos" "s35c_g6a_lo9 (clamp LO=-9, teste do mecanismo) e s35c_g6a_s34 (basin re-draw do híbrido) terminaram."
