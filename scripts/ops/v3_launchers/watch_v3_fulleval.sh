#!/usr/bin/env bash
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
OUT=outputs/v3_s35cg6a_full_20260612
send() { curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/projeto_vlc_ia" >/dev/null 2>&1; }
while docker ps --format '{{.Names}}' 2>/dev/null | grep -qx cvae_v3_s35cg6a_fulleval; do sleep 300; done
RES=$(python3 - "$REPO/$OUT" <<'PY'
import csv, glob, sys
paths = sorted(glob.glob(sys.argv[1] + "/exp_*/tables/protocol_leaderboard.csv"))
if not paths:
    print("sem leaderboard — eval falhou de novo, verificar run_eval.log"); raise SystemExit
r = list(csv.DictReader(open(paths[-1])))[0]
n = r.get("n_regimes", "?")
print(f"twin {r.get('n_pass','?')}/{n} | full {r.get('n_full_pass','?')}/{n} | screen {r.get('stat_screen_pass','?')}/{n}")
PY
)
send "🏆 V3 FULL-DATA s35c_g6a — eval concluído" "stat full (20k, 2000 perms): ${RES}"
