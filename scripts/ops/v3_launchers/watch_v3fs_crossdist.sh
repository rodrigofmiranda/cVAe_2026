#!/usr/bin/env bash
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
OUT=outputs/v3fs_crossdist_20260613
send() { curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/projeto_vlc_ia" >/dev/null 2>&1; }
while docker ps --format '{{.Names}}' 2>/dev/null | grep -qx cvae_v3fs_crossdist; do sleep 300; done
# container saiu — reportar estado final
EVAL=$(ls -d $REPO/$OUT/exp_*/  2>/dev/null | tail -1)
if ls $REPO/$OUT/exp_*/tables/protocol_leaderboard.csv >/dev/null 2>&1; then
  RES=$(python3 - "$REPO/$OUT" <<'PY'
import csv, glob, sys
paths=sorted(glob.glob(sys.argv[1]+"/exp_*/tables/protocol_leaderboard.csv"))
r=list(csv.DictReader(open(paths[-1])))[0]; n=r.get("n_regimes","?")
print(f"twin {r.get('n_pass','?')}/{n} | full {r.get('n_full_pass','?')}/{n} | screen {r.get('stat_screen_pass','?')}/{n}")
PY
)
  send "🏁 cross-dist FS — container encerrado" "leaderboard: ${RES}. Unseen=0.9/1.16/1.25m a comparar."
else
  send "⚠️ cross-dist FS — container saiu SEM leaderboard" "verificar run_train.log / run_eval63.log (possível erro)."
fi
