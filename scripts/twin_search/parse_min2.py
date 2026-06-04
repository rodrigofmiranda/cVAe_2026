import re, sys
WINDOW = 30
vals = []
for line in open(sys.argv[1], errors="ignore"):
    m = re.search(r"val_recon_loss:\s*(-?[0-9.]+e?[+-]?[0-9]*)", line)
    if m:
        try: vals.append(float(m.group(1)))
        except ValueError: pass
if not vals:
    print("0 nan nan"); sys.exit(0)
n = len(vals)
rmin = min(vals)
prev = vals[: max(1, n - WINDOW)]
rmin_prev = min(prev)
print(f"{n} {rmin:.4f} {rmin_prev:.4f}")
