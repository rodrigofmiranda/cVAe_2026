import re, sys
vals = []
for line in open(sys.argv[1], errors="ignore"):
    m = re.search(r"val_recon_loss:\s*(-?[0-9.]+e?[+-]?[0-9]*)", line)
    if m:
        try:
            vals.append(float(m.group(1)))
        except ValueError:
            pass
if vals:
    print(f"{len(vals)} {min(vals):.4f}")
else:
    print("0 nan")
