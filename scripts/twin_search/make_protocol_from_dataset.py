#!/usr/bin/env python3
"""Generate a protocol JSON by auto-discovering regimes from a dataset root.

Onboarding a NEW dataset version (e.g. the 5-LED set) without hand-editing
absolute paths: scan <dataset_root> for experiments (metadata.json + IQ_data),
group by regime (distance, current), and emit a protocol compatible with
`python -m src.protocol.run --protocol <out.json>`.

Usage:
  python3 make_protocol_from_dataset.py <dataset_root> [out.json]

Each experiment is a dir whose IQ_data/ holds X.npy + Y.npy and whose
metadata.json carries distance (distance_m|dist_m) and current (curr_mA|current_mA).
regime_id is taken from metadata when present, else derived (e.g. dist_0p8m__curr_100mA).
"""
import json, os, sys


def fmt_dist(d):
    return ("%g" % float(d)).replace(".", "p")  # 0.8->0p8, 1.0->1, 1.5->1p5


def regime_id(dist, curr):
    return f"dist_{fmt_dist(dist)}m__curr_{int(round(float(curr)))}mA"


def find_experiments(root):
    exps = []
    for dirpath, _dirs, files in os.walk(root):
        if "metadata.json" not in files:
            continue
        iq = os.path.join(dirpath, "IQ_data")
        # metadata may sit at exp dir or inside IQ_data
        exp = os.path.dirname(dirpath) if os.path.basename(dirpath) == "IQ_data" else dirpath
        iq = os.path.join(exp, "IQ_data")
        if not (os.path.exists(os.path.join(iq, "X.npy")) and os.path.exists(os.path.join(iq, "Y.npy"))):
            continue
        try:
            meta = json.load(open(os.path.join(exp, "metadata.json")))
        except Exception:
            meta = json.load(open(os.path.join(dirpath, "metadata.json")))
        dist = meta.get("distance_m", meta.get("dist_m"))
        curr = meta.get("curr_mA", meta.get("current_mA"))
        rid = meta.get("regime_id") or (regime_id(dist, curr) if dist is not None and curr is not None else None)
        if dist is None or curr is None or rid is None:
            continue
        exps.append((exp, float(dist), float(curr), rid))
    return exps


def main(root, out):
    root = os.path.abspath(root)
    exps = find_experiments(root)
    if not exps:
        sys.exit(f"No valid experiments (IQ_data/X.npy+Y.npy + metadata) under {root}")
    # unique regimes
    regimes = {}
    sel = set()
    for exp, dist, curr, rid in exps:
        regimes.setdefault(rid, (dist, curr))
        sel.add(os.path.dirname(exp))  # curr-level dir (parent of the experiment)
    order = sorted(regimes, key=lambda r: regimes[r])
    proto = {
        "protocol_version": "1.0",
        "description": f"Auto-generated from {root} ({len(exps)} experiments, {len(order)} regimes).",
        "global_settings": {"_selected_experiments": sorted(sel)},
        "regimes": [
            {
                "regime_id": r,
                "description": f"{regimes[r][0]:g} m / {int(regimes[r][1])} mA",
                "distance_m": regimes[r][0],
                "current_mA": int(regimes[r][1]),
                "_study": "within_regime",
                "_split_strategy": "per_experiment",
            }
            for r in order
        ],
        "_studies": [
            {"name": "within_regime", "split_strategy": "per_experiment", "regime_ids": order}
        ],
    }
    json.dump(proto, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"Wrote {out}: {len(order)} regimes, {len(exps)} experiments, {len(sel)} selected dirs.")
    print("Regimes:", ", ".join(order))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "configs/protocol_generated.json")
