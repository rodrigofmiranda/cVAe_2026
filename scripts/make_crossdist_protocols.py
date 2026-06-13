#!/usr/bin/env python3
"""Gera protocolos train/eval para o estudo de cross-distance generalization.

Treina num subconjunto de distâncias (todas as correntes) e avalia em TODAS as
7 distâncias (as não-treinadas = inferência em distância desconhecida, com
gabarito real). Reutilizável por modulação.

Uso: make_crossdist_protocols.py <MOD_DIR_NAME> <train_dists_csv> [out_prefix]
ex:  make_crossdist_protocols.py FULLSQUARE_2026_V3_ORGANIZED 0.75,1.0,1.5 v3fs
"""
import json, sys, os, glob, re

ROOT_IN_CONTAINER = "/data/Dataset/V3"
HOST_ROOT = "/home/rodrigo/1-Data/Dataset/V3"

def rid(dist, curr):
    ds = ("%g" % dist).replace(".", "p")
    return f"dist_{ds}m__curr_{int(curr)}mA"

def main():
    mod = sys.argv[1]
    train_dists = [float(x) for x in sys.argv[2].split(",")]
    prefix = sys.argv[3] if len(sys.argv) > 3 else "v3cross"
    mod_host = os.path.join(HOST_ROOT, mod)
    # mapear distância(float) -> nome REAL da pasta no disco (ex.: 1.0 -> "1.0",
    # não "1"; "%g" quebraria o casamento de caminho para distâncias inteiras)
    dist_dirname = {}
    for d in glob.glob(mod_host + "/dist_*"):
        s = re.search(r"dist_([0-9.]+)m", os.path.basename(d)).group(1)
        dist_dirname[float(s)] = s
    dists = sorted(dist_dirname)
    currs = sorted({int(re.search(r"curr_([0-9]+)mA", c).group(1))
                    for c in glob.glob(mod_host + "/dist_*/curr_*")})
    test_dists = [d for d in dists if d not in train_dists]

    def build(dist_list, tag, desc):
        sel, regimes, rids = [], [], []
        for d in dist_list:
            dstr = dist_dirname[d]  # nome real da pasta
            for c in currs:
                sel.append(f"{ROOT_IN_CONTAINER}/{mod}/dist_{dstr}m/curr_{c}mA")
                regimes.append({"regime_id": rid(d, c),
                                "description": f"{dstr} m / {c} mA",
                                "distance_m": d, "current_mA": c,
                                "_study": "within_regime",
                                "_split_strategy": "per_experiment"})
                rids.append(rid(d, c))
        return {"protocol_version": "1.0", "description": desc,
                "global_settings": {"_selected_experiments": sel},
                "regimes": regimes,
                "_studies": [{"name": "within_regime",
                              "split_strategy": "per_experiment",
                              "regime_ids": rids}]}

    outdir = "/home/rodrigo/cVAe_2026_full_square_v3det/configs"
    tr = build(train_dists, "train",
               f"{mod} cross-dist TRAIN: dists {train_dists} x {len(currs)} currents (all data).")
    ev = build(dists, "eval_all",
               f"{mod} cross-dist EVAL: ALL {len(dists)} dists x {len(currs)} currents; "
               f"unseen={test_dists} are distance-extrapolation inference vs real ground truth.")
    ftr = f"{outdir}/protocol_{prefix}_train.json"
    fev = f"{outdir}/protocol_{prefix}_evalall.json"
    json.dump(tr, open(ftr, "w"), indent=2, ensure_ascii=False)
    json.dump(ev, open(fev, "w"), indent=2, ensure_ascii=False)
    print(f"modulação: {mod}")
    print(f"  distâncias no disco: {dists}")
    print(f"  correntes no disco:  {currs}")
    print(f"  TREINO ({len(train_dists)} dist x {len(currs)} curr = {len(train_dists)*len(currs)} regimes): {train_dists}")
    print(f"  TESTE/UNSEEN ({len(test_dists)} dist): {test_dists}")
    print(f"  escrito: {ftr}")
    print(f"  escrito: {fev}")

if __name__ == "__main__":
    main()
