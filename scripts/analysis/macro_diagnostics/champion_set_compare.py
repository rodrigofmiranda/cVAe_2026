#!/usr/bin/env python3
"""Cross-champion macro comparison: is the per-regime failure pattern consistent
across a SET of near-champion models (not just one)?

Reads each champion's eval `summary_by_regime.csv` (consistent V3 columns:
regime_id, validation_status=pass/fail, gate_g1..g6, stat_screen_pass) and builds:
  - a regime x champion PASS/FAIL matrix (+ how many of N champions pass each regime)
  - per-distance PASS rate per champion + 'all-champions-fail' count
  - consistency buckets: robust_fail (0/N) = robust data-separation targets;
    robust_pass (N/N); model_dependent (mixed)
  - per-gate culprit on the robust-fail regimes (which gate drives consistent fails)

Pure stdlib, CPU, reuses existing eval artifacts (no GPU/training).
"""
from __future__ import annotations
import argparse, csv, os, re, statistics
from datetime import datetime, UTC
from pathlib import Path

OUTPUTS = Path("/home/rodrigo/cVAe_2026_full_square_v3det/outputs")

# All available near-champions w/ eval63 + per-regime gates. name -> summary_by_regime.csv
_PATHS = {
    "base_FC_mdn":    OUTPUTS / "v3fc_crossdist_20260613/exp_20260614_144009/tables/summary_by_regime.csv",
    "base_FS_mdn":    OUTPUTS / "v3fs_crossdist_20260613/exp_20260614_143056/tables/summary_by_regime.csv",
    "E1_light_gauss": OUTPUTS / "v3fc_e1light_eval63_20260624/exp_20260624_032850/tables/summary_by_regime.csv",
    "E1_heavy_gauss": OUTPUTS / "v3fc_e1gauss_20260617/exp_20260618_054004/tables/summary_by_regime.csv",
    "seed7_gauss":    OUTPUTS / "v3fc_e1gauss_seed7_full_20260617/exp_20260618_175253/tables/summary_by_regime.csv",
    "E2_densified":   OUTPUTS / "v3fc_e2gauss_20260620/exp_20260621_064919/tables/summary_by_regime.csv",
}
# Named sets (--set). E2 = NEGATIVE control (regressed 12/63), não é campeão; FS = contraste de geometria.
CHAMPION_SETS = {
    "fc_line": ["base_FC_mdn", "E1_light_gauss", "E1_heavy_gauss", "seed7_gauss"],
    "good5":   ["base_FC_mdn", "base_FS_mdn", "E1_light_gauss", "E1_heavy_gauss", "seed7_gauss"],
    "all":     ["base_FC_mdn", "base_FS_mdn", "E1_light_gauss", "E1_heavy_gauss", "seed7_gauss", "E2_densified"],
}
GATES = ["gate_g1", "gate_g2", "gate_g3", "gate_g4", "gate_g5", "gate_g6", "stat_screen_pass"]


def _b(v) -> bool:
    return str(v).strip().lower() in ("true", "pass", "1", "ok", "yes")


def _dist_curr(regime_id: str):
    md = re.search(r"dist_([0-9p.]+)m", regime_id)
    mc = re.search(r"curr_([0-9]+)mA", regime_id)
    d = float(md.group(1).replace("p", ".")) if md else float("nan")
    c = int(mc.group(1)) if mc else -1
    return d, c


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rid = r.get("regime_id") or r.get("regime") or r.get("regime_label")
            if not rid:
                continue
            out[rid] = {
                "pass": _b(r.get("validation_status")),
                "gates": {g: _b(r.get(g)) for g in GATES},
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="fc_line", choices=list(CHAMPION_SETS))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    CHAMPIONS = {n: _PATHS[n] for n in CHAMPION_SETS[args.set]}
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"/home/rodrigo/comparison_v3/macro_diagnostics/runs/{stamp}_champion_set_{args.set}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {name: _load(p) for name, p in CHAMPIONS.items()}
    names = [n for n in CHAMPIONS if data[n]]
    missing = [n for n in CHAMPIONS if not data[n]]
    # common regimes (intersection)
    regimes = sorted(set.intersection(*[set(data[n]) for n in names]), key=lambda r: _dist_curr(r))
    N = len(names)

    # ---- matrix + consistency ----
    rows = []
    robust_fail, robust_pass, mixed = [], [], []
    for rid in regimes:
        d, c = _dist_curr(rid)
        passes = {n: data[n][rid]["pass"] for n in names}
        npass = sum(passes.values())
        rows.append({"regime_id": rid, "dist_m": d, "curr_mA": c,
                     **{n: ("pass" if passes[n] else "fail") for n in names},
                     "n_pass_across": npass, "n_total": N})
        if npass == 0:
            robust_fail.append(rid)
        elif npass == N:
            robust_pass.append(rid)
        else:
            mixed.append(rid)

    with (out_dir / "champion_set_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- per-distance pass rate ----
    dists = sorted({_dist_curr(r)[0] for r in regimes})
    per_dist = {}
    for d in dists:
        rr = [r for r in regimes if abs(_dist_curr(r)[0] - d) < 1e-9]
        per_dist[d] = {
            "n": len(rr),
            "per_champ": {n: sum(data[n][r]["pass"] for r in rr) for n in names},
            "all_fail": sum(1 for r in rr if all(not data[n][r]["pass"] for n in names)),
            "all_pass": sum(1 for r in rr if all(data[n][r]["pass"] for n in names)),
        }

    # ---- per-gate culprit on robust-fail regimes ----
    gate_fail_frac = {}
    if robust_fail:
        for g in GATES:
            # fraction of (regime, champion) pairs where this gate FAILS, among robust-fail regimes
            tot = len(robust_fail) * N
            nf = sum(1 for rid in robust_fail for n in names if not data[n][rid]["gates"].get(g, True))
            gate_fail_frac[g] = nf / tot if tot else 0.0

    # ---- REPORT ----
    lines = []
    lines.append(f"# Comparação Macro Cross-Champion — set '{args.set}'\n")
    lines.append(f"Gerado: {stamp} UTC · fonte: `summary_by_regime.csv` (coluna `validation_status`, régua V3 twin).\n")
    lines.append(f"Modelos ({N}): " + ", ".join(f"`{n}`" for n in names) + ".")
    if missing:
        lines.append(f"\n⚠️ Ausentes (sem summary_by_regime): {', '.join(missing)}")
    lines.append("\n> **Caveat de consistência**: o `base_FC_mdn` foi avaliado em 2026-06-14 "
                 "(antes da adoção formal da régua V3 em 06-16); os Gaussianos em 06-18/06-24. "
                 "Mesma coluna `validation_status`, mas a régua pode diferir levemente — tratar o "
                 "base_FC como referência, não como medida idêntica.\n")

    lines.append("## Veredito por modelo (n_pass / total)\n")
    for n in names:
        npass = sum(data[n][r]["pass"] for r in regimes)
        lines.append(f"- `{n}`: **{npass}/{len(regimes)}**")

    lines.append("\n## Consistência da falha entre os campeões\n")
    lines.append(f"- **Falha ROBUSTA** (falha em TODOS os {N}): **{len(robust_fail)}** regimes → "
                 "alvos sólidos de separação de dados (não é idiossincrasia de 1 modelo).")
    lines.append(f"- **Passa em TODOS**: {len(robust_pass)} regimes (bem modelados por toda a família).")
    lines.append(f"- **Dependente do modelo** (passa em alguns, falha em outros): {len(mixed)} regimes.")

    lines.append("\n## Taxa de PASS por distância (e falha consensual)\n")
    lines.append("| dist (m) | " + " | ".join(names) + " | all-fail | all-pass |")
    lines.append("|---|" + "|".join(["---"] * N) + "|---|---|")
    for d in dists:
        pd = per_dist[d]
        cells = " | ".join(f"{pd['per_champ'][n]}/{pd['n']}" for n in names)
        lines.append(f"| {d:g} | {cells} | {pd['all_fail']}/{pd['n']} | {pd['all_pass']}/{pd['n']} |")

    if gate_fail_frac:
        lines.append("\n## Qual gate dirige as falhas robustas (entre os regimes que falham em todos)\n")
        lines.append("Fração de pares (regime×modelo) em que cada gate FALHA, nos regimes de falha robusta:\n")
        for g, fr in sorted(gate_fail_frac.items(), key=lambda kv: -kv[1]):
            bar = "█" * int(round(fr * 20))
            lines.append(f"- `{g:18}` {fr*100:5.1f}%  {bar}")

    lines.append("\n## Leitura\n")
    nf_dist = [f"{d:g} m" for d in dists if per_dist[d]["all_fail"] == per_dist[d]["n"] and per_dist[d]["n"] > 0]
    if nf_dist:
        lines.append(f"- Distâncias que falham em **TODOS** os campeões e correntes: **{', '.join(nf_dist)}** "
                     "→ falha estrutural da família (separação de dados / loss, não escolha de modelo).")
    lines.append(f"- {len(mixed)} regimes dependem do modelo → onde a escolha de arquitetura/loss ainda move o placar.")

    (out_dir / "REPORT_champion_set.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"escrito: {out_dir}/REPORT_champion_set.md  (+ champion_set_matrix.csv)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
