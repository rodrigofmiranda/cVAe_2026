#!/usr/bin/env python3
"""Teste de validação física da generalização de distância (gray-box OWC).

Pergunta decisiva (antes de construir qualquer gray-box):
  os momentos condicionais do canal — ganho a(d,c) e ruído σ(d,c) — nas distâncias
  NÃO vistas {0.9, 1.16, 1.25} m são PREVISÍVEIS a partir das 4 âncoras de treino
  {0.75, 1.0, 1.35, 1.5} m por uma lei física/suave de POUCOS parâmetros?

  - Se SIM (uma lei de baixa ordem ajusta as 4 e acerta as 3) → o cVAE falha por
    não ter o viés indutivo certo; o gray-box (a,σ paramétricos) interpola por
    construção. TESE CONFIRMADA.
  - Se NÃO (as não-vistas estão fora do trend suave das âncoras) → nenhum modelo
    interpola; o held-out não é prova possível (artefato de medição/geometria).
    Veredito honesto ANTES de gastar GPU.

Convenções herdadas de regime_census.py:
  - a = <X,Y>/<X,X> por eixo (ganho LS; ~1 porque os dados são normalizados).
  - δ = Y - a·X  (resíduo equalizado); σ = std(δ).
  - SNR = mean|X|² / mean|Y-X|².
Puro numpy, CPU, sem modelo/treino. Reusa só os dados medidos (X.npy/Y.npy).
"""
from __future__ import annotations
import argparse, glob, json, os, re
from pathlib import Path
import numpy as np

TRAIN_D = [0.75, 1.0, 1.35, 1.5]
HELD_D = [0.9, 1.16, 1.25]
ALL_D = sorted(TRAIN_D + HELD_D)


def _parse_d(p: str):
    m = re.search(r"dist_([0-9.]+)m", p)
    return float(m.group(1)) if m else float("nan")


def _parse_c(p: str):
    m = re.search(r"curr_([0-9]+)mA", p)
    return int(m.group(1)) if m else -1


def _load_xy(curr_dir: str, n_cap: int):
    """Acha o primeiro IQ_data/X.npy,Y.npy sob curr_dir e carrega (N,2) float64."""
    xs = sorted(glob.glob(os.path.join(curr_dir, "*", "IQ_data", "X.npy")))
    if not xs:
        return None, None
    X = np.load(xs[0])[:n_cap].astype(np.float64)
    Y = np.load(xs[0].replace("X.npy", "Y.npy"))[:n_cap].astype(np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
    return X, Y


def _regime_moments(X, Y):
    """Ganho LS por eixo a, σ do resíduo equalizado, SNR. Retorna escalares
    (a, σ agregados por média entre eixos I/Q)."""
    a_ax, sig_ax = [], []
    for ax in range(X.shape[1]):
        x, y = X[:, ax], Y[:, ax]
        denom = float(np.dot(x, x))
        a = float(np.dot(x, y) / denom) if denom > 0 else float("nan")
        d = y - a * x
        a_ax.append(a)
        sig_ax.append(float(np.std(d)))
    sp = float(np.mean(np.sum(X ** 2, axis=1)))
    npow = float(np.mean(np.sum((Y - X) ** 2, axis=1)))
    snr = sp / npow if npow > 0 else float("inf")
    return float(np.mean(a_ax)), float(np.mean(sig_ax)), snr


# ---------- leis candidatas: ajustam em (d_train -> v_train), preveem em d ------
def _fit_const(d, v):
    c = float(np.mean(v))
    return lambda dd: np.full_like(np.asarray(dd, float), c), f"const={c:.4g}"


def _fit_linear(d, v):
    b1, b0 = np.polyfit(d, v, 1)
    return lambda dd: b0 + b1 * np.asarray(dd, float), f"lin(slope={b1:.4g})"


def _fit_power(d, v):
    """v = K · d^p  (lei de potência; p=-2 reproduz 1/d²). Ajuste em log-log;
    exige v>0."""
    d = np.asarray(d, float); v = np.asarray(v, float)
    if np.any(v <= 0):
        return None, "power(n/a: v<=0)"
    p, logK = np.polyfit(np.log(d), np.log(v), 1)
    K = np.exp(logK)
    return lambda dd: K * np.asarray(dd, float) ** p, f"power(K={K:.4g},p={p:.3f})"


LAWS = {"const": _fit_const, "linear": _fit_linear, "power": _fit_power}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root com dist_*/curr_*")
    ap.add_argument("--model", default="FC")
    ap.add_argument("--n-cap", type=int, default=60000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---- 1) momentos medidos por regime (todas as 7 distâncias) ----
    rows = []
    for dd in sorted(glob.glob(f"{args.root}/dist_*")):
        d = _parse_d(dd)
        for cc in sorted(glob.glob(f"{dd}/curr_*")):
            c = _parse_c(cc)
            X, Y = _load_xy(cc, args.n_cap)
            if X is None:
                print(f"  [skip] sem X/Y: dist={d} curr={c}", flush=True)
                continue
            a, sig, snr = _regime_moments(X, Y)
            rows.append({"dist_m": d, "curr_mA": c, "a_gain": a, "sigma": sig,
                         "snr": snr, "snr_db": 10 * np.log10(snr) if snr > 0 else float("nan"),
                         "n": int(X.shape[0])})
            print(f"  d={d:<5} c={c:<4} a={a:.4f} sigma={sig:.4f} snr_db={10*np.log10(snr):5.2f}", flush=True)

    import csv
    with (out / "regime_moments.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # index: (d,c) -> row
    by = {(r["dist_m"], r["curr_mA"]): r for r in rows}
    currents = sorted({r["curr_mA"] for r in rows})

    # ---- 2) held-out fit/predict por corrente, para cada variável-alvo ----
    report = []
    report.append(f"# Teste de validação física — generalização de distância ({args.model})\n")
    report.append(f"Treino (âncoras): {TRAIN_D} · Held-out (prova): {HELD_D}\n")
    report.append("Para cada corrente, ajusta leis (const/linear/power) nas 4 âncoras e prevê as 3 "
                  "não-vistas. Erro = |pred−medido|/|medido|. Lei vencedora por menor erro mediano.\n")

    # sanidade física
    a_all = np.array([r["a_gain"] for r in rows])
    report.append(f"**Sanidade**: ganho a — média {a_all.mean():.4f}, faixa [{a_all.min():.4f}, {a_all.max():.4f}] "
                  f"(≈1 confirma dados normalizados → física vive em σ/SNR, não no ganho).\n")

    verdict_acc = {v: {law: [] for law in list(LAWS) + ["linear_interp"]} for v in ["a_gain", "sigma", "snr_db"]}

    for var in ["a_gain", "sigma", "snr_db"]:
        report.append(f"\n## Alvo: `{var}`\n")
        report.append("| corrente | medido(held) | melhor lei | pred(held) | err% por lei (const/lin/power) | interp.linear err% |")
        report.append("|---|---|---|---|---|---|")
        for c in currents:
            try:
                dtr = np.array([d for d in TRAIN_D if (d, c) in by])
                vtr = np.array([by[(d, c)][var] for d in dtr])
                dhe = np.array([d for d in HELD_D if (d, c) in by])
                vhe = np.array([by[(d, c)][var] for d in dhe])
            except KeyError:
                continue
            if len(dtr) < 3 or len(dhe) == 0:
                continue
            errs = {}
            preds = {}
            for law, fitter in LAWS.items():
                fn, _desc = fitter(dtr, vtr)
                if fn is None:
                    errs[law] = float("nan"); continue
                pred = fn(dhe)
                preds[law] = pred
                e = np.abs(pred - vhe) / (np.abs(vhe) + 1e-12)
                errs[law] = float(np.median(e))
                verdict_acc[var][law].extend(list(e))
            # baseline: interpolação linear pura (proxy do que um MLP faz)
            li = np.interp(dhe, dtr, vtr)
            eli = np.abs(li - vhe) / (np.abs(vhe) + 1e-12)
            verdict_acc[var]["linear_interp"].extend(list(eli))
            best = min(errs, key=lambda k: (errs[k] if errs[k] == errs[k] else 9e9))
            fnb, _ = LAWS[best](dtr, vtr)
            pb = fnb(dhe)
            errstr = "/".join(f"{errs[k]*100:.1f}" if errs[k] == errs[k] else "—" for k in ["const", "linear", "power"])
            report.append(f"| {c} | {np.array2string(vhe, precision=3)} | {best} | "
                          f"{np.array2string(pb, precision=3)} | {errstr} | {np.median(eli)*100:.1f} |")

    # ---- 3) veredito agregado ----
    report.append("\n## Veredito agregado (erro mediano % nos held-out, todas correntes)\n")
    report.append("| alvo | const | linear | power | interp.linear (proxy MLP) |")
    report.append("|---|---|---|---|---|")
    for var in ["a_gain", "sigma", "snr_db"]:
        cells = []
        for law in ["const", "linear", "power", "linear_interp"]:
            arr = np.array(verdict_acc[var][law]) if verdict_acc[var][law] else np.array([np.nan])
            cells.append(f"{np.nanmedian(arr)*100:.1f}")
        report.append(f"| `{var}` | " + " | ".join(cells) + " |")

    report.append("\n## Leitura\n")
    report.append("- Se a melhor lei física prevê σ/SNR nos held-out com erro pequeno (≪ erro do "
                  "interp.linear) → os momentos SÃO previsíveis por física: **gray-box justificado**.\n"
                  "- Se TODAS as leis erram muito (held-out fora do trend) → as não-vistas não estão "
                  "num manifold suado; nenhum modelo interpola. **Veredito honesto: held-out impossível "
                  "como prova sem mais dados/geometria.**\n")

    (out / "REPORT_physics_test.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nescrito: {out}/REPORT_physics_test.md  +  regime_moments.csv")


if __name__ == "__main__":
    main()
