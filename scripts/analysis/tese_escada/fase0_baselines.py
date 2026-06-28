#!/usr/bin/env python3
"""Fase 0 da escada da tese — baselines analíticos N0/N1/N2 em UM regime.

Executa o protocolo congelado em docs/FASE0_PRE_REGISTRO.md (TESE-F0-FC-1p0m-400mA-v1):
  N0  Y = X + ε              (AWGN ganho unitário; Σ0 = cov(Y-X) no treino)
  N1  Y = a·X + b + ε        (escalar a compartilhado, offset I/Q, Σ1 cheia)
  N2  Y = A·X + b + ε        (matriz A 2x2, offset I/Q, Σ2 cheia)
N3/N4 (regressor Gaussiano / cVAE) só se N2 falhar — fora deste script.

Reusa as definições OFICIAIS da régua-V3 (não reescreve):
  src.evaluation.metrics.{calculate_evm, calculate_snr, residual_distribution_metrics}
  src.evaluation.validation_summary.TWIN_GATE_THRESHOLDS
  src.evaluation.stat_tests.mmd.mmd_rbf
  scripts.analysis.compute_ber.qam_ber

Tolerância = block-bootstrap REAL-vs-REAL (piso de variabilidade intra-captura): para
cada métrica, o IC95 de M(realA, realB) sob reamostragem em blocos. Um modelo "≈ real"
se sua métrica ficar <= P97.5 do piso. CPU/numpy. Saídas em
comparison_v3/tese_escada/fase0/<run_id>/.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import argparse, csv, hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from src.evaluation.metrics import (
    calculate_evm, calculate_snr, residual_distribution_metrics,
)
from src.evaluation.validation_summary import TWIN_GATE_THRESHOLDS
try:
    from src.evaluation.stat_tests.mmd import mmd_rbf
except Exception:
    mmd_rbf = None
from scripts.analysis.compute_ber import qam_ber

# ----------------------- CONFIG CONGELADA (pré-registro) -----------------------
RUN_ID = "TESE-F0-FC-1p0m-400mA-v1"
DATA = "/home/rodrigo/1-Data/Dataset/V3"
REGIME = "dist_1.0m/curr_400mA/full_circle_1.0m_400mA_001_20260608_155858/IQ_data"
FC_DIR = f"{DATA}/FULL_CIRCLE_2026_V3_ORGANIZED/{REGIME}"
QAM_DIRS = {
    "4QAM":  f"{DATA}/4QAM_2026_V3_ORGANIZED/dist_1.0m/curr_400mA/4QAM_1.0m_400mA_001_20260608_155634/IQ_data",
    "16QAM": f"{DATA}/16QAM_2026_V3_ORGANIZED/dist_1.0m/curr_400mA/16QAM_1.0m_400mA_001_20260608_155710/IQ_data",
    "64QAM": f"{DATA}/64QAM_2026_V3_ORGANIZED/dist_1.0m/curr_400mA/64QAM_1.0m_400mA_001_20260608_155746/IQ_data",
}
GAP = 4096
FRAC = (0.70, 0.15, 0.15)
SEEDS = [7, 21, 33, 42, 79]
COV_DRAWS = 50            # nº de amostras MC p/ coverage
MMD_SUB = 5000           # subamostra p/ MMD
OUT_BASE = Path("/home/rodrigo/comparison_v3/tese_escada/fase0")
GATE_KEYS = {            # métrica -> threshold da régua-V3
    "rel_evm_error": TWIN_GATE_THRESHOLDS["rel_evm_error"],
    "rel_snr_error": TWIN_GATE_THRESHOLDS["rel_snr_error"],
    "mean_rel_sigma": TWIN_GATE_THRESHOLDS["mean_rel_sigma"],
    "delta_psd_l2": TWIN_GATE_THRESHOLDS["delta_psd_l2"],
    "delta_skew_l2": TWIN_GATE_THRESHOLDS["delta_skew_l2"],
}


def _load(d):
    X = np.load(f"{d}/X.npy").astype(np.float64)
    Y = np.load(f"{d}/Y.npy").astype(np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1); Y = Y.reshape(-1, 1)
    return X, Y


def _sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n))   # hash do início (rápido; identifica o arquivo)
    return h.hexdigest()[:16]


def split_blocks(N):
    M = N - 2 * GAP
    n_tr = int(M * FRAC[0]); n_va = int(M * FRAC[1]); n_te = M - n_tr - n_va
    tr = (0, n_tr)
    va = (n_tr + GAP, n_tr + GAP + n_va)
    te = (n_tr + GAP + n_va + GAP, n_tr + GAP + n_va + GAP + n_te)
    return {"train": tr, "val": va, "test": te}


# ----------------------------- fits N0 / N1 / N2 ------------------------------
def fit_N0(X, Y):
    d = Y - X
    return {"name": "N0", "A": np.eye(2), "b": np.zeros(2), "Sigma": np.cov(d.T)}


def fit_N1(X, Y):
    N = len(X)
    # Y_ax = a·X_ax + b_ax, mesmo a; design 2N x 3 = [a, bI, bQ]
    D = np.zeros((2 * N, 3)); t = np.zeros(2 * N)
    D[:N, 0] = X[:, 0]; D[:N, 1] = 1.0; t[:N] = Y[:, 0]
    D[N:, 0] = X[:, 1]; D[N:, 2] = 1.0; t[N:] = Y[:, 1]
    sol, *_ = np.linalg.lstsq(D, t, rcond=None)
    a, bI, bQ = sol
    A = np.array([[a, 0.0], [0.0, a]]); b = np.array([bI, bQ])
    res = Y - (X @ A.T + b)
    return {"name": "N1", "A": A, "b": b, "Sigma": np.cov(res.T), "a_scalar": float(a)}


def fit_N2(X, Y):
    Xa = np.hstack([X, np.ones((len(X), 1))])         # N x 3
    W, *_ = np.linalg.lstsq(Xa, Y, rcond=None)        # 3 x 2
    A = W[:2].T; b = W[2]
    res = Y - (X @ A.T + b)
    return {"name": "N2", "A": A, "b": b, "Sigma": np.cov(res.T)}


def predict_mean(m, X):
    return X @ m["A"].T + m["b"]


def sample(m, X, rng):
    return predict_mean(m, X) + rng.multivariate_normal(np.zeros(2), m["Sigma"], size=len(X))


def gauss_nll(Y, mean, Sigma):
    d = Y - mean
    Si = np.linalg.inv(Sigma)
    sign, logdet = np.linalg.slogdet(Sigma)
    maha = np.einsum("ni,ij,nj->n", d, Si, d)
    return float(np.mean(0.5 * (2 * np.log(2 * np.pi) + logdet + maha)))


# ------------------------------- métricas régua ------------------------------
def gate_metrics(X, Yreal, Ypred, Ysamples=None):
    """Calcula as métricas da régua-V3 reusando residual_distribution_metrics +
    calculate_evm/snr. Ypred = 1 realização; Ysamples = (K,N,2) p/ coverage."""
    rdm = residual_distribution_metrics(
        X, Yreal, Ypred, Y_samples=Ysamples, coverage_target=Yreal,
    )
    evm_real = abs(calculate_evm(X, Yreal)[0]); evm_pred = abs(calculate_evm(X, Ypred)[0])
    snr_real = abs(calculate_snr(X, Yreal)); snr_pred = abs(calculate_snr(X, Ypred))
    sigma_real = float(np.sqrt(rdm["var_real_delta"])) if rdm["var_real_delta"] > 0 else float("nan")
    m = {
        "rel_evm_error": abs(evm_pred - evm_real) / evm_real if evm_real > 0 else np.nan,
        "rel_snr_error": abs(snr_pred - snr_real) / snr_real if snr_real > 0 else np.nan,
        "mean_rel_sigma": rdm["delta_mean_l2"] / sigma_real if sigma_real > 0 else np.nan,
        "delta_psd_l2": rdm["delta_psd_l2"],
        "delta_skew_l2": rdm["delta_skew_l2"],
        "delta_kurt_l2": rdm["delta_kurt_l2"],
        "var_ratio": rdm["var_pred_delta"] / rdm["var_real_delta"] if rdm["var_real_delta"] > 0 else np.nan,
        "coverage_95": rdm.get("coverage_95", np.nan),
        "coverage_80": rdm.get("coverage_80", np.nan),
        "coverage_50": rdm.get("coverage_50", np.nan),
    }
    return m


def block_resample(N, block, rng):
    idx = []
    while len(idx) < N:
        s = int(rng.integers(0, max(1, N - block)))
        idx.extend(range(s, s + block))
    return np.array(idx[:N])


def real_self_floor(X, Y, B, block, seed):
    """Piso real-vs-real: M(realA, realB) sob reamostragem em blocos -> P97.5 por métrica."""
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in GATE_KEYS}
    for _ in range(B):
        ia = block_resample(len(X), block, rng)
        ib = block_resample(len(X), block, rng)
        # realB faz o papel de "predito"; compara distribuições reais independentes
        m = gate_metrics(X[ia], Y[ia], Y[ib])
        for k in GATE_KEYS:
            acc[k].append(m[k])
    return {k: float(np.nanpercentile(acc[k], 97.5)) for k in GATE_KEYS}


# ----------------------------------- BER -------------------------------------
def ber_table(models, seeds):
    rows = []
    for qam, d in QAM_DIRS.items():
        Xq, Yq = _load(d)
        br = qam_ber(Xq, Yq)              # BER real
        ber_real = br.get("ber", br.get("BER", np.nan)) if isinstance(br, dict) else float(br)
        for m in models:
            for s in seeds:
                rng = np.random.default_rng(s)
                Yp = sample(m, Xq, rng)
                bp = qam_ber(Xq, Yp)
                ber_pred = bp.get("ber", bp.get("BER", np.nan)) if isinstance(bp, dict) else float(bp)
                rows.append({"qam": qam, "model": m["name"], "seed": s,
                             "ber_real": ber_real, "ber_pred": ber_pred,
                             "abs_err": abs(ber_pred - ber_real)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--boot-block", type=int, default=2048)
    ap.add_argument("--cap", type=int, default=0, help="0 = usar tudo")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = OUT_BASE / f"{stamp}__{RUN_ID}"
    out.mkdir(parents=True, exist_ok=True)

    X, Y = _load(FC_DIR)
    if args.cap:
        X, Y = X[:args.cap], Y[:args.cap]
    N = len(X)
    sp = split_blocks(N)
    (tr0, tr1) = sp["train"]; (te0, te1) = sp["test"]
    Xtr, Ytr = X[tr0:tr1], Y[tr0:tr1]
    Xte, Yte = X[te0:te1], Y[te0:te1]
    print(f"N={N} | train={tr1-tr0} test={te1-te0} | split={sp}", flush=True)

    models = [fit_N0(Xtr, Ytr), fit_N1(Xtr, Ytr), fit_N2(Xtr, Ytr)]

    # ---- piso real-vs-real (tolerância) no bloco de teste ----
    print(f"bootstrap real-vs-real (B={args.boot}, block={args.boot_block})...", flush=True)
    floor = real_self_floor(Xte, Yte, args.boot, args.boot_block, seed=12345)
    print("piso (P97.5) por métrica:", {k: round(v, 4) for k, v in floor.items()}, flush=True)

    # ---- métricas por modelo × seed no teste ----
    rows_seed = []
    Ksamp = COV_DRAWS
    for m in models:
        nll = gauss_nll(Yte, predict_mean(m, Xte), m["Sigma"])
        for s in SEEDS:
            rng = np.random.default_rng(s)
            Yp = sample(m, Xte, rng)
            Ys = np.stack([sample(m, Xte, np.random.default_rng(1000 + s * 10 + j)) for j in range(Ksamp)], axis=0)
            gm = gate_metrics(Xte, Yte, Yp, Ysamples=Ys)
            # MMD (resíduo real vs pred, subamostra)
            mmd_val = np.nan
            if mmd_rbf is not None:
                try:
                    dr = (Yte - Xte)[:MMD_SUB]; dp = (Yp - Xte)[:MMD_SUB]
                    r = mmd_rbf(dr, dp)
                    mmd_val = float(r[0] if isinstance(r, (tuple, list)) else (r.get("mmd2", np.nan) if isinstance(r, dict) else r))
                except Exception as e:
                    mmd_val = np.nan
            row = {"model": m["name"], "seed": s, "nll": nll, "mmd2": mmd_val, **gm}
            # gates pass/fail vs régua E vs piso
            for k, thr in GATE_KEYS.items():
                row[f"{k}__pass_regua"] = bool(gm[k] < thr) if gm[k] == gm[k] else False
                row[f"{k}__pass_piso"] = bool(gm[k] <= floor[k]) if gm[k] == gm[k] else False
            rows_seed.append(row)

    # ---- BER ----
    print("BER 4/16/64-QAM...", flush=True)
    ber_rows = ber_table(models, SEEDS)

    # ---- saídas ----
    with (out / "metrics_by_model_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_seed[0].keys())); w.writeheader(); w.writerows(rows_seed)
    with (out / "qam_ber_by_model_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ber_rows[0].keys())); w.writeheader(); w.writerows(ber_rows)
    params = {}
    for m in models:
        p = {"A": m["A"].tolist(), "b": m["b"].tolist(), "Sigma": m["Sigma"].tolist()}
        if "a_scalar" in m:
            p["a_scalar"] = m["a_scalar"]
        params[m["name"]] = p
    (out / "model_parameters.json").write_text(json.dumps(params, indent=2))
    (out / "split.json").write_text(json.dumps({"N": N, **{k: list(v) for k, v in sp.items()}, "gap": GAP, "frac": FRAC}, indent=2))
    try:
        commit = subprocess.check_output(["git", "-C", "/home/rodrigo/cVAe_2026_full_square_v3det", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        commit = "unknown"
    (out / "manifest.json").write_text(json.dumps({
        "run_id": RUN_ID, "stamp": stamp, "commit": commit,
        "fc_dir": FC_DIR, "qam_dirs": QAM_DIRS,
        "X_sha16": _sha(f"{FC_DIR}/X.npy"), "Y_sha16": _sha(f"{FC_DIR}/Y.npy"),
        "boot": args.boot, "boot_block": args.boot_block, "seeds": SEEDS,
        "thresholds": GATE_KEYS, "floor_p975": floor,
    }, indent=2))

    # ---- agregação + decisão ----
    def agg(model, key):
        vals = [r[key] for r in rows_seed if r["model"] == model and r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    report = [f"# Fase 0 — baselines analíticos ({RUN_ID})\n",
              f"Gerado {stamp} UTC · commit `{commit[:10]}` · N={N} (test={te1-te0}).",
              f"Piso real-vs-real (P97.5, block-bootstrap B={args.boot}/{args.boot_block}): " +
              ", ".join(f"`{k}`={floor[k]:.4f}" for k in GATE_KEYS) + "\n",
              "## NLL primária (nat/amostra, teste) + métricas-chave (média sobre seeds)\n",
              "| modelo | NLL | rel_evm | rel_snr | mean_rel_sigma | psd_l2 | skew_l2 | var_ratio | cov95 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        nm = m["name"]
        report.append(f"| {nm} | {agg(nm,'nll'):.4f} | {agg(nm,'rel_evm_error'):.4f} | "
                      f"{agg(nm,'rel_snr_error'):.4f} | {agg(nm,'mean_rel_sigma'):.4f} | "
                      f"{agg(nm,'delta_psd_l2'):.4f} | {agg(nm,'delta_skew_l2'):.4f} | "
                      f"{agg(nm,'var_ratio'):.3f} | {agg(nm,'coverage_95'):.3f} |")
    report.append("\n## Gates régua-V3 (pass = abaixo do threshold) — média seeds\n")
    report.append("| modelo | " + " | ".join(GATE_KEYS) + " |")
    report.append("|---|" + "|".join(["---"] * len(GATE_KEYS)) + "|")
    for m in models:
        nm = m["name"]
        cells = []
        for k in GATE_KEYS:
            frac = np.mean([1.0 if r[f"{k}__pass_regua"] else 0.0 for r in rows_seed if r["model"] == nm])
            cells.append(f"{frac:.0%}")
        report.append(f"| {nm} | " + " | ".join(cells) + " |")
    report.append("\n## BER (média seeds) por QAM\n| qam | " + " | ".join(m["name"] for m in models) + " | real |")
    report.append("|---|" + "|".join(["---"] * (len(models) + 1)) + "|")
    for qam in QAM_DIRS:
        real = np.mean([r["ber_real"] for r in ber_rows if r["qam"] == qam])
        cells = []
        for m in models:
            bp = np.mean([r["ber_pred"] for r in ber_rows if r["qam"] == qam and r["model"] == m["name"]])
            cells.append(f"{bp:.2e}")
        report.append(f"| {qam} | " + " | ".join(cells) + f" | {real:.2e} |")
    report.append("\n## N1 vs N2 (regra 7.1)\n")
    n2 = next(m for m in models if m["name"] == "N2")
    A = n2["A"]; offdiag = np.hypot(A[0, 1], A[1, 0]); diag = np.hypot(A[0, 0], A[1, 1])
    iq_gain_diff = abs(A[0, 0] - A[1, 1]) / max(abs(A[0, 0]), abs(A[1, 1]))
    nll_n1 = agg("N1", "nll"); nll_n2 = agg("N2", "nll")
    report.append(f"- offdiag/diag de A(N2) = {offdiag/diag:.4f} (limiar 0.02)")
    report.append(f"- diff relativa ganho I/Q = {iq_gain_diff:.4f} (limiar 0.02)")
    report.append(f"- N2 reduz NLL vs N1? {(nll_n1-nll_n2)/abs(nll_n1)*100:.2f}% (limiar 1%)")
    report.append(f"- N1.a_scalar = {next(m for m in models if m['name']=='N1')['a_scalar']:.4f} (medido físico ≈0.62)")
    (out / "REPORT_FASE0.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nescrito: {out}/REPORT_FASE0.md")


if __name__ == "__main__":
    main()
