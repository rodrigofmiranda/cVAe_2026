#!/usr/bin/env python3
"""Fase 0 — isolar a causa do gap de BER do 16QAM: DRIFT entre capturas vs FORMA do ruído.

Decomposição (CPU, numpy; reusa qam_ber/equalize_axis):
  real         = qam_ber(Xq, Yq)                         # canal real
  exact        = qam_ber(Xq, a_q·Xq + δ_q)               # sanidade (≈ real)
  perm_resid   = qam_ber(Xq, a_q·Xq + permuta(δ_q))      # ruído REAL, i.i.d.-izado (sem memória)
  self_gauss   = qam_ber(Xq, a_q·Xq + N(0,Σ_q))          # SEM drift, ruído GAUSSIANO (forma)
  rich_gauss   = qam_ber(Xq, a_r·Xq + N(0,Σ_r))          # = o que o N1 faz (drift + Gauss)

Leitura:
  - self_gauss ≈ real        → forma OK; o gap do N1 era DRIFT (Σ_rich ≠ Σ_qam).
  - self_gauss << real e perm_resid ≈ real → FORMA do ruído (Gauss subconta erros).
  - perm_resid ≠ exact       → há MEMÓRIA/correlação no resíduo.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from scripts.analysis.compute_ber import qam_ber, equalize_axis

DATA = "/home/rodrigo/1-Data/Dataset/V3"
RICH = f"{DATA}/FULL_CIRCLE_2026_V3_ORGANIZED/dist_1.0m/curr_400mA/full_circle_1.0m_400mA_001_20260608_155858/IQ_data"
QAM = {
    "4QAM":  f"{DATA}/4QAM_2026_V3_ORGANIZED/dist_1.0m/curr_400mA/4QAM_1.0m_400mA_001_20260608_155634/IQ_data",
    "16QAM": f"{DATA}/16QAM_2026_V3_ORGANIZED/dist_1.0m/curr_400mA/16QAM_1.0m_400mA_001_20260608_155710/IQ_data",
    "64QAM": f"{DATA}/64QAM_2026_V3_ORGANIZED/dist_1.0m/curr_400mA/64QAM_1.0m_400mA_001_20260608_155746/IQ_data",
}
SEEDS = [7, 21, 33]


def _load(d):
    X = np.load(f"{d}/X.npy").astype(np.float64)
    Y = np.load(f"{d}/Y.npy").astype(np.float64)
    return X, Y


def fit_lin(X, Y):
    """Y ≈ a·X + b por eixo (a = <X,Y>/<X,X>, ganho CORRETO; b offset) + resíduo δ + Σ."""
    a = np.zeros(2); b = np.zeros(2)
    for k in (0, 1):
        a[k], b[k] = np.polyfit(X[:, k], Y[:, k], 1)
    d = Y - (a * X + b)
    return a, b, d, np.cov(d.T)


def ber_of(Xq, Ypred):
    r = qam_ber(Xq, Ypred)
    return float(r.get("ber", r.get("BER", np.nan))) if isinstance(r, dict) else float(r)


def main():
    Xr, Yr = _load(RICH)
    a_r, b_r, d_r, S_r = fit_lin(Xr, Yr)
    eff_r = float(np.mean(np.sqrt(np.diag(S_r)) / a_r))  # ruído efetivo (pós-equalização) do rico
    print(f"RICA: a={a_r.round(4)} sigma_eff(pós-eq)={eff_r:.5f} kurt_delta={_kurt(d_r):.3f}")

    for name, dd in QAM.items():
        Xq, Yq = _load(dd)
        a_q, b_q, d_q, S_q = fit_lin(Xq, Yq)
        eff_q = float(np.mean(np.sqrt(np.diag(S_q)) / a_q))
        base_q = a_q * Xq + b_q          # média do canal na captura QAM (drift-free)
        base_r = a_r * Xq + b_r          # média do canal segundo a captura RICA (= N1)
        real = ber_of(Xq, Yq)
        exact = ber_of(Xq, base_q + d_q)
        perm = np.mean([ber_of(Xq, base_q + d_q[np.random.default_rng(s).permutation(len(d_q))]) for s in SEEDS])
        selfg = np.mean([ber_of(Xq, base_q + np.random.default_rng(s).multivariate_normal([0, 0], S_q, len(Xq))) for s in SEEDS])
        richg = np.mean([ber_of(Xq, base_r + np.random.default_rng(s).multivariate_normal([0, 0], S_r, len(Xq))) for s in SEEDS])
        print(f"\n=== {name} ===")
        print(f"  a_qam={a_q.round(4)}  sigma_eff_qam={eff_q:.5f}  (rica={eff_r:.5f}; razão={eff_q/eff_r:.3f})  kurt_delta={_kurt(d_q):.3f}")
        print(f"  BER real        = {real:.3e}")
        print(f"  BER exact recon = {exact:.3e}   (sanidade ≈ real)")
        print(f"  BER perm_resid  = {perm:.3e}   (ruído REAL i.i.d.)")
        print(f"  BER self_gauss  = {selfg:.3e}   (sem drift, Gauss)")
        print(f"  BER rich_gauss  = {richg:.3e}   (= N1: drift + Gauss)")
        _verdict(name, real, exact, perm, selfg, richg, eff_q, eff_r)


def _kurt(d):
    z = (d - d.mean(0)) / (d.std(0) + 1e-12)
    return float(np.mean(np.mean(z ** 4, 0) - 3.0))


def _verdict(name, real, exact, perm, selfg, richg, eff_q, eff_r):
    if name != "16QAM":
        return
    print("  --- veredito 16QAM ---")
    drift = abs(eff_q - eff_r) / eff_r
    print(f"  drift de SNR efetivo rica→qam: {drift*100:.1f}%")
    if richg < real * 0.6 and selfg >= real * 0.7:
        print("  → DRIFT domina: corrigir o Σ p/ a captura certa já aproxima a BER.")
    elif selfg < real * 0.6 <= (perm / real if real else 1):
        print("  → FORMA do ruído: Gauss subconta erros; o ruído REAL reproduz a BER.")
    else:
        print("  → misto/ambíguo: ver razões acima (drift + forma).")
    if real and abs(perm - exact) / real > 0.3:
        print("  → também há MEMÓRIA/correlação no resíduo (perm ≠ exact).")


if __name__ == "__main__":
    main()
