#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Macro post-modeling diagnostic aggregator (pure stdlib, no numpy).

Consolidates every comparison_v3 diagnostic into ONE verdict per (model, regime)
and turns the per-regime failure map into a concrete data-separation
recommendation for the next cVAE training.

Layers consumed (all optional — the report degrades gracefully if absent):
  0  regime_census.csv                      real-channel delta stats (model-free)
  1  cross_correlation/xcorr_table_*.csv     linear-response interpolation error
  2  awgn/<m>/metrics_table.csv              distributional fidelity (cvae vs real)
  2  awgn/shot_noise_coefficients_*.csv      heteroscedastic slope (real/cvae/awgn)
  4  fs_vs_fc_crossdist/gates_summary_*.csv  per-gate PASS/FAIL map (G1..G5+stat)
  5  modulations*/.../ber_table_*.csv        application BER fidelity
  6  awgn/<m>/metrics_table.csv (awgn rows)  baseline contrast

Join key is (model, dist_m: float, curr_mA: int) — NOT the regime string, because
the gate table writes `dist_1m` while the metrics table writes `dist_1p0m`.

Writes: summary_master.csv, REPORT.md, manifest.json under <out_dir>.

Pure stdlib so the default --fast path runs without docker/numpy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COMPARISON_V3 = Path("/home/rodrigo/comparison_v3")
MODELS = ("FC", "FS")
GATE_COLS = ("G1", "G2", "G3", "G4", "G5", "stat")
TRAINED_DISTS = (0.75, 1.0, 1.35, 1.5)

# Data-source subdirs relative to ``root``. Default = legacy scattered folders;
# ``--in-dir`` swaps these to the self-contained per-run layout (N_*/ subfolders),
# so a single run folder holds AWGN + cross-dist + modulations + everything.
SRC = {
    "gates": "fs_vs_fc_crossdist",
    "awgn": "awgn",
    "xcorr": "cross_correlation",
    "mod": ("modulations", "modulations_unseen"),
    "census": ".",
}
SRC_RUN = {
    "gates": "3_gates",
    "awgn": "2_awgn",
    "xcorr": "1_xcorr",
    # modulações são 100% inferência (o treino é só FC/FS no canal) — sem split trained/unseen
    "mod": ("5_modulations",),
    "census": "0_census",
}
# Human titles for the self-contained figure index.
FIG_SECTIONS = (
    ("0_census", "Camada 0 — Censo / ruído (LED, cauda, decomposição)"),
    ("1_xcorr", "Camada 1 — Resposta linear (xcorr)"),
    ("2_awgn", "Camada 2 — AWGN vs cVAE (heterocedasticidade, SNR, radar)"),
    ("3_gates", "Gates (trained)"),
    ("4_crossdist", "Inferência em outras distâncias (gates unseen/crossdist)"),
    ("5_modulations", "Modulações 4/16/64-QAM (BER — inferência, todas as distâncias)"),
)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _f(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _dist_from_label(label: str) -> float:
    """'0.75 m' -> 0.75 ; '1 m' -> 1.0 ; '1.16 m' -> 1.16."""
    return _f(str(label).replace("m", "").strip())


def _key(dist_m: float, curr_mA: float) -> tuple[float, int]:
    return (round(float(dist_m), 4), int(round(float(curr_mA))))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _minmax_norm(values: dict[Any, float]) -> dict[Any, float]:
    """Normalise finite values to [0, 1]; non-finite -> 0.0."""
    finite = [v for v in values.values() if math.isfinite(v)]
    if not finite:
        return {k: 0.0 for k in values}
    lo, hi = min(finite), max(finite)
    span = hi - lo
    out: dict[Any, float] = {}
    for k, v in values.items():
        if not math.isfinite(v) or span <= 0:
            out[k] = 0.0
        else:
            out[k] = (v - lo) / span
    return out


# --------------------------------------------------------------------------
# layer parsers -> indexed by (model, dist, curr)
# --------------------------------------------------------------------------
def parse_gates(root: Path) -> dict[tuple, dict[str, Any]]:
    """Wide gate table -> melt to (model, dist, curr)."""
    rows = _read_csv(root / SRC["gates"] / "gates_summary_FS_vs_FC_crossdist.csv")
    out: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        dist = _dist_from_label(r.get("distance", ""))
        curr = _f(r.get("current_mA"))
        if not math.isfinite(dist) or not math.isfinite(curr):
            continue
        unseen = str(r.get("unseen", "")).strip().lower() in {"yes", "1", "true", "sim"}
        for model in MODELS:
            gates = {g: str(r.get(f"{model}_{g}", "")).strip().upper() for g in GATE_COLS}
            n_fail = sum(1 for g in GATE_COLS if gates[g] == "FAIL")
            failed = [g for g in GATE_COLS if gates[g] == "FAIL"]
            out[(model, *_key(dist, curr))] = {
                "model": model, "dist_m": round(dist, 4), "curr_mA": int(round(curr)),
                "unseen": unseen, "status": str(r.get(f"{model}_status", "")).strip().upper(),
                "n_fail": n_fail, "failed_gates": "|".join(failed), **{f"gate_{g}": gates[g] for g in GATE_COLS},
            }
    return out


def parse_metrics(root: Path) -> dict[tuple, dict[str, float]]:
    """awgn/<m>/metrics_table.csv -> distributional fidelity (cvae) + awgn contrast."""
    out: dict[tuple, dict[str, float]] = {}
    for model, sub in (("FC", "fc"), ("FS", "fs")):
        rows = _read_csv(root / SRC["awgn"] / sub / "metrics_table.csv")
        by_key: dict[tuple, dict[str, dict]] = {}
        for r in rows:
            k = (model, *_key(_f(r.get("dist_m")), _f(r.get("curr_mA"))))
            by_key.setdefault(k, {})[str(r.get("method", ""))] = r
        for k, methods in by_key.items():
            cv = methods.get("cvae", {})
            aw = methods.get("awgn", {})
            out[k] = {
                "real_channel_snr_db": _f(cv.get("real_channel_snr_db")),
                "cvae_delta_channel_snr_db": _f(cv.get("delta_channel_snr_db")),
                "awgn_delta_channel_snr_db": _f(aw.get("delta_channel_snr_db")),
                "cvae_kurt_l2": _f(cv.get("delta_residual_kurt_l2")),
                "cvae_skew_l2": _f(cv.get("delta_residual_skew_l2")),
                "cvae_var_l2": _f(cv.get("delta_residual_var_l2")),
                "cvae_w1_mag": _f(cv.get("wasserstein1_mag")),
                "awgn_kurt_l2": _f(aw.get("delta_residual_kurt_l2")),
                "awgn_w1_mag": _f(aw.get("wasserstein1_mag")),
                "cvae_snr_fidelity_db": _f(cv.get("snr_db")),
                "awgn_snr_fidelity_db": _f(aw.get("snr_db")),
            }
    return out


def parse_shot_noise(root: Path) -> dict[tuple, dict[str, float]]:
    """shot_noise_coefficients_*.csv -> het slope a for real vs cvae."""
    out: dict[tuple, dict[str, float]] = {}
    for model, suffix in (("FC", "FC"), ("FS", "FS")):
        rows = _read_csv(root / SRC["awgn"] / f"shot_noise_coefficients_{suffix}.csv")
        by_key: dict[tuple, dict[str, float]] = {}
        for r in rows:
            k = (model, *_key(_f(r.get("dist_m")), _f(r.get("curr_mA"))))
            by_key.setdefault(k, {})[str(r.get("method", ""))] = _f(r.get("a"))
        for k, methods in by_key.items():
            a_real = methods.get("real", float("nan"))
            a_cvae = methods.get("cvae", float("nan"))
            mism = abs(a_cvae - a_real) / max(abs(a_real), 1e-9) if math.isfinite(a_real) and math.isfinite(a_cvae) else float("nan")
            out[k] = {"het_a_real": a_real, "het_a_cvae": a_cvae, "het_mismatch": mism}
    return out


def parse_ber(root: Path) -> dict[tuple, dict[str, float]]:
    """All ber_table_*.csv (trained + unseen) -> mean |BER err| + 64QAM err per regime."""
    out: dict[tuple, list[tuple[str, float]]] = {}
    for folder in SRC["mod"]:
        for path in (root / folder).glob("*/ber_table_*.csv"):
            for r in _read_csv(path):
                model = str(r.get("label", "")).strip().upper()
                if model not in MODELS:
                    continue
                k = (model, *_key(_f(r.get("dist_m")), _f(r.get("curr_mA"))))
                out.setdefault(k, []).append((str(r.get("modulation", "")), _f(r.get("ber_abs_err"))))
    agg: dict[tuple, dict[str, float]] = {}
    for k, items in out.items():
        errs = [e for _, e in items if math.isfinite(e)]
        err64 = [e for m, e in items if "64" in m and math.isfinite(e)]
        agg[k] = {
            "ber_abs_err_mean": (sum(errs) / len(errs)) if errs else float("nan"),
            "ber_abs_err_64qam": (sum(err64) / len(err64)) if err64 else float("nan"),
        }
    return agg


def parse_xcorr(root: Path) -> dict[tuple, float]:
    """xcorr_table_*.csv -> linear-response L2 per (model, dist) (curr 500 only)."""
    out: dict[tuple, float] = {}
    for model in MODELS:
        for r in _read_csv(root / SRC["xcorr"] / f"xcorr_table_{model}.csv"):
            dist = _f(r.get("dist_m"))
            l2 = max(_f(r.get("xcorr_l2_I"), 0.0), _f(r.get("xcorr_l2_Q"), 0.0))
            out[(model, round(dist, 4))] = l2
    return out


def parse_census(out_dir: Path, root: Path) -> dict[tuple, dict[str, float]]:
    """regime_census.csv (model-free real stats) if it was precomputed."""
    for cand in (out_dir / "regime_census.csv", out_dir / SRC["census"] / "regime_census.csv",
                 root / "macro_diagnostics" / "regime_census.csv"):
        rows = _read_csv(cand)
        if rows:
            out: dict[tuple, dict[str, float]] = {}
            for r in rows:
                model = str(r.get("model", "")).strip().upper()
                k = (model, *_key(_f(r.get("dist_m")), _f(r.get("curr_mA"))))
                out[k] = {c: _f(r.get(c)) for c in r if c not in ("model", "dist_m", "curr_mA")}
            return out
    return {}


# --------------------------------------------------------------------------
# scoring + clustering
# --------------------------------------------------------------------------
def build_master(root: Path, out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    gates = parse_gates(root)
    metrics = parse_metrics(root)
    shot = parse_shot_noise(root)
    ber = parse_ber(root)
    xcorr = parse_xcorr(root)
    census = parse_census(out_dir, root)

    sources = {
        "gates": bool(gates), "metrics": bool(metrics), "shot_noise": bool(shot),
        "ber": bool(ber), "xcorr": bool(xcorr), "census": bool(census),
    }
    if not gates:
        raise SystemExit("FATAL: gate table is required (fs_vs_fc_crossdist/gates_summary_FS_vs_FC_crossdist.csv).")

    # assemble raw rows keyed by gate regimes (most complete: 7 dist x 9 curr x 2 models)
    rows: list[dict[str, Any]] = []
    for k, g in gates.items():
        model, dist, curr = k[0], g["dist_m"], g["curr_mA"]
        row: dict[str, Any] = dict(g)
        row.update(metrics.get(k, {}))
        row.update(shot.get(k, {}))
        row.update(ber.get(k, {}))
        row["xcorr_l2"] = xcorr.get((model, round(dist, 4)), float("nan"))
        row.update(census.get(k, {}))
        rows.append(row)

    # difficulty signals, normalised WITHIN each model
    for model in MODELS:
        sub = [r for r in rows if r["model"] == model]
        idx = {id(r): r for r in sub}
        sig_dist = {id(r): (_f(r.get("cvae_kurt_l2"), 0.0) + _f(r.get("cvae_skew_l2"), 0.0)
                            + 10.0 * _f(r.get("cvae_w1_mag"), 0.0)) for r in sub}
        sig_het = {id(r): _f(r.get("het_mismatch"), 0.0) for r in sub}
        sig_ber = {id(r): _f(r.get("ber_abs_err_mean"), 0.0) for r in sub}
        n_dist, n_het, n_ber = _minmax_norm(sig_dist), _minmax_norm(sig_het), _minmax_norm(sig_ber)
        for r in sub:
            gate_frac = r["n_fail"] / len(GATE_COLS)
            parts, weights = [], []
            parts.append(gate_frac); weights.append(0.40)
            if math.isfinite(_f(r.get("cvae_kurt_l2"))):
                parts.append(n_dist[id(r)]); weights.append(0.25)
            if math.isfinite(_f(r.get("het_mismatch"))):
                parts.append(n_het[id(r)]); weights.append(0.15)
            if math.isfinite(_f(r.get("ber_abs_err_mean"))):
                parts.append(n_ber[id(r)]); weights.append(0.20)
            wsum = sum(weights)
            r["difficulty_score"] = round(sum(p * w for p, w in zip(parts, weights)) / wsum, 4) if wsum else 0.0

    # cluster + separation action
    for r in rows:
        status = r.get("status", "")
        if status == "PASS":
            r["cluster"] = "bem_modelado"
            r["separation_action"] = "manter (baseline/screening)"
        elif r.get("unseen"):
            r["cluster"] = "extrapolacao_distancia"
            r["separation_action"] = "hold-out de generalizacao; adicionar capturas intermediarias se houver"
        else:
            r["cluster"] = "dificil_distribucional"
            near = r["dist_m"] <= 0.8
            hi = r["curr_mA"] >= 700
            tag = "near-field" if near else ("alta-corrente" if hi else "regime-isolado")
            r["separation_action"] = f"oversample + candidato a especialista ({tag})"

    rows.sort(key=lambda r: (r["model"], r["dist_m"], r["curr_mA"]))
    return rows, sources


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------
MASTER_COLS = [
    "model", "dist_m", "curr_mA", "unseen", "status", "n_fail", "failed_gates",
    *[f"gate_{g}" for g in GATE_COLS],
    "difficulty_score", "cluster", "separation_action",
    "real_channel_snr_db", "cvae_delta_channel_snr_db", "awgn_delta_channel_snr_db",
    "cvae_kurt_l2", "cvae_skew_l2", "cvae_w1_mag", "awgn_kurt_l2",
    "cvae_snr_fidelity_db", "awgn_snr_fidelity_db",
    "het_a_real", "het_a_cvae", "het_mismatch",
    "ber_abs_err_mean", "ber_abs_err_64qam", "xcorr_l2",
    "real_kurt_canon", "het_slope_canon", "real_kurt_eq", "het_slope_eq",
    "frac_linear", "frac_nonlinear", "frac_noise", "real_kurt_nl", "het_slope_nl",
    "real_var_mag", "n_samples",
]


def write_master_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in MASTER_COLS})


def _pass_rate_by(rows: list[dict], model: str, key: str) -> list[tuple[Any, int, int]]:
    sub = [r for r in rows if r["model"] == model]
    buckets: dict[Any, list[bool]] = {}
    for r in sub:
        buckets.setdefault(r[key], []).append(r.get("status") == "PASS")
    return [(k, sum(v), len(v)) for k, v in sorted(buckets.items())]


def write_report(rows: list[dict[str, Any]], sources: dict[str, bool], path: Path,
                 run_dir: Path | None = None) -> None:
    L: list[str] = []
    L.append("# Relatório Macro de Diagnóstico Pós-Modelagem — cVAE V3 (FS & FC)\n")
    L.append(f"Gerado: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    present = ", ".join(k for k, v in sources.items() if v) or "nenhuma"
    L.append(f"Camadas com dados: **{present}**.\n")

    # veredito de topo
    L.append("## Veredito\n")
    for model in MODELS:
        sub = [r for r in rows if r["model"] == model]
        n = len(sub)
        npass = sum(1 for r in sub if r.get("status") == "PASS")
        clusters = {}
        for r in sub:
            clusters[r["cluster"]] = clusters.get(r["cluster"], 0) + 1
        cl = ", ".join(f"{k}={v}" for k, v in sorted(clusters.items()))
        L.append(f"- **{model}**: {npass}/{n} regimes PASS. Clusters: {cl}.")
    L.append("")

    # camada 1 — resposta linear
    if sources["xcorr"]:
        L.append("## Camada 1 — Resposta linear (xcorr)\n")
        worst = max((r.get("xcorr_l2", 0.0) for r in rows if math.isfinite(_f(r.get("xcorr_l2")))), default=0.0)
        L.append(f"Erro L2 máximo de `R_xy(τ)` real vs gêmeo: **{worst:.4f}** (< 1% ⇒ resposta "
                 "linear determinística interpolada; a falha é distribucional, não de ganho).\n")

    # camada 2 — heterocedasticidade
    if sources["shot_noise"]:
        L.append("## Camada 2 — Heterocedasticidade (shot noise)\n")
        for model in MODELS:
            ms = [r.get("het_mismatch") for r in rows if r["model"] == model and math.isfinite(_f(r.get("het_mismatch")))]
            if ms:
                L.append(f"- **{model}**: mismatch médio do slope Var(δ)~|X| (cvae vs real) = "
                         f"**{sum(ms)/len(ms):.2%}** (AWGN é plano por construção).")
        L.append("")

    # camada 0 — interpretação física dos dois resíduos (census)
    if sources["census"]:
        L.append("## Camada 0 — Estrutura do ruído real (dois resíduos)\n")
        for model in MODELS:
            kc = [_f(r.get("real_kurt_canon")) for r in rows if r["model"] == model and math.isfinite(_f(r.get("real_kurt_canon")))]
            ke = [_f(r.get("real_kurt_eq")) for r in rows if r["model"] == model and math.isfinite(_f(r.get("real_kurt_eq")))]
            hc = [abs(_f(r.get("het_slope_canon"))) for r in rows if r["model"] == model and math.isfinite(_f(r.get("het_slope_canon")))]
            he = [abs(_f(r.get("het_slope_eq"))) for r in rows if r["model"] == model and math.isfinite(_f(r.get("het_slope_eq")))]
            if kc and ke:
                L.append(f"- **{model}**: canônico (Y−X) kurt={sum(kc)/len(kc):+.2f}, het~{sum(hc)/len(hc):.2e}  →  "
                         f"equalizado (Y−a·X) kurt={sum(ke)/len(ke):+.2f}, het~{sum(he)/len(he):.2e}.")
        L.append("")
        L.append("**Interpretação física** (cards locais — sínteses, não papers primários): "
                 "a heterocedasticidade do resíduo canônico Var(Y−X)∝|X|² é a assinatura do "
                 "**mapa de ganho determinístico** do LED — saturação Hammerstein "
                 "([[dang_2022]]) / quadrador Volterra ([[bojarczuk_2023]]), que isolam a distorção "
                 "de SINAL e explicitamente **não** modelam ruído. Removido o ganho linear, o resíduo "
                 "equalizado fica **≈Gaussiano e ≈homocedástico** (kurt≈0, het≈0): o ruído é benigno. "
                 "Ver [[liu_2024]] para as famílias estatísticas por regime.\n")

        # opção 1b — decomposição de variância (veredito para o reprojeto do treino)
        if any(math.isfinite(_f(r.get("frac_noise"))) for r in rows):
            L.append("### Decomposição de variância (opção 1b — alvo do reprojeto do treino)\n")
            L.append("Fit de um mapa de ganho não-linear g(P)·x por regime (treino/teste). "
                     "Decompõe Var(Y−X) — o 'ruído aparente' que o AWGN teria de absorver — em "
                     "ganho linear / extra não-linear / ruído irredutível.\n")
            L.append("| modelo | frac. linear | frac. não-linear | frac. ruído | kurt do resíduo final |")
            L.append("|---|---|---|---|---|")
            verdicts = []
            for model in MODELS:
                sub = [r for r in rows if r["model"] == model and math.isfinite(_f(r.get("frac_noise")))]
                if not sub:
                    continue
                fl = sum(_f(r["frac_linear"]) for r in sub) / len(sub)
                fn = sum(_f(r["frac_nonlinear"]) for r in sub) / len(sub)
                fz = sum(_f(r["frac_noise"]) for r in sub) / len(sub)
                kn = sum(_f(r["real_kurt_nl"]) for r in sub) / len(sub)
                L.append(f"| {model} | {fl:.1%} | {fn:.1%} | {fz:.1%} | {kn:+.2f} |")
                verdicts.append((model, fl, fn, fz, kn))
            L.append("")
            # high-current near-field focus: where nonlinearity (saturation) should peak
            hot = [r for r in rows if r["model"] == "FS" and _f(r.get("dist_m")) <= 0.8
                   and _f(r.get("curr_mA")) >= 700 and math.isfinite(_f(r.get("frac_nonlinear")))]
            if hot:
                fn_hot = sum(_f(r["frac_nonlinear"]) for r in hot) / len(hot)
                L.append(f"- Foco near-field/alta-corrente (FS ≤0.8 m, ≥700 mA), onde a saturação "
                         f"deveria ser máxima: fração não-linear média = **{fn_hot:.1%}**.\n")
            # automated verdict line
            if verdicts:
                fn_max = max(v[2] for v in verdicts)
                kn_max = max(abs(v[4]) for v in verdicts)
                if fn_max < 0.05 and kn_max < 0.2:
                    L.append("> **Veredito**: a parcela não-linear extra é pequena e o resíduo final é "
                             "Gaussiano. A dificuldade por regime é **ganho/atenuação + interpolação "
                             "entre regimes**, NÃO cauda de ruído. Reprojeto: priorizar média condicional, "
                             "condicionamento em amplitude e capturas intermediárias; **não** investir em "
                             "capacidade de cauda do MDN (alinha com os negativos S28/S29/S33 do histórico).\n")
                else:
                    L.append("> **Veredito**: há parcela não-linear material e/ou resíduo não-Gaussiano. "
                             "Reprojeto: dar capacidade ao **mapa determinístico não-linear** (warp + "
                             "condicionamento em amplitude) nos regimes de maior fração não-linear; o ruído "
                             "em si permanece benigno (não priorizar cauda do MDN).\n")

    # camada 4 — mapa de gates
    L.append("## Camada 4 — Mapa de falhas por regime\n")
    for model in MODELS:
        L.append(f"### {model} — taxa de PASS por distância")
        L.append("| dist (m) | PASS/total |")
        L.append("|---|---|")
        for d, p, t in _pass_rate_by(rows, model, "dist_m"):
            L.append(f"| {d:g} | {p}/{t} |")
        L.append("")

    # camada 5 — BER
    if sources["ber"]:
        L.append("## Camada 5 — Fidelidade de BER (aplicação)\n")
        for model in MODELS:
            es = [r.get("ber_abs_err_mean") for r in rows if r["model"] == model and math.isfinite(_f(r.get("ber_abs_err_mean")))]
            e64 = [r.get("ber_abs_err_64qam") for r in rows if r["model"] == model and math.isfinite(_f(r.get("ber_abs_err_64qam")))]
            if es:
                L.append(f"- **{model}**: |BER_gêmeo − BER_real| médio = **{sum(es)/len(es):.2e}** "
                         f"(64-QAM: {sum(e64)/len(e64):.2e})." if e64 else
                         f"- **{model}**: |BER_gêmeo − BER_real| médio = **{sum(es)/len(es):.2e}**.")
        L.append("")

    # ranking de dificuldade
    L.append("## Ranking de dificuldade (top 8 por modelo)\n")
    for model in MODELS:
        sub = sorted([r for r in rows if r["model"] == model], key=lambda r: -r.get("difficulty_score", 0.0))[:8]
        L.append(f"### {model}")
        L.append("| regime | difficulty | gates falhos | cluster |")
        L.append("|---|---|---|---|")
        for r in sub:
            L.append(f"| {r['dist_m']:g} m / {r['curr_mA']} mA | {r.get('difficulty_score', 0):.3f} "
                     f"| {r.get('failed_gates') or '—'} | {r['cluster']} |")
        L.append("")

    # recomendação de separação de dados
    L.append("## Recomendação de separação de dados (objetivo do pipeline)\n")
    L.append("Agrupamento dos regimes que falham, com a ação sugerida para o próximo treino.\n")
    L.append("| cluster | ação | regimes (model: dist/curr) |")
    L.append("|---|---|---|")
    by_action: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        if r["cluster"] == "bem_modelado":
            continue
        by_action.setdefault((r["cluster"], r["separation_action"]), []).append(
            f"{r['model']}:{r['dist_m']:g}/{r['curr_mA']}")
    for (cluster, action), regs in sorted(by_action.items()):
        shown = ", ".join(regs[:12]) + (f" … (+{len(regs)-12})" if len(regs) > 12 else "")
        L.append(f"| {cluster} | {action} | {shown} |")
    L.append("")

    # governança
    L.append("## Nota de governança de dados\n")
    L.append("Papéis sugeridos dos regimes (lacuna #1 da síntese de validação do twin):\n")
    L.append("- **Screening** (desenvolvimento rápido): regimes `bem_modelado` em 1.0/1.35 m.")
    L.append("- **Final-validation** (aceite de tese): cobertura completa treinada (0.75–1.5 m).")
    L.append("- **Generalização** (advisory): distâncias não-vistas 0.9/1.16/1.25 m — nunca no treino.\n")
    L.append("> A separação proposta NÃO altera o dataset nem retreina; entrega o mapa + a ação.\n")

    # figuras auto-contidas (layout N_*/) — links relativos ao run folder
    if run_dir is not None:
        L.append("## Figuras\n")
        any_fig = False
        for sub, title in FIG_SECTIONS:
            d = run_dir / sub
            if not d.is_dir():
                continue
            pngs = sorted(d.rglob("*.png"))
            if not pngs:
                continue
            any_fig = True
            L.append(f"### {title}\n")
            for p in pngs:
                rel = p.relative_to(run_dir)
                L.append(f"- [{rel.as_posix()}]({rel.as_posix()})")
            L.append("")
        if not any_fig:
            L.append("_(nenhuma figura coletada neste run)_\n")

    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(COMPARISON_V3), help="comparison_v3 root")
    ap.add_argument("--out-dir", default=None, help="output dir (default: macro_diagnostics/runs/<stamp>)")
    ap.add_argument("--in-dir", default=None,
                    help="self-contained run dir (N_*/ layout): read sources from it, "
                         "write REPORT inside it, and index its figures")
    args = ap.parse_args()

    run_dir = None
    if args.in_dir:
        # self-contained run: everything lives under the run dir in the N_*/ layout
        SRC.update(SRC_RUN)
        run_dir = Path(args.in_dir).resolve()
        root = run_dir
        out_dir = run_dir
    else:
        root = Path(args.root).resolve()
        if args.out_dir:
            out_dir = Path(args.out_dir).resolve()
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_dir = root / "macro_diagnostics" / "runs" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, sources = build_master(root, out_dir)
    write_master_csv(rows, out_dir / "summary_master.csv")
    write_report(rows, sources, out_dir / "REPORT.md", run_dir=run_dir)
    (out_dir / "manifest.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "sources_present": sources,
        "n_rows": len(rows), "models": list(MODELS),
    }, indent=2), encoding="utf-8")
    print(f"Done: {out_dir}")
    print(f"  summary_master.csv ({len(rows)} rows)  REPORT.md  manifest.json")


if __name__ == "__main__":
    main()
