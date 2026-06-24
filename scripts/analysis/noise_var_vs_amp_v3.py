#!/usr/bin/env python3
"""noise_var_vs_amp_v3.py

Generates:
  1. noise_variance_vs_amplitude_grid_{FS,FC}.png  — diagnostic grid (all regimes)
  2. shot_noise_coefficients.csv                   — fitted a,b per regime/method
  3. article_fig_A_shot_noise.png                  — Strategy A: exemplar + scatter
  4. article_fig_B_heatmap.png                     — Strategy B: r=b_cvae/b_real heatmaps

Usage (inside docker with PYTHONPATH set to repo root):
  python scripts/analysis/noise_var_vs_amp_v3.py [--skip-fs] [--skip-fc]
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

FS_MODEL = os.path.join(REPO, "outputs/v3fs_crossdist_20260613/exp_20260613_205805/train/models/best_model_full.keras")
FC_MODEL = os.path.join(REPO, "outputs/v3fc_crossdist_20260613/exp_20260613_211826/train/models/best_model_full.keras")
FS_DATA  = "/home/rodrigo/1-Data/Dataset/V3/FULLSQUARE_2026_V3_ORGANIZED"
FC_DATA  = "/home/rodrigo/1-Data/Dataset/V3/FULL_CIRCLE_2026_V3_ORGANIZED"
OUT_DIR  = "/home/rodrigo/comparison_v3/awgn"

TRAINED_DISTS  = [0.75, 1.0, 1.35, 1.5]
ARTICLE_DISTS  = [1.0, 1.35, 1.5]   # exclude near-field from article figs
ALL_CURRS      = [100, 200, 300, 400, 500, 600, 700, 800, 900]
EXEMPLAR       = (1.35, 500)         # (dist_m, curr_mA) used as exemplar
VAL_SPLIT      = 0.2
SEED           = 33
NORM           = {"D_min": 0.75, "D_max": 1.5, "C_min": 100.0, "C_max": 900.0}
N_BINS         = 12

# ── visual style ──────────────────────────────────────────────────────────────
PALETTE = {
    "real": "#1d4ed8", "cvae": "#ea580c", "awgn": "#475569",
    "fs":   "#2563eb", "fc":   "#ea580c",
    "axis": "#1f2937", "grid": "#cbd5e1",
}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": PALETTE["axis"], "axes.labelcolor": PALETTE["axis"],
    "axes.titleweight": "bold", "axes.titlesize": 10.0, "axes.labelsize": 9.0,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": PALETTE["axis"], "ytick.color": PALETTE["axis"],
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "grid.color": PALETTE["grid"], "grid.linestyle": "-",
    "grid.linewidth": 0.55, "grid.alpha": 0.55,
    "legend.fontsize": 8.5, "font.family": "DejaVu Sans",
    "font.size": 9.0,
})


# ── helpers ───────────────────────────────────────────────────────────────────
def _complex(a: np.ndarray) -> np.ndarray:
    return a[:, 0].astype(np.float64) + 1j * a[:, 1].astype(np.float64)


def _norm_cond(d: float, c: float) -> tuple[float, float]:
    dn = (d - NORM["D_min"]) / (NORM["D_max"] - NORM["D_min"])
    cn = (c - NORM["C_min"]) / (NORM["C_max"] - NORM["C_min"])
    return float(dn), float(cn)


def _find_xy(dataset_root: str, dist_m: float, curr_mA: int) -> tuple[np.ndarray, np.ndarray]:
    candidates = glob.glob(os.path.join(dataset_root, f"dist_{dist_m:g}m", f"curr_{curr_mA}mA", "*", "IQ_data"))
    if not candidates:
        candidates = glob.glob(os.path.join(dataset_root, f"dist_{dist_m:.2g}m", f"curr_{curr_mA}mA", "*", "IQ_data"))
    if not candidates:
        for ddir in glob.glob(os.path.join(dataset_root, "dist_*m")):
            dname = os.path.basename(ddir).removeprefix("dist_").removesuffix("m")
            try:
                if abs(float(dname) - dist_m) < 1e-6:
                    candidates = glob.glob(os.path.join(ddir, f"curr_{curr_mA}mA", "*", "IQ_data"))
                    break
            except ValueError:
                continue
    if not candidates:
        raise FileNotFoundError(f"No IQ_data for dist={dist_m}m curr={curr_mA}mA in {dataset_root}")
    iq_dir = candidates[0]
    X = np.load(os.path.join(iq_dir, "X.npy"), allow_pickle=False)
    Y = np.load(os.path.join(iq_dir, "Y.npy"), allow_pickle=False)
    if X.ndim == 1: X = X.reshape(-1, 2)
    if Y.ndim == 1: Y = Y.reshape(-1, 2)
    n = min(len(X), len(Y))
    return X[:n].astype(np.float32), Y[:n].astype(np.float32)


def _val_slice(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_train = max(1, int(round((1.0 - VAL_SPLIT) * len(X))))
    return X[n_train:], Y[n_train:]


def _matched_awgn(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise_power = float(np.mean(np.abs(_complex(Y) - _complex(X)) ** 2))
    sigma = math.sqrt(max(noise_power, 1e-18) / 2.0)
    noise = rng.normal(0.0, sigma, len(X)) + 1j * rng.normal(0.0, sigma, len(X))
    realized = float(np.mean(np.abs(noise) ** 2))
    if realized > 1e-18:
        noise *= math.sqrt(max(noise_power, 1e-18) / realized)
    yc = _complex(X) + noise
    return np.column_stack([yc.real, yc.imag]).astype(np.float32)


def _load_model_and_predict(model_path: str, X: np.ndarray, dist_m: float, curr_mA: float,
                             _cache: dict = {}) -> np.ndarray:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from src.models.cvae import create_inference_model_from_full
    from src.models.cvae_sequence import load_seq_model

    if model_path not in _cache:
        model = load_seq_model(model_path)
        try:
            prior = model.get_layer("prior_net")
            is_seq = len(prior.inputs[0].shape) == 3
            window_size = int(prior.inputs[0].shape[1]) if is_seq else None
        except Exception:
            is_seq, window_size = False, None
        inf = create_inference_model_from_full(model, deterministic=False)
        _cache[model_path] = (inf, is_seq, window_size)

    inf, is_seq, window_size = _cache[model_path]
    N = len(X)
    dn, cn = _norm_cond(dist_m, curr_mA)
    Dn = np.full((N, 1), dn, dtype=np.float32)
    Cn = np.full((N, 1), cn, dtype=np.float32)

    if is_seq and window_size:
        from src.data.windowing import build_windows_single_experiment
        Xin, _, _, _ = build_windows_single_experiment(
            X, X, Dn, Cn, window_size=window_size, stride=1, pad_mode="edge")
    else:
        Xin = X

    pred = inf.predict([Xin, Dn, Cn], batch_size=8192, verbose=0)
    return np.asarray(pred, dtype=np.float32)


# ── curve computation ─────────────────────────────────────────────────────────
def _bin_variance(mag_x: np.ndarray, residual: np.ndarray, edges: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    vars_ = np.full(len(centers), np.nan)
    for k in range(len(centers)):
        mask = (mag_x >= edges[k]) & (mag_x < edges[k + 1])
        if int(mask.sum()) < 30:
            continue
        vars_[k] = float(np.var(residual[mask, 0]) + np.var(residual[mask, 1]))
    return centers, vars_


def _fit_shot_noise(centers: np.ndarray, vs: np.ndarray) -> tuple[float, float, float]:
    """Fit var = a + b*|X|^2. Returns (a, b, r2). b>0 => shot noise component."""
    mask = np.isfinite(vs)
    if mask.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    xx = centers[mask] ** 2
    yy = vs[mask]
    A = np.column_stack([np.ones_like(xx), xx])
    sol, *_ = np.linalg.lstsq(A, yy, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    yhat = a + b * xx
    ss_res = float(np.sum((yy - yhat) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return a, b, r2


def _compute_curves(X: np.ndarray, Yv: np.ndarray, Yp: np.ndarray, Ya: np.ndarray
                    ) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    mx = np.abs(_complex(X)).astype(np.float64)
    residuals = {
        "real": np.column_stack([(_complex(Yv) - _complex(X)).real, (_complex(Yv) - _complex(X)).imag]),
        "cvae": np.column_stack([(_complex(Yp) - _complex(X)).real, (_complex(Yp) - _complex(X)).imag]),
        "awgn": np.column_stack([(_complex(Ya) - _complex(X)).real, (_complex(Ya) - _complex(X)).imag]),
    }
    qs = np.linspace(0.0, 1.0, N_BINS + 1)
    edges = np.maximum.accumulate(np.quantile(mx, qs))
    curves = {key: _bin_variance(mx, res, edges) for key, res in residuals.items()}
    return mx, curves


# ── diagnostic grid (all regimes) ─────────────────────────────────────────────
def _make_grid(label: str, regime_data: dict, out_path: str
               ) -> tuple[list[dict], dict]:
    """Returns (coefs_list, exemplar_dict)."""
    series_def = [
        ("real", PALETTE["real"], "-",  "o", 2.0, "Real (canal)"),
        ("cvae", PALETTE["cvae"], "-",  "s", 1.9, "cVAE"),
        ("awgn", PALETTE["awgn"], "--", "^", 1.7, "AWGN"),
    ]
    dist_order = sorted({d for d, c in regime_data})
    curr_order = sorted({c for d, c in regime_data})
    nr, nc = len(dist_order), len(curr_order)

    panel_curves: dict = {}
    row_y_max: dict[float, float] = {}
    coefs_list: list[dict] = []
    exemplar_dict: dict = {}

    for (dist, curr), (X, Yv, Yp, Ya) in regime_data.items():
        mx, curves = _compute_curves(X, Yv, Yp, Ya)
        panel_curves[(dist, curr)] = (curves, mx)

        for key, *_ in series_def:
            c_arr, v_arr = curves[key]
            finite = v_arr[np.isfinite(v_arr)]
            if finite.size:
                row_y_max[dist] = max(row_y_max.get(dist, 0.0), float(np.nanmax(finite)))
            a, b, r2 = _fit_shot_noise(c_arr, v_arr)
            coefs_list.append({"model": label, "dist_m": dist, "curr_mA": curr,
                                "method": key, "a": a, "b": b, "r2": r2})

        if (dist, curr) == EXEMPLAR:
            exemplar_dict = {
                "dist": dist, "curr": curr,
                "real": curves["real"], "cvae": curves["cvae"], "awgn": curves["awgn"],
            }

    cell = 4.6
    fig, axes = plt.subplots(nr, nc, figsize=(cell * nc, cell * nr), facecolor="white")
    if nr == 1: axes = [axes]
    if nc == 1: axes = [[ax] for ax in axes]

    for i, dist in enumerate(dist_order):
        for j, curr in enumerate(curr_order):
            ax = axes[i][j]
            item = panel_curves.get((dist, curr))
            if item is None:
                ax.set_visible(False)
                continue
            curves, mx = item

            for key, color, ls, marker, lw, lbl in series_def:
                centers, vs = curves[key]
                ax.plot(centers, vs, color=color, ls=ls, lw=lw,
                        marker=marker, markersize=5.5,
                        markerfacecolor=color, markeredgecolor="white",
                        markeredgewidth=0.7, label=lbl, zorder=3 if key == "cvae" else 2)

            cr, vr = curves["real"]
            a_fit, b_fit, _ = _fit_shot_noise(cr, vr)
            mask = np.isfinite(vr)
            if mask.sum() >= 4 and np.isfinite(a_fit):
                xs_fit = np.linspace(cr[mask].min(), cr[mask].max(), 80)
                ax.plot(xs_fit, a_fit + b_fit * xs_fit ** 2,
                        color="#0f172a", lw=1.0, ls=":", alpha=0.7, zorder=1,
                        label=r"ajuste: $a + b|X|^2$" if (i == 0 and j == 0) else None)
                xmax = float(cr[mask].max())
                if xmax > 0 and b_fit > 0:
                    pct = 100.0 * b_fit * xmax**2 / max(a_fit + b_fit * xmax**2, 1e-18)
                    ax.text(0.97, 0.03, f"shot≈{pct:.0f}%\n@ |X|max",
                            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                            color="#1f2937", bbox=dict(facecolor="white", edgecolor=PALETTE["grid"],
                                                       lw=0.6, alpha=0.85, pad=2.5))

            ax.set_ylim(0, row_y_max.get(dist, 1.0) * 1.12)
            ax.grid(True, alpha=0.25, lw=0.4)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if i == 0:
                ax.set_title(f"{curr} mA", fontsize=13, fontweight="bold",
                             color=PALETTE["axis"], pad=10)
            if i == nr - 1:
                ax.set_xlabel("|X| (amplitude enviada)", fontsize=10)
            if j == 0:
                ax.set_ylabel(r"Var($\delta$ | |X|)", fontsize=10)
                ax.text(-0.28, 0.5, f"{dist:g} m", transform=ax.transAxes,
                        ha="right", va="center", fontsize=13, fontweight="bold",
                        color=PALETTE["axis"])

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=11.5,
               frameon=True, framealpha=0.96, edgecolor=PALETTE["grid"], handlelength=2.6)
    fig.text(0.02, 0.5, "Distância", va="center", ha="center", rotation=90,
             fontsize=14, fontweight="bold", color=PALETTE["axis"])
    fig.text(0.5, 0.995, "Corrente (mA)", va="top", ha="center",
             fontsize=14, fontweight="bold", color=PALETTE["axis"])
    fig.text(0.5, 0.030,
             "Ruído térmico ⇒ curva plana. "
             "Componente shot ⇒ Var(δ) cresce com |X|² (linha pontilhada = ajuste $a+b|X|^2$ no Real). "
             "AWGN é plano por construção; cVAE deve seguir o Real.",
             ha="center", fontsize=9.5, style="italic", color="#475569")
    fig.suptitle(f"{label} — variância condicional do ruído Var(δ | |X|) por regime",
                 fontsize=16, fontweight="bold", color=PALETTE["axis"], y=1.005)
    fig.tight_layout(rect=[0.035, 0.05, 1, 0.985])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out_path}")
    return coefs_list, exemplar_dict


# ── Article Figure A: exemplar + scatter ──────────────────────────────────────
def _make_article_fig_A(fs_coefs: list[dict], fc_coefs: list[dict],
                         fs_exemplar: dict, fc_exemplar: dict,
                         out_path: str) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.2, 3.8),
                                             gridspec_kw={"wspace": 0.38})

    # ── Panel (a): exemplar curves ─────────────────────────────────────────
    dist_e, curr_e = EXEMPLAR
    series = [
        ("real",   fs_exemplar, PALETTE["real"],  "-",  "D", 1.8, 5.5, "Real (canal)"),
        ("cvae",   fs_exemplar, PALETTE["fs"],    "--", "s", 1.5, 5.0, "FS — cVAE"),
        ("cvae",   fc_exemplar, PALETTE["fc"],    "-",  "o", 1.8, 5.5, "FC — cVAE"),
        ("awgn",   fs_exemplar, PALETTE["awgn"],  ":",  "^", 1.4, 4.5, "AWGN"),
    ]
    for key, expl, color, ls, marker, lw, ms, lbl in series:
        if not expl:
            continue
        centers, vs = expl[key]
        ax_left.plot(centers, vs, color=color, ls=ls, lw=lw,
                     marker=marker, markersize=ms,
                     markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.6,
                     label=lbl, zorder=4 if "cVAE" in lbl else 2)

    # fit line on Real
    if fs_exemplar:
        cr, vr = fs_exemplar["real"]
        a_fit, b_fit, _ = _fit_shot_noise(cr, vr)
        if np.isfinite(a_fit):
            mask = np.isfinite(vr)
            xs = np.linspace(cr[mask].min(), cr[mask].max(), 80)
            ax_left.plot(xs, a_fit + b_fit * xs**2,
                         color="#0f172a", lw=1.0, ls=(0, (3, 1, 1, 1)), alpha=0.7,
                         label=r"ajuste $a + b|X|^2$", zorder=1)

    ax_left.set_xlabel(r"$|X|$ — amplitude do sinal enviado", fontsize=9)
    ax_left.set_ylabel(r"$\mathrm{Var}(\delta\mid|X|)$", fontsize=9)
    ax_left.set_title(f"(a) Regime exemplar: {dist_e:g} m · {curr_e} mA", fontsize=9.5)
    ax_left.set_ylim(bottom=0)
    ax_left.grid(True, alpha=0.25, lw=0.4)
    ax_left.legend(fontsize=7.8, loc="upper left", framealpha=0.9,
                   edgecolor=PALETTE["grid"], handlelength=2.2)

    # ── Panel (b): scatter b_cvae vs b_real ───────────────────────────────
    def _coef_map(coefs: list[dict], method: str) -> dict[tuple, float]:
        return {(r["dist_m"], r["curr_mA"]): r["b"]
                for r in coefs if r["method"] == method}

    fs_real  = _coef_map(fs_coefs, "real")
    fs_cvae  = _coef_map(fs_coefs, "cvae")
    fs_awgn  = _coef_map(fs_coefs, "awgn")
    fc_real  = _coef_map(fc_coefs, "real")
    fc_cvae  = _coef_map(fc_coefs, "cvae")
    fc_awgn  = _coef_map(fc_coefs, "awgn")

    # collect (b_real, b_method) for article distances only
    def _pairs(real_map, method_map):
        xs, ys = [], []
        for (d, c), br in real_map.items():
            if d not in ARTICLE_DISTS:
                continue
            bm = method_map.get((d, c))
            if bm is None or not np.isfinite(br) or not np.isfinite(bm):
                continue
            if br < 1e-8:
                continue
            xs.append(br); ys.append(bm)
        return xs, ys

    scatter_series = [
        (_pairs(fs_real, fs_cvae), PALETTE["fs"],   "s",  42, "FS — cVAE"),
        (_pairs(fc_real, fc_cvae), PALETTE["fc"],   "o",  42, "FC — cVAE"),
        (_pairs(fs_real, fs_awgn), PALETTE["awgn"], "^",  28, "AWGN (FS·FC)"),
    ]

    all_b = []
    for (xs, ys), color, marker, ms, lbl in scatter_series:
        if not xs:
            continue
        ax_right.scatter(xs, ys, color=color, marker=marker, s=ms,
                         alpha=0.80, edgecolors="white", linewidths=0.5,
                         label=lbl, zorder=3)
        all_b.extend(xs); all_b.extend(ys)

    # also add AWGN from FC (same AWGN values, so overlay)
    fc_awgn_pairs = _pairs(fc_real, fc_awgn)
    if fc_awgn_pairs[0]:
        ax_right.scatter(fc_awgn_pairs[0], fc_awgn_pairs[1],
                         color=PALETTE["awgn"], marker="v", s=28,
                         alpha=0.80, edgecolors="white", linewidths=0.5, zorder=3)

    if all_b:
        bmax = max(b for b in all_b if np.isfinite(b)) * 1.08
        bmax = max(bmax, 1e-10)
        ax_right.plot([0, bmax], [0, bmax], color="#0f172a", lw=1.1, ls="--",
                      alpha=0.6, label="y = x (ideal)", zorder=1)
        ax_right.set_xlim(0, bmax)
        ax_right.set_ylim(-bmax * 0.05, bmax * 1.08)

    ax_right.axhline(0, color=PALETTE["awgn"], lw=0.7, ls=":", alpha=0.5)
    ax_right.set_xlabel(r"$b_{\rm real}$ — coeficiente shot noise Real", fontsize=9)
    ax_right.set_ylabel(r"$b_{\rm método}$ — coeficiente shot noise Método", fontsize=9)
    ax_right.set_title(r"(b) Reprodução do shot noise — $b_{\rm cVAE}$ vs $b_{\rm real}$"
                        + f"\n(distâncias: {', '.join(f'{d:g} m' for d in ARTICLE_DISTS)})",
                        fontsize=9.5)
    ax_right.legend(fontsize=7.8, loc="upper left", framealpha=0.9,
                    edgecolor=PALETTE["grid"], handlelength=1.8)
    ax_right.grid(True, alpha=0.25, lw=0.4)

    fig.subplots_adjust(top=0.72, bottom=0.15, left=0.08, right=0.97, wspace=0.32)
    fig.suptitle("cVAE reproduz a heterocedasticidade do canal VLC (ruído shot)\n"
                 "AWGN não captura a dependência da variância com a amplitude do sinal",
                 fontsize=9.5, style="italic", color="#475569", y=0.91)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out_path}")


# ── Article Figure B: heatmaps r = b_cvae / b_real ────────────────────────────
def _make_article_fig_B(fs_coefs: list[dict], fc_coefs: list[dict],
                         out_path: str) -> None:
    dists = ARTICLE_DISTS
    currs = ALL_CURRS

    def _ratio_matrix(coefs: list[dict], method: str = "cvae") -> np.ndarray:
        real_map = {(r["dist_m"], r["curr_mA"]): r["b"]
                    for r in coefs if r["method"] == "real"}
        method_map = {(r["dist_m"], r["curr_mA"]): r["b"]
                      for r in coefs if r["method"] == method}
        mat = np.full((len(dists), len(currs)), np.nan)
        for i, d in enumerate(dists):
            for j, c in enumerate(currs):
                br = real_map.get((d, c))
                bm = method_map.get((d, c))
                if br and bm is not None and np.isfinite(br) and np.isfinite(bm) and br > 1e-8:
                    mat[i, j] = np.clip(bm / br, 0.0, 2.5)
        return mat

    mat_fs = _ratio_matrix(fs_coefs, "cvae")
    mat_fc = _ratio_matrix(fc_coefs, "cvae")
    mat_awgn = _ratio_matrix(fs_coefs, "awgn")

    # divergent colormap centered at 1.0
    norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=2.5)
    cmap = plt.cm.RdBu_r

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.7), gridspec_kw={"wspace": 0.28})

    panels = [
        (axes[0], mat_fs, "(a) FS — cVAE"),
        (axes[1], mat_fc, "(b) FC — cVAE"),
        (axes[2], mat_awgn, "(c) AWGN")
    ]

    im = None
    for ax, mat, title in panels:
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(currs)))
        ax.set_xticklabels([str(c) for c in currs], fontsize=7.0, rotation=45, ha="right")
        ax.set_yticks(range(len(dists)))
        ax.set_yticklabels([f"{d:g} m" for d in dists], fontsize=7.5)
        ax.set_xlabel("Corrente (mA)", fontsize=8)
        ax.set_ylabel("Distância", fontsize=8)
        ax.set_title(title, fontsize=9.0)

    # single shared colorbar on the right
    fig.subplots_adjust(top=0.70, bottom=0.18, left=0.08, right=0.86, wspace=0.38)
    cbar_ax = fig.add_axes([0.89, 0.18, 0.02, 0.52]) # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"Razão $r = b_{\rm método}/b_{\rm real}$", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.0)
    cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])

    fig.suptitle(r"Razão de reprodução do shot noise: $r = b_{\rm método}/b_{\rm real}$"
                 "\n$r = 1$ = reprodução perfeita (branco) · $r = 0$ = nenhuma (azul)",
                 fontsize=9.0, style="italic", color="#475569", y=0.90)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def run_geometry(label: str, model_path: str, dataset_root: str,
                 grid_out: str) -> tuple[list[dict], dict]:
    regime_data: dict = {}
    rng_seed = SEED + 9999
    for dist in TRAINED_DISTS:
        for curr in ALL_CURRS:
            print(f"[{label}] {dist:g}m {curr}mA", flush=True)
            try:
                X_full, Y_full = _find_xy(dataset_root, dist, curr)
            except FileNotFoundError as e:
                print(f"  skip: {e}")
                continue
            Xv, Yv = _val_slice(X_full, Y_full)
            rng = np.random.default_rng(rng_seed); rng_seed += 1
            Yp = _load_model_and_predict(model_path, Xv, dist, curr)
            Ya = _matched_awgn(Xv, Yv, rng)
            regime_data[(dist, curr)] = (Xv, Yv, Yp, Ya)

    return _make_grid(label, regime_data, grid_out)


def _save_coefs_csv(coefs: list[dict], path: str) -> None:
    if not coefs:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(coefs[0].keys()))
        w.writeheader(); w.writerows(coefs)
    print(f"saved: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fs", action="store_true")
    ap.add_argument("--skip-fc", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    fs_coefs, fs_exemplar, fc_coefs, fc_exemplar = [], {}, [], {}

    if not args.skip_fs:
        fs_coefs, fs_exemplar = run_geometry(
            "Full Square (FS)", FS_MODEL, FS_DATA,
            os.path.join(OUT_DIR, "noise_variance_vs_amplitude_grid_FS.png"))
        _save_coefs_csv(fs_coefs, os.path.join(OUT_DIR, "shot_noise_coefficients_FS.csv"))

    if not args.skip_fc:
        fc_coefs, fc_exemplar = run_geometry(
            "Full Circle (FC)", FC_MODEL, FC_DATA,
            os.path.join(OUT_DIR, "noise_variance_vs_amplitude_grid_FC.png"))
        _save_coefs_csv(fc_coefs, os.path.join(OUT_DIR, "shot_noise_coefficients_FC.csv"))

    if fs_coefs and fc_coefs:
        _make_article_fig_A(
            fs_coefs, fc_coefs, fs_exemplar, fc_exemplar,
            os.path.join(OUT_DIR, "article_fig_A_shot_noise.png"))
        _make_article_fig_B(
            fs_coefs, fc_coefs,
            os.path.join(OUT_DIR, "article_fig_B_heatmap.png"))


if __name__ == "__main__":
    sys.exit(main())
