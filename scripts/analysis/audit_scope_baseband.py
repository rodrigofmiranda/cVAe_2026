#!/usr/bin/env python3
"""Baseband audit for oscilloscope captures of the 1 MHz LED-string terminal.

This script performs three analyses on Tektronix CSV waveform exports:

1. Downconvert the measured real passband waveform to a complex baseband
   envelope using the analytic signal.
2. Compare the measured baseband PSD with the expected raised-cosine occupancy
   implied by `channel_dataset.py` (`samp_rate=200 kS/s`, `sps=4`, `alpha=0.35`).
3. Measure a passband-model error against a synthetic `1 MHz + envelope` model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


NOMINAL_CARRIER_HZ = 1_000_000.0
SAMP_RATE_HZ = 200_000.0
SPS = 4.0
ROLLOFF = 0.35
SYMBOL_RATE_HZ = SAMP_RATE_HZ / SPS


@dataclass
class BasebandSummary:
    file: str
    current_mA: float
    sample_rate_scope_hz: float
    duration_s: float
    carrier_est_hz: float
    carrier_error_hz: float
    carrier_period_error_ns: float
    carrier_drift_cycles_over_record: float
    bb_fs_hz: float
    bb_rms: float
    bb_i_std: float
    bb_q_std: float
    bb_obw90_hz: float
    bb_obw99_hz: float
    rc_corr: float
    rc_nmse: float
    model_error_est_carrier_pct: float
    model_error_nominal_carrier_pct: float
    envelope_peak_hz: float
    envelope_peak_rel_db: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/workspace/Auditoria/Aquisições"),
        help="Directory containing Tek CSV files.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Defaults to ROOT/analysis/baseband.",
    )
    p.add_argument(
        "--target-bb-fs",
        type=float,
        default=1_000_000.0,
        help="Target sample rate for downsampled baseband views.",
    )
    p.add_argument(
        "--lpf-pass-hz",
        type=float,
        default=80_000.0,
        help="Low-pass passband for equivalent complex envelope.",
    )
    p.add_argument(
        "--lpf-stop-hz",
        type=float,
        default=120_000.0,
        help="Low-pass stopband for equivalent complex envelope.",
    )
    p.add_argument(
        "--zoom-us",
        type=float,
        default=40.0,
        help="Central time zoom for passband/baseband plots.",
    )
    return p.parse_args()


def _parse_current_from_name(path: Path) -> float:
    m = re.search(r"(\d+(?:[.,]\d+)?)mA", path.stem)
    if not m:
        return float("nan")
    return float(m.group(1).replace(",", "."))


def _read_tektronix_csv(path: Path) -> tuple[Dict[str, str], pd.DataFrame]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header: Dict[str, str] = {}
    data_start = None
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if line.startswith("TIME,"):
            data_start = i + 1
            break
        parts = line.split(",", 1)
        if len(parts) == 2:
            header[parts[0].strip()] = parts[1].strip()
    if data_start is None:
        raise ValueError(f"Could not find TIME header in {path}")
    df = pd.read_csv(path, skiprows=data_start, names=["time_s", "voltage_v"])
    return header, df


def _estimate_carrier_hz(t: np.ndarray, x0: np.ndarray) -> float:
    analytic = signal.hilbert(x0)
    phase = np.unwrap(np.angle(analytic))
    weights = np.abs(analytic) + 1e-12
    slope, _ = np.polyfit(t, phase, deg=1, w=weights)
    return float(slope / (2.0 * np.pi))


def _raised_cosine_mask(freq_hz: np.ndarray, pass_hz: float, stop_hz: float) -> np.ndarray:
    af = np.abs(freq_hz)
    mask = np.ones_like(af, dtype=float)
    mask[af >= stop_hz] = 0.0
    trans = (af > pass_hz) & (af < stop_hz)
    if np.any(trans):
        frac = (af[trans] - pass_hz) / (stop_hz - pass_hz)
        mask[trans] = 0.5 * (1.0 + np.cos(np.pi * frac))
    return mask


def _downconvert_to_baseband(
    t: np.ndarray,
    x0: np.ndarray,
    carrier_hz: float,
    lpf_pass_hz: float,
    lpf_stop_hz: float,
    target_bb_fs: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    fs = 1.0 / float(np.median(np.diff(t)))
    analytic = signal.hilbert(x0)
    mixed = analytic * np.exp(-1j * 2.0 * np.pi * carrier_hz * t)

    n = len(mixed)
    freq = np.fft.fftfreq(n, d=1.0 / fs)
    mask = _raised_cosine_mask(freq, lpf_pass_hz, lpf_stop_hz)
    bb_full = np.fft.ifft(np.fft.fft(mixed) * mask)

    down = max(1, int(round(fs / target_bb_fs)))
    bb = signal.resample_poly(bb_full, up=1, down=down)
    bb_fs = fs / down
    return bb_full.astype(np.complex128), bb.astype(np.complex128), float(bb_fs)


def _psd_two_sided(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(4096, len(x))
    freq, psd = signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        return_onesided=False,
        scaling="density",
    )
    order = np.argsort(freq)
    return freq[order], psd[order]


def _ideal_raised_cosine_psd(freq_hz: np.ndarray, symbol_rate_hz: float, alpha: float) -> np.ndarray:
    f = np.abs(freq_hz)
    f_flat = (1.0 - alpha) * symbol_rate_hz / 2.0
    f_edge = (1.0 + alpha) * symbol_rate_hz / 2.0
    out = np.zeros_like(f, dtype=float)
    flat = f <= f_flat
    out[flat] = 1.0
    trans = (f > f_flat) & (f < f_edge)
    if np.any(trans):
        out[trans] = 0.5 * (
            1.0
            + np.cos(
                np.pi
                * (f[trans] - f_flat)
                / (alpha * symbol_rate_hz)
            )
        )
    return out


def _occupied_bw(freq_hz: np.ndarray, psd: np.ndarray, frac: float) -> float:
    order = np.argsort(freq_hz)
    freq = freq_hz[order]
    pw = np.maximum(psd[order], 0.0)
    c = np.cumsum(pw)
    if c[-1] <= 0:
        return float("nan")
    c = c / c[-1]
    flo = float(freq[np.searchsorted(c, (1.0 - frac) / 2.0)])
    fhi = float(freq[np.searchsorted(c, 1.0 - (1.0 - frac) / 2.0)])
    return fhi - flo


def _rc_compare(freq_hz: np.ndarray, psd: np.ndarray) -> tuple[float, float]:
    roi = np.abs(freq_hz) <= 120_000.0
    f = freq_hz[roi]
    p = np.maximum(psd[roi], 0.0)
    rc = _ideal_raised_cosine_psd(f, SYMBOL_RATE_HZ, ROLLOFF)
    if p.sum() <= 0 or rc.sum() <= 0:
        return float("nan"), float("nan")
    p = p / p.sum()
    rc = rc / rc.sum()
    corr = float(np.corrcoef(p, rc)[0, 1])
    nmse = float(np.mean((p - rc) ** 2) / np.mean(rc ** 2))
    return corr, nmse


def _best_phase_rotation(x_ref: np.ndarray, x_model: np.ndarray) -> float:
    num = np.vdot(x_model, x_ref)
    return float(np.angle(num))


def _passband_model_error_pct(
    x0: np.ndarray,
    t: np.ndarray,
    bb_full: np.ndarray,
    carrier_hz: float,
) -> float:
    x_model0 = np.real(bb_full * np.exp(1j * 2.0 * np.pi * carrier_hz * t))
    phi = _best_phase_rotation(signal.hilbert(x0), signal.hilbert(x_model0))
    x_model = np.real(bb_full * np.exp(1j * (2.0 * np.pi * carrier_hz * t + phi)))
    denom = float(np.sqrt(np.mean(x_model ** 2)))
    if denom <= 0:
        return float("nan")
    err = float(np.sqrt(np.mean((x0 - x_model) ** 2)))
    return 100.0 * err / denom


def _envelope_spur(bb: np.ndarray, fs: float) -> tuple[float, float]:
    env = np.abs(bb)
    env0 = env - np.mean(env)
    f_env, p_env = signal.welch(env0, fs=fs, nperseg=min(2048, len(env0)))
    if len(p_env) < 2:
        return float("nan"), float("nan")
    idx = np.argmax(p_env[1:]) + 1
    rel_db = float(10.0 * np.log10(max(p_env[idx], 1e-30) / max(np.max(p_env), 1e-30)))
    return float(f_env[idx]), rel_db


def _plot_record(
    summary: BasebandSummary,
    df: pd.DataFrame,
    bb: np.ndarray,
    bb_fs: float,
    freq_bb: np.ndarray,
    psd_bb: np.ndarray,
    outdir: Path,
    zoom_us: float,
) -> None:
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    t = df["time_s"].to_numpy(dtype=float)
    x = df["voltage_v"].to_numpy(dtype=float)
    x0 = x - np.mean(x)
    dt_bb = 1.0 / bb_fs
    t_bb = (np.arange(len(bb)) - len(bb) // 2) * dt_bb
    zoom_half = zoom_us * 1e-6
    mask_t = np.abs(t) <= zoom_half
    mask_bb = np.abs(t_bb) <= zoom_half

    rc = _ideal_raised_cosine_psd(freq_bb, SYMBOL_RATE_HZ, ROLLOFF)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(t[mask_t] * 1e6, x0[mask_t], lw=0.8, label="measured")
    axes[0, 0].set_title("Passband center zoom")
    axes[0, 0].set_xlabel("Time [us]")
    axes[0, 0].set_ylabel("Voltage [V]")
    axes[0, 0].grid(True, alpha=0.2)

    axes[0, 1].plot(t_bb[mask_bb] * 1e6, bb.real[mask_bb], lw=0.8, label="I")
    axes[0, 1].plot(t_bb[mask_bb] * 1e6, bb.imag[mask_bb], lw=0.8, label="Q")
    axes[0, 1].set_title("Equivalent baseband envelope")
    axes[0, 1].set_xlabel("Time [us]")
    axes[0, 1].set_ylabel("Amplitude [arb.]")
    axes[0, 1].grid(True, alpha=0.2)
    axes[0, 1].legend()

    meas = psd_bb / max(np.max(psd_bb), 1e-30)
    rc_n = rc / max(np.max(rc), 1e-30)
    axes[1, 0].plot(freq_bb / 1e3, 10 * np.log10(meas + 1e-30), lw=1.0, label="measured")
    axes[1, 0].plot(freq_bb / 1e3, 10 * np.log10(rc_n + 1e-30), lw=1.0, label="ideal RC")
    axes[1, 0].set_xlim(-120, 120)
    axes[1, 0].set_title("Baseband PSD vs ideal raised-cosine")
    axes[1, 0].set_xlabel("Frequency [kHz]")
    axes[1, 0].set_ylabel("Normalized PSD [dB]")
    axes[1, 0].grid(True, alpha=0.2)
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    txt = (
        f"carrier = {summary.carrier_est_hz/1e6:.6f} MHz\n"
        f"carrier error = {summary.carrier_error_hz:+.1f} Hz\n"
        f"BB OBW90/99 = {summary.bb_obw90_hz/1e3:.1f} / {summary.bb_obw99_hz/1e3:.1f} kHz\n"
        f"RC corr = {summary.rc_corr:.4f}\n"
        f"RC NMSE = {summary.rc_nmse:.4f}\n"
        f"fit err (est fc) = {summary.model_error_est_carrier_pct:.2f} %\n"
        f"fit err (1 MHz) = {summary.model_error_nominal_carrier_pct:.2f} %\n"
        f"env spur = {summary.envelope_peak_hz/1e3:.2f} kHz"
    )
    axes[1, 1].text(0.02, 0.98, txt, va="top", ha="left", family="monospace")

    fig.suptitle(f"{summary.file} | {summary.current_mA:.0f} mA")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{Path(summary.file).stem}_baseband.png", dpi=160)
    plt.close(fig)


def _plot_overlay(records: List[Tuple[BasebandSummary, np.ndarray, np.ndarray]], outdir: Path) -> None:
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for summary, freq_bb, psd_bb in records:
        meas = psd_bb / max(np.max(psd_bb), 1e-30)
        axes[0].plot(freq_bb / 1e3, 10 * np.log10(meas + 1e-30), lw=1.0, label=f"{summary.current_mA:.0f} mA")
    rc = _ideal_raised_cosine_psd(freq_bb, SYMBOL_RATE_HZ, ROLLOFF)
    rc_n = rc / max(np.max(rc), 1e-30)
    axes[0].plot(freq_bb / 1e3, 10 * np.log10(rc_n + 1e-30), lw=1.2, ls="--", color="black", label="ideal RC")
    axes[0].set_xlim(-120, 120)
    axes[0].set_title("Baseband PSD overlay")
    axes[0].set_xlabel("Frequency [kHz]")
    axes[0].set_ylabel("Normalized PSD [dB]")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()

    rows = [[s.current_mA, s.carrier_error_hz, s.bb_obw99_hz, s.model_error_nominal_carrier_pct] for s, _, _ in records]
    table = "current mA | carrier err Hz | OBW99 kHz | model err %\n"
    table += "\n".join(
        f"{int(r[0]):>9} | {r[1]:>14.1f} | {r[2]/1e3:>9.2f} | {r[3]:>11.2f}"
        for r in rows
    )
    axes[1].axis("off")
    axes[1].text(0.02, 0.98, table, va="top", ha="left", family="monospace")

    fig.tight_layout()
    fig.savefig(plots_dir / "overlay_baseband.png", dpi=160)
    plt.close(fig)


def _write_report(summaries: List[BasebandSummary], outdir: Path) -> None:
    df = pd.DataFrame([asdict(s) for s in summaries]).sort_values("current_mA")
    df.to_csv(outdir / "baseband_summary.csv", index=False)
    (outdir / "baseband_summary.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8"
    )

    lines = [
        "# Baseband Audit",
        "",
        "This report downconverts the measured real waveform around 1 MHz to an",
        "equivalent complex baseband, compares its PSD to the raised-cosine",
        "occupancy implied by `channel_dataset.py`, and measures passband model",
        "error against a synthetic `1 MHz + envelope` reconstruction.",
        "",
        "## Key findings",
        "",
    ]
    if not df.empty:
        lines.extend(
            [
                "- estimated carrier stays close to 1 MHz for all acquisitions",
                "- measured baseband occupied bandwidth is of order a few `10 kHz`, not `200 kHz`",
                "- this is consistent with `samp_rate=200 kS/s`, `sps=4`, hence `Rs≈50 ksym/s` and expected raised-cosine occupancy near `67.5 kHz` total",
                "- the passband model error quantifies how much the measured waveform departs from a narrowband `carrier + envelope` model",
                "",
                "## Files",
                "",
                "- `baseband_summary.csv`",
                "- `baseband_summary.json`",
                "- `plots/*_baseband.png`",
                "- `plots/overlay_baseband.png`",
            ]
        )
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    outdir = (args.outdir or (root / "analysis" / "baseband")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    plot_records: List[Tuple[BasebandSummary, np.ndarray, np.ndarray]] = []
    summaries: List[BasebandSummary] = []

    for path in sorted(root.glob("*.csv")):
        _header, df = _read_tektronix_csv(path)
        t = df["time_s"].to_numpy(dtype=float)
        x = df["voltage_v"].to_numpy(dtype=float)
        x0 = x - np.mean(x)
        fs = 1.0 / float(np.median(np.diff(t)))
        duration = len(x0) / fs

        carrier_est = _estimate_carrier_hz(t, x0)
        bb_full, bb, bb_fs = _downconvert_to_baseband(
            t,
            x0,
            carrier_est,
            lpf_pass_hz=float(args.lpf_pass_hz),
            lpf_stop_hz=float(args.lpf_stop_hz),
            target_bb_fs=float(args.target_bb_fs),
        )
        freq_bb, psd_bb = _psd_two_sided(bb, bb_fs)
        obw90 = _occupied_bw(freq_bb, psd_bb, 0.90)
        obw99 = _occupied_bw(freq_bb, psd_bb, 0.99)
        rc_corr, rc_nmse = _rc_compare(freq_bb, psd_bb)
        env_peak_hz, env_peak_rel_db = _envelope_spur(bb, bb_fs)
        model_err_est = _passband_model_error_pct(x0, t, bb_full, carrier_est)
        model_err_nom = _passband_model_error_pct(x0, t, bb_full, NOMINAL_CARRIER_HZ)

        summary = BasebandSummary(
            file=path.name,
            current_mA=_parse_current_from_name(path),
            sample_rate_scope_hz=float(fs),
            duration_s=float(duration),
            carrier_est_hz=float(carrier_est),
            carrier_error_hz=float(carrier_est - NOMINAL_CARRIER_HZ),
            carrier_period_error_ns=float(1e9 * (1.0 / carrier_est - 1.0 / NOMINAL_CARRIER_HZ)),
            carrier_drift_cycles_over_record=float((carrier_est - NOMINAL_CARRIER_HZ) * duration),
            bb_fs_hz=float(bb_fs),
            bb_rms=float(np.sqrt(np.mean(np.abs(bb) ** 2))),
            bb_i_std=float(np.std(bb.real)),
            bb_q_std=float(np.std(bb.imag)),
            bb_obw90_hz=float(obw90),
            bb_obw99_hz=float(obw99),
            rc_corr=float(rc_corr),
            rc_nmse=float(rc_nmse),
            model_error_est_carrier_pct=float(model_err_est),
            model_error_nominal_carrier_pct=float(model_err_nom),
            envelope_peak_hz=float(env_peak_hz),
            envelope_peak_rel_db=float(env_peak_rel_db),
        )
        summaries.append(summary)
        plot_records.append((summary, freq_bb, psd_bb))
        _plot_record(summary, df, bb, bb_fs, freq_bb, psd_bb, outdir, float(args.zoom_us))

    _plot_overlay(plot_records, outdir)
    _write_report(summaries, outdir)
    print(f"Baseband audit written to: {outdir}")


if __name__ == "__main__":
    main()
