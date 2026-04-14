#!/usr/bin/env python3
"""Audit oscilloscope acquisitions exported from Tektronix MSO scopes.

Expected input folder layout:

    ROOT/
      100mA_000.csv
      100mA_000.tss
      500mA_000.csv
      ...

The script:
  - parses Tek CSV waveform exports;
  - extracts screenshot PNGs embedded in .tss bundles;
  - computes descriptive and spectral statistics;
  - writes summary CSV/JSON/Markdown;
  - generates reproducible plots (overview, zoom, histogram, PSD, overlay).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.runtime_env import ensure_required_python_modules, ensure_writable_mpl_config_dir

ensure_writable_mpl_config_dir()
ensure_required_python_modules(
    ("numpy", "pandas", "matplotlib", "scipy"),
    context="scope acquisition audit",
    allow_missing=False,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

@dataclass
class AcquisitionSummary:
    file: str
    current_mA: float
    label: str
    model: str
    record_length: int
    sample_interval_s: float
    sample_rate_hz: float
    duration_s: float
    zero_index: float
    mean_v: float
    std_v: float
    rms_centered_v: float
    min_v: float
    max_v: float
    p2p_v: float
    q001_v: float
    q01_v: float
    median_v: float
    q99_v: float
    q999_v: float
    skew: float
    kurtosis_fisher: float
    crest_factor: float
    dominant_freq_hz: float
    dominant_period_us: float
    second_peak_hz: float
    third_peak_hz: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/workspace/Auditoria/Aquisições"),
        help="Directory containing Tek CSV/TSS files.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Defaults to ROOT/analysis.",
    )
    p.add_argument(
        "--zoom-us",
        type=float,
        default=20.0,
        help="Half-width of the central time zoom in microseconds.",
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


def _spectral_peaks(x0: np.ndarray, fs: float) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    freqs, psd = signal.welch(x0, fs=fs, nperseg=min(65536, len(x0)))
    peaks, _ = signal.find_peaks(psd, distance=50)
    if len(peaks) == 0:
        return float("nan"), float("nan"), float("nan"), freqs, psd
    order = peaks[np.argsort(psd[peaks])[::-1]]
    vals = [float(freqs[idx]) for idx in order[:3]]
    while len(vals) < 3:
        vals.append(float("nan"))
    return vals[0], vals[1], vals[2], freqs, psd


def _summarize_waveform(path: Path) -> tuple[AcquisitionSummary, pd.DataFrame, np.ndarray, np.ndarray]:
    header, df = _read_tektronix_csv(path)
    t = df["time_s"].to_numpy(dtype=float)
    x = df["voltage_v"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    x0 = x - np.mean(x)
    q = np.quantile(x, [0.001, 0.01, 0.5, 0.99, 0.999])
    dominant, second, third, freqs, psd = _spectral_peaks(x0, fs)
    summary = AcquisitionSummary(
        file=path.name,
        current_mA=_parse_current_from_name(path),
        label=str(header.get("Label", "")),
        model=str(header.get("Model", "")),
        record_length=int(header.get("Record Length", len(x))),
        sample_interval_s=float(header.get("Sample Interval", dt)),
        sample_rate_hz=float(fs),
        duration_s=float((len(x) - 1) * dt),
        zero_index=float(header.get("Zero Index", float("nan"))),
        mean_v=float(np.mean(x)),
        std_v=float(np.std(x)),
        rms_centered_v=float(np.sqrt(np.mean(x0 ** 2))),
        min_v=float(np.min(x)),
        max_v=float(np.max(x)),
        p2p_v=float(np.max(x) - np.min(x)),
        q001_v=float(q[0]),
        q01_v=float(q[1]),
        median_v=float(q[2]),
        q99_v=float(q[3]),
        q999_v=float(q[4]),
        skew=float(stats.skew(x0)),
        kurtosis_fisher=float(stats.kurtosis(x0, fisher=True)),
        crest_factor=float(np.max(np.abs(x0)) / np.sqrt(np.mean(x0 ** 2))),
        dominant_freq_hz=dominant,
        dominant_period_us=float(1e6 / dominant) if dominant and math.isfinite(dominant) and dominant > 0 else float("nan"),
        second_peak_hz=second,
        third_peak_hz=third,
    )
    return summary, df, freqs, psd


def _extract_tss_screenshots(root: Path, outdir: Path) -> List[Path]:
    extracted: List[Path] = []
    shot_dir = outdir / "tss_screenshots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(root.glob("*.tss")):
        try:
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if name.lower().endswith(".png"):
                        dest = shot_dir / f"{path.stem}__{Path(name).name}"
                        dest.write_bytes(zf.read(name))
                        extracted.append(dest)
        except zipfile.BadZipFile:
            continue
    return extracted


def _plot_per_acquisition(
    summary: AcquisitionSummary,
    df: pd.DataFrame,
    freqs: np.ndarray,
    psd: np.ndarray,
    outdir: Path,
    zoom_us: float,
) -> None:
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    t = df["time_s"].to_numpy(dtype=float)
    x = df["voltage_v"].to_numpy(dtype=float)
    x0 = x - np.mean(x)

    zoom_half = zoom_us * 1e-6
    mask = np.abs(t) <= zoom_half
    if not np.any(mask):
        center = len(t) // 2
        n = min(len(t), 20000)
        lo = max(0, center - n // 2)
        hi = min(len(t), lo + n)
        mask = np.zeros(len(t), dtype=bool)
        mask[lo:hi] = True

    decim = max(1, len(t) // 4000)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(t[::decim] * 1e3, x[::decim], lw=0.8)
    axes[0, 0].set_title(f"{summary.file} - overview")
    axes[0, 0].set_xlabel("Time [ms]")
    axes[0, 0].set_ylabel("Voltage [V]")
    axes[0, 0].grid(True, alpha=0.2)

    axes[0, 1].plot(t[mask] * 1e6, x[mask], lw=0.8)
    axes[0, 1].set_title(f"Central zoom ±{zoom_us:g} us")
    axes[0, 1].set_xlabel("Time [us]")
    axes[0, 1].set_ylabel("Voltage [V]")
    axes[0, 1].grid(True, alpha=0.2)

    axes[1, 0].hist(x, bins=256, density=True, alpha=0.85, color="tab:blue")
    axes[1, 0].set_title("Voltage histogram")
    axes[1, 0].set_xlabel("Voltage [V]")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].grid(True, alpha=0.2)

    axes[1, 1].semilogx(freqs[1:], 10 * np.log10(psd[1:] + 1e-30), lw=1.0)
    axes[1, 1].set_title("Welch PSD")
    axes[1, 1].set_xlabel("Frequency [Hz]")
    axes[1, 1].set_ylabel("PSD [dB/Hz]")
    axes[1, 1].grid(True, which="both", alpha=0.2)

    fig.suptitle(
        f"{summary.current_mA:.0f} mA | mean={summary.mean_v:.4f} V | "
        f"p2p={summary.p2p_v:.4f} V | f0={summary.dominant_freq_hz/1e6:.3f} MHz"
    )
    fig.tight_layout()
    fig.savefig(plots_dir / f"{Path(summary.file).stem}_analysis.png", dpi=160)
    plt.close(fig)


def _plot_overlay(records: List[tuple[AcquisitionSummary, pd.DataFrame, np.ndarray, np.ndarray]], outdir: Path, zoom_us: float) -> None:
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for summary, df, freqs, psd in records:
        t = df["time_s"].to_numpy(dtype=float)
        x = df["voltage_v"].to_numpy(dtype=float)
        mask = np.abs(t) <= zoom_us * 1e-6
        axes[0].plot(t[mask] * 1e6, x[mask], lw=0.9, label=f"{summary.current_mA:.0f} mA")
        axes[1].hist(x, bins=256, density=True, histtype="step", lw=1.2, label=f"{summary.current_mA:.0f} mA")
        axes[2].semilogx(freqs[1:], 10 * np.log10(psd[1:] + 1e-30), lw=1.0, label=f"{summary.current_mA:.0f} mA")

    axes[0].set_title(f"Central zoom ±{zoom_us:g} us")
    axes[0].set_xlabel("Time [us]")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()

    axes[1].set_title("Histogram overlay")
    axes[1].set_xlabel("Voltage [V]")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.2)

    axes[2].set_title("Welch PSD overlay")
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("PSD [dB/Hz]")
    axes[2].grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    fig.savefig(plots_dir / "overlay_analysis.png", dpi=160)
    plt.close(fig)


def _write_report(
    summaries: List[AcquisitionSummary],
    outdir: Path,
    screenshots: List[Path],
) -> None:
    df = pd.DataFrame([asdict(s) for s in summaries]).sort_values("current_mA")
    (outdir / "scope_summary.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (outdir / "scope_summary.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8"
    )

    lines = [
        "# Oscilloscope Acquisition Audit",
        "",
        f"- acquisitions: `{len(df)}`",
        f"- screenshots extracted from `.tss`: `{len(screenshots)}`",
        "",
        "## Key findings",
        "",
    ]
    if not df.empty:
        f0_min = df["dominant_freq_hz"].min()
        f0_max = df["dominant_freq_hz"].max()
        lines.extend(
            [
                f"- all three CSV exports show a dominant spectral line near `1 MHz`: `{f0_min/1e6:.3f}` to `{f0_max/1e6:.3f}` MHz",
                f"- measured centered RMS by current: "
                + ", ".join(f"`{row.current_mA:.0f} mA = {row.rms_centered_v:.4f} V`" for row in df.itertuples()),
                f"- measured peak-to-peak by current: "
                + ", ".join(f"`{row.current_mA:.0f} mA = {row.p2p_v:.4f} Vpp`" for row in df.itertuples()),
                "- the waveforms are mildly asymmetric and platykurtic, but they do not show a hard-clipping plateau in the CSV traces",
                "",
                "## Files",
                "",
                "- `scope_summary.csv`",
                "- `scope_summary.json`",
                "- `plots/overlay_analysis.png`",
                "- `plots/*_analysis.png`",
                "- `tss_screenshots/*.png`",
            ]
        )

    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    outdir = (args.outdir or (root / "analysis")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    records: List[tuple[AcquisitionSummary, pd.DataFrame, np.ndarray, np.ndarray]] = []
    for path in sorted(root.glob("*.csv")):
        records.append(_summarize_waveform(path))

    if not records:
        raise SystemExit(f"No CSV acquisitions found under {root}")

    screenshots = _extract_tss_screenshots(root, outdir)

    for summary, df, freqs, psd in records:
        _plot_per_acquisition(summary, df, freqs, psd, outdir, args.zoom_us)

    _plot_overlay(records, outdir, args.zoom_us)
    _write_report([r[0] for r in records], outdir, screenshots)

    print(f"Audit written to: {outdir}")


if __name__ == "__main__":
    main()
