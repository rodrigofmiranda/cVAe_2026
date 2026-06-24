#!/usr/bin/env python3
"""Plot BER fidelity (absolute difference from Real): |BER_model - BER_real|.

Reads the generated CSV tables and plots the absolute discrepancy of cVAE FC,
cVAE FS, and AWGN against the Real channel BER. Grid layout: rows = distances,
columns = modulations (matches ber_comparison.png layout).
"""
import argparse
import csv
import glob
import os
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing the ber_table_*.csv files")
    ap.add_argument("--out", required=True, help="Output image file path (e.g. ber_fidelity.png)")
    ap.add_argument("--title", default="Fidelidade de BER: Discrepância Absoluta contra o Real")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Find all CSV files
    csv_files = glob.glob(os.path.join(args.dir, "**/ber_table_*.csv"), recursive=True)
    if not csv_files:
        csv_files = glob.glob(os.path.join(args.dir, "ber_table_*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {args.dir}")
        return

    print(f"Found {len(csv_files)} CSV files. Parsing for BER fidelity...")

    # Load all data
    data = []
    for fpath in csv_files:
        with open(fpath, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                data.append({
                    "label": r["label"],
                    "modulation": r["modulation"],
                    "dist_m": float(r["dist_m"]),
                    "curr_mA": int(float(r["curr_mA"])),
                    "ber_real": float(r["ber_real"]),
                    "ber_twin": float(r["ber_twin"]),
                    "ber_awgn": float(r.get("ber_awgn", 0.0))
                })

    mods = sorted(list({d["modulation"] for d in data}))
    dists = sorted(list({d["dist_m"] for d in data}))
    
    nr, nc = len(dists), len(mods)  # Rows = distances, Cols = modulations
    
    # We will generate both shared scale and adaptive scale plots for fidelity
    for mode in ["shared", "adaptive"]:
        fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.6 * nr), sharex=True, squeeze=False)
        
        colors = {
            "cVAE_FC": "#e66101",   # Orange
            "cVAE_FS": "#5e3c99",   # Purple
            "AWGN": "#7f7f7f"       # Grey
        }
        
        floor = 1e-6

        for r_idx, d in enumerate(dists):
            for c_idx, m in enumerate(mods):
                ax = axes[r_idx, c_idx]
                
                # Extract data for this cell
                cell_data = [x for x in data if x["modulation"] == m and x["dist_m"] == d]
                if not cell_data:
                    ax.text(0.5, 0.5, "Sem dados", ha="center", va="center")
                    continue
                
                currents = sorted(list({x["curr_mA"] for x in cell_data}))
                
                err_fc = []
                err_fs = []
                err_awgn = []
                
                for curr in currents:
                    sub = [x for x in cell_data if x["curr_mA"] == curr]
                    if not sub:
                        continue
                    
                    # Real BER (reference)
                    real_val = np.mean([x["ber_real"] for x in sub])
                    
                    # cVAE FC absolute error
                    fc_val = [x["ber_twin"] for x in sub if x["label"] == "FC"]
                    err_fc.append(abs(fc_val[0] - real_val) if fc_val else 0.0)
                    
                    # cVAE FS absolute error
                    fs_val = [x["ber_twin"] for x in sub if x["label"] == "FS"]
                    err_fs.append(abs(fs_val[0] - real_val) if fs_val else 0.0)
                    
                    # AWGN absolute error
                    awgn_val = np.mean([x["ber_awgn"] for x in sub])
                    err_awgn.append(abs(awgn_val - real_val))

                y_fc = np.array(err_fc)
                y_fs = np.array(err_fs)
                y_awgn = np.array(err_awgn)

                # Clamped values for log-scale plotting
                p_fc = np.clip(y_fc, floor, None)
                p_fs = np.clip(y_fs, floor, None)
                p_awgn = np.clip(y_awgn, floor, None)

                # Plot lines
                ax.plot(currents, p_fc, marker="s", ms=4, color=colors["cVAE_FC"], lw=1.3, label="cVAE FC")
                ax.plot(currents, p_fs, marker="^", ms=4, color=colors["cVAE_FS"], lw=1.3, label="cVAE FS")
                ax.plot(currents, p_awgn, marker="x", ms=4, color=colors["AWGN"], lw=1.1, ls="--", label="AWGN")
                
                # Plot reference zero-error baseline at the floor
                ax.axhline(floor, color="#1a1a1a", lw=0.8, ls=":", alpha=0.5)
                
                ax.set_yscale("log")
                ax.grid(True, which="both", ls=":", alpha=0.45)
                
                # Determine Y-limits
                if mode == "shared":
                    ax.set_ylim(5e-7, 0.2)
                else:
                    # Adaptive scale
                    all_vals = np.concatenate([y_fc, y_fs, y_awgn])
                    nonzero = all_vals[all_vals > 0.0]
                    if len(nonzero) == 0:
                        ax.set_ylim(5e-7, 1e-5)
                    else:
                        max_val = np.max(nonzero)
                        min_val = np.min(nonzero)
                        y_max = min(max_val * 2.0, 0.2)
                        y_min = max(min_val * 0.4, 5e-7)
                        if y_max <= y_min:
                            y_max = y_min * 10.0
                        ax.set_ylim(y_min, y_max)
                
                # Subplot titles
                ax.set_title(f"{d:g} m — {m}", fontsize=10, fontweight="bold")
                
                if c_idx == 0:
                    ax.set_ylabel("|BER_modelo - BER_real| (Log)", fontsize=9, fontweight="bold")
                if r_idx == nr - 1:
                    ax.set_xlabel("Corrente (mA)", fontsize=10)
                
                # Add legend only in top-left panel
                if r_idx == 0 and c_idx == 0:
                    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

        suffix = " (Escala Compartilhada)" if mode == "shared" else " (Escalas Adaptáveis)"
        fig.suptitle(args.title + suffix, fontsize=13, fontweight="bold", y=0.995)
        plt.tight_layout()
        
        if mode == "shared":
            out_path = args.out
        else:
            base, ext = os.path.splitext(args.out)
            out_path = f"{base}_adaptive{ext}"
            
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"Fidelity plot saved to {out_path}")

if __name__ == "__main__":
    main()
