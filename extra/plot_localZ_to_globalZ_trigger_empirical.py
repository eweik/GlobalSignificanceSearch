#!/usr/bin/env python3
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse
import os
import json

# Setup paths to import local modules
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from src.config import ATLAS_BINS
from src.models import FiveParam
from src.stats import fast_bumphunter_stat

def main():
    parser = argparse.ArgumentParser(description="Map Local Z to Global Z and Evaluate Empirical Data.")
    parser.add_argument("--ExpectedLocalZvalue", type=float, default=5.0,
                        help="Expected local significance (default: 5.0)")
    parser.add_argument("--trigger", type=str, default="t2",
                        help="Trigger to analyze (default: t2)")
    parser.add_argument("--bkg", choices=["func", "matrix"], default="func",
                        help="Background model used for the pseudo-experiments (func or matrix)")
    parser.add_argument("--cms", type=float, default=13000., 
                        help="Center of mass energy (default: 13000.)")
    args = parser.parse_args()

    ExpectedLocalZvalue = args.ExpectedLocalZvalue
    trigger = args.trigger
    bkg_tag = "BKGfunc" if args.bkg == "func" else "BKGmatrix"

    methods = ["naive", "copula", "poisson_event", "decorrelated_bootstrap"]
    colors = {"naive": "red", "linear": "blue", "copula": "orange",
              "poisson_event": "green", "exclusive_categories": "purple",
              "decorrelated_bootstrap": "olive"}
    method_label_map = {"naive": "Independent", "linear": "Overlap", "copula": "Copula",
                        "poisson_event": "Poisson Bootstrap", "decorrelated_bootstrap": "Decorrelated Bootstrap"}

    os.makedirs("plots", exist_ok=True)
    base_dir = os.getcwd() if os.path.exists("data") else repo_root

    print(f"\n############## START ################")
    print(f"Searching for bumps with Local Z >= {ExpectedLocalZvalue}")
    print(f"Trigger: {trigger.upper()} | Background Model: {bkg_tag}")

    # --- 1. CALCULATE OBSERVED EMPIRICAL DATA STATISTIC ---
    t_data_max = 0.0
    z_data_local = -np.inf
    mass_path = os.path.join(base_dir, "data", f"masses_{trigger}.npz")

    if os.path.exists(mass_path):
        print("\nEvaluating raw data to find observed empirical significance...")
        f_mass = np.load(mass_path)
        mass_matrix = f_mass['masses']
        col_names = list(f_mass['columns'])
        mass_types = ["jj", "bb", "jb", "je", "jm", "jg", "be", "bm", "bg"]

        for m in mass_types:
            fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{m}.json")
            if not os.path.exists(fitfile): continue

            with open(fitfile, "r") as f: d_nom = json.load(f)

            fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
            v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
            c = (v_bins[:-1] + v_bins[1:]) / 2

            idx = col_names.index(f"M{m}")
            valid_masses = mass_matrix[mass_matrix[:, idx] > 0, idx] * args.cms
            data_counts, _ = np.histogram(valid_masses, bins=v_bins)

            if args.bkg == "func":
                bkg_counts = FiveParam(args.cms, c, *d_nom['parameters'])
                bkg_counts = np.maximum(bkg_counts, 1e-15)
            else:
                bkg_counts = np.maximum(data_counts.copy(), 1e-15)

            if np.sum(data_counts) < 50: continue

            t_ch = fast_bumphunter_stat(data_counts, bkg_counts)
            t_data_max = max(t_data_max, t_ch)

        if t_data_max > 0:
            p_data_local = np.exp(-t_data_max)
            p_data_local = np.clip(p_data_local, 1e-300, 0.999999)
            z_data_local = stats.norm.isf(p_data_local)
            print(f"Observed Maximum Local Test Statistic (t): {t_data_max:.3f}")
            print(f"Observed Maximum Local Z-score: {z_data_local:.3f}σ")
        else:
            print("Data perfectly matches background (t=0). Local Z is negligible.")
    else:
        print(f"Warning: Mass matrix not found at {mass_path}. Cannot compute empirical data point.")

    # --- 2. EVALUATE TOY METHODS ---
    plt.figure(figsize=(11, 7))

    for method in methods:
        file_pattern = f"results/global_stat_{trigger}_{method}_*_{bkg_tag}.npy"
        file_list = glob.glob(file_pattern)
        
        if not file_list:
            file_list = glob.glob(f"results/merged/final_{trigger}_{method}_{bkg_tag}.npy")
            if not file_list:
                file_list = glob.glob(f"results/merged_5param/final_{trigger}_{method}.npy")
                if not file_list:
                    print(f"Warning: No data found for {method} with tag {bkg_tag}. Skipping.")
                    continue

        arrays = [np.load(f) for f in file_list]
        t_max_dist = np.concatenate(arrays)
        t_max_dist = t_max_dist[np.isfinite(t_max_dist)]
        MaxEvents = len(t_max_dist)

        if MaxEvents == 0: continue

        # Local Z conversions for plotting
        p_local_dist = np.exp(-t_max_dist)
        p_local_dist = np.clip(p_local_dist, 1e-300, 0.999999)
        z_local_dist = stats.norm.isf(p_local_dist)

        # Target Expected Stats
        NrFound = np.sum(z_local_dist >= ExpectedLocalZvalue)
        p_global = NrFound / MaxEvents
        Z_global = stats.norm.isf(p_global) if p_global > 0 else np.inf 

        # Empirical Observed Stats
        emp_p_global = np.sum(t_max_dist >= t_data_max) / MaxEvents
        emp_Z_global = stats.norm.isf(emp_p_global) if emp_p_global > 0 else np.inf

        print(f"\n###### RESULT: {method.upper()} ######")
        print(f" Total pseudo-experiments = {MaxEvents}")
        print(f" --- Target (Local Z = {ExpectedLocalZvalue}) ---")
        print(f" Toys >= Target: {NrFound}")
        if p_global > 0:
            print(f" Expected Global p-value = {p_global:.2e}  (Global Z = {Z_global:.2f})")
        else:
            print(f" Expected Global p-value = < {1/MaxEvents:.2e}")
            
        if t_data_max > 0:
            emp_toys_found = np.sum(t_max_dist >= t_data_max)
            print(f" --- Empirical Data (Local Z = {z_data_local:.2f}) ---")
            print(f" Toys >= Observed Data: {emp_toys_found}")
            if emp_p_global > 0:
                print(f" Empirical Global p-value = {emp_p_global:.2e}  (Global Z = {emp_Z_global:.2f})")
            else:
                print(f" Empirical Global p-value = < {1/MaxEvents:.2e} (Exceeds toy limits)")
        print(f"###### END RESULT ######")

        # Prepare the Survival Curve Map
        z_local_sorted = np.sort(z_local_dist)[::-1]
        ranks = np.arange(1, MaxEvents + 1)
        p_global_curve = ranks / MaxEvents
        z_global_curve = stats.norm.isf(p_global_curve)

        valid = (z_global_curve > -10) & np.isfinite(z_global_curve)
        
        label_name = method_label_map.get(method, method.capitalize())
        c_color = colors.get(method, "black")
        
        # Plot Curve
        plt.plot(z_local_sorted[valid], z_global_curve[valid],
                 label=f"{label_name} (N={MaxEvents})", color=c_color, lw=2)
                 
        # Plot Empirical Point Intersection
        if z_data_local > 0 and emp_Z_global < np.inf:
            plt.plot(z_data_local, emp_Z_global, marker='o', color=c_color, markersize=8, zorder=5)

    # --- 3. FORMAT THE PLOT ---
    bkg_display = "5-param" if args.bkg == "func" else "Raw Bin Counts"
    plt.title(f"Trigger-Wide Global Significance vs. BumpHunter Significance\n{trigger.upper()} | {bkg_display} Background", fontsize=15, fontweight='bold')
    plt.xlabel("Highest Observed BumpHunter Significance Across All Mass Channels ($Z_{local}$)", fontsize=12)
    plt.ylabel("Global Significance ($Z_{global}$)", fontsize=12)
    
    # Discovery thresholds
    plt.axhline(3, color='grey', linestyle='--', alpha=0.7, label='3σ Global Evidence')
    plt.axhline(5, color='black', linestyle=':', alpha=0.7, label='5σ Global Discovery')

    # Empirical Data Line
    if z_data_local > 0:
        plt.axvline(z_data_local, color='magenta', linestyle='-.', lw=2, alpha=0.8, label=f'Observed Data ($Z_{{local}}={z_data_local:.2f}$)')

    # No LEE baseline
    lims = [max(0, plt.xlim()[0]), min(8, plt.xlim()[1])]
    plt.plot(lims, lims, 'k--', alpha=0.3, label="No LEE ($Z_{global} = Z_{local}$)")

    plt.legend(loc="lower right", fontsize=10, framealpha=0.9, edgecolor='black')
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plot_out = f"plots/Local_vs_Global_Z_{trigger}_{bkg_tag}_Empirical.png"
    plt.savefig(plot_out, dpi=300)
    print(f"\nPlot saved to {plot_out}")

if __name__ == "__main__":
    main()
