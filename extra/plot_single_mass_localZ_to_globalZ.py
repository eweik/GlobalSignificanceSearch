#!/usr/bin/env python3
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Map Local Z to Histogram-Wide Z for a single channel.")
    parser.add_argument("--ExpectedLocalZvalue", type=float, default=5.0)
    parser.add_argument("--trigger", type=str, default="t2")
    parser.add_argument("--channel", type=str, default="jj", help="Channel to analyze (e.g., jj)")
    args = parser.parse_args()

    ExpectedLocalZvalue = args.ExpectedLocalZvalue
    trigger = args.trigger.lower()
    channel = args.channel.lower()

    # Only plotting the two relevant methods for a single channel
    methods = ["naive", "poisson_bootstrap"]
    colors = {"naive": "red", "poisson_bootstrap": "blue"}
    method_label_map = {"naive": "Naive (Independent Poisson)", 
                        "poisson_bootstrap": "Poisson Bootstrap (Data Resampled)"}

    os.makedirs("plots", exist_ok=True)

    print(f"\n############## START ################")
    print(f"Searching for bumps with Local Z >= {ExpectedLocalZvalue}")
    print(f"Trigger: {trigger.upper()} | Channel: {channel.upper()}")

    plt.figure(figsize=(10, 6))

    for method in methods:
        file_pattern = f"results/single_stat_{trigger}_{channel}_{method}.npy"
        file_list = glob.glob(file_pattern)
        
        if not file_list:
            print(f"Warning: No data found for {method} at {file_pattern}. Skipping.")
            continue

        arrays = [np.load(f) for f in file_list]
        t_max_dist = np.concatenate(arrays)
        t_max_dist = t_max_dist[np.isfinite(t_max_dist)]
        MaxEvents = len(t_max_dist)

        if MaxEvents == 0: continue

        p_local_dist = np.exp(-t_max_dist)
        p_local_dist = np.clip(p_local_dist, 1e-300, 0.999999)
        z_local_dist = stats.norm.isf(p_local_dist)

        NrFound = np.sum(z_local_dist >= ExpectedLocalZvalue)
        p_global = NrFound / MaxEvents
        
        if p_global > 0:
            Z_global = stats.norm.isf(p_global)
        else:
            Z_global = np.inf 

        print(f"\n###### RESULT: {method.upper()} ######")
        print(f" Total pseudo-experiments = {MaxEvents}")
        print(f" Found toys with Local Z >= {ExpectedLocalZvalue} = {NrFound}")
        print(f" Expected Local Z = {ExpectedLocalZvalue}")
        if p_global > 0:
            print(f" Found histogram-wide p-value = {p_global:.2e}  or Z = {Z_global:.2f}")
        else:
            print(f" Found histogram-wide p-value = < {1/MaxEvents:.2e}")
        print(f"###### END RESULT ######")

        z_local_sorted = np.sort(z_local_dist)[::-1]
        ranks = np.arange(1, MaxEvents + 1)
        p_global_curve = ranks / MaxEvents
        z_global_curve = stats.norm.isf(p_global_curve)

        valid = (z_global_curve > -10) & np.isfinite(z_global_curve)
        
        label_name = method_label_map.get(method, method)
        plt.plot(z_local_sorted[valid], z_global_curve[valid],
                 label=f"{label_name} (N={MaxEvents})", color=colors.get(method, "black"), lw=2)

    plt.title(f"Histogram-Wide Significance vs. Local Significance\n{trigger.upper()} | Channel: {channel.upper()}", fontsize=14)
    plt.xlabel("Highest Observed BumpHunter Significance in Histogram ($Z_{local}$)", fontsize=12)
    plt.ylabel("Histogram-Wide Significance ($Z_{hist-wide}$)", fontsize=12)
    
    plt.axhline(3, color='grey', linestyle='--', alpha=0.7, label='3σ Evidence')
    plt.axhline(5, color='black', linestyle=':', alpha=0.7, label='5σ Discovery')

    lims = [max(0, plt.xlim()[0]), min(8, plt.xlim()[1])]
    plt.plot(lims, lims, 'k--', alpha=0.3, label="No LEE ($Z_{hist-wide} = Z_{local}$)")

    plt.legend(loc="lower right")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plot_out = f"plots/Local_vs_HistWide_Z_single_{trigger}_{channel}.png"
    plt.savefig(plot_out, dpi=300)
    print(f"\nPlot saved to {plot_out}")

if __name__ == "__main__":
    main()
