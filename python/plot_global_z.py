#!/usr/bin/env python3
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def main():
    os.makedirs("plots", exist_ok=True)
    trigger = "t1"
    methods = ["naive", "linear", "copula"]
    colors = {"naive": "red", "linear": "blue", "copula": "green"}

    plt.figure(figsize=(10, 6))

    print(f"\n{'='*65}")
    print(f" Global Z-Score Thresholds for {trigger.upper()} (Empirical)")
    print(f"{'='*65}")
    print(f"{'Method':<10} | {'t_obs required for Global: 1σ, 2σ, 3σ, 4σ':<45}")
    print("-" * 65)

    for method in methods:
        # Load all chunked arrays for this method
        file_list = glob.glob(f"results/global_stat_{trigger}_{method}_*.npy")
        if not file_list:
            continue
            
        arrays = [np.load(f) for f in file_list]
        t_max_dist = np.concatenate(arrays)
        t_max_dist = t_max_dist[np.isfinite(t_max_dist)]
        
        total_toys = len(t_max_dist)
        if total_toys == 0:
            continue
            
        # 1. Sort from highest t_max to lowest
        t_max_sorted = np.sort(t_max_dist)[::-1]
        
        # 2. Calculate empirical Global p-value for every single toy
        ranks = np.arange(1, total_toys + 1)
        p_global = ranks / total_toys
        
        # 3. Convert Global p-value to Global Z-score
        z_global = norm.isf(p_global)
        
        # Plot only valid, non-infinite Z-scores
        valid = (z_global > 0) & np.isfinite(z_global)
        plt.plot(t_max_sorted[valid], z_global[valid], 
                 label=f"{method.capitalize()} (N={total_toys})", 
                 color=colors[method], lw=2)
        
        # Calculate specific thresholds for the table
        z_targets = [1, 2, 3, 4]
        thresholds = []
        for z in z_targets:
            p_target = norm.sf(z)
            # Check if we have enough toys to resolve this Z-score
            if p_target >= (1.0 / total_toys):
                idx = np.searchsorted(p_global, p_target)
                thresh = t_max_sorted[idx]
                thresholds.append(f"{thresh:>6.2f}")
            else:
                thresholds.append("   N/A")
                
        print(f"{method.capitalize():<10} | {', '.join(thresholds)}")

    # Format the plot
    plt.title(f"Global Significance vs. Local Test Statistic ({trigger.upper()})", fontsize=14)
    plt.xlabel(r"Observed BumpHunter Test Statistic ($t_{obs}$)", fontsize=12)
    plt.ylabel(r"Global Significance ($Z_{global}$)", fontsize=12)
    
    # Add reference lines for standard discovery thresholds
    plt.axhline(3, color='grey', linestyle='--', alpha=0.7, label='3σ Evidence')
    plt.axhline(5, color='black', linestyle=':', alpha=0.7, label='5σ Discovery')
    
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plot_out = f"plots/Global_Z_Mapping_{trigger}.png"
    plt.savefig(plot_out, dpi=300)
    print(f"{'-'*65}\nPlot saved to {plot_out}\n")

if __name__ == "__main__":
    main()
