#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Import your analysis tools
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path:
    sys.path.append(repo_root)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.config import ATLAS_BINS
from src.models import FiveParam
from src.stats import fast_bumphunter_stat

def get_observed_tval_single(base_dir, trigger, channel, cms=13000.):
    """Calculates the BumpHunter statistic observed in the raw data for a single channel."""
    mass_path = os.path.join(base_dir, "data", f"masses_{trigger}.npz")
    if not os.path.exists(mass_path):
        print(f"Error: Raw mass data not found at {mass_path}")
        return 0.0
        
    f_mass = np.load(mass_path)
    mass_matrix = f_mass['masses']
    cols_mass = list(f_mass['columns'])

    fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{channel}.json")
    try:
        with open(fitfile, "r") as j_nom:
            d_nom = json.load(j_nom)
            
            # Setup bins and centers
            fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
            v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
            c = (v_bins[:-1] + v_bins[1:]) / 2
            
            # 1. Expected Background (5-Param Model)
            bkg_func = FiveParam(cms, c, *d_nom['parameters'])
            
            # 2. Observed Data Counts
            idx = cols_mass.index(f"M{channel}")
            masses = mass_matrix[:, idx]
            valid_masses = masses[masses > 0] * cms
            data_counts, _ = np.histogram(valid_masses, bins=v_bins)
            
            # Calculate stat
            if np.sum(bkg_func) > 0 and np.sum(data_counts) > 0:
                return fast_bumphunter_stat(data_counts, bkg_func)
                
    except Exception as e:
        print(f"Error calculating observed stat: {e}")
        return 0.0
        
    return 0.0

def main():
    parser = argparse.ArgumentParser(description="Plot Test Statistic Distribution for a Single Channel")
    parser.add_argument('--trigger', type=str, default='t2', help="Trigger stream (e.g., t1, t2)")
    parser.add_argument('--channel', type=str, default='jj', help="Channel (e.g., jj, jb)")
    args = parser.parse_args()

    trigger = args.trigger.lower()
    channel = args.channel.lower()

    # For a single channel, we only compare Naive vs Bootstrap
    methods = {
        "naive": {"color": "red", "label": "Naive (Independent Poisson)"},
        "poisson_bootstrap": {"color": "blue", "label": "Poisson Bootstrap (Data Resampled)"}
    }
                
    plt.figure(figsize=(10, 7))

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Calculating observed t_val for {trigger}_{channel}...")
    t_obs = get_observed_tval_single(base_dir, trigger, channel)
    print(f"Observed t_val: {t_obs:.2f}")

    for m, settings in methods.items():
        filename = os.path.join(results_dir, f"single_stat_{trigger}_{channel}_{m}.npy")
        
        if not os.path.exists(filename):
            print(f"Warning: Data file not found for {m} ({filename}). Skipping.")
            continue
            
        data = np.load(filename)
        
        plt.hist(data, bins=50, histtype='step',
                 label=f"{settings['label']}",
                 color=settings['color'], linewidth=2.5, density=True)

    if t_obs > 0:
        plt.axvline(t_obs, color='black', linestyle='--', linewidth=2.5, 
                    label=rf'Observed Data ($t_{{obs}} = {t_obs:.1f}$)')

    plt.yscale('log')
    plt.xlabel(r'Test Statistic ($t_{max}$)', fontsize=16)
    plt.ylabel('Probability Density', fontsize=16)
    plt.title(f'Single Channel Toy Distributions vs Observation - {trigger.upper()} {channel.upper()}', fontsize=16)
    
    plt.legend(fontsize=11, frameon=True, loc='upper right')

    out_filename = os.path.join(plot_dir, f"t_distribution_single_{trigger}_{channel}.png")
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {out_filename}.")

if __name__ == "__main__":
    main()
