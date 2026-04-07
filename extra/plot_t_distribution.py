import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Import your analysis tools
# Adjust these imports if your directory structure requires sys.path.append()
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path:
    sys.path.append(repo_root)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.config import ATLAS_BINS
from src.models import FiveParam
from src.stats import fast_bumphunter_stat

def get_observed_tmax(base_dir, trigger, cms=13000.):
    """Calculates the maximum BumpHunter statistic observed in the raw data against the 5-parameter fit."""
    mass_path = os.path.join(base_dir, "data", f"masses_{trigger}.npz")
    if not os.path.exists(mass_path):
        print(f"Error: Raw mass data not found at {mass_path}")
        return 0.0
        
    f_mass = np.load(mass_path)
    mass_matrix = f_mass['masses']
    cols_mass = list(f_mass['columns'])

    mass_types = ["jj", "bb", "jb", "je", "jm", "jg", "be", "bm", "bg"]
    t_obs_max = 0.0

    for m in mass_types:
        fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{m}.json")
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
                idx = cols_mass.index(f"M{m}")
                masses = mass_matrix[:, idx]
                valid_masses = masses[masses > 0] * cms
                data_counts, _ = np.histogram(valid_masses, bins=v_bins)
                
                # Calculate stat if both exist
                if np.sum(bkg_func) > 0 and np.sum(data_counts) > 0:
                    t_val = fast_bumphunter_stat(data_counts, bkg_func)
                    t_obs_max = max(t_obs_max, t_val)
                    
        except Exception as e:
            continue
            
    return t_obs_max

def main():
    parser = argparse.ArgumentParser(description="Plot Global Significance Test Statistics")
    parser.add_argument('--trigger', type=str, default='t2', help="Trigger stream (e.g., t1, t2)")
    args = parser.parse_args()

    trigger = args.trigger.lower()

    methods = {
        "naive": {"color": "red", "label": "Naive (Independent)"},
        # "copula": {"color": "green", "label": "Empirical Copula (Migrated)"},
        "poisson_event": {"color": "blue", "label": "Poisson Bootstrap"},
        "decorrelated_bootstrap": {"color": "orange", "label": "Decorrelated Bootstrap"}
    }
                
    plt.figure(figsize=(10, 7))

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results/merged_5param")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # --- NEW: Calculate observed max t-value ---
    print("Calculating observed t_max from raw data...")
    t_obs = get_observed_tmax(base_dir, trigger)
    print(f"Observed global t_max: {t_obs:.2f}")

    for m, settings in methods.items():
        # Adjust file naming convention if yours differs slightly
        filename = os.path.join(results_dir, f"global_stat_{trigger}_{m}_local_NOFIT_BKGfunc.npy")
        if not os.path.exists(filename):
            # Fallback to the original naming convention in your script
            filename = os.path.join(results_dir, f"final_{trigger}_{m}.npy")
            
        if not os.path.exists(filename):
            print(f"Warning: Data file not found for {m}. Skipping.")
            continue
            
        data = np.load(filename)
        
        plt.hist(data, bins=50, histtype='step',
                 label=f"{settings['label']}",
                 color=settings['color'], linewidth=2.5, density=True)

    # --- NEW: Plot the vertical line for the observed data ---
    if t_obs > 0:
        plt.axvline(t_obs, color='black', linestyle='--', linewidth=2.5, 
                    label=rf'Observed Data ($t_{{max}} = {t_obs:.1f}$)')

    plt.yscale('log')
    plt.xlabel(r'Global Test Statistic ($t_{max}$)', fontsize=16)
    plt.ylabel('Probability Density', fontsize=16)
    plt.title(f'Global Significance Toy Distributions vs Observation - {trigger.upper()}', fontsize=18)
    
    # Put legend outside if it overlaps with the data
    plt.legend(fontsize=11, frameon=True, loc='upper right')

    out_filename = os.path.join(plot_dir, f"t_distribution_{trigger}_with_obs.png")
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {out_filename}.")

if __name__ == "__main__":
    main()
