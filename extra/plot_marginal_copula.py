#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Setup paths to import local modules
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from src.config import ATLAS_BINS
from src.models import FiveParam

def get_channel_data(base_dir, trigger, channel, cms, mass_matrix, col_names):
    """
    Extracts raw data, calculates the 5-param fit expectation, builds the discrete CDF,
    and generates the continuous function curve for plotting.
    """
    idx = col_names.index(f"M{channel}")
    masses = mass_matrix[:, idx]
    valid_masses = masses[masses > 0] * cms
    
    # 1. Load 5-Parameter Fit
    fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{channel}.json")
    if not os.path.exists(fitfile):
        raise FileNotFoundError(f"Fit file not found: {fitfile}")

    with open(fitfile, "r") as j_nom:
        d_nom = json.load(j_nom)
        
    fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
    v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
    c = (v_bins[:-1] + v_bins[1:]) / 2
    
    # 2. Histogram raw data into the fit bins
    data_counts, _ = np.histogram(valid_masses, bins=v_bins)
    
    # 3. Evaluate Fit at discrete bin centers (Expectations)
    expected_counts = FiveParam(cms, c, *d_nom['parameters']) 
    expected_counts = np.maximum(expected_counts, 0)
    
    # --- NEW: Evaluate exact continuous function for plotting ---
    x_dense = np.linspace(v_bins[0], v_bins[-1], 1000)
    y_dense = FiveParam(cms, x_dense, *d_nom['parameters'])
    y_dense = np.maximum(y_dense, 0)
    
    # 4. Create Discrete CDF
    cdf = np.cumsum(expected_counts) / np.sum(expected_counts)

    # 5. Calculate Phase-Space Truncation Bounds in Uniform Space
    actual_fmin = v_bins[0]
    actual_fmax = v_bins[-1]
    N_valid = len(valid_masses)
    
    if N_valid > 0:
        u_min = np.sum(valid_masses < actual_fmin) / N_valid
        u_max = np.sum(valid_masses <= actual_fmax) / N_valid
    else:
        u_min, u_max = 0.0, 1.0
        
    return data_counts, expected_counts, cdf, v_bins, c, (u_min, u_max), x_dense, y_dense

# --- FIX 1: Pass in col_names_cop to map the columns safely ---
def generate_expected_copula_marginal(copula_matrix, col_names_cop, channel, u_bounds, cdf, centers, bins):
    """
    Generates a massive sample from the copula, applies the exact pseudo-experiment
    truncation logic, and averages the result to find the expected unscaled toy marginal.
    """
    idx_cop = col_names_cop.index(f"M{channel}")
    N_data = len(copula_matrix)
    
    N_mult = 1000
    N_draw_total = N_data * N_mult
    
    # 1. Sample from the copula globally
    sampled_rows = copula_matrix[np.random.choice(N_data, size=N_draw_total, replace=True)]
    
    # 2. Extract raw uniform values using the safe index
    u_raw = sampled_rows[sampled_rows[:, idx_cop] >= 0, idx_cop]
    
    # 3. Apply phase-space cuts in uniform space
    u_min, u_max = u_bounds
    mask_in_window = (u_raw >= u_min) & (u_raw <= u_max)
    u_in_window = u_raw[mask_in_window]
    
    if len(u_in_window) == 0:
        return np.zeros(len(bins) - 1)
        
    # Jitter to break ties
    u_jittered = u_in_window + np.random.uniform(-0.0002, 0.0002, size=len(u_in_window))
    
    # 4. Transform to strictly local truncated [0, 1) space
    u_trunc = (u_jittered - u_min) / max(u_max - u_min, 1e-10)
    
    # Bound reflections
    u_trunc = np.abs(u_trunc)
    u_trunc = np.where(u_trunc >= 1.0, 1.99999 - u_trunc, u_trunc)
    
    # 5. Map to physical binned mass via discrete Inverse CDF
    idx_mapped = np.searchsorted(cdf, u_trunc)
    idx_mapped = np.clip(idx_mapped, 0, len(centers) - 1)
    m_toy = centers[idx_mapped]
    
    # Histogram the total mapped toys
    toy_counts_total, _ = np.histogram(m_toy, bins=bins)
    
    # Average back to a single unscaled dataset
    expected_toy_counts = toy_counts_total / N_mult
    
    return expected_toy_counts

def main():
    parser = argparse.ArgumentParser(description="Plot Marginal Agreement: Data vs Fit vs Copula Toys.")
    parser.add_argument('--trigger', type=str, required=True, help="Trigger name (e.g., t2)")
    parser.add_argument('--ch1', type=str, default='jj', help="First channel to plot (e.g., jj)")
    parser.add_argument('--ch2', type=str, default='jb', help="Second channel to plot (e.g., jb)")
    parser.add_argument('--cms', type=float, default=13000., help="Center of mass energy")
    args = parser.parse_args()

    base_dir = os.getcwd() if os.path.exists("data") and os.path.exists("fits") else repo_root

    print("Loading data matrices...")
    mass_path = os.path.join(base_dir, "data", f"masses_{args.trigger}.npz")
    copula_path = os.path.join(base_dir, "data", f"copula_{args.trigger}.npz")
    
    f_mass = np.load(mass_path)
    f_copula = np.load(copula_path)
    mass_matrix = f_mass['masses']
    copula_matrix = f_copula['copula']
    
    # Extract both lists to prevent mismatch errors
    col_names_mass = list(f_mass['columns'])
    col_names_cop = list(f_copula['columns'])

    print("Loading fits and calculating CDFs & Bounds...")
    d1, f1, cdf1, bins1, centers1, bounds1, x_dense1, y_dense1 = get_channel_data(base_dir, args.trigger, args.ch1, args.cms, mass_matrix, col_names_mass)
    d2, f2, cdf2, bins2, centers2, bounds2, x_dense2, y_dense2 = get_channel_data(base_dir, args.trigger, args.ch2, args.cms, mass_matrix, col_names_mass)

    print("Generating and mapping Copula toys (applying exact truncation logic)...")
    # Pass col_names_cop to the toy generator
    expected_toys1 = generate_expected_copula_marginal(copula_matrix, col_names_cop, args.ch1, bounds1, cdf1, centers1, bins1)
    expected_toys2 = generate_expected_copula_marginal(copula_matrix, col_names_cop, args.ch2, bounds2, cdf2, centers2, bins2)

    print("Generating plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    def plot_marginal(ax, channel, bins, centers, data, fit, toys, x_dense, y_dense):
        
        # 1. Plot Exact Continuous 5-Param Fit (Smooth Solid Line)
        ax.plot(x_dense, y_dense, color='dodgerblue', linewidth=2, alpha=0.6, label='5-Param Fit (Continuous Function)', zorder=2)
        
        # 2. Plot Fit Expectation Points (Squares on the line)
        ax.plot(centers, fit, color='dodgerblue', marker='s', linestyle='none', markersize=5, label='Fit Expectation (Bin Centers)', zorder=3)
        
        # 3. Plot Copula Toys as Dots
        # Adding a slight transparency and edge color to make the dots pop
        ax.plot(centers, toys, color='red', marker='o', linestyle='none', markersize=6, alpha=0.9, markeredgecolor='darkred', label='Copula Toys (Unscaled Expectation)', zorder=4)
        
        # 4. Plot Raw Data (Black Points with Error Bars)
        err = np.sqrt(data)
        err[err == 0] = 1.0 
        ax.errorbar(centers, data, yerr=err, fmt='ko', markersize=4, capsize=3, label='Raw Data', zorder=10)

        ax.set_title(f"Marginal Agreement: {channel.upper()} Channel", fontsize=15)
        ax.set_xlabel(f"${channel.upper()}$ Mass [GeV]", fontsize=14)
        ax.set_ylabel("Events / Bin", fontsize=14)
        ax.set_yscale('log')
        ax.set_xlim(bins[0], bins[-1])
        
        # Dynamic Y-limits
        min_y = max(0.1, np.min(data[data > 0]) * 0.5)
        max_y = np.max(data) * 5.0
        ax.set_ylim(min_y, max_y)
        
        ax.legend(fontsize=11, frameon=True, edgecolor='black', framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    # Plot Ch1
    plot_marginal(axes[0], args.ch1, bins1, centers1, d1, f1, expected_toys1, x_dense1, y_dense1)
    # Plot Ch2
    plot_marginal(axes[1], args.ch2, bins2, centers2, d2, f2, expected_toys2, x_dense2, y_dense2)

    # --- FIX 2: Apply fontweight and tight_layout rect ---
    plt.suptitle(f"Copula Marginal Fidelity Validation | Trigger: {args.trigger.upper()}", fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    out_path = os.path.join(base_dir, "plots", f"marginal_agreement_dots_{args.trigger}_{args.ch1}_{args.ch2}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved marginal agreement plot to: {out_path}")

if __name__ == "__main__":
    main()
