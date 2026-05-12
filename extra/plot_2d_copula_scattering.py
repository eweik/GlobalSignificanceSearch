#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from scipy.stats import norm, chi2, t as student_t, pearsonr

# Setup paths to import local modules
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from src.config import ATLAS_BINS

def get_u_bounds(base_dir, trigger, channel, cms, mass_matrix, col_names):
    """Calculates the [u_min, u_max] uniform bounds for a specific channel's fit window."""
    idx = col_names.index(f"M{channel}")
    masses = mass_matrix[:, idx]
    valid_masses = masses[masses > 0] * cms
    
    fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{channel}.json")
    if not os.path.exists(fitfile):
        print(f"Warning: Fit file not found for {channel}. Defaulting bounds to [0, 1].")
        return 0.0, 1.0

    with open(fitfile, "r") as j_nom:
        d_nom = json.load(j_nom)
        
    fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
    v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
    actual_fmin = v_bins[0]
    actual_fmax = v_bins[-1]
    
    N_valid = len(valid_masses)
    if N_valid > 0:
        u_min = np.sum(valid_masses < actual_fmin) / N_valid
        u_max = np.sum(valid_masses <= actual_fmax) / N_valid
    else:
        u_min, u_max = 0.0, 1.0
        
    return u_min, u_max

def main():
    parser = argparse.ArgumentParser(description="Plot 2D Rank Density: 2x2 Grid with Empirical and Parametric Models")
    parser.add_argument('--trigger', type=str, required=True, help="Trigger name (e.g., t2)")
    parser.add_argument('--ch1', type=str, default='jj', help="First channel (e.g., jj)")
    parser.add_argument('--ch2', type=str, default='jb', help="Second channel (e.g., jb)")
    parser.add_argument('--cms', type=float, default=13000., help="Center of mass energy")
    parser.add_argument('--nu', type=float, default=5.0, help="Degrees of freedom for Student-t copula")
    args = parser.parse_args()

    base_dir = os.getcwd() if os.path.exists("data") and os.path.exists("fits") else repo_root

    print(f"Loading matrices for {args.trigger.upper()}...")
    mass_path = os.path.join(base_dir, "data", f"masses_{args.trigger}.npz")
    copula_path = os.path.join(base_dir, "data", f"copula_{args.trigger}.npz")
    
    if not os.path.exists(mass_path) or not os.path.exists(copula_path):
        print("Error: Could not find masses or copula npz files.")
        sys.exit(1)

    f_mass = np.load(mass_path)
    f_copula = np.load(copula_path)
    mass_matrix = f_mass['masses']
    copula_matrix = f_copula['copula']
    
    col_names = list(f_mass['columns'])
    
    if f"M{args.ch1}" not in col_names or f"M{args.ch2}" not in col_names:
        print(f"Error: Columns M{args.ch1} or M{args.ch2} not found in data.")
        sys.exit(1)

    # 1. Extract valid overlap topologies
    idx1 = col_names.index(f"M{args.ch1}")
    idx2 = col_names.index(f"M{args.ch2}")
    
    u1_all = copula_matrix[:, idx1]
    u2_all = copula_matrix[:, idx2]
    
    # Keep only events where BOTH channels physically exist
    valid_mask = (u1_all >= 0) & (u2_all >= 0)
    u1_emp = u1_all[valid_mask]
    u2_emp = u2_all[valid_mask]
    
    N_events = len(u1_emp)
    print(f"Found {N_events:,} events containing both {args.ch1.upper()} and {args.ch2.upper()}.")

    if N_events < 2:
        print("Not enough overlapping events to calculate correlations. Exiting.")
        sys.exit(0)

    # 2. Get Fit Window Bounds
    u_min1, u_max1 = get_u_bounds(base_dir, args.trigger, args.ch1, args.cms, mass_matrix, col_names)
    u_min2, u_max2 = get_u_bounds(base_dir, args.trigger, args.ch2, args.cms, mass_matrix, col_names)

    # 3. Generate Parametric Toys
    print("Generating Parametric and Independent equivalents...")
    rho_s = pd.Series(u1_emp).corr(pd.Series(u2_emp), method='spearman')
    if np.isnan(rho_s): rho_s = 0.0
    
    rho_p = 2 * np.sin(rho_s * np.pi / 6)
    cov_matrix = np.array([[1.0, rho_p], [rho_p, 1.0]])
    
    # Gaussian
    Z = np.random.multivariate_normal([0, 0], cov_matrix, size=N_events)
    u1_gauss = norm.cdf(Z[:, 0])
    u2_gauss = norm.cdf(Z[:, 1])

    # Student-t
    S_t = chi2.rvs(args.nu, size=N_events)
    T_student = Z * np.sqrt(args.nu / S_t[:, None])
    u1_student = student_t.cdf(T_student[:, 0], df=args.nu)
    u2_student = student_t.cdf(T_student[:, 1], df=args.nu)

    # Independent
    u1_indep = np.random.uniform(0, 1, size=N_events)
    u2_indep = np.random.uniform(0, 1, size=N_events)

    # 4. Plotting
    print("Plotting 2x2 Rank Densities...")
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    
    datasets = [
        (u1_emp, u2_emp, f"Empirical Copula (Data)\nGlobal $\\rho = {rho_s:.2f}$"),
        (u1_gauss, u2_gauss, f"Gaussian Copula (Toys)\nGlobal $\\rho = {rho_s:.2f}$"),
        (u1_student, u2_student, f"Student-t Copula (Toys, $\\nu={args.nu}$)\nGlobal $\\rho = {rho_s:.2f}$"),
        (u1_indep, u2_indep, "Independent Poisson (Toys)\nGlobal $\\rho = 0.00$")
    ]
    
    prof_ch1 = f"$U_{{{args.ch1}}}$"
    prof_ch2 = f"$U_{{{args.ch2}}}$"

    for i, (x, y, title) in enumerate(datasets):
        ax = axes_flat[i]
        
        # 2D Histogram (Density Map)
        ax.hist2d(x, y, bins=75, cmap='Blues', cmin=1, density=True)
        
        # Overlay Fit Window Bounding Box
        width = u_max1 - u_min1
        height = u_max2 - u_min2
        rect = Rectangle((u_min1, u_min2), width, height, 
                         linewidth=2, edgecolor='red', facecolor='none', linestyle='--', 
                         label='High-Mass Fit Region')
        ax.add_patch(rect)
        
        # Calculate Local Correlation INSIDE the Red Box
        box_mask = (x >= u_min1) & (x <= u_max1) & (y >= u_min2) & (y <= u_max2)
        x_in_box = x[box_mask]
        y_in_box = y[box_mask]
        
        if len(x_in_box) > 1 and np.std(x_in_box) > 0 and np.std(y_in_box) > 0:
            local_r, _ = pearsonr(x_in_box, y_in_box)
        else:
            local_r = 0.0
            
        # Display local r inside the red box
        text_x = u_min1 + (width * 0.05)
        text_y = u_max2 - (height * 0.05)
        ax.text(text_x, text_y, f"Local $r$ = {local_r:.2f}", 
                color='red', fontsize=14, fontweight='bold', 
                va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.3'))

        ax.set_title(title, fontsize=18, pad=15)
        
        # Format axes cleanly for 2x2 grid
        if i >= 2: ax.set_xlabel(f"{prof_ch1} (Uniform Rank)", fontsize=16)
        if i % 2 == 0: ax.set_ylabel(f"{prof_ch2} (Uniform Rank)", fontsize=16)
            
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.6)
        
        if i == 0:
            ax.legend(loc='upper left', fontsize=12)

    plt.suptitle(f"2D Rank Dependence Structure: {args.ch1.upper()} vs {args.ch2.upper()} (Trigger {args.trigger.upper()})", 
                 fontsize=24, fontweight='bold', y=1.03)
    plt.tight_layout()

    out_path = os.path.join(base_dir, "plots", f"copula_2d_scatter_2x2_{args.trigger}_{args.ch1}_{args.ch2}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Successfully saved 2x2 Scatter plot to: {out_path}")

if __name__ == "__main__":
    main()
