#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Setup paths to import local modules (matching your main script)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from src.config import ATLAS_BINS
from src.models import FiveParam

def get_discrete_cdf_and_bins(base_dir, trigger, channel, cms):
    """
    Loads the 5-parameter fit, returns the discrete CDF, bin edges, and bin centers.
    This exactly mirrors your pseudo-experiment logic.
    """
    fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{channel}.json")
    
    if not os.path.exists(fitfile):
        raise FileNotFoundError(f"Fit file not found: {fitfile}")

    with open(fitfile, "r") as j_nom:
        d_nom = json.load(j_nom)
        
    fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
    v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
    c = (v_bins[:-1] + v_bins[1:]) / 2
    
    # 1. Evaluate the 5-Parameter Background Function
    counts_nom = FiveParam(cms, c, *d_nom['parameters']) 
    counts_nom = np.maximum(counts_nom, 0)
    
    # 2. Create the exact discrete CDF used in your toy generation
    cdf = np.cumsum(counts_nom) / np.sum(counts_nom)
        
    return cdf, v_bins, c, fmin_val, fmax_val

def main():
    parser = argparse.ArgumentParser(description="Visualize the Discrete Empirical Copula Transformation.")
    parser.add_argument('--trigger', type=str, required=True, help="Trigger name (e.g., t2)")
    parser.add_argument('--ch1', type=str, default='jj', help="First channel to plot (e.g., jj)")
    parser.add_argument('--ch2', type=str, default='jb', help="Second channel to plot (e.g., jb)")
    parser.add_argument('--cms', type=float, default=13000., help="Center of mass energy for scaling")
    args = parser.parse_args()

    base_dir = os.getcwd() if os.path.exists("data") and os.path.exists("fits") else repo_root

    # 1. Load the actual raw data and empirical copula
    mass_path = os.path.join(base_dir, "data", f"masses_{args.trigger}.npz")
    copula_path = os.path.join(base_dir, "data", f"copula_{args.trigger}.npz")

    if not os.path.exists(mass_path) or not os.path.exists(copula_path):
        print(f"Error: Missing data files in {os.path.join(base_dir, 'data')}")
        return

    print("Loading data matrices...")
    f_mass = np.load(mass_path)
    f_copula = np.load(copula_path)

    mass_matrix = f_mass['masses']
    copula_matrix = f_copula['copula']
    col_names = list(f_mass['columns'])

    try:
        idx1 = col_names.index(f"M{args.ch1}")
        idx2 = col_names.index(f"M{args.ch2}")
    except ValueError:
        print(f"Error: Channels not found.")
        return

    # Extract raw mass data and scale
    m1_raw = mass_matrix[:, idx1] * args.cms
    m2_raw = mass_matrix[:, idx2] * args.cms

    # Filter out empty events
    valid_mask = (m1_raw > 0) & (m2_raw > 0)
    m1_raw = m1_raw[valid_mask]
    m2_raw = m2_raw[valid_mask]

    # Extract corresponding uniform data
    u1_raw = copula_matrix[valid_mask, idx1]
    u2_raw = copula_matrix[valid_mask, idx2]

    # --- SIMULATE THE DISCRETE BACKGROUND MODEL ---
    print("Generating discrete CDFs from 5-parameter fits...")
    try:
        cdf1, bins1, centers1, fmin1, fmax1 = get_discrete_cdf_and_bins(base_dir, args.trigger, args.ch1, args.cms)
        cdf2, bins2, centers2, fmin2, fmax2 = get_discrete_cdf_and_bins(base_dir, args.trigger, args.ch2, args.cms)
    except Exception as e:
        print(f"Failed to load fits: {e}")
        return

    print("Sampling massive dataset and mapping to discrete bins...")
    N_toys = 100000 # Increased for a smoother 2D binned plot
    random_indices = np.random.choice(len(u1_raw), size=N_toys, replace=True)
    
    # Jitter to break ties empirically (matching your script)
    u1_toy = u1_raw[random_indices] + np.random.uniform(-0.0002, 0.0002, size=N_toys)
    u2_toy = u2_raw[random_indices] + np.random.uniform(-0.0002, 0.0002, size=N_toys)
    
    # Bound reflections
    u1_toy = np.abs(u1_toy)
    u1_toy = np.where(u1_toy >= 1.0, 1.99999 - u1_toy, u1_toy)
    u2_toy = np.abs(u2_toy)
    u2_toy = np.where(u2_toy >= 1.0, 1.99999 - u2_toy, u2_toy)

    # MAP TO EXACT BIN CENTERS USING SEARCHSORTED (The core correction)
    idx_m1 = np.searchsorted(cdf1, u1_toy)
    idx_m1 = np.clip(idx_m1, 0, len(centers1) - 1) # Safety clip
    m1_toy = centers1[idx_m1]

    idx_m2 = np.searchsorted(cdf2, u2_toy)
    idx_m2 = np.clip(idx_m2, 0, len(centers2) - 1)
    m2_toy = centers2[idx_m2]

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    scatter_kws = {'alpha': 0.3, 's': 10, 'edgecolors': 'none'}

    xlims = (fmin1, fmax1)
    ylims = (fmin2, fmax2)

    # Plot 1: Raw Data
    axes[0].scatter(m1_raw, m2_raw, color='#d62728', **scatter_kws)
    axes[0].set_title(f"1. Raw Data Space\n(Empirical Marginals + Correlation)", fontsize=14)
    axes[0].set_xlabel(f"${args.ch1.upper()}$ Mass [GeV]", fontsize=12)
    axes[0].set_ylabel(f"${args.ch2.upper()}$ Mass [GeV]", fontsize=12)
    axes[0].set_xlim(xlims); axes[0].set_ylim(ylims)

    # Plot 2: Uniform Space (The Copula)
    axes[1].scatter(u1_raw, u2_raw, color='#1f77b4', **scatter_kws)
    axes[1].set_title(f"2. Uniform Space (The Copula)\n(Pure Correlation Structure)", fontsize=14)
    axes[1].set_xlabel(f"$U_{{{args.ch1.upper()}}}$", fontsize=12)
    axes[1].set_ylabel(f"$U_{{{args.ch2.upper()}}}$", fontsize=12)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    axes[1].set_aspect('equal', adjustable='box') 

    # Plot 3: Final Model Space (2D Binned Histogram)
    print("Drawing binned 2D histogram...")
    # We use hist2d with your exact ATLAS_BINS to show the discrete nature
    h = axes[2].hist2d(m1_toy, m2_toy, bins=[bins1, bins2], cmap="YlOrBr", norm=LogNorm(vmin=1))
    
    # Add the colorbar scale
    cbar = fig.colorbar(h[3], ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Toy Event Count (per bin)', rotation=270, labelpad=15, fontsize=12)

    axes[2].set_title(f"3. Binned Copula Model Space\n(Discrete Mapping via searchsorted)", fontsize=14)
    axes[2].set_xlabel(f"${args.ch1.upper()}'$ Mass (Model) [GeV]", fontsize=12)
    axes[2].set_ylabel(f"${args.ch2.upper()}'$ Mass (Model) [GeV]", fontsize=12)
    axes[2].set_xlim(xlims); axes[2].set_ylim(ylims)

    plt.suptitle(f"Discrete Copula Transformation: {args.ch1.upper()} vs {args.ch2.upper()} | {args.trigger.upper()}", fontsize=16, y=1.05)
    plt.tight_layout()

    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    out_path = os.path.join(base_dir, "plots", f"copula_concept_discrete_{args.trigger}_{args.ch1}_{args.ch2}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccessfully generated discrete conceptual plot: {out_path}")

if __name__ == "__main__":
    main()
