#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Setup paths to import local modules
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from src.config import ATLAS_BINS
from src.models import FiveParam

def get_discrete_cdf_and_bins(base_dir, trigger, channel, cms):
    fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{channel}.json")
    if not os.path.exists(fitfile):
        raise FileNotFoundError(f"Fit file not found: {fitfile}")

    with open(fitfile, "r") as j_nom:
        d_nom = json.load(j_nom)
        
    fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
    v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
    c = (v_bins[:-1] + v_bins[1:]) / 2
    
    counts_nom = FiveParam(cms, c, *d_nom['parameters']) 
    counts_nom = np.maximum(counts_nom, 0)
    cdf = np.cumsum(counts_nom) / np.sum(counts_nom)
        
    return cdf, v_bins, c, fmin_val, fmax_val

def main():
    parser = argparse.ArgumentParser(description="Visualize the Discrete Empirical Copula Transformation.")
    parser.add_argument('--trigger', type=str, required=True, help="Trigger name")
    parser.add_argument('--ch1', type=str, default='jj', help="First channel")
    parser.add_argument('--ch2', type=str, default='jb', help="Second channel")
    parser.add_argument('--cms', type=float, default=13000.)
    args = parser.parse_args()

    base_dir = os.getcwd() if os.path.exists("data") else repo_root

    f_mass = np.load(os.path.join(base_dir, "data", f"masses_{args.trigger}.npz"))
    f_copula = np.load(os.path.join(base_dir, "data", f"copula_{args.trigger}.npz"))

    mass_matrix = f_mass['masses']
    copula_matrix = f_copula['copula']
    col_names_mass = list(f_mass['columns'])
    col_names_cop = list(f_copula['columns'])

    # Explicit indices for column alignment safety
    idx_m1, idx_m2 = col_names_mass.index(f"M{args.ch1}"), col_names_mass.index(f"M{args.ch2}")
    idx_u1, idx_u2 = col_names_cop.index(f"M{args.ch1}"), col_names_cop.index(f"M{args.ch2}")

    m1_raw = mass_matrix[:, idx_m1] * args.cms
    m2_raw = mass_matrix[:, idx_m2] * args.cms
    valid_mask = (m1_raw > 0) & (m2_raw > 0)
    
    m1_raw, m2_raw = m1_raw[valid_mask], m2_raw[valid_mask]
    u1_raw, u2_raw = copula_matrix[valid_mask, idx_u1], copula_matrix[valid_mask, idx_u2]

    cdf1, bins1, centers1, fmin1, fmax1 = get_discrete_cdf_and_bins(base_dir, args.trigger, args.ch1, args.cms)
    cdf2, bins2, centers2, fmin2, fmax2 = get_discrete_cdf_and_bins(base_dir, args.trigger, args.ch2, args.cms)

    # Oversampling for smooth model visualization
    N_toys = 1_000_000 
    random_indices = np.random.choice(len(u1_raw), size=N_toys, replace=True)
    
    # Apply Jitter and reflections
    def process_u(u):
        u_j = u + np.random.uniform(-0.0002, 0.0002, size=len(u))
        u_j = np.abs(u_j)
        return np.where(u_j >= 1.0, 1.99999 - u_j, u_j)

    u1_toy, u2_toy = process_u(u1_raw[random_indices]), process_u(u2_raw[random_indices])

    # Discrete Inverse CDF Mapping
    m1_toy = centers1[np.clip(np.searchsorted(cdf1, u1_toy), 0, len(centers1)-1)]
    m2_toy = centers2[np.clip(np.searchsorted(cdf2, u2_toy), 0, len(centers2)-1)]

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    # Panel 1: Raw Data (Crimson)
    axes[0].scatter(m1_raw, m2_raw, color='#d62728', s=12, alpha=0.3, edgecolors='none')
    axes[0].set_title("1. Raw Data Space\n(Empirical Mass Distributions)", fontsize=14, fontweight='bold')
    axes[0].set_xlim(fmin1, fmax1); axes[0].set_ylim(fmin2, fmax2)

    # Panel 2: Uniform Space (Emerald Green - Requested Color Change)
    axes[1].scatter(u1_raw, u2_raw, color='#2ca02c', s=12, alpha=0.3, edgecolors='none')
    axes[1].set_title("2. Uniform Space (The Copula)\n(Pure Phase-Space Correlation)", fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    axes[1].set_aspect('equal', adjustable='box')

    # Panel 3: Binned Model (Gold/Orange Heatmap)
    h = axes[2].hist2d(m1_toy, m2_toy, bins=[bins1, bins2], cmap="YlOrBr", norm=LogNorm(vmin=1))
    fig.colorbar(h[3], ax=axes[2], fraction=0.046, pad=0.04).set_label('Toy Count', rotation=270, labelpad=15)
    axes[2].set_title("3. Binned Copula Model\n(Smooth Fit + Discrete Mapping)", fontsize=14, fontweight='bold')
    axes[2].set_xlim(fmin1, fmax1); axes[2].set_ylim(fmin2, fmax2)

    plt.suptitle(f"The Copula Pipeline: {args.ch1.upper()} vs {args.ch2.upper()} Correlation | Trigger {args.trigger.upper()}", 
                 fontsize=20, fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_path = os.path.join(base_dir, "plots", f"copula_logic_{args.trigger}_{args.ch1}_{args.ch2}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved concept plot: {out_path}")

if __name__ == "__main__":
    main()
