#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

# Setup paths
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from src.config import ATLAS_BINS
from src.models import FiveParam

def get_channel_data(base_dir, trigger, channel, cms, mass_matrix, col_names):
    idx = col_names.index(f"M{channel}")
    masses = mass_matrix[:, idx]
    valid_masses = masses[masses > 0] * cms
    
    fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{channel}.json")
    with open(fitfile, "r") as j_nom:
        d_nom = json.load(j_nom)
        
    fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
    v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
    c = (v_bins[:-1] + v_bins[1:]) / 2
    
    expected_counts = FiveParam(cms, c, *d_nom['parameters']) 
    expected_counts = np.maximum(expected_counts, 0)
    cdf = np.cumsum(expected_counts) / np.sum(expected_counts)

    if len(valid_masses) > 0:
        u_min = np.sum(valid_masses < v_bins[0]) / len(valid_masses)
        u_max = np.sum(valid_masses <= v_bins[-1]) / len(valid_masses)
    else:
        u_min, u_max = 0.0, 1.0
        
    return cdf, v_bins, c, (u_min, u_max)

def map_uniform_to_mass(u_array, u_bounds, cdf, centers, apply_jitter=False):
    u_min, u_max = u_bounds
    mask = (u_array >= u_min) & (u_array <= u_max)
    u_valid = u_array[mask]
    if len(u_valid) == 0: return np.array([]), mask
    if apply_jitter:
        u_valid = u_valid + np.random.uniform(-0.0002, 0.0002, size=len(u_valid))
    u_trunc = (u_valid - u_min) / max(u_max - u_min, 1e-10)
    u_trunc = np.abs(u_trunc)
    u_trunc = np.where(u_trunc >= 1.0, 1.99999 - u_trunc, u_trunc)
    idx = np.searchsorted(cdf, u_trunc)
    idx = np.clip(idx, 0, len(centers) - 1)
    return centers[idx], mask

def main():
    parser = argparse.ArgumentParser(description="Generate 2D Correlation Validation Plot.")
    parser.add_argument('--trigger', type=str, required=True, help="Trigger name")
    parser.add_argument('--ch1', type=str, default='jj', help="First channel")
    parser.add_argument('--ch2', type=str, default='jb', help="Second channel")
    parser.add_argument('--cms', type=float, default=13000., help="Center of mass energy")
    args = parser.parse_args()

    base_dir = os.getcwd() if os.path.exists("data") else repo_root

    print("Loading data...")
    f_mass = np.load(os.path.join(base_dir, "data", f"masses_{args.trigger}.npz"))
    f_copula = np.load(os.path.join(base_dir, "data", f"copula_{args.trigger}.npz"))
    
    mass_matrix, copula_matrix = f_mass['masses'], f_copula['copula']
    col_names_mass, col_names_cop = list(f_mass['columns']), list(f_copula['columns'])

    idx_mass1, idx_mass2 = col_names_mass.index(f"M{args.ch1}"), col_names_mass.index(f"M{args.ch2}")
    idx_cop1, idx_cop2 = col_names_cop.index(f"M{args.ch1}"), col_names_cop.index(f"M{args.ch2}")

    cdf1, bins1, centers1, b1 = get_channel_data(base_dir, args.trigger, args.ch1, args.cms, mass_matrix, col_names_mass)
    cdf2, bins2, centers2, b2 = get_channel_data(base_dir, args.trigger, args.ch2, args.cms, mass_matrix, col_names_mass)

    # 1. Raw Data Extraction
    m1_all, m2_all = mass_matrix[:, idx_mass1] * args.cms, mass_matrix[:, idx_mass2] * args.cms
    valid_raw_mask = (m1_all >= bins1[0]) & (m1_all <= bins1[-1]) & (m2_all >= bins2[0]) & (m2_all <= bins2[-1])
    m1_raw, m2_raw = m1_all[valid_raw_mask], m2_all[valid_raw_mask]
    n_raw = len(m1_raw)

    # 2. Toy Generation
    n_toy_target = 1_000_000
    print(f"Generating {n_toy_target} toys for validation...")
    random_indices = np.random.choice(len(copula_matrix), size=n_toy_target, replace=True)
    u1_sampled, u2_sampled = copula_matrix[random_indices, idx_cop1], copula_matrix[random_indices, idx_cop2]
    
    _, mask1 = map_uniform_to_mass(u1_sampled, b1, cdf1, centers1)
    _, mask2 = map_uniform_to_mass(u2_sampled, b2, cdf2, centers2)
    survivor_mask = mask1 & mask2
    
    m1_copula, _ = map_uniform_to_mass(u1_sampled[survivor_mask], b1, cdf1, centers1, apply_jitter=True)
    m2_copula, _ = map_uniform_to_mass(u2_sampled[survivor_mask], b2, cdf2, centers2, apply_jitter=True)

    m1_indep, _ = map_uniform_to_mass(np.random.uniform(b1[0], b1[1], size=len(m1_copula)), b1, cdf1, centers1, apply_jitter=True)
    m2_indep, _ = map_uniform_to_mass(np.random.uniform(b2[0], b2[1], size=len(m1_copula)), b2, cdf2, centers2, apply_jitter=True)

    # --- THE FIX: SHARED NORMALIZATION AND WEIGHTS ---
    # We want toys to be scaled down to the raw data size so 1 color = 1 physical event density
    weight_cop = np.ones_like(m1_copula) * (n_raw / len(m1_copula))
    weight_ind = np.ones_like(m1_indep) * (n_raw / len(m1_indep))

    # Determine unified VMAX from the raw data histogram
    raw_hist, _, _ = np.histogram2d(m1_raw, m2_raw, bins=[bins1, bins2])
    vmax_val = np.max(raw_hist) * 1.2 # Add 20% headroom
    unified_norm = LogNorm(vmin=0.5, vmax=vmax_val)

    def calc_metrics(x, y):
        if len(x) < 2: return 0.0, 0.0
        return stats.pearsonr(x, y)[0], stats.spearmanr(x, y)[0]

    metrics = {
        "raw": calc_metrics(m1_raw, m2_raw),
        "cop": calc_metrics(m1_copula, m2_copula),
        "ind": calc_metrics(m1_indep, m2_indep)
    }


    # --- 4. PLOTTING ---
    # Create a figure and a GridSpec with 4 columns: 3 for plots, 1 for the colorbar
    # width_ratios makes the first 3 columns wide and the last one narrow (5%)
    fig = plt.figure(figsize=(24, 7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax_cbar = fig.add_subplot(gs[0, 3]) # This is our dedicated colorbar axis

    axes = [ax0, ax1, ax2]
    configs = [
        (m1_raw, m2_raw, None, "1. Raw Data (In-Window)", metrics["raw"]),
        (m1_copula, m2_copula, weight_cop, "2. Copula Toys (Coupled)", metrics["cop"]),
        (m1_indep, m2_indep, weight_ind, "3. Independent Toys", metrics["ind"])
    ]

    h_last = None
    for ax, (x, y, w, title, (pr, sr)) in zip(axes, configs):
        # We use 'unified_norm' (LogNorm) calculated in previous step
        h = ax.hist2d(x, y, bins=[bins1, bins2], weights=w, cmap="YlOrBr", norm=unified_norm)
        h_last = h
        
        ax.set_title(f"{title}\nPearson $\\rho$: {pr:.3f} | Spearman $\\rho$: {sr:.3f}", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"${args.ch1.upper()}$ Mass [GeV]", fontsize=12)
        ax.set_ylabel(f"${args.ch2.upper()}$ Mass [GeV]", fontsize=12)
        ax.set_xlim(bins1[0], bins1[-1])
        ax.set_ylim(bins2[0], bins2[-1])

    # --- EXPLICIT COLORBAR PLACEMENT ---
    # We point the colorbar specifically to the 'ax_cbar' axis we created
    cbar = fig.colorbar(h_last[3], cax=ax_cbar)
    cbar.set_label('Equivalent Event Counts (Weighted)', rotation=270, labelpad=20, fontsize=12)

    plt.suptitle(f"2D Correlation Fidelity | {args.trigger.upper()} | {args.ch1.upper()} vs {args.ch2.upper()}", 
                 fontsize=20, fontweight='bold', y=1.02)
    
    # Using rect to ensure suptitle doesn't bleed into plots
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    out_path = os.path.join(base_dir, "plots", f"corr_fidelity_{args.trigger}_{args.ch1}_{args.ch2}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved with production layout: {out_path}")

if __name__ == "__main__":
    main()
