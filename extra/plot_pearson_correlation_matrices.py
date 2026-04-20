#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Suppress standard SciPy warnings globally just in case
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

# Setup paths to import local modules
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from src.config import ATLAS_BINS
from src.models import FiveParam

def get_channel_data(base_dir, trigger, channel, cms):
    """Extracts discrete CDFs, bin centers, and exact uniform bounds."""
    fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{channel}.json")
    if not os.path.exists(fitfile):
        return None
        
    with open(fitfile, "r") as j_nom:
        d_nom = json.load(j_nom)
        
    fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
    v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
    c = (v_bins[:-1] + v_bins[1:]) / 2
    
    expected_counts = FiveParam(cms, c, *d_nom['parameters']) 
    expected_counts = np.maximum(expected_counts, 0)
    cdf = np.cumsum(expected_counts) / np.sum(expected_counts)
        
    return cdf, v_bins, c, (fmin_val, fmax_val)

def map_uniform_to_mass(u_array, u_bounds, cdf, centers, apply_jitter=False):
    """Maps uniform [0,1] variables to discrete bins, with optional jitter."""
    u_min, u_max = u_bounds
    
    if apply_jitter:
        u_array = u_array + np.random.uniform(-0.0002, 0.0002, size=len(u_array))
    
    # Local scaling and reflection bounds
    u_trunc = (u_array - u_min) / max(u_max - u_min, 1e-10)
    u_trunc = np.abs(u_trunc)
    u_trunc = np.where(u_trunc >= 1.0, 1.99999 - u_trunc, u_trunc)
    
    # Discrete mapping via searchsorted
    idx = np.searchsorted(cdf, u_trunc)
    idx = np.clip(idx, 0, len(centers) - 1)
    
    return centers[idx]

def safe_pearson(x, y):
    """Safely calculates Pearson correlation, returning 0.0 for zero-variance arrays."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    
    r, _ = stats.pearsonr(x, y)
    
    if np.isnan(r):
        return 0.0
        
    return r

def format_axes_labels(ax):
    """Helper function to apply professional formatting to heatmap axes."""
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, rotation=0, va='center')

def main():
    parser = argparse.ArgumentParser(description="Plot Full Pearson Matrix: Raw vs Empirical Bootstrap vs Copula vs Independent")
    parser.add_argument('--trigger', type=str, required=True, help="Trigger name")
    parser.add_argument('--cms', type=float, default=13000., help="Center of mass energy")
    args = parser.parse_args()

    base_dir = os.getcwd() if os.path.exists("data") else repo_root

    print("Loading data matrices...")
    mass_path = os.path.join(base_dir, "data", f"masses_{args.trigger}.npz")
    copula_path = os.path.join(base_dir, "data", f"copula_{args.trigger}.npz")
    
    if not os.path.exists(mass_path) or not os.path.exists(copula_path):
        print("Error: Could not find masses or copula npz files.")
        sys.exit(1)

    f_mass = np.load(mass_path)
    f_copula = np.load(copula_path)
    mass_matrix = f_mass['masses']
    copula_matrix = f_copula['copula']
    
    col_names_mass = list(f_mass['columns'])
    col_names_cop = list(f_copula['columns'])
    
    # Make labels professional (e.g., convert "Mjj" to mathematical "$m_{jj}$")
    prof_labels = [f"$m_{{{col.replace('M', '')}}}$" if col.startswith("M") else col for col in col_names_mass]

    n_cols = len(col_names_mass)
    
    channel_info = {}
    print("Loading 5-parameter fits and calculating phase-space bounds...")
    for i, col in enumerate(col_names_mass):
        channel = col.replace("M", "")
        data = get_channel_data(base_dir, args.trigger, channel, args.cms)
        if data is None:
            print(f"Warning: Missing fit for {channel}. Skipping in matrix.")
            continue
            
        cdf, bins, centers, mass_bounds = data
        
        valid_masses = mass_matrix[:, i] * args.cms
        valid_masses = valid_masses[valid_masses > 0]
        if len(valid_masses) > 0:
            u_min = np.sum(valid_masses < mass_bounds[0]) / len(valid_masses)
            u_max = np.sum(valid_masses <= mass_bounds[1]) / len(valid_masses)
        else:
            u_min, u_max = 0.0, 1.0
            
        channel_info[i] = {
            'cdf': cdf, 'centers': centers, 
            'mass_bounds': mass_bounds, 'u_bounds': (u_min, u_max)
        }

    corr_raw = np.eye(n_cols)
    corr_boot = np.eye(n_cols)
    corr_copula = np.eye(n_cols)
    corr_indep = np.eye(n_cols)

    N_toys = 10_000_000 
    print(f"Generating global sampling indices for {N_toys} events...")
    
    # Generate 10M indices globally to preserve correlations across all channel pairwise calculations
    boot_indices = np.random.choice(len(mass_matrix), size=N_toys, replace=True)
    cop_indices = np.random.choice(len(copula_matrix), size=N_toys, replace=True)

    print("Calculating pairwise Pearson Linear correlations...")
    
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            if i not in channel_info or j not in channel_info:
                continue
                
            info_i, info_j = channel_info[i], channel_info[j]
            col_i, col_j = col_names_mass[i], col_names_mass[j]
            
            # --- 1. RAW DATA ---
            m_i_all = mass_matrix[:, i] * args.cms
            m_j_all = mass_matrix[:, j] * args.cms
            
            valid_raw_mask = (m_i_all >= info_i['mass_bounds'][0]) & (m_i_all <= info_i['mass_bounds'][1]) & \
                             (m_j_all >= info_j['mass_bounds'][0]) & (m_j_all <= info_j['mass_bounds'][1])
                             
            m_i_raw = m_i_all[valid_raw_mask]
            m_j_raw = m_j_all[valid_raw_mask]
            
            corr_raw[i, j] = corr_raw[j, i] = safe_pearson(m_i_raw, m_j_raw)

            # --- 2. EMPIRICAL BOOTSTRAP TOYS (10M Asymptotic Sample) ---
            m_i_sampled = mass_matrix[boot_indices, i] * args.cms
            m_j_sampled = mass_matrix[boot_indices, j] * args.cms
            
            valid_boot_mask = (m_i_sampled >= info_i['mass_bounds'][0]) & (m_i_sampled <= info_i['mass_bounds'][1]) & \
                              (m_j_sampled >= info_j['mass_bounds'][0]) & (m_j_sampled <= info_j['mass_bounds'][1])
                              
            m_i_boot = m_i_sampled[valid_boot_mask]
            m_j_boot = m_j_sampled[valid_boot_mask]
            
            corr_boot[i, j] = corr_boot[j, i] = safe_pearson(m_i_boot, m_j_boot)

            # --- 3. COPULA TOYS (10M Asymptotic Sample) ---
            idx_cop_i = col_names_cop.index(col_i)
            idx_cop_j = col_names_cop.index(col_j)
            
            u_i_sampled = copula_matrix[cop_indices, idx_cop_i]
            u_j_sampled = copula_matrix[cop_indices, idx_cop_j]
            
            u_min_i, u_max_i = info_i['u_bounds']
            u_min_j, u_max_j = info_j['u_bounds']
            
            survivor_mask = (u_i_sampled >= u_min_i) & (u_i_sampled <= u_max_i) & \
                            (u_j_sampled >= u_min_j) & (u_j_sampled <= u_max_j)
                            
            u_i_survivors = u_i_sampled[survivor_mask]
            u_j_survivors = u_j_sampled[survivor_mask]
            
            m_i_toy = map_uniform_to_mass(u_i_survivors, info_i['u_bounds'], info_i['cdf'], info_i['centers'], apply_jitter=True)
            m_j_toy = map_uniform_to_mass(u_j_survivors, info_j['u_bounds'], info_j['cdf'], info_j['centers'], apply_jitter=True)
            
            corr_copula[i, j] = corr_copula[j, i] = safe_pearson(m_i_toy, m_j_toy)

            # --- 4. INDEPENDENT POISSON TOYS (Uncorrelated Sample) ---
            # Generate completely independent uniform arrays to destroy correlation,
            # matching the size of the copula survivors for consistent statistical power.
            n_survivors = len(u_i_survivors)
            if n_survivors > 0:
                u_i_indep = np.random.uniform(u_min_i, u_max_i, size=n_survivors)
                u_j_indep = np.random.uniform(u_min_j, u_max_j, size=n_survivors)

                m_i_indep = map_uniform_to_mass(u_i_indep, info_i['u_bounds'], info_i['cdf'], info_i['centers'], apply_jitter=True)
                m_j_indep = map_uniform_to_mass(u_j_indep, info_j['u_bounds'], info_j['cdf'], info_j['centers'], apply_jitter=True)

                corr_indep[i, j] = corr_indep[j, i] = safe_pearson(m_i_indep, m_j_indep)

    # --- 5. PLOTTING ---
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    vmin, vmax = -0.1, 1.0 
    
    # 5a. COMBINED PLOT (1x4 Layout)
    fig, axes = plt.subplots(1, 4, figsize=(32, 8))

    sns.heatmap(corr_raw, ax=axes[0], cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=prof_labels, yticklabels=prof_labels, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    axes[0].set_title(f"Raw Data Pearson $r$\n(In-Window)", fontsize=16, pad=15)
    format_axes_labels(axes[0])

    sns.heatmap(corr_boot, ax=axes[1], cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=prof_labels, yticklabels=prof_labels, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    axes[1].set_title(f"Empirical Bootstrap Pearson $r$\n($10^7$ Event Toys)", fontsize=16, pad=15)
    format_axes_labels(axes[1])

    sns.heatmap(corr_copula, ax=axes[2], cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=prof_labels, yticklabels=prof_labels, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    axes[2].set_title(f"Copula Pearson $r$\n($10^7$ Event Toys)", fontsize=16, pad=15)
    format_axes_labels(axes[2])

    sns.heatmap(corr_indep, ax=axes[3], cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=prof_labels, yticklabels=prof_labels, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    axes[3].set_title(f"Independent Poisson Pearson $r$\n(Uncorrelated Control)", fontsize=16, pad=15)
    format_axes_labels(axes[3])

    plt.suptitle(f"Global Correlation Preservation Comparison - Trigger {args.trigger.upper()}", fontsize=20, y=1.05)
    plt.tight_layout()
    
    out_path_combined = os.path.join(base_dir, "plots", f"full_pearson_matrix_comparison_{args.trigger}.png")
    fig.savefig(out_path_combined, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"\nSuccessfully saved combined Pearson correlation matrix plot to: {out_path_combined}")

    # 5b. INDIVIDUAL RAW DATA PLOT
    fig_raw, ax_raw = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_raw, ax=ax_raw, cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=prof_labels, yticklabels=prof_labels, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    ax_raw.set_title(f"Raw Data Pearson Correlation Coefficient $r$\nTrigger {args.trigger.upper()}", fontsize=16, pad=15)
    format_axes_labels(ax_raw)
    plt.tight_layout()
    
    out_path_raw = os.path.join(base_dir, "plots", f"pearson_matrix_raw_{args.trigger}.png")
    fig_raw.savefig(out_path_raw, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_raw)
    print(f"Successfully saved individual Raw Pearson matrix plot to: {out_path_raw}")

    # 5c. INDIVIDUAL BOOTSTRAP TOYS PLOT
    fig_boot, ax_boot = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_boot, ax=ax_boot, cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=prof_labels, yticklabels=prof_labels, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    ax_boot.set_title(f"Empirical Bootstrap Pearson $r$\nTrigger {args.trigger.upper()} ($10^7$ Event Toys)", fontsize=16, pad=15)
    format_axes_labels(ax_boot)
    plt.tight_layout()
    
    out_path_boot = os.path.join(base_dir, "plots", f"pearson_matrix_bootstrap_{args.trigger}.png")
    fig_boot.savefig(out_path_boot, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_boot)
    print(f"Successfully saved individual Bootstrap Pearson matrix plot to: {out_path_boot}")

    # 5d. INDIVIDUAL COPULA TOYS PLOT
    fig_cop, ax_cop = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_copula, ax=ax_cop, cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=prof_labels, yticklabels=prof_labels, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    ax_cop.set_title(f"Copula Toys Pearson $r$ (Mapped to 5-Param)\nTrigger {args.trigger.upper()} ($10^7$ Event Toys)", fontsize=16, pad=15)
    format_axes_labels(ax_cop)
    plt.tight_layout()
    
    out_path_cop = os.path.join(base_dir, "plots", f"pearson_matrix_copula_{args.trigger}.png")
    fig_cop.savefig(out_path_cop, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_cop)
    print(f"Successfully saved individual Copula Pearson matrix plot to: {out_path_cop}")

    # 5e. INDIVIDUAL INDEPENDENT POISSON TOYS PLOT
    fig_indep, ax_indep = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_indep, ax=ax_indep, cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=prof_labels, yticklabels=prof_labels, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    ax_indep.set_title(f"Independent Poisson Pearson $r$ (Uncorrelated)\nTrigger {args.trigger.upper()} ($10^7$ Event Toys)", fontsize=16, pad=15)
    format_axes_labels(ax_indep)
    plt.tight_layout()
    
    out_path_indep = os.path.join(base_dir, "plots", f"pearson_matrix_independent_{args.trigger}.png")
    fig_indep.savefig(out_path_indep, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_indep)
    print(f"Successfully saved individual Independent Poisson matrix plot to: {out_path_indep}")

if __name__ == "__main__":
    main()
