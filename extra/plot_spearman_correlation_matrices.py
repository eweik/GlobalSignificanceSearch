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


def safe_spearman(x, y):
    """Safely calculates Spearman correlation, returning 0.0 for zero-variance arrays."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    s, _ = stats.spearmanr(x, y)

    # Catch any residual NaN returns from scipy just in case
    if np.isnan(s):
        return 0.0

    return s

def main():
    parser = argparse.ArgumentParser(description="Plot Full Spearman Matrix: Raw vs Copula Transformed")
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
    
    # FIX 1: Extract both lists of column names for correct mapping
    col_names_mass = list(f_mass['columns'])
    col_names_cop = list(f_copula['columns'])

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
    corr_copula = np.eye(n_cols)

    print("Calculating pairwise Spearman Rank correlations...")
    # FIX 2: Oversample Copula to ensure survival in the deep tails
    N_toys = 10_000_000 
    
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            if i not in channel_info or j not in channel_info:
                continue
                
            info_i, info_j = channel_info[i], channel_info[j]
            col_i, col_j = col_names_mass[i], col_names_mass[j]
            
            # --- 1. RAW DATA (In-Window) ---
            m_i_all = mass_matrix[:, i] * args.cms
            m_j_all = mass_matrix[:, j] * args.cms
            
            valid_raw_mask = (m_i_all >= info_i['mass_bounds'][0]) & (m_i_all <= info_i['mass_bounds'][1]) & \
                             (m_j_all >= info_j['mass_bounds'][0]) & (m_j_all <= info_j['mass_bounds'][1])
                             
            m_i_raw = m_i_all[valid_raw_mask]
            m_j_raw = m_j_all[valid_raw_mask]
            
            corr_raw[i, j] = corr_raw[j, i] = safe_spearman(m_i_raw, m_j_raw)

            # --- 2. COPULA TOYS (Mapped to 5-Param) ---
            # FIX 3: Map the indices safely using string names
            idx_cop_i = col_names_cop.index(col_i)
            idx_cop_j = col_names_cop.index(col_j)
            
            random_indices = np.random.choice(len(copula_matrix), size=N_toys, replace=True)
            u_i_sampled = copula_matrix[random_indices, idx_cop_i]
            u_j_sampled = copula_matrix[random_indices, idx_cop_j]
            
            u_min_i, u_max_i = info_i['u_bounds']
            u_min_j, u_max_j = info_j['u_bounds']
            
            survivor_mask = (u_i_sampled >= u_min_i) & (u_i_sampled <= u_max_i) & \
                            (u_j_sampled >= u_min_j) & (u_j_sampled <= u_max_j)
                            
            u_i_survivors = u_i_sampled[survivor_mask]
            u_j_survivors = u_j_sampled[survivor_mask]
            
            # Map valid survivors to physical mass bins
            m_i_toy = map_uniform_to_mass(u_i_survivors, info_i['u_bounds'], info_i['cdf'], info_i['centers'], apply_jitter=True)
            m_j_toy = map_uniform_to_mass(u_j_survivors, info_j['u_bounds'], info_j['cdf'], info_j['centers'], apply_jitter=True)
            
            corr_copula[i, j] = corr_copula[j, i] = safe_spearman(m_i_toy, m_j_toy)

    # --- 3. PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    vmin, vmax = -0.1, 1.0 

    sns.heatmap(corr_raw, ax=axes[0], cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=col_names_mass, yticklabels=col_names_mass, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    axes[0].set_title(f"Raw Data Spearman $\\rho$ (In-Window)\n{args.trigger.upper()}", fontsize=14)

    sns.heatmap(corr_copula, ax=axes[1], cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=col_names_mass, yticklabels=col_names_mass, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    axes[1].set_title(f"Copula Toys Spearman $\\rho$ (Mapped to 5-Param)\n{args.trigger.upper()}", fontsize=14)

    # FIX 4: Add fontweight bold and tight_layout rect to prevent text bleed
    plt.suptitle("Validation of Global Rank Correlation (Spearman) Preservation", fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    out_path = os.path.join(base_dir, "plots", f"full_spearman_matrix_comparison_{args.trigger}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccessfully saved full Spearman correlation matrix plot to: {out_path}")

if __name__ == "__main__":
    main()
