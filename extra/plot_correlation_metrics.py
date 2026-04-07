#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

# Setup paths to import local modules
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from src.config import ATLAS_BINS
from src.models import FiveParam

def get_channel_data(base_dir, trigger, channel, cms, mass_matrix, col_names):
    """Extracts discrete CDFs, bin centers, and exact uniform bounds."""
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
    """Maps uniform [0,1] variables to discrete bins, with optional jitter."""
    u_min, u_max = u_bounds
    
    # 1. Truncate to valid window
    mask = (u_array >= u_min) & (u_array <= u_max)
    u_valid = u_array[mask]
    
    if len(u_valid) == 0:
        return np.array([]), mask
        
    # --- JITTER APPLIED HERE ---
    if apply_jitter:
        u_valid = u_valid + np.random.uniform(-0.0002, 0.0002, size=len(u_valid))
    
    # 2. Local scaling and reflection bounds
    u_trunc = (u_valid - u_min) / max(u_max - u_min, 1e-10)
    u_trunc = np.abs(u_trunc)
    u_trunc = np.where(u_trunc >= 1.0, 1.99999 - u_trunc, u_trunc)
    
    # 3. Discrete mapping via searchsorted
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
    mass_matrix = f_mass['masses']
    copula_matrix = f_copula['copula']
    col_names = list(f_mass['columns'])

    idx1 = col_names.index(f"M{args.ch1}")
    idx2 = col_names.index(f"M{args.ch2}")

    # Load Fits & Bounds first so we know the fit windows
    cdf1, bins1, centers1, b1 = get_channel_data(base_dir, args.trigger, args.ch1, args.cms, mass_matrix, col_names)
    cdf2, bins2, centers2, b2 = get_channel_data(base_dir, args.trigger, args.ch2, args.cms, mass_matrix, col_names)

    # --- 1. FAIR RAW DATA EXTRACTION ---
    # To compare fairly, we must look at the raw data STRICTLY inside the fit windows
    m1_all = mass_matrix[:, idx1] * args.cms
    m2_all = mass_matrix[:, idx2] * args.cms
    
    valid_raw_mask = (m1_all >= bins1[0]) & (m1_all <= bins1[-1]) & \
                     (m2_all >= bins2[0]) & (m2_all <= bins2[-1])
                     
    m1_raw = m1_all[valid_raw_mask]
    m2_raw = m2_all[valid_raw_mask]

    # --- 2. GENERATE TOYS ---
    N_toys = 200000 
    print(f"Generating ~{N_toys} toys for validation...")

    # A) COPULA TOYS
    N_data = len(copula_matrix)
    random_indices = np.random.choice(N_data, size=N_toys, replace=True)
    
    u1_sampled = copula_matrix[random_indices, idx1]
    u2_sampled = copula_matrix[random_indices, idx2]
    
    # Identify which pairs survive the truncation bounds
    _, mask1 = map_uniform_to_mass(u1_sampled, b1, cdf1, centers1, apply_jitter=False)
    _, mask2 = map_uniform_to_mass(u2_sampled, b2, cdf2, centers2, apply_jitter=False)
    survivor_mask = mask1 & mask2
    
    # Map the survivors and apply jitter
    m1_copula, _ = map_uniform_to_mass(u1_sampled[survivor_mask], b1, cdf1, centers1, apply_jitter=True)
    m2_copula, _ = map_uniform_to_mass(u2_sampled[survivor_mask], b2, cdf2, centers2, apply_jitter=True)

    # B) INDEPENDENT TOYS
    # Generate completely random uniform variables within the valid bounds
    u1_indep_raw = np.random.uniform(b1[0], b1[1], size=len(m1_copula)) 
    u2_indep_raw = np.random.uniform(b2[0], b2[1], size=len(m1_copula))
    
    m1_indep, _ = map_uniform_to_mass(u1_indep_raw, b1, cdf1, centers1, apply_jitter=True)
    m2_indep, _ = map_uniform_to_mass(u2_indep_raw, b2, cdf2, centers2, apply_jitter=True)

    # Clean up array lengths just in case
    min_len_cop = min(len(m1_copula), len(m2_copula))
    m1_copula, m2_copula = m1_copula[:min_len_cop], m2_copula[:min_len_cop]
    
    min_len_ind = min(len(m1_indep), len(m2_indep))
    m1_indep, m2_indep = m1_indep[:min_len_ind], m2_indep[:min_len_ind]

    # --- 3. CALCULATE METRICS ---
    def calc_metrics(x, y):
        if len(x) < 2 or len(y) < 2: return 0.0, 0.0
        p_r, _ = stats.pearsonr(x, y)
        s_r, _ = stats.spearmanr(x, y)
        return p_r, s_r

    pr_raw, sr_raw = calc_metrics(m1_raw, m2_raw)
    pr_cop, sr_cop = calc_metrics(m1_copula, m2_copula)
    pr_ind, sr_ind = calc_metrics(m1_indep, m2_indep)

    print(f"\nResults (Evaluated inside fit window):")
    print(f"Raw Data (in-window): Pearson = {pr_raw:.3f}, Spearman = {sr_raw:.3f}")
    print(f"Copula Toys:          Pearson = {pr_cop:.3f}, Spearman = {sr_cop:.3f}")
    print(f"Independent Toys:     Pearson = {pr_ind:.3f}, Spearman = {sr_ind:.3f}\n")

    # --- 4. PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    xlims = (bins1[0], bins1[-1])
    ylims = (bins2[0], bins2[-1])

    def plot_panel(ax, x, y, title, p_corr, s_corr):
        if len(x) > 0 and len(y) > 0:
            h = ax.hist2d(x, y, bins=[bins1, bins2], cmap="YlOrBr", norm=LogNorm(vmin=1))
        else:
            h = None
        ax.set_title(f"{title}\nPearson $\\rho$ = {p_corr:.3f} | Spearman $\\rho$ = {s_corr:.3f}", fontsize=14)
        ax.set_xlabel(f"${args.ch1.upper()}$ Mass [GeV]", fontsize=12)
        ax.set_ylabel(f"${args.ch2.upper()}$ Mass [GeV]", fontsize=12)
        ax.set_xlim(xlims); ax.set_ylim(ylims)
        return h

    h1 = plot_panel(axes[0], m1_raw, m2_raw, "1. Raw Data (Inside Fit Window)", pr_raw, sr_raw)
    h2 = plot_panel(axes[1], m1_copula, m2_copula, "2. Copula Toys (Coupled + Jitter)", pr_cop, sr_cop)
    h3 = plot_panel(axes[2], m1_indep, m2_indep, "3. Independent Poisson Toys", pr_ind, sr_ind)

    if h1 is not None:
        cbar = fig.colorbar(h1[3], ax=axes.ravel().tolist(), fraction=0.015, pad=0.04)
        cbar.set_label('Counts per bin', rotation=270, labelpad=15, fontsize=12)

    plt.suptitle(f"2D Correlation Structure Validation: {args.ch1.upper()} vs {args.ch2.upper()} | {args.trigger.upper()}", fontsize=18, y=1.05)

    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    out_path = os.path.join(base_dir, "plots", f"correlation_comparison_jitter_{args.trigger}_{args.ch1}_{args.ch2}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved 2D Correlation plot to: {out_path}")

if __name__ == "__main__":
    main()
