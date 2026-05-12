#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Setup paths to import local modules
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
    if not os.path.exists(fitfile):
        raise FileNotFoundError(f"Fit file not found: {fitfile}")

    with open(fitfile, "r") as j_nom:
        d_nom = json.load(j_nom)
        
    fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
    v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
    c = (v_bins[:-1] + v_bins[1:]) / 2
    
    chi2_val = float(d_nom.get('chi2', 0.0))
    ndf = int(d_nom.get('ndf', 1))
    n_params = len(d_nom['parameters'])
    
    data_counts, _ = np.histogram(valid_masses, bins=v_bins)
    expected_counts = np.maximum(FiveParam(cms, c, *d_nom['parameters']), 0)
    
    x_dense = np.linspace(v_bins[0], v_bins[-1], 1000)
    y_dense = np.maximum(FiveParam(cms, x_dense, *d_nom['parameters']), 0)
    
    cdf = np.cumsum(expected_counts) / np.sum(expected_counts)

    actual_fmin = v_bins[0]
    actual_fmax = v_bins[-1]
    N_valid = len(valid_masses)
    
    if N_valid > 0:
        u_min = np.sum(valid_masses < actual_fmin) / N_valid
        u_max = np.sum(valid_masses <= actual_fmax) / N_valid
    else:
        u_min, u_max = 0.0, 1.0
        
    return data_counts, expected_counts, cdf, v_bins, c, (u_min, u_max), x_dense, y_dense, n_params, chi2_val, ndf

def generate_expected_marginal(copula_matrix, col_names_cop, channel, u_bounds, cdf, centers, bins, method='empirical'):
    """Generates the unscaled toy marginals for Empirical, Gaussian, or Student-t."""
    idx_cop = col_names_cop.index(f"M{channel}")
    u_col_all = copula_matrix[:, idx_cop]
    u_col_valid = u_col_all[u_col_all >= 0]
    
    N_valid = len(u_col_valid)
    if N_valid == 0: return np.zeros(len(bins) - 1)
        
    N_mult = 10000 
    N_draw_total = N_valid * N_mult
    u_min, u_max = u_bounds
    toy_counts_total = np.zeros(len(bins) - 1, dtype=np.float64)
    
    chunk_size = 5_000_000
    num_chunks = int(np.ceil(N_draw_total / chunk_size))
    
    for _ in range(num_chunks):
        if method == 'empirical':
            u_raw = np.random.choice(u_col_valid, size=chunk_size, replace=True)
        else:
            # For 1D marginals, Gaussian/Student-t mathematically reduce to independent U(0,1)
            u_raw = np.random.uniform(0, 1, size=chunk_size)
        
        mask_in_window = (u_raw >= u_min) & (u_raw <= u_max)
        u_in_window = u_raw[mask_in_window]
        
        if len(u_in_window) == 0: continue
            
        u_jittered = u_in_window + np.random.uniform(-0.0002, 0.0002, size=len(u_in_window))
        u_trunc = (u_jittered - u_min) / max(u_max - u_min, 1e-10)
        u_trunc = np.abs(u_trunc)
        u_trunc = np.where(u_trunc >= 1.0, 1.99999 - u_trunc, u_trunc)
        
        idx_mapped = np.searchsorted(cdf, u_trunc)
        idx_mapped = np.clip(idx_mapped, 0, len(centers) - 1)
        m_toy = centers[idx_mapped]
        
        chunk_counts, _ = np.histogram(m_toy, bins=bins)
        toy_counts_total += chunk_counts

    actual_toys_simulated = (num_chunks * chunk_size) / N_valid
    return toy_counts_total / actual_toys_simulated

def plot_marginal(ax, channel, bins, centers, data, fit, toys_emp, toys_gauss, toys_student, x_dense, y_dense, chi2_val, ndf, p_val, n_params):
    reduced_chi2 = chi2_val / ndf if ndf > 0 else 0.0
    fit_label = f'{n_params}-Param Fit ($\\chi^2/\\mathrm{{ndf}}$ = {reduced_chi2:.2f})'
    
    # 1. Fit Lines
    ax.plot(x_dense, y_dense, color='dodgerblue', linewidth=2, alpha=0.6, label=fit_label, zorder=2)
    ax.plot(centers, fit, color='dodgerblue', marker='s', linestyle='none', markersize=5, zorder=3)
    
    # 2. Copula Toys (Shift markers slightly to avoid perfect overlap)
    ax.plot(centers, toys_emp, color='green', marker='o', linestyle='none', markersize=6, alpha=0.8, markeredgecolor='darkgreen', label='Empirical Copula Toys', zorder=6)
    ax.plot(centers, toys_gauss, color='purple', marker='^', linestyle='none', markersize=5, alpha=0.8, markeredgecolor='indigo', label='Gaussian Copula Toys', zorder=5)
    ax.plot(centers, toys_student, color='hotpink', marker='v', linestyle='none', markersize=5, alpha=0.8, markeredgecolor='mediumvioletred', label='Student-t Copula Toys', zorder=4)
    
    # 3. Raw Data
    err = np.sqrt(data)
    err[err == 0] = 1.0 
    ax.errorbar(centers, data, yerr=err, fmt='ko', markersize=4, capsize=3, label='Raw Data', zorder=10)

    prof_channel = f"$m_{{{channel}}}$"
    ax.set_title(f"{channel.upper()} Channel", fontsize=28)
    ax.set_xlabel(f"{prof_channel} [GeV]", fontsize=24)
    ax.set_ylabel("Events / Bin", fontsize=24)
    ax.set_yscale('log')
    
    x_range = bins[-1] - bins[0]
    ax.set_xlim(bins[0], bins[-1] + (0.15 * x_range))
    
    valid_data_min = np.min(data[data > 0]) if np.any(data > 0) else 1.0
    valid_fit_min = np.min(y_dense[y_dense > 0]) if np.any(y_dense > 0) else 1.0
    ax.set_ylim(max(1e-3, min(valid_data_min, valid_fit_min) * 0.1), np.max(data) * 5.0)
    
    ax.legend(fontsize=12, frameon=True, edgecolor='black', framealpha=0.9)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

def main():
    parser = argparse.ArgumentParser(description="Plot Marginal Agreement: Data vs Fit vs Parametric/Empirical Toys")
    parser.add_argument('--trigger', type=str, required=True, help="Trigger name (e.g., t2)")
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
    
    col_names_mass = list(f_mass['columns'])
    col_names_cop = list(f_copula['columns'])
    n_cols = len(col_names_mass)
    
    print(f"Detected {n_cols} channels. Initializing plot grid...")
    fig_grid, axes = plt.subplots(3, 3, figsize=(24, 18))
    axes_flat = axes.flatten()

    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)

    for i, col in enumerate(col_names_mass):
        channel = col.replace("M", "")
        print(f"Processing channel: {channel.upper()} ({i+1}/{n_cols})...")
        
        d, f, cdf, bins, centers, bounds, x_dense, y_dense, n_params, chi2_val, ndf = get_channel_data(
            base_dir, args.trigger, channel, args.cms, mass_matrix, col_names_mass
        )
        
        p_val = stats.chi2.sf(chi2_val, ndf)
        
        # Generate all three types of toys
        toys_emp = generate_expected_marginal(copula_matrix, col_names_cop, channel, bounds, cdf, centers, bins, method='empirical')
        toys_gauss = generate_expected_marginal(copula_matrix, col_names_cop, channel, bounds, cdf, centers, bins, method='gaussian')
        toys_student = generate_expected_marginal(copula_matrix, col_names_cop, channel, bounds, cdf, centers, bins, method='student')
        
        # 1. 9-panel grid
        plot_marginal(axes_flat[i], channel, bins, centers, d, f, toys_emp, toys_gauss, toys_student, x_dense, y_dense, chi2_val, ndf, p_val, n_params)

        # 2. Standalone figure
        fig_indiv, ax_indiv = plt.subplots(figsize=(10, 8))
        plot_marginal(ax_indiv, channel, bins, centers, d, f, toys_emp, toys_gauss, toys_student, x_dense, y_dense, chi2_val, ndf, p_val, n_params)
        
        fig_indiv.suptitle(f"Marginal Fidelity Validation | Trigger: {args.trigger.upper()}", fontsize=24, fontweight='bold', y=1.02)
        fig_indiv.tight_layout()
        out_path_indiv = os.path.join(base_dir, "plots", f"marginal_agreement_all_copulas_{args.trigger}_{channel}.png")
        fig_indiv.savefig(out_path_indiv, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig_indiv)

    for j in range(n_cols, len(axes_flat)): axes_flat[j].set_visible(False)

    fig_grid.suptitle(f"Marginal Fidelity Validation (All Methods) | Trigger: {args.trigger.upper()}", fontsize=36, fontweight='bold', y=1.02)
    fig_grid.tight_layout(rect=[0, 0, 1, 0.98], h_pad=3.0, w_pad=3.0)

    out_path_grid = os.path.join(base_dir, "plots", f"marginal_agreement_all_copulas_9panel_{args.trigger}.png")
    fig_grid.savefig(out_path_grid, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_grid)
    print(f"\nSuccessfully saved 9-panel marginal agreement plot to: {out_path_grid}")

if __name__ == "__main__":
    main()
