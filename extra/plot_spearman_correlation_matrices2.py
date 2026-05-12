#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, chi2
import pandas as pd
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
    u_min, u_max = u_bounds
    
    if apply_jitter:
        u_array = u_array + np.random.uniform(-0.0002, 0.0002, size=len(u_array))
    
    u_trunc = (u_array - u_min) / max(u_max - u_min, 1e-10)
    u_trunc = np.abs(u_trunc)
    u_trunc = np.where(u_trunc >= 1.0, 1.99999 - u_trunc, u_trunc)
    
    idx = np.searchsorted(cdf, u_trunc)
    idx = np.clip(idx, 0, len(centers) - 1)
    
    return centers[idx]

def safe_spearman(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    s, _ = stats.spearmanr(x, y)
    return 0.0 if np.isnan(s) else s

def format_axes_labels(ax):
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, rotation=0, va='center')

def main():
    parser = argparse.ArgumentParser(description="Plot Full Spearman Matrix: Empirical vs Parametric Copulas")
    parser.add_argument('--trigger', type=str, required=True, help="Trigger name")
    parser.add_argument('--cms', type=float, default=13000., help="Center of mass energy")
    parser.add_argument('--nu', type=float, default=5.0, help="Degrees of freedom for Student-t copula")
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
    prof_labels = [f"$m_{{{col.replace('M', '')}}}$" if col.startswith("M") else col for col in col_names_mass]
    n_cols = len(col_names_mass)
    
    channel_info = {}
    print("Loading fits and calculating phase-space bounds...")
    for i, col in enumerate(col_names_mass):
        channel = col.replace("M", "")
        data = get_channel_data(base_dir, args.trigger, channel, args.cms)
        if data is None: continue
            
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

    # ---------------------------------------------------------
    # GLOBAL COVARIANCE FOR PARAMETRIC MODELS
    # ---------------------------------------------------------
    print("Computing global covariance matrix for parametric models...")
    df_ranks = pd.DataFrame(np.where(copula_matrix >= 0, copula_matrix, np.nan))
    rho_matrix = df_ranks.corr(method='spearman').fillna(0).values
    cov_matrix = 2 * np.sin(rho_matrix * np.pi / 6)
    
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    if np.any(eigvals < 0):
        eigvals[eigvals < 0] = 1e-8
        cov_matrix = eigvecs.dot(np.diag(eigvals)).dot(eigvecs.T)
        d = np.diag(1.0 / np.sqrt(np.diag(cov_matrix)))
        cov_matrix = d.dot(cov_matrix).dot(d)

    # ---------------------------------------------------------
    # MASSIVE TOY GENERATION (10M EVENTS)
    # ---------------------------------------------------------
    N_toys = 10_000_000 
    print(f"Generating global kinematics for {N_toys} events...")
    
    boot_indices = np.random.choice(len(mass_matrix), size=N_toys, replace=True)
    cop_indices = np.random.choice(len(copula_matrix), size=N_toys, replace=True)
    
    # Extract empirical topology mask
    sampled_mask = copula_matrix[cop_indices] >= 0

    # Generate Kinematics
    U_emp = copula_matrix[cop_indices]
    
    Z_gauss = np.random.multivariate_normal(np.zeros(n_cols), cov_matrix, size=N_toys)
    U_gauss = norm.cdf(Z_gauss)
    
    S_t = chi2.rvs(args.nu, size=N_toys)
    T_student = Z_gauss * np.sqrt(args.nu / S_t[:, None])
    U_student = stats.t.cdf(T_student, df=args.nu)
    
    U_indep = np.random.uniform(0, 1, size=(N_toys, n_cols))

    # Output Matrices
    corr_raw, corr_boot = np.eye(n_cols), np.eye(n_cols)
    corr_emp, corr_gauss = np.eye(n_cols), np.eye(n_cols)
    corr_student, corr_indep = np.eye(n_cols), np.eye(n_cols)

    print("Calculating pairwise Spearman correlations...")
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            if i not in channel_info or j not in channel_info: continue
                
            info_i, info_j = channel_info[i], channel_info[j]
            u_min_i, u_max_i = info_i['u_bounds']
            u_min_j, u_max_j = info_j['u_bounds']
            
            # 1. RAW DATA
            m_i_all, m_j_all = mass_matrix[:, i] * args.cms, mass_matrix[:, j] * args.cms
            raw_mask = (m_i_all >= info_i['mass_bounds'][0]) & (m_i_all <= info_i['mass_bounds'][1]) & \
                       (m_j_all >= info_j['mass_bounds'][0]) & (m_j_all <= info_j['mass_bounds'][1])
            corr_raw[i, j] = corr_raw[j, i] = safe_spearman(m_i_all[raw_mask], m_j_all[raw_mask])

            # 2. BOOTSTRAP
            m_i_boot, m_j_boot = m_i_all[boot_indices], m_j_all[boot_indices]
            boot_mask = (m_i_boot >= info_i['mass_bounds'][0]) & (m_i_boot <= info_i['mass_bounds'][1]) & \
                        (m_j_boot >= info_j['mass_bounds'][0]) & (m_j_boot <= info_j['mass_bounds'][1])
            corr_boot[i, j] = corr_boot[j, i] = safe_spearman(m_i_boot[boot_mask], m_j_boot[boot_mask])

            # Helper for uniform models
            def calc_toy_corr(U_matrix):
                u_i, u_j = U_matrix[:, i], U_matrix[:, j]
                mask = sampled_mask[:, i] & sampled_mask[:, j] & \
                       (u_i >= u_min_i) & (u_i <= u_max_i) & \
                       (u_j >= u_min_j) & (u_j <= u_max_j)
                
                if np.sum(mask) < 2: return 0.0
                
                m_i = map_uniform_to_mass(u_i[mask], info_i['u_bounds'], info_i['cdf'], info_i['centers'], apply_jitter=True)
                m_j = map_uniform_to_mass(u_j[mask], info_j['u_bounds'], info_j['cdf'], info_j['centers'], apply_jitter=True)
                return safe_spearman(m_i, m_j)

            # 3-6. COPULAS & INDEPENDENT
            corr_emp[i, j] = corr_emp[j, i] = calc_toy_corr(U_emp)
            corr_gauss[i, j] = corr_gauss[j, i] = calc_toy_corr(U_gauss)
            corr_student[i, j] = corr_student[j, i] = calc_toy_corr(U_student)
            corr_indep[i, j] = corr_indep[j, i] = calc_toy_corr(U_indep)

    # ---------------------------------------------------------
    # PLOTTING (2x3 Grid)
    # ---------------------------------------------------------
    print("Generating plots...")
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    vmin, vmax = -0.1, 1.0 
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes_flat = axes.flatten()

    matrices = [
        (corr_raw, "Raw Data Spearman $\\rho$\n(In-Window)", "raw"),
        (corr_boot, f"Empirical Bootstrap Spearman $\\rho$\n($10^7$ Event Toys)", "bootstrap"),
        (corr_emp, f"Empirical Copula Spearman $\\rho$\n($10^7$ Event Toys)", "empirical_copula"),
        (corr_gauss, f"Gaussian Copula Spearman $\\rho$\n($10^7$ Event Toys)", "gaussian_copula"),
        (corr_student, f"Student-t Copula Spearman $\\rho$\n($10^7$ Event Toys)", "student_copula"),
        (corr_indep, f"Independent Poisson Spearman $\\rho$\n(Uncorrelated Control)", "independent")
    ]

    for idx, (mat, title, _) in enumerate(matrices):
        sns.heatmap(mat, ax=axes_flat[idx], cmap=cmap, vmin=vmin, vmax=vmax,
                    xticklabels=prof_labels, yticklabels=prof_labels, 
                    annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
        axes_flat[idx].set_title(title, fontsize=16, pad=15)
        format_axes_labels(axes_flat[idx])

    plt.suptitle(f"Global Correlation Preservation Comparison (Spearman) - Trigger {args.trigger.upper()}", fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(base_dir, "plots", f"full_spearman_matrix_comparison_6panel_{args.trigger}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"Successfully saved 6-panel Spearman correlation matrix plot to: {out_path}")

    # ---------------------------------------------------------
    # INDIVIDUAL PLOTS
    # ---------------------------------------------------------
    print("Generating individual correlation matrix plots...")
    for mat, title, suffix in matrices:
        fig_indiv, ax_indiv = plt.subplots(figsize=(10, 8))
        sns.heatmap(mat, ax=ax_indiv, cmap=cmap, vmin=vmin, vmax=vmax,
                    xticklabels=prof_labels, yticklabels=prof_labels, 
                    annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
        
        individual_title = f"{title}\nTrigger {args.trigger.upper()}"
        ax_indiv.set_title(individual_title, fontsize=16, pad=15)
        format_axes_labels(ax_indiv)
        plt.tight_layout()
        
        out_path_indiv = os.path.join(base_dir, "plots", f"spearman_matrix_{suffix}_{args.trigger}.png")
        fig_indiv.savefig(out_path_indiv, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig_indiv)
        print(f"  -> Saved individual {suffix} matrix plot: {out_path_indiv}")

    print("\nAll plotting routines completed successfully.")

if __name__ == "__main__":
    main()
