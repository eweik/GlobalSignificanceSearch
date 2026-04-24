#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

def get_fit_bounds(base_dir, trigger, col_names):
    """Loads the fmin and fmax bounds from the fit JSON files."""
    bounds = {}
    for col in col_names:
        channel = col.replace("M", "")
        fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{channel}.json")
        if os.path.exists(fitfile):
            with open(fitfile, 'r') as f:
                d = json.load(f)
                bounds[col] = (float(d['fmin']), float(d['fmax']))
        else:
            print(f"Warning: Fit file not found for {channel}. Using full range.")
            bounds[col] = (0.0, np.inf)
    return bounds

def plot_decorrelation_matrices(mass_matrix, col_names, trigger_name, bounds, cms):
    """
    Calculates, plots, and saves the correlation matrices before and after decorrelation,
    strictly within the fit boundaries for each channel pair.
    """
    n_cols = len(col_names)
    corr_raw = np.ones((n_cols, n_cols))
    corr_shuffled = np.ones((n_cols, n_cols))

    # Apply Decorrelated Bootstrap Logic (Independent Column Shuffling)
    # Done on the FULL matrix before windowing to preserve true 1D marginal distributions
    mass_matrix_shuffled = np.copy(mass_matrix)
    for col_idx in range(n_cols):
        np.random.shuffle(mass_matrix_shuffled[:, col_idx])

    # Convert to physical GeV for boundary comparisons
    m_raw_gev = mass_matrix * cms
    m_shuf_gev = mass_matrix_shuffled * cms

    print("Calculating pairwise correlations inside fit windows...")
    
    # Calculate pairwise correlation dynamically applying the specific bounds for the pair
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            col_i, col_j = col_names[i], col_names[j]
            b_i, b_j = bounds[col_i], bounds[col_j]

            # --- Raw Data Pairwise Mask ---
            mask_raw = (m_raw_gev[:, i] >= b_i[0]) & (m_raw_gev[:, i] <= b_i[1]) & \
                       (m_raw_gev[:, j] >= b_j[0]) & (m_raw_gev[:, j] <= b_j[1])

            if np.sum(mask_raw) > 1:
                c_raw = np.corrcoef(m_raw_gev[mask_raw, i], m_raw_gev[mask_raw, j])[0, 1]
                corr_raw[i, j] = corr_raw[j, i] = c_raw
            else:
                corr_raw[i, j] = corr_raw[j, i] = 0.0

            # --- Shuffled Data Pairwise Mask ---
            mask_shuf = (m_shuf_gev[:, i] >= b_i[0]) & (m_shuf_gev[:, i] <= b_i[1]) & \
                        (m_shuf_gev[:, j] >= b_j[0]) & (m_shuf_gev[:, j] <= b_j[1])

            if np.sum(mask_shuf) > 1:
                c_shuf = np.corrcoef(m_shuf_gev[mask_shuf, i], m_shuf_gev[mask_shuf, j])[0, 1]
                corr_shuffled[i, j] = corr_shuffled[j, i] = c_shuf
            else:
                corr_shuffled[i, j] = corr_shuffled[j, i] = 0.0

    # 4. Visualization Setup
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # We set vmin to -0.1 to keep the colorbar centered near 0, but allow 1.0 for the diagonal
    vmin, vmax = -0.1, 1.0 

    # Plot Raw Matrix
    sns.heatmap(corr_raw, ax=axes[0], cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=col_names, yticklabels=col_names, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    axes[0].set_title(f"Raw Data Correlation Matrix | {trigger_name.upper()}\n(Off-diagonals show true trigger overlaps)", fontsize=14)

    # Plot Shuffled Matrix
    sns.heatmap(corr_shuffled, ax=axes[1], cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=col_names, yticklabels=col_names, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    axes[1].set_title(f"Decorrelated/Shuffled Correlation Matrix | {trigger_name.upper()}\n(Correlation structure destroyed)", fontsize=14)

    plt.tight_layout()
    
    # 5. Save the figure
    os.makedirs("plots", exist_ok=True)
    out_path = os.path.join("plots", f"decorrelation_matrix_{trigger_name}_in_window.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved correlation matrix plot to: {out_path}")

def main():
    p = ArgumentParser(description="Visualize trigger channel decorrelation.")
    p.add_argument('--trigger', required=True, help="Name of the trigger (e.g., t2)")
    p.add_argument('--cms', type=float, default=13000., help="Center of mass energy (for GeV conversion)")
    args = p.parse_args()

    # Determine base directory dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(current_dir)
    base_dir = os.getcwd() if os.path.exists("data") and os.path.exists("fits") else repo_root

    # Locate the data file
    data_path = os.path.join(base_dir, "data", f"masses_{args.trigger}.npz")
    if not os.path.exists(data_path):
        print(f"Error: Could not find data file at {data_path}")
        return

    # Load data
    print(f"Loading data from {data_path}...")
    f = np.load(data_path)
    mass_matrix = f['masses']
    col_names = list(f['columns'])
    
    print(f"Loaded matrix with {mass_matrix.shape[0]} events and {mass_matrix.shape[1]} channels.")
    
    # Load Boundaries
    bounds = get_fit_bounds(base_dir, args.trigger, col_names)

    plot_decorrelation_matrices(mass_matrix, col_names, args.trigger, bounds, args.cms)

if __name__ == '__main__':
    main()
