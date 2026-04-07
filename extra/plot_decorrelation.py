#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

def plot_decorrelation_matrices(mass_matrix, col_names, trigger_name):
    """
    Calculates, plots, and saves the correlation matrices before and after decorrelation.
    """
    # 1. Calculate Correlation Matrix of Raw Data
    # rowvar=False means columns are variables (channels), rows are observations (events)
    corr_raw = np.corrcoef(mass_matrix, rowvar=False)

    # 2. Apply Decorrelated Bootstrap Logic (Independent Column Shuffling)
    mass_matrix_shuffled = np.copy(mass_matrix)
    for col_idx in range(mass_matrix_shuffled.shape[1]):
        np.random.shuffle(mass_matrix_shuffled[:, col_idx])

    # 3. Calculate Correlation Matrix of Shuffled Data
    corr_shuffled = np.corrcoef(mass_matrix_shuffled, rowvar=False)

    # 4. Visualization Setup
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # We set vmin to -0.1 to keep the colorbar centered near 0, but allow 1.0 for the diagonal
    vmin, vmax = -0.1, 1.0 

    # Plot Raw Matrix
    sns.heatmap(corr_raw, ax=axes[0], cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=col_names, yticklabels=col_names, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    axes[0].set_title(f"Raw Data Correlation Matrix | {trigger_name}\n(Off-diagonals show true trigger overlaps)", fontsize=14)

    # Plot Shuffled Matrix
    sns.heatmap(corr_shuffled, ax=axes[1], cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=col_names, yticklabels=col_names, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    axes[1].set_title(f"Decorrelated Bootstrap Matrix | {trigger_name}\n(Correlation structure destroyed)", fontsize=14)

    plt.tight_layout()
    
    # 5. Save the figure
    os.makedirs("plots", exist_ok=True)
    out_path = os.path.join("plots", f"decorrelation_matrix_{trigger_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved correlation matrix plot to: {out_path}")
    
    # Optional: Still display it if running interactively
    # plt.show()

def main():
    p = ArgumentParser(description="Visualize trigger channel decorrelation.")
    p.add_argument('--trigger', required=True, help="Name of the trigger (e.g., HLTMass)")
    args = p.parse_args()

    # Locate the data file
    data_path = os.path.join("data", f"masses_{args.trigger}.npz")
    if not os.path.exists(data_path):
        print(f"Error: Could not find data file at {data_path}")
        return

    # Load data
    print(f"Loading data from {data_path}...")
    f = np.load(data_path)
    mass_matrix = f['masses']
    col_names = list(f['columns'])
    
    print(f"Loaded matrix with {mass_matrix.shape[0]} events and {mass_matrix.shape[1]} channels.")
    
    plot_decorrelation_matrices(mass_matrix, col_names, args.trigger)

if __name__ == '__main__':
    main()
