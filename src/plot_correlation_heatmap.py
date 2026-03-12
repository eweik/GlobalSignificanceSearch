import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trigger', type=str, default='t2')
    args = parser.parse_args()
    trigger = args.trigger

    filename = f"copula_{trigger}.npz"
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    # 1. Load the Copula Matrix
    print(f"Loading {filename}...")
    f = np.load(filename)
    matrix = f['copula']
    
    # Clean up column names for the plot (e.g., "Mjj" -> "jj")
    raw_cols = f['columns']
    clean_cols = [c.replace("M", "") if c.startswith("M") else c for c in raw_cols]

    # 2. Convert to Pandas DataFrame
    df = pd.DataFrame(matrix, columns=clean_cols)

    # 3. Handle the Physics Cuts
    # In our matrix, an event that fails a b-tag gets a rank of -1.0.
    # We replace these with NaN so pandas automatically ignores them 
    # when calculating the pairwise correlation for those specific channels.
    df = df.replace(-1.0, np.nan)

    # 4. Calculate the Spearman Rank Correlation
    print("Calculating pairwise Spearman rank correlations...")
    corr_matrix = df.corr(method='spearman')

    # 5. Plotting the Heatmap
    
    plt.figure(figsize=(10, 8))
    
    # Set up a mask so we only show the bottom triangle (optional, but looks cleaner)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Draw the heatmap
    sns.heatmap(corr_matrix, 
                mask=mask, 
                cmap='coolwarm', 
                vmin=-0.1, vmax=1.0, # Scale from slightly negative to perfect 1.0
                annot=True,          # Write the numbers in the boxes
                fmt=".2f",           # 2 decimal places
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .8, "label": "Spearman Rank Correlation ($\\rho$)"})

    plt.title(f"Empirical Copula Cross-Channel Rank Correlation (Trigger: {trigger.upper()})", fontsize=16, pad=20)
    
    # Rotate axis labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    plt.tight_layout()
    outname = f"spearman_heatmap_{trigger}.png"
    plt.savefig(outname, dpi=300)
    print(f"Saved correlation heatmap to {outname}")

if __name__ == "__main__":
    main()
