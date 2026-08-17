import ROOT
import numpy as np
from scipy.stats import rankdata
import sys

def extract_copula(input_root, output_npz):
    print(f"Reading TTree from {input_root}...")
    df = ROOT.RDataFrame("output", input_root)
    mass_vars = ["Mjj", "Mbb", "Mjb", "Mje", "Mjm", "Mjg", "Mbe", "Mbm", "Mbg"] 
    
    try:
        data_dict = df.AsNumpy(columns=mass_vars)
    except Exception as e:
        print(f"Failed to read columns: {e}")
        return

    N = len(data_dict["Mjj_data100percent"])
    # N = len(data_dict["Mjj"])
    print(f"Loaded {N} total events.")
    copula_matrix = np.zeros((N, len(mass_vars)))
    
    # Dictionary to store the event counts for our summary
    event_counts = {}

    print("Converting valid masses to empirical CDF quantiles...")
    for i, var in enumerate(mass_vars):
        var += "_data100percent"
        data = data_dict[var]

        # Find the physically valid masses (ignoring exact 0s or near-0 floats)
        valid_mask = data > 0.001 
        
        # Store the total number of valid events for this channel
        event_counts[var] = np.sum(valid_mask)
        
        # Initialize the whole column to -1.0 (our "missing particle" flag)
        U = np.full(N, -1.0) 
        
        # Only rank the events that actually contain these particles
        valid_data = data[valid_mask]
        if len(valid_data) > 0:
            ranks = rankdata(valid_data)
            U[valid_mask] = ranks / (len(valid_data) + 1.0)
            
        copula_matrix[:, i] = U

    np.savez(output_npz, copula=copula_matrix, columns=mass_vars)
    print(f"Successfully saved Copula matrix to {output_npz}")
    
    # Print a clean summary table at the end
    print("\n" + "="*40)
    print("      EVENT COUNTS PER CHANNEL")
    print("="*40)
    for var in mass_vars:
        print(f"{var:>8} : {event_counts[var]:,} valid events")
    print("="*40 + "\n")
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_copula.py <input.root> <output.npz>")
        sys.exit(1)
    extract_copula(sys.argv[1], sys.argv[2])
