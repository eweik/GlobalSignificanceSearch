import os
import sys
import json
import numpy as np
from scipy.stats import rankdata

# Make src.config importable (for ATLAS_BINS) whether run from repo root or python/
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

import ROOT
from src.config import ATLAS_BINS

CMS = 13000.0  # use 13000 even for Run 3

# Per-trigger anomaly-region cuts (AR_HAE_RUN23_100PERCENT_M100.txt)
AR_CUTS = {
    1: -9.374504251376864,
    2: -9.829198438298226,
    3: -9.238664051471718,
    4: -9.744067418703318,
    5: -8.295795788735555,
    6: -9.841564554356868,
    7: -9.344795288421562,
}

def extract_copula(input_root, output_npz, trig):
    cut = AR_CUTS[trig]
    print(f"Reading TTree from {input_root} (LogLoss >= {cut})...")
    df = ROOT.RDataFrame("output", input_root)
    mass_vars = ["Mjj", "Mbb", "Mjb", "Mje", "Mjm", "Mjg", "Mbe", "Mbm", "Mbg"]

    try:
        data_dict = df.AsNumpy(columns=mass_vars + ["LogLoss"])
    except Exception as e:
        print(f"Failed to read columns: {e}")
        return

    # Apply the anomaly-region cut before ranking
    keep = data_dict["LogLoss"] >= cut
    N = int(keep.sum())
    print(f"Loaded {len(keep)} events, {N} pass the cut.")

    copula_matrix = np.zeros((N, len(mass_vars)))
    # Phase-space truncation bounds (u_min, u_max) per channel, so downstream
    # toy generation needs only this file (no masses). Depends on each channel's
    # fit window, so the fit files must exist when this is run.
    u_bounds = np.tile([0.0, 1.0], (len(mass_vars), 1))
    event_counts = {}

    fits_dir = os.path.join(os.getcwd() if os.path.isdir("fits") else repo_root, "fits")

    print("Converting valid masses to empirical CDF quantiles...")
    for i, var in enumerate(mass_vars):
        data = data_dict[var][keep]

        # Find the physically valid masses (ignoring exact 0s or near-0 floats)
        valid_mask = data > 0.001
        event_counts[var] = np.sum(valid_mask)

        # Initialize the whole column to -1.0 (our "missing particle" flag)
        U = np.full(N, -1.0)

        valid_data = data[valid_mask]
        if len(valid_data) > 0:
            ranks = rankdata(valid_data)
            U[valid_mask] = ranks / (len(valid_data) + 1.0)

            # Truncation bounds: fraction of this channel's ranked population
            # falling inside its fit window [v_bins[0], v_bins[-1]].
            ch = var[1:]  # "Mjj" -> "jj"
            fitfile = os.path.join(fits_dir, f"fitme_p5_t{trig}_{ch}.json")
            if os.path.isfile(fitfile):
                d = json.load(open(fitfile))
                v_bins = ATLAS_BINS[(ATLAS_BINS >= float(d['fmin'])) & (ATLAS_BINS <= float(d['fmax']))]
                if len(v_bins) >= 2:
                    phys = valid_data * CMS
                    n = len(phys)
                    u_bounds[i] = [np.sum(phys < v_bins[0]) / n, np.sum(phys <= v_bins[-1]) / n]
            else:
                print(f"  Warning: no fit file for {ch} (t{trig}); u_bounds left at (0,1).")

        copula_matrix[:, i] = U

    np.savez(output_npz, copula=copula_matrix, columns=mass_vars, u_bounds=u_bounds)
    print(f"Successfully saved Copula matrix to {output_npz}")

    # Print a clean summary table at the end
    print("\n" + "="*52)
    print("      EVENT COUNTS PER CHANNEL   (u_min, u_max)")
    print("="*52)
    for i, var in enumerate(mass_vars):
        print(f"{var:>8} : {event_counts[var]:>8,} valid   ({u_bounds[i,0]:.3f}, {u_bounds[i,1]:.3f})")
    print("="*52 + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_copula.py <input.root> <output.npz> <trigger>")
        sys.exit(1)
    extract_copula(sys.argv[1], sys.argv[2], int(sys.argv[3]))
