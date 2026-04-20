#!/usr/bin/env python3
import os
import sys
import json
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from argparse import ArgumentParser

# Setup local module imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from src.config import ATLAS_BINS
from src.models import FiveParam

def nll_poisson(params, cms, c, data):
    """Negative Log-Likelihood for Poisson data against 5-Param model."""
    bkg = FiveParam(cms, c, *params)
    bkg = np.maximum(bkg, 1e-9) # Prevent log(0)
    # L = Sum(B - D * ln(B))
    return np.sum(bkg - data * np.log(bkg))

def main(args):
    os.makedirs("plots/absorption", exist_ok=True)
    base_dir = os.getcwd() if os.path.exists("fits") else repo_root

    print(f"\n{'='*60}")
    print(f" SIGNAL ABSORPTION SCAN (BACKGROUND-ONLY REFIT)")
    print(f" Channel: {args.channel} | Trigger: {args.trigger.upper()}")
    print(f" Injected Z: {args.sig_inj}σ | Width: {args.width_frac*100}% | Toys/Bin: {args.toys}")
    print(f"{'='*60}")

    # 1. Load Nominal Background from JSON
    fitfile_nom = os.path.join(base_dir, "fits", f"fitme_p5_{args.trigger}_{args.channel}.json")
    if not os.path.exists(fitfile_nom):
        print(f"Error: Cannot find JSON fit file at {fitfile_nom}")
        sys.exit(1)

    with open(fitfile_nom, "r") as j_nom:
        d_nom = json.load(j_nom)
        fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
        nominal_params = np.array([float(p) for p in d_nom['parameters']])
        print(f"Loaded JSON Parameters: {nominal_params}")

    # 2. Setup Binning and True Background
    v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
    c = (v_bins[:-1] + v_bins[1:]) / 2
    bin_widths = np.diff(v_bins)
    
    # Establish "True" expected background counts
    B_nom = FiveParam(args.cms, c, *nominal_params)

    # 3. Parameter Bounds for the Minimizer (To prevent the 5-param fit from exploding)
    bounds = [
        (nominal_params[0] * 0.1, nominal_params[0] * 10.0), # Norm
        (nominal_params[1] - 5.0, nominal_params[1] + 5.0),  # p1
        (nominal_params[2] - 5.0, nominal_params[2] + 5.0),  # p2
        (nominal_params[3] - 2.0, nominal_params[3] + 2.0),  # p3
        (nominal_params[4] - 2.0, nominal_params[4] + 2.0)   # p4
    ]

    mass_points = []
    eff_means, eff_errs = [], []
    sig_means, sig_errs = [], []

    start_time = time.time()

    # 4. Loop over mass bins
    for idx, mass_peak in enumerate(c):
        width = mass_peak * args.width_frac
        
        # Injection window (+/- 2 sigma)
        window_min = mass_peak - 2 * width
        window_max = mass_peak + 2 * width
        
        if window_min < fmin_val or window_max > fmax_val: continue
        
        # Calculate expected background in window
        window_mask = (c >= window_min) & (c <= window_max)
        B_expected = np.sum(B_nom[window_mask])
        
        if B_expected < 5: continue # Require some background
        
        # Calculate target signal yield
        N_inj = args.sig_inj * np.sqrt(B_expected)
        if N_inj < 1: continue

        # Generate Gaussian Signal Array
        sig_pdf = norm.pdf(c, loc=mass_peak, scale=width)
        S_arr = N_inj * (sig_pdf * bin_widths) / np.sum(sig_pdf * bin_widths)

        eff_toys = []
        sig_toys = []

        # 5. Run Pseudo-Experiments
        for toy in range(args.toys):
            # Generate Toy Data: Poisson(True Bkg + Signal)
            D_toy = np.random.poisson(B_nom + S_arr)

            # REFIT: Background-Only
            # We seed it with the nominal parameters so it converges quickly
            res = minimize(nll_poisson, nominal_params, args=(args.cms, c, D_toy), 
                           bounds=bounds, method='L-BFGS-B')

            if not res.success:
                continue

            # Evaluate the fitted background model
            B_fit = FiveParam(args.cms, c, *res.x)

            # Calculate Recovered Signal in Window
            N_rec = np.sum(D_toy[window_mask] - B_fit[window_mask])
            B_fit_window = np.sum(B_fit[window_mask])
            
            N_rec = max(N_rec, 0.0) # Floor at 0
            
            eff_toys.append(N_rec / N_inj)
            sig_toys.append(N_rec / np.sqrt(B_fit_window) if B_fit_window > 0 else 0.0)

        if len(eff_toys) == 0: continue

        mean_eff = np.mean(eff_toys)
        err_eff = np.std(eff_toys) / np.sqrt(len(eff_toys))
        mean_sig = np.mean(sig_toys)
        err_sig = np.std(sig_toys) / np.sqrt(len(sig_toys))

        mass_points.append(mass_peak / 1000.0) # Store in TeV
        eff_means.append(mean_eff)
        eff_errs.append(err_eff)
        sig_means.append(mean_sig)
        sig_errs.append(err_sig)

        print(f"Mass: {mass_peak:5.0f} GeV | N_inj: {N_inj:6.1f} | N_rec: {mean_eff*N_inj:6.1f} | Eff: {mean_eff*100:5.1f}% | Rec Z: {mean_sig:4.2f}σ")

    if len(mass_points) == 0:
        print("Error: No valid mass points processed.")
        sys.exit(1)

    # 6. Plotting with Matplotlib
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Efficiency Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(mass_points, eff_means, yerr=eff_errs, fmt='o-', color='blue', label='Recovery Efficiency', capsize=3)
    ax.axhline(1.0, color='red', linestyle='--', label='Ideal (No Absorption)')
    ax.set_title(f'Signal Absorption by 5-Param Model\n{args.trigger.upper()} | Channel: {args.channel} | Injected Z = {args.sig_inj}σ', fontsize=12, fontweight='bold')
    ax.set_xlabel('Injected Mass [TeV]', fontsize=12)
    ax.set_ylabel('Efficiency ($N_{rec} / N_{inj}$)', fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"plots/absorption/eff_{args.trigger}_{args.channel}_inj{args.sig_inj}.png", dpi=300)
    plt.close()

    # Significance Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(mass_points, sig_means, yerr=sig_errs, fmt='o-', color='green', label='Recovered Z', capsize=3)
    ax.axhline(args.sig_inj, color='red', linestyle='--', label='Injected Z')
    ax.set_title(f'Recovered Significance after Bkg Refit\n{args.trigger.upper()} | Channel: {args.channel}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Injected Mass [TeV]', fontsize=12)
    ax.set_ylabel('Recovered Local Significance [$Z_{local}$]', fontsize=12)
    ax.set_ylim(0, args.sig_inj + 1)
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"plots/absorption/sig_{args.trigger}_{args.channel}_inj{args.sig_inj}.png", dpi=300)
    plt.close()

    print(f"\n--- Done. Plots saved to plots/absorption/ ---")

if __name__ == '__main__':
    p = ArgumentParser(description="Signal Absorption Scan via Bkg Refit")
    p.add_argument('--trigger', type=str, default="t2")
    p.add_argument('--channel', type=str, default="jj")
    p.add_argument('--sig_inj', type=float, default=5.0)
    p.add_argument('--width_frac', type=float, default=0.05)
    p.add_argument('--toys', type=int, default=20, help="Toys per mass point")
    p.add_argument('--cms', type=float, default=13000.)
    main(p.parse_args())
