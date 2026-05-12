#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, norm
import argparse
import os

def mean_residual_life(data, thresholds):
    """Calculates the mean excess for various thresholds."""
    mrl = []
    mrl_err = []
    for u in thresholds:
        excesses = data[data > u] - u
        if len(excesses) > 0:
            mrl.append(np.mean(excesses))
            # Standard error of the mean
            mrl_err.append(np.std(excesses, ddof=1) / np.sqrt(len(excesses)))
        else:
            mrl.append(np.nan)
            mrl_err.append(np.nan)
    return np.array(mrl), np.array(mrl_err)

def parameter_stability(data, thresholds):
    """Fits GPD across thresholds to check parameter stability."""
    shapes = []
    shape_errs = [] # Asymptotic standard errors
    mod_scales = []
    
    for u in thresholds:
        excesses = data[data > u] - u
        if len(excesses) < 20: # Require at least 20 events to attempt a fit
            shapes.append(np.nan)
            mod_scales.append(np.nan)
            continue
            
        # scipy genpareto uses 'c' for shape (xi) and 'scale' for beta
        # Notice: The standard scipy GPD fit is unconstrained. 
        # In HEP, xi is often ~0 (exponential tail) or slightly negative.
        c, loc, scale = genpareto.fit(excesses, floc=0)
        
        shapes.append(c)
        # Modified scale parameter: beta* = beta - xi * u
        mod_scales.append(scale - c * u)
        
    return np.array(shapes), np.array(mod_scales)

def extrapolate_significance(data, u, t_obs):
    """Fits GPD at chosen threshold and extrapolates to t_obs."""
    N = len(data)
    excesses = data[data > u] - u
    N_u = len(excesses)
    
    if N_u == 0:
        raise ValueError("Threshold too high, no excesses found.")

    # Fit the GPD
    xi, loc, beta = genpareto.fit(excesses, floc=0)
    
    # Calculate GPD survival function (1 - CDF)
    # P(T > t_obs | T > u)
    surv_prob = genpareto.sf(t_obs - u, c=xi, loc=0, scale=beta)
    
    # Global p-value: P(T > u) * P(T > t_obs | T > u)
    p_global = (N_u / N) * surv_prob
    
    # Convert to one-sided significance (Z-score)
    z_global = norm.isf(p_global)
    
    return xi, beta, p_global, z_global, N_u

def main():
    parser = argparse.ArgumentParser(description="EVT/POT Extrapolation for Global Significance")
    parser.add_argument('--input', type=str, required=True, help="Path to the generated numpy file containing test stats")
    parser.add_argument('--obs', type=float, default=6.61, help="Observed t_global in unblinded data")
    parser.add_argument('--plot-only', action='store_true', help="Only plot diagnostics to help choose threshold")
    parser.add_argument('--threshold', type=float, help="Chosen threshold (u) to perform the final extrapolation")
    args = parser.parse_args()

    # 1. Load data
    t_global = np.load(args.input)
    t_global = np.sort(t_global)
    
    # Define a range of thresholds to test (e.g., 80th to 99.5th percentile)
    percentiles = np.linspace(80, 99.5, 50)
    thresholds = np.percentile(t_global, percentiles)

    # 2. Run Diagnostics
    mrl, mrl_err = mean_residual_life(t_global, thresholds)
    shapes, mod_scales = parameter_stability(t_global, thresholds)

    # 3. Plot Diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # MRL Plot
    axes[0].errorbar(thresholds, mrl, yerr=mrl_err, fmt='o-', markersize=4)
    axes[0].set_title("Mean Residual Life Plot")
    axes[0].set_xlabel("Threshold (u)")
    axes[0].set_ylabel("Mean Excess")
    axes[0].grid(True, alpha=0.3)

    # Shape Parameter Stability
    axes[1].plot(thresholds, shapes, 'o-', markersize=4)
    axes[1].set_title("Shape Parameter ($\\xi$) Stability")
    axes[1].set_xlabel("Threshold (u)")
    axes[1].set_ylabel("Shape $\\xi$")
    axes[1].axhline(0, color='r', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    # Modified Scale Stability
    axes[2].plot(thresholds, mod_scales, 'o-', markersize=4)
    axes[2].set_title("Modified Scale ($\\beta^*$) Stability")
    axes[2].set_xlabel("Threshold (u)")
    axes[2].set_ylabel("Modified Scale")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/evt_diagnostics.png")
    print("Saved diagnostic plots to evt_diagnostics.png")

    # 4. Final Extrapolation (if threshold is provided)
    if not args.plot_only and args.threshold is not None:
        u = args.threshold
        xi, beta, p_global, z_global, N_u = extrapolate_significance(t_global, u, args.obs)
        
        print("\n" + "="*50)
        print(f"EXTRAPOLATION RESULTS")
        print("="*50)
        print(f"Total Toys (N)        : {len(t_global)}")
        print(f"Chosen Threshold (u)  : {u:.3f}")
        print(f"Excesses (N_u)        : {N_u}")
        print(f"Fitted Shape (xi)     : {xi:.4f}")
        print(f"Fitted Scale (beta)   : {beta:.4f}")
        print(f"Observed stat (t_obs) : {args.obs:.3f}")
        print("-" * 50)
        print(f"Global p-value        : {p_global:.2e}")
        print(f"Global Significance   : {z_global:.3f} sigma")
        print("="*50)

if __name__ == '__main__':
    main()
