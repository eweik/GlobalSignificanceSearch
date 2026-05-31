#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, norm
from scipy.optimize import curve_fit
import argparse

# --- HELPER FUNCTIONS ---

def z_to_p(z):
    """Convert Significance (Z-score) to one-sided p-value."""
    return norm.sf(z)

def p_to_z(p):
    """Convert one-sided p-value to Significance (Z-score)."""
    # Clip p to avoid inf Z-scores, capping at ~8 sigma
    return norm.isf(np.clip(p, 1e-15, 1.0)) 

def t_to_zlocal(t):
    """Convert BumpHunter test statistic t = -ln(p) to local Z-score."""
    p_local = np.exp(-t)
    return p_to_z(p_local)

def empirical_survival(data, t_eval):
    """Calculate the empirical probability P(T > t)."""
    N = len(data)
    # Using searchsorted for speed on sorted arrays
    idx = np.searchsorted(data, t_eval)
    return (N - idx) / N

def naive_fit_func(t, A, B):
    """Naive exponential decay for the survival function tail."""
    return A * np.exp(-B * t)

# --- MAIN SCRIPT ---

def main():
    parser = argparse.ArgumentParser(description="Final LEE Validation Plots")
    parser.add_argument('--toys100k', type=str, default="results/merged/copula100k.npy", help="Path to 100k toys .npy file")
    parser.add_argument('--toys10M', type=str, default="results/merged/copula10M.npy", help="Path to 10M toys .npy file")
    # parser.add_argument('--toys100k', type=str, required=True, help="Path to 10M toys .npy file")
    # parser.add_argument('--toys10M', type=str, required=True, help="Path to 10M toys .npy file")
    parser.add_argument('--threshold', type=float, default=8.5, help="EVT Threshold (u)")
    args = parser.parse_args()

    print("Loading data...")
    t_100k = np.sort(np.load(args.toys100k)[:10000])
    # t_100k = np.sort(np.load(args.toys100k))
    t_10M = np.sort(np.load(args.toys10M))
    N_100k = len(t_100k)
    print(f"number of 100k toys: {N_100k}")
    N_10M = len(t_10M)
    u = args.threshold

    # Define the t range we want to evaluate (up to Local 7-sigma)
    t_eval = np.linspace(2, 28, 500)
    z_local_eval = t_to_zlocal(t_eval)

    # 1. EMPIRICAL TRUTH (10M Toys)
    print("Calculating Empirical 10M Truth...")
    p_global_10M = np.array([empirical_survival(t_10M, t) for t in t_eval])
    # Filter out exactly 0 probabilities to avoid log/Z-score errors
    valid_10M = p_global_10M > 0
    z_global_10M = np.zeros_like(p_global_10M)
    z_global_10M[valid_10M] = p_to_z(p_global_10M[valid_10M])

    # 2. POT EXTRAPOLATION (100k Toys)
    print(f"Fitting POT EVT (u={u})...")
    excesses = t_100k[t_100k > u] - u
    N_u = len(excesses)
    xi, loc, beta = genpareto.fit(excesses, floc=0)
    
    p_global_POT = np.zeros_like(t_eval)
    for i, t in enumerate(t_eval):
        if t <= u:
            p_global_POT[i] = empirical_survival(t_100k, t)
        else:
            surv = genpareto.sf(t - u, c=xi, loc=0, scale=beta)
            p_global_POT[i] = (N_u / N_100k) * surv
    z_global_POT = p_to_z(p_global_POT)

    # 3. NAIVE CURVE FIT (100k Toys)
    print("Fitting Naive Exponential to 100k tail...")
    # Fit the exponential function only to the semi-tail (e.g., t between 4 and 10)
    # where the 100k dataset still has smooth statistics
    fit_mask = (t_100k > 4) & (t_100k < 10)
    t_fit_data = t_100k[fit_mask]
    p_fit_data = np.array([empirical_survival(t_100k, t) for t in t_fit_data])
    
    # We fit ln(p) = ln(A) - B*t for stability
    popt, _ = curve_fit(lambda t, lnA, B: lnA - B*t, t_fit_data, np.log(p_fit_data))
    lnA_opt, B_opt = popt
    A_opt = np.exp(lnA_opt)
    
    p_global_Naive = naive_fit_func(t_eval, A_opt, B_opt)
    z_global_Naive = p_to_z(p_global_Naive)

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: The Closure Test (Survival Function)
    axes[0].step(t_eval[valid_10M], p_global_10M[valid_10M], label='Empirical Truth (10M)', color='black', linewidth=2, zorder=3)
    
    # Generate the jagged empirical curve for 100k
    p_empirical_100k = np.array([empirical_survival(t_100k, t) for t in t_eval])
    valid_100k = p_empirical_100k > 0
    axes[0].step(t_eval[valid_100k], p_empirical_100k[valid_100k], label='Empirical (100k)', color='gray', alpha=0.5, zorder=2)
    
    axes[0].plot(t_eval, p_global_POT, label=f'POT EVT (u={u})', color='blue', linestyle='--', linewidth=2, zorder=4)
    axes[0].plot(t_eval, p_global_Naive, label='Naive Exponential Fit', color='red', linestyle=':', linewidth=2, zorder=1)
    
    axes[0].axvline(u, color='green', linestyle='-', alpha=0.3, label='EVT Threshold $u$')
    axes[0].set_yscale('log')
    axes[0].set_ylim(1e-9, 1.0)
    axes[0].set_xlim(2, 28)
    axes[0].set_xlabel(r'Test Statistic $t_{\mathrm{global}}$')
    axes[0].set_ylabel(r'Global p-value $P(T > t)$')
    axes[0].set_title('Closure Test: Survival Function Tail')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Final LEE Significance Curve
    axes[1].plot(z_local_eval, z_local_eval, label='No LEE ($Z_{\mathrm{global}} = Z_{\mathrm{local}}$)', color='black', linestyle=':', alpha=0.5)
    axes[1].plot(z_local_eval[valid_10M], z_global_10M[valid_10M], label='Actual $Z_{\mathrm{global}}$ (10M Toys)', color='black', linewidth=2, zorder=3)
    axes[1].plot(z_local_eval, z_global_POT, label='POT Extrapolation (100k Toys)', color='blue', linestyle='--', linewidth=2, zorder=4)
    axes[1].plot(z_local_eval, z_global_Naive, label='Naive Curve Fit (100k Toys)', color='red', linestyle='-.', linewidth=2, zorder=1)
    
    # Add vertical lines for benchmark significances
    for z_bench in [3, 4, 5, 6]:
        axes[1].axvline(z_bench, color='gray', linestyle='-', alpha=0.2)
        
    axes[1].set_xlim(2, 7)
    axes[1].set_ylim(0, 7)
    axes[1].set_xlabel(r'Local Significance $Z_{\mathrm{local}}$ [$\sigma$]')
    axes[1].set_ylabel(r'Global Significance $Z_{\mathrm{global}}$ [$\sigma$]')
    axes[1].set_title('Look-Elsewhere Effect (LEE) Comparisons')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/lee_final_validation.png', dpi=300)
    print("Saved plots to lee_final_validation.png")

if __name__ == '__main__':
    main()
