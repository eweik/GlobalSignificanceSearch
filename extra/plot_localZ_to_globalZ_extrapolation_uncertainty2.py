#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, expon, norm
import argparse

# --- HELPER FUNCTIONS ---

def z_to_p(z):
    return norm.sf(z)

def p_to_z(p):
    return norm.isf(np.clip(p, 1e-15, 1.0)) 

def t_to_zlocal(t):
    p_local = np.exp(-t)
    return p_to_z(p_local)

def empirical_survival(data, t_eval):
    N = len(data)
    idx = np.searchsorted(data, t_eval)
    return (N - idx) / N

# --- UNCERTAINTY BAND GENERATION ---

def calculate_evt_bands(excesses, N_total, u, t_eval, n_boot=500):
    print(f"Running {n_boot} bootstrap iterations for uncertainty bands...")
    N_u_obs = len(excesses)
    p_global_boot = np.zeros((n_boot, len(t_eval)))
    
    for i in range(n_boot):
        N_u_boot = np.random.poisson(N_u_obs)
        if N_u_boot > 0:
            excesses_boot = np.random.choice(excesses, size=N_u_boot, replace=True)
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    xi_b, loc_b, beta_b = genpareto.fit(excesses_boot, floc=0)
                
                for j, t in enumerate(t_eval):
                    if t <= u:
                        p_global_boot[i, j] = np.nan 
                    else:
                        surv = genpareto.sf(t - u, c=xi_b, loc=0, scale=beta_b)
                        p_global_boot[i, j] = (N_u_boot / N_total) * surv
            except RuntimeError:
                p_global_boot[i, :] = np.nan
        else:
            p_global_boot[i, :] = np.nan

    lower_band = np.nanpercentile(p_global_boot, 16, axis=0)
    upper_band = np.nanpercentile(p_global_boot, 84, axis=0)
    
    return lower_band, upper_band

# --- MAIN SCRIPT ---

def main():
    parser = argparse.ArgumentParser(description="LEE Validation: POT vs Conditional Exponential")
    parser.add_argument('--toys100k', type=str, default="results/merged/copula100k.npy", help="Path to 100k toys .npy file")
    parser.add_argument('--toys10M', type=str, default="results/merged/copula20M.npy", help="Path to 10M toys .npy file")
    # parser.add_argument('--toys100k', type=str, required=True, help="Path to 100k toys .npy file")
    # parser.add_argument('--toys10M', type=str, required=True, help="Path to 10M toys .npy file")
    parser.add_argument('--threshold', type=float, default=8.5, help="EVT Threshold (u)")
    parser.add_argument('--bootstraps', type=int, default=500, help="Number of bootstrap iterations for bands")
    args = parser.parse_args()

    print("Loading data...")
    t_100k = np.sort(np.load(args.toys100k))
    t_10M = np.sort(np.load(args.toys10M))
    N_100k = len(t_100k)
    N_10M = len(t_10M)
    u = args.threshold

    t_eval = np.linspace(2, 28, 500)
    z_local_eval = t_to_zlocal(t_eval)

    # 1. EMPIRICAL TRUTH (10M Toys)
    print("Calculating Empirical 10M Truth...")
    p_global_10M = np.array([empirical_survival(t_10M, t) for t in t_eval])
    valid_10M = p_global_10M > 0
    z_global_10M = np.zeros_like(p_global_10M)
    z_global_10M[valid_10M] = p_to_z(p_global_10M[valid_10M])

    # 2. POT GPD EXTRAPOLATION (Flexible xi)
    print(f"Fitting POT GPD (u={u})...")
    excesses = t_100k[t_100k > u] - u
    N_u = len(excesses)
    
    # GPD Fit (xi is free to float)
    xi, loc, beta = genpareto.fit(excesses, floc=0)
    
    p_global_POT = np.zeros_like(t_eval)
    for i, t in enumerate(t_eval):
        if t <= u:
            p_global_POT[i] = empirical_survival(t_100k, t)
        else:
            # Here is the explicit conditioning step!
            surv = genpareto.sf(t - u, c=xi, loc=0, scale=beta)
            p_global_POT[i] = (N_u / N_100k) * surv
            
    z_global_POT = p_to_z(p_global_POT)

    # Calculate Uncertainty Bands for the POT GPD
    p_lower, p_upper = calculate_evt_bands(excesses, N_100k, u, t_eval, n_boot=args.bootstraps)
    valid_bands = ~np.isnan(p_lower) & ~np.isnan(p_upper) & (p_lower > 0) & (p_upper > 0)
    z_lower_band = np.zeros_like(t_eval)
    z_upper_band = np.zeros_like(t_eval)
    z_lower_band[valid_bands] = p_to_z(p_upper[valid_bands]) 
    z_upper_band[valid_bands] = p_to_z(p_lower[valid_bands]) 

    # 3. CONDITIONAL EXPONENTIAL FIT (Forced xi = 0)
    print(f"Fitting Conditional Exponential (u={u})...")
    # Fit standard exponential strictly to the excesses
    loc_exp, scale_exp = expon.fit(excesses, floc=0)
    
    p_global_CondExp = np.zeros_like(t_eval)
    for i, t in enumerate(t_eval):
        if t <= u:
            p_global_CondExp[i] = empirical_survival(t_100k, t)
        else:
            # Applying the exact same conditioning factorization, but with an exponential tail
            surv_exp = expon.sf(t - u, loc=0, scale=scale_exp)
            p_global_CondExp[i] = (N_u / N_100k) * surv_exp
            
    z_global_CondExp = p_to_z(p_global_CondExp)

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: The Closure Test
    axes[0].step(t_eval[valid_10M], p_global_10M[valid_10M], label='Empirical Truth (10M)', color='black', linewidth=2, zorder=3)
    
    p_empirical_100k = np.array([empirical_survival(t_100k, t) for t in t_eval])
    valid_100k = p_empirical_100k > 0
    axes[0].step(t_eval[valid_100k], p_empirical_100k[valid_100k], label='Empirical (100k)', color='gray', alpha=0.5, zorder=2)
    
    axes[0].fill_between(t_eval[valid_bands], p_lower[valid_bands], p_upper[valid_bands], color='blue', alpha=0.2, label=r'POT GPD $\pm 1\sigma$ Band')
    axes[0].plot(t_eval, p_global_POT, label=f'POT GPD (Flexible $\\xi$)', color='blue', linestyle='--', linewidth=2, zorder=4)
    axes[0].plot(t_eval, p_global_CondExp, label=r'Conditional Exp ($\xi=0$)', color='red', linestyle=':', linewidth=2, zorder=1)
    
    axes[0].axvline(u, color='green', linestyle='-', alpha=0.3, label='Threshold $u$')
    axes[0].set_yscale('log')
    axes[0].set_ylim(1e-9, 1.0)
    axes[0].set_xlim(2, 28)
    axes[0].set_xlabel(r'Test Statistic $t_{\mathrm{global}}$')
    axes[0].set_ylabel(r'Global p-value $P(T > t)$')
    axes[0].set_title('Closure Test: Conditioning on the Tail')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Final LEE Significance Curve
    axes[1].plot(z_local_eval, z_local_eval, label='No LEE ($Z_{\mathrm{global}} = Z_{\mathrm{local}}$)', color='black', linestyle=':', alpha=0.5)
    axes[1].plot(z_local_eval[valid_10M], z_global_10M[valid_10M], label='Actual $Z_{\mathrm{global}}$ (10M)', color='black', linewidth=2, zorder=3)
    
    axes[1].fill_between(z_local_eval[valid_bands], z_lower_band[valid_bands], z_upper_band[valid_bands], color='blue', alpha=0.2)
    axes[1].plot(z_local_eval, z_global_POT, label='POT GPD (Flexible $\\xi$)', color='blue', linestyle='--', linewidth=2, zorder=4)
    axes[1].plot(z_local_eval, z_global_CondExp, label=r'Conditional Exp ($\xi=0$)', color='red', linestyle='-.', linewidth=2, zorder=1)
    
    for z_bench in [3, 4, 5, 6]:
        axes[1].axvline(z_bench, color='gray', linestyle='-', alpha=0.2)
        
    axes[1].set_xlim(2, 7)
    axes[1].set_ylim(0, 7)
    axes[1].set_xlabel(r'Local Significance $Z_{\mathrm{local}}$ [$\sigma$]')
    axes[1].set_ylabel(r'Global Significance $Z_{\mathrm{global}}$ [$\sigma$]')
    axes[1].set_title('Look-Elsewhere Effect (LEE) Calibration')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/lee_final_validation_conditioned.png', dpi=300)
    print("Saved plots to lee_final_validation_conditioned.png")

if __name__ == '__main__':
    main()
