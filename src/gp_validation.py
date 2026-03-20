import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_gp_validation(centers, widths, data_counts, gp_density, gp_err_density, channel, trigger, out_dir="plots"):
    """Generates the mass spectrum fit and the pull distribution."""
    
    data_density = data_counts / widths
    data_err_density = np.sqrt(np.maximum(data_counts, 0)) / widths
    
    # Mask empty bins for the pull calculation to avoid division by zero
    mask = data_counts > 0
    pulls = (data_density[mask] - gp_density[mask]) / data_err_density[mask]

    fig, (ax_main, ax_pull, ax_hist) = plt.subplots(
        3, 1, figsize=(10, 12), 
        gridspec_kw={'height_ratios': [3, 1, 1.5]}
    )

    # --- Top Panel: Spectrum ---
    ax_main.errorbar(centers, data_density, yerr=data_err_density, fmt='ko', markersize=3, label='1% Data')
    ax_main.plot(centers, gp_density, 'b-', label='GP Fit')
    ax_main.fill_between(centers, gp_density - gp_err_density, gp_density + gp_err_density, color='b', alpha=0.2, label='GP Uncertainty')
    ax_main.set_yscale('log')
    ax_main.set_ylabel('Events / GeV')
    ax_main.set_title(f'GP Background Fit: Trigger {trigger}, Channel {channel}')
    ax_main.legend()

    # --- Middle Panel: Pulls vs Mass ---
    ax_pull.axhline(0, color='black', linestyle='--')
    ax_pull.scatter(centers[mask], pulls, color='black', s=10)
    ax_pull.set_ylabel('Pull (Data-Fit)/Err')
    ax_pull.set_xlabel('m [GeV]')
    ax_pull.set_ylim(-4, 4)

    # --- Bottom Panel: Pull Distribution ---
    counts, bins, _ = ax_hist.hist(pulls, bins=np.linspace(-4, 4, 30), density=True, alpha=0.6, color='gray')
    mu, std = norm.fit(pulls)
    x_pdf = np.linspace(-4, 4, 100)
    p = norm.pdf(x_pdf, mu, std)
    ax_hist.plot(x_pdf, p, 'k', linewidth=2, label=f'Fit: $\mu={mu:.2f}, \sigma={std:.2f}$')
    ax_hist.set_xlabel('Pull')
    ax_hist.set_ylabel('Density')
    ax_hist.legend()

    plt.tight_layout()
    os.makedirs(f"{out_dir}", exist_ok=True)
    plt.savefig(f"{out_dir}/GP_Validation_{trigger}_{channel}.png", dpi=300)
    plt.close()
    return mu, std
