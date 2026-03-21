import numpy as np
import json
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse
import array
import ROOT

# Suppress ROOT popups and fitting print spam
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kFatal

# ==========================================
# ROOT FITTING CLASS
# ==========================================
class FiveParam2015:
    """ROOT callable for the 5-parameter background fit"""
    def __call__(self, x, par):
        xx = x[0] / 13600.0  # Assuming 13.6 TeV CMS energy for Run 3
        if xx <= 0 or xx >= 1: return 0.0
        ff1 = par[0] * ROOT.TMath.Power((1.0 - xx), par[1])
        ff2 = ROOT.TMath.Power(xx, (par[2] + par[3] * ROOT.TMath.Log(xx) + par[4] * ROOT.TMath.Log(xx) * ROOT.TMath.Log(xx)))
        return ff1 * ff2

def fit_and_get_chi2(counts, bins, params, fmin, fmax, name="hist"):
    """Creates a ROOT TH1D, fits it with TF1, and returns the chi2/ndf."""
    edges = array.array('d', bins)
    h = ROOT.TH1D(name, name, len(edges)-1, edges)
    h.SetDirectory(0)
    
    for i, val in enumerate(counts):
        if val > 0:
            h.SetBinContent(i+1, val)
            h.SetBinError(i+1, ROOT.TMath.Sqrt(val))
            
    tf1 = ROOT.TF1(f"tf1_{name}", FiveParam2015(), fmin, fmax, 5)
    for i, p in enumerate(params[:5]):
        tf1.SetParameter(i, p)
        if p == 0.0: 
            tf1.FixParameter(i, 0.0)
            
    fit_res = h.Fit(tf1, "ISMR0Q")
    ndf = tf1.GetNDF()
    chi2 = tf1.GetChisquare()
    return chi2 / ndf if ndf > 0 else float('inf')


# ==========================================
# VISUALIZATION CONFIGURATION FLAGS
# ==========================================
PLOT_AS_POINTS        = True 
SHOW_ERRORS           = True 
SHOW_COPULA_PEAK_LINE = False

ATLAS_BINS = np.array([99,112,125,138,151,164,177,190, 203, 216, 229, 243, 257, 272, 287, 303, 319, 335, 352, 369, 387, 405, 424, 443, 462, 482, 502, 523, 544, 566, 588, 611, 634, 657, 681, 705, 730, 755, 781, 807, 834, 861, 889, 917, 946, 976, 1006, 1037, 1068, 1100, 1133, 1166, 1200, 1234, 1269, 1305, 1341, 1378, 1416, 1454, 1493, 1533, 1573, 1614, 1656, 1698, 1741, 1785, 1830, 1875, 1921, 1968, 2016, 2065, 2114, 2164, 2215, 2267, 2320, 2374, 2429, 2485, 2542, 2600, 2659, 2719, 2780, 2842, 2905, 2969, 3034, 3100, 3167, 3235, 3305, 3376, 3448, 3521, 3596, 3672, 3749, 3827, 3907, 3988, 4070, 4154, 4239, 4326, 4414, 4504, 4595, 4688, 4782, 4878, 4975, 5074, 5175, 5277, 5381, 5487, 5595, 5705, 5817, 5931, 6047, 6165, 6285, 6407, 6531, 6658, 6787, 6918, 7052, 7188, 7326, 7467, 7610, 7756, 7904, 8055, 8208, 8364, 8523, 8685, 8850, 9019, 9191, 9366, 9544, 9726, 9911, 10100, 10292, 10488, 10688, 10892, 11100, 11312, 11528, 11748, 11972, 12200, 12432, 12669, 12910, 13156])
CHANNELS_8 = ["jb", "bb", "je", "jm", "jg", "be", "bm", "bg"] # 8 propagation channels

TRIGGER_OVERLAPS = {
    "t1": { "jj": 1.000, "bb": 0.754, "jb": 0.845, "je": 0.852, "jm": 0.849, "jg": 0.729, "be": 0.772, "bm": 0.753, "bg": 0.622 },
    "t2": { "jj": 1.000, "bb": 0.577, "jb": 0.770, "je": 0.831, "jm": 0.827, "jg": 0.572, "be": 0.636, "bm": 0.634, "bg": 0.430 },
    "t3": { "jj": 1.000, "bb": 0.208, "jb": 0.528, "je": 0.727, "jm": 0.741, "jg": 0.341, "be": 0.341, "bm": 0.364, "bg": 0.256 },
    "t4": { "jj": 1.000, "bb": 0.544, "jb": 0.741, "je": 0.573, "jm": 0.587, "jg": 0.785, "be": 0.429, "bm": 0.476, "bg": 0.631 },
    "t5": { "jj": 1.000, "bb": 0.333, "jb": 0.562, "je": 0.455, "jm": 0.737, "jg": 0.849, "be": 0.300, "bm": 1.000, "bg": 0.405 },
    "t6": { "jj": 1.000, "bb": 0.830, "jb": 0.900, "je": 0.943, "jm": 0.952, "jg": 0.969, "be": 0.860, "bm": 0.874, "bg": 0.852 },
    "t7": { "jj": 1.000, "bb": 0.923, "jb": 0.951, "je": 0.984, "jm": 0.986, "jg": 0.998, "be": 0.975, "bm": 0.974, "bg": 0.993 },
    "default": { "jj": 1.000, "bb": 0.0, "jb": 0.0, "je": 0.0, "jm": 0.0, "jg": 0.0, "be": 0.0, "bm": 0.0, "bg": 0.0 }
}

def FiveParam_NP(Ecm, x_center, p1, p2, p3, p4, p5):
    x = x_center / Ecm
    nlog = np.log(x)
    return p1 * np.power((1.0 - x), p2) * np.power(x, (p3 + p4 * nlog + p5 * nlog * nlog))

def load_fit(trigger, channel):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{channel}.json")
    
    if not os.path.exists(fitfile): 
        print(f"Warning: Fit file not found -> {fitfile}")
        return None, None, None, None, None
        
    with open(fitfile, "r") as j:
        d = json.load(j)
        fmin, fmax = float(d['fmin']), float(d['fmax'])
        v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin) & (ATLAS_BINS <= fmax)]
        c = (v_bins[:-1] + v_bins[1:]) / 2
        widths = np.diff(v_bins)
        params = [float(p) for p in d['parameters']]
        counts = FiveParam_NP(13600., c, *params[:5]) * widths
        return counts, v_bins, params, fmin, fmax

def main():
    parser = argparse.ArgumentParser(description="Visualize signal injection migration (9-Panel Grid).")
    parser.add_argument('--trigger', type=str, default='t1', help="Trigger stream (e.g., t1, t2)")
    parser.add_argument('--mass', type=float, default=2000.0, help="Mass of the injected Gaussian signal (GeV)")
    parser.add_argument('--width', type=float, default=80.0, help="Width of the injected Gaussian signal (GeV)")
    parser.add_argument('--events', type=int, default=5000, help="Number of signal events to inject into the hub (M_jj)")
    args = parser.parse_args()
    
    trigger = args.trigger.lower()
    overlap_map = TRIGGER_OVERLAPS.get(trigger, TRIGGER_OVERLAPS["default"])
    np.random.seed(42) 
    
    # Setup Figure and 3x3 Grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten() # Flatten to 1D array of 9 axes for easier indexing
    
    # ---------------------------------------------------------
    # 1. PROCESS AND PLOT PRIMARY HUB (M_jj) ON axes[0]
    # ---------------------------------------------------------
    B_jj, bins_jj, params_jj, fmin_jj, fmax_jj = load_fit(trigger, "jj")
    if B_jj is None:
        print(f"Error: Could not load M_jj fit for {trigger}.")
        return
        
    centers_jj = (bins_jj[:-1] + bins_jj[1:]) / 2
    cdf_jj = np.cumsum(B_jj) / np.sum(B_jj)
    
    toy_jj_random = np.random.poisson(B_jj)
    sig_events_jj = np.random.normal(args.mass, args.width, args.events)
    sig_hist_jj, _ = np.histogram(sig_events_jj, bins=bins_jj)
    
    toy_jj_base = np.random.poisson(B_jj) + sig_hist_jj
    residual_jj = np.where(B_jj > 0, (toy_jj_base - B_jj) / B_jj, 0)

    chi2_jj_rand = fit_and_get_chi2(toy_jj_random, bins_jj, params_jj, fmin_jj, fmax_jj, "h_jj_rand")
    chi2_jj_base = fit_and_get_chi2(toy_jj_base, bins_jj, params_jj, fmin_jj, fmax_jj, "h_jj_base")
    
    ax_jj = axes[0]
    ax_jj.plot(centers_jj, B_jj, color='gray', linestyle='--', linewidth=1.5, label=f'Analytic Fit $H_0$')
    
    label_jj_rand = f'Random Toy [$\chi^2$={chi2_jj_rand:.1f}]'
    label_jj_base = f'Injected Signal [$\chi^2$={chi2_jj_base:.1f}]'

    if PLOT_AS_POINTS:
        err_base = np.sqrt(toy_jj_base) if SHOW_ERRORS else None
        err_rand = np.sqrt(toy_jj_random) if SHOW_ERRORS else None
        ax_jj.errorbar(centers_jj, toy_jj_random, yerr=err_rand, fmt='o', color='green', markersize=2, elinewidth=0.8, label=label_jj_rand, alpha=0.4)
        ax_jj.errorbar(centers_jj, toy_jj_base, yerr=err_base, fmt='s', color='blue', markersize=3, elinewidth=0.8, label=label_jj_base, alpha=0.8)
    else:
        ax_jj.step(centers_jj, toy_jj_random, where='mid', color='green', linewidth=1, label=label_jj_rand, alpha=0.4)
        ax_jj.step(centers_jj, toy_jj_base, where='mid', color='blue', linewidth=1.5, label=label_jj_base, alpha=0.8)
    
    ax_jj.axvline(args.mass, color='blue', linestyle=':', linewidth=1.5, label=f'Injection ({args.mass} GeV)')
    ax_jj.set_xscale('log')
    ax_jj.set_yscale('log')
    ax_jj.set_xlim(bins_jj[0], 6500) 
    ax_jj.set_ylim(0.5, np.max(toy_jj_base) * 100)
    ax_jj.set_title(f'PRIMARY CHANNEL: $M_{{jj}}$', fontsize=14, fontweight='bold')
    ax_jj.set_xlabel('Mass [GeV]', fontsize=11)
    ax_jj.set_ylabel('Events per Bin', fontsize=11)
    ax_jj.legend(frameon=True, fontsize=9, loc='upper right')
    ax_jj.grid(True, alpha=0.15)

    # ---------------------------------------------------------
    # 2. LOAD COPULA MATRIX FOR SUB-CHANNELS
    # ---------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    copula_path = os.path.join(base_dir, "data", f"copula_{trigger}.npz")
    try:
        f = np.load(copula_path)
        matrix, cols = f['copula'], list(f['columns'])
        idx_jj = cols.index("Mjj")
        sort_idx = np.argsort(matrix[:, idx_jj])
        sorted_matrix = matrix[sort_idx]
    except Exception as e:
        print(f"Error loading copula matrix from {copula_path}: {e}")
        return

    # ---------------------------------------------------------
    # 3. PROCESS AND PLOT THE 8 SUB-CHANNELS ON axes[1] to axes[8]
    # ---------------------------------------------------------
    for i, channel in enumerate(CHANNELS_8):
        ax = axes[i + 1] # Offset by 1 because axes[0] is jj
        
        B_xy, bins_xy, params_xy, fmin_xy, fmax_xy = load_fit(trigger, channel)
        if B_xy is None: 
            ax.set_title(f'$M_{{{channel}}}$ (Data Missing)')
            ax.axis('off')
            continue
            
        centers_xy = (bins_xy[:-1] + bins_xy[1:]) / 2
        eff = overlap_map.get(channel, 0.1)
        cdf_xy = np.cumsum(B_xy) / np.sum(B_xy)
        
        toy_naive = np.random.poisson(B_xy)
        
        # --- FIXED LINEAR TOY GENERATION ---
        aligned_residual_jj = np.interp(centers_xy, centers_jj, residual_jj)
        ov_counts = (B_xy * eff) * (1 + aligned_residual_jj)
        
        ind_b = np.maximum(0, B_xy * (1 - eff))
        ind_counts = np.random.poisson(ind_b)
        
        toy_linear = np.maximum(0, np.round(ov_counts + ind_counts).astype(int))
        # -----------------------------------
        
        idx_xy = cols.index(f"M{channel}")
        toy_copula = np.random.poisson(B_xy)
        copula_peak_mass = None 
        
        ranks_jj = np.interp(sig_events_jj, centers_jj, cdf_jj)
        idx_closest = np.searchsorted(sorted_matrix[:, idx_jj], ranks_jj)
        idx_closest = np.clip(idx_closest, 0, len(sorted_matrix)-1)
        
        migrated_ranks_xy = sorted_matrix[idx_closest, idx_xy]
        valid_ranks = migrated_ranks_xy[migrated_ranks_xy >= 0]
        
        if len(valid_ranks) > 0:
            migrated_masses_xy = np.interp(valid_ranks, cdf_xy, centers_xy)
            sig_hist_xy, _ = np.histogram(migrated_masses_xy, bins=bins_xy)
            
            if np.sum(sig_hist_xy) > 0:
                sig_hist_xy = sig_hist_xy * ( (args.events * eff) / np.sum(sig_hist_xy) )
                toy_copula += np.random.poisson(sig_hist_xy)
                peak_idx = np.argmax(sig_hist_xy)
                copula_peak_mass = centers_xy[peak_idx]
        
        chi2_naive = fit_and_get_chi2(toy_naive, bins_xy, params_xy, fmin_xy, fmax_xy, f"h_{channel}_naive")
        chi2_linear = fit_and_get_chi2(toy_linear, bins_xy, params_xy, fmin_xy, fmax_xy, f"h_{channel}_linear")
        chi2_copula = fit_and_get_chi2(toy_copula, bins_xy, params_xy, fmin_xy, fmax_xy, f"h_{channel}_copula")

        ax.plot(centers_xy, B_xy, color='gray', linestyle='--', linewidth=1.5, label='Analytic Fit $H_0$')
        
        plot_configs = [
            (f'Naive [$\chi^2$={chi2_naive:.1f}]', toy_naive, 'red', 'o'),
            (f'Linear [$\chi^2$={chi2_linear:.1f}]', toy_linear, 'orange', 's'),
            # (f'Copula [$\chi^2$={chi2_copula:.1f}]', toy_copula, 'green', 'D')
        ]
        
        for label, counts, color, marker in plot_configs:
            if PLOT_AS_POINTS:
                errors = np.sqrt(counts) if SHOW_ERRORS else None
                ax.errorbar(centers_xy, counts, yerr=errors, fmt=marker, color=color, markersize=3, elinewidth=0.8, label=label, alpha=0.7)
            else:
                ax.step(centers_xy, counts, where='mid', color=color, linewidth=1.5, label=label, alpha=0.7)
        
        ax.axvline(args.mass, color='blue', linestyle=':', linewidth=1.5) # Removed label to save space
        if SHOW_COPULA_PEAK_LINE and copula_peak_mass is not None:
            ax.axvline(copula_peak_mass, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
        
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(bins_xy[0], 6500) 
        
        ymax = max(np.max(toy_naive), np.max(toy_copula)) + (np.max(np.sqrt(toy_copula)) if SHOW_ERRORS else 0)
        ax.set_ylim(0.5, ymax * 50)
        
        ax.set_title(f'Shift to $M_{{{channel}}}$', fontsize=16)
        ax.set_xlabel('Mass [GeV]', fontsize=11)
        ax.set_ylabel('Events per Bin', fontsize=11)
        ax.legend(frameon=True, fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.15)

    # ---------------------------------------------------------
    # 4. FINALIZE AND SAVE GRID FIGURE
    # ---------------------------------------------------------
    fig.suptitle(f'Kinematic Shift: {trigger.upper()} Trigger | {args.events} injected @ {args.mass} GeV', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Leave room for suptitle
    
    results_dir = os.path.join(base_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f"grid_spectra_{trigger}_M{int(args.mass)}.png")
    
    plt.savefig(filename, dpi=200) # Lowered DPI slightly for a large 9-panel plot to keep file size reasonable
    plt.close()
    print(f"\nSuccess! 9-Panel grid saved to: {filename}")

if __name__ == "__main__":
    main()
