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
            
    # I: Integral over bin, S: Save result, M: Improve fit, R: Use TF1 range, 0: No draw, Q: Quiet
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
CHANNELS = ["jb", "bb", "je", "jm", "jg", "be", "bm", "bg"]

# Extracted from the row-normalized event overlap matrix
TRIGGER_OVERLAPS = {
    "t1": {
        "jj": 1.000, "bb": 0.754, "jb": 0.845, "je": 0.852, "jm": 0.849, 
        "jg": 0.729, "be": 0.772, "bm": 0.753, "bg": 0.622
    },
    "t2": {
        "jj": 1.000, "bb": 0.577, "jb": 0.770, "je": 0.831, "jm": 0.827, 
        "jg": 0.572, "be": 0.636, "bm": 0.634, "bg": 0.430
    },
    "t3": {
        "jj": 1.000, "bb": 0.208, "jb": 0.528, "je": 0.727, "jm": 0.741, 
        "jg": 0.341, "be": 0.341, "bm": 0.364, "bg": 0.256
    },
    "t4": {
        "jj": 1.000, "bb": 0.544, "jb": 0.741, "je": 0.573, "jm": 0.587, 
        "jg": 0.785, "be": 0.429, "bm": 0.476, "bg": 0.631
    },
    "t5": {
        "jj": 1.000, "bb": 0.333, "jb": 0.562, "je": 0.455, "jm": 0.737, 
        "jg": 0.849, "be": 0.300, "bm": 1.000, "bg": 0.405
    },
    "t6": {
        "jj": 1.000, "bb": 0.830, "jb": 0.900, "je": 0.943, "jm": 0.952, 
        "jg": 0.969, "be": 0.860, "bm": 0.874, "bg": 0.852
    },
    "t7": {
        "jj": 1.000, "bb": 0.923, "jb": 0.951, "je": 0.984, "jm": 0.986, 
        "jg": 0.998, "be": 0.975, "bm": 0.974, "bg": 0.993
    },
    "default": {
        "jj": 1.000, "bb": 0.0, "jb": 0.0, "je": 0.0, "jm": 0.0, 
        "jg": 0.0, "be": 0.0, "bm": 0.0, "bg": 0.0
    }
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
    parser = argparse.ArgumentParser(description="Visualize signal injection migration via empirical copula.")
    parser.add_argument('--trigger', type=str, default='t1', help="Trigger stream (e.g., t1, t2)")
    parser.add_argument('--mass', type=float, default=2000.0, help="Mass of the injected Gaussian signal (GeV)")
    parser.add_argument('--width', type=float, default=80.0, help="Width of the injected Gaussian signal (GeV)")
    parser.add_argument('--events', type=int, default=5000, help="Number of signal events to inject into the hub (M_jj)")
    args = parser.parse_args()
    
    trigger = args.trigger.lower()
    overlap_map = TRIGGER_OVERLAPS.get(trigger, TRIGGER_OVERLAPS["default"])

    np.random.seed(42) 
    
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

    # Perform ROOT Fits for M_jj labels
    chi2_jj_rand = fit_and_get_chi2(toy_jj_random, bins_jj, params_jj, fmin_jj, fmax_jj, "h_jj_rand")
    chi2_jj_base = fit_and_get_chi2(toy_jj_base, bins_jj, params_jj, fmin_jj, fmax_jj, "h_jj_base")
    
    plt.figure(figsize=(12, 7))
    plt.plot(centers_jj, B_jj, color='gray', linestyle='--', linewidth=1.5, label=f'Analytic Fit $H_0$ ($M_{{jj}}$)')
    
    label_jj_rand = f'Random $M_{{jj}}$ Toy [$\chi^2$/ndf={chi2_jj_rand:.2f}]'
    label_jj_base = f'Injected $M_{{jj}}$ Gaussian [$\chi^2$/ndf={chi2_jj_base:.2f}]'

    if PLOT_AS_POINTS:
        errors_jj_base = np.sqrt(toy_jj_base) if SHOW_ERRORS else None
        errors_jj_rand = np.sqrt(toy_jj_random) if SHOW_ERRORS else None
        plt.errorbar(centers_jj, toy_jj_random, yerr=errors_jj_rand, fmt='o', color='green', 
                     markersize=2, elinewidth=0.8, capsize=0, label=label_jj_rand, alpha=0.4)
        plt.errorbar(centers_jj, toy_jj_base, yerr=errors_jj_base, fmt='s', color='blue', 
                     markersize=3, elinewidth=0.8, capsize=0, label=label_jj_base, alpha=0.8)
    else:
        plt.step(centers_jj, toy_jj_random, where='mid', color='green', linewidth=1, label=label_jj_rand, alpha=0.4)
        plt.step(centers_jj, toy_jj_base, where='mid', color='blue', linewidth=1.5, label=label_jj_base, alpha=0.8)
    
    plt.axvline(args.mass, color='blue', linestyle=':', linewidth=1.5, label=f'Injection Peak ({args.mass} GeV)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(bins_jj[0], 6500) 
    plt.ylim(0.5, np.max(toy_jj_base) * 100)
    
    plt.title(f'Gaussian Injection: $M_{{jj}}$ Hub (Trigger: {trigger.upper()})', fontsize=16)
    plt.xlabel(f'Invariant Mass $M_{{jj}}$ [GeV]', fontsize=14)
    plt.ylabel('Events per Bin', fontsize=14)
    plt.legend(frameon=True, fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.15)
    plt.tight_layout()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    
    filename_jj = os.path.join(results_dir, f"spike_points_{trigger}_jj_original.png")
    plt.savefig(filename_jj, dpi=300)
    plt.close()
    print(f"Saved: {filename_jj}")

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

    for channel in CHANNELS:
        B_xy, bins_xy, params_xy, fmin_xy, fmax_xy = load_fit(trigger, channel)
        if B_xy is None: continue
        
        centers_xy = (bins_xy[:-1] + bins_xy[1:]) / 2
        eff = overlap_map.get(channel, 0.1)
        cdf_xy = np.cumsum(B_xy) / np.sum(B_xy)
        
        toy_naive = np.random.poisson(B_xy)
        
        aligned_residual_jj = np.interp(centers_xy, centers_jj, residual_jj)
        expected_linear = B_xy * (1 - eff) + B_xy * eff * (1 + aligned_residual_jj)
        toy_linear = np.random.poisson(np.maximum(expected_linear, 0))
        
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
        
        # Calculate chi2/ndf for each generated toy against the analytic hypothesis
        chi2_naive = fit_and_get_chi2(toy_naive, bins_xy, params_xy, fmin_xy, fmax_xy, f"h_{channel}_naive")
        chi2_linear = fit_and_get_chi2(toy_linear, bins_xy, params_xy, fmin_xy, fmax_xy, f"h_{channel}_linear")
        chi2_copula = fit_and_get_chi2(toy_copula, bins_xy, params_xy, fmin_xy, fmax_xy, f"h_{channel}_copula")

        plt.figure(figsize=(12, 7))
        plt.plot(centers_xy, B_xy, color='gray', linestyle='--', linewidth=1.5, label=f'Analytic Fit $H_0$ ($M_{{{channel}}}$)')
        
        plot_configs = [
            (f'Naive (No Transfer) [$\chi^2$/ndf={chi2_naive:.2f}]', toy_naive, 'red', 'o'),
            (f'Linear-Overlap [$\chi^2$/ndf={chi2_linear:.2f}]', toy_linear, 'orange', 's'),
            (f'Empirical Copula [$\chi^2$/ndf={chi2_copula:.2f}]', toy_copula, 'green', 'D')
        ]
        
        for label, counts, color, marker in plot_configs:
            if PLOT_AS_POINTS:
                errors = np.sqrt(counts) if SHOW_ERRORS else None
                plt.errorbar(centers_xy, counts, yerr=errors, fmt=marker, 
                             color=color, markersize=3, elinewidth=0.8, capsize=0, 
                             label=label, alpha=0.7)
            else:
                plt.step(centers_xy, counts, where='mid', color=color, linewidth=1.5, label=label, alpha=0.7)
        
        plt.axvline(args.mass, color='blue', linestyle=':', linewidth=1.5, label=f'Original $M_{{jj}}$ Injection ({args.mass} GeV)')
        
        if SHOW_COPULA_PEAK_LINE and copula_peak_mass is not None:
            plt.axvline(copula_peak_mass, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Copula Shift Peak (~{copula_peak_mass:.0f} GeV)')
        
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(bins_xy[0], 6500) 
        
        ymax_naive = np.max(toy_naive) + (np.max(np.sqrt(toy_naive)) if SHOW_ERRORS else 0)
        ymax_copula = np.max(toy_copula) + (np.max(np.sqrt(toy_copula)) if SHOW_ERRORS else 0)
        plt.ylim(0.5, max(ymax_naive, ymax_copula) * 50)
        
        plt.title(f'Kinematic Gaussian Propagation: $M_{{jj}} \\rightarrow M_{{{channel}}}$', fontsize=16)
        plt.xlabel(f'Invariant Mass $M_{{{channel}}}$ [GeV]', fontsize=14)
        plt.ylabel('Events per Bin', fontsize=14)
        plt.legend(frameon=True, fontsize=11, loc='upper right')
        plt.grid(True, alpha=0.15)
        
        plt.tight_layout()
        filename = os.path.join(results_dir, f"spike_points_{trigger}_{channel}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")

if __name__ == "__main__":
    main()
