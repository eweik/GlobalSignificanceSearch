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
CHANNELS_8 = ["jb", "bb", "je", "jm", "jg", "be", "bm", "bg"]
ALL_CHANNELS = ["jj"] + CHANNELS_8

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
    parser.add_argument('--events', type=int, default=5000, help="Number of signal events to inject")
    parser.add_argument('--inject', type=str, default='bb', help="Mass channel to inject the signal into (e.g., jj, bb, jb)")
    args = parser.parse_args()
    
    trigger = args.trigger.lower()
    inj_channel = args.inject.lower()
    overlap_map = TRIGGER_OVERLAPS.get(trigger, TRIGGER_OVERLAPS["default"])
    np.random.seed(42) 
    
    # ---------------------------------------------------------
    # 1. LOAD COPULA MATRIX FOR PROPAGATION
    # ---------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    copula_path = os.path.join(base_dir, "data", f"copula_{trigger}.npz")
    try:
        f = np.load(copula_path)
        matrix, cols = f['copula'], list(f['columns'])
        idx_jj = cols.index("Mjj")
        sort_idx_jj = np.argsort(matrix[:, idx_jj])
        sorted_matrix_jj = matrix[sort_idx_jj] # Sorted by Mjj for forward propagation
    except Exception as e:
        print(f"Error loading copula matrix from {copula_path}: {e}")
        return

    # ---------------------------------------------------------
    # 2. GENERATE SIGNAL & BACK-PROPAGATE VIA COPULA ONLY
    # ---------------------------------------------------------
    # Generate the initial raw signal in the injection channel
    sig_events_inj = np.random.normal(args.mass, args.width, args.events)
    
    B_jj, bins_jj, params_jj, fmin_jj, fmax_jj = load_fit(trigger, "jj")
    centers_jj = (bins_jj[:-1] + bins_jj[1:]) / 2
    cdf_jj = np.cumsum(B_jj) / np.sum(B_jj)
    
    # 2A: COPULA HUB (Correlated Back-propagation)
    if inj_channel == 'jj':
        sig_events_copula_jj = sig_events_inj
    else:
        B_inj, bins_inj, _, _, _ = load_fit(trigger, inj_channel)
        if B_inj is None:
            print(f"Error loading fit for injection channel {inj_channel}. Falling back to jj.")
            sig_events_copula_jj = sig_events_inj
        else:
            centers_inj = (bins_inj[:-1] + bins_inj[1:]) / 2
            cdf_inj = np.cumsum(B_inj) / np.sum(B_inj)
            
            ranks_inj = np.interp(sig_events_inj, centers_inj, cdf_inj)
            
            idx_inj = cols.index(f"M{inj_channel}")
            sort_idx_inj = np.argsort(matrix[:, idx_inj])
            sorted_matrix_inj = matrix[sort_idx_inj] 
            
            idx_closest_inj = np.searchsorted(sorted_matrix_inj[:, idx_inj], ranks_inj)
            idx_closest_inj = np.clip(idx_closest_inj, 0, len(sorted_matrix_inj) - 1)
            
            migrated_ranks_jj = sorted_matrix_inj[idx_closest_inj, idx_jj]
            valid_mask = migrated_ranks_jj >= 0
            sig_events_copula_jj = np.interp(migrated_ranks_jj[valid_mask], cdf_jj, centers_jj)

    # 2B: LINEAR HUB (Strictly literal - No back-propagation)
    # The Linear method assumes it can only propagate what exists natively in M_jj. 
    # If the signal wasn't injected into jj, the linear method sees absolutely nothing.
    if inj_channel == 'jj':
        sig_events_linear_jj = np.random.normal(args.mass, args.width, args.events)
    else:
        sig_events_linear_jj = np.array([])
        
    if len(sig_events_linear_jj) > 0:
        sig_hist_linear_jj, _ = np.histogram(sig_events_linear_jj, bins=bins_jj)
    else:
        sig_hist_linear_jj = np.zeros_like(B_jj)
        
    residual_jj_linear = np.where(B_jj > 0, sig_hist_linear_jj / B_jj, 0)

    # ---------------------------------------------------------
    # 3. PLOT ALL 9 PANELS UNIFORMLY
    # ---------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, channel in enumerate(ALL_CHANNELS):
        ax = axes[i]
        
        B_ch, bins_ch, params_ch, fmin_ch, fmax_ch = load_fit(trigger, channel)
        if B_ch is None: 
            ax.set_title(f'$M_{{{channel}}}$ (Data Missing)')
            ax.axis('off')
            continue
            
        centers_ch = (bins_ch[:-1] + bins_ch[1:]) / 2
        cdf_ch = np.cumsum(B_ch) / np.sum(B_ch)
        
        # =========================================================
        # BRANCH A: THIS IS THE INJECTION CHANNEL
        # Show exactly 2 things: The isolated background toy, and the background + true signal bump
        # =========================================================
        if channel == inj_channel:
            sig_hist_inj, _ = np.histogram(sig_events_inj, bins=bins_ch)
            toy_random = np.random.poisson(B_ch)
            toy_injected = np.random.poisson(B_ch) + sig_hist_inj
            
            chi2_rand = fit_and_get_chi2(toy_random, bins_ch, params_ch, fmin_ch, fmax_ch, f"h_{channel}_rand")
            chi2_inj = fit_and_get_chi2(toy_injected, bins_ch, params_ch, fmin_ch, fmax_ch, f"h_{channel}_inj")
            
            ax.plot(centers_ch, B_ch, color='gray', linestyle='--', linewidth=1.5, label='Analytic Fit $H_0$')
            
            plot_configs = [
                (f'Random Toy [$\chi^2$={chi2_rand:.1f}]', toy_random, 'green', 'o'),
                (f'Injected Signal [$\chi^2$={chi2_inj:.1f}]', toy_injected, 'blue', 's')
            ]
            
            for label, counts, color, marker in plot_configs:
                if PLOT_AS_POINTS:
                    errors = np.sqrt(counts) if SHOW_ERRORS else None
                    ax.errorbar(centers_ch, counts, yerr=errors, fmt=marker, color=color, markersize=3, elinewidth=0.8, label=label, alpha=0.7)
                else:
                    ax.step(centers_ch, counts, where='mid', color=color, linewidth=1.5, label=label, alpha=0.7)
                    
            ax.axvline(args.mass, color='blue', linestyle=':', linewidth=2, label=f'True Inj ({args.mass})')
            ax.set_title(f'INJECTION CHANNEL: $M_{{{channel}}}$', fontsize=14, fontweight='bold', color='navy')

        # =========================================================
        # BRANCH B: ALL OTHER CHANNELS (INCLUDING M_jj IF IT WAS NOT INJECTED)
        # Show the Migration Methods: Naive, Linear Overlap, and Copula
        # =========================================================
        else:
            eff = 1.0 if channel == 'jj' else overlap_map.get(channel, 0.1)
            
            toy_naive = np.random.poisson(B_ch)
            
            # --- Linear Overlap (Propagated from the strictly forward M_jj literal assumption) ---
            aligned_residual_linear = np.interp(centers_ch, centers_jj, residual_jj_linear)
            ov_counts_linear = (B_ch * eff) * (1 + aligned_residual_linear)
            ind_b = np.maximum(0, B_ch * (1 - eff))
            ind_counts_linear = np.random.poisson(ind_b)
            toy_linear = np.maximum(0, np.round(ov_counts_linear + ind_counts_linear).astype(int))
            
            # --- Copula Forward Propagation (From the properly back-propagated M_jj hub) ---
            toy_copula = np.random.poisson(B_ch)
            copula_peak_mass = None 
            
            if channel == 'jj':
                # If we are plotting the jj hub, the copula signal is the back-propagated data
                sig_hist_ch, _ = np.histogram(sig_events_copula_jj, bins=bins_ch)
                if np.sum(sig_hist_ch) > 0:
                    toy_copula += np.random.poisson(sig_hist_ch)
                    copula_peak_mass = centers_ch[np.argmax(sig_hist_ch)]
            else:
                # If we are plotting a sub-channel, forward propagate from the back-propagated jj hub
                idx_ch = cols.index(f"M{channel}")
                ranks_jj = np.interp(sig_events_copula_jj, centers_jj, cdf_jj)
                idx_closest = np.searchsorted(sorted_matrix_jj[:, idx_jj], ranks_jj)
                idx_closest = np.clip(idx_closest, 0, len(sorted_matrix_jj)-1)
                
                migrated_ranks_ch = sorted_matrix_jj[idx_closest, idx_ch]
                valid_ranks = migrated_ranks_ch[migrated_ranks_ch >= 0]
                
                if len(valid_ranks) > 0:
                    migrated_masses_ch = np.interp(valid_ranks, cdf_ch, centers_ch)
                    sig_hist_ch, _ = np.histogram(migrated_masses_ch, bins=bins_ch)
                    
                    if np.sum(sig_hist_ch) > 0:
                        b_ch_local = np.interp(args.mass, centers_ch, B_ch)
                        b_jj_local = np.interp(args.mass, centers_jj, B_jj)
                        acceptance_ratio = b_ch_local / b_jj_local if b_jj_local > 0 else 0
                        
                        target_events = args.events * eff * acceptance_ratio
                        sig_hist_ch = sig_hist_ch * (target_events / np.sum(sig_hist_ch))
                        
                        toy_copula += np.random.poisson(sig_hist_ch)
                        copula_peak_mass = centers_ch[np.argmax(sig_hist_ch)]
            
            chi2_naive = fit_and_get_chi2(toy_naive, bins_ch, params_ch, fmin_ch, fmax_ch, f"h_{channel}_naive")
            chi2_linear = fit_and_get_chi2(toy_linear, bins_ch, params_ch, fmin_ch, fmax_ch, f"h_{channel}_linear")
            chi2_copula = fit_and_get_chi2(toy_copula, bins_ch, params_ch, fmin_ch, fmax_ch, f"h_{channel}_copula")

            ax.plot(centers_ch, B_ch, color='gray', linestyle='--', linewidth=1.5, label='Analytic Fit $H_0$')
            
            plot_configs = [
                (f'Naive [$\chi^2$={chi2_naive:.1f}]', toy_naive, 'red', 'o'),
                (f'Linear [$\chi^2$={chi2_linear:.1f}]', toy_linear, 'orange', 's'),
                (f'Copula [$\chi^2$={chi2_copula:.1f}]', toy_copula, 'green', 'D')
            ]
            
            for label, counts, color, marker in plot_configs:
                if PLOT_AS_POINTS:
                    errors = np.sqrt(counts) if SHOW_ERRORS else None
                    ax.errorbar(centers_ch, counts, yerr=errors, fmt=marker, color=color, markersize=3, elinewidth=0.8, label=label, alpha=0.7)
                else:
                    ax.step(centers_ch, counts, where='mid', color=color, linewidth=1.5, label=label, alpha=0.7)
            
            if SHOW_COPULA_PEAK_LINE and copula_peak_mass is not None:
                ax.axvline(copula_peak_mass, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label='Copula Peak')

            if channel == 'jj':
                ax.set_title(f'HUB Comparison: $M_{{jj}}$', fontsize=14, fontweight='bold', color='purple')
            else:
                ax.set_title(f'Shift to $M_{{{channel}}}$', fontsize=14)

        # =========================================================
        # COMMON AESTHETICS FOR ALL PANELS
        # =========================================================
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(bins_ch[0], 6500) 
        
        if channel == inj_channel:
            ymax = max(np.max(toy_random), np.max(toy_injected))
        else:
            ymax = max(np.max(toy_naive), np.max(toy_copula))
            
        ax.set_ylim(0.5, (ymax + np.sqrt(ymax)) * 50)
        ax.set_xlabel('Mass [GeV]', fontsize=11)
        ax.set_ylabel('Events per Bin', fontsize=11)
        ax.legend(frameon=True, fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.15)

    # ---------------------------------------------------------
    # 4. FINALIZE AND SAVE GRID FIGURE
    # ---------------------------------------------------------
    fig.suptitle(f'Kinematic Shift: {trigger.upper()} Trigger | {args.events} events initially injected into M_{{{inj_channel}}} @ {args.mass} GeV', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    
    results_dir = os.path.join(base_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f"grid_spectra_{trigger}_M{int(args.mass)}_inj_{inj_channel}.png")
    
    plt.savefig(filename, dpi=200) 
    plt.close()
    print(f"\nSuccess! 9-Panel grid saved to: {filename}")

if __name__ == "__main__":
    main()
