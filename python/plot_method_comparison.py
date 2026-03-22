#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from argparse import ArgumentParser

current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from src.config import ATLAS_BINS, TRIGGER_OVERLAPS
from src.models import FiveParam
from src.stats import fast_bumphunter_stat

ALL_CHANNELS = ["jj", "bb", "jb", "je", "jm", "jg", "be", "bm", "bg"]

def t_to_z(t):
    """Converts BumpHunter test statistic (-ln(p)) to a Z-score"""
    if t <= 0: return 0.0
    p = np.exp(-t)
    if p < 1e-15: return np.sqrt(2*t) # Asymptotic approximation for crazy Copula spikes
    return st.norm.isf(p)

def load_channel_data(trigger, base_dir, cms=13600.0):
    bkg, info, cdfs = {}, {}, {}
    for m in ALL_CHANNELS:
        fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{m}.json")
        try:
            with open(fitfile, "r") as j:
                d = json.load(j)
                fmin, fmax = float(d['fmin']), float(d['fmax'])
                v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin) & (ATLAS_BINS <= fmax)]
                c = (v_bins[:-1] + v_bins[1:]) / 2
                counts = FiveParam(cms, c, *d['parameters']) * np.diff(v_bins)
                if np.sum(counts) > 0:
                    bkg[m] = counts
                    info[m] = {'centers': c, 'bins': v_bins}
                    cdfs[m] = np.cumsum(counts) / np.sum(counts)
        except Exception:
            continue
    return bkg, info, cdfs

def main(args):
    base_dir = repo_root
    bkg, info, cdfs = load_channel_data(args.trigger, base_dir)
    
    if 'jj' not in bkg:
        print("Error: Missing M_jj fit, cannot build hubs."); sys.exit(1)

    # Load Copula
    copula_path = os.path.join(base_dir, "data", f"copula_{args.trigger}.npz")
    f = np.load(copula_path)
    matrix, col_names = f['copula'], list(f['columns'])
    n_mother_exp = np.sum(bkg['jj'])
    scales = {m: np.sum(b) / n_mother_exp for m, b in bkg.items()}
    overlap_map = TRIGGER_OVERLAPS.get(args.trigger.lower(), TRIGGER_OVERLAPS["default"])

    print(f"Generating Universe for {args.trigger.upper()}...")
    
    attempts = 0
    while True:
        attempts += 1
        toys_naive, toys_linear, toys_copula = {}, {}, {}
        z_naive, z_linear, z_copula = {}, {}, {}
        
        # 1. NAIVE TOYS
        for m, b in bkg.items():
            toys_naive[m] = np.random.poisson(b)
            t = fast_bumphunter_stat(toys_naive[m], b)
            z_naive[m] = t_to_z(t)

        # 2. LINEAR TOYS
        jj_pseudo = np.random.poisson(bkg['jj'])
        jj_res_raw = np.where(bkg['jj'] > 0, (jj_pseudo - bkg['jj']) / bkg['jj'], 0)
        for m, b in bkg.items():
            if m == 'jj':
                toys_linear[m] = jj_pseudo
            else:
                eff = overlap_map.get(m, 0.1)
                mapped_res = np.interp(info[m]['centers'], info['jj']['centers'], jj_res_raw)
                ov_counts = (b * eff) * (1 + mapped_res)
                ind_counts = np.random.poisson(b * (1 - eff))
                toys_linear[m] = np.maximum(0, np.round(ov_counts + ind_counts).astype(int))
            t = fast_bumphunter_stat(toys_linear[m], b)
            z_linear[m] = t_to_z(t)

        # 3. COPULA TOYS
        sampled = matrix[np.random.choice(len(matrix), size=np.random.poisson(n_mother_exp), replace=True)]
        
        for m, b in bkg.items():
            idx = col_names.index(f"M{m}")
            expected_yield = np.sum(b)
            target_n = np.random.poisson(expected_yield)
            
            if target_n == 0:
                toys_copula[m] = np.zeros(len(b), dtype=int)
                z_copula[m] = 0.0
                continue
                
            v_correlated = sampled[sampled[:, idx] >= 0, idx]
            k = len(v_correlated)
            
            if k >= target_n:
                U_final = np.random.choice(v_correlated, size=target_n, replace=False)
            else:
                independent_n = target_n - k
                U_independent = np.random.uniform(0, 1, size=independent_n)
                U_final = np.concatenate([v_correlated, U_independent])
                
            # Apply a safe uniform dither to break up duplicates
            U_final += np.random.uniform(-0.0002, 0.0002, size=target_n)
            
            # SAFE BOUNDARY REFLECTION (Prevents the tail wall spike)
            U_final = np.abs(U_final) # Bounces negative values slightly positive
            U_final = np.where(U_final >= 1.0, 1.99999 - U_final, U_final) # Bounces >1.0 values back into the tail
            
            # Map to physical histogram
            toys_copula[m] = np.bincount(np.searchsorted(cdfs[m], U_final), minlength=len(b))
            
            t = fast_bumphunter_stat(toys_copula[m], b)
            z_copula[m] = t_to_z(t)

        # Hunt Logic: Break if we find a Copula spike, OR if we aren't hunting
        max_copula_z = max(z_copula.values())
        if not args.hunt or max_copula_z >= 5.0:
            if args.hunt: print(f"Anomaly found after {attempts} attempts! (Max Copula Z = {max_copula_z:.2f})")
            break

    # --- PLOTTING ---
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    for i, m in enumerate(ALL_CHANNELS):
        ax = axes[i]
        if m not in bkg:
            ax.axis('off'); continue

        c, bins, B = info[m]['centers'], info[m]['bins'], bkg[m]
        
        ax.plot(c, B, color='gray', linestyle='--', lw=2, label='Analytic Bkg')
        
        # Plot the 3 methods using step plots so they don't completely overlap
        ax.step(c, toys_naive[m], where='mid', color='red', alpha=0.6, lw=1.5, 
                label=f'Naive (Z={z_naive[m]:.1f})')
        ax.step(c, toys_linear[m], where='mid', color='blue', alpha=0.6, lw=1.5, 
                label=f'Linear (Z={z_linear[m]:.1f})')
        ax.step(c, toys_copula[m], where='mid', color='green', alpha=0.8, lw=2.0, 
                label=f'Copula (Z={z_copula[m]:.1f})')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(bins[0], 6500)
        ax.set_ylim(0.5, np.max(toys_copula[m]) * 10)
        
        ax.set_title(f'Channel: $M_{{{m}}}$', fontsize=14, fontweight='bold')
        ax.set_xlabel('Mass [GeV]')
        ax.set_ylabel('Events')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f'Method Comparison & BumpHunter Z-Scores | Trigger: {args.trigger.upper()}', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_dir = os.path.join(base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_anomaly" if args.hunt else ""
    out_file = os.path.join(out_dir, f"method_comparison_{args.trigger}{suffix}.png")
    plt.savefig(out_file, dpi=200)
    print(f"Plot saved to: {out_file}")

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('--trigger', required=True)
    p.add_argument('--hunt', action='store_true', help="Keep generating until Copula breaks (Z > 5)")
    main(p.parse_args())
