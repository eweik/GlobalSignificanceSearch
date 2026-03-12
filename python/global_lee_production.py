#!/usr/bin/env python
import os
import sys
import json
import numpy as np
from argparse import ArgumentParser

# Add the parent directory to the path so it can find the 'src' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ATLAS_BINS, TRIGGER_OVERLAPS
from src.models import FiveParam, FiveParam_alt
from src.stats import fast_bumphunter_stat

def main(args):
    mass_types = ["jj", "bb", "jb", "je", "jm", "jg", "be", "bm", "bg"]
    bkg_expectations = {}
    syst_envelopes = {}
    channel_info = {}
    
    overlap_map = TRIGGER_OVERLAPS.get(args.trigger.lower(), TRIGGER_OVERLAPS["default"])
    
    # 1. Load Nominal and Alternative Fits
    for m in mass_types:
        fitfile_nom = f"/eos/atlas/atlascerngroupdisk/phys-exotics/jdm/lepdijet/AnomalyDetect23/stat/pyBH/figs/fitme_p5_{args.trigger}_{m}.json"
        fitfile_alt = f"/eos/atlas/atlascerngroupdisk/phys-exotics/jdm/lepdijet/AnomalyDetect23/stat/pyBH/figs/fitme_p5alt_{args.trigger}_{m}.json"
        
        try:
            with open(fitfile_nom, "r") as j_nom, open(fitfile_alt, "r") as j_alt:
                d_nom = json.load(j_nom)
                d_alt = json.load(j_alt)
                
                v_bins = ATLAS_BINS[(ATLAS_BINS >= float(d_nom['fmin'])) & (ATLAS_BINS <= float(d_nom['fmax']))]
                c = (v_bins[:-1] + v_bins[1:]) / 2
                widths = np.diff(v_bins)
                
                counts_nom = FiveParam(args.cms, c, *d_nom['parameters']) * widths
                counts_alt = FiveParam_alt(args.cms, c, *d_alt['parameters']) * widths
                
                if np.sum(counts_nom) > 0:
                    bkg_expectations[m] = counts_nom
                    syst_envelopes[m] = np.abs(counts_alt - counts_nom)
                    channel_info[m] = {'centers': c, 'bins': v_bins}
        except Exception as e:
            continue

    if not bkg_expectations:
        print("Error: No background fits found."); sys.exit(1)

    # 2. Setup Copula Matrices
    if args.method == "copula":
        f = np.load(f"copula_{args.trigger}.npz")
        matrix, col_names = f['copula'], list(f['columns'])
        cdfs = {m: np.cumsum(b) / np.sum(b) for m, b in bkg_expectations.items()}
        
        mother_key = 'jj' if 'jj' in bkg_expectations else list(bkg_expectations.keys())[0]
        n_mother_exp = np.sum(bkg_expectations[mother_key])
        channel_scales = {m: np.sum(b) / n_mother_exp for m, b in bkg_expectations.items()}

    stats = []
    print(f"Generating {args.toys} {args.method} toys for {args.trigger}...")
    
    # 3. Main Toy Loop
    for i in range(args.toys):
        max_t = 0.0
        
        if args.method == "naive":
            for m, b in bkg_expectations.items():
                env = syst_envelopes[m]
                b_fluct = np.maximum(0, b + (np.random.normal(0, 1, size=len(b)) * env))
                toy = np.random.poisson(b_fluct)
                max_t = max(max_t, fast_bumphunter_stat(toy, b))
        
        elif args.method == "linear":
            if 'jj' not in bkg_expectations:
                print("Error: Hub-and-Spoke requires jj channel."); sys.exit(1)
            
            jj_b = bkg_expectations['jj']
            jj_fluct = np.maximum(0, jj_b + (np.random.normal(0, 1, size=len(jj_b)) * syst_envelopes['jj']))
            jj_pseudo = np.random.poisson(jj_fluct)
            
            jj_res_raw = np.where(jj_b > 0, (jj_pseudo - jj_b) / jj_b, 0)
            jj_centers = channel_info['jj']['centers']
            
            for m, b in bkg_expectations.items():
                if m == 'jj':
                    toy = jj_pseudo
                else:
                    overlap_frac = overlap_map.get(m, 0.1)
                    this_centers = channel_info[m]['centers']
                    
                    mapped_res = np.interp(this_centers, jj_centers, jj_res_raw)
                    ov_counts = (b * overlap_frac) * (1 + mapped_res)
                    
                    ind_b = b * (1 - overlap_frac)
                    ind_fluct = np.maximum(0, ind_b + (np.random.normal(0, 1, size=len(ind_b)) * syst_envelopes[m] * (1-overlap_frac)))
                    ind_counts = np.random.poisson(ind_fluct)
                    
                    toy = np.maximum(0, ov_counts + ind_counts)
                
                max_t = max(max_t, fast_bumphunter_stat(toy, b))

        elif args.method == "copula":
            n_mother = np.random.poisson(n_mother_exp)
            sampled = matrix[np.random.choice(len(matrix), size=n_mother, replace=True)]
            
            for m, b in bkg_expectations.items():
                idx = col_names.index(f"M{m}")
                v = sampled[sampled[:, idx] >= 0, idx]
                if len(v) == 0: continue
                
                target_n = int(len(v) * channel_scales[m])
                if target_n < len(v):
                    v = np.random.choice(v, size=target_n, replace=False)
                
                U = np.clip(v - np.random.uniform(0, 1e-6, size=len(v)), 0, 1)
                toy = np.bincount(np.clip(np.searchsorted(cdfs[m], U), 0, len(b)-1), minlength=len(b))
                
                syst_shift = np.random.normal(0, 1, size=len(b)) * syst_envelopes[m]
                toy = np.maximum(0, np.round(toy + syst_shift).astype(int))
                
                max_t = max(max_t, fast_bumphunter_stat(toy, b))
        
        stats.append(max_t)

    np.save(f"global_stat_{args.trigger}_{args.method}.npy", stats)

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--trigger', required=True)
    p.add_argument('--toys', type=int, default=1000)
    p.add_argument('--method', choices=["naive", "copula", "linear"], required=True)
    p.add_argument('--cms', type=float, default=13600.)
    main(p.parse_args())
