#!/usr/bin/env python3
import os
import sys
import json
import time
import numpy as np
from argparse import ArgumentParser

# Setup paths so it can find the src module whether on lxplus or Condor
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path:
    sys.path.append(repo_root)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from src.config import ATLAS_BINS, TRIGGER_OVERLAPS
from src.models import FiveParam, FiveParam_alt
from src.stats import fast_bumphunter_stat
from src.fitting import setup_root_env, create_tf1_template, do_fit_and_get_bkg

def main(args):
    # Condor requires the output directory to exist in the worker node's local slot
    os.makedirs("results", exist_ok=True)
    
    # Path logic: Worker nodes are flat, local runs have structure
    if os.path.exists("data") and os.path.exists("fits"):
        base_dir = os.getcwd()
    else:
        base_dir = repo_root

    if args.fit or args.batch:
        setup_root_env(batch=args.batch, fit_enabled=args.fit)

    mass_types = ["jj", "bb", "jb", "je", "jm", "jg", "be", "bm", "bg"]
    bkg_expectations, syst_envelopes, channel_info, tf1_templates = {}, {}, {}, {}
    overlap_map = TRIGGER_OVERLAPS.get(args.trigger.lower(), TRIGGER_OVERLAPS["default"])
    
    for m in mass_types:
        fitfile_nom = os.path.join(base_dir, "fits", f"fitme_p5_{args.trigger}_{m}.json")
        fitfile_alt = os.path.join(base_dir, "fits", f"fitme_p5alt_{args.trigger}_{m}.json")
        
        try:
            with open(fitfile_nom, "r") as j_nom, open(fitfile_alt, "r") as j_alt:
                d_nom, d_alt = json.load(j_nom), json.load(j_alt)
                
                fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
                v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
                c = (v_bins[:-1] + v_bins[1:]) / 2
                widths = np.diff(v_bins)
                
                counts_nom = FiveParam(args.cms, c, *d_nom['parameters']) * widths
                counts_alt = FiveParam_alt(args.cms, c, *d_alt['parameters']) * widths
                
                if np.sum(counts_nom) > 0:
                    bkg_expectations[m] = counts_nom
                    syst_envelopes[m] = np.abs(counts_alt - counts_nom)
                    channel_info[m] = {'centers': c, 'bins': v_bins}
                    
                    if args.fit:
                        name = f"back_{args.trigger}_{m}"
                        tf1_templates[m] = create_tf1_template(name, args.cms, fmin_val, fmax_val, d_nom['parameters'])
        except Exception:
            continue

    if not bkg_expectations:
        print(f"Error: No background fits found in {os.path.join(base_dir, 'fits')}"); sys.exit(1)

    if args.method == "copula":
        copula_path = os.path.join(base_dir, "data", f"copula_{args.trigger}.npz")
        f = np.load(copula_path)
        matrix, col_names = f['copula'], list(f['columns'])
        cdfs = {m: np.cumsum(b) / np.sum(b) for m, b in bkg_expectations.items()}
        
        mother_key = 'jj' if 'jj' in bkg_expectations else list(bkg_expectations.keys())[0]
        n_mother_exp = np.sum(bkg_expectations[mother_key])
        channel_scales = {m: np.sum(b) / n_mother_exp for m, b in bkg_expectations.items()}

    stats = []
    fit_failures, attempts = 0, 0
    max_attempts = args.toys * 50  # Give it plenty of room to fail and try again
    
    print(f"Generating {args.toys} {args.method} toys for {args.trigger}...")
    start_time = time.time()
    
    while len(stats) < args.toys and attempts < max_attempts:
        attempts += 1
        max_t = 0.0
        toy_successful = True
        
        completed = len(stats)
        if completed > 0 and completed % max(1, (args.toys // 20)) == 0:
            progress = int((completed / args.toys) * 100)
            sys.stdout.write(f"\rProgress: [{('=' * (progress//5)).ljust(20)}] {progress}% (Attempts: {attempts}) ")
            sys.stdout.flush()

        if args.method == "naive":
            for m, b in bkg_expectations.items():
                env = syst_envelopes[m]
                toy = np.random.poisson(np.maximum(0, b + (np.random.normal(0, 1, size=len(b)) * env)))
                
                active_bkg, fit_ok = do_fit_and_get_bkg(toy, m, b, channel_info, tf1_templates, args)
                if not fit_ok: 
                    toy_successful = False
                    break
                
                max_t = max(max_t, fast_bumphunter_stat(toy, active_bkg))
        
        elif args.method == "linear":
            jj_b = bkg_expectations['jj']
            jj_pseudo = np.random.poisson(np.maximum(0, jj_b + (np.random.normal(0, 1, size=len(jj_b)) * syst_envelopes['jj'])))
            jj_res_raw = np.where(jj_b > 0, (jj_pseudo - jj_b) / jj_b, 0)
            
            for m, b in bkg_expectations.items():
                if m == 'jj':
                    toy = jj_pseudo
                else:
                    ov_frac = overlap_map.get(m, 0.1)
                    mapped_res = np.interp(channel_info[m]['centers'], channel_info['jj']['centers'], jj_res_raw)
                    ov_counts = (b * ov_frac) * (1 + mapped_res)
                    
                    ind_b = b * (1 - ov_frac)
                    ind_counts = np.random.poisson(np.maximum(0, ind_b + (np.random.normal(0, 1, size=len(ind_b)) * syst_envelopes[m] * (1-ov_frac))))
                    toy = np.maximum(0, ov_counts + ind_counts)
                
                active_bkg, fit_ok = do_fit_and_get_bkg(toy, m, b, channel_info, tf1_templates, args)
                if not fit_ok: 
                    toy_successful = False
                    break
                
                max_t = max(max_t, fast_bumphunter_stat(toy, active_bkg))

        elif args.method == "copula":
            sampled = matrix[np.random.choice(len(matrix), size=np.random.poisson(n_mother_exp), replace=True)]
            
            for m, b in bkg_expectations.items():
                idx = col_names.index(f"M{m}")
                v = sampled[sampled[:, idx] >= 0, idx]
                if len(v) == 0: continue
                
                target_n = int(len(v) * channel_scales[m])
                if target_n < len(v): v = np.random.choice(v, size=target_n, replace=False)
                
                U = np.clip(v - np.random.uniform(0, 1e-6, size=len(v)), 0, 1)
                toy_base = np.bincount(np.clip(np.searchsorted(cdfs[m], U), 0, len(b)-1), minlength=len(b))
                syst_shift = np.random.normal(0, 1, size=len(b)) * syst_envelopes[m]
                toy = np.maximum(0, np.round(toy_base + syst_shift).astype(int))
                
                active_bkg, fit_ok = do_fit_and_get_bkg(toy, m, b, channel_info, tf1_templates, args)
                if not fit_ok: 
                    toy_successful = False
                    break
                
                max_t = max(max_t, fast_bumphunter_stat(toy, active_bkg))
        
        if toy_successful:
            stats.append(max_t)
        else:
            fit_failures += 1

    sys.stdout.write(f"\rProgress: [{'=' * 20}] 100% \n")
    sys.stdout.flush()

    out_file = os.path.join("results", f"global_stat_{args.trigger}_{args.method}_{args.jobid}.npy")
    np.save(out_file, stats)
    
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print("-" * 50)
    print(f"Successfully saved {len(stats)} toys to {out_file}")
    if args.fit: 
        print(f"Total toys thrown out completely (after 5 fit retries): {fit_failures}")
        print(f"Overall Acceptance Rate: {(len(stats) / attempts) * 100:.2f}%")
    print(f"Time Elapsed: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("-" * 50)

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--trigger', required=True)
    p.add_argument('--toys', type=int, default=1000)
    p.add_argument('--method', choices=["naive", "copula", "linear"], required=True)
    p.add_argument('--cms', type=float, default=13600.)
    p.add_argument('-b', '--batch', action='store_true')
    p.add_argument('--fit', action='store_true')
    p.add_argument('--chimax', type=float, default=2.0)
    p.add_argument('--jobid', type=str, default="local")
    main(p.parse_args())
