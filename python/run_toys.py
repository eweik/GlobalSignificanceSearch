#!/usr/bin/env python3
import os
import sys
import json
import time
import numpy as np
from argparse import ArgumentParser

current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from src.config import ATLAS_BINS, TRIGGER_OVERLAPS
from src.models import FiveParam, FiveParam_alt
from src.stats import fast_bumphunter_stat

try:
    from src.fitting import setup_root_env, create_tf1_template, do_fit_and_get_bkg
    FITTING_AVAILABLE = True
except ImportError:
    FITTING_AVAILABLE = False

def main(args):
    os.makedirs("results", exist_ok=True)
    base_dir = os.getcwd() if os.path.exists("data") and os.path.exists("fits") else repo_root

    if args.fit:
        if not FITTING_AVAILABLE:
            print("Error: --fit requested but fitting tools not found."); sys.exit(1)
        setup_root_env(batch=args.batch, fit_enabled=True)

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
                
                counts_nom = FiveParam(args.cms, c, *d_nom['parameters']) 
                counts_alt = FiveParam_alt(args.cms, c, *d_alt['parameters']) 
                
                if np.sum(counts_nom) > 0:
                    bkg_expectations[m] = counts_nom
                    syst_envelopes[m] = np.abs(counts_alt - counts_nom)
                    channel_info[m] = {'centers': c, 'bins': v_bins}
                    
                    if args.fit:
                        name = f"back_{args.trigger}_{m}"
                        tf1_templates[m] = create_tf1_template(name, args.cms, fmin_val, fmax_val, d_nom['parameters'])
        except Exception: continue

    if not bkg_expectations:
        print(f"Error: No background fits found in {os.path.join(base_dir, 'fits')}"); sys.exit(1)

    # --- DATA LOADING & PREPARATION ---
    if args.method == "copula":
        copula_path = os.path.join(base_dir, "data", f"copula_{args.trigger}.npz")
        f = np.load(copula_path)
        matrix, col_names = f['copula'], list(f['columns'])
        cdfs = {m: np.cumsum(b) / np.sum(b) for m, b in bkg_expectations.items()}
        mother_key = 'jj' if 'jj' in bkg_expectations else list(bkg_expectations.keys())[0]
        # Use the expected inclusive yield as N_events to sample from
        N_events = np.sum(bkg_expectations[mother_key])

    elif args.method in ["poisson_event", "exclusive_categories"]:
        mass_path = os.path.join(base_dir, "data", f"masses_{args.trigger}.npz")
        f = np.load(mass_path)
        mass_matrix, col_names = f['masses'], list(f['columns'])
        N_events = len(mass_matrix)

        if args.method == "exclusive_categories":
            # Map events to orthogonal 9-bit categories
            valid_mask = mass_matrix > 0
            powers = 2 ** np.arange(len(col_names))
            event_patterns = valid_mask.dot(powers)
            unique_patterns = np.unique(event_patterns)
            pattern_indices = {p: np.where(event_patterns == p)[0] for p in unique_patterns}

    stats = []
    fit_failures, attempts = 0, 0
    max_attempts = args.toys * 50 
    
    mode_str = "FIXED-DOF REFIT" if args.fit else "NO FIT (FROZEN BKG)"
    print(f"Generating {args.toys} {args.method} toys for {args.trigger} | Mode: {mode_str}")
    start_time = time.time()

    
    while len(stats) < args.toys and attempts < max_attempts:
        attempts += 1
        max_t, channels_searched = 0.0, 0
        
        completed = len(stats)
        if not args.batch and completed > 0 and completed % max(1, (args.toys // 20)) == 0:
            progress = int((completed / args.toys) * 100)
            sys.stdout.write(f"\rProgress: [{('=' * (progress//5)).ljust(20)}] {progress}% (Attempts: {attempts}) ")
            sys.stdout.flush()

        # ---------------------------------------------------------
        # TOY GENERATION LOGIC
        # ---------------------------------------------------------
        if args.method == "naive":
            for m, b in bkg_expectations.items():
                toy = np.random.poisson(b)
                if np.sum(toy) < 50: continue
                
                if args.fit:
                    active_bkg, fit_ok = do_fit_and_get_bkg(toy, m, b, channel_info, tf1_templates[m], args, syst_envelopes[m])
                    if not fit_ok: fit_failures += 1; continue 
                else: 
                    active_bkg = b
                
                max_t = max(max_t, fast_bumphunter_stat(toy, active_bkg))
                channels_searched += 1
        
        elif args.method == "linear":
            jj_b = bkg_expectations['jj']
            jj_pseudo = np.random.poisson(jj_b)
            jj_centers = channel_info['jj']['centers']

            for m, b in bkg_expectations.items():
                if m == 'jj':
                    toy = jj_pseudo
                else:
                    ov_frac = overlap_map.get(m, 0.1)
                    m_centers = channel_info[m]['centers']

                    # 1. Exact Bin Alignment
                    jj_b_aligned = np.zeros(len(b))
                    jj_pseudo_int = np.zeros(len(b), dtype=int)

                    for i, mc in enumerate(m_centers):
                        dist = np.abs(jj_centers - mc)
                        min_idx = np.argmin(dist)
                        if dist[min_idx] < 1.0:
                            jj_b_aligned[i] = jj_b[min_idx]
                            jj_pseudo_int[i] = jj_pseudo[min_idx]

                    # 2. Bivariate Poisson math
                    lambda_shared = np.minimum(b * ov_frac, jj_b_aligned)
                    p_transfer = lambda_shared / np.maximum(jj_b_aligned, 1e-15)

                    # 3. Draw correlated events
                    ov_counts = np.random.binomial(jj_pseudo_int, p_transfer)

                    # 4. Draw independent events to PERFECTLY restore target sub-channel mean
                    ind_b = np.maximum(0, b - lambda_shared)
                    ind_counts = np.random.poisson(ind_b)

                    toy = ov_counts + ind_counts

                if np.sum(toy) < 50: continue
                if args.fit:
                    active_bkg, fit_ok = do_fit_and_get_bkg(toy, m, b, channel_info, tf1_templates[m], args, syst_envelopes[m])
                    if not fit_ok: fit_failures += 1; continue
                else: 
                    active_bkg = b
                max_t = max(max_t, fast_bumphunter_stat(toy, active_bkg))
                channels_searched += 1

        elif args.method == "copula":
            # 1. Sample N times globally for this toy
            N_draw = np.random.poisson(N_events)
            sampled_rows = matrix[np.random.choice(len(matrix), size=N_draw, replace=True)]
            
            for m, b in bkg_expectations.items():
                idx = col_names.index(f"M{m}")
                
                # 2. Extract valid values for this specific sub-channel
                v_correlated = sampled_rows[sampled_rows[:, idx] >= 0, idx]
                
                if len(v_correlated) == 0:
                    toy = np.zeros(len(b), dtype=int)
                else:
                    # Smear slightly to prevent binning artifacts inherited from the copula extraction
                    U_final = v_correlated + np.random.uniform(-0.0002, 0.0002, size=len(v_correlated))
                    
                    # Reflect boundaries to strictly bound within [0, 1)
                    U_final = np.where(U_final < 0.0, np.abs(U_final), U_final)
                    U_final = np.where(U_final >= 1.0, 1.99999 - U_final, U_final)
                    
                    # 3. Invert back into un-binned target distribution via CDF
                    toy = np.bincount(np.searchsorted(cdfs[m], U_final), minlength=len(b))

                if np.sum(toy) < 50: continue

                if args.fit:
                    active_bkg, fit_ok = do_fit_and_get_bkg(toy, m, b, channel_info, tf1_templates[m], args, syst_envelopes[m])
                    if not fit_ok: fit_failures += 1; continue
                else: 
                    active_bkg = b

                max_t = max(max_t, fast_bumphunter_stat(toy, active_bkg))
                channels_searched += 1

        elif args.method in ["poisson_event", "exclusive_categories"]:
            if args.method == "poisson_event":
                N_draw = np.random.poisson(N_events)
                sampled_rows = mass_matrix[np.random.choice(N_events, size=N_draw, replace=True)]
            else:
                sampled_rows_list = []
                for p, indices in pattern_indices.items():
                    n_obs = len(indices)
                    n_toy = np.random.poisson(n_obs)
                    if n_toy > 0:
                        sampled_rows_list.append(mass_matrix[np.random.choice(indices, size=n_toy, replace=True)])
                
                if len(sampled_rows_list) > 0:
                    sampled_rows = np.concatenate(sampled_rows_list, axis=0)
                else:
                    sampled_rows = np.empty((0, len(col_names)))

            for m, b in bkg_expectations.items():
                idx = col_names.index(f"M{m}")
                masses = sampled_rows[:, idx]
                valid_masses = masses[masses > 0]
                physical_masses = valid_masses * args.cms

                toy, _ = np.histogram(physical_masses, bins=channel_info[m]['bins'])
                
                if np.sum(toy) < 50: continue

                if args.fit:
                    active_bkg, fit_ok = do_fit_and_get_bkg(toy, m, b, channel_info, tf1_templates[m], args, syst_envelopes[m])
                    if not fit_ok: fit_failures += 1; continue
                else: 
                    active_bkg = b 

                max_t = max(max_t, fast_bumphunter_stat(toy, active_bkg))
                channels_searched += 1

        # ---------------------------------------------------------
        
        if channels_searched > 0: stats.append(max_t)

    sys.stdout.write(f"\rProgress: [{'=' * 20}] 100% \n")
    sys.stdout.flush()

    suffix = "FIT" if args.fit else "NOFIT"
    out_file = os.path.join("results", f"global_stat_{args.trigger}_{args.method}_{args.jobid}_{suffix}.npy")
    np.save(out_file, stats)
    
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print("-" * 50)
    print(f"Successfully saved {len(stats)} toys to {out_file}")
    if args.fit: print(f"Total individual channel fits failed/skipped: {fit_failures}")
    print(f"Overall Acceptance Rate: {(len(stats) / attempts) * 100:.2f}%")
    print(f"Time Elapsed: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("-" * 50)

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--trigger', required=True)
    p.add_argument('--toys', type=int, default=1000)
    p.add_argument('--method', choices=["naive", "copula", "linear", "poisson_event", "exclusive_categories"], required=True)
    p.add_argument('--cms', type=float, default=13000.)
    p.add_argument('-b', '--batch', action='store_true')
    p.add_argument('--fit', action='store_true')
    p.add_argument('--chimax', type=float, default=2.0)
    p.add_argument('--jobid', type=str, default="local")
    main(p.parse_args())
