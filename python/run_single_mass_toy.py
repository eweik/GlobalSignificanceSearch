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

from src.config import ATLAS_BINS
from src.models import FiveParam
from src.stats import fast_bumphunter_stat

def main(args):
    os.makedirs("results", exist_ok=True)
    base_dir = os.getcwd() if os.path.exists("data") and os.path.exists("fits") else repo_root

    # --- 1. LOAD FIT & DEFINE BACKGROUND ---
    fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{args.trigger}_{args.channel}.json")
    if not os.path.exists(fitfile):
        print(f"Error: Fit file not found -> {fitfile}")
        sys.exit(1)

    with open(fitfile, "r") as j_nom:
        d_nom = json.load(j_nom)
        
    fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
    v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
    centers = (v_bins[:-1] + v_bins[1:]) / 2
    
    # Evaluate smooth 5-parameter background function
    bkg_expected = FiveParam(args.cms, centers, *d_nom['parameters'])
    bkg_expected = np.maximum(bkg_expected, 0)
    
    if np.sum(bkg_expected) <= 0:
        print("Error: Background expectation evaluates to 0.")
        sys.exit(1)

    # --- 2. PREPARE RAW DATA (If using Bootstrap) ---
    raw_valid_masses = None
    if args.method == "poisson_bootstrap":
        mass_path = os.path.join(base_dir, "data", f"masses_{args.trigger}.npz")
        if not os.path.exists(mass_path):
            print(f"Error: Raw mass data required for bootstrap not found -> {mass_path}")
            sys.exit(1)
            
        f_mass = np.load(mass_path)
        mass_matrix = f_mass['masses']
        col_names = list(f_mass['columns'])
        
        try:
            idx = col_names.index(f"M{args.channel}")
        except ValueError:
            print(f"Error: Channel M{args.channel} not found in mass matrix.")
            sys.exit(1)
            
        masses = mass_matrix[:, idx]
        raw_valid_masses = masses[masses > 0] * args.cms
        N_raw_events = len(raw_valid_masses)
        print(f"Loaded {N_raw_events} valid raw events for bootstrap sampling.")

    # --- 3. TOY GENERATION & BUMPHUNTER SCANS ---
    stats = []
    attempts = 0
    max_attempts = args.toys * 10 
    
    print(f"Generating {args.toys} {args.method} toys for {args.trigger}_{args.channel}...")
    start_time = time.time()
    
    while len(stats) < args.toys and attempts < max_attempts:
        attempts += 1
        
        # Progress bar
        completed = len(stats)
        if completed > 0 and completed % max(1, (args.toys // 20)) == 0:
            progress = int((completed / args.toys) * 100)
            sys.stdout.write(f"\rProgress: [{('=' * (progress//5)).ljust(20)}] {progress}% ")
            sys.stdout.flush()

        # Generate Toy
        if args.method == "naive":
            # Independent Poisson fluctuations around the smooth fit
            toy_counts = np.random.poisson(bkg_expected)
            
        elif args.method == "poisson_bootstrap":
            # Resample actual data events (Poisson-weighted bootstrap)
            N_draw = np.random.poisson(N_raw_events)
            sampled_masses = np.random.choice(raw_valid_masses, size=N_draw, replace=True)
            toy_counts, _ = np.histogram(sampled_masses, bins=v_bins)

        if np.sum(toy_counts) < 10: 
            continue

        # Evaluate BumpHunter statistic against the SMOOTH expected background
        t_val = fast_bumphunter_stat(toy_counts, bkg_expected)
        stats.append(t_val)

    sys.stdout.write(f"\rProgress: [{'=' * 20}] 100% \n")
    sys.stdout.flush()

    # --- 4. SAVE RESULTS ---
    out_file = os.path.join("results", f"single_stat_{args.trigger}_{args.channel}_{args.method}.npy")
    np.save(out_file, stats)
    
    elapsed_time = time.time() - start_time
    print("-" * 50)
    print(f"Successfully saved {len(stats)} toys to {out_file}")
    print(f"Time Elapsed: {elapsed_time:.2f}s")
    print("-" * 50)

if __name__ == '__main__':
    p = ArgumentParser(description="Run single-histogram pseudo-experiments.")
    p.add_argument('--trigger', type=str, required=True, help="Trigger name (e.g., t2)")
    p.add_argument('--channel', type=str, required=True, help="Channel name (e.g., jj, jb)")
    p.add_argument('--toys', type=int, default=10000, help="Number of pseudo-experiments to run")
    p.add_argument('--method', choices=["naive", "poisson_bootstrap"], required=True, help="Toy generation method")
    p.add_argument('--cms', type=float, default=13000., help="Center of mass energy")
    main(p.parse_args())
