#!/usr/bin/env python3
"""
run_toys_combined.py

Generate pseudo-experiments for the WHOLE dataset (all triggers × channels)
using the combined copula matrix from extract_copula_combined.py.

The combined copula has one row per unique physical event and 63 columns
(7 triggers × 9 channels). Resampling ROWS preserves cross-trigger
correlations (shared events stay together). The per-toy global test
statistic is  max_t = max over all 63 columns of BumpHunter(toy, bkg).

Usage:
    python run_toys_combined.py --toys 1000 --jobid job1
"""
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
    fits_dir = os.path.join(base_dir, "fits")

    # ------------------------------------------------------------------
    # 1. LOAD COMBINED COPULA
    # ------------------------------------------------------------------
    copula_path = os.path.join(base_dir, "data", args.copula)
    f = np.load(copula_path)
    matrix    = f['copula']          # (N_events, 63)
    col_names = list(f['columns'])   # ["t1_Mjj", ..., "t7_Mbg"]
    if 'u_bounds' not in f.files:
        print("Error: copula file has no 'u_bounds'. "
              "Re-run extract_copula_combined.py."); sys.exit(1)
    ub_arr = f['u_bounds']           # (63, 2)

    N_events = len(matrix)
    print(f"Loaded combined copula: {matrix.shape[0]} events × {matrix.shape[1]} columns")

    # ------------------------------------------------------------------
    # 2. LOAD FITS & BUILD BACKGROUND FOR EVERY COLUMN
    # ------------------------------------------------------------------
    bkg = {}       # col_name -> expected counts array
    cdfs = {}      # col_name -> cumulative CDF for inverse mapping
    u_bounds = {}  # col_name -> (u_min, u_max)

    skipped = []
    for ci, cn in enumerate(col_names):
        # parse "t3_Mjj" -> trig="t3", channel="jj"
        trig, mass_var = cn.split("_", 1)
        ch = mass_var[1:]  # "Mjj" -> "jj"

        fitfile = os.path.join(fits_dir, f"fitme_p5_{trig}_{ch}.json")
        if not os.path.isfile(fitfile):
            skipped.append(cn)
            continue

        with open(fitfile) as fh:
            d = json.load(fh)

        fmin, fmax = float(d['fmin']), float(d['fmax'])
        v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin) & (ATLAS_BINS <= fmax)]
        if len(v_bins) < 2:
            skipped.append(cn); continue

        centers = (v_bins[:-1] + v_bins[1:]) / 2
        bin_widths = np.diff(v_bins)
        counts = FiveParam(args.cms, centers, *d['parameters']) * bin_widths
        counts = np.maximum(counts, 0)

        if np.sum(counts) <= 0:
            skipped.append(cn); continue

        bkg[cn]      = counts
        cdfs[cn]     = np.cumsum(counts) / np.sum(counts)
        u_bounds[cn] = tuple(ub_arr[ci])

    active_cols = [(ci, cn) for ci, cn in enumerate(col_names) if cn in bkg]

    print(f"Active columns: {len(active_cols)} / {len(col_names)}")
    if skipped:
        print(f"Skipped (no fit or empty): {skipped}")
    if not active_cols:
        print("Error: no active columns."); sys.exit(1)

    # Summarize per trigger
    trigs_seen = sorted(set(cn.split("_")[0] for _, cn in active_cols))
    for trig in trigs_seen:
        chs = [cn.split("_")[1] for _, cn in active_cols if cn.startswith(trig + "_")]
        print(f"  {trig}: {len(chs)} channels -> {chs}")

    # ------------------------------------------------------------------
    # 3. TOY LOOP — empirical copula, max over all 63 columns
    # ------------------------------------------------------------------
    stats = []
    attempts = 0
    max_attempts = args.toys * 50

    print(f"\nGenerating {args.toys} combined-copula toys | Bkg: func")
    start_time = time.time()

    while len(stats) < args.toys and attempts < max_attempts:
        attempts += 1

        completed = len(stats)
        if not args.batch and completed > 0 and completed % max(1, (args.toys // 20)) == 0:
            progress = int((completed / args.toys) * 100)
            sys.stdout.write(f"\rProgress: [{('=' * (progress // 5)).ljust(20)}] "
                             f"{progress}% (Attempts: {attempts}) ")
            sys.stdout.flush()

        # Resample rows — this is where cross-trigger correlations survive:
        # a shared event's ranks in t1_Mjj AND t2_Mjj move together.
        N_draw = np.random.poisson(N_events)
        sampled_rows = matrix[np.random.choice(N_events, size=N_draw, replace=True)]

        max_t = 0.0
        channels_searched = 0

        for ci, cn in active_cols:
            b = bkg[cn]
            u_min, u_max = u_bounds[cn]

            # Extract valid ranks for this column from the resampled rows
            u_raw = sampled_rows[sampled_rows[:, ci] >= 0, ci]

            # Phase-space truncation to fit window
            mask_in_window = (u_raw >= u_min) & (u_raw <= u_max)
            u_in_window = u_raw[mask_in_window]

            if len(u_in_window) == 0:
                toy = np.zeros(len(b), dtype=int)
            else:
                u_jittered = u_in_window + np.random.uniform(-0.0002, 0.0002,
                                                              size=len(u_in_window))
                u_trunc = (u_jittered - u_min) / max(u_max - u_min, 1e-10)
                u_trunc = np.abs(u_trunc)
                u_trunc = np.where(u_trunc >= 1.0, 1.99999 - u_trunc, u_trunc)

                toy = np.bincount(np.searchsorted(cdfs[cn], u_trunc),
                                  minlength=len(b))

            if np.sum(toy) < 50:
                continue

            max_t = max(max_t, fast_bumphunter_stat(toy, b))
            channels_searched += 1

        if channels_searched > 0:
            stats.append(max_t)

    sys.stdout.write(f"\rProgress: [{'=' * 20}] 100% \n")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # 4. SAVE
    # ------------------------------------------------------------------
    out_file = os.path.join("results",
                            f"global_stat_combined_copula_{args.jobid}_BKGfunc.npy")
    np.save(out_file, stats)

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print("-" * 50)
    print(f"Saved {len(stats)} toys -> {out_file}")
    print(f"Columns searched per toy: {len(active_cols)}")
    print(f"Acceptance rate: {(len(stats) / max(attempts, 1)) * 100:.2f}%")
    print(f"Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("-" * 50)


if __name__ == '__main__':
    p = ArgumentParser(description="Combined-copula pseudo-experiments across all triggers.")
    p.add_argument('--toys',   type=int,   default=1000)
    p.add_argument('--cms',    type=float, default=13000.)
    p.add_argument('--jobid',  type=str,   default="local")
    p.add_argument('--copula', type=str,   default="copula_combined.npz",
                   help="Filename (inside data/) of the combined copula .npz")
    p.add_argument('-b', '--batch', action='store_true')
    main(p.parse_args())
