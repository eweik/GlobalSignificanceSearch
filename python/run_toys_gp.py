#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import uproot
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# Import your existing configurations and BumpHunter
from src.config import ATLAS_BINS, TRIGGER_OVERLAPS
from src.stats import fast_bumphunter_stat
from src.gp_validation import plot_gp_validation

def fit_gp_density(centers, counts, widths, min_len_scale=300.0):
    """Fits GP to density (counts/width) in log space."""
    density = counts / widths
    errors = np.sqrt(np.maximum(counts, 0)) / widths
    
    mask = counts > 0
    if np.sum(mask) < 10:
        return density, np.zeros_like(density), False

    X = centers[mask].reshape(-1, 1)
    y = density[mask]
    y_err = errors[mask]

    y_log = np.log(y)
    y_err_log = y_err / y

    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=max(min_len_scale * 2, 1000.0), 
        length_scale_bounds=(min_len_scale, 1e4)
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_log**2, n_restarts_optimizer=3, normalize_y=True)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X, y_log)
        
        X_full = centers.reshape(-1, 1)
        y_pred_log, y_pred_log_std = gp.predict(X_full, return_std=True)
        
        # Transform back to linear density
        pred_density = np.exp(y_pred_log)
        pred_density_err = pred_density * y_pred_log_std
        return pred_density, pred_density_err, True
    except Exception:
        return density, np.zeros_like(density), False

def main(args):
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    mass_types = ["jj", "bb", "jb", "je", "jm", "jg", "be", "bm", "bg"]
    overlap_map = TRIGGER_OVERLAPS.get(args.trigger.lower(), TRIGGER_OVERLAPS["default"])
    
    # Setup data path based on your prompt
    trigger_num = args.trigger.replace('t', '')
    root_file_path = f"/eos/atlas/atlascerngroupdisk/phys-exotics/jdm/lepdijet/AnomalyDetect23/ana/root/data1percent_t{trigger_num}_HAE_RUN23_nominal_10PB.root"
    
    bkg_expectations = {}
    channel_info = {}
    
    print(f"Loading data from: {root_file_path}")
    try:
        root_file = uproot.open(root_file_path)
    except Exception as e:
        print(f"Failed to open ROOT file: {e}")
        sys.exit(1)

    # 1. Establish Null Hypothesis from Real Data
    for m in mass_types:
        # NOTE: Adjust the key below ("h_m_{m}") to match the exact histogram names in your ROOT file
        hist_name = f"h_m_{m}" 
        if hist_name not in root_file:
            continue
            
        hist = root_file[hist_name]
        counts, edges = hist.to_numpy()
        
        # Map to ATLAS Bins
        v_bins = ATLAS_BINS[(ATLAS_BINS >= edges[0]) & (ATLAS_BINS <= edges[-1])]
        c = (v_bins[:-1] + v_bins[1:]) / 2
        widths = np.diff(v_bins)
        
        # Rebin counts if necessary (assuming uproot yields raw bins, you may need a rebin function here if they don't match ATLAS_BINS natively)
        # For this script, we assume the ROOT file is already binned to ATLAS_BINS.
        
        min_len = c[0] * 0.05 * 3.0 # Dynamic length scale constraint
        gp_density, gp_err, ok = fit_gp_density(c, counts, widths, min_len_scale=min_len)
        
        if ok:
            bkg_expectations[m] = gp_density * widths # Store as Counts
            channel_info[m] = {'centers': c, 'widths': widths, 'min_len': min_len}
            
            # Validate the fit!
            plot_gp_validation(c, widths, counts, gp_density, gp_err, m, args.trigger)

    if not bkg_expectations:
        print("Error: No successful baseline GP fits."); sys.exit(1)

    # 2 & 3 & 4. Toy Generation, GP Refitting, and BumpHunter
    stats = []
    attempts = 0
    start_time = time.time()
    
    print(f"Generating {args.toys} {args.method} toys for {args.trigger} using GP Background...")
    
    while len(stats) < args.toys:
        attempts += 1
        max_t = 0.0
        toy_successful = True
        
        if len(stats) > 0 and len(stats) % (args.toys // 20) == 0:
            sys.stdout.write(f"\rProgress: {int(len(stats)/args.toys*100)}% ")
            sys.stdout.flush()

        # --- 1. GENERATE TOYS BASED ON METHOD ---
        toy_dict = {} # Store generated toys before fitting

        if args.method == "naive":
            for m, b in bkg_expectations.items():
                toy_dict[m] = np.random.poisson(b)

        elif args.method == "linear":
            jj_b = bkg_expectations.get('jj', list(bkg_expectations.values())[0])
            jj_pseudo = np.random.poisson(jj_b)
            jj_res_raw = np.where(jj_b > 0, (jj_pseudo - jj_b) / jj_b, 0)

            for m, b in bkg_expectations.items():
                if m == 'jj':
                    toy_dict[m] = jj_pseudo
                else:
                    ov_frac = overlap_map.get(m, 0.1)
                    mapped_res = np.interp(channel_info[m]['centers'], channel_info['jj']['centers'], jj_res_raw)
                    ov_counts = (b * ov_frac) * (1 + mapped_res)
                    ind_counts = np.random.poisson(b * (1 - ov_frac))
                    toy_dict[m] = np.maximum(0, np.round(ov_counts + ind_counts).astype(int))

        elif args.method == "copula":
            sampled = matrix[np.random.choice(len(matrix), size=np.random.poisson(n_mother_exp), replace=True)]
            for m, b in bkg_expectations.items():
                idx = col_names.index(f"M{m}")
                v = sampled[sampled[:, idx] >= 0, idx]
                if len(v) == 0:
                    toy_dict[m] = np.zeros_like(b)
                    continue

                target_n = int(len(v) * channel_scales[m])
                if target_n < len(v): v = np.random.choice(v, size=target_n, replace=False)

                # Applying a stronger Gaussian smear to fight the 1% staircase clumping
                U = np.clip(v + np.random.normal(0, 0.005, size=len(v)), 0, 1)
                toy_dict[m] = np.bincount(np.clip(np.searchsorted(cdfs[m], U), 0, len(b)-1), minlength=len(b))

        # --- 2. FIT THE GP TO THE GENERATED TOYS ---
        for m, toy_counts in toy_dict.items():
            widths = channel_info[m]['widths']
            c = channel_info[m]['centers']
            min_len = channel_info[m]['min_len']

            gp_density_toy, _, fit_ok = fit_gp_density(c, toy_counts, widths, min_len_scale=min_len)

            if not fit_ok:
                toy_successful = False
                break

            active_bkg_counts = gp_density_toy * widths
            max_t = max(max_t, fast_bumphunter_stat(toy_counts, active_bkg_counts))

        if toy_successful:
            stats.append(max_t)

    np.save(f"results/global_stat_{args.trigger}_{args.method}_{args.jobid}.npy", stats)
    print(f"\nDone. Acceptance Rate: {(len(stats)/attempts)*100:.1f}%. Saved to results/.")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--trigger', required=True, help="e.g., t1, t2")
    p.add_argument('--toys', type=int, default=1000)
    p.add_argument('--method', choices=["naive", "linear"], default="linear")
    p.add_argument('--jobid', type=str, default="local")
    main(p.parse_args())
