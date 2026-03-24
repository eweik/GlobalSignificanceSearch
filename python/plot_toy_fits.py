#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from array import array
from argparse import ArgumentParser
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kFatal

current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)

from src.config import ATLAS_BINS, TRIGGER_OVERLAPS
from src.models import FiveParam

def create_tf1_base(name, cms, fmin, fmax):
    formula = (f"[0] * TMath::Power(1.0 - (x/{cms}), [1]) * "
               f"TMath::Power((x/{cms}), [2] + [3]*TMath::Log(x/{cms}) + [4]*TMath::Log(x/{cms})*TMath::Log(x/{cms}))")
    return ROOT.TF1(name, formula, fmin, fmax)

def perform_sequential_fit(toy, bkg_nom, channel_info, tf1_base, orig_params, args):
    edges = channel_info['bins']
    edges_root = array('d', edges)
    widths = channel_info['widths']
    h_tmp = ROOT.TH1D("h_tmp", "h_tmp", len(edges_root)-1, edges_root)
    h_tmp.SetDirectory(0)

    for j, val in enumerate(toy):
        h_tmp.SetBinContent(j+1, float(val))
        err = np.sqrt(val)
        if args.colleague_mode and val == 0:
            h_tmp.SetBinError(j+1, 0.0)
        else:
            h_tmp.SetBinError(j+1, err if err > 0 else 1.0)

    if np.sum(toy) < 50:
        return bkg_nom, None, False, 0, 0

    best_chi2 = float('inf')
    best_curve = None
    best_nparams = 0
    attempts_total = 0
    is_valid = False
    
    fit_string = "ISMRQ0" if args.colleague_mode else args.fitopts

    # SEQUENTIAL FIT LOOP: Try 2, 3, 4, then 5 parameters
    for n_params in range(2, 6):
        tf1 = tf1_base.Clone(f"tf1_{n_params}par")
        
        for i in range(5):
            if i < n_params:
                # Free the parameter. Give it a slight nudge if the nominal was 0.0
                start_val = orig_params[i] if orig_params[i] != 0.0 else 0.1
                tf1.SetParameter(i, start_val)
                if i >= 2: tf1.SetParLimits(i, -100.0, 100.0) # Straitjacket
            else:
                # Force the higher-order parameters to be exactly 0
                tf1.FixParameter(i, 0.0)

        passed_this_stage = False
        
        # Retry loop for this specific N-parameter configuration
        for attempt in range(args.retries):
            attempts_total += 1
            if attempt > 0:
                for i in range(n_params):
                    val = tf1.GetParameter(i)
                    tf1.SetParameter(i, val * np.random.uniform(0.9, 1.1))

            fit_status = int(h_tmp.Fit(tf1, fit_string))
            
            if fit_status == 0:
                ndf = tf1.GetNDF()
                chi2ndf = tf1.GetChisquare() / ndf if ndf > 0 else 999.0
                
                # If this parameter count hits the threshold, STOP adding parameters.
                if chi2ndf <= args.chimax:
                    best_chi2 = chi2ndf
                    centers = channel_info['centers']
                    best_curve = np.array([tf1.Eval(c) for c in centers])
                    best_nparams = n_params
                    is_valid = True
                    passed_this_stage = True
                    break

        # Break out of the sequential loop if we found a valid, simpler model
        if passed_this_stage:
            break
            
        # If we failed the 5-parameter fit, log it as the final attempt
        if n_params == 5 and not passed_this_stage:
            best_chi2 = chi2ndf
            centers = channel_info['centers']
            best_curve = np.array([tf1.Eval(c) for c in centers])
            best_nparams = 5

    return best_curve, best_chi2, is_valid, attempts_total, best_nparams

def main(args):
    os.makedirs("plots/toyfits", exist_ok=True)
    base_dir = repo_root

    bkg_expectations, channel_info, tf1_bases, orig_params_dict = {}, {}, {}, {}
    mass_types = ["jj", "bb", "jb", "je", "jm", "jg", "be", "bm", "bg"]
    
    for m in mass_types:
        fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{args.trigger}_{m}.json")
        try:
            with open(fitfile, "r") as jf:
                d = json.load(jf)
                fmin, fmax = float(d['fmin']), float(d['fmax'])
                v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin) & (ATLAS_BINS <= fmax)]
                c = (v_bins[:-1] + v_bins[1:]) / 2
                
                counts = FiveParam(args.cms, c, *d['parameters'])
                if np.sum(counts) > 0:
                    bkg_expectations[m] = counts
                    channel_info[m] = {'centers': c, 'bins': v_bins, 'widths': np.diff(v_bins)}
                    tf1_bases[m] = create_tf1_base(f"tf1_base_{m}", args.cms, fmin, fmax)
                    orig_params_dict[m] = [float(p) for p in d['parameters'][:5]]
        except Exception: pass

    if args.channel not in bkg_expectations:
        print(f"Error: Channel {args.channel} not found."); sys.exit(1)

    np.random.seed(args.seed)
    toys = {}
    
    toys['Naive'] = np.random.poisson(bkg_expectations[args.channel])
    
    jj_b = bkg_expectations['jj']
    jj_pseudo = np.random.poisson(jj_b)
    if args.channel == 'jj':
        toys['Linear'] = jj_pseudo
    else:
        jj_res = np.where(jj_b > 0, (jj_pseudo - jj_b) / jj_b, 0)
        ov_frac = TRIGGER_OVERLAPS.get(args.trigger, TRIGGER_OVERLAPS["default"]).get(args.channel, 0.1)
        mapped_res = np.interp(channel_info[args.channel]['centers'], channel_info['jj']['centers'], jj_res)
        b = bkg_expectations[args.channel]
        ov_counts = (b * ov_frac) * (1 + mapped_res)
        toys['Linear'] = np.maximum(0, np.round(ov_counts + np.random.poisson(b * (1 - ov_frac))).astype(int))

    copula_path = os.path.join(base_dir, "data", f"copula_{args.trigger}.npz")
    if os.path.exists(copula_path):
        f = np.load(copula_path)
        matrix, col_names = f['copula'], list(f['columns'])
        idx = col_names.index(f"M{args.channel}")
        b = bkg_expectations[args.channel]
        target_n = np.random.poisson(np.sum(b))
        
        v_correlated = matrix[matrix[:, idx] >= 0, idx]
        U_final = np.random.choice(v_correlated, size=target_n, replace=True)
        U_final += np.random.uniform(-0.0002, 0.0002, size=target_n)
        U_final = np.clip(np.abs(U_final), 0.0, 0.9999)
        
        cdf = np.cumsum(b) / np.sum(b)
        toys['Copula'] = np.bincount(np.searchsorted(cdf, U_final), minlength=len(b))
    else:
        toys['Copula'] = np.zeros_like(bkg_expectations[args.channel])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    mode_title = "COLLEAGUE MOCK (ISMRQ0 + Ignore 0s)" if args.colleague_mode else f"Fit: '{args.fitopts}'"
    title_str = f"Sequential Fitting: Trigger {args.trigger.upper()} | Channel {args.channel.upper()} | {mode_title}"
    fig.suptitle(title_str, fontsize=16)

    centers = channel_info[args.channel]['centers']
    widths = channel_info[args.channel]['widths']
    nom_plot = bkg_expectations[args.channel] / widths
    
    for ax, (method, toy) in zip(axes, toys.items()):
        fit_curve, chi2, is_valid, attempts, nparams = perform_sequential_fit(
            toy, bkg_expectations[args.channel], channel_info[args.channel], 
            tf1_bases[args.channel], orig_params_dict[args.channel], args
        )
        
        toy_plot = toy / widths
        toy_err = np.sqrt(toy) / widths
        
        ax.errorbar(centers, toy_plot, yerr=toy_err, fmt='ko', markersize=4, label='Toy Data (dN/dm)')
        ax.plot(centers, nom_plot, 'b--', alpha=0.6, label='Nominal Bkg')
        
        if chi2 is not None:
            fit_plot = fit_curve / widths
            color = 'r' if is_valid else 'orange'
            status = "PASS" if is_valid else "FAIL"
            label_str = f'Fit ({nparams}-Param, $\chi^2/ndf$={chi2:.2f} [{status}])\nRetries: {attempts}'
            ax.plot(centers, fit_plot, color, linewidth=2, label=label_str)
        else:
            ax.plot(centers, nom_plot, 'gray', label=f'Fit Crash/Skip\nRetries: {attempts}')
            
        ax.set_title(f"{method} Method")
        ax.set_xlabel("Mass [GeV]")
        if ax == axes[0]: ax.set_ylabel("Events / GeV")
        ax.set_yscale('log')
        ax.set_ylim(0.001, max(np.max(toy_plot)*2, 10))
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = "ISMRQ0" if args.colleague_mode else f"{args.fitopts}"
    out_name = f"plots/toyfits/methods_seq_fit_{args.trigger}_{args.channel}_{suffix}.png"
    plt.savefig(out_name, dpi=300)
    print(f"Saved sequential fit plot to {out_name}")

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--trigger', required=True)
    p.add_argument('--channel', required=True)
    p.add_argument('--cms', type=float, default=13600.)
    p.add_argument('--chimax', type=float, default=2.0)
    p.add_argument('--fitopts', type=str, default="ILSRQ0")
    p.add_argument('--retries', type=int, default=10)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--colleague_mode', action='store_true')
    main(p.parse_args())
