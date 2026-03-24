import numpy as np
import random
from array import array

try:
    import ROOT
except ImportError:
    ROOT = None

def setup_root_env(batch=True, fit_enabled=False):
    if not ROOT:
        raise ImportError("ROOT module not found. Please ensure PyROOT is set up.")
    if batch:
        ROOT.gROOT.SetBatch(True)
    if fit_enabled:
        ROOT.gErrorIgnoreLevel = ROOT.kFatal 

def create_tf1_template(name, cms, fmin, fmax, params):
    formula = (
        f"[0] * TMath::Power(1.0 - (x/{cms}), [1]) * "
        f"TMath::Power((x/{cms}), [2] + [3]*TMath::Log(x/{cms}) + [4]*TMath::Log(x/{cms})*TMath::Log(x/{cms}))"
    )
    back = ROOT.TF1(name, formula, fmin, fmax)

    for idx, val in enumerate(params[:5]):
        back.SetParameter(idx, float(val))
        # This locks the degrees of freedom to exactly match the JSON Null Hypothesis
        if float(val) == 0.0:
            back.FixParameter(idx, 0.0)
        elif idx >= 2:
            # The Straitjacket to prevent the function from numerical divergence
            margin = abs(float(val)) * 0.50
            back.SetParLimits(idx, float(val) - margin, float(val) + margin)
            
    return back

def do_fit_and_get_bkg(toy_data, m, original_bkg, channel_info, tf1_template, args, syst_env):
    if not args.fit:
        return original_bkg, True

    edges = channel_info[m]['bins']
    edges_root = array('d', edges)
    widths = np.diff(edges)
    h_name = f"h_tmp_{m}"

    h_tmp = ROOT.TH1D(h_name, h_name, len(edges_root)-1, edges_root)
    h_tmp.SetDirectory(0)

    total_events = 0
    for j, val in enumerate(toy_data):
        if val > 0:
            total_events += val
            h_tmp.SetBinContent(j+1, val / widths[j])

            stat_err = np.sqrt(val)
            syst_err = syst_env[j]
            tot_err = np.sqrt(stat_err**2 + syst_err**2)

            h_tmp.SetBinError(j+1, tot_err / widths[j])

    if total_events < 50: 
        return original_bkg, False

    n_params = tf1_template.GetNpar()
    orig_params = [tf1_template.GetParameter(i) for i in range(n_params)]

    best_chi2 = float('inf')
    best_params = None
    max_fit_attempts = 15 # Give MINUIT a few chances to find the minimum for this shape

    for attempt in range(max_fit_attempts):
        if attempt == 0:
            for i in range(n_params):
                tf1_template.SetParameter(i, orig_params[i])
        else:
            for i in range(n_params):
                if orig_params[i] != 0.0: # Only nudge parameters that are floating
                    shift = random.uniform(0.9, 1.1)
                    if i == 3 and random.random() > 0.5: shift *= -1
                    tf1_template.SetParameter(i, orig_params[i] * shift)

        # RM0QN: Range, Improve, Zero Graphics, Quiet, No Store
        # fit_status = int(h_tmp.Fit(tf1_template, "RM0QN"))
        fit_status = int(h_tmp.Fit(tf1_template, "IRM0QN"))

        if fit_status == 0:
            ndf = tf1_template.GetNDF()
            chi2ndf = tf1_template.GetChisquare() / ndf if ndf > 0 else float('inf')
            
            if chi2ndf < best_chi2:
                best_chi2 = chi2ndf
                best_params = [tf1_template.GetParameter(i) for i in range(n_params)]
                if chi2ndf <= args.chimax:
                    break

    if best_params is None or best_chi2 > args.chimax:
        return original_bkg, False

    for i in range(n_params):
        tf1_template.SetParameter(i, best_params[i])

    centers = channel_info[m]['centers']
    active_bkg = np.clip(np.array([tf1_template.Eval(c) for c in centers]) * widths, 0.0, 1e7)

    return active_bkg, True
