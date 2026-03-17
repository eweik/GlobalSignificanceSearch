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
        if float(val) == 0.0:
            back.FixParameter(idx, 0.0)
    return back

def do_fit_and_get_bkg(toy_data, m, original_bkg, channel_info, tf1_templates, args, syst_env):
    if not args.fit:
        return original_bkg, True
        
    edges = channel_info[m]['bins']
    edges_root = array('d', edges)
    widths = np.diff(edges)
    h_name = f"h_tmp_{m}"
    
    h_tmp = ROOT.TH1D(h_name, h_name, len(edges_root)-1, edges_root)
    h_tmp.SetDirectory(0)
    
    for j, val in enumerate(toy_data):
        if val > 0:
            # Fit requires Density (Counts / Width) to remain smooth
            h_tmp.SetBinContent(j+1, val / widths[j])
            
            # THE MATHEMATICAL FIX: Combine Statistical and Systematic Errors
            stat_err = np.sqrt(val)
            syst_err = syst_env[j]
            tot_err = np.sqrt(stat_err**2 + syst_err**2)
            
            h_tmp.SetBinError(j+1, tot_err / widths[j])
            
    base_tf1 = tf1_templates[m]
    n_params = base_tf1.GetNpar()
    orig_params = [base_tf1.GetParameter(i) for i in range(n_params)]
    
    max_fit_attempts = 100
    best_tf1 = None
    min_chi2ndf = float('inf')
    
    for attempt in range(max_fit_attempts):
        tf1_try = base_tf1.Clone(f"tf1_try_{m}_{attempt}")
        
        if attempt > 0:
            for i in range(n_params):
                if orig_params[i] != 0.0:
                    shift = random.uniform(0.5, 1.5)
                    if i == 3 and random.random() > 0.5:
                        shift *= -1
                    tf1_try.SetParameter(i, orig_params[i] * shift)
                    
        #fit_result = h_tmp.Fit(tf1_try, "ISR0Q")
        fit_result = h_tmp.Fit(tf1_try, "ISRM0Q")
        
        ndf = tf1_try.GetNDF()
        chi2ndf = tf1_try.GetChisquare() / ndf if ndf > 0 else float('inf')
        
        if fit_result.IsValid() and chi2ndf < min_chi2ndf:
            min_chi2ndf = chi2ndf
            best_tf1 = tf1_try.Clone(f"best_tf1_{m}")
            
            if chi2ndf <= args.chimax:
                break
                
    if best_tf1 is None or min_chi2ndf > args.chimax:
        h_tmp.Delete()
        return original_bkg, False 
        
    centers = channel_info[m]['centers']
    active_bkg = np.clip(np.array([best_tf1.Eval(c) for c in centers]) * widths, 0.0, 1e7)
    
    h_tmp.Delete()
    return active_bkg, True

