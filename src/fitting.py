import numpy as np
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

def do_fit_and_get_bkg(toy_data, m, original_bkg, channel_info, tf1_templates, args):
    if not args.fit:
        return original_bkg, True
        
    edges = channel_info[m]['bins']
    edges_root = array('d', edges)
    h_name = f"h_tmp_{m}"
    
    h_tmp = ROOT.TH1D(h_name, h_name, len(edges_root)-1, edges_root)
    h_tmp.SetDirectory(0)
    
    for j, val in enumerate(toy_data):
        if val > 0:
            h_tmp.SetBinContent(j+1, val)
            h_tmp.SetBinError(j+1, np.sqrt(val))
            
    base_tf1 = tf1_templates[m]
    n_params = base_tf1.GetNpar()
    orig_params = [base_tf1.GetParameter(i) for i in range(n_params)]
    
    max_fit_attempts = 5
    fit_passed = False
    best_tf1 = None
    
    # Retry loop with parameter shifting
    for attempt in range(max_fit_attempts):
        tf1_try = base_tf1.Clone(f"tf1_try_{m}_{attempt}")
        
        # Shift parameters by +/- 10% on subsequent attempts
        if attempt > 0:
            for i in range(n_params):
                if orig_params[i] != 0.0: # Don't shift fixed parameters
                    shift_factor = np.random.uniform(0.9, 1.1)
                    tf1_try.SetParameter(i, orig_params[i] * shift_factor)
                    
        fit_result = h_tmp.Fit(tf1_try, "ISMR0Q")
        
        ndf = tf1_try.GetNDF()
        chi2ndf = tf1_try.GetChisquare() / ndf if ndf > 0 else float('inf')
        
        if fit_result.IsValid() and chi2ndf <= args.chimax:
            fit_passed = True
            best_tf1 = tf1_try
            break
            
    if not fit_passed:
        h_tmp.Delete()
        return original_bkg, False # Completely failed after all retries
        
    centers = channel_info[m]['centers']
    widths = np.diff(edges)
    active_bkg = np.clip(np.array([best_tf1.Eval(c) for c in centers]) * widths, 0.0, 1e7)
    
    h_tmp.Delete()
    return active_bkg, True
