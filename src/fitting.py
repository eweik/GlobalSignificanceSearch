import numpy as np
from array import array

try:
    import ROOT
except ImportError:
    ROOT = None

class FiveParam2015:
    """ROOT TF1 callable class for 5-parameter background function."""
    def __init__(self, cms):
        self.cms = cms
        
    def __call__(self, x, par):
        xx = x[0] / self.cms
        if xx <= 0 or xx >= 1: return 0.0
        ff1 = par[0] * ROOT.TMath.Power((1.0 - xx), par[1])
        ff2 = ROOT.TMath.Power(xx, (par[2] + par[3] * ROOT.TMath.Log(xx) + par[4] * ROOT.TMath.Log(xx) * ROOT.TMath.Log(xx)))
        return ff1 * ff2

def setup_root_env(batch=True, fit_enabled=False):
    """Configures global ROOT settings."""
    if not ROOT:
        raise ImportError("ROOT module not found. Please ensure PyROOT is set up.")
    if batch:
        ROOT.gROOT.SetBatch(True)
    if fit_enabled:
        ROOT.gErrorIgnoreLevel = ROOT.kFatal # Suppress GSL roundoff noise

def create_tf1_template(name, cms, fmin, fmax, params):
    """Creates and returns a pre-configured ROOT TF1 object."""
    back = ROOT.TF1(name, FiveParam2015(cms), fmin, fmax, 5)
    for idx, val in enumerate(params[:5]):
        back.SetParameter(idx, float(val))
        if float(val) == 0.0: 
            back.FixParameter(idx, 0.0)
    return back

def do_fit_and_get_bkg(toy_data, m, original_bkg, channel_info, tf1_templates, args):
    """Fits pseudo-data and returns updated active background if successful."""
    if not args.fit:
        return original_bkg, True
        
    h_name = f"h_tmp_{m}"
    edges = channel_info[m]['bins']
    edges_root = array('d', edges)
    
    h_tmp = ROOT.TH1D(h_name, h_name, len(edges_root)-1, edges_root)
    h_tmp.SetDirectory(0)
    
    for j, val in enumerate(toy_data):
        if val > 0:
            h_tmp.SetBinContent(j+1, val)
            h_tmp.SetBinError(j+1, np.sqrt(val))
            
    tf1_template = tf1_templates[m].Clone()
    fit_result = h_tmp.Fit(tf1_template, "ISMR0Q")
    
    ndf = tf1_template.GetNDF()
    chi2ndf = tf1_template.GetChisquare() / ndf if ndf > 0 else float('inf')
    
    if not fit_result.IsValid() or chi2ndf > args.chimax:
        h_tmp.Delete()
        return original_bkg, False
        
    centers = channel_info[m]['centers']
    widths = np.diff(edges)
    active_bkg = np.clip(np.array([tf1_template.Eval(c) for c in centers]) * widths, 0.0, 1e7)
    
    h_tmp.Delete()
    return active_bkg, True
