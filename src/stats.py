import numpy as np
from scipy.stats import poisson

def fast_bumphunter_stat(data_hist, bkg_hist):
    """Vectorized sliding window equivalent to pyBumpHunter."""
    max_window = len(data_hist) // 2
    max_t = 0.0
    for w in range(2, max_window + 1):
        k = np.ones(w)
        D = np.convolve(data_hist, k, mode='valid')
        B = np.convolve(bkg_hist, k, mode='valid')
        mask = D >= B
        if not np.any(mask): 
            continue
        p = np.clip(poisson.sf(D[mask] - 1, B[mask]), 1e-300, 1.0)
        max_t = max(max_t, np.max(-np.log(p)))
    return max_t
