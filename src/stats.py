import numpy as np
from scipy.stats import poisson
from scipy.special import gammainc

# =================================================================
# 1. Fast Vectorized BumpHunter Engine
# =================================================================
def fast_bumphunter_stat(data_hist, bkg_hist, max_width=30):
    """Vectorized sliding window BumpHunter. Returns max local test stat t = -ln(p)."""
    # 1. Replicate pyBumpHunter's array cropping (first to last non-zero bin)
    non_zero_idx = np.where(bkg_hist > 0)[0]
    if len(non_zero_idx) == 0:
        return 0.0

    h_inf, h_sup = non_zero_idx[0], non_zero_idx[-1] + 1
    d_crop = data_hist[h_inf:h_sup]
    b_crop = bkg_hist[h_inf:h_sup]

    max_t = 0.0
    for w in range(2, max_width + 1):
        k = np.ones(w)
        D = np.convolve(d_crop, k, mode='valid')
        B = np.convolve(b_crop, k, mode='valid')

        # 2. Only look for excesses where background > 0
        mask = (D > B) & (B > 0)
        if not np.any(mask):
            continue

        # 3. Use continuous incomplete gamma to perfectly match pyBH's fractional handling
        p = np.clip(gammainc(D[mask], B[mask]), 1e-300, 1.0)
        max_t = max(max_t, np.max(-np.log(p)))

    return max_t
