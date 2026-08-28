#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse


def combine_across_triggers(methods, z_scores, percentiles, p_values, args):
    """All-triggers/all-channels LEE, treating triggers as independent data.

    Each trigger's merged null (final_{trig}_{method}.npy) is already the max
    over that trigger's 9 channels (cross-channel correlations handled by the
    copula in run_toys). Under trigger independence, the combined global t_max
    is the max over triggers, so:
        P(t_max >= t) = 1 - prod_i F_i(t),   F_i = per-trigger empirical CDF.
    Thresholds and the observed p-value are computed exactly from this product;
    resampling is used only to save/plot a representative combined null.
    """
    triggers = [f"t{i}" for i in range(1, 8)]
    rng = np.random.default_rng(0)
    print(f"\n{'='*50}\nALL-TRIGGER LEE (triggers treated as INDEPENDENT)\n{'='*50}")

    plt.figure(figsize=(10, 6))
    for method in methods:
        # Collect each trigger's merged null
        arrays, used = [], []
        for trig in triggers:
            path = f"results/merged/final_{trig}_{method}.npy"
            if os.path.exists(path):
                a = np.load(path)
                a = a[np.isfinite(a)]
                if len(a) > 0:
                    arrays.append(a); used.append(trig)
        if len(arrays) < 2:
            print(f"[{method}] only {len(arrays)} trigger(s) available; "
                  f"cross-trigger LEE needs >=2. Skipping.")
            continue

        Ns = [len(a) for a in arrays]
        sorted_arr = [np.sort(a) for a in arrays]

        # Exact combined upper tail S(t) = P(max over triggers >= t)
        def S(t):
            prod = 1.0
            for s in sorted_arr:
                prod *= np.searchsorted(s, t, side='left') / len(s)
            return 1.0 - prod

        grid = np.unique(np.concatenate(arrays))
        Sgrid = np.array([S(t) for t in grid])
        min_resolvable = Sgrid[Sgrid > 0].min() if np.any(Sgrid > 0) else 1.0

        # Resampled combined null (for the saved array + plot only)
        M = args.ncombine if args.ncombine else max(Ns)
        combined = np.full(M, -np.inf)
        for a in arrays:
            combined = np.maximum(combined, rng.choice(a, size=M, replace=True))
        np.save(f"results/merged/final_ALLTRIG_{method}.npy", combined)

        print(f"\n[{method.upper()}] combined {len(arrays)} triggers {used}")
        print(f"  toys per trigger: {Ns}")
        print(f"  Combined mean t_max: {combined.mean():.2f} | "
              f"empirical max t_max: {combined.max():.2f}")
        print("  Global (all-trigger) thresholds (t_max required):")
        for z, p in zip(z_scores, p_values):
            if p < min_resolvable:
                print(f"    {z} sigma global -> need more per-trigger toys "
                      f"(tail not resolved; min resolvable p ~ {min_resolvable:.1e}).")
                continue
            below = grid[Sgrid <= p]
            thr = below[0] if len(below) else grid[-1]
            print(f"    {z} sigma global -> t_max > {thr:.2f}")

        if args.observed is not None:
            t_obs = args.observed
            p_i = [float((a >= t_obs).mean()) for a in arrays]
            p_glob = 1.0 - np.prod([1.0 - pi for pi in p_i])
            print(f"  Observed t_obs = {t_obs:.3f}")
            print(f"    per-trigger p_i: {['%.2e' % pi for pi in p_i]}")
            if p_glob <= 0.0 or any(pi == 0.0 for pi in p_i):
                print(f"    -> GLOBAL p < ~{min_resolvable:.1e} "
                      f"(t_obs beyond toys in >=1 trigger; needs tail extrapolation)")
            else:
                print(f"    -> GLOBAL p = {p_glob:.3e}  ({norm.isf(p_glob):.2f} sigma)")

        plt.hist(combined, bins=100, range=(0, max(50, combined.max())),
                 histtype='step', linewidth=2, density=True, log=True,
                 label=f"{method.capitalize()} (all-trig, N={M})")

    plt.title("All-Trigger Global $t_{max}$ (independent-trigger LEE)")
    plt.xlabel(r"$t_{max}$ (max over all triggers & channels)")
    plt.ylabel("Probability Density")
    plt.legend(loc="upper right")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/LEE_Distributions_ALLTRIG.png", dpi=300)
    plt.close()
    print("\nSaved plot to plots/LEE_Distributions_ALLTRIG.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge toys and compute LEE.")
    parser.add_argument("--observed", type=float, default=None,
                        help="Observed global t_max of the excess; prints its all-trigger global p-value.")
    parser.add_argument("--ncombine", type=int, default=None,
                        help="Number of resampled combined toys for the saved/plotted combined null.")
    args = parser.parse_args()

    z_scores = [1, 2, 3, 4, 5, 6]
    p_values = [norm.sf(z) for z in z_scores]
    percentiles = [(1.0 - p) * 100 for p in p_values]
    # methods = ["naive", "linear", "copula", "poisson_event",
    #            "exclusive_categories", "decorrelated_bootstrap",
    #            "decorrelated_copula", "gaussian_copula", "student_t_copula"]
    methods = ["copula"]
    combine_across_triggers(methods, z_scores, percentiles, p_values, args)
