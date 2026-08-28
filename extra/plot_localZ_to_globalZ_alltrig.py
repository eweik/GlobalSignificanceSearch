#!/usr/bin/env python3
"""
Local Z vs Global Z for the ENTIRE dataset searched (all triggers combined),
treating the trigger streams as independent ("distinct data").

Each trigger's null (final_{trig}_{method}...) is already the max over that
trigger's 9 channels (cross-channel correlations handled by the copula in
run_toys). Under trigger independence the whole-dataset global tail is

    S(t) = P(t_max over all triggers & channels >= t) = 1 - prod_i F_i(t),

with F_i the per-trigger empirical CDF. The curve is the parametric map
    t  ->  ( Z_local(t), Z_global(t) ) = ( isf(exp(-t)), isf(S(t)) ),
which reduces to the usual per-trigger rank method when there is one trigger.
"""
import glob
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def find_trigger_arrays(trig, method, bkg_tag):
    """Locate a trigger's merged null, trying the known naming conventions."""
    candidates = [
        f"results/merged/final_{trig}_{method}_{bkg_tag}.npy",
        f"results/merged/final_{trig}_{method}.npy",
        # f"results/merged_5param/final_{trig}_{method}.npy",
    ]
    for pat in candidates:
        hits = glob.glob(pat)
        if hits:
            a = np.concatenate([np.load(f) for f in hits])
            return a[np.isfinite(a)]
    return None


def main():
    parser = argparse.ArgumentParser(description="All-trigger Local Z -> Global Z (independent triggers).")
    parser.add_argument("--ExpectedLocalZvalue", type=float, default=6.0,
                        help="Expected local significance (default: 5.0)")
    parser.add_argument("--bkg", choices=["func", "matrix"], default="func",
                        help="Background model used for the pseudo-experiments")
    parser.add_argument("--triggers", type=str, default="1-7",
                        help="Triggers to combine, e.g. '1-7' or '1,2,4' (default: 1-7)")
    args = parser.parse_args()

    bkg_tag = "BKGfunc" if args.bkg == "func" else "BKGmatrix"

    if "-" in args.triggers:
        lo, hi = args.triggers.split("-")
        trig_list = [f"t{i}" for i in range(int(lo), int(hi) + 1)]
    else:
        trig_list = [f"t{i.strip()}" for i in args.triggers.split(",")]

    # methods = ["naive", "copula", "poisson_event", "decorrelated_bootstrap", "decorrelated_copula"]
    methods = ["copula"]
    colors = {"naive": "red", "linear": "blue", "copula": "green",
              "poisson_event": "blue", "decorrelated_copula": "purple",
              "decorrelated_bootstrap": "olive", "gaussian_copula": "rebeccapurple",
              "student_t_copula": "lightcoral"}
    method_label_map = {"naive": "Independent", "linear": "Overlap", "copula": "Empirical Copula",
                        "poisson_event": "Poisson Bootstrap", "decorrelated_bootstrap": "Decorrelated Bootstrap",
                        "decorrelated_copula": "Decorrelated Copula", "gaussian_copula": "Gaussian Copula",
                        "student_t_copula": "Student-t Copula"}

    os.makedirs("plots", exist_ok=True)

    print(f"\n############## START (ALL-TRIGGER) ################")
    print(f"Whole-dataset LEE for Local Z >= {args.ExpectedLocalZvalue}")
    print(f"Triggers combined (independent): {trig_list} | Background: {bkg_tag}")

    # Local-Z threshold -> BumpHunter t threshold: z = isf(exp(-t)) => t = -ln(sf(z))
    t_thresh = -np.log(stats.norm.sf(args.ExpectedLocalZvalue))

    plt.figure(figsize=(10, 6))

    for method in methods:
        per_trig, used = [], []
        for trig in trig_list:
            a = find_trigger_arrays(trig, method, bkg_tag)
            if a is not None and len(a) > 0:
                per_trig.append(a); used.append(trig)

        if len(per_trig) == 0:
            print(f"Warning: no data for {method} (tag {bkg_tag}). Skipping.")
            continue
        if len(per_trig) < 2:
            print(f"Note: only {len(per_trig)} trigger ({used}) for {method}; "
                  f"this is a single-trigger curve, not the full combination.")

        Ns = [len(a) for a in per_trig]
        sorted_arr = [np.sort(a) for a in per_trig]
        floor_p = 1.0 / max(Ns)  # best resolvable global p (one toy in the deepest tail)

        # --- Combined tail S(t) = 1 - prod_i P(t_i < t), evaluated on a fine grid ---
        tmax = max(a.max() for a in per_trig)
        tgrid = np.linspace(0.0, tmax, 4000)
        prod = np.ones_like(tgrid)
        for s, N in zip(sorted_arr, Ns):
            prod *= np.searchsorted(s, tgrid, side="left") / N
        S = 1.0 - prod  # P(combined t_max >= t)

        z_local = stats.norm.isf(np.clip(np.exp(-tgrid), 1e-300, 0.999999))
        z_global = stats.norm.isf(np.clip(S, 1e-300, 0.999999))
        valid = (S > 0) & np.isfinite(z_global) & np.isfinite(z_local) & (z_global > -10) & (z_local >= 0)

        # --- Whole-dataset global p at the requested local Z ---
        p_i = [float((a >= t_thresh).mean()) for a in per_trig]
        p_global = 1.0 - np.prod([1.0 - pi for pi in p_i])

        print(f"\n###### RESULT: {method.upper()} ######")
        print(f" Triggers used = {used} (toys each: {Ns})")
        print(f" Per-trigger global p_i at Local Z={args.ExpectedLocalZvalue}: {['%.2e' % pi for pi in p_i]}")
        if p_global > 0 and not any(pi == 0.0 for pi in p_i):
            print(f" Whole-dataset global p = {p_global:.2e}  or Global Z = {stats.norm.isf(p_global):.2f}")
        else:
            print(f" Whole-dataset global p < {floor_p:.2e} "
                  f"(Local Z={args.ExpectedLocalZvalue} beyond toys in >=1 trigger; needs tail extrapolation)")
        print(f"###### END RESULT ######")

        label_name = method_label_map.get(method, method.capitalize())
        plt.plot(z_local[valid], z_global[valid],
                 label=f"{label_name}",
                 color=colors.get(method, "black"), lw=2)

    plt.title("Whole-Dataset Global vs. Local Significance\n(All Triggers, Independent)", fontsize=18)
    plt.xlabel("Highest Observed Local Significance Across All Windows & Triggers ($Z_{local}$)", fontsize=14)
    plt.ylabel("Global Significance ($Z_{global}$)", fontsize=14)

    plt.axhline(3, color='grey', linestyle='--', alpha=0.7, label='3σ Global Evidence')
    plt.axhline(5, color='black', linestyle=':', alpha=0.7, label='5σ Global Discovery')

    lims = [max(0, plt.xlim()[0]), min(8, plt.xlim()[1])]
    plt.plot(lims, lims, 'k--', alpha=0.3, label="No LEE ($Z_{global} = Z_{local}$)")

    plt.legend(loc="lower right")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plot_out = f"plots/Local_vs_Global_Z_ALLTRIG_{bkg_tag}.png"
    plt.savefig(plot_out, dpi=300)
    print(f"\nPlot saved to {plot_out}")


if __name__ == "__main__":
    main()
