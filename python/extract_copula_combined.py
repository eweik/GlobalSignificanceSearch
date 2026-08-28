#!/usr/bin/env python3
"""
extract_copula_combined.py

Build a SINGLE copula matrix spanning all triggers and channels, so that
cross-trigger correlations (from shared events) are captured in the ranks.

Layout
------
- Rows: one per unique physical event (identified by number_run + number_event)
  across all 7 trigger files.
- Columns: 7 triggers x 9 mass channels = 63.  Column order is
      t1_Mjj, t1_Mbb, ..., t1_Mbg, t2_Mjj, ..., t7_Mbg.
- Values: rank/(n_valid+1) in (0,1) for events that (a) appear in that
  trigger's file, (b) pass that trigger's LogLoss cut, AND (c) have valid
  mass > 0.001 in that channel.  Everything else is -1.0 (missing).
- u_bounds: (63, 2) array of per-column phase-space truncation bounds.

Usage
-----
    python extract_copula_combined.py --root-dir data --out data/copula_combined.npz

The script expects files named
    {root-dir}/data{fraction}_t{i}_HAE_RUN23_nominal_10PB.root
for i in 1..7.  Adjust --fraction if needed.
"""
import os
import sys
import json
import argparse
import numpy as np
from scipy.stats import rankdata

current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

import ROOT
from src.config import ATLAS_BINS

CMS = 13000.0
MASS_VARS = ["Mjj", "Mbb", "Mjb", "Mje", "Mjm", "Mjg", "Mbe", "Mbm", "Mbg"]
ID_VARS   = ["number_run", "number_event"]

AR_CUTS = {
    1: -9.374504251376864,
    2: -9.829198438298226,
    3: -9.238664051471718,
    4: -9.744067418703318,
    5: -8.295795788735555,
    6: -9.841564554356868,
    7: -9.344795288421562,
}


def resolve_input(root_dir, fraction, trig):
    """Find the ROOT file for a trigger, tolerating the run2_ infix variant."""
    candidates = [
        os.path.join(root_dir, f"data{fraction}_t{trig}_HAE_RUN23_nominal_10PB.root"),
        # os.path.join(root_dir, f"data{fraction}_run2_t{trig}_HAE_RUN23_nominal_10PB.root"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[0]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root-dir",  default="/eos/atlas/atlascerngroupdisk/phys-exotics/jdm/lepdijet/AnomalyDetect23/ana/root",
                   help="Directory holding the per-trigger ROOT files.")
    p.add_argument("--fraction",  default="100percent",
                   help="Dataset fraction token in filenames.")
    p.add_argument("--out",       default="/tmp/edweik/edweik/copula_combined.npz",
                   help="Output path for the combined copula .npz.")
    args = p.parse_args()

    fits_dir = os.path.join(os.getcwd() if os.path.isdir("fits") else repo_root, "fits")
    triggers = list(range(1, 8))

    # ------------------------------------------------------------------
    # Pass 1: read every trigger file, apply its LogLoss cut, collect
    #         (run, event) IDs and the 9 mass arrays for passing events.
    # ------------------------------------------------------------------
    trig_data = {}  # trig -> {ids: set of (run,evt), ...}

    for trig in triggers:
        infile = resolve_input(args.root_dir, args.fraction, trig)
        if not os.path.isfile(infile):
            print(f"  t{trig}: file not found ({infile}), skipping.")
            continue

        print(f"  t{trig}: reading {infile} ...")
        df = ROOT.RDataFrame("output", infile)
        cols = ID_VARS + ["LogLoss"] + MASS_VARS
        try:
            d = df.AsNumpy(columns=cols)
        except Exception as e:
            print(f"  t{trig}: failed to read columns: {e}")
            continue

        cut = AR_CUTS[trig]
        keep = d["LogLoss"] >= cut
        n_pass = int(keep.sum())
        print(f"  t{trig}: {len(keep)} events, {n_pass} pass LogLoss >= {cut:.4f}")

        runs  = d["number_run"][keep]
        evts  = d["number_event"][keep]
        ids   = list(zip(runs.astype(np.int64), evts.astype(np.int64)))
        masses = {v: d[v][keep] for v in MASS_VARS}

        trig_data[trig] = {"ids": ids, "masses": masses}

    if not trig_data:
        print("No trigger files loaded. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build global event index: union of all (run, event) pairs.
    # ------------------------------------------------------------------
    all_ids = set()
    for td in trig_data.values():
        all_ids.update(td["ids"])
    all_ids = sorted(all_ids)  # deterministic row order
    id_to_row = {eid: row for row, eid in enumerate(all_ids)}
    N = len(all_ids)
    print(f"\nGlobal event index: {N} unique (run, event) pairs.")

    # ------------------------------------------------------------------
    # Build the combined matrix: 7 triggers x 9 channels = 63 columns.
    # ------------------------------------------------------------------
    n_cols = len(triggers) * len(MASS_VARS)
    col_names = [f"t{trig}_{var}" for trig in triggers for var in MASS_VARS]

    copula = np.full((N, n_cols), -1.0)
    u_bounds = np.tile([0.0, 1.0], (n_cols, 1))
    event_counts = {}

    for ti, trig in enumerate(triggers):
        if trig not in trig_data:
            for var in MASS_VARS:
                event_counts[f"t{trig}_{var}"] = 0
            continue

        td = trig_data[trig]
        rows = np.array([id_to_row[eid] for eid in td["ids"]], dtype=np.int64)

        for vi, var in enumerate(MASS_VARS):
            col_idx = ti * len(MASS_VARS) + vi
            col_name = col_names[col_idx]
            data = td["masses"][var]

            valid_mask = data > 0.001
            n_valid = int(valid_mask.sum())
            event_counts[col_name] = n_valid

            if n_valid > 0:
                valid_data = data[valid_mask]
                valid_rows = rows[valid_mask]
                ranks = rankdata(valid_data)
                copula[valid_rows, col_idx] = ranks / (n_valid + 1.0)

                # u_bounds from this trigger/channel's fit window
                ch = var[1:]  # "Mjj" -> "jj"
                fitfile = os.path.join(fits_dir, f"fitme_p5_t{trig}_{ch}.json")
                if os.path.isfile(fitfile):
                    d_fit = json.load(open(fitfile))
                    v_bins = ATLAS_BINS[(ATLAS_BINS >= float(d_fit['fmin'])) &
                                       (ATLAS_BINS <= float(d_fit['fmax']))]
                    if len(v_bins) >= 2:
                        phys = valid_data * CMS
                        u_bounds[col_idx] = [np.sum(phys < v_bins[0]) / n_valid,
                                             np.sum(phys <= v_bins[-1]) / n_valid]
                else:
                    print(f"  Warning: no fit file for t{trig}_{ch}; u_bounds left at (0,1).")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, copula=copula, columns=col_names, u_bounds=u_bounds)
    print(f"\nSaved combined copula matrix {copula.shape} -> {args.out}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    # Per-trigger overlap statistics
    print("\n" + "=" * 60)
    print("  CROSS-TRIGGER EVENT OVERLAP")
    print("=" * 60)
    for i, ti in enumerate(triggers):
        if ti not in trig_data: continue
        si = set(trig_data[ti]["ids"])
        for tj in triggers[i+1:]:
            if tj not in trig_data: continue
            sj = set(trig_data[tj]["ids"])
            ovlp = len(si & sj)
            if ovlp > 0:
                print(f"  t{ti} & t{tj}: {ovlp:>8,} shared events "
                      f"({100*ovlp/min(len(si),len(sj)):.1f}% of smaller stream)")

    # Per-column valid counts and u_bounds
    print("\n" + "=" * 60)
    print("  PER-COLUMN SUMMARY (trigger_channel : valid events, u_bounds)")
    print("=" * 60)
    for ci, cn in enumerate(col_names):
        ec = event_counts.get(cn, 0)
        ub = u_bounds[ci]
        print(f"  {cn:>10} : {ec:>8,} valid   ({ub[0]:.3f}, {ub[1]:.3f})")
    print("=" * 60)
    print(f"  Total rows (unique events): {N}")
    print(f"  Total columns: {n_cols}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
