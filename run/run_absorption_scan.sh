#!/bin/bash

# Configuration
TRIGGER="t2"
INJ_SIG=5.0
WIDTH=0.05
TOYS_PER_BIN=20

CHANNELS=("jj" "jb" "bb" "je" "jm" "jg" "be" "bm" "bg")

# Create output log directory
mkdir -p logs/absorption

echo "======================================================"
echo " Starting Background-Only Signal Absorption Scan"
echo " Trigger: ${TRIGGER^^} | Target Significance: ${INJ_SIG} sigma"
echo "======================================================"

for chan in "${CHANNELS[@]}"; do
    echo "Submitting ${TRIGGER^^} | Channel ${chan}..."
    
    # Run in the background (&) to parallelize
    python3 python/scan_signal_absorption.py \
        --trigger "${TRIGGER}" \
        --channel "${chan}" \
        --sig_inj "${INJ_SIG}" \
        --width_frac "${WIDTH}" \
        --toys "${TOYS_PER_BIN}" \
        > "logs/absorption/scan_${TRIGGER}_${chan}.log" 2>&1
done

# Wait for all background jobs to finish
wait

echo "======================================================"
echo " All channels complete. Check plots/absorption/ directory."
echo "======================================================"
