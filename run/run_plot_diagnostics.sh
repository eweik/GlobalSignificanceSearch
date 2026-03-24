#!/bin/bash
# Usage: ./run_plot_methods.sh <trigger> <channel> <seed> <chimax>
# Example: ./run_plot_methods.sh t2 jb 42 2.0

TRIGGER=${1:-t2}
CHANNEL=${2:-jb}
SEED=${3:-42}
CHIMAX=${4:-5.0}

echo "=========================================================="
echo " Plotting Generation Methods: Trigger $TRIGGER, Channel $CHANNEL"
echo " Seed: $SEED | Max Chi2: $CHIMAX"
echo "=========================================================="

python3 python/plot_toy_fits.py \
    --trigger $TRIGGER \
    --channel $CHANNEL \
    --seed $SEED \
    --chimax $CHIMAX \
    --fitopts "ILSRQ0" \
    --retries 20

echo "Done. Check the figs/ directory for the output PNG."
