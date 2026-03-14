#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Default variables
TRIGGER=${1:-"t1"}      # First argument is the trigger, defaults to "t1"
TOYS=${2:-100000}       # Second argument is number of toys, defaults to 100000
CHIMAX=2.0

echo "======================================================"
echo " Starting $TOYS Pseudo-Experiments for Trigger: $TRIGGER"
echo "======================================================"

# Loop through all 3 methods
for METHOD in naive linear copula; do
    echo ""
    echo ">>> Running method: $METHOD"

    # Assuming this script is run from the 'scripts/' directory
    python3 ../python/run_toys.py \
        --trigger "$TRIGGER" \
        --method "$METHOD" \
        --toys "$TOYS" \
        --fit \
        --chimax "$CHIMAX" \
        -b

    echo ">>> Completed $METHOD"
done

echo ""
echo "======================================================"
echo " All methods completed successfully for $TRIGGER!"
echo " Results are stored in the ../results/ directory."
echo "======================================================"

