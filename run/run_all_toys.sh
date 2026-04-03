#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Default variables
TRIGGER=${1:-"t2"}      # First argument is the trigger, defaults to "t1"
TOYS=${2:-100}       # Second argument is number of toys, defaults to 100000
USE_GP=${3:-"default"}  # Third argument switches to GP mode

echo "======================================================"
echo " Starting $TOYS Pseudo-Experiments for Trigger: $TRIGGER"
echo "======================================================"

if [ "$USE_GP" == "gp-toy" ]; then
    TARGET_SCRIPT="python/run_toys_gp.py"
    EXTRA_ARGS=""
    echo " Background Model: Gaussian Process"
else
    TARGET_SCRIPT="python/run_toys.py"
    CHIMAX=2.0
    # EXTRA_ARGS="--fit --chimax $CHIMAX -b"
    echo " Background Model: 5-Parameter Minuit Fit"
fi
echo "======================================================"

# Loop through all 3 methods
# for METHOD in poisson_event exclusive_categories; do
# for METHOD in naive linear copula poisson_event exclusive_categories; do
for METHOD in linear; do
    echo ""
    echo ">>> Running method: $METHOD"

    # Assuming this script is run from the root directory
    python3 "$TARGET_SCRIPT" \
        --trigger "$TRIGGER" \
        --method "$METHOD" \
        --toys "$TOYS" \
        --jobid "local" \
        $EXTRA_ARGS

    echo ">>> Completed $METHOD"
done
    
echo ""
echo "======================================================"
echo " All methods completed successfully for $TRIGGER!"
echo " Results are stored in the results/ directory."
echo "======================================================"
