#!/bin/bash
# run_grid.sh - Generate a 3x3 visualization grid of copula shifts

# Default parameters
TRIGGER="t1"
MASS=2000.0
WIDTH=80.0
EVENTS=5000

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --trigger) TRIGGER="$2"; shift ;;
        --mass) MASS="$2"; shift ;;
        --width) WIDTH="$2"; shift ;;
        --events) EVENTS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================="
echo "Running 9-Panel Grid Simulation"
echo "Trigger:   $TRIGGER"
echo "Mass:      $MASS GeV"
echo "Width:     $WIDTH GeV"
echo "Events:    $EVENTS"
echo "========================================="

# Execute the new grid python script
python3 python/plot_9panel_grid.py \
    --trigger "$TRIGGER" \
    --mass "$MASS" \
    --width "$WIDTH" \
    --events "$EVENTS"

echo "Done!"
