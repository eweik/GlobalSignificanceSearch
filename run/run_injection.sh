#!/bin/bash
# run_injection.sh - Run signal injection visualizations manually from bash

# Default parameters
TRIGGER="t1"
MASS=2000.0
WIDTH=80.0
EVENTS=5000

# Parse command line arguments (optional overrides)
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
echo "Running Signal Injection Visualization"
echo "Trigger:   $TRIGGER"
echo "Mass:      $MASS GeV"
echo "Width:     $WIDTH GeV"
echo "Events:    $EVENTS"
echo "========================================="

# Execute the python script with the provided parameters
python ../python/visualize_copula_shift.py \
    --trigger "$TRIGGER" \
    --mass "$MASS" \
    --width "$WIDTH" \
    --events "$EVENTS"

echo "Done! Check the results/spike_points_*.png files for results."
