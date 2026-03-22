#!/bin/bash

TRIGGER=${1:-t2}
HUNT_FLAG=$2

echo "========================================================="
echo "Generating 9-Panel Method Comparison for Trigger: $TRIGGER"
echo "========================================================="

if [ "$HUNT_FLAG" == "--hunt" ]; then
    echo "HUNT MODE ENABLED: Scanning universes until Copula breaks (Local Z > 5)..."
    python3 python/plot_method_comparison.py --trigger $TRIGGER --hunt
else
    python3 python/plot_method_comparison.py --trigger $TRIGGER
fi
