#!/bin/bash

# Default to "all" if no trigger is specified
TRIGGER=${1:-"all"}

echo "Starting merge process for trigger: $TRIGGER..."

# Run the python script
python3 python/merge_results.py --trigger "$TRIGGER"

echo "Merge complete."
echo "You can view your plots in the plots/ directory."
echo "Your consolidated numpy arrays are in results/merged/"
