#!/bin/bash
# run_all_toys.sh - Generate global significance toys and plot test statistics

TRIGGER="t1"
TOYS=10000
CMS=13600.0
FORCE=0

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --trigger) TRIGGER="$2"; shift ;;
        --toys) TOYS="$2"; shift ;;
        --cms) CMS="$2"; shift ;;
        --force) FORCE=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================================="
echo "Global Significance Toy Production - Trigger: $TRIGGER"
echo "========================================================="

METHODS=("naive" "linear" "copula")

for METHOD in "${METHODS[@]}"; do
    FILE="../results/global_stat_${TRIGGER}_${METHOD}.npy"

    # Check if toys already exist and we are not forcing an overwrite
    if [ -f "$FILE" ] && [ "$FORCE" -eq 0 ]; then
        echo "Found existing toy data for $METHOD ($FILE). Skipping generation..."
    else
        echo "Running $METHOD toy generation..."
        python ../python/global_lee_production.py \
            --trigger "$TRIGGER" \
            --toys "$TOYS" \
            --method "$METHOD" \
            --cms "$CMS"
    fi
done

echo "---------------------------------------------------------"
echo "Plotting Test Statistics..."
python ../python/plot_test_statistics.py --trigger "$TRIGGER"

echo "Pipeline complete!"
