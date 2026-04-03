#!/bin/bash

ROOT_DIR="data"

echo "=========================================================="
echo "Starting Mass Extraction for All Triggers"
echo "=========================================================="

for i in {1..7}; do
    INPUT_FILE="${ROOT_DIR}/data1percent_t${i}_HAE_RUN23_nominal_10PB.root"
    # INPUT_FILE="data1percent_t${i}_HAE_RUN23_nominal_10PB.root"
    OUTPUT_FILE="${ROOT_DIR}/masses_t${i}.npz"

    if [ -f "$INPUT_FILE" ]; then
        echo "--> Extracting Masses for T${i}..."
        python python/extract_masses.py "$INPUT_FILE" "$OUTPUT_FILE"
    else
        echo "--> Warning: $INPUT_FILE not found. Skipping T${i}."
    fi
done

echo "=========================================================="
echo "Done! Check for masses_t*.npz files."
