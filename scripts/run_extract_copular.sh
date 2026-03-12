#!/bin/bash

ROOT_DIR="root"

echo "=========================================================="
echo "Starting Copula Extraction for All Triggers"
echo "=========================================================="

for i in {1..7}; do
    # INPUT_FILE="${ROOT_DIR}/data1percent_t${i}_HAE_RUN23_nominal_10PB.root"
    INPUT_FILE="data1percent_t${i}_HAE_RUN23_nominal_10PB.root"
    OUTPUT_FILE="copula_t${i}.npz"

    if [ -f "$INPUT_FILE" ]; then
        echo "--> Extracting Copula for T${i}..."
        python extract_copula.py "$INPUT_FILE" "$OUTPUT_FILE"
    else
        echo "--> Warning: $INPUT_FILE not found. Skipping T${i}."
    fi
done

echo "=========================================================="
echo "Done! Check for copula_t*.npz files."
