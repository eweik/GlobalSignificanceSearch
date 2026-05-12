#!/bin/bash

# Check if a trigger argument was provided
if [ -z "$1" ]; then
    echo "Error: No trigger provided."
    echo "Usage: ./run_all_2d_scatters.sh <trigger_name>"
    echo "Example: ./run_all_2d_scatters.sh t2"
    exit 1
fi

TRIGGER=$1
echo "=========================================================="
echo "Starting 2D Copula Scatter Plot Generation for Trigger: ${TRIGGER^^}"
echo "=========================================================="

# Define the 9 mass channels
CHANNELS=("jj" "bb" "jb" "je" "jm" "jg" "be" "bm" "bg")
LEN=${#CHANNELS[@]}

# Loop over all unique pairs
for (( i=0; i<$LEN; i++ )); do
    for (( j=i+1; j<$LEN; j++ )); do
        CH1=${CHANNELS[$i]}
        CH2=${CHANNELS[$j]}
        
        echo "--> Generating plot for pair: ${CH1^^} vs ${CH2^^} ..."
        
        # Execute the python script
        python3 extra/plot_2d_copula_scattering.py --trigger "$TRIGGER" --ch1 "$CH1" --ch2 "$CH2"
        
        # Optional: Check if the script failed for this pair
        if [ $? -ne 0 ]; then
            echo "    [!] Warning: Failed to generate plot for ${CH1^^} vs ${CH2^^}."
        fi
    done
done

echo "=========================================================="
echo "Done! All 2D scatter plots for trigger ${TRIGGER^^} are saved in the 'plots/' directory."
