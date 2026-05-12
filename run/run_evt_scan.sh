#!/bin/bash

# ==========================================
# Configuration
# ==========================================
INPUT_FILE="results/merged_5param/final_t2_copula.npy"
PYTHON_SCRIPT="python/event_extrapolate.py" 
EVT_THRESHOLD=8.5                 # <-- UPDATE THIS based on your parameter stability plots!

# Local significance benchmarks (t = -ln(p_local))
T_OBS_VALUES=(6.61 10.36 15.06 20.74 27.38)
LABELS=("3-sigma" "4-sigma" "5-sigma" "6-sigma" "7-sigma")

# Check if the python script and input file exist
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found."
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    exit 1
fi

echo "=================================================="
echo " Running EVT Extrapolation Scan (LEE Calibration)"
echo " Input File        : $INPUT_FILE"
echo " EVT Threshold (u) : $EVT_THRESHOLD"
echo "=================================================="

# Loop through the benchmark values
for i in "${!T_OBS_VALUES[@]}"; do
    t_obs="${T_OBS_VALUES[$i]}"
    label="${LABELS[$i]}"

    echo ""
    echo "--> Testing Local $label (t_obs = $t_obs)"
    
    # Run the python script and filter the output for a clean summary
    python3 "$PYTHON_SCRIPT" \
        --input "$INPUT_FILE" \
        --threshold "$EVT_THRESHOLD" \
        --obs "$t_obs" | grep -E "^(Fitted Shape|Global p-value|Global Significance)"
done

echo ""
echo "=================================================="
echo " Scan complete."
