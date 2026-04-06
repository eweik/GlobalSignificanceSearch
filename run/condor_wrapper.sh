#!/bin/bash
TRIGGER=$1
METHOD=$2
TOYS=$3
JOBID=$4

echo "Starting job $JOBID on $(hostname)"

# Load ROOT/Python environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

# Move into Condor's temporary worker directory if it exists
if [ -n "$_CONDOR_SCRATCH_DIR" ]; then
    cd "$_CONDOR_SCRATCH_DIR" || exit 1
fi

TARGET_SCRIPT="run_toys.py"
EXTRA_ARGS="-b"
echo "Mode: 5-Parameter Minuit Fit ($TARGET_SCRIPT)"

# Find where the target script actually ended up
if [ -f "python/$TARGET_SCRIPT" ]; then
    PY_PATH="python/$TARGET_SCRIPT"
elif [ -f "$TARGET_SCRIPT" ]; then
    PY_PATH="$TARGET_SCRIPT"
else
    echo "ERROR: $TARGET_SCRIPT not found in root or python/ directory!"
    ls -R
    exit 1
fi

# ==========================================
# STRICT SINGLE-THREADING ENFORCEMENT
# Prevents HTCondor from throttling the job
# ==========================================
# export OMP_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export VECLIB_MAXIMUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
mkdir -p run/logs results

# Run the python script using the dynamically detected PY_PATH and arguments
python3 "$PY_PATH" \
    --trigger "$TRIGGER" \
    --method "$METHOD" \
    --toys "$TOYS" \
    --jobid "$JOBID" \
    $EXTRA_ARGS

echo "Job $JOBID finished."
