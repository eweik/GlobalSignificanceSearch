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
    cd $_CONDOR_SCRATCH_DIR
fi

# Find where run_toys.py actually ended up
if [ -f "python/run_toys.py" ]; then
    PY_PATH="python/run_toys.py"
elif [ -f "run_toys.py" ]; then
    PY_PATH="run_toys.py"
else
    echo "ERROR: run_toys.py not found in root or python/ directory!"
    ls -R
    exit 1
fi

# Run the python script (pointing to the python/ folder)
python3 python/run_toys.py \
    --trigger "$TRIGGER" \
    --method "$METHOD" \
    --toys "$TOYS" \
    --fit \
    --chimax 2.0 \
    -b \
    --jobid "$JOBID"

echo "Job $JOBID finished."
