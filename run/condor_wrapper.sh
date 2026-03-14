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
