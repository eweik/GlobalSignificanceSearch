#!/bin/bash
# condor_wrapper.sh

# Assign arguments to variables for readability
TRIGGER=$1
METHOD=$2
TOYS=$3
JOBID=$4

echo "Starting job $JOBID on $(hostname)"

# --- SET UP YOUR ENVIRONMENT HERE ---
# If you use an LCG release for ROOT/Python, source it. 
# (Update this path to whatever you normally source on lxplus)
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

# Navigate to the directory where the job was submitted from
cd $TMPDIR || cd $_CONDOR_SCRATCH_DIR

# Run the python script
python3 ../python/run_toys.py \
    --trigger "$TRIGGER" \
    --method "$METHOD" \
    --toys "$TOYS" \
    --fit \
    --chimax 2.0 \
    -b \
    --jobid "$JOBID"

echo "Job $JOBID finished."
