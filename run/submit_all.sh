#!/bin/bash

# Default values if no arguments are provided
TRIGGER=${1:-"t2"}
TOTAL_TOYS=${2:-100000}
TOYS_PER_JOB=${3:-2000}

# Calculate how many jobs are needed per method
N_JOBS=$((TOTAL_TOYS / TOYS_PER_JOB))

# Ensure local directories exist before submission
mkdir -p run/logs results

echo "Directories verified. Commencing submission..."

echo "================================================="
echo " Trigger:          $TRIGGER"
echo " Total Toys/Method: $TOTAL_TOYS"
echo " Toys per Job:     $TOYS_PER_JOB"
echo " Jobs per Method:  $N_JOBS"
echo "================================================="

# Loop through all three methods and submit them
# for METHOD in naive linear copula; do
for METHOD in poisson_event naive; do
# for METHOD in poisson_event exclusive_categories; do
# for METHOD in copula; do
    echo "Submitting $METHOD..."
    condor_submit run/submit_toys.sub \
        trigger=$TRIGGER \
        method=$METHOD \
        toys_per_job=$TOYS_PER_JOB \
        n_jobs=$N_JOBS
done

echo "================================================="
echo "All jobs submitted! Type 'condor_q' to check status."
