# GlobalSignificanceSearch: Multi-Channel Global LEE Framework

This repository provides a high-performance statistical framework for evaluating global significance and the Look-Elsewhere Effect (LEE) in multi-channel invariant mass searches. 
Developed for the Model Independent searches, it implements and compares three distinct methods for modeling channel correlations.

## Statistical Methods

This framework handles the penalization of p-values when searching across multiple mass spectra (e.g., $M_{jj}, $M_{jb}$, $M_{bb}$) by modeling how these channels share events and fluctuate together.

1. **Naive (Independent Poisson):** Assumes zero correlation between channels. This is the most conservative approach, resulting in the highest trial factor penalization.
2. **Linear Overlap:** Uses row-normalized overlap matrices to lock the fluctuations of exclusive channels to the inclusive $M_{jj}$ "hub." This accounts for the fact that many events in sub-channels are subsets of the dijet stream.
3. **Empirical Copula:** The most sophisticated method. It uses event-by-event rank dependencies to preserve exact kinematic correlations. It accurately models mass migration (e.g., energy loss in b-jets or detector resolution effects) by propagating fluctuations through an empirical CDF transformation.
4. **Poisson (Event-Weight) Bootstrap:**  Uses the data itself to generate pseudo-experiments by assigning each event a weight drawn from a Poisson distribution with $\lambda=1$. This method naturally preserves the exact kinematic correlations, overlaps, and physical boundary limits of the actual data without requiring complex analytical modeling.


## Project Structure

### Core Directories
* **data/**: Pre-computed empirical copula matrices (*.npz)
* **fits/**: JSON parameters for nominal and alternative fits
* **results/**: Output directory for LEE toys (*.npy) 
* **run/**: Bash scripts (Entry points for the pipeline, cluster submission, and merging)
* **src/**: Core Python package (reusable logic)

### **`python/`** (Execution Scripts)
Contains the primary Python scripts for data processing and pseudo-experiment generation.
* `extract_copula.py`: Parses ROOT ntuples to extract rank-order matrices and saves empirical copula data.
* `extract_masses.py`: Extracts invariant mass distributions for all defined channels.
* `run_toys.py`: Main script for generating the multi-channel pseudo-experiments.
* `run_single_mass_toys.py`: Generates toys restricted to a single mass definition.
* `merge_results.py`: Combines individual `.npy` outputs from cluster jobs into unified arrays.

### **`run/`** (Shell Wrappers & Cluster Submission)
Bash scripts serving as entry points for local execution and HTCondor cluster submission. 
Always run these from the `GlobalSignificanceSearch` directory.
* **HTCondor Submission**
    * `submit_all.sh` / `submit_all_triggers.sh`: Scripts to batch-submit toy generation jobs.
    * `submit_toys.sub`: HTCondor job configuration requirements.
    * `condor_wrapper.sh`: Sets up the LCG environment on worker nodes before execution.
* **Local Wrappers**: `run_all_toys.sh`, `run_comparison_plot.sh`, `run_extract_copular.sh`, `run_extract_masses.sh`, `run_injection.sh`, `run_plot_9panel_grid.sh`, `run_plot_diagnostics.sh`, `run_absorption_scan.sh`


### **`extra/`** (Visualizations & Diagnostics)
A comprehensive suite of scripts dedicated strictly to plotting results, generating grids, and visualizing correlation metrics.

## Implementation Note: Fast BumpHunter

The `fast_bumphunter_stat` utility included in `src/stats.py` is a highly optimized version of the standard BumpHunter algorithm. While it strips away side-band normalization and complex object-state management to achieve massive speed gains (up to 100x) for toy production, it is **mathematically exact** for the local p-value calculation. It utilizes `numpy` convolutions and the Poisson survival function to yield results identical to the full `pyBumpHunter` suite.

## Installation

Ensure you have Python 3.8+ and the standard scientific stack (`numpy`, `scipy`, `matplotlib`) installed (with lsetup)

```bash
git clone <repo-url>
cd GlobalSignificanceSearch
lsetup "views LCG_106 x86_64-el9-gcc13-opt"   # on lxplus for libraries
chmod +x run/*.sh
```

## Usage

The `run/` directory contains all necessary bash scripts to execute the workflow locally or on the HTCondor cluster.
*Important: Run all scripts from the main `GlobalSignificanceSearch` directory, not from within the `run` directory.*

### 1. Extracting Empirical Copula and Masses
Before generating toys using the Copula method, the rank-order matrix must be extracted from the ATLAS ROOT ntuple.
For AD Analysis, ROOT ntuples must be created with `analysis_root_chunky.py` with `save_tree=True` enabled. 

```bash
./run/run_extract_copular.sh /path/to/input/root/file /path/to/output/file/
./run/run_extract_masses.sh /path/to/input/root/file /path/to/output/file/

```
* **Process:** This parses the events and maps their percentile ranks across all invariant mass definitions, saving the output to `data/copula_t1.npz`, for Copula toys.


### 2. Global Significance (LEE) Toys (Local Run)
To calculate the global test statistic distribution and account for the Look-Elsewhere Effect (LEE) locally:

```bash
./run/run_all_toys.sh --trigger t1 --toys 100 --cms 13600.0
```
* **Process:** Generates local test statistic distributions using the Naive, Linear, Copula, and Poisson-Bootstrap methods.
* **Storage:** Data (BumpHunter test statistic) is stored in the `results/` directory as `.npy` files.


### 3. HTCondor Mass Production
To generate the massive toy datasets required for discovery-level significance, submit the jobs to the HTCondor cluster. The submission scripts rely on `submit_toys.sub` for the job requirements and `condor_wrapper.sh` to set up the LCG environment on the worker nodes. 
Saves BumpHunter test statistic for each pseudo-experiment into `.npy` files.

**To submit jobs for a single trigger:**
```bash
# Format: ./run/submit_all.sh <trigger> <total_toys> <toys_per_job>
./run/submit_all.sh t1 100000 1000
```

**To submit jobs for ALL triggers simultaneously:**
```bash
# Format: ./run/submit_all_triggers.sh <total_toys> <toys_per_job>
./run/submit_all_triggers.sh 100000 1000
```


### 4. Merging Cluster Results
Once all HTCondor jobs have finished, merge the thousands of individual `.npy` outputs into single arrays for the final global significance calculation.

```bash
./run/merge_all.sh
```
* **Process:** Scans the `results/` directory, concatenates all valid arrays, and outputs the final merged files (e.g., `final_t1_linear.npy`).


### 5. Visualizing the Significance Curve (Global vs. Local Z)
Once all trigger results are merged, generate the final Trigger-Level Global Significance plot to compare the Look-Elsewhere Effect (LEE) penalty across methods.
```bash
python python/localZ_to_global_z.py --trigger <t2>  # plot Trigger Level Significance Curve
```
* **Process** This script loads the merged `.npy` files from the input trigger, calculates the p-values, and maps them to global Z-scores using the normal survival function.
* **Output** Generates the master "Survival Curve" plot (e.g., `plots/Local_vs_Global_{trigger}.png`), visually demonstrating how the Linear method safely recovers global significance compared to the Naive baseline.

To see the Analysis-Level Global Significance plot:
```bash
python python/analysis_global_lee.py 
```
* **Process** This script loads the merged `.npy` files across all triggers, calculates the global p-values, and maps them to global Z-scores using the normal survival function.
* **Output** Generates the master "Survival Curve" plot (e.g., `plots/Analysis_Wide_Global_Z.png`), visually demonstrating how the different methods recovers global significance compared to the Independent Poisson baseline.


