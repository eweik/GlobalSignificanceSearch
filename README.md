# GlobalSignificanceSearch: Multi-Channel Global LEE Framework

This repository provides a high-performance statistical framework for evaluating global significance and the Look-Elsewhere Effect (LEE) in multi-channel invariant mass searches. 
Developed for the Model Independent searches, it implements and compares three distinct methods for modeling channel correlations.

## Statistical Methods

This framework handles the penalization of p-values when searching across multiple mass spectra (e.g., $M_{jj}, $M_{jb}$, $M_{bb}$) by modeling how these channels share events and fluctuate together.

1. **Naive (Independent):** Assumes zero correlation between channels. This is the most conservative approach, resulting in the highest trial factor penalization.
2. **Linear (Hub-and-Spoke):** Uses row-normalized overlap matrices to lock the fluctuations of exclusive channels to the inclusive $M_{jj}$ "hub." This accounts for the fact that many events in sub-channels are subsets of the dijet stream.
3. **Empirical Copula (Migrated):** The most sophisticated method. It uses event-by-event rank dependencies to preserve exact kinematic correlations. It accurately models mass migration (e.g., energy loss in b-jets or detector resolution effects) by propagating fluctuations through an empirical CDF transformation.

## Project Structure

* **data/**: Pre-computed empirical copula matrices (*.npz)
* **fits/**: JSON parameters for nominal and alternative fits
* **results/**: Output directory for LEE toys (*.npy) and plots (*.png)
* **run/**: Bash scripts (Entry points for the pipeline, cluster submission, and merging)
* **src/**: Core Python package (reusable logic)
* **python/**: Execution scripts (Global LEE and visualization)

## Implementation Note: Fast BumpHunter

The `fast_bumphunter_stat` utility included in `src/stats.py` is a highly optimized version of the standard BumpHunter algorithm. While it strips away side-band normalization and complex object-state management to achieve massive speed gains (up to 100x) for toy production, it is **mathematically exact** for the local p-value calculation. It utilizes `numpy` convolutions and the Poisson survival function to yield results identical to the full `pyBumpHunter` suite.

## Installation

Ensure you have Python 3.8+ and the standard scientific stack (`numpy`, `scipy`, `matplotlib`) installed. 

```bash
git clone <repo-url>
cd GlobalSignificanceSearch
chmod +x run/*.sh
```

## Usage

The `run/` directory contains all necessary bash scripts to execute the workflow locally or on the HTCondor cluster.
But run code from the `GlobalSignificanceSearch` directory, not the `run` directory.

### 1. Extracting Empirical Copula
Before generating toys using the Copula method, the rank-order matrix must be extracted from the ATLAS ROOT ntuple.

```bash
./run/run_extract_copular.sh /path/to/input/root/file /path/to/output/file/
```
* **Process:** This parses the events and maps their percentile ranks across all invariant mass definitions, saving the output to `data/copula_t1.npz`.

### 2. Global Significance (LEE) Toys (Local Run)
To calculate the global test statistic distribution and account for the Look-Elsewhere Effect (LEE) locally:

```bash
./run/run_all_toys.sh --trigger t1 --toys 100 --cms 13600.0
```
* **Process:** Generates local test statistic distributions using the Naive, Linear, and Copula methods.
* **Storage:** Data is stored in the `results/` directory as `.npy` files.

### 3. Visualizing Background Pseudo-Experiments (Method Comparison)
To generate a single synchronized pseudo-experiment across all 9 mass channels and visualize how the Naive, Linear, and Copula methods physically model background fluctuations:

```bash
# Format: ./run/run_comparison_plot.sh <trigger> [--hunt]
./run/run_comparison_plot.sh t1
```
* **Process:** Overlays the Poisson toys from all three methods on top of the analytic background fits. Also calculates and displays the local BumpHunter Z-score for each toy.
* **Hunt Mode:** Append `--hunt` to continuously generate random universes until the Copula method yields a $>5\sigma$ anomaly. This is highly useful for visually diagnosing boundary effects and clumping artifacts in low-statistics triggers.
* **Output:** Saves a 9-panel grid to `plots/method_comparison_<trigger>.png`

### 4. Signal Injection Visualization
To visually verify the physical "migration" of a Gaussian signal peak from the inclusive hub into exclusive sub-channels via the empirical copula:

```bash
./run/run_injection.sh --trigger t1 --mass 2000 --width 80 --events 5000
```
* **Process:** Injects a hypothetical signal into the $M_{jj}$ distribution and maps the corresponding events into $M_{jb}$, $M_{bb}$, etc.
* **Output:** Plots showing the source spike and the resulting migrated peaks are saved to the `results/` directory as `.png` files.

To see all 9 mass channels in a panel-grid figure:
```
# Example: Injecting a 2000 GeV signal into the M_jj distribution
./run/run_plot_9panel_grid.sh --trigger t1 --mass 2000 --width 80 --events 5000
```

### 5. HTCondor Mass Production
To generate the massive toy datasets required for discovery-level significance, submit the jobs to the HTCondor cluster. The submission scripts rely on `submit_toys.sub` for the job requirements and `condor_wrapper.sh` to set up the LCG environment on the worker nodes.

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

### 6. Merging Cluster Results
Once all HTCondor jobs have finished, merge the thousands of individual `.npy` outputs into single arrays for the final global significance calculation.

```bash
./run/merge_all.sh
```
* **Process:** Scans the `results/` directory, concatenates all valid arrays, and outputs the final merged files (e.g., `final_t1_linear.npy`).


### 7. Visualizing the Survival Curve (Global vs. Local Z)
Once all trigger results are merged, generate the final Trigger-Level Global Significance plot to compare the Look-Elsewhere Effect (LEE) penalty across methods.
```bash
python python/local_to_global_z.py --trigger "t1"
```
* **Process** This script loads the merged `.npy` files from the input trigger, calculates the p-values, and maps them to global Z-scores using the normal survival function.
* **Output** Generates the master "Survival Curve" plot (e.g., `plots/Local_vs_Global_{trigger}.png`), visually demonstrating how the Linear method safely recovers global significance compared to the Naive baseline.

To see the Analysis-Level Global Significance plot:
```bash
python python/experiment_global_lee.py 
```
* **Process** This script loads the merged `.npy` files across all triggers, calculates the global p-values, and maps them to global Z-scores using the normal survival function.
* **Output** Generates the master "Survival Curve" plot (e.g., `plots/Experiment_Wide_Global_Z.png`), visually demonstrating how the Linear method safely recovers global significance compared to the Naive baseline.


