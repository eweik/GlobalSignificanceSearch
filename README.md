# GlobalSignificanceSearch: Multi-Channel Global LEE Framework

This repository provides a high-performance statistical framework for evaluating global significance and the Look-Elsewhere Effect (LEE) in multi-channel invariant mass searches. 
Developed for the Model Independent searches, it implements and compares three distinct methods for modeling channel correlations.

## Statistical Methods

This framework handles the penalization of p-values when searching across multiple mass spectra (e.g., $M_{jj}$, $M_{jb}$, $M_{bb}$) by modeling how these channels share events and fluctuate together.

1. **Naive (Independent):** Assumes zero correlation between channels. This is the most conservative approach, resulting in the highest trial factor penalization.
2. **Linear (Hub-and-Spoke):** Uses row-normalized overlap matrices to lock the fluctuations of exclusive channels to the inclusive $M_{jj}$ "hub." This accounts for the fact that many events in sub-channels are subsets of the dijet stream.
3. **Empirical Copula (Migrated):** The most sophisticated method. It uses event-by-event rank dependencies to preserve exact kinematic correlations. It accurately models mass migration (e.g., energy loss in b-jets or detector resolution effects) by propagating fluctuations through an empirical CDF transformation.

## Project Structure

* **data/**: Pre-computed empirical copula matrices (*.npz)
* **fits/**: JSON parameters for nominal and alternative fits
* **results/**: Output directory for LEE toys (*.npy) and plots (*.png)
* **run/**: Bash scripts (Entry points for the pipeline)
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

## Usage

### 1. Extracting Empirical Copula
Before generating toys, the rank-order matrix must be extracted from ATLAS ROOT ntuple

```bash
# Example usage (adjust arguments based on your script setup)
python python/extract_copula.py /path/to/input/root/file /path/to/output/file/
```
* This parses the outlier events and maps their percentile ranks across all invariant mass definitions, saving the output to `data/copula_t1.npz`

### 2. Global Significance (LEE) Toys
To calculate the global test statistic distribution and account for the Look-Elsewhere Effect (LEE) across all search channels simultaneously:

```bash
cd run
./run_all_toys.sh --trigger t1 --toys 10000 --cms 13600.0
```
* Process: This generates $t_{max}$ distributions using the Naive, Linear, and Copula methods.
* Storage: Data is stored in the results/ directory as .npy files.
* Output: The script automatically generates a comparison plot of the three methods once the pseudo-experiments are complete.

### 3. Signal Injection Visualization
To visually verify the physical "migration" of a Gaussian signal peak from the inclusive hub into exclusive sub-channels via the empirical copula:

```bash
cd run
./run_injection.sh --trigger t1 --mass 2000 --width 80 --events 5000
```

* Process: This injects a hypothetical signal into the $M_{jj}$ distribution and maps the corresponding events into $M_{jb}$, $M_{bb}$, etc.
* Output: Plots showing the source spike and the resulting migrated peaks are saved to the `results/` directory as `.png` files.

