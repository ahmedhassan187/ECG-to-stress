# ECG-to-Stress Analysis CLI Usage Guide

## Overview

The `main.py` script provides a command-line interface (CLI) for ECG signal analysis, HRV feature extraction, **correlation / reliability analysis across recording durations**, FFT frequency analysis, machine-learning model training, and **prediction on new data** for the paper *"Investigating the Effect of ECG Recording Duration on HRV Reliability and Stress Classification"* (AIMEH conference, WiMoB group).

## Table of Contents

1. [General Usage](#general-usage)
2. [Common Options](#common-options)
3. [Label Schemes](#label-schemes)
4. [Correlation Analysis Command](#correlation-analysis)
5. [Full Signal Visualization Command](#full-signal-visualization)
6. [Machine Learning Training Command](#machine-learning-training)
7. [FFT Frequency Analysis Command](#fft-frequency-analysis)
8. [Prediction Mode Command](#prediction-mode)
9. [Examples](#examples)
10. [Output Structure](#output-structure)
11. [Troubleshooting](#troubleshooting)

---

## General Usage

### Get Help

```bash
# Show main help message
python src/main.py --help
python src/main.py -h
```

### Command Structure

```bash
python src/main.py <COMMAND> [OPTIONS]
```

---

## Common Options

The following options apply to all commands:

| Option | Alias | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `-i` | `--input` | string | `data/WESAD` | Path to the WESAD dataset directory |
| `-d` | `--dataset` | int+ | `30 120 300` | Dataset durations in seconds |
| `-l` | `--labels` | string | `binary` | Label scheme: `binary` or `3class` |
| `-o` | `--output` | string | varies | Custom output directory |

```bash
# Specify a custom dataset path (works with all commands)
python src/main.py -i /path/to/WESAD -c
python src/main.py -i ./my_dataset -m
```

**Note:** `-f` is reserved for the **Full Signal Visualization** command. To select HRV features in the correlation command use the long flag `--features`.

---

## Label Schemes

WESAD raw labels (1–4) are mapped to target labels via `src/label_config.py`.

| Raw label | Condition |
|-----------|-----------|
| 1 | Baseline |
| 2 | Stress |
| 3 | Amusement |
| 4 | Meditation |

### Binary (`-l binary`, default)

- 1 → 0 (No Stress), 2 → 1 (Stress), 3 → 0 (No Stress), 4 → 0 (No Stress)

### Three-class (`-l 3class`)

- 1 → 0 (No Stress / Amusement), 2 → 1 (Stress), 3 → 0 (No Stress / Amusement), 4 → 2 (Meditation)

```bash
python src/main.py -c -l 3class
python src/main.py -m -l 3class
python src/main.py --fft -l 3class
```

---

## Correlation Analysis

### Command Syntax

```bash
python src/main.py -c [OPTIONS]
python src/main.py --corr [OPTIONS]
```

### Purpose

Extracts HRV features at each window duration and computes **cross-duration reliability metrics** — Pearson correlation (`r`), Intraclass Correlation Coefficient (ICC 2,1) and Mean Absolute Error (MAE) — between the short windows (e.g., 30 s) and longer recordings (120 s / 300 s). This is the core statistical analysis of the paper.

### Options

| Option | Alias | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--features` | — | string+ | `all` | HRV features to analyze |
| `--by-condition` | — | flag | off | Also compute metrics separately for each condition group |
| `-i`, `-d`, `-l`, `-o` | | | | See [Common Options](#common-options) |

### Available Features

- `mean_rr` - Mean RR interval
- `mean_hr` - Mean heart rate
- `sdnn` - Standard deviation of NN intervals
- `rmssd` - Root mean square of successive differences
- `pnn50` - Percentage of NN50 count
- `lf_power` - Low frequency power
- `hf_power` - High frequency power
- `lf_hf_ratio` - LF/HF ratio

### Examples

```bash
# All features, all durations
python src/main.py -c

# Specific features for specific durations
python src/main.py -c --features mean_rr mean_hr sdnn -d 30 120

# Split metrics by stress / non-stress condition
python src/main.py -c --by-condition

# Custom output directory
python src/main.py -c -o ./my_results

# Custom dataset path
python src/main.py -i /path/to/WESAD -c
```

### Outputs

Written to `../results/correlation_figures/` (or `-o`):

- `cross_duration_comparison.csv` — one row per (small, large) duration pair and feature with `r`, `icc`, `mae`, `n`
- `comparison_table_r.csv`, `comparison_table_icc.csv`, `comparison_table_mae.csv` — N×N pairwise tables per metric
- Heatmaps (`comparison_r_heatmap.png`, ...), per-comparison and per-feature bar charts
- Styled metric summary tables and feature bar charts

---

## Full Signal Visualization

### Command Syntax

```bash
python src/main.py -f [OPTIONS]
python src/main.py --full [OPTIONS]
```

### Purpose

Plots ECG signals with label-aware coloring and an adjustable chunk size (points per plot).

### Options

| Option | Alias | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `-p` | `--points` | int | `5000` | Points per plot chunk |
| `-s` | `--subjects` | int+ | all | Subject IDs to plot |
| `-i`, `-o` | | | | Dataset path / output dir |

### Examples

```bash
# Default: 5000 points per chunk, all subjects
python src/main.py -f

# Custom chunk size / subjects
python src/main.py -f -p 10000 -s 0 1 2

# Custom output
python src/main.py -f -p 7500 -o ./plots
```

**Point Size Guidelines:** 2000–5000 (high detail), 5000–10000 (balanced), 10000+ (overview).

### Outputs

PNG files in `../results/signal_plots/` colored by label.

---

## Machine Learning Training

### Command Syntax

```bash
python src/main.py -m [OPTIONS]
python src/main.py --ml [OPTIONS]
```

### Purpose

Trains **7 classifiers** with stratified k-fold cross-validation (default 5 folds) on ECG chunks. Trained models and metadata are saved as `.pkl` files for later use in prediction mode.

### Options

| Option | Alias | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `-mo` | `--models` | string+ | all 7 | Models to train |
| `-cv` | `--cross-val` | int | `5` | Number of CV folds |
| `-i`, `-d`, `-l`, `-o` | | | | See [Common Options](#common-options) |

### Models

`knn`, `svm`, `decision_tree`, `random_forest`, `gradient_boosting`, `logistic_regression`, `xgboost`

### Examples

```bash
# All models, all durations
python src/main.py -m

# Specific durations / models
python src/main.py -m -d 30 120 -mo knn svm xgboost

# Custom CV folds / 3-class
python src/main.py -m -d 30 -cv 10 -l 3class
```

### Outputs

- `../results/ml_results/ml_results_<duration>s_<mode>.csv` — per-model accuracy, F1, precision, recall, AUC (mean ± std across folds)
- `../results/ml_results/saved_models/<model>_<duration>s_<mode>.pkl` and `.meta.pkl` (features, label mapping, duration, CV summary)

---

## FFT Frequency Analysis

### Command Syntax

```bash
python src/main.py --fft [OPTIONS]
```

### Purpose

Computes the FFT of every ECG chunk at each duration, plots mean spectra per class (with LF / HF bands), overlays all durations, and compares spectra using **cosine similarity** — cross-class (stress ↔ non-stress) and within-class.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--fft-max-pairs` | int | `500` | Max random pairs per comparison for cosine similarity |
| `--fft-freq-max` | float | `40.0` | Max frequency (Hz) to plot |
| `-d`, `-l`, `-i`, `-o` | | | See [Common Options](#common-options) |

### Examples

```bash
# All durations
python src/main.py --fft

# Specific durations, more pairs
python src/main.py --fft -d 30 120 --fft-max-pairs 1000

# 3-class FFT analysis
python src/main.py --fft -l 3class
```

### Outputs

- `fft_spectra_all_durations.png`, `fft_overlay_durations.png`
- `fft_cosine_similarity_distributions.png` (KDE grids per duration × comparison)
- `fft_cosine_summary.csv`, `fft_cosine_summary_bar.png`

---

## Prediction Mode

### Command Syntax

```bash
python src/main.py --predict [OPTIONS]
```

### Purpose

Loads saved models (`.pkl`) and predicts stress labels on new data. Three input modes are supported:

| Mode | Flag | Description |
|------|------|-------------|
| **WESAD dataset** (default) | *(none)* | Uses subject 0 of the WESAD dataset as test data |
| **Pavia HRV** | `--pavia` | Loads `pavia_features.csv` / `pavia_labels.csv` (from `data/` or a custom folder) |
| **Custom CSV** | `--test-data` / `--test-labels` | CSV files with (optionally) a `label` column |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model-dir` | string | `../results/ml_results/saved_models` | Directory with trained models |
| `--pavia` | string (optional) | `default` | Pavia data folder (default: `data/`) |
| `--test-data` | string | — | Test features CSV |
| `--test-labels` | string | — | Test labels CSV |
| `-d`, `-l`, `-i`, `-o` | | | See [Common Options](#common-options) |

### Examples

```bash
# Default: WESAD subject 0 as test data, 30s models (default)
python src/main.py --predict

# Specific duration
python src/main.py --predict -d 30

# Pavia HRV data (default folder)
python src/main.py --predict --pavia

# Pavia HRV data (custom folder)
python src/main.py --predict --pavia /custom/path

# Custom CSV test data
python src/main.py --predict --test-data test_features.csv --test-labels test_labels.csv

# Custom model directory
python src/main.py --predict --model-dir results/ml_results/saved_models
```

### Outputs

- `predictions_<duration>s.csv` — features + true labels + per-model prediction columns
- `prediction_metrics_<duration>s.png` and `prediction_comparison_<duration>s.png`

---

## Examples

### Complete Analysis Pipeline

```bash
# 1. Reliability / correlation analysis across durations
python src/main.py -c -d 30 120 300

# 2. Visualize the raw signals to inspect data quality
python src/main.py -f -p 5000 -s 0 1 2

# 3. Train models on 30s chunks (baseline)
python src/main.py -m -d 30

# 4. Compare performance across durations
python src/main.py -m

# 5. FFT frequency analysis
python src/main.py --fft

# 6. Validate on Pavia data with trained models
python src/main.py --predict --pavia
```

---

## Output Structure

```
results/
├── correlation_figures/
│   ├── cross_duration_comparison.csv
│   ├── comparison_table_{r|icc|mae}.csv
│   ├── comparison_{r|icc|mae}_heatmap.png
│   └── [styled tables, bar charts]
│
├── signal_plots/
│   ├── subject_00_ecg.png
│   └── ...
│
├── ml_results/
│   ├── ml_results_30s_binary.csv
│   ├── ml_results_30s_3class.csv
│   ├── saved_models/*.pkl
│   └── [comparison plots]
│
├── fft_analysis/
│   ├── fft_spectra_all_durations.png
│   ├── fft_cosine_summary.csv
│   └── ...
│
└── predictions/
    ├── predictions_30s.csv
    └── prediction_metrics_30s.png
```

---

## Performance Notes

- **30s chunks**: ~1491 samples (fastest training, most chunks)
- **120s chunks**: ~369 samples (balanced)
- **300s chunks**: ~144 samples (fewest chunks)

### Model Training Times (Approximate)

- KNN / Logistic Regression: very fast
- Decision Tree: fast
- SVM / Random Forest: medium
- Gradient Boosting / XGBoost: slow

---

## Troubleshooting

### "Module not found" errors

Ensure you're running from the project root:

```bash
cd g:\Master\Thesis\FLT\Code\ECG-to-stress
python src/main.py -m
```

### "Dataset not found" errors

Verify WESAD data structure or specify the correct path with `-i`:

```bash
# Default structure
data/WESAD/
├── S2/S2.pkl
├── S3/S3.pkl
└── ...
```

### "Model not found" during prediction

Train (and save) the models first:

```bash
python src/main.py -m -d 30 -mo knn svm
python src/main.py --predict -d 30 --pavia
```

### Out of memory errors

Reduce the number of durations or models:

```bash
python src/main.py -c -d 30
python src/main.py -m -d 300 -mo knn svm
```

---

## Additional Notes

- All commands support relative and absolute output paths
- Cross-validation is stratified to maintain class balance
- Default binary mapping: Stress (label 2) vs. No Stress (labels 1, 3, 4)
- Sampling frequency: 700 Hz (WESAD standard)
- XGBoost is optional; missing it simply skips that model