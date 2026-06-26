# ECG-to-Stress Analysis CLI Usage Guide

## Overview

The `main.py` script provides a command-line interface (CLI) for ECG signal analysis, feature extraction, correlation analysis, and machine learning model training on the WESAD dataset.

## Table of Contents
1. [General Usage](#general-usage)
2. [Dataset Path Configuration](#dataset-path-configuration)
3. [Correlation Analysis Command](#correlation-analysis)
4. [Full Signal Visualization Command](#full-signal-visualization)
5. [Machine Learning Training Command](#machine-learning-training)
6. [Examples](#examples)

---

## General Usage

### Get Help
```bash
# Show main help message
python src/main.py --help
python src/main.py -h

# Show help for specific command
python src/main.py correlation --help
python src/main.py full --help
python src/main.py ml --help
```

### Command Structure
```bash
python src/main.py <COMMAND> [OPTIONS]
```

---

## Dataset Path Configuration

The dataset path can be specified with the `-i` / `--input` option. If not provided, it defaults to `data/WESAD` relative to the project root.

```bash
# Use default dataset path (data/WESAD)
python src/main.py -c

# Specify custom dataset path (relative)
python src/main.py -i data/WESAD -c

# Specify custom dataset path (absolute)
python src/main.py --input /absolute/path/to/WESAD -c

# The -i flag works with all commands
python src/main.py -i /data/WESAD -f
python src/main.py -i ./my_dataset -m
```

The `-i` flag is available as a common option for all three commands (correlation, visualization, ML).

---

## Correlation Analysis

### Command Syntax
```bash
python src/main.py -c [OPTIONS]
python src/main.py --corr [OPTIONS]
python src/main.py correlation [OPTIONS]
```

### Purpose
Extracts HRV (Heart Rate Variability) features from ECG signals and generates correlation analysis figures.

### Options

| Option | Alias | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `-i` | `--input` | string | `data/WESAD` | Path to the WESAD dataset directory |
| `-f` | `--features` | string+ | `all` | Features to analyze (space-separated list) |
| `-d` | `--dataset` | int+ | `30 120 300` | Dataset durations in seconds |
| `-o` | `--output` | string | `../results/correlation_figures` | Output directory for figures |

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
# Analyze ALL features for all datasets (30s, 120s, 300s)
python src/main.py -c

# Analyze specific features for all datasets
python src/main.py -c -f mean_rr mean_hr sdnn

# Analyze all features for specific dataset durations
python src/main.py -c -d 30 120

# Analyze specific features for specific durations
python src/main.py -c -f mean_rr rmssd -d 30 300

# Save results to custom output directory
python src/main.py -c -o ./my_correlation_results

# Use custom dataset path
python src/main.py -i /path/to/WESAD -c
```

---

## Full Signal Visualization

### Command Syntax
```bash
python src/main.py -f [OPTIONS]
python src/main.py --full [OPTIONS]
python src/main.py full [OPTIONS]
```

### Purpose
Plots the complete ECG signals with adjustable chunk size for visualization. Displays signal segments with label-based coloring.

### Options

| Option | Alias | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `-i` | `--input` | string | `data/WESAD` | Path to the WESAD dataset directory |
| `-p` | `--points` | int | `5000` | Number of points per plot chunk |
| `-s` | `--subjects` | int+ | None (all) | Specific subject IDs to plot |
| `-o` | `--output` | string | `../results/signal_plots` | Output directory for plots |

### Examples

```bash
# Plot all subjects with default 5000 points per chunk
python src/main.py -f

# Plot all subjects with 10000 points per chunk (larger segments)
python src/main.py -f -p 10000

# Plot all subjects with 2000 points per chunk (smaller segments)
python src/main.py --full -p 2000

# Plot only subjects 0, 1, 2 with default chunk size
python src/main.py -f -s 0 1 2

# Plot subjects 0, 3, 5 with custom chunk size
python src/main.py -f -s 0 3 5 -p 8000

# Save results to custom directory
python src/main.py -f -o ./my_signal_plots -p 7500

# Use custom dataset path
python src/main.py -i /path/to/WESAD -f
```

### Point Recommendations
- **2000-5000**: High detail, many plots per signal
- **5000-10000**: Balanced detail and overview
- **10000+**: Large segments, fewer plots

---

## Machine Learning Training

### Command Syntax
```bash
python src/main.py -m [OPTIONS]
python src/main.py --ml [OPTIONS]
python src/main.py ml [OPTIONS]
```

### Purpose
Trains machine learning models on ECG chunks using 5-fold stratified cross-validation. Supports multiple models and dataset durations.

### Options

| Option | Alias | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `-i` | `--input` | string | `data/WESAD` | Path to the WESAD dataset directory |
| `-d` | `--dataset` | int+ | `30 120 300` | Dataset durations in seconds |
| `-mo` | `--models` | string+ | (all 7) | Models to train (space-separated) |
| `-cv` | `--cross-val` | int | `5` | Number of CV folds |
| `-o` | `--output` | string | `../results/ml_results` | Output directory |

### Available Models
- `knn` - K-Nearest Neighbors
- `svm` - Support Vector Machine
- `decision_tree` - Decision Tree Classifier
- `random_forest` - Random Forest
- `gradient_boosting` - Gradient Boosting
- `logistic_regression` - Logistic Regression
- `xgboost` - XGBoost Classifier

### Examples

```bash
# Train ALL models on ALL datasets (30s, 120s, 300s)
python src/main.py -m

# Train all models on specific datasets (30s, 120s only)
python src/main.py -m -d 30 120

# Train specific models on all datasets
python src/main.py -m -mo knn svm xgboost

# Train specific models on specific datasets
python src/main.py -m -d 30 -mo random_forest gradient_boosting

# Train on 300s dataset with Random Forest and XGBoost
python src/main.py -m -d 300 -mo random_forest xgboost

# Change number of CV folds (10-fold cross-validation)
python src/main.py -m -cv 10

# Custom CV folds and models
python src/main.py -m -cv 10 -mo knn svm decision_tree

# Save results to custom directory
python src/main.py -m -o ./my_ml_results

# Use custom dataset path
python src/main.py -i /path/to/WESAD -m
```

### Classification Task
- **Binary Classification**: Stress vs. No-Stress
  - No-Stress: Labels 1 (Baseline), 3 (Meditation)
  - Stress: Labels 2 (Amusement), 4 (Stress)

### Output Files
- `cv_results_30s.csv` - Cross-validation metrics for 30s chunks
- `cv_results_120s.csv` - Cross-validation metrics for 120s chunks
- `cv_results_300s.csv` - Cross-validation metrics for 300s chunks
- `cv_summary_all_results.csv` - Comprehensive summary
- Individual model comparison plots (PNG)
- Heatmaps and line plots with error bars

---

## Examples

### Complete Analysis Pipeline

```bash
# 1. First, analyze correlations to understand feature relationships
python src/main.py -c -f mean_rr mean_hr sdnn rmssd -d 30 120 300

# 2. Visualize the raw signals to inspect data quality
python src/main.py -f -p 5000 -s 0 1 2

# 3. Train models on 30s chunks (baseline)
python src/main.py -m -d 30

# 4. Compare performance across durations
python src/main.py -m

# 5. Fine-tune best performers
python src/main.py -m -d 30 -mo random_forest gradient_boosting -cv 10
```

### Specific Use Cases

**Quick Overview:**
```bash
python src/main.py --help
python src/main.py -c
python src/main.py -f -p 5000
python src/main.py -m -d 30 -mo knn svm
```

**Detailed Analysis:**
```bash
python src/main.py -c -f mean_rr mean_hr sdnn rmssd pnn50 -d 30 120 300
python src/main.py -f -p 2500 -s 0 1 2 3 4
python src/main.py -m -cv 10 -mo random_forest gradient_boosting xgboost
```

**Full Feature Extraction:**
```bash
# Extract all features, visualize all signals, train all models
python src/main.py -c
python src/main.py -f
python src/main.py -m
```

---

## Output Structure

```
results/
├── correlation_figures/
│   ├── features_30s.csv
│   ├── features_120s.csv
│   ├── features_300s.csv
│   └── [correlation plots]
│
├── signal_plots/
│   ├── subject_00_ecg.png
│   ├── subject_01_ecg.png
│   └── ...
│
└── ml_results/
    ├── cv_results_30s.csv
    ├── cv_results_120s.csv
    ├── cv_results_300s.csv
    ├── cv_summary_all_results.csv
    └── [model comparison plots]
```

---

## Performance Notes

- **30s chunks**: ~196 samples per subject (fastest training)
- **120s chunks**: ~49 samples per subject (balanced)
- **300s chunks**: ~19 samples per subject (most data per chunk, slowest)

### Model Training Times (Approximate)
- KNN: Very fast
- Logistic Regression: Very fast
- Decision Tree: Fast
- SVM: Medium
- Random Forest: Medium
- Gradient Boosting: Slow
- XGBoost: Slow

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
# Using default path
data/WESAD/
├── S2/
│   └── S2.pkl
├── S3/
│   └── S3.pkl
└── ...

# Or specify a custom path
python src/main.py -i /path/to/your/dataset -c
```

### Out of memory errors
Reduce dataset size or use smaller chunk sizes:
```bash
python src/main.py -c -d 30  # Smaller chunks
python src/main.py -f -p 10000  # Larger visualization chunks
python src/main.py -m -d 300 -mo knn svm  # Fewer models
```

---

## Additional Notes

- All commands support relative and absolute output paths
- Default dataset includes all available subjects
- Cross-validation is stratified to maintain class balance
- Binary stress classification: {1,3}→0 (No-Stress), {2,4}→1 (Stress)
- Sampling frequency: 700 Hz (WESAD standard)