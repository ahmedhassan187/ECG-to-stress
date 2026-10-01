# ECG-to-Stress Analysis

**Investigating the effect of ECG recording duration on HRV reliability and stress classification.**

This is the official code repository for the paper **"Investigating the Effect of ECG Recording Duration on HRV Reliability and Stress Classification"**, presented by the **WiMoB** group at the **AIMEH** conference.

The project provides a complete pipeline for loading raw ECG recordings from the [WESAD](https://archive.ics.uci.edu/dataset/421/wesad+wearable+stress+and+affect+detection) (Wearable Stress and Affect Detection) dataset, extracting HRV (Heart Rate Variability) features at multiple recording/window durations, evaluating the statistical reliability of those features across durations (Pearson **r**, **ICC**, **MAE**), and training/evaluating machine-learning classifiers to detect stress in both binary and three-class settings.

The core research question: **do short ECG recordings (e.g., 30 s) provide HRV features and stress-classification performance comparable to conventional longer recordings (120 s / 300 s)?**

---

## 📋 Overview

This repository provides tools to:

- **Load and process** WESAD ECG signals from raw pickle files
- **Extract** 8 HRV features (`mean_rr`, `mean_hr`, `sdnn`, `rmssd`, `pnn50`, `lf_power`, `hf_power`, `lf_hf_ratio`)
- **Visualize** full ECG signals with label-aware coloring at adjustable chunk sizes
- **Analyze feature reliability across durations** — cross-duration comparisons using Pearson `r`, ICC(2,1) and MAE, including pairwise duration heatmaps, per-feature bar charts, and styled summary tables
- **Run FFT frequency analysis** — mean spectra per class, overlays across durations, and cosine-similarity comparisons (cross- vs. within-class)
- **Train and evaluate** 7 ML models with cross-validation for **binary** and **3-class** stress classification
- **Save trained models** (`.pkl` + metadata) and **make predictions** on new data: the Pavia HRV dataset, custom CSV files, or held-out WESAD subjects

The goal is to explore how well standard HRV metrics — computed from chest-worn ECG — can discriminate between psychological conditions (baseline, stress, amusement, meditation) in a controlled lab setting, and which window durations and models yield the best performance.

---

## 📖 Publication

> **Investigating the Effect of ECG Recording Duration on HRV Reliability and Stress Classification**
>
> Presented at the **AIMEH** conference — **WiMoB** group.
>
> Authors: Ahmed H. Aly (University of Pisa), Vittorio Meini (CNR, Pisa), Lucia Billeci (CNR, Pisa)

Key findings from the paper:

- Reducing the recording duration generally **decreases HRV feature agreement** (r / ICC) with longer recordings, most notably for SDNN.
- **Mean HR and Mean RR remain relatively stable** across durations (e.g., 30 s vs. 120 s / 300 s).
- Despite reduced feature agreement, HRV features from **30 s ECG recordings achieve classification performance comparable to (and in the binary task better than) longer recordings**.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`, `seaborn`, `neurokit2`, `pingouin`, `joblib`
- Optional: `xgboost`

### Installation

```bash
# Navigate to the project root
cd ECG-to-stress

# Install core dependencies
pip install numpy pandas scikit-learn matplotlib scipy seaborn neurokit2 pingouin joblib

# Optional: XGBoost
pip install xgboost
```

### Run Your First Analysis

```bash
# See all available commands
python src/main.py --help

# Correlation / cross-duration reliability analysis (default dataset path: data/WESAD)
python src/main.py -c

# Visualize ECG signals
python src/main.py -f

# Train ML models (binary classification by default)
python src/main.py -m

# Run FFT frequency analysis
python src/main.py --fft

# Make predictions with trained models (default: WESAD subject 0)
python src/main.py --predict
```

---

## 🎯 CLI Commands

The CLI (`src/main.py`) provides **five** main commands:

| Command | Alias | Purpose |
|---------|-------|---------|
| **Correlation Analysis** | `-c` / `--corr` | HRV feature reliability: cross-duration Pearson `r`, ICC, MAE + visualizations |
| **Full Signal Visualization** | `-f` / `--full` | Plot ECG signals with adjustable detail level |
| **Machine Learning Training** | `-m` / `--ml` | Train 7 models with cross-validation; save models for later prediction |
| **FFT Frequency Analysis** | `--fft` | FFT spectra per class + cosine-similarity comparison across durations |
| **Prediction Mode** | `--predict` | Use saved models to predict stress on new data (Pavia / CSV / WESAD) |

### Common Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i`, `--input` | Path to the WESAD dataset directory | `data/WESAD` |
| `-d`, `--dataset` | Dataset durations in seconds | `30 120 300` |
| `-l`, `--labels` | Label scheme: `binary` or `3class` | `binary` |
| `-o`, `--output` | Custom output directory | Varies by command |

The dataset path defaults to `data/WESAD` relative to the project root. Use `-i` to specify an alternative location:

```bash
python src/main.py -i /path/to/WESAD -c
python src/main.py -i ./my_dataset -m
```

---

### 1. Correlation Analysis (`-c` / `--corr`)

Extracts HRV features at each window duration and compares them **across durations** (e.g., 30 s vs. 120 s, 30 s vs. 300 s) using Pearson correlation (`r`), Intraclass Correlation Coefficient (ICC 2,1) and Mean Absolute Error (MAE).

```bash
python src/main.py -c                                        # All features & durations
python src/main.py -c --features mean_rr mean_hr -d 30      # Specific features / durations
python src/main.py -c -l 3class                              # Use 3-class label scheme
python src/main.py -c --by-condition                         # Split metrics by stress / non-stress
python src/main.py -i /path/to/WESAD -c                      # Custom dataset path
```

**Options:** `--features` (space-separated list or `all`), `--by-condition` (compute metrics for each condition group separately).

**Outputs** → `../results/correlation_figures/`:

- `cross_duration_comparison.csv` and per-metric tables (`comparison_table_r.csv`, `..._icc.csv`, `..._mae.csv`)
- Pairwise heatmaps (`comparison_r_heatmap.png`, ...), per-comparison and per-feature bar charts
- Styled metric summary tables and feature bar charts (R / ICC / MAE)

---

### 2. Full Signal Visualization (`-f` / `--full`)

Plots ECG signals with label-aware coloring and an adjustable chunk size (points per plot).

```bash
python src/main.py -f                                        # Default 5000 points / subject
python src/main.py -f -p 10000                               # Larger chunks
python src/main.py -f -s 0 1 2                               # Specific subjects
```

**Options:** `-p` / `--points` (points per plot, default `5000`), `-s` / `--subjects` (default: all).

**Outputs** → `../results/signal_plots/` — PNG files showing ECG waveforms colored by label.

---

### 3. Machine Learning Training (`-m` / `--ml`)

Trains 7 classifiers with stratified k-fold cross-validation (default 5 folds) on ECG chunks. Trained models and their metadata are saved as `.pkl` files so they can be reused by the prediction mode.

```bash
python src/main.py -m                                        # All models, all durations
python src/main.py -m -d 30                                  # Specific duration
python src/main.py -m -mo knn svm random_forest              # Specific models
python src/main.py -m -cv 10                                 # 10-fold cross-validation
python src/main.py -m -l 3class                              # 3-class stress classification
```

**Options:** `-mo` / `--models`, `-cv` / `--cross-val` (default `5`).

**Outputs** → `../results/ml_results/`:

- `ml_results_<duration>s_<mode>.csv` — per-model accuracy, F1, precision, recall, AUC (mean ± std)
- `saved_models/` — `<model>_<duration>s_<mode>.pkl` models + `.meta.pkl` metadata (features, label mapping, duration)

---

### 4. FFT Frequency Analysis (`--fft`)

Computes the FFT of every ECG chunk, plots mean spectra per class (with LF/HF bands highlighted), overlays all durations, and compares spectra using **cosine similarity** — both cross-class and within-class.

```bash
python src/main.py --fft                                     # All durations (30/120/300 s)
python src/main.py --fft -d 30 120                           # Specific durations
python src/main.py --fft --fft-max-pairs 1000                # More cosine-similarity pairs
python src/main.py --fft -l 3class                           # 3-class FFT analysis
```

**Options:** `--fft-max-pairs` (default `500`), `--fft-freq-max` (default `40.0` Hz), plus common options.

**Outputs** → `../results/fft_analysis/`:

- `fft_spectra_all_durations.png`, `fft_overlay_durations.png`
- `fft_cosine_similarity_distributions.png` (KDE grids per duration × comparison)
- `fft_cosine_summary.csv` and `fft_cosine_summary_bar.png`

### 5. Prediction Mode (`--predict`)

Loads saved models and predicts stress labels on new data. Supports **three input modes**:

| Mode | Flag | Description |
|------|------|-------------|
| **WESAD dataset** (default) | *(none)* | Uses subject 0 of the WESAD dataset as test data |
| **Pavia HRV** | `--pavia` | Loads `pavia_features.csv` / `pavia_labels.csv` from `data/` (or a custom folder) |
| **Custom CSV** | `--test-data` / `--test-labels` | CSV file(s) with (optionally) a `label` column |

```bash
python src/main.py --predict                                 # WESAD subject 0 + 30s models (default)
python src/main.py --predict -d 30                           # Use 30 s models
python src/main.py --predict --pavia                         # Pavia data in default folder
python src/main.py --predict --pavia /custom/path            # Pavia data in custom folder
python src/main.py --predict --test-data t.csv --test-labels t_labels.csv
python src/main.py --predict --model-dir results/ml_results/saved_models
```

**Options:** `--model-dir` (default `../results/ml_results/saved_models`), `--test-data`, `--test-labels`, `--pavia`.

**Outputs** → `../results/predictions/`:

- `predictions_<duration>s.csv` — features + true labels + per-model predictions
- `prediction_metrics_<duration>s.png`, `prediction_comparison_<duration>s.png`

---

## 🧠 Label Schemes

WESAD raw labels (1–4) are mapped to target labels using `src/label_config.py`:

| Raw label | Condition |
|-----------|-----------|
| 1 | Baseline |
| 2 | Stress |
| 3 | Amusement |
| 4 | Meditation |

### Binary (`-l binary`, default)

| Raw label | Target | Name |
|-----------|:------:|------|
| 1 | 0 | No Stress |
| 2 | 1 | Stress |
| 3 | 0 | No Stress |
| 4 | 0 | No Stress |

### Three-class (`-l 3class`)

| Raw label | Target | Name |
|-----------|:------:|------|
| 1 | 0 | No Stress / Amusement |
| 2 | 1 | Stress |
| 3 | 0 | No Stress / Amusement |
| 4 | 2 | Meditation |

---

## 🧬 Features, Models & Durations

- **8 HRV Features**: `mean_rr`, `mean_hr`, `sdnn`, `rmssd`, `pnn50`, `lf_power`, `hf_power`, `lf_hf_ratio`
- **3 Recording Durations**: 30 s, 120 s, 300 s
- **7 ML Models**: KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, Logistic Regression, XGBoost
- **Cross-Validation**: Stratified k-fold (default 5) with per-fold metrics; LOSO (leave-one-subject-out) evaluation support
- **Statistical Agreement Metrics**: Pearson `r`, ICC(2,1), MAE

---

## 📁 Project Structure

```
ECG-to-stress/
├── src/
│   ├── main.py                  # CLI entry point (5 commands)
│   ├── data.py                  # WESAD + Pavia data loading & processing
│   ├── features.py              # HRV feature extraction + FFT computation
│   ├── visualization.py         # Plotting and visualization
│   ├── correlation.py           # Pearson r, ICC, MAE analysis
│   ├── ml.py                    # Machine learning evaluation (k-fold / LOSO)
│   ├── label_config.py          # Label schemes (binary / 3-class)
│   └── figures/
│       ├── bar_plot_accuracy_f1.py   # Accuracy / F1 / AUC bar plot from ML results
│       └── ml_summary_table.py       # Styled summary table of best model per duration
│
├── notebooks/
│   ├── 01_explore_the_dataset.ipynb
│   ├── 02_feature_visualization.ipynb
│   ├── 03_full_signal_view.ipynb
│   ├── 04_features_correlation.ipynb
│   ├── 05_ml_models.ipynb
│   ├── 06_FFT_analysis.ipynb
│   └── 07_pavia_predict.ipynb
│
├── data/                        # Datasets (not fully tracked in git)
│   ├── WESAD/                   # WESAD dataset (pickle files per subject)
│   ├── HRV_Pavia_per_soggetto.xlsx
│   ├── pavia_features.csv
│   └── pavia_labels.csv
│
├── results/                     # Generated results (plots, CSVs)
│   ├── correlation_figures/
│   ├── signal_plots/
│   ├── ml_results/
│   ├── fft_analysis/
│   └── predictions/
│
├── README.md                    # This file
├── CLI_README.md                # Comprehensive CLI usage guide
├── CLI_QUICK_REFERENCE.md       # Quick command lookup
├── CLI_OVERVIEW.md              # CLI overview and walkthrough
├── CLI_COMMAND_STRUCTURE.md     # Visual command structure diagrams
├── CLI_IMPLEMENTATION_SUMMARY.md # Implementation details summary
└── .gitignore
```

---

## 📚 Documentation

| File | Description |
|------|-------------|
| [CLI_OVERVIEW.md](CLI_OVERVIEW.md) | **Start here** — Quick overview and getting started guide |
| [CLI_README.md](CLI_README.md) | Comprehensive CLI usage guide with detailed examples |
| [CLI_QUICK_REFERENCE.md](CLI_QUICK_REFERENCE.md) | Fast command lookup and cheat sheet |
| [CLI_COMMAND_STRUCTURE.md](CLI_COMMAND_STRUCTURE.md) | Visual diagrams of command hierarchy and data flow |
| [CLI_IMPLEMENTATION_SUMMARY.md](CLI_IMPLEMENTATION_SUMMARY.md) | Summary of CLI features and design decisions |

### Notebooks

- `01_explore_the_dataset.ipynb` — Dataset exploration
- `02_feature_visualization.ipynb` — Feature extraction & plots
- `03_full_signal_view.ipynb` — Full signal viewing with labels
- `04_features_correlation.ipynb` — Feature correlation analysis
- `05_ml_models.ipynb` — Machine learning model training
- `06_FFT_analysis.ipynb` — FFT spectra + cosine-similarity analysis
- `07_pavia_predict.ipynb` — Build the Pavia HRV dataset from the Excel file and export CSVs

---

## ☕ Datasets

### WESAD (training / main analysis)

Public dataset used for the paper experiments. ECG chest signals are sampled at **700 Hz** for 15 subjects across four conditions (baseline, stress, amusement, meditation). Download from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/421/wesad+wearable+stress+and+affect+detection) and place the subject folders under `data/WESAD/` (each subject folder contains a `.pkl` file).

### Pavia (validation / prediction)

An internal HRV dataset (`data/HRV_Pavia_per_soggetto.xlsx`) exported to `data/pavia_features.csv` + `data/pavia_labels.csv` by `notebooks/07_pavia_predict.ipynb`. Used to validate the trained WESAD models via the `--predict` command.

---

## 🤝 Contributing

This is a research project for Master's thesis work (WiMoB group). Contributions and suggestions are welcome.

## 📄 License

This project is for academic and research purposes.