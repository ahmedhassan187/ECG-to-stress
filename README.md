# ECG-to-Stress Analysis

A comprehensive toolkit for ECG signal analysis, feature extraction, correlation analysis, and machine learning model training on the [WESAD](https://archive.ics.uci.edu/dataset/421/wesad+wearable+stress+and+affect+detection) (Wearable Stress and Affect Detection) dataset.

## 📋 Overview

This project provides tools to:
- **Load and process** WESAD ECG signals
- **Extract** HRV (Heart Rate Variability) features
- **Visualize** full ECG signals with adjustable chunk sizes
- **Analyze** feature correlations and distributions
- **Train and evaluate** machine learning models for stress classification

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`
- Optional: `xgboost`

### Installation

```bash
# Navigate to the project root
cd ECG-to-stress

# Install dependencies
pip install numpy pandas scikit-learn matplotlib scipy
pip install xgboost  # optional
```

### Run Your First Analysis

```bash
# See all available commands
python src/main.py --help

# Run correlation analysis (default dataset path: data/WESAD)
python src/main.py -c

# Visualize ECG signals
python src/main.py -f

# Train ML models
python src/main.py -m
```

## 📁 Project Structure

```
ECG-to-stress/
├── src/
│   ├── main.py                 # CLI entry point
│   ├── data.py                  # WESAD data loading & processing
│   ├── features.py              # HRV feature extraction
│   ├── visualization.py         # Plotting and visualization
│   ├── correlation.py           # Correlation analysis
│   └── ml.py                    # Machine learning evaluation
│
├── notebooks/
│   ├── 01_explore_the_dataset.ipynb
│   ├── 02_feature_visualization.ipynb
│   ├── 03_full_signal_view.ipynb
│   ├── 04_features_correlation.ipynb
│   └── 05_ml_models.ipynb
│
├── results/                     # Generated results (plots, CSVs)
│
├── data/WESAD/                  # WESAD dataset (not tracked in git)
│
├── README.md                    # This file
├── CLI_README.md                # Comprehensive CLI usage guide
├── CLI_QUICK_REFERENCE.md       # Quick command lookup
├── CLI_OVERVIEW.md              # CLI overview and walkthrough
├── CLI_COMMAND_STRUCTURE.md     # Visual command structure diagrams
├── CLI_IMPLEMENTATION_SUMMARY.md # Implementation details summary
│
└── .gitignore
```

## 📚 Documentation

### Main Documentation Files

| File | Description |
|------|-------------|
| [CLI_OVERVIEW.md](CLI_OVERVIEW.md) | **Start here** - Quick overview and getting started guide |
| [CLI_README.md](CLI_README.md) | Comprehensive CLI usage guide with detailed examples |
| [CLI_QUICK_REFERENCE.md](CLI_QUICK_REFERENCE.md) | Fast command lookup and cheat sheet |
| [CLI_COMMAND_STRUCTURE.md](CLI_COMMAND_STRUCTURE.md) | Visual diagrams of command hierarchy and data flow |
| [CLI_IMPLEMENTATION_SUMMARY.md](CLI_IMPLEMENTATION_SUMMARY.md) | Summary of CLI features and design decisions |

### Notebooks

The `notebooks/` directory contains Jupyter notebooks that explore the dataset and demonstrate the analysis workflow:
- `01_explore_the_dataset.ipynb` - Dataset exploration
- `02_feature_visualization.ipynb` - Feature visualization
- `03_full_signal_view.ipynb` - Full signal viewing
- `04_features_correlation.ipynb` - Feature correlation analysis
- `05_ml_models.ipynb` - Machine learning model training

## 🎯 CLI Commands

The CLI (`src/main.py`) provides three main commands:

### Correlation Analysis (`-c` / `--corr`)
Extract HRV features and generate correlation visualizations.
```bash
python src/main.py -c
python src/main.py -c -f mean_rr mean_hr -d 30
```

### Full Signal Visualization (`-f` / `--full`)
Plot ECG signals with adjustable detail level.
```bash
python src/main.py -f
python src/main.py -f -p 10000 -s 0 1 2
```

### Machine Learning Training (`-m` / `--ml`)
Train models with cross-validation on ECG chunks.
```bash
python src/main.py -m
python src/main.py -m -d 30 -mo knn svm random_forest
```

### Common Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i`, `--input` | Path to the WESAD dataset directory | `data/WESAD` |
| `-d`, `--dataset` | Dataset durations in seconds | `30 120 300` |
| `-o`, `--output` | Custom output directory | Varies by command |

The dataset path defaults to `data/WESAD` relative to the project root. Use `-i` to specify an alternative location:
```bash
python src/main.py -i /path/to/WESAD -c
python src/main.py -i ./my_dataset -m
```

## 🧠 Features

- **8 HRV Features**: mean_rr, mean_hr, sdnn, rmssd, pnn50, lf_power, hf_power, lf_hf_ratio
- **3 Dataset Durations**: 30s, 120s, 300s window sizes
- **7 ML Models**: KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, Logistic Regression, XGBoost
- **Cross-Validation**: Stratified k-fold with configurable folds
- **Binary Classification**: Stress (labels 2,4) vs. No-Stress (labels 1,3)

## 🤝 Contributing

This is a research project for Master's thesis work. Contributions and suggestions are welcome.

## 📄 License

This project is for academic and research purposes.