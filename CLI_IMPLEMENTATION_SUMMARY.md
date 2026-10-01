# CLI Interface Implementation Summary

> Supporting the paper **"Investigating the Effect of ECG Recording Duration on HRV Reliability and Stress Classification"** (AIMEH conference, WiMoB group).

## ✅ What's Been Created

A comprehensive command-line interface (CLI) for the ECG-to-Stress project. The CLI provides easy access to all analysis functionality — correlation / reliability analysis, signal visualization, machine learning training, FFT analysis, and prediction — through simple commands.

---

## 🔧 Five Main Commands

### 1. Correlation / Reliability Analysis (`-c` / `--corr`)

Extracts HRV features at each window duration and computes **cross-duration reliability metrics** (Pearson `r`, ICC, MAE).

```bash
# Default: all features, all durations
python src/main.py -c

# Custom features / durations
python src/main.py -c --features mean_rr mean_hr sdnn -d 30 120

# Metrics split by condition group
python src/main.py -c --by-condition

# Custom dataset path
python src/main.py -i /path/to/WESAD -c
```

**Key Options:**
- `--features`: HRV features to analyze (`all` or a subset)
- `--by-condition`: compute metrics separately for stress / non-stress
- `-d / --dataset`: durations (default: `30 120 300`)
- `-l / --labels`: `binary` or `3class`

**Available Features (8):**
`mean_rr`, `mean_hr`, `sdnn`, `rmssd`, `pnn50`, `lf_power`, `hf_power`, `lf_hf_ratio`

### 2. Full Signal Visualization (`-f` / `--full`)

Plot complete ECG signals with adjustable chunk size.

```bash
python src/main.py -f                                # Default: 5000 points
python src/main.py -f -p 10000 -s 0 1 2              # Custom points / subjects
python src/main.py -i /path/to/WESAD -f              # Custom dataset path
```

**Key Options:** `-p / --points` (default `5000`), `-s / --subjects` (default: all)

### 3. Machine Learning Training (`-m` / `--ml`)

Trains ML models with cross-validation and saves them for prediction.

```bash
python src/main.py -m                                # All models, all durations
python src/main.py -m -d 30 -mo knn svm              # Specific duration / models
python src/main.py -m -cv 10 -l 3class               # 10-fold, 3-class
```

**Key Options:**
- `-mo / --models`: choose from 7 models
- `-cv / --cross-val`: number of CV folds (default `5`)
- `-l / --labels`: `binary` or `3class`

**Available Models (7):** KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, Logistic Regression, XGBoost

### 4. FFT Frequency Analysis (`--fft`)

Frequency-domain analysis with cosine-similarity comparison.

```bash
python src/main.py --fft                             # All durations
python src/main.py --fft -d 30 --fft-max-pairs 1000  # 30s, more pairs
```

**Key Options:** `--fft-max-pairs` (default `500`), `--fft-freq-max` (default `40.0` Hz)

### 5. Prediction Mode (`--predict`)

Loads saved models and predicts stress on new data.

```bash
python src/main.py --predict --pavia                 # Pavia HRV data (default folder)
python src/main.py --predict --test-data t.csv       # Custom CSV
python src/main.py --predict                          # WESAD subject 0
```

**Key Options:** `--model-dir`, `--pavia`, `--test-data`, `--test-labels`

---

## 🧠 Label Schemes (`src/label_config.py`)

| Mode | Mapping | Classes |
|------|---------|---------|
| **binary** (default) | 1→0, 2→1, 3→0, 4→0 | {No Stress, Stress} |
| **3class** | 1→0, 2→1, 3→0, 4→2 | {No Stress/Amusement, Stress, Meditation} |

---

## ✨ Extra Functionality Added

- **Cross-duration reliability analysis** — pair short windows with longer windows and compute Pearson `r`, ICC(2,1) and MAE; outputs pairwise comparison tables, heatmaps, per-feature and per-comparison bar charts, and styled metric summary tables.
- **FFT frequency analysis** — mean spectra per class, duration overlays, and cosine-similarity distributions (cross-class vs within-class).
- **Prediction mode** — load saved `.pkl` models and predict on Pavia HRV data, custom CSVs, or a WESAD subject.
- **Saved models** — every trained model is serialized with a `.meta.pkl` metadata file (features, label mapping, duration).
- **Series of figure scripts** (`src/figures/`):
  - `bar_plot_accuracy_f1.py` — grouped bar chart of Accuracy / F1 / AUC across models (binary & 3-class).
  - `ml_summary_table.py` — styled summary table of the best model per duration.

---

## 📝 Design Features

✅ **Argparse**: professional, standard Python CLI library
✅ **Mutually exclusive commands**: `-c`, `-f`, `-m`, `--fft`, `--predict`
✅ **Sensible defaults**: works out of the box with `data/WESAD`
✅ **Label schemes**: binary & 3-class via a single `-l / --labels` flag
✅ **Informative messages**: progress with clear file paths for outputs
✅ **Flexible paths**: relative and absolute input/output support
✅ **Adjustable model dir**: prediction can point to any saved-model folder
✅ **Cross-validation**: stratified k-fold (configurable folds)

---

## 📊 Output Examples

### Correlation / Reliability Output

```
📂 Loading dataset from: data/WESAD
✓ Loaded 15 subjects

🔗 Comparing 30s vs 120s (ratio 4:1)...
   • mean_rr     R=0.902  ICC=0.899  MAE=37.219  n=...
✓ Saved comparison CSV: results/correlation_figures/cross_duration_comparison.csv
✅ Correlation analysis complete!
```

### ML Training Output

```
🤖 Models to train: knn, svm, random_forest
📊 Cross-validation folds: 5
🚀 Training 3 models...
    → KNN ... ✓
    → SVM ... ✓
    → RANDOM FOREST ... ✓
✓ Saved results CSV: results/ml_results/ml_results_30s_binary.csv
✓ Saved models: results/ml_results/saved_models/*.pkl
✅ ML model training complete!
```

### Prediction Output

```
📋 Label mode: binary
🤖 Models to use: knn, svm
   ✓ Loaded KNN from saved_models/knn_30s.pkl
   ✓ Pavia data ready: 34 samples, 8 features
✓ Saved predictions to: results/predictions/predictions_30s.csv
✅ Prediction complete!
```

---

## 📚 Documentation Files

### `README.md` (Main)
Project overview, publication info, quick start. References all other documentation files.

### `CLI_README.md`
Comprehensive usage guide with:
- Detailed command descriptions for all 5 commands
- All available options
- Feature / model / label-mode explanations
- Examples for each command
- Output structure
- Troubleshooting guide

### `CLI_QUICK_REFERENCE.md`
Quick lookup sheet with:
- All commands at a glance
- Common usage patterns
- Available options summary

### `CLI_OVERVIEW.md`
Welcome and getting-started walkthrough with common workflows.

### `CLI_COMMAND_STRUCTURE.md`
Visual diagrams of command hierarchy, data flow, and a decision tree.

---

## ✨ Summary

You now have a production-ready CLI that provides:

- **5 main commands** (correlation/reliability, visualization, ML, FFT, prediction)
- **2 label schemes** (binary, 3-class)
- **8 HRV features** across **3 recording durations** (30/120/300 s)
- **7 ML models** with configurable k-fold cross-validation
- **Saved models** for reuse in prediction
- **Cross-duration reliability metrics** (Pearson r, ICC, MAE)

Run complex analysis with simple, intuitive commands!