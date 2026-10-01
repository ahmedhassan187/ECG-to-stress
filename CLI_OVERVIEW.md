# ECG-to-Stress CLI — Complete Overview

> Supporting the paper **"Investigating the Effect of ECG Recording Duration on HRV Reliability and Stress Classification"** (AIMEH conference, WiMoB group).

## Welcome! 👋

The CLI (Command-Line Interface) provides an easy way to run all ECG analysis workflows — **correlation / reliability analysis**, **signal visualization**, **machine learning training**, **FFT frequency analysis**, and **prediction on new data** — without opening Jupyter notebooks.

---

## 🚀 Get Started in 30 Seconds

### Installation Check

```bash
# Navigate to the project root
cd g:\Master\Thesis\FLT\Code\ECG-to-stress

# Verify WESAD data is present
ls data/WESAD/
```

### Run Your First Command

```bash
# See what's available
python src/main.py --help

# Run correlation / cross-duration reliability analysis
python src/main.py -c

# Visualize signals
python src/main.py -f

# Train ML models
python src/main.py -m

# Run FFT frequency analysis
python src/main.py --fft

# Make predictions with saved models (default: WESAD subject 0)
python src/main.py --predict
```

**That's it!** Results are saved to `results/` subdirectories.

---

## 📚 Documentation Files

| File | Purpose | Best For |
|------|---------|----------|
| **README.md** | Main repository readme (paper, overview, install) | Getting started |
| **CLI_README.md** | Comprehensive CLI usage guide | Detailed learning |
| **CLI_QUICK_REFERENCE.md** | Quick command lookup | Fast command reference |
| **CLI_COMMAND_STRUCTURE.md** | Visual command structure diagrams | Understanding flow |
| **CLI_IMPLEMENTATION_SUMMARY.md** | Implementation overview | Getting oriented |

---

## 🔧 5 Main Commands

### 1. Correlation / Reliability Analysis (`-c`)

**What it does:** Extracts HRV features and compares them **across recording durations** using Pearson `r`, ICC and MAE — the core statistical analysis of the paper.

```bash
python src/main.py -c                                        # All features & durations
python src/main.py -c --features mean_rr mean_hr sdnn -d 30 120   # Specific features / durations
python src/main.py -c --by-condition                         # Metrics per condition group
```

📊 **Output:** CSV tables + heatmaps + bar charts + styled summary tables

### 2. Full Signal Visualization (`-f`)

**What it does:** Creates visual plots of ECG signals with adjustable detail level.

```bash
python src/main.py -f                                        # Default 5000 points
python src/main.py -f -p 10000 -s 0 1 2                      # Custom points / subjects
```

📈 **Output:** PNG files showing ECG waveforms with color-coded labels

### 3. Machine Learning Training (`-m`)

**What it does:** Trains 7 different models with k-fold cross-validation on ECG chunks and saves the trained models as `.pkl` files.

```bash
python src/main.py -m                                        # All models, all durations
python src/main.py -m -d 30 -mo knn svm random_forest        # Specific models
python src/main.py -m -cv 10 -l 3class                       # 10-fold, 3-class
```

🤖 **Output:** CSV results + saved models + performance metrics

### 4. FFT Frequency Analysis (`--fft`)

**What it does:** Frequency-domain analysis — mean FFT spectra per class across durations + cosine-similarity comparison.

```bash
python src/main.py --fft                                     # All durations
python src/main.py --fft -d 30 --fft-max-pairs 1000          # 30 s + more pairs
```

🌊 **Output:** spectra plots + cosine-similarity distributions + `fft_cosine_summary.csv`

### 5. Prediction Mode (`--predict`)

**What it does:** Loads saved models and predicts stress labels on new data. Supports three input modes:

- **WESAD dataset** (default) — uses subject 0 of WESAD as test data
- **Pavia HRV data** — `--pavia` (optionally a path to the folder with the CSVs)
- **Custom CSV** — `--test-data` / `--test-labels`

```bash
python src/main.py --predict                                 # WESAD subject 0 (default)
python src/main.py --predict --pavia                         # Pavia data in default folder
python src/main.py --predict --test-data t.csv --test-labels t_labels.csv
```

🔮 **Output:** `predictions_<dur>s.csv` + comparison / metrics figures

---

## 💡 Common Workflows

### Workflow 1: Explore Your Data (5 minutes)

```bash
# Understand HRV feature reliability across recording durations
python src/main.py -c -d 30

# Look at raw signals
python src/main.py -f -p 5000 -s 0

# Quick ML test
python src/main.py -m -d 30 -mo knn svm
```

### Workflow 2: Complete Analysis (30 minutes)

```bash
# Full reliability analysis (r / ICC / MAE) across durations
python src/main.py -c

# Visualize all signals
python src/main.py -f -p 7500

# Train all models with 10-fold CV
python src/main.py -m -cv 10

# FFT frequency analysis
python src/main.py --fft
```

### Workflow 3: Validate New Data (10 minutes)

```bash
# Ensure models are trained first
python src/main.py -m -d 30

# Predict on Pavia HRV data (or custom CSVs)
python src/main.py --predict --pavia
python src/main.py --predict --test-data t.csv --test-labels t_labels.csv
```

---

## 📞 Troubleshooting

### Issue: Command not found
```bash
# Solution: Use the correct invocation
python src/main.py -c
# NOT: python -c  or  main.py -c  or  python main -c
```

### Issue: Output files not created
```bash
# Check if the results directory was created
ls results/

# Check the error message carefully
python src/main.py -m 2>&1 | head -20
```

### Issue: Model not found during prediction
```bash
# Train (and save) models first, then predict
python src/main.py -m -d 30 -mo knn svm
python src/main.py --predict -d 30 --pavia
```

### Issue: Out of memory with all datasets
```bash
# Run one duration at a time
python src/main.py -m -d 30
python src/main.py -m -d 120
python src/main.py -m -d 300
```

---

## 📖 Reading Order

If you're new to the CLI, read these files in order:

1. **README.md** (Main repo readme — start here)
2. **This file** (CLI_OVERVIEW.md) ← You are here
3. **CLI_QUICK_REFERENCE.md** (2 min read)
4. **CLI_COMMAND_STRUCTURE.md** (5 min read)
5. **CLI_README.md** (15 min read)

---

## 🎓 Educational Overview

This CLI demonstrates:

- **Signal Processing**: ECG chunking, HRV feature extraction, FFT frequency analysis
- **Statistical Reliability**: Pearson r, ICC, MAE across recording durations
- **Machine Learning**: Model training, cross-validation, saved models, prediction
- **Software Engineering**: CLI design, argparse, modular architecture

Perfect for learning or research!

---

## 🚀 Ready? Let's Go!

```bash
# Start here:
python src/main.py --help

# Or try this:
python src/main.py -c

# Happy analyzing! 🎉
```