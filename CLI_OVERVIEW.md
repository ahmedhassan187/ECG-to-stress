# ECG-to-Stress CLI - Complete Overview

## Welcome! 👋

The CLI (Command-Line Interface) provides an easy way to run all ECG analysis, visualization, and machine learning workflows without opening Jupyter notebooks.

---

## 🚀 Get Started in 30 Seconds

### Installation Check
```bash
# Navigate to project root
cd g:\Master\Thesis\FLT\Code\ECG-to-stress

# Verify WESAD data is present
ls data/WESAD/
```

### Run Your First Command
```bash
# See what's available
python src/main.py --help

# Run correlation analysis
python src/main.py -c

# Visualize signals
python src/main.py -f

# Train ML models
python src/main.py -m
```

**That's it!** Results are saved to `results/` directories.

---

## 📚 Documentation Files

| File | Purpose | Best For |
|------|---------|----------|
| **README.md** | Main repository readme | Getting started |
| **CLI_README.md** | Comprehensive guide | Detailed learning |
| **CLI_QUICK_REFERENCE.md** | Quick lookup | Fast command lookup |
| **CLI_COMMAND_STRUCTURE.md** | Visual diagrams | Understanding flow |
| **CLI_IMPLEMENTATION_SUMMARY.md** | Overview | Getting oriented |

---

## 3️⃣ Main Commands

### 1. Correlation Analysis (`-c`)
**What it does:** Extracts HRV features and analyzes their correlations

```bash
python src/main.py -c                              # All features
python src/main.py -c -f mean_rr mean_hr sdnn      # Specific features
python src/main.py -c -d 30 120                    # Specific durations
python src/main.py -i /path/to/WESAD -c            # Custom dataset path
```

📊 **Output:** CSV files with correlation data + visualization plots

---

### 2. Full Signal Visualization (`-f`)
**What it does:** Creates visual plots of ECG signals with adjustable detail level

```bash
python src/main.py -f                              # Default 5000 points
python src/main.py -f -p 10000                     # Larger chunks
python src/main.py -f -s 0 1 2                     # Specific subjects
python src/main.py -i /path/to/WESAD -f            # Custom dataset path
```

📈 **Output:** PNG files showing ECG waveforms with color-coded labels

---

### 3. Machine Learning Training (`-m`)
**What it does:** Trains 7 different models with cross-validation on ECG chunks

```bash
python src/main.py -m                              # All models, all durations
python src/main.py -m -d 30                        # Specific duration
python src/main.py -m -mo knn svm random_forest    # Specific models
python src/main.py -i /path/to/WESAD -m            # Custom dataset path
```

🤖 **Output:** CSV results + comparison plots + performance metrics

---

## 💡 Common Workflows

### Workflow 1: Explore Your Data (5 minutes)
```bash
# Understand feature relationships
python src/main.py -c -d 30

# Look at raw signals
python src/main.py -f -p 5000 -s 0

# Quick ML test
python src/main.py -m -d 30 -mo knn svm
```

### Workflow 2: Complete Analysis (30 minutes)
```bash
# Full correlation analysis
python src/main.py -c

# Visualize all signals
python src/main.py -f -p 7500

# Train all models with 10-fold CV
python src/main.py -m -cv 10
```

### Workflow 3: Model Comparison (10 minutes)
```bash
# Test 30s chunks
python src/main.py -m -d 30 -mo random_forest gradient_boosting xgboost

# Compare with 120s chunks
python src/main.py -m -d 120 -mo random_forest gradient_boosting xgboost

# Analyze winning features
python src/main.py -c -f mean_rr sdnn rmssd -d 30 120
```

---

## 🎯 Quick Command Cheat Sheet

```bash
# Get help
python src/main.py --help
python src/main.py -c --help

# Dataset path (optional, defaults to data/WESAD)
python src/main.py -i /path/to/WESAD -c

# Correlation (3 variants)
python src/main.py -c
python src/main.py --corr -f mean_rr
python src/main.py correlation -d 30 120

# Visualization (3 variants)
python src/main.py -f
python src/main.py --full -p 10000
python src/main.py full -s 0 1 2

# Machine Learning (3 variants)
python src/main.py -m
python src/main.py --ml -d 30
python src/main.py ml -mo knn svm xgboost
```

---

## 📊 What's Happening Behind the Scenes?

### Correlation Analysis Flow
```
Load WESAD Dataset
    ↓
Extract HRV Features
    ├─ Mean RR, Mean HR
    ├─ SDNN, RMSSD, PNN50
    └─ LF Power, HF Power, LF/HF Ratio
    ↓
Compute Correlations
    ↓
Generate Plots & Export CSV
```

### Visualization Flow
```
Load WESAD Dataset
    ↓
Iterate Through Subjects
    ↓
Split Signal into Chunks (adjustable size)
    ↓
Plot with Color-Coded Labels
    ↓
Save PNG Files
```

### ML Training Flow
```
Load WESAD Dataset
    ↓
Extract HRV Features
    ↓
Create Binary Labels (Stress vs No-Stress)
    ↓
For Each Duration & Model:
    ├─ Normalize Features
    ├─ Apply 5-Fold Cross-Validation
    ├─ Train Model
    └─ Evaluate Metrics
    ↓
Export Results & Plots
```

---

## 🎨 Features at a Glance

| Feature | Description |
|---------|-------------|
| **Adjustable Input** | Dataset path via `-i`/`--input`, defaults to `data/WESAD` |
| **8 HRV Features** | Mean RR, Mean HR, SDNN, RMSSD, PNN50, LF Power, HF Power, LF/HF Ratio |
| **3 Durations** | 30s, 120s, 300s chunks |
| **7 ML Models** | KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, Logistic Regression, XGBoost |
| **Multiple Controls** | Input path, feature selection, duration selection, model selection, CV folds, custom outputs |
| **Smart Defaults** | Works out-of-the-box with no arguments |

---

## ❓ Common Questions

**Q: Do I need to modify any files to use the CLI?**
A: No! The CLI is self-contained in `src/main.py` and ready to use.

**Q: What if my dataset is in a different location?**
A: Use the `-i` flag to specify a custom path:
```bash
python src/main.py -i /path/to/your/WESAD -m
```

**Q: What if I get "Module not found" errors?**
A: Make sure you're in the project root directory:
```bash
cd g:\Master\Thesis\FLT\Code\ECG-to-stress
python src/main.py -m
```

**Q: Can I run multiple commands together?**
A: Yes! Use `&&` to chain commands:
```bash
python src/main.py -c && python src/main.py -f && python src/main.py -m
```

**Q: How long do commands take?**
A: - Correlation: < 5 seconds
- Visualization (1 subject): 1-2 seconds
- ML (1 model, 1 duration): 5-10 seconds
- Full ML pipeline: 60-120 seconds

**Q: Where are results saved?**
A: In `results/` subdirectories:
```
results/
├── correlation_figures/
├── signal_plots/
└── ml_results/
```

---

## 📖 Reading Order

If you're new to the CLI, read these files in order:

1. **README.md** (Main repo readme - start here)
2. **This file** (CLI_OVERVIEW.md) ← You are here
3. **CLI_QUICK_REFERENCE.md** (2 min read)
4. **CLI_COMMAND_STRUCTURE.md** (5 min read)
5. **CLI_README.md** (15 min read)

---

## 🔗 Next Steps

### Option 1: Learn by Doing
```bash
# Just run commands and see what happens
python src/main.py -c
python src/main.py -f -p 5000
python src/main.py -m -d 30
```

### Option 2: Deep Dive
- Read `CLI_README.md` for comprehensive guide
- Check `CLI_COMMAND_STRUCTURE.md` for visual diagrams
- Refer to `CLI_QUICK_REFERENCE.md` for quick lookup

### Option 3: Copy-Paste Workflows
- Use examples from **CLI_IMPLEMENTATION_SUMMARY.md**
- Modify parameters as needed
- Save commands you like for later

---

## 📞 Troubleshooting

### Issue: Command not found
```bash
# Solution: Use correct path and format
python src/main.py -c
# NOT: python -c or main.py -c or python main -c
```

### Issue: Output files not created
```bash
# Check if results directory was created
ls results/

# If not found, check error message carefully
python src/main.py -m 2>&1 | head -20
```

### Issue: Out of memory with all datasets
```bash
# Solution: Run one duration at a time
python src/main.py -m -d 30
python src/main.py -m -d 120
python src/main.py -m -d 300
```

---

## ✨ Tips for Power Users

### Tip 1: Save Favorite Commands
Create a `commands.sh` file:
```bash
#!/bin/bash
# My favorite analysis commands

# Quick test
python src/main.py -m -d 30 -mo knn svm

# Deep dive
python src/main.py -c -f mean_rr sdnn rmssd -d 30 120 300
python src/main.py -f -p 5000
python src/main.py -m -cv 10

# Full pipeline
python src/main.py -c && python src/main.py -f && python src/main.py -m
```

### Tip 2: Organize Results
```bash
# Create timestamped results
python src/main.py -m -o results/ml_$(date +%Y%m%d_%H%M%S)

# Or use descriptive names
python src/main.py -m -d 30 -o results/ml_30s_experiment1
```

### Tip 3: Log Everything
```bash
# Save command output to file
python src/main.py -m -d 30 2>&1 | tee analysis_log.txt
```

---

## 🎓 Educational Overview

This CLI demonstrates:
- **Signal Processing**: ECG chunking and HRV feature extraction
- **Data Analysis**: Correlation analysis and statistical visualization
- **Machine Learning**: Model training with cross-validation
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

---

**Last Updated:** 2026-06-27
**Version:** 1.1
**Status:** Production Ready ✅