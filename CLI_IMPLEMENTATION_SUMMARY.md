# CLI Interface Implementation Summary

## ✅ What's Been Created

I've implemented a comprehensive command-line interface (CLI) for your ECG-to-Stress analysis project. The CLI provides easy access to all major functionality through simple commands.

---

## 📋 Three Main Commands

### 1. **Correlation Analysis** (`-c` / `--corr`)
Extract HRV features and generate correlation analysis figures.

```bash
# Default: all features, all durations
python src/main.py -c

# Custom features
python src/main.py -c -f mean_rr mean_hr sdnn rmssd

# Custom durations
python src/main.py -c -d 30 120

# Both custom
python src/main.py -c -f mean_rr -d 30 120 -o ./results
```

**Key Options:**
- `-f / --features`: Specify which HRV features to analyze (default: all)
- `-d / --dataset`: Specify dataset durations in seconds (default: 30 120 300)
- `-o / --output`: Custom output directory

**Available Features:**
- `mean_rr` - Mean RR interval
- `mean_hr` - Mean heart rate
- `sdnn` - Standard deviation of NN intervals
- `rmssd` - Root mean square of successive differences
- `pnn50` - Percentage of NN50 count
- `lf_power` - Low frequency power
- `hf_power` - High frequency power
- `lf_hf_ratio` - LF/HF ratio

---

### 2. **Full Signal Visualization** (`-f` / `--full`)
Plot complete ECG signals with adjustable chunk size.

```bash
# Default: 5000 points per chunk
python src/main.py -f

# Larger chunks (less detail, fewer plots)
python src/main.py -f -p 10000

# Smaller chunks (more detail, more plots)
python src/main.py -f -p 2000

# Specific subjects
python src/main.py -f -s 0 1 2

# Custom combination
python src/main.py -f -p 8000 -s 0 3 5 -o ./my_plots
```

**Key Options:**
- `-p / --points`: Adjust chunk size (points per plot) (default: 5000)
- `-s / --subjects`: Specify subject IDs to plot (default: all)
- `-o / --output`: Custom output directory

**Point Size Guidelines:**
- 2000-5000: High detail visualization
- 5000-10000: Balanced overview
- 10000+: Large segments, fewer plots

---

### 3. **Machine Learning Training** (`-m` / `--ml`)
Train ML models with cross-validation on ECG chunks.

```bash
# All models, all durations
python src/main.py -m

# Specific durations
python src/main.py -m -d 30 120

# Specific models
python src/main.py -m -mo knn svm xgboost

# Both specific
python src/main.py -m -d 30 -mo random_forest gradient_boosting

# Adjust cross-validation folds
python src/main.py -m -cv 10

# Complete customization
python src/main.py -m -d 30 -mo knn svm -cv 10 -o ./results
```

**Key Options:**
- `-d / --dataset`: Dataset durations (30, 120, 300 seconds) (default: all)
- `-mo / --models`: Specify models to train (default: all)
- `-cv / --cross-val`: Number of CV folds (default: 5)
- `-o / --output`: Custom output directory

**Available Models:**
- `knn` - K-Nearest Neighbors
- `svm` - Support Vector Machine
- `decision_tree` - Decision Tree Classifier
- `random_forest` - Random Forest
- `gradient_boosting` - Gradient Boosting
- `logistic_regression` - Logistic Regression
- `xgboost` - XGBoost Classifier

---

## 📁 File Structure

```
ECG-to-stress/
├── src/
│   ├── main.py                    ✨ NEW - CLI interface
│   ├── data.py                    (existing classes used)
│   ├── features.py
│   ├── visualization.py
│   └── correlation.py
├── CLI_README.md                  ✨ NEW - Comprehensive guide
├── CLI_QUICK_REFERENCE.md         ✨ NEW - Quick lookup
└── notebooks/
    └── 05_ml_models.ipynb         (existing)
```

---

## 💡 Example Usage Patterns

### Pattern 1: Quick Overview
```bash
python src/main.py -c
python src/main.py -f -p 5000
python src/main.py -m -d 30
```

### Pattern 2: Detailed Analysis
```bash
# Analyze correlations for specific features
python src/main.py -c -f mean_rr mean_hr sdnn rmssd pnn50 -d 30 120 300

# Visualize signals with high detail
python src/main.py -f -p 2500 -s 0 1 2 3 4

# Train best models with 10-fold CV
python src/main.py -m -cv 10 -mo random_forest gradient_boosting xgboost
```

### Pattern 3: Specific Investigation
```bash
# Test 30s dataset with specific models
python src/main.py -m -d 30 -mo knn svm random_forest

# Visualize particular subjects
python src/main.py -f -s 2 5 8 -p 7500

# Correlate specific features
python src/main.py -c -f sdnn rmssd lf_power hf_power -d 120
```

---

## 🔧 Features

✅ **Comprehensive Help System**
- `python src/main.py --help` - Main help
- `python src/main.py -c --help` - Correlation help
- `python src/main.py -f --help` - Visualization help
- `python src/main.py -m --help` - ML training help

✅ **Flexible Argument Parsing**
- Short flags (`-c`, `-f`, `-m`)
- Long flags (`--corr`, `--full`, `--ml`)
- Subcommand aliases for flexibility

✅ **Smart Defaults**
- Correlation: All features, all durations (30, 120, 300s)
- Visualization: 5000 points per chunk, all subjects
- ML: All 7 models, all durations, 5-fold CV

✅ **Custom Configuration**
- Specify exact features, durations, models
- Adjust visualization parameters
- Configure output directories
- Customize cross-validation folds

✅ **Informative Output**
- Progress messages with emojis
- Clear file paths for outputs
- Sample count and data statistics
- Result summaries

---

## 📊 Output Examples

### Correlation Analysis Output
```
📂 Loading dataset from: ../data/WESAD
✓ Loaded 15 subjects

30s Dataset:
✓ Created 196 chunks with 8 HRV features
📈 Generating correlation visualizations for 3 features...
✓ Saved: results/correlation_figures/features_30s.csv

✅ Correlation analysis complete!
```

### Full Signal Visualization Output
```
📂 Loading dataset from: ../data/WESAD
✓ Loaded 15 subjects
📊 Chunk size: 10000 points per plot

📌 Processing Subject 0...
✓ Saved: results/signal_plots/subject_00_ecg.png

✅ ECG signal visualization complete!
```

### ML Training Output
```
🤖 Models to train: knn, svm, xgboost
📊 Cross-validation folds: 5

⏱️  Processing 30s chunks...
✓ Created 196 chunks
✓ Extracted 196 samples × 8 features
🚀 Training 3 models...
   → KNN... ✓
   → SVM... ✓
   → XGBOOST... ✓
✓ 30s dataset processing complete

✅ ML model training complete!
```

---

## 📚 Documentation Files

### `CLI_README.md`
Comprehensive usage guide with:
- Detailed command descriptions
- All available options
- Feature explanations
- Model descriptions
- Examples for each command
- Output structure
- Troubleshooting guide

### `CLI_QUICK_REFERENCE.md`
Quick lookup sheet with:
- All commands at a glance
- Common usage patterns
- Available options summary
- Parameter ranges
- Quick copy-paste examples

---

## 🚀 Next Steps (Optional)

The CLI is designed to be extensible. You can enhance it further by:

1. **Integrate with Jupyter**: Add command to export notebook results to CLI
2. **Add Batch Processing**: Support running multiple analysis pipelines
3. **Add Result Comparison**: Compare results across different parameter combinations
4. **Add Performance Metrics**: Display real-time training metrics
5. **Add Configuration Files**: Support loading arguments from config files

---

## 📝 Key Design Decisions

✅ **Argparse**: Professional, standard Python CLI library
✅ **Subcommands**: Clear separation of concerns
✅ **Aliases**: Multiple ways to invoke same command
✅ **Sensible Defaults**: Works without arguments
✅ **Informative Messages**: User knows what's happening
✅ **Flexible Paths**: Relative and absolute path support
✅ **Error Handling**: Validation of inputs with helpful messages

---

## ✨ Summary

You now have a production-ready CLI that provides:
- 3 main commands (correlation, visualization, ML)
- 7 different ML models
- 3 dataset durations
- 8 HRV features
- Fully customizable parameters
- Comprehensive documentation

Users can now perform complex analysis with simple, intuitive commands!

