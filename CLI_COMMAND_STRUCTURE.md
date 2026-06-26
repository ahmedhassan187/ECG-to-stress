# CLI Command Structure Diagram

## Command Hierarchy

```
python src/main.py
│
├─ --help / -h
│  └─ Shows comprehensive help for all commands
│
├─ CORRELATION (-c / --corr)
│  ├─ -i / --input [path]
│  │  └─ Default: data/WESAD
│  │
│  ├─ -f / --features [feature1 feature2 ...]
│  │  └─ Options: mean_rr, mean_hr, sdnn, rmssd, pnn50, lf_power, hf_power, lf_hf_ratio
│  │  └─ Default: all
│  │
│  ├─ -d / --dataset [30 120 300]
│  │  └─ Default: 30 120 300 (all)
│  │
│  └─ -o / --output [path]
│     └─ Default: ../results/correlation_figures
│
├─ FULL SIGNAL (-f / --full)
│  ├─ -i / --input [path]
│  │  └─ Default: data/WESAD
│  │
│  ├─ -p / --points [int]
│  │  └─ Range: 2000-15000
│  │  └─ Default: 5000
│  │
│  ├─ -s / --subjects [0 1 2 ...]
│  │  └─ Default: all subjects
│  │
│  └─ -o / --output [path]
│     └─ Default: ../results/signal_plots
│
└─ MACHINE LEARNING (-m / --ml)
   ├─ -i / --input [path]
   │  └─ Default: data/WESAD
   │
   ├─ -d / --dataset [30 120 300]
   │  └─ Default: 30 120 300 (all)
   │
   ├─ -mo / --models [model1 model2 ...]
   │  └─ Options: knn, svm, decision_tree, random_forest, gradient_boosting, logistic_regression, xgboost
   │  └─ Default: all 7 models
   │
   ├─ -cv / --cross-val [int]
   │  └─ Default: 5
   │
   └─ -o / --output [path]
      └─ Default: ../results/ml_results
```

---

## Quick Command Reference

### One-Liners

| Task | Command |
|------|---------|
| Help | `python src/main.py --help` |
| Correlations (all) | `python src/main.py -c` |
| Correlations (specific) | `python src/main.py -c -f mean_rr mean_hr -d 30` |
| Correlations (custom path) | `python src/main.py -i /path/to/data -c` |
| Visualizations (default) | `python src/main.py -f` |
| Visualizations (custom) | `python src/main.py -f -p 8000 -s 0 1 2` |
| Visualizations (custom path) | `python src/main.py -i /path/to/data -f` |
| ML (all) | `python src/main.py -m` |
| ML (custom) | `python src/main.py -m -d 30 -mo knn svm xgboost` |
| ML (custom path) | `python src/main.py -i /path/to/data -m` |

---

## Data Flow Diagram

```
ECG DATASET (WESAD)
        │
        ├─────────────────────┬──────────────────────┬─────────────────
        │                     │                      │
        ▼                     ▼                      ▼
   ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
   │ CORRELATION │    │ VISUALIZATION│    │ ML TRAINING      │
   └─────────────┘    └──────────────┘    └──────────────────┘
        │                     │                      │
        │ 1. Extract HRV      │ 1. Load Signal      │ 1. Extract HRV
        │    Features         │ 2. Create Chunks   │    Features
        │ 2. Analyze          │    (adjustable)     │ 2. Normalize
        │    Correlations     │ 3. Plot with        │ 3. Split/Cross-Val
        │ 3. Generate         │    Labels           │ 4. Train Models
        │    Figures          │ 4. Export PNG       │ 5. Evaluate
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
   │ CSV + Plots │    │ PNG Files    │    │ CSV + Plots      │
   │ (Features)  │    │ (Signals)    │    │ (Metrics)        │
   └─────────────┘    └──────────────┘    └──────────────────┘
```

---

## Example Command Chains

### Chain 1: Complete Analysis
```bash
# Step 1: Understand feature relationships
python src/main.py -c -f mean_rr mean_hr sdnn rmssd -d 30 120 300

# Step 2: Inspect raw signals
python src/main.py -f -p 5000

# Step 3: Train models
python src/main.py -m
```

### Chain 2: Focused Investigation
```bash
# Only analyze 30s chunks with specific models
python src/main.py -m -d 30 -mo random_forest gradient_boosting xgboost

# Visualize those signals
python src/main.py -f -p 5000

# Analyze specific features for that duration
python src/main.py -c -f sdnn rmssd -d 30
```

### Chain 3: Performance Optimization
```bash
# Quick test with few models
python src/main.py -m -d 30 -mo knn svm

# Compare visualization chunk sizes
python src/main.py -f -p 3000
python src/main.py -f -p 5000
python src/main.py -f -p 10000

# Deep analysis of best features
python src/main.py -c -f mean_rr sdnn rmssd lf_power hf_power
```

---

## Argument Patterns

### Pattern 1: Minimal (Use Defaults)
```bash
python src/main.py -c
python src/main.py -f
python src/main.py -m
```
**Result:** All features, all durations, all subjects, all models, default parameters

### Pattern 2: Specific Values
```bash
python src/main.py -c -f mean_rr mean_hr
python src/main.py -f -p 10000
python src/main.py -m -mo knn svm
```
**Result:** Specific features/models/parameters only

### Pattern 3: Partial Specification
```bash
python src/main.py -c -d 30
python src/main.py -m -d 30 -mo random_forest
```
**Result:** Mix of specific and default values

### Pattern 4: Complete Control
```bash
python src/main.py -c -f mean_rr sdnn -d 30 120 -o ./my_results
python src/main.py -f -p 8000 -s 0 1 2 -o ./signals
python src/main.py -m -d 30 -mo knn svm -cv 10 -o ./ml
python src/main.py -i /path/to/WESAD -c -f mean_rr -d 30
```
**Result:** Fully customized behavior with custom output paths

---

## Decision Tree: Which Command to Use?

```
What do you want to do?
│
├─ "Understand feature relationships?"
│  └─ USE: python src/main.py -c [options]
│     └─ Generates correlation analysis
│
├─ "Inspect the raw ECG signals?"
│  └─ USE: python src/main.py -f [options]
│     └─ Creates visualization plots
│
├─ "Train and evaluate models?"
│  └─ USE: python src/main.py -m [options]
│     └─ Performs ML training with CV
│
└─ "Not sure where to start?"
   └─ RUN: python src/main.py --help
      └─ Shows comprehensive help
```

---

## Parameter Combinations

### For Correlation Analysis
```
Features × Durations = Total Analyses

Examples:
- 1 feature  × 1 duration  = 1 analysis
- 3 features × 1 duration  = 3 analyses
- 8 features × 3 durations = 24 analyses (all)
```

### For Visualization
```
Chunk Size Options:
- 2000 points  = 7-8 plots per subject (high detail)
- 5000 points  = 3-4 plots per subject (balanced)
- 10000 points = 2 plots per subject (overview)
```

### For ML Training
```
Models × Durations × CV Folds = Total CV Runs

Examples:
- 1 model  × 1 duration  × 5 folds = 5 CV runs
- 3 models × 2 durations × 10 folds = 60 CV runs
- 7 models × 3 durations × 5 folds = 105 CV runs (all)
```

---

## Performance Expectations

### Processing Time (Approximate)

| Command | Dataset | Time |
|---------|---------|------|
| `-c` (correlation) | 30s | < 5 seconds |
| `-c` (correlation) | 120s | < 5 seconds |
| `-c` (correlation) | 300s | < 5 seconds |
| `-f` (visualization) | 1 subject, 5000pts | 1-2 seconds |
| `-f` (visualization) | All subjects | 10-20 seconds |
| `-m` (ML, 1 model) | 30s, 5-fold | 5-10 seconds |
| `-m` (ML, 7 models) | All, 5-fold | 60-120 seconds |

### Output File Sizes (Approximate)

| Output | Size |
|--------|------|
| CSV per dataset | 50-500 KB |
| PNG plot per subject | 100-500 KB |
| Summary CSV | 10-50 KB |

---

## Tips & Tricks

### Tip 1: Start Simple
```bash
# Good starting point
python src/main.py -c
python src/main.py -f -p 5000
python src/main.py -m -d 30
```

### Tip 2: Test Before Full Run
```bash
# Quick test with 1 model
python src/main.py -m -d 30 -mo knn

# Then run full if satisfied
python src/main.py -m
```

### Tip 3: Organize Outputs
```bash
python src/main.py -c -o ./results/correlation
python src/main.py -f -o ./results/visualization
python src/main.py -m -o ./results/ml
```

### Tip 4: Combine with Piping
```bash
# Run analysis and save log
python src/main.py -m -d 30 > ml_training.log 2>&1

# Run multiple analyses sequentially
python src/main.py -c && python src/main.py -f && python src/main.py -m