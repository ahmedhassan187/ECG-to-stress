# CLI Command Structure Diagram

> Supporting the paper **"Investigating the Effect of ECG Recording Duration on HRV Reliability and Stress Classification"** (AIMEH conference, WiMoB group).

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
│  ├─ --features [feature1 feature2 ...]
│  │  └─ Options: mean_rr, mean_hr, sdnn, rmssd, pnn50, lf_power, hf_power, lf_hf_ratio
│  │  └─ Default: all
│  ├─ -d / --dataset [30 120 300]
│  │  └─ Default: 30 120 300 (all)
│  ├─ -l / --labels [binary | 3class]
│  │  └─ Default: binary
│  ├─ --by-condition
│  │  └─ Compute metrics separately per condition group
│  └─ -o / --output [path]
│     └─ Default: results/correlation_figures
│
├─ FULL SIGNAL (-f / --full)
│  ├─ -i / --input [path]
│  │  └─ Default: data/WESAD
│  ├─ -p / --points [int]
│  │  └─ Range: 2000-15000
│  │  └─ Default: 5000
│  ├─ -s / --subjects [0 1 2 ...]
│  │  └─ Default: all subjects
│  └─ -o / --output [path]
│     └─ Default: results/signal_plots
│
├─ MACHINE LEARNING (-m / --ml)
│  ├─ -i / --input [path]
│  │  └─ Default: data/WESAD
│  ├─ -d / --dataset [30 120 300]
│  │  └─ Default: 30 120 300 (all)
│  ├─ -mo / --models [model1 model2 ...]
│  │  └─ Options: knn, svm, decision_tree, random_forest, gradient_boosting, logistic_regression, xgboost
│  │  └─ Default: all 7 models
│  ├─ -cv / --cross-val [int]
│  │  └─ Default: 5
│  ├─ -l / --labels [binary | 3class]
│  │  └─ Default: binary
│  └─ -o / --output [path]
│     └─ Default: results/ml_results
│
├─ FFT ANALYSIS (--fft)
│  ├─ -d / --dataset [30 120 300]
│  │  └─ Default: 30 120 300 (all)
│  ├─ -l / --labels [binary | 3class]
│  │  └─ Default: binary
│  ├─ --fft-max-pairs [int]
│  │  └─ Default: 500
│  ├─ --fft-freq-max [float]
│  │  └─ Default: 40.0
│  └─ -o / --output [path]
│     └─ Default: results/fft_analysis
│
└─ PREDICTION (--predict)
   ├─ --model-dir [path]
   │  └─ Default: results/ml_results/saved_models
   ├─ -d / --dataset [30 120 300]
   │  └─ Default: [30] (overridden if specified)
   ├─ --pavia [path | default]
   │  └─ Use Pavia HRV data (default folder: data/)
   ├─ --test-data [path]
   │  └─ Test features CSV
   ├─ --test-labels [path]
   │  └─ Test labels CSV
   ├─ -l / --labels [binary | 3class]
   │  └─ Default: binary
   └─ -o / --output [path]
      └─ Default: results/predictions
```

---

## Quick Command Reference

| Task | Command |
|------|---------|
| Help | `python src/main.py --help` |
| Correlation / reliability (all) | `python src/main.py -c` |
| Correlation (specific) | `python src/main.py -c --features mean_rr mean_hr -d 30` |
| Correlation (custom path) | `python src/main.py -i /path/to/data -c` |
| Visualizations (default) | `python src/main.py -f` |
| Visualizations (custom) | `python src/main.py -f -p 8000 -s 0 1 2` |
| ML (all) | `python src/main.py -m` |
| ML (custom) | `python src/main.py -m -d 30 -mo knn svm xgboost -cv 10` |
| ML (3-class) | `python src/main.py -m -l 3class` |
| FFT (all) | `python src/main.py --fft` |
| FFT (custom) | `python src/main.py --fft -d 30 120 --fft-max-pairs 1000` |
| Predict (Pavia) | `python src/main.py --predict --pavia` |
| Predict (custom CSV) | `python src/main.py --predict --test-data t.csv --test-labels t_lab.csv` |

---

## Data Flow Diagram

```
ECG DATASET (WESAD)
        │
        ├──────────────────────┬──────────────────────┬─────────────────┬────────────
        │                      │                      │                 │
        ▼                      ▼                      ▼                 ▼
   ┌──────────────┐     ┌──────────────┐     ┌─────────────────┐  ┌─────────────┐
   │ CORRELATION  │     │ VISUALIZATION│     │ ML TRAINING     │  │ FFT ANALYSIS│
   │ RELIABILITY  │     └──────────────┘     └─────────────────┘  └─────────────┘
   └──────────────┘              │                     │                 │
        │                        │                     │                 │
        │ 1. Extract HRV         │ 1. Load Signal      │ 1. Extract HRV  │ 1. Extract
        │    Features            │ 2. Create Chunks    │    Features     │    chunks
        │ 2. Pair small/large    │ 3. Plot with        │ 2. Normalize    │ 2. FFT each
        │    windows             │    Labels           │ 3. k-fold / LOSO│ 3. Mean spectra
        │ 3. Compute r/ICC/MAE   │ 4. Export PNG       │ 4. Train Models │ 4. Cosine sim
        │ 4. Export tables/plots │                     │ 5. Save .pkl    │    cross/within
        │                        │                     │                 │
        ▼                        ▼                     ▼                 ▼
   ┌─────────────┐         ┌──────────────┐      ┌────────────────┐  ┌───────────────┐
   │CSV + Plots  │         │ PNG Files    │      │CSV + Saved      │  │CSV + Plots    │
   │(r,ICC,MAE)  │         │(Signals)     │      │Models (.pkl)    │  │(Spectra, Cos)│
   └─────────────┘         └──────────────┘      └────────────────┘  └───────────────┘
        │                                                            │
        └────────────────────── PREDICTION (--predict) ──────────────┘
                                │ loads saved models +
                                │ test data (WESAD / Pavia / CSV)
                                ▼
                          ┌──────────────┐
                          │Predictions + │
                          │metrics plots │
                          └──────────────┘
```

---

## Label Flow

```
WESAD raw labels (1–4):
  1 Baseline · 2 Stress · 3 Amusement · 4 Meditation

Binary (-l binary, default):   1→0  2→1  3→0  4→0   → {No Stress, Stress}
3-class (-l 3class):           1→0  2→1  3→0  4→2   → {No Stress/Amusement, Stress, Meditation}
```

---

## Decision Tree: Which Command to Use?

```
What do you want to do?
│
├─ "How reliable are HRV features across recording durations?"
│  └─ USE: python src/main.py -c [options]
│     └─ Generates cross-duration r / ICC / MAE analysis
│
├─ "Inspect the raw ECG signals?"
│  └─ USE: python src/main.py -f [options]
│     └─ Creates visualization plots
│
├─ "Train and evaluate stress classifiers?"
│  └─ USE: python src/main.py -m [options]
│     └─ Performs ML training with CV and saves models
│
├─ "Analyze frequency content / class separability?"
│  └─ USE: python src/main.py --fft [options]
│     └─ FFT spectra + cosine similarity
│
├─ "Predict stress on new data (Pavia / CSV / WESAD)?"
│  └─ USE: python src/main.py --predict [options]
│     └─ Uses saved models
│
└─ "Not sure where to start?"
   └─ RUN: python src/main.py --help
      └─ Shows comprehensive help
```

---

## Parameter Combinations

### For Correlation Analysis

```
Features × Duration-pairs = Total Analyses

Examples:
- 1 feature  × 1 pair (30s vs 120s) = 1 analysis
- 8 features × 2 pairs (30-120, 30-300) = 16 analyses (all)
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
- 1 model  × 1 duration  × 5 folds  = 5 CV runs
- 7 models × 3 durations × 5 folds  = 105 CV runs (all)
```

### For FFT

```
Durations × Comparison Types (cross/within-stress/within-non-stress)
```

---

## Output Directories

| Command | Default Output |
|---------|----------------|
| `-c` | `results/correlation_figures/` |
| `-f` | `results/signal_plots/` |
| `-m` | `results/ml_results/` (with `saved_models/`) |
| `--fft` | `results/fft_analysis/` |
| `--predict` | `results/predictions/` |

---

## Tips & Tricks

### Tip 1: Start Simple

```bash
python src/main.py -c
python src/main.py -f
python src/main.py -m -d 30 -mo knn
```

### Tip 2: For Prediction, Train Models First

```bash
python src/main.py -m -d 30 -mo knn svm random_forest
python src/main.py --predict -d 30 --pavia
```

### Tip 3: Organize Outputs

```bash
python src/main.py -c -o ./results/rel
python src/main.py -m -o ./results/ml
python src/main.py --fft -o ./results/fft
python src/main.py --predict -o ./results/pred
```

### Tip 4: Combine with Piping

```bash
python src/main.py -m -d 30 > ml_training.log 2>&1
```