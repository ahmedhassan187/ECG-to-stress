# CLI Quick Reference

## Common Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i`, `--input` | Path to the WESAD dataset directory | `data/WESAD` |
| `-d`, `--dataset` | Dataset durations in seconds | `30 120 300` |
| `-l`, `--labels` | Label scheme: `binary` or `3class` | `binary` |
| `-o`, `--output` | Custom output directory | Varies by command |

**Label schemes**
- `binary` (default): 1→0 (No Stress), 2→1 (Stress), 3→0, 4→0
- `3class`: 1→0 (No Stress/Amusement), 2→1 (Stress), 3→0, 4→2 (Meditation)

---

## Correlation Analysis (`-c` / `--corr`)

```bash
# All features, all durations
python src/main.py -c

# Specific features / durations
python src/main.py -c -f mean_rr mean_hr sdnn -d 30 120

# 3-class or binary label scheme
python src/main.py -c -l 3class

# Cross-duration reliability metrics (Pearson r, ICC, MAE)
python src/main.py -c --by-condition

# Custom dataset path
python src/main.py -i /path/to/WESAD -c
```

**Features**: `mean_rr`, `mean_hr`, `sdnn`, `rmssd`, `pnn50`, `lf_power`, `hf_power`, `lf_hf_ratio`

**Outputs**: `results/correlation_figures/` — `cross_duration_comparison.csv`, `comparison_table_{r|icc|mae}.csv`, heatmaps, bar charts, styled tables

---

## Full Signal Visualization (`-f` / `--full`)

```bash
# Default: 5000 points per chunk, all subjects
python src/main.py -f

# Custom chunk size / specific subjects
python src/main.py -f -p 10000 -s 0 1 2

# Custom output
python src/main.py -f -p 7500 -o ./my_plots
```

**Options**: `-p` / `--points` (default `5000`), `-s` / `--subjects` (default all)

**Outputs**: `results/signal_plots/` — ECG waveform PNGs colored by label

---

## Machine Learning Training (`-m` / `--ml`)

```bash
# All models, all durations
python src/main.py -m

# Specific durations / models
python src/main.py -m -d 30 120 -mo knn svm xgboost

# Custom CV folds
python src/main.py -m -cv 10

# 3-class classification
python src/main.py -m -l 3class
```

**Models**: `knn`, `svm`, `decision_tree`, `random_forest`, `gradient_boosting`, `logistic_regression`, `xgboost`

**Outputs**: `results/ml_results/` — `ml_results_<dur>s_<mode>.csv`, `saved_models/*.pkl` + `.meta.pkl`

---

## FFT Frequency Analysis (`--fft`)

```bash
# All durations (30/120/300 s)
python src/main.py --fft

# Specific durations
python src/main.py --fft -d 30 120

# More cosine-similarity pairs
python src/main.py --fft --fft-max-pairs 1000
```

**Options**: `--fft-max-pairs` (default `500`), `--fft-freq-max` (default `40.0` Hz)

**Outputs**: `results/fft_analysis/` — spectra overlays, cosine-similarity KDE grids, `fft_cosine_summary.csv`

---

## Prediction Mode (`--predict`)

```bash
# Default: WESAD subject 0 (test data) with saved 30s models
python src/main.py --predict

# Specific durations
python src/main.py --predict -d 30

# Pavia HRV data (default folder or custom)
python src/main.py --predict --pavia
python src/main.py --predict --pavia /custom/path

# Custom CSV test data (optionally with labels)
python src/main.py --predict --test-data t.csv --test-labels t_labels.csv

# Custom model directory
python src/main.py --predict --model-dir results/ml_results/saved_models
```

**Options**: `--model-dir` (default `../results/ml_results/saved_models`), `--test-data`, `--test-labels`, `--pavia`

**Outputs**: `results/predictions/` — `predictions_<dur>s.csv`, metrics + comparison figures

---

## Complete Pipeline

```bash
# Reliability analysis → visualization → ML → FFT → prediction
python src/main.py -c
python src/main.py -f -p 5000
python src/main.py -m
python src/main.py --fft
python src/main.py --predict
```

## Help

```bash
python src/main.py --help
```