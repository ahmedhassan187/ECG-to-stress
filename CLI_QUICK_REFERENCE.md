# CLI Quick Reference

## Dataset Path (`-i` / `--input`)
Specify custom WESAD dataset path (default: `data/WESAD`):
```bash
python src/main.py -i /path/to/WESAD -c
python src/main.py -i ./my_dataset -f
python src/main.py -i data/WESAD -m
```

---

## Correlation Analysis (`-c` / `--corr`)
```bash
# All features, all durations
python src/main.py -c

# Specific features
python src/main.py -c -f mean_rr mean_hr sdnn

# Specific durations
python src/main.py -c -d 30 120

# Both specific
python src/main.py -c -f mean_rr -d 30 120 -o ./results

# Custom dataset path
python src/main.py -i /path/to/WESAD -c
```

**Available Features**: `mean_rr`, `mean_hr`, `sdnn`, `rmssd`, `pnn50`, `lf_power`, `hf_power`, `lf_hf_ratio`

---

## Full Signal Visualization (`-f` / `--full`)
```bash
# Default: 5000 points per chunk, all subjects
python src/main.py -f

# Custom chunk size
python src/main.py -f -p 10000

# Specific subjects
python src/main.py -f -s 0 1 2 -p 5000

# Custom output
python src/main.py -f -p 7500 -o ./my_plots

# Custom dataset path
python src/main.py -i /path/to/WESAD -f
```

**Points Range**: 2000-15000 (2000 for detail, 10000+ for overview)

---

## Machine Learning Training (`-m` / `--ml`)
```bash
# All models, all durations
python src/main.py -m

# Specific durations (30s, 120s, 300s)
python src/main.py -m -d 30 120

# Specific models
python src/main.py -m -mo knn svm xgboost

# Both specific
python src/main.py -m -d 30 -mo random_forest gradient_boosting

# Custom CV folds
python src/main.py -m -cv 10

# Custom output
python src/main.py -m -o ./ml_results

# Custom dataset path
python src/main.py -i /path/to/WESAD -m
```

**Available Models**: `knn`, `svm`, `decision_tree`, `random_forest`, `gradient_boosting`, `logistic_regression`, `xgboost`

**Durations**: 30, 120, 300 (seconds)

---

## Complete Pipeline
```bash
python src/main.py -c -f mean_rr mean_hr sdnn
python src/main.py -f -p 5000
python src/main.py -m -d 30
```

---

## Help
```bash
python src/main.py --help
python src/main.py -c --help
python src/main.py -f --help
python src/main.py -m --help