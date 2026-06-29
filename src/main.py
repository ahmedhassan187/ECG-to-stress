"""
ECG-to-Stress Analysis CLI
Command-line interface for WESAD dataset analysis, feature extraction, correlation analysis,
machine learning model training, and FFT frequency analysis.

Label mapping (WESAD):
    1 → Baseline   → Non-Stress (binary 0)
    2 → Stress     → Stress     (binary 1)
    3 → Amusement  → Non-Stress (binary 0)
    4 → Meditation → Stress     (binary 1)

Usage Examples:
    py src/main.py --help                           # Show help message
    py src/main.py -i data/WESAD -c                 # Run correlation analysis with custom dataset path
    py src/main.py --input /path/to/data -c         # Same with long flag
    py src/main.py -c                               # Run correlation analysis (default dataset path)
    py src/main.py --corr --features feature1 feature2  # Run correlation on specific features
    py src/main.py -f                               # Plot full ECG signals (default 5000 points)
    py src/main.py --full -p 10000                  # Plot with 10000 points per chunk
    py src/main.py -m                               # Train all models on all datasets
    py src/main.py --ml -d 30 120                   # Train models on 30s and 120s datasets
    py src/main.py -m -mo knn svm random_forest     # Train specific models on all datasets
    py src/main.py --fft                            # Run FFT analysis (30s/120s/300s chunks)
    py src/main.py --fft -d 30 120                  # FFT on specific durations only
    py src/main.py --fft --fft-max-pairs 1000       # More cosine-similarity pairs
    
    # PREDICTION MODE
    py src/main.py -p predict -d 30                 # Predict on test data using 30s models
    py src/main.py --predict -i data/test_data      # Predict on custom test data
    py src/main.py -p predict -mo knn random_forest # Use specific models for prediction
"""

import argparse
import sys
import os
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, ConfusionMatrixDisplay, accuracy_score, f1_score, classification_report
import joblib

# Automatically change to the ECG-to-stress directory if running from parent
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent  # ECG-to-stress folder
if Path.cwd() != project_root:
    os.chdir(project_root)

# Add src directory to path for imports
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

# Import classes from src
try:
    from data import Data
    from features import Features
    from visualization import Visualization
    from correlation import Correlation
    from ml import ML
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("   Make sure you're in the ECG-to-stress project directory:")
    print("   cd G:\\Master\\Thesis\\FLT\\Code\\ECG-to-stress")
    print("   py src/main.py --help")
    sys.exit(1)

# Try to import xgboost (optional dependency)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')


def parse_arguments():
    """
    Parse command-line arguments using argparse.

    Returns:
        Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog='py src/main.py',
        description='ECG-to-Stress Analysis Tool - WESAD Dataset Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Correlation Analysis
  py src/main.py -c                              # All features
  py src/main.py --corr --features mean_rr mean_hr  # Specific features
  py src/main.py -i data/WESAD -c                # Custom dataset path

  # Full Signal Visualization
  py src/main.py -f                              # Default 5000 points per chunk
  py src/main.py --full -p 10000                 # 10000 points per chunk

  # Machine Learning Models
  py src/main.py -m                              # All models, all datasets
  py src/main.py -m -d 30 120 300                # Specific datasets (30s, 120s, 300s)
  py src/main.py -m -mo knn svm xgboost          # Specific models
  py src/main.py -m -d 30 -mo random_forest      # Specific dataset + models

  # FFT Frequency Analysis
  py src/main.py --fft                           # All durations (30s/120s/300s)
  py src/main.py --fft -d 30 120                 # FFT for 30s and 120s only
  py src/main.py --fft --fft-max-pairs 1000      # More cosine-similarity pairs
  
  # PREDICTION MODE
  py src/main.py -p predict -d 30                # Predict on test data using 30s models
  py src/main.py --predict -i data/test_data     # Predict on custom test data
  py src/main.py -p predict -mo knn random_forest # Use specific models for prediction
        """
    )

    # Create mutually exclusive group for commands
    commands = parser.add_mutually_exclusive_group()

    # ========== CORRELATION COMMAND ==========
    commands.add_argument(
        '-c', '--corr',
        action='store_true',
        dest='correlation',
        help='Generate correlation analysis of HRV features'
    )

    # ========== FULL SIGNAL COMMAND ==========
    commands.add_argument(
        '-f', '--full',
        action='store_true',
        dest='full_signal',
        help='Plot full ECG signals with adjustable chunk size'
    )

    # ========== ML MODELS COMMAND ==========
    commands.add_argument(
        '-m', '--ml',
        action='store_true',
        dest='ml_training',
        help='Train machine learning models with cross-validation'
    )

    # ========== FFT ANALYSIS COMMAND ==========
    commands.add_argument(
        '--fft',
        action='store_true',
        dest='fft_analysis',
        help='Run FFT frequency analysis with cosine similarity (stress vs non-stress)'
    )

    # ========== PREDICTION COMMAND ==========
    commands.add_argument(
        '--predict',
        action='store_true',
        dest='prediction',
        help='Load trained models and make predictions on test data'
    )

    # ========== CORRELATION OPTIONS ==========
    corr_group = parser.add_argument_group('Correlation Analysis Options')
    corr_group.add_argument(
        '--features',
        nargs='+',
        type=str,
        default=['all'],
        help='Features to analyze (default: all). Available: mean_rr, mean_hr, sdnn, rmssd, pnn50, lf_power, hf_power, lf_hf_ratio'
    )

    # ========== SHARED OPTIONS ==========
    shared_group = parser.add_argument_group('Common Options')
    shared_group.add_argument(
        '-i', '--input',
        type=str,
        default=None,
        help='Path to the WESAD dataset directory (default: data/WESAD relative to project root)'
    )
    shared_group.add_argument(
        '-d', '--dataset',
        nargs='+',
        type=int,
        default=[30, 120, 300],
        help='Dataset durations in seconds (default: 30 120 300)'
    )
    shared_group.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory (default: results/{correlation_figures|signal_plots|ml_results|fft_analysis})'
    )

    # ========== VISUALIZATION OPTIONS ==========
    viz_group = parser.add_argument_group('Full Signal Visualization Options')
    viz_group.add_argument(
        '-p', '--points',
        type=int,
        default=5000,
        help='Number of points per plot chunk (default: 5000)'
    )
    viz_group.add_argument(
        '-s', '--subjects',
        nargs='+',
        type=int,
        default=None,
        help='Specific subject IDs to plot (default: all subjects)'
    )

    # ========== ML OPTIONS ==========
    ml_group = parser.add_argument_group('Machine Learning Options')
    ml_group.add_argument(
        '-mo', '--models',
        nargs='+',
        type=str,
        default=['knn', 'svm', 'decision_tree', 'random_forest', 'gradient_boosting', 'logistic_regression', 'xgboost'],
        help='Models to train. Available: knn, svm, decision_tree, random_forest, gradient_boosting, logistic_regression, xgboost'
    )
    ml_group.add_argument(
        '-cv', '--cross-val',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )

    # ========== FFT OPTIONS ==========
    fft_group = parser.add_argument_group('FFT Analysis Options')
    fft_group.add_argument(
        '--fft-max-pairs',
        type=int,
        default=500,
        dest='fft_max_pairs',
        help='Maximum random pairs for cosine similarity (default: 500)'
    )
    fft_group.add_argument(
        '--fft-freq-max',
        type=float,
        default=40.0,
        dest='fft_freq_max',
        help='Upper frequency limit for FFT plots in Hz (default: 40.0)'
    )

    # ========== PREDICTION OPTIONS ==========
    pred_group = parser.add_argument_group('Prediction Options')
    pred_group.add_argument(
        '--model-dir',
        type=str,
        default='../results/ml_results/saved_models',
        help='Directory containing trained models (default: ../results/ml_results/saved_models)'
    )
    pred_group.add_argument(
        '--test-data',
        type=str,
        default=None,
        help='Path to test data CSV file with features (if not using WESAD dataset)'
    )
    pred_group.add_argument(
        '--test-labels',
        type=str,
        default=None,
        help='Path to test labels CSV file (if using separate label file)'
    )
    pred_group.add_argument(
        '--pavia',
        action='store_true',
        default=False,
        help='Use Pavia HRV data (data/pavia_features.csv + data/pavia_labels.csv) for prediction. '
             'Automatically maps Pavia column names (HR,SDNN,rMSSD,...) to standard feature names '
             'and filters out empty rows.'
    )


    
    return parser


def _get_dataset_path(args):
    """
    Resolve the dataset path from the -i argument or use the default.
    
    Parameters:
        args: parsed arguments
    
    Returns:
        Path: resolved path to the WESAD dataset directory
    """
    if args.input is not None:
        # If user provided a path, resolve it relative to the project root if relative
        input_path = Path(args.input)
        if not input_path.is_absolute():
            return project_root / input_path
        return input_path
    else:
        # Default path: data/WESAD relative to project root
        return project_root / "data" / "WESAD"


def _get_model(model_name):
    """Create an sklearn model instance by name."""
    models = {
        'knn': KNeighborsClassifier(n_neighbors=5),
        'svm': SVC(kernel='rbf', probability=True, random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    }
    if 'xgboost' == model_name:
        if XGBOOST_AVAILABLE:
            return XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        else:
            return None
    return models.get(model_name)


def _extract_features_for_duration(ecgs, labels, duration, feature_extractor):
    """
    Extract HRV features for all subjects at a given window duration.

    For each subject we keep a per-subject list of feature dicts so that
    values from the smaller window can later be matched against the larger
    window (e.g. four 30s chunks per one 120s chunk).

    Parameters:
        ecgs: list of ECG arrays (one per subject)
        labels: list of label arrays (one per subject)
        duration: chunk size in seconds
        feature_extractor: Features instance

    Returns:
        per_subject_features: list (per subject) of list (per chunk) of dicts
        per_subject_labels:   list (per subject) of list (per chunk) of labels
    """
    chunk_size = duration * 700  # 700 Hz sampling rate
    per_subject_features = []
    per_subject_labels = []

    for ecg, label in zip(ecgs, labels):
        valid_mask = np.isin(label, [1, 2, 3, 4])
        valid_ecg = ecg[valid_mask]
        valid_label = label[valid_mask]

        subj_feats = []
        subj_labels = []
        for i in range(0, len(valid_ecg) - chunk_size + 1, chunk_size):
            chunk = valid_ecg[i:i + chunk_size]
            chunk_label = int(valid_label[i])
            try:
                feats = {
                    'mean_rr': feature_extractor.get_mean_rr(chunk),
                    'mean_hr': feature_extractor.get_mean_hr(chunk),
                    'sdnn': feature_extractor.get_sdnn(chunk),
                    'rmssd': feature_extractor.get_rmssd(chunk),
                    'pnn50': feature_extractor.get_pnn50(chunk),
                    'lf_power': feature_extractor.get_lf_power(chunk),
                    'hf_power': feature_extractor.get_hf_power(chunk),
                    'lf_hf_ratio': feature_extractor.get_lf_hf_ratio(chunk),
                }
                subj_feats.append(feats)
                subj_labels.append(chunk_label)
            except Exception:
                continue

        per_subject_features.append(subj_feats)
        per_subject_labels.append(subj_labels)

    return per_subject_features, per_subject_labels


def _build_pairwise_arrays(per_subject_features_small,
                           per_subject_features_large,
                           feat_name, ratio):
    """
    Align values from a smaller window with a larger window by repetition.

    For each subject, group the small-window values in blocks of ``ratio``
    (e.g. four 30s values per one 120s value) and pair them with the
    corresponding large-window value. The large-window value is repeated
    ``ratio`` times so both arrays have the same length.

    Example:
        30s values for one subject : [1, 2, 3, 4]
        120s values for same subject: [2, 5, 1, 3]
        ratio = 120 / 30 = 4

        small = [1, 2, 3, 4]
        large (repeated) = [2, 2, 2, 2, 5, 5, 5, 5, 1, 1, 1, 1, 3, 3, 3, 3]

        Only the first ``ratio`` small values are kept (those matching the
        first 120s chunk), giving [1, 2, 3, 4] vs [2, 2, 2, 2] for that
        subject. The process repeats for every 120s chunk.

    Parameters:
        per_subject_features_small: per-subject list of per-chunk feature dicts
        per_subject_features_large: per-subject list of per-chunk feature dicts
        feat_name: feature to extract
        ratio: large_duration / small_duration (must be integer)

    Returns:
        small_arr: 1D numpy array of small-window values
        large_arr: 1D numpy array of large-window values, each repeated `ratio` times
    """
    small_vals = []
    large_vals = []

    for subj_small, subj_large in zip(per_subject_features_small,
                                       per_subject_features_large):
        n_large = len(subj_large)
        for j in range(n_large):
            start = j * ratio
            end = start + ratio
            if end > len(subj_small):
                break
            large_val = subj_large[j].get(feat_name, np.nan)
            for k in range(start, end):
                sv = subj_small[k].get(feat_name, np.nan)
                small_vals.append(sv)
                large_vals.append(large_val)

    return np.array(small_vals, dtype=float), np.array(large_vals, dtype=float)


def _compute_comparison_metrics(small_arr, large_arr, corr_analyzer):
    """
    Compute Pearson R, ICC2 and MAE between two aligned feature arrays.
    Returns a dict with metric values or NaN on failure.
    """
    mask = ~(np.isnan(small_arr) | np.isnan(large_arr))
    s = small_arr[mask]
    l = large_arr[mask]
    if len(s) < 2:
        return {'r': np.nan, 'icc': np.nan, 'mae': np.nan, 'n': int(len(s))}

    try:
        r, _ = corr_analyzer.get_r(s, l, method='pearson')
    except Exception:
        r = np.nan
    try:
        icc, _ = corr_analyzer.get_icc(s, l, icc_type='ICC2')
    except Exception:
        icc = np.nan
    try:
        mae = float(mean_absolute_error(s, l))
    except Exception:
        mae = np.nan

    return {'r': r, 'icc': icc, 'mae': mae, 'n': int(len(s))}


def _make_pairwise_table(durations, comparison_table, metric):
    """
    Build an NxN comparison table (rows = duration, cols = duration) for the
    given metric ('r', 'icc' or 'mae').

    Each off-diagonal cell is the mean of that metric averaged across all HRV
    features for that duration pair.  The diagonal is 1.0 (R / ICC) or 0.0 (MAE).
    """
    n    = len(durations)
    diag = 0.0 if metric == 'mae' else 1.0
    table = pd.DataFrame(index=durations, columns=durations, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                table.iloc[i, j] = diag
            else:
                small = durations[min(i, j)]
                large = durations[max(i, j)]
                feat_dict = comparison_table.get((small, large), {})
                # feat_dict: {feature_name: {'r', 'icc', 'mae', 'n'}}
                vals = [v[metric] for v in feat_dict.values()
                        if isinstance(v, dict) and not np.isnan(v.get(metric, np.nan))]
                table.iloc[i, j] = np.mean(vals) if vals else np.nan

    table.index.name   = 'Duration (s)'
    table.columns.name = 'Duration (s)'
    return table


def _make_pairwise_table_per_feature(durations, comparison_table, metric, feature):
    """
    Same as _make_pairwise_table but for a single HRV feature instead of
    averaging across all features.
    """
    n    = len(durations)
    diag = 0.0 if metric == 'mae' else 1.0
    table = pd.DataFrame(index=durations, columns=durations, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                table.iloc[i, j] = diag
            else:
                small = durations[min(i, j)]
                large = durations[max(i, j)]
                feat_dict = comparison_table.get((small, large), {})
                val = feat_dict.get(feature, {}).get(metric, np.nan)
                table.iloc[i, j] = val

    table.index.name   = 'Duration (s)'
    table.columns.name = 'Duration (s)'
    return table


def _plot_pairwise_table(table, metric, output_path, title):
    """
    Save a heat-map figure of a pairwise comparison table.
    """
    data = table.astype(float).values
    if np.all(np.isnan(data)):
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, cmap='viridis', aspect='auto',
                   vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.set_xticks(range(len(table.columns)))
    ax.set_yticks(range(len(table.index)))
    ax.set_xticklabels([f"{c}s" for c in table.columns])
    ax.set_yticklabels([f"{r}s" for r in table.index])
    ax.set_xlabel('Large window')
    ax.set_ylabel('Small window')
    ax.set_title(title)
    mean_val = np.nanmean(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                        color='white' if val < mean_val else 'black', fontsize=10)
    fig.colorbar(im, ax=ax, label=metric.upper())
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_per_comparison_bars(comparison_table, metric, output_path):
    """
    Save a bar chart of per-comparison values for a given metric,
    averaged across all HRV features for each duration pair.
    """
    keys   = list(comparison_table.keys())
    labels = [f"{a}s vs {b}s" for a, b in keys]
    vals   = []
    for k in keys:
        feat_dict = comparison_table[k]
        pair_vals = [v[metric] for v in feat_dict.values()
                     if isinstance(v, dict) and not np.isnan(v.get(metric, np.nan))]
        vals.append(np.mean(pair_vals) if pair_vals else np.nan)

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 5))
    bars = ax.bar(labels, vals, color='steelblue', edgecolor='black')
    ax.set_ylabel(f'{metric.upper()} (mean across features)')
    ax.set_title(f"{metric.upper()} for each duration pair (mean across HRV features)")
    finite = [v for v in vals if not np.isnan(v)]
    upper  = max(finite + [1]) * 1.15
    ax.set_ylim(0, upper)
    ax.tick_params(axis='x', rotation=30)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                    ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_metric_summary_tables(durations, comparison_table, feature_list, output_path):
    """
    Render three styled NxN table images — one per metric (ICC, R, MAE) —
    and one combined figure that shows all three side-by-side.

    Layout of each NxN table (example with 30 / 120 / 300 s):

              ICC
           30    120   300
      30 [  1   val  val ]
     120 [ val   1   val ]
     300 [ val  val   1  ]

    Diagonal cells are grey (self-comparison sentinel: 1.0 for R/ICC, 0.0 for MAE).
    Off-diagonal cells are colour-coded with RdYlGn (green = good agreement).
    For MAE the colour map is inverted (lower = better).

    Additionally saves per-feature tables so the user can drill down per feature.

    Output files
    ─────────────
      corr_summary_table_<metric>.png     – standalone NxN table per metric
      corr_summary_tables_combined.png    – all three tables in one figure
      corr_per_feature_<feat>_<metric>.png – NxN table per feature × metric
    """
    METRICS = [
        ('icc', 'ICC',  'RdYlGn',  False),
        ('r',   'R',    'RdYlGn',  False),
        ('mae', 'MAE',  'RdYlGn_r', True),   # reversed: lower MAE = better
    ]

    def _render_table_ax(ax, table, metric_label, cmap_name, invert, title):
        """Draw a single NxN styled table onto ax."""
        data = table.astype(float).values
        n    = data.shape[0]
        cols = [f'{c}s' for c in table.columns]
        rows = [f'{r}s' for r in table.index]

        cmap      = plt.get_cmap(cmap_name)
        diag_val  = 0.0 if invert else 1.0   # sentinel on diagonal

        # colour each cell
        cell_colors = []
        for i in range(n):
            row_c = []
            for j in range(n):
                if i == j:
                    row_c.append((0.88, 0.88, 0.88, 1.0))   # grey diagonal
                else:
                    v = data[i, j]
                    if np.isnan(v):
                        row_c.append((1.0, 1.0, 1.0, 1.0))  # white = no data
                    else:
                        # normalise into [0,1] for colour map
                        vmin = np.nanmin(data[~np.eye(n, dtype=bool)])
                        vmax = np.nanmax(data[~np.eye(n, dtype=bool)])
                        norm = (v - vmin) / (vmax - vmin + 1e-9)
                        row_c.append(cmap(norm))
            cell_colors.append(row_c)

        # cell text
        cell_text = []
        for i in range(n):
            row_t = []
            for j in range(n):
                v = data[i, j]
                if i == j:
                    row_t.append('—')
                elif np.isnan(v):
                    row_t.append('N/A')
                else:
                    row_t.append(f'{v:.3f}')
            cell_text.append(row_t)

        ax.axis('off')
        tbl = ax.table(
            cellText=cell_text,
            rowLabels=rows,
            colLabels=cols,
            cellColours=cell_colors,
            cellLoc='center',
            loc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.4, 2.0)

        # bold the header row and index column
        for (r, c), cell in tbl.get_celld().items():
            if r == 0 or c == -1:
                cell.set_text_props(fontweight='bold')

        ax.set_title(title, fontsize=13, fontweight='bold', pad=14)

    # ── 1. individual table images ────────────────────────────────────────────
    for metric_key, metric_label, cmap_name, invert in METRICS:
        tbl = _make_pairwise_table(durations, comparison_table, metric_key)

        cell_h  = max(0.6, 0.5 * len(durations))
        fig_h   = max(3.5, cell_h * len(durations) + 1.5)
        fig_w   = max(5.0, 1.6 * len(durations) + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        _render_table_ax(
            ax, tbl, metric_label, cmap_name, invert,
            title=f'{metric_label} – Mean across HRV features\n'
                  f'(rows = reference duration, cols = comparison duration)'
        )
        fig.tight_layout()
        p = output_path / f'corr_summary_table_{metric_key}.png'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✓ Saved {metric_label} summary table: {p}")

    # ── 2. combined figure (ICC | R | MAE side-by-side) ───────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(max(18, 6 * len(durations)), max(4, 1.8 * len(durations))))
    for ax, (metric_key, metric_label, cmap_name, invert) in zip(axes, METRICS):
        tbl = _make_pairwise_table(durations, comparison_table, metric_key)
        _render_table_ax(ax, tbl, metric_label, cmap_name, invert,
                         title=metric_label)

    fig.suptitle('Cross-Duration Agreement Summary (mean across all HRV features)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    p = output_path / 'corr_summary_tables_combined.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved combined summary tables: {p}")

    # ── 3. per-feature breakdown tables ──────────────────────────────────────
    print(f"   📋 Saving per-feature tables for {len(feature_list)} features × 3 metrics...")
    for feature in feature_list:
        fig, axes = plt.subplots(
            1, 3,
            figsize=(max(18, 6 * len(durations)), max(4, 1.8 * len(durations)))
        )
        for ax, (metric_key, metric_label, cmap_name, invert) in zip(axes, METRICS):
            tbl = _make_pairwise_table_per_feature(
                durations, comparison_table, metric_key, feature
            )
            _render_table_ax(ax, tbl, metric_label, cmap_name, invert,
                             title=metric_label)

        fig.suptitle(f'Cross-Duration Agreement – {feature}',
                     fontsize=13, fontweight='bold', y=1.02)
        fig.tight_layout()
        p = output_path / f'corr_per_feature_{feature}.png'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"   ✓ Saved per-feature tables → corr_per_feature_<feature>.png")


def _plot_per_feature_bars(rows_for_csv, feat, metric, output_path):
    """
    Save a per-feature bar chart of metric values across duration pairs.
    """
    pairs = sorted({(r['small_duration_s'], r['large_duration_s'])
                    for r in rows_for_csv if r['feature'] == feat})
    if not pairs:
        return
    labels = [f"{a}s vs {b}s" for a, b in pairs]
    vals = [next(r[metric] for r in rows_for_csv
                 if r['feature'] == feat and r['small_duration_s'] == a
                 and r['large_duration_s'] == b) for a, b in pairs]
    fig, ax = plt.subplots(figsize=(max(6, len(pairs) * 1.2), 4.5))
    bars = ax.bar(labels, vals, color='teal', edgecolor='black')
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{feat} — {metric.upper()} by duration pair")
    finite = [v for v in vals if not np.isnan(v)]
    ax.set_ylim(0, max(finite + [1]) * 1.15)
    ax.tick_params(axis='x', rotation=30)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                    ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_correlation_analysis(args):
    """
    Execute correlation analysis on HRV features.

    The analysis runs in two stages:

    1.  Per-duration extraction — features are computed for every requested
        window size (e.g. 30s, 120s, 300s) and the original chunks are kept
        separately so smaller windows can later be mapped to larger windows
        via repetition (e.g. four 30s chunks per one 120s chunk).

    2.  Cross-duration comparison — for every ordered pair (small, large)
        we align values by repeating the large-window value to match the
        number of small chunks it covers, then compute Pearson R, ICC2 and
        MAE. Results are written to a CSV and visualised as a heat-map and
        a per-comparison bar chart.
    """
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    # Set default output directory if not specified
    output_dir = args.output or '../results/correlation_figures'

    # Initialize components
    data_loader = Data(fs=700)
    feature_extractor = Features(fs=700)
    corr_analyzer = Correlation()
    viz = Visualization()

    # Get dataset path
    dataset_path = _get_dataset_path(args)

    print(f"📂 Loading dataset from: {dataset_path}")

    # Load data
    try:
        ecgs, labels = data_loader.read_dataset(str(dataset_path))
        print(f"✓ Loaded {len(ecgs)} subjects")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_path}")

    # Make sure durations are sorted and unique for the pairwise mapping
    durations = sorted(set(args.dataset))
    print(f"📏 Window durations: {durations}")

    # Feature list
    all_feature_names = ['mean_rr', 'mean_hr', 'sdnn', 'rmssd',
                         'pnn50', 'lf_power', 'hf_power', 'lf_hf_ratio']
    if 'all' in args.features:
        feature_list = all_feature_names
    else:
        feature_list = [f for f in args.features if f in all_feature_names]

    # ------------------------------------------------------------------
    # Stage 1: extract features for every duration
    # ------------------------------------------------------------------
    per_duration_features = {}
    per_duration_labels   = {}
    per_duration_dfs      = {}

    for duration in durations:
        print(f"\n📊 Processing {duration}s chunks...")

        per_subj_feats, per_subj_labels = _extract_features_for_duration(
            ecgs, labels, duration, feature_extractor
        )

        if not any(len(s) for s in per_subj_feats):
            print(f"⚠️  No features extracted for {duration}s duration")
            continue

        per_duration_features[duration] = per_subj_feats
        per_duration_labels[duration] = per_subj_labels

        # Flatten for the per-duration outputs (legacy behaviour)
        flat_rows = []
        for subj_feats, subj_labels in zip(per_subj_feats, per_subj_labels):
            for fdict, lbl in zip(subj_feats, subj_labels):
                row = dict(fdict)
                row['label'] = lbl
                flat_rows.append(row)
        df = pd.DataFrame(flat_rows)
        per_duration_dfs[duration] = df
        print(f"✓ Extracted {len(df)} chunks with 8 HRV features")

        # ---- per-duration outputs ----
        if len(feature_list) >= 2:
            print(f"📈 Generating correlation matrix heatmap for {len(feature_list)} features...")
            corr_features = [f for f in feature_list if f in df.columns]
            if len(corr_features) >= 2:
                fig, ax = viz.plot_features_corr_mat(df, feature_cols=corr_features)
                corr_path = output_path / f"correlation_matrix_{duration}s.png"
                fig.savefig(corr_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"✓ Saved: {corr_path}")

        if 'label' in df.columns:
            valid_features = [f for f in feature_list if df[f].notna().sum() >= 2]
            if len(valid_features) >= 2:
                df_clean = df.dropna(subset=valid_features, how='all').copy()
                if not df_clean.empty:
                    print(f"📈 Generating feature distribution plots for {len(valid_features)} features...")
                    df_clean['label'] = df_clean['label'].apply(lambda x: 1 if x in [2, 4] else 0)
                    fig, axes = viz.plot_features_dist(df_clean, feature_cols=valid_features)
                    dist_path = output_path / f"feature_distributions_{duration}s.png"
                    fig.savefig(dist_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"✓ Saved: {dist_path}")

        csv_path = output_path / f"features_{duration}s.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")

    # ------------------------------------------------------------------
    # Stage 2: pairwise comparison across durations
    # ------------------------------------------------------------------
    if len(durations) < 2:
        print("\nℹ️  Only one duration requested — skipping cross-duration comparison.")
        print("\n✅ Correlation analysis complete!")
        return

    print("\n" + "-" * 80)
    print("CROSS-DURATION COMPARISON (small vs large windows)")
    print("-" * 80)

    # comparison_table[(small, large)][feature] = {'r', 'icc', 'mae', 'n'}
    comparison_table = {}
    rows_for_csv = []

    for i, small in enumerate(durations):
        for j, large in enumerate(durations):
            if i >= j:
                continue
            ratio = large // small
            if large % small != 0:
                print(f"⚠️  Skipping {small}s vs {large}s (not an integer ratio)")
                continue
            if small not in per_duration_features or large not in per_duration_features:
                continue

            print(f"\n🔗 Comparing {small}s vs {large}s (ratio {ratio}:1)...")

            comparison_table[(small, large)] = {}

            for feat in feature_list:
                s_arr, l_arr = _build_pairwise_arrays(
                    per_duration_features[small],
                    per_duration_features[large],
                    feat, ratio,
                )
                metrics = _compute_comparison_metrics(s_arr, l_arr, corr_analyzer)
                comparison_table[(small, large)][feat] = metrics
                rows_for_csv.append({
                    'small_duration_s': small,
                    'large_duration_s': large,
                    'feature': feat,
                    'r': metrics['r'],
                    'icc': metrics['icc'],
                    'mae': metrics['mae'],
                    'n_samples': metrics['n'],
                })
                print(f"   • {feat:11s}  R={metrics['r']:.3f}  "
                      f"ICC={metrics['icc']:.3f}  MAE={metrics['mae']:.3f}  "
                      f"n={metrics['n']}")

    if not rows_for_csv:
        print("⚠️  No cross-duration comparisons produced.")
        print("\n✅ Correlation analysis complete!")
        return

    # ---- save the long-form CSV ----
    long_csv = output_path / "cross_duration_comparison.csv"
    pd.DataFrame(rows_for_csv).to_csv(long_csv, index=False)
    print(f"\n✓ Saved comparison CSV: {long_csv}")

    # ---- build per-metric NxN tables (rows = small, cols = large) ----
    for metric in ('r', 'icc', 'mae'):
        tbl = _make_pairwise_table(durations, comparison_table, metric)
        tbl_csv = output_path / f"comparison_table_{metric}.csv"
        tbl.to_csv(tbl_csv)
        print(f"✓ Saved {metric.upper()} table: {tbl_csv}")
        print(f"\n{metric.upper()} comparison (mean across features):")
        print(tbl.to_string())

        # heat-map for the table
        _plot_pairwise_table(
            tbl, metric,
            output_path / f"comparison_{metric}_heatmap.png",
            f"{metric.upper()} — pairwise duration comparison"
        )
        # per-comparison bar chart (one figure, all features, one metric)
        _plot_per_comparison_bars(
            comparison_table, metric,
            output_path / f"comparison_{metric}_bars.png"
        )

        # per-feature figure for the metric
        features_in_table = sorted({row['feature'] for row in rows_for_csv})
        for feat in features_in_table:
            _plot_per_feature_bars(
                rows_for_csv, feat, metric,
                output_path / f"{feat}_{metric}_comparison.png"
            )

    # ---- styled NxN summary table images (ICC / R / MAE) ----
    print("\n📋 Generating styled summary table images...")
    _plot_metric_summary_tables(durations, comparison_table, feature_list, output_path)

    print("\n✅ Correlation analysis complete!")


def run_full_signal_visualization(args):
    """
    Execute full ECG signal visualization with adjustable chunk size.
    
    Parameters:
        args: parsed arguments containing points per chunk and output settings
    """
    print("\n" + "="*80)
    print("FULL ECG SIGNAL VISUALIZATION")
    print("="*80)
    
    # Set default output directory if not specified
    output_dir = args.output or '../results/signal_plots'
    
    # Initialize components
    data_loader = Data(fs=700)
    viz = Visualization()
    
    # Get dataset path
    dataset_path = _get_dataset_path(args)
    
    print(f"📂 Loading dataset from: {dataset_path}")
    print(f"📊 Chunk size: {args.points} points per plot")
    
    # Load data
    try:
        ecgs, labels = data_loader.read_dataset(str(dataset_path))
        print(f"✓ Loaded {len(ecgs)} subjects")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_path}")
    # Determine which subject(s) to plot
    if args.subjects:
        # User-specified subjects
        subjects = args.subjects
        print(f"\U0001f3af Using specified subject(s): {subjects}")
    else:
        # Pick one random subject
        subjects = [random.randint(0, len(ecgs) - 1)]
        print(f"\U0001f3b2 No subject specified \u2014 randomly selected Subject {subjects[0]}")
    
    for idx in subjects:
        if idx >= len(ecgs):
            print(f"\u26a0\ufe0f  Subject {idx} not found (only {len(ecgs)} subjects available)")
            continue
        
        ecg = ecgs[idx]
        label = labels[idx]
        
        total_chunks = max(1, (len(ecg) + args.points - 1) // args.points)
        print(f"\n\U0001f4cc Processing Subject {idx} ({len(ecg)} samples = {total_chunks} chunks)\n")
        
        try:
            # Generate ALL chunks (no cap) for the full signal
            print(f"\U0001f4ca Generating all {total_chunks} plots for full signal...")
            figs = viz.plot_ecg_rolling(
                ecg=ecg,
                fs=700,
                chunk_size=args.points,
                max_chunks=None,  # No cap — plot the ENTIRE signal
                label=label,
                title=f"Subject {idx} - Full ECG Signal ({len(ecg)} samples)"
            )
            # Save each figure
            for fig_idx, fig in enumerate(figs):
                plot_path = output_path / f"subject_{idx:02d}_ecg_chunk_{fig_idx:04d}.png"
                fig.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                if (fig_idx + 1) % 10 == 0:
                    print(f"  Saved {fig_idx + 1}/{len(figs)} plots...")
            
            print(f"✓ Saved {len(figs)} plots for subject {idx}")
        except Exception as e:
            print(f"❌ Error processing subject {idx}: {e}")
    
    print("\n✅ ECG signal visualization complete!")


def _get_available_models(available_models):
    """Check which models are actually available (xgboost may not be installed)."""
    result = []
    for m in available_models:
        if m == 'xgboost' and not XGBOOST_AVAILABLE:
            continue
        result.append(m)
    return result


def run_ml_training(args):
    """
    Execute machine learning model training with cross-validation.
    
    Parameters:
        args: parsed arguments containing models, datasets, and CV settings
    """
    print("\n" + "="*80)
    print("MACHINE LEARNING MODEL TRAINING")
    print("="*80)
    
    # Set default output directory if not specified
    output_dir = args.output or '../results/ml_results'
    
    # Initialize components
    data_loader = Data(fs=700)
    feature_extractor = Features(fs=700)
    ml_evaluator = ML(random_state=42)
    
    # Get dataset path
    dataset_path = _get_dataset_path(args)
    
    print(f"📂 Loading dataset from: {dataset_path}")
    
    # Load data
    try:
        ecgs, labels = data_loader.read_dataset(str(dataset_path))
        print(f"✓ Loaded {len(ecgs)} subjects")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Validate and normalize model names
    available_models = [
        'knn', 'svm', 'decision_tree', 'random_forest', 
        'gradient_boosting', 'logistic_regression', 'xgboost'
    ]
    
    requested_models = [m.lower() for m in args.models]
    invalid_models = [m for m in requested_models if m not in available_models]
    
    if invalid_models:
        print(f"⚠️  Unknown models: {invalid_models}")
        print(f"   Available: {', '.join(available_models)}")
    
    models_to_train = [m for m in requested_models if m in available_models]
    models_to_train = _get_available_models(models_to_train)
    
    if not models_to_train:
        print("❌ No valid models to train.")
        return
    
    if not XGBOOST_AVAILABLE and 'xgboost' in requested_models:
        print("⚠️  xgboost not installed — skipping XGBoost model")
    
    print(f"🤖 Models to train: {', '.join(models_to_train)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_path}")
    
    # Create model saving directory
    model_dir = output_path / 'saved_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Model save directory: {model_dir}")
    
    # Process each dataset duration
    print(f"📊 Cross-validation folds: {args.cross_val}")
    
    for duration in args.dataset:
        print(f"\n⏱️  Processing {duration}s chunks...")
        
        chunk_size = duration * 700  # 700 Hz sampling rate
        all_chunks = []
        all_labels = []
        all_subject_ids = []
        
        # Extract features for all chunks, tracking subject IDs
        for subj_id, (ecg, label) in enumerate(zip(ecgs, labels)):
            # Filter for labels 1, 2, 3, 4
            valid_mask = np.isin(label, [1, 2, 3, 4])
            valid_ecg = ecg[valid_mask]
            valid_label = label[valid_mask]
            
            # Chunk the signal
            for i in range(0, len(valid_ecg) - chunk_size + 1, chunk_size):
                chunk = valid_ecg[i:i + chunk_size]
                chunk_label = valid_label[i]
                
                all_chunks.append(chunk)
                # Binary classification: {1,3}->0 (No Stress), {2,4}->1 (Stress)
                binary_label = 1 if chunk_label in [2, 4] else 0
                all_labels.append(binary_label)
                all_subject_ids.append(subj_id)
        
        print(f"✓ Created {len(all_chunks)} chunks")
        
        # Extract HRV features from chunks
        feature_list = []
        for chunk in all_chunks:
            try:
                features_dict = feature_extractor.get_hrv_features(chunk)
                feature_values = [
                    features_dict.get('mean_rr', np.nan),
                    features_dict.get('mean_hr', np.nan),
                    features_dict.get('sdnn', np.nan),
                    features_dict.get('rmssd', np.nan),
                    features_dict.get('pnn50', np.nan),
                    features_dict.get('lf_power', np.nan),
                    features_dict.get('hf_power', np.nan),
                    features_dict.get('lf_hf_ratio', np.nan)
                ]
                feature_list.append(feature_values)
            except:
                feature_list.append([np.nan] * 8)
        
        # Create DataFrame with features, labels, and subject IDs
        feature_names = ['mean_rr', 'mean_hr', 'sdnn', 'rmssd', 'pnn50', 'lf_power', 'hf_power', 'lf_hf_ratio']
        df = pd.DataFrame(feature_list, columns=feature_names)
        df['label'] = all_labels
        df['subject_id'] = all_subject_ids
        
        # Handle NaN values - impute with column medians (fallback to 0)
        n_before = len(df)
        nan_count = df[feature_names].isna().sum().sum()
        if nan_count > 0:
            fill_values = df[feature_names].median()
            # If any column median is NaN (entirely NaN column), use 0 instead
            fill_values = fill_values.fillna(0)
            df[feature_names] = df[feature_names].fillna(fill_values)
            print(f"⚠️  Imputed {nan_count} NaN values with feature medians")
        n_after = len(df)
        
        X = df[feature_names].values
        y = df['label'].values
        
        print(f"✓ Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"✓ Class distribution: {np.sum(y==0)} no-stress, {np.sum(y==1)} stress")
        
        # Train each model
        print(f"🚀 Training {len(models_to_train)} models with {args.cross_val}-fold cross-validation...")

        all_results = {}
        for model_name in models_to_train:
            model = _get_model(model_name)
            if model is None:
                print(f"   → {model_name.upper()}... ⚠️ skipped (not available)")
                continue

            try:
                # Use the ML class for evaluation (only for CV metrics)
                models_dict = {model_name: model}
                results = ml_evaluator.eval_all(
                    df, models_dict, feature_names,
                    subject_col='subject_id', label_col='label',
                    cv_method='kfold'
                )
                all_results[model_name] = results[model_name]
                overall = results[model_name]['overall']

                # Refit the model on the FULL dataset so the saved file is
                # usable for inference on new data (e.g. PAVIA). CV above
                # only fits per-fold clones which are then discarded.
                try:
                    from sklearn.base import clone as _clone
                    final_model = _clone(model)
                    final_model.fit(X, y)
                except Exception as fit_err:
                    print(f"   → {model_name.upper()}... "
                          f"⚠️ Could not refit on full data: {fit_err}. "
                          f"Saving unfitted model (prediction will fail).")
                    final_model = model

                # Save the trained model + a small metadata sidecar so
                # downstream scripts (e.g. on PAVIA) know the expected
                # feature order and label mapping.
                model_path = model_dir / f"{model_name}_{duration}s.pkl"
                joblib.dump(final_model, model_path)

                meta_path = model_dir / f"{model_name}_{duration}s.meta.pkl"
                joblib.dump({
                    'feature_names': feature_names,
                    'duration': duration,
                    'label_mapping': {0: 'no_stress', 1: 'stress'},
                    'label_source': 'wesad_binary (1,3)->0 ; (2,4)->1',
                    'trained_on': 'wesad',
                }, meta_path)

                print(f"   → {model_name.upper()}... "
                      f"Accuracy: {overall['accuracy_mean']:.4f} "
                      f"(±{overall['accuracy_std']:.4f}), "
                      f"F1: {overall['f1_mean']:.4f} "
                      f"(±{overall['f1_std']:.4f}) "
                      f"✓ Saved: {model_path}")
            except Exception as e:
                print(f"   → {model_name.upper()}... ❌ Error: {e}")
        
        # ── Save results for this duration ────────────────────────────────────
        if all_results:
            summary_rows = []
            for model_name, result in all_results.items():
                overall = result['overall']
                summary_rows.append({
                    'model'           : model_name,
                    'accuracy_mean'   : overall['accuracy_mean'],
                    'accuracy_std'    : overall['accuracy_std'],
                    'f1_mean'         : overall['f1_mean'],
                    'f1_std'          : overall['f1_std'],
                    'precision_mean'  : overall['precision_mean'],
                    'precision_std'   : overall['precision_std'],
                    'recall_mean'     : overall['recall_mean'],
                    'recall_std'      : overall['recall_std'],
                })

            summary_df = pd.DataFrame(summary_rows).set_index('model')

            # ── 1. CSV summary ────────────────────────────────────────────────
            summary_path = output_path / f"ml_results_{duration}s.csv"
            summary_df.to_csv(summary_path)
            print(f"✓ Saved results CSV: {summary_path}")

            # ── 2. Model-comparison bar chart (accuracy + F1 side-by-side) ───
            _plot_model_comparison(summary_df, duration, output_path)

            # ── 3. Results table image ────────────────────────────────────────
            _plot_results_table(summary_df, duration, output_path)

            # ── 4. Individual confusion matrix per model ──────────────────────
            _plot_confusion_matrices_grid(all_results, duration, output_path)

        print(f"✓ {duration}s dataset processing complete")

    print("\n✅ ML model training complete!")


# ── ML output helpers ─────────────────────────────────────────────────────────

def _plot_model_comparison(summary_df, duration, output_path):
    """
    Grouped bar chart comparing Accuracy and F1 across all models for one
    chunk duration.  Error bars show ±1 std from cross-validation.
    Saved as  ml_model_comparison_{duration}s.png
    """
    models  = summary_df.index.tolist()
    x       = np.arange(len(models))
    width   = 0.35

    metrics = {
        'Accuracy': ('accuracy_mean', 'accuracy_std', '#1976D2'),
        'F1 Score': ('f1_mean',       'f1_std',       '#388E3C'),
    }

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.6), 6))

    for i, (label, (mean_col, std_col, color)) in enumerate(metrics.items()):
        offset = (i - len(metrics) / 2 + 0.5) * width
        means  = summary_df[mean_col].values
        stds   = summary_df[std_col].values
        bars   = ax.bar(x + offset, means, width,
                        label=label, color=color, alpha=0.88,
                        edgecolor='black', linewidth=0.6)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt='none', color='black', capsize=4, linewidth=1)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.007,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title(f'Model Comparison – {duration}s Chunks\n(mean ± std over CV folds)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(0.5, color='grey', lw=0.8, ls='--', alpha=0.5, label='Chance')
    ax.grid(axis='y', ls='--', alpha=0.4)

    fig.tight_layout()
    save_path = output_path / f"ml_model_comparison_{duration}s.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved comparison chart: {save_path}")


def _plot_results_table(summary_df, duration, output_path):
    """
    Render the results summary as a styled table image.
    Cells are colour-coded by value (green = high, red = low).
    Saved as  ml_results_table_{duration}s.png
    """
    display_cols = ['accuracy_mean', 'accuracy_std',
                    'f1_mean', 'f1_std',
                    'precision_mean', 'precision_std',
                    'recall_mean', 'recall_std']
    col_labels   = ['Acc\nmean', 'Acc\n±std',
                    'F1\nmean', 'F1\n±std',
                    'Prec\nmean', 'Prec\n±std',
                    'Rec\nmean', 'Rec\n±std']

    data    = summary_df[display_cols].values
    rows    = summary_df.index.tolist()
    n_rows  = len(rows)
    n_cols  = len(col_labels)

    fig_h = max(3, 0.55 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(n_cols * 1.35, fig_h))
    ax.axis('off')

    # build cell colours: only colour mean columns (even indices), grey for std
    cell_colors = []
    cmap = plt.cm.RdYlGn
    for r in range(n_rows):
        row_colors = []
        for c in range(n_cols):
            if c % 2 == 0:                        # mean column → colour map
                val = data[r, c]
                rgba = cmap(val) if not np.isnan(val) else (0.9, 0.9, 0.9, 1)
            else:                                 # std column → light grey
                rgba = (0.96, 0.96, 0.96, 1)
            row_colors.append(rgba)
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=[[f'{v:.4f}' for v in row] for row in data],
        rowLabels=rows,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)

    ax.set_title(f'ML Results Summary – {duration}s Chunks\n'
                 f'(green = high performance)',
                 fontsize=12, fontweight='bold', pad=12)

    fig.tight_layout()
    save_path = output_path / f"ml_results_table_{duration}s.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved results table: {save_path}")


def _plot_confusion_matrices_grid(all_results, duration, output_path):
    """
    One subplot per model, arranged in a grid.
    Each subplot is a confusion matrix for the aggregated CV predictions.
    Saved as  ml_confusion_matrices_{duration}s.png

    Also saves individual per-model PNGs as
    ml_cm_{model}_{duration}s.png  for easy inclusion in reports.
    """
    CLASS_NAMES = ['Non-Stress', 'Stress']
    models      = list(all_results.keys())
    n           = len(models)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.2))
    # flatten axes always into 1-D list
    if n == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for ax, model_name in zip(axes, models):
        result  = all_results[model_name]
        y_true  = result.get('y_true', [])
        y_pred  = result.get('y_pred', [])

        if len(y_true) == 0:
            ax.set_visible(False)
            continue

        disp = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=CLASS_NAMES,
            cmap='Blues',
            colorbar=False,
            ax=ax,
        )
        overall = result['overall']
        ax.set_title(
            f'{model_name.replace("_", " ").title()}\n'
            f'Acc={overall["accuracy_mean"]:.3f}  '
            f'F1={overall["f1_mean"]:.3f}',
            fontsize=10, fontweight='bold'
        )

        # ── also save standalone individual CM ──────────────────────────────
        fig_ind, ax_ind = plt.subplots(figsize=(4.5, 4.0))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=CLASS_NAMES,
            cmap='Blues',
            colorbar=False,
            ax=ax_ind,
        )
        ax_ind.set_title(
            f'{model_name.replace("_", " ").title()} – {duration}s\n'
            f'Acc={overall["accuracy_mean"]:.3f}  F1={overall["f1_mean"]:.3f}',
            fontsize=10, fontweight='bold'
        )
        fig_ind.tight_layout()
        ind_path = output_path / f"ml_cm_{model_name}_{duration}s.png"
        fig_ind.savefig(ind_path, dpi=150, bbox_inches='tight')
        plt.close(fig_ind)

    # hide any unused subplots in the grid
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f'Confusion Matrices – {duration}s Chunks',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    grid_path = output_path / f"ml_confusion_matrices_{duration}s.png"
    fig.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved confusion matrix grid : {grid_path}")
    print(f"✓ Saved individual CMs        : ml_cm_<model>_{duration}s.png")


# ── FFT Analysis ──────────────────────────────────────────────────────────────

def run_fft_analysis(args):
    """
    FFT frequency analysis with cosine-similarity comparison.

    For each chunk duration (-d, default 30/120/300 s):
      1. Chunks all subjects into non-overlapping windows.
      2. Computes FFT per chunk (via Features.compute_fft).
      3. Plots mean spectrum ± std for stress vs. non-stress.
      4. Overlays all durations on one figure per class.
      5. Computes cosine similarity for three pair types:
           • Stress  ↔  Non-stress   (cross-class)
           • Stress  ↔  Stress       (within-stress)
           • Non-stress ↔ Non-stress (within non-stress)
      6. Saves KDE distribution plots and a grouped summary bar chart.

    Label mapping used (binary):
        1 (Baseline)   → 0  Non-Stress
        2 (Stress)     → 1  Stress
        3 (Amusement)  → 0  Non-Stress
        4 (Meditation) → 1  Stress
    """
    print("\n" + "=" * 80)
    print("FFT FREQUENCY ANALYSIS")
    print("=" * 80)

    # ── config ────────────────────────────────────────────────────────────────
    FS          = 700
    VALID_LABELS = [1, 2, 3, 4]
    LABEL_MAP    = {1: 0, 2: 1, 3: 0, 4: 1}
    LABEL_NAME   = {0: 'Non-Stress', 1: 'Stress'}
    COLORS       = {0: '#2196F3', 1: '#F44336'}     # blue / red
    FREQ_MAX     = args.fft_freq_max
    MAX_PAIRS    = args.fft_max_pairs
    durations    = sorted(set(args.dataset))

    output_dir  = args.output or '../results/fft_analysis'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory : {output_path}")
    print(f"⏱️  Durations        : {durations} s")
    print(f"🔢 Max cosine pairs : {MAX_PAIRS}")
    print(f"📡 Freq limit       : {FREQ_MAX} Hz")

    # ── load data ─────────────────────────────────────────────────────────────
    data_loader       = Data(fs=FS)
    feature_extractor = Features(fs=FS)
    dataset_path      = _get_dataset_path(args)

    print(f"\n📂 Loading dataset from: {dataset_path}")
    try:
        ecgs, labels = data_loader.read_dataset(str(dataset_path))
        print(f"✓ Loaded {len(ecgs)} subjects")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # ── helper: spectrum interpolation ────────────────────────────────────────
    def interp_spectrum(freqs, mag, common_freqs):
        return np.interp(common_freqs, freqs, mag, left=0.0, right=0.0)

    # ── step 1: chunk + FFT ───────────────────────────────────────────────────
    results = {}   # results[duration] = {'freqs': ..., 'spectra': {0:[…], 1:[…]}}

    for duration in durations:
        chunk_size    = duration * FS
        common_freqs  = np.fft.rfftfreq(chunk_size, d=1.0 / FS)
        freq_mask     = common_freqs <= FREQ_MAX
        common_freqs  = common_freqs[freq_mask]
        spectra       = {0: [], 1: []}
        n_chunks      = {0: 0, 1: 0}

        print(f"\n⚙️  Computing FFT – {duration}s chunks...")
        for ecg, label in zip(ecgs, labels):
            valid_mask  = np.isin(label, VALID_LABELS)
            valid_ecg   = ecg[valid_mask]
            valid_label = label[valid_mask]

            for start in range(0, len(valid_ecg) - chunk_size + 1, chunk_size):
                chunk     = valid_ecg[start: start + chunk_size]
                raw_lbl   = int(valid_label[start])
                bin_lbl   = LABEL_MAP[raw_lbl]
                try:
                    f, m = feature_extractor.compute_fft(chunk)
                    if f.size == 0:
                        continue
                    spectra[bin_lbl].append(interp_spectrum(f, m, common_freqs))
                    n_chunks[bin_lbl] += 1
                except Exception:
                    pass

        results[duration] = {'freqs': common_freqs, 'spectra': spectra}
        total = n_chunks[0] + n_chunks[1]
        print(f"   ✓ {total} chunks  (non-stress: {n_chunks[0]}, stress: {n_chunks[1]})")

    # ── step 2: stacked mean-spectrum figure (one panel per duration) ─────────
    print("\n📊 Plotting mean FFT spectra...")
    fig, axes = plt.subplots(len(durations), 1,
                             figsize=(14, 4.5 * len(durations)), sharex=False)
    if len(durations) == 1:
        axes = [axes]

    for ax, duration in zip(axes, durations):
        res     = results[duration]
        freqs   = res['freqs']
        spectra = res['spectra']
        for bin_lbl, lname in LABEL_NAME.items():
            mags = spectra[bin_lbl]
            if not mags:
                continue
            arr  = np.stack(mags)
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            ax.plot(freqs, mean, color=COLORS[bin_lbl], lw=1.8, label=lname)
            ax.fill_between(freqs, mean - std, mean + std,
                            color=COLORS[bin_lbl], alpha=0.18)
        ax.axvspan(0.04, 0.15, color='gold',   alpha=0.12, label='LF (0.04–0.15 Hz)')
        ax.axvspan(0.15, 0.40, color='orchid', alpha=0.12, label='HF (0.15–0.40 Hz)')
        ax.set_xlim(0, FREQ_MAX)
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('Magnitude',      fontsize=11)
        ns, ss = len(spectra[0]), len(spectra[1])
        ax.set_title(f'Mean FFT – {duration}s  (non-stress n={ns}, stress n={ss})',
                     fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, ls='--', alpha=0.4)

    fig.suptitle('ECG FFT: Stress vs. Non-Stress across Chunk Durations',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    p = output_path / 'fft_spectra_all_durations.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved: {p}")

    # ── step 3: overlay plot – all durations on same axes ────────────────────
    dur_colors  = {d: c for d, c in zip(durations, ['#1565C0', '#2E7D32', '#6A1B9A',
                                                     '#E65100', '#4A148C'])}
    dur_ls      = {d: ls for d, ls in zip(durations, ['-', '--', ':', '-.', (0, (3, 1, 1, 1))])}
    fig, axes2  = plt.subplots(1, 2, figsize=(16, 5))
    for bin_lbl, lname in LABEL_NAME.items():
        ax = axes2[bin_lbl]
        for dur in durations:
            mags = results[dur]['spectra'][bin_lbl]
            if not mags:
                continue
            arr  = np.stack(mags)
            mean = arr.mean(axis=0)
            ax.plot(results[dur]['freqs'], mean,
                    color=dur_colors.get(dur, 'grey'),
                    ls=dur_ls.get(dur, '-'),
                    lw=1.8, label=f'{dur}s  (n={len(mags)})')
        ax.axvspan(0.04, 0.15, color='gold',   alpha=0.10)
        ax.axvspan(0.15, 0.40, color='orchid', alpha=0.10)
        ax.set_xlim(0, FREQ_MAX)
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('Magnitude',      fontsize=11)
        ax.set_title(f'{lname} – all chunk sizes', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, ls='--', alpha=0.4)
    fig.suptitle('FFT Overlay – Effect of Chunk Duration',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p = output_path / 'fft_overlay_durations.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved: {p}")

    # ── step 4: cosine similarity ─────────────────────────────────────────────
    print("\n📐 Computing cosine similarities...")
    random.seed(42)
    np.random.seed(42)

    COMP_NAMES   = ['Stress ↔ Non-stress', 'Stress ↔ Stress', 'Non-stress ↔ Non-stress']
    COMP_COLORS  = ['#E53935',             '#1E88E5',          '#43A047']

    def _cos_sim(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else np.nan

    def _sample_pairs(list_a, list_b, max_p, same=False):
        sims = []
        if same:
            candidates = [(i, j) for i in range(len(list_a))
                          for j in range(i + 1, len(list_a))]
        else:
            candidates = [(i, j) for i in range(len(list_a))
                          for j in range(len(list_b))]
        chosen = random.sample(candidates, min(max_p, len(candidates)))
        for i, j in chosen:
            s = _cos_sim(list_a[i], list_b[j] if not same else list_a[j])
            if not np.isnan(s):
                sims.append(s)
        return sims

    cos_results = {}
    for duration in durations:
        spec_ns = results[duration]['spectra'][0]
        spec_s  = results[duration]['spectra'][1]
        cross     = _sample_pairs(spec_s,  spec_ns, MAX_PAIRS, same=False)
        within_s  = _sample_pairs(spec_s,  spec_s,  MAX_PAIRS, same=True)
        within_ns = _sample_pairs(spec_ns, spec_ns, MAX_PAIRS, same=True)
        cos_results[duration] = {
            'Stress ↔ Non-stress'    : cross,
            'Stress ↔ Stress'        : within_s,
            'Non-stress ↔ Non-stress': within_ns,
        }
        print(f"\n   {duration}s chunks:")
        for cname, vals in cos_results[duration].items():
            if vals:
                print(f"     {cname:<30s} mean={np.mean(vals):.4f}  "
                      f"std={np.std(vals):.4f}  n={len(vals)}")

    # ── step 5: KDE distribution grid ────────────────────────────────────────
    print("\n📊 Plotting cosine similarity distributions...")
    fig, axes = plt.subplots(len(durations), len(COMP_NAMES),
                             figsize=(18, 4 * len(durations)))
    if len(durations) == 1:
        axes = [axes]

    for row, duration in enumerate(durations):
        ax_row = axes[row] if len(durations) > 1 else axes[0]
        for col, (comp, color) in enumerate(zip(COMP_NAMES, COMP_COLORS)):
            ax   = ax_row[col]
            vals = cos_results[duration][comp]
            if vals:
                sns.kdeplot(vals, ax=ax, color=color, fill=True, alpha=0.35, lw=2)
                ax.axvline(np.mean(vals), color=color, lw=1.5, ls='--',
                           label=f'mean={np.mean(vals):.3f}')
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Cosine Similarity', fontsize=10)
            ax.set_ylabel('Density',           fontsize=10)
            ax.set_title(f'{duration}s – {comp}', fontsize=10)
            ax.grid(True, ls='--', alpha=0.3)

    fig.suptitle('Cosine Similarity of FFT Spectra – Stress vs. Non-Stress',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    p = output_path / 'fft_cosine_similarity_distributions.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved: {p}")

    # ── step 6: summary bar chart + CSV ──────────────────────────────────────
    summary_rows = []
    for duration in durations:
        for comp in COMP_NAMES:
            vals = cos_results[duration][comp]
            summary_rows.append({
                'Duration (s)': f'{duration}s',
                'Comparison'  : comp,
                'Mean'        : np.mean(vals) if vals else np.nan,
                'Std'         : np.std(vals)  if vals else np.nan,
                'N pairs'     : len(vals),
            })

    summary_df = pd.DataFrame(summary_rows)
    csv_path   = output_path / 'fft_cosine_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"   ✓ Saved CSV: {csv_path}")

    x      = np.arange(len(durations))
    n_comp = len(COMP_NAMES)
    width  = 0.22
    fig, ax = plt.subplots(figsize=(max(10, len(durations) * 3.5), 5))

    for i, (comp, color) in enumerate(zip(COMP_NAMES, COMP_COLORS)):
        means = [np.mean(cos_results[d][comp]) if cos_results[d][comp] else np.nan
                 for d in durations]
        stds  = [np.std(cos_results[d][comp])  if cos_results[d][comp] else np.nan
                 for d in durations]
        offset = (i - n_comp / 2 + 0.5) * width
        bars   = ax.bar(x + offset, means, width,
                        color=color, alpha=0.88, label=comp,
                        edgecolor='black', linewidth=0.5)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt='none', color='black', capsize=3, lw=1)
        for bar, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{m:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{d}s' for d in durations], fontsize=11)
    ax.set_xlabel('Chunk Duration', fontsize=11)
    ax.set_ylabel('Mean Cosine Similarity', fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title('Mean FFT Cosine Similarity by Chunk Duration & Comparison Type',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', ls='--', alpha=0.4)
    fig.tight_layout()
    p = output_path / 'fft_cosine_summary_bar.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved: {p}")

    print("\n✅ FFT analysis complete!")
    print(f"   All outputs → {output_path.resolve()}")


# ── PREDICTION MODE ──────────────────────────────────────────────────────────

def _load_pavia_data(data_dir=None):
    """
    Load and prepare Pavia HRV data for prediction.

    The Pavia CSV uses column names:
        HR, SDNN, rMSSD, pNN50, SE, LF, HF, LFHF
    which differ from the standard feature names:
        mean_rr, mean_hr, sdnn, rmssd, pnn50, lf_power, hf_power, lf_hf_ratio

    This function:
        1. Loads pavia_features.csv and pavia_labels.csv
        2. Removes all-NaN (empty) rows
        3. Maps Pavia column names to standard feature names
        4. Computes mean_rr from HR (mean_rr = 60000 / HR)
        5. Drops SE (Sample Entropy) which is not in the standard set
        6. Aligns labels with the filtered feature rows

    Parameters:
        data_dir: path to the data directory (default: project_root / 'data')

    Returns:
        X: np.ndarray of shape (n_samples, 8) with standard feature ordering
        y: np.ndarray of shape (n_samples,) with binary labels (0/1)
        feature_names: list of standard feature column names
    """
    from pathlib import Path
    data_dir = Path(data_dir) if data_dir else project_root / 'data'

    features_path = data_dir / 'pavia_features.csv'
    labels_path   = data_dir / 'pavia_labels.csv'

    if not features_path.exists():
        print(f"   ❌ Pavia features not found: {features_path}")
        return None, None, None
    if not labels_path.exists():
        print(f"   ❌ Pavia labels not found: {labels_path}")
        return None, None, None

    # 1. Load raw data
    raw_df  = pd.read_csv(features_path)
    raw_labels = pd.read_csv(labels_path)
    print(f"   ✓ Loaded {features_path.name} - shape {raw_df.shape}")
    print(f"   ✓ Loaded {labels_path.name}   - shape {raw_labels.shape}")

    # 2. Remove all-NaN rows (empty separators in CSV)
    all_nan = raw_df.isna().all(axis=1)
    n_empty = int(all_nan.sum())
    if n_empty > 0:
        print(f"   Removing {n_empty} empty row(s) from features")
        valid_df = raw_df.dropna(how='all').reset_index(drop=True)
        # Align labels: drop labels at same indices as empty feature rows
        valid_labels = raw_labels.loc[~all_nan].reset_index(drop=True)
    else:
        valid_df = raw_df
        valid_labels = raw_labels

    # 3. Map Pavia column names to standard names
    pavia_to_standard = {
        'HR':    'mean_hr',
        'SDNN':  'sdnn',
        'rMSSD': 'rmssd',
        'pNN50': 'pnn50',
        'SE':    None,          # Sample Entropy - not in standard features
        'LF':    'lf_power',
        'HF':    'hf_power',
        'LFHF':  'lf_hf_ratio',
    }

    # Rename columns
    mapped = valid_df.rename(
        columns={k: v for k, v in pavia_to_standard.items() if v is not None}
    )
    # Drop unmapped columns (e.g. SE)
    cols_to_drop = [c for c in pavia_to_standard if pavia_to_standard[c] is None and c in mapped.columns]
    if cols_to_drop:
        mapped = mapped.drop(columns=cols_to_drop)

    # 4. Compute mean_rr from HR
    if 'mean_hr' in mapped.columns:
        mapped['mean_rr'] = 60000.0 / mapped['mean_hr']

    # 5. Select only the 8 standard features in the correct order
    standard_features = ['mean_rr', 'mean_hr', 'sdnn', 'rmssd', 'pnn50',
                         'lf_power', 'hf_power', 'lf_hf_ratio']
    available_features = [f for f in standard_features if f in mapped.columns]
    missing = [f for f in standard_features if f not in mapped.columns]
    if missing:
        print(f"   Missing features (will be filled as NaN): {missing}")

    X = mapped[available_features].values.astype(np.float64)
    y = valid_labels['label'].values.ravel().astype(int)

    # 6. Handle any remaining NaN values
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"   {int(nan_count)} NaN value(s) detected - imputing with column medians")
        for col_idx in range(X.shape[1]):
            col_median = np.nanmedian(X[:, col_idx])
            if np.isnan(col_median):
                col_median = 0.0
            mask = np.isnan(X[:, col_idx])
            X[mask, col_idx] = col_median

    print(f"   Pavia data ready - {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Labels - {len(y)} samples ({y.sum()} stress, {len(y)-y.sum()} non-stress)")
    return X, y, standard_features



def run_prediction(args):
    """
    Load trained models and make predictions on test data.
    
    This function supports three modes:
    1. Pavia mode (--pavia): Load Pavia HRV data from CSV with automatic column mapping
    2. Custom CSV mode (--test-data): Load features from CSV files
    3. WESAD dataset mode: Load test data from the same WESAD dataset
    
    Parameters:
    """
    print("\n" + "=" * 80)
    print("PREDICTION MODE")
    print("=" * 80)

    # Prediction mode defaults to the 30 s trained models, which is the
    # canonical choice for cross-dataset evaluation (e.g. PAVIA). Pass
    # `-d 30 120 300` (or any subset) to override.
    if args.dataset == [30, 120, 300]:
        durations_to_use = [30]
        print("ℹ️  Defaulting to 30 s models (override with -d).")
    else:
        durations_to_use = args.dataset

    # Set default output directory
    output_dir = args.output or '../results/predictions'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_path}")
    
    # Determine which models to use
    available_models = [
        'knn', 'svm', 'decision_tree', 'random_forest', 
        'gradient_boosting', 'logistic_regression', 'xgboost'
    ]
    
    requested_models = [m.lower() for m in args.models]
    models_to_use = [m for m in requested_models if m in available_models]
    models_to_use = _get_available_models(models_to_use)
    
    if not models_to_use:
        print("❌ No valid models specified for prediction.")
        return
    
    print(f"🤖 Models to use: {', '.join(models_to_use)}")
    
    # Process each duration
    feature_names = ['mean_rr', 'mean_hr', 'sdnn', 'rmssd', 'pnn50', 'lf_power', 'hf_power', 'lf_hf_ratio']
    all_predictions = {}

    for duration in durations_to_use:
        print(f"\n⏱️  Processing {duration}s chunks...")

        # Load the trained models
        models = {}
        model_dir = Path(args.model_dir)
        
        for model_name in models_to_use:
            model_path = model_dir / f"{model_name}_{duration}s.pkl"
            if model_path.exists():
                try:
                    models[model_name] = joblib.load(model_path)
                    print(f"   ✓ Loaded {model_name.upper()} from {model_path}")
                except Exception as e:
                    print(f"   ❌ Failed to load {model_name.upper()}: {e}")
            else:
                print(f"   ⚠️  Model not found: {model_path}")
        
        if not models:
            print(f"   ❌ No models found for {duration}s duration. Skipping...")
            continue
        
        # ── PAVIA MODE ────────────────────────────────────────────────────────────
        if args.pavia:
            print("   📋 Using Pavia HRV data for prediction...")
            data_dir = args.input if args.input else project_root / 'data'
            X_test, test_labels, loaded_feature_names = _load_pavia_data(data_dir)
            if X_test is None:
                print("   ❌ Failed to load Pavia data. Skipping...")
                continue
            feature_names = loaded_feature_names
            print(f"   ✓ Pavia data ready: {X_test.shape[0]} samples, {X_test.shape[1]} features")

        # ── CUSTOM CSV MODE ────────────────────────────────────────────────────────
        elif args.test_data:
            try:
                test_data = pd.read_csv(args.test_data)
                if args.test_labels:
                    test_labels = pd.read_csv(args.test_labels).values.ravel()
                elif 'label' in test_data.columns:
                    test_labels = test_data['label'].values
                    test_data = test_data.drop('label', axis=1)
                
                # Ensure we have the right features
                available_features = [f for f in feature_names if f in test_data.columns]
                if len(available_features) != len(feature_names):
                    print(f"   ⚠️  Missing some features. Available: {available_features}")
                    # Use available features only
                    X_test = test_data[available_features].values
                else:
                    X_test = test_data[feature_names].values
                    
                print(f"   ✓ Loaded test data: {X_test.shape[0]} samples")
                
            except Exception as e:
                print(f"   ❌ Error loading test data: {e}")
                continue
                
        # ── WESAD DATASET MODE ─────────────────────────────────────────────────────
        else:
            try:
                # Load WESAD data
                dataset_path = _get_dataset_path(args)
                data_loader = Data(fs=700)
                feature_extractor = Features(fs=700)
                
                print(f"   📂 Loading WESAD data from: {dataset_path}")
                ecgs, labels = data_loader.read_dataset(str(dataset_path))
                
                # Use first subject as test (or random split could be implemented)
                # For simplicity, use subject 0 as test
                test_subject = 0
                test_ecg = ecgs[test_subject]
                test_label = labels[test_subject]
                
                # Filter valid labels
                valid_mask = np.isin(test_label, [1, 2, 3, 4])
                valid_ecg = test_ecg[valid_mask]
                valid_label = test_label[valid_mask]
                
                # Chunk the test data
                chunk_size = duration * 700
                X_test = []
                y_test = []
                
                for i in range(0, len(valid_ecg) - chunk_size + 1, chunk_size):
                    chunk = valid_ecg[i:i + chunk_size]
                    chunk_label = valid_label[i]
                    
                    try:
                        features_dict = feature_extractor.get_hrv_features(chunk)
                        feature_values = [
                            features_dict.get(f, np.nan) for f in feature_names
                        ]
                        X_test.append(feature_values)
                        # Binary classification: {1,3}->0 (No Stress), {2,4}->1 (Stress)
                        binary_label = 1 if chunk_label in [2, 4] else 0
                        y_test.append(binary_label)
                    except:
                        continue
                
                X_test = np.array(X_test)
                y_test = np.array(y_test)
                
                # Handle NaN values
                nan_mask = ~np.isnan(X_test).any(axis=1)
                X_test = X_test[nan_mask]
                y_test = y_test[nan_mask]
                
                # Impute remaining NaN with column medians
                for col_idx in range(X_test.shape[1]):
                    col_median = np.nanmedian(X_test[:, col_idx])
                    if np.isnan(col_median):
                        col_median = 0
                    X_test[np.isnan(X_test[:, col_idx]), col_idx] = col_median
                
                print(f"   ✓ Loaded test subject {test_subject}: {X_test.shape[0]} chunks")
                test_labels = y_test
                
            except Exception as e:
                print(f"   ❌ Error loading WESAD test data: {e}")
                continue
        
        if X_test is None or len(X_test) == 0:
            print(f"   ⚠️  No test data available for {duration}s")
            continue
        
        # Make predictions with each model
        predictions = {}
        y_pred_all = []
        y_true = test_labels if test_labels is not None else None
        
        for model_name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                predictions[model_name] = y_pred
                y_pred_all.append(y_pred)
                
                # Calculate metrics if we have true labels
                if y_true is not None:
                    acc = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, average='weighted')
                    print(f"   → {model_name.upper()}: Accuracy={acc:.4f}, F1={f1:.4f}")
                else:
                    print(f"   → {model_name.upper()}: Predictions made (no labels for evaluation)")
                    
            except Exception as e:
                print(f"   ❌ Error making predictions with {model_name.upper()}: {e}")
        
        all_predictions[duration] = {
            'models': models,
            'predictions': predictions,
            'X_test': X_test,
            'y_true': y_true
        }
        
        # ── Plot predictions ──────────────────────────────────────────────────
        if y_true is not None and predictions:
            _plot_prediction_results(predictions, y_true, duration, output_path, feature_names)
            _plot_prediction_comparison(predictions, y_true, duration, output_path)
    
    # ── Save all predictions to CSV ──────────────────────────────────────────
    for duration, data in all_predictions.items():
        if data['predictions']:
            pred_df = pd.DataFrame(data['X_test'], columns=feature_names)
            if data['y_true'] is not None:
                pred_df['y_true'] = data['y_true']
            for model_name, y_pred in data['predictions'].items():
                pred_df[f'pred_{model_name}'] = y_pred
            
            csv_path = output_path / f"predictions_{duration}s.csv"
            pred_df.to_csv(csv_path, index=False)
            print(f"\n✓ Saved predictions to: {csv_path}")
    
    print("\n✅ Prediction complete!")


def _plot_prediction_results(predictions, y_true, duration, output_path, feature_names):
    """
    Plot prediction results including confusion matrices and metrics.
    """
    CLASS_NAMES = ['Non-Stress', 'Stress']
    n_models = len(predictions)
    
    # ── Confusion matrices grid ──────────────────────────────────────────────
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, 
                             figsize=(ncols * 4.5, nrows * 4.0))
    if n_models == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]
    
    metrics_data = []
    
    for ax, (model_name, y_pred) in zip(axes, predictions.items()):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        metrics_data.append({'model': model_name, 'accuracy': acc, 'f1': f1})
        
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=CLASS_NAMES,
            cmap='Blues',
            colorbar=False,
            ax=ax,
        )
        ax.set_title(
            f'{model_name.replace("_", " ").title()}\n'
            f'Acc={acc:.3f}  F1={f1:.3f}',
            fontsize=10, fontweight='bold'
        )
    
    # Hide unused subplots
    for ax in axes[n_models:]:
        ax.set_visible(False)
    
    fig.suptitle(f'Prediction Results – {duration}s Chunks (Test Set)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_path = output_path / f'prediction_confusion_{duration}s.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved confusion matrices: {save_path}")
    
    # ── Metrics bar chart ─────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(metrics_data)
    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.6), 5))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, metrics_df['accuracy'], width, 
                   label='Accuracy', color='#1976D2', alpha=0.8)
    bars2 = ax.bar(x + width/2, metrics_df['f1'], width,
                   label='F1 Score', color='#388E3C', alpha=0.8)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_df['model']], 
                        rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title(f'Model Performance on Test Set – {duration}s Chunks')
    ax.legend()
    ax.grid(axis='y', ls='--', alpha=0.4)
    
    fig.tight_layout()
    save_path = output_path / f'prediction_metrics_{duration}s.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved metrics chart: {save_path}")
    
    # ── Print detailed classification report ──────────────────────────────────
    print(f"\n   📊 Classification Report for {duration}s:")
    print("-" * 50)
    for model_name in predictions.keys():
        y_pred = predictions[model_name]
        print(f"\n   {model_name.upper()}:")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))


def _plot_prediction_comparison(predictions, y_true, duration, output_path):
    """
    Plot comparison of predictions across models for a sample of test points.
    """
    n_samples = min(50, len(y_true))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot true labels
    x = np.arange(n_samples)
    ax.scatter(x, y_true[:n_samples], color='black', s=30, 
               marker='s', label='True Labels', zorder=5)
    
    # Plot predictions for each model with slight offset
    offsets = np.linspace(-0.1, 0.1, len(predictions))
    for i, (model_name, y_pred) in enumerate(predictions.items()):
        offset = offsets[i]
        ax.scatter(x + offset, y_pred[:n_samples], s=20, alpha=0.6,
                   label=model_name.upper(), marker='o')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Class (0=Non-Stress, 1=Stress)')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Non-Stress', 'Stress'])
    ax.set_title(f'Model Predictions Comparison – {duration}s Chunks (First {n_samples} samples)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, ls='--', alpha=0.3)
    
    fig.tight_layout()
    save_path = output_path / f'prediction_comparison_{duration}s.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved predictions comparison: {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Main entry point for CLI application."""
    parser = parse_arguments()

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return

    # Parse arguments
    args = parser.parse_args()

    # Execute appropriate command based on flags
    if args.correlation:
        run_correlation_analysis(args)
    elif args.full_signal:
        run_full_signal_visualization(args)
    elif args.ml_training:
        run_ml_training(args)
    elif args.fft_analysis:
        run_fft_analysis(args)
    elif args.prediction:
        run_prediction(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()