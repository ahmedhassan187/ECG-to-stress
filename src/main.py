"""
ECG-to-Stress Analysis CLI
Command-line interface for WESAD dataset analysis, feature extraction, correlation analysis, 
and machine learning model training.

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
"""

import argparse
import sys
import os
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error

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
        help='Output directory (default: results/{correlation_figures|signal_plots|ml_results})'
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
    Build an NxN comparison table (rows = small window, cols = large window)
    for the given metric ('r', 'icc' or 'mae'). The diagonal is 1.0 (R/ICC)
    or 0.0 (MAE); the table is symmetric.
    """
    n = len(durations)
    table = pd.DataFrame(index=durations, columns=durations, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                table.iloc[i, j] = 1.0 if metric != 'mae' else 0.0
            elif i < j:
                key = (durations[i], durations[j])
                table.iloc[i, j] = comparison_table.get(key, {}).get(metric, np.nan)
            else:
                key = (durations[j], durations[i])
                table.iloc[i, j] = comparison_table.get(key, {}).get(metric, np.nan)
    table.index.name = 'small_duration_s'
    table.columns.name = 'large_duration_s'
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
    Save a bar chart of per-comparison values for a given metric.
    """
    keys = list(comparison_table.keys())
    labels = [f"{a}s vs {b}s" for a, b in keys]
    vals = [comparison_table[k].get(metric, np.nan) for k in keys]

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 5))
    bars = ax.bar(labels, vals, color='steelblue', edgecolor='black')
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} for each duration pair")
    finite = [v for v in vals if not np.isnan(v)]
    upper = max(finite + [1]) * 1.15
    ax.set_ylim(0, upper)
    ax.tick_params(axis='x', rotation=30)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                    ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


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
        print(f"\n{metric.upper()} comparison (rows = small window, cols = large window):")
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
                # Use the ML class for evaluation
                models_dict = {model_name: model}
                results = ml_evaluator.eval_all(
                    df, models_dict, feature_names,
                    subject_col='subject_id', label_col='label',
                    cv_method='kfold'
                )
                all_results[model_name] = results[model_name]
                
                overall = results[model_name]['overall']
                print(f"   → {model_name.upper()}... "
                      f"Accuracy: {overall['accuracy_mean']:.4f} (±{overall['accuracy_std']:.4f}), "
                      f"F1: {overall['f1_mean']:.4f} (±{overall['f1_std']:.4f})")
            except Exception as e:
                print(f"   → {model_name.upper()}... ❌ Error: {e}")
        
        # Save results summary
        if all_results:
            summary_rows = []
            for model_name, result in all_results.items():
                overall = result['overall']
                summary_rows.append({
                    'model': model_name,
                    'accuracy_mean': overall['accuracy_mean'],
                    'accuracy_std': overall['accuracy_std'],
                    'f1_mean': overall['f1_mean'],
                    'f1_std': overall['f1_std'],
                    'precision_mean': overall['precision_mean'],
                    'precision_std': overall['precision_std'],
                    'recall_mean': overall['recall_mean'],
                    'recall_std': overall['recall_std']
                })
            
            summary_df = pd.DataFrame(summary_rows).set_index('model')
            summary_path = output_path / f"ml_results_{duration}s.csv"
            summary_df.to_csv(summary_path)
            print(f"✓ Saved results: {summary_path}")
            
            # Generate and save confusion matrices
            viz = Visualization()
            n_models = len(all_results)
            if n_models > 0:
                try:
                    all_predictions = {}
                    for model_name, result in all_results.items():
                        all_predictions[model_name] = (result['y_true'], result['y_pred'])
                    
                    fig, axes = viz.plot_multiple_conf_mats(
                        all_predictions,
                        title=f'Confusion Matrices ({duration}s windows)'
                    )
                    cm_path = output_path / f"confusion_matrices_{duration}s.png"
                    fig.savefig(cm_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"✓ Saved confusion matrices: {cm_path}")
                except Exception as e:
                    print(f"⚠️  Could not generate confusion matrix plots: {e}")
        
        print(f"✓ {duration}s dataset processing complete")
    
    print("\n✅ ML model training complete!")


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
    else:
        parser.print_help()


if __name__ == '__main__':
    main()