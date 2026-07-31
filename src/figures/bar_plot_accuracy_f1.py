"""
Generate a bar plot comparing model Accuracy, F1, and AUC scores.

Supports:
  - Single label mode:  py src/figures/bar_plot_accuracy_f1.py -l binary
  - Single label mode:  py src/figures/bar_plot_accuracy_f1.py -l 3class
  - Comparison mode:    py src/figures/bar_plot_accuracy_f1.py -l both   (default)

Reads results from:
    ../results/ml_results/ml_results_30s_<mode>.csv

Usage:
    py src/figures/bar_plot_accuracy_f1.py              # Both modes (default)
    py src/figures/bar_plot_accuracy_f1.py -l binary     # Binary only
    py src/figures/bar_plot_accuracy_f1.py -l 3class     # 3-Class only
    py src/figures/bar_plot_accuracy_f1.py -l both       # Both (explicit)
    py src/figures/bar_plot_accuracy_f1.py -d 30 120     # Custom durations

Output:
    ../results/ml_results/ml_barplot_accuracy_f1.png
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src is on path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Change to project root
project_root = src_dir.parent
os.chdir(str(project_root))

RESULTS_DIR = Path("../results/ml_results")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate bar plot of ML model performance.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  py src/figures/bar_plot_accuracy_f1.py              # Both modes (default)
  py src/figures/bar_plot_accuracy_f1.py -l binary     # Binary only
  py src/figures/bar_plot_accuracy_f1.py -l 3class     # 3-Class only
  py src/figures/bar_plot_accuracy_f1.py -l both       # Both (explicit)
  py src/figures/bar_plot_accuracy_f1.py -d 30 120     # Custom durations
        """
    )
    parser.add_argument(
        '-l', '--labels',
        type=str,
        default='both',
        choices=['binary', '3class', 'both'],
        help='Label mode: binary, 3class, or both (default: both)'
    )
    parser.add_argument(
        '-d', '--dataset',
        nargs='+',
        type=int,
        default=[30],
        help='Dataset durations in seconds (default: 30)'
    )
    return parser.parse_args()


def load_results(csv_path):
    """Load ML results CSV and return dict of {model: {accuracy, f1, auc}}."""
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path, index_col='model')
    result = {}
    for row in df.index:
        entry = {
            'accuracy': df.loc[row, 'accuracy_mean'],
            'f1': df.loc[row, 'f1_mean'],
            'auc': df.loc[row, 'auc_mean'] if 'auc_mean' in df.columns else np.nan,
        }
        result[row] = entry
    return result


def plot_single_mode(data, label_mode, duration, output_path):
    """
    Create a bar plot for a single label mode.
    Shows Accuracy, F1, and AUC bars for each model.
    """
    models = list(data.keys())
    x = np.arange(len(models))
    width = 0.25  # Width of each bar (3 bars per model)

    fig, ax = plt.subplots(figsize=(20, 10))

    # Bar positions
    positions = {
        'Accuracy': x - width,
        'F1':       x,
        'AUC':      x + width,
    }

    colors = {
        'Accuracy': '#1976D2',  # Blue
        'F1':       '#388E3C',  # Green
        'AUC':      '#7B1FA2',  # Purple
    }

    mode_label = 'Binary' if label_mode == 'binary' else '3-Class'

    for metric, pos in positions.items():
        values = [data[m][metric.lower()] for m in models]
        bars = ax.bar(pos, values, width, label=metric,
                      color=colors[metric], edgecolor='black', linewidth=0.6,
                      alpha=0.88)

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=15, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n').title() for m in models],
                       fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_title(f'Model Performance — {mode_label} Classification\n'
                 f'({duration}s Window Duration, Mean over CV Folds)',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right', ncol=3)
    ax.axhline(0.5, color='grey', lw=0.8, ls='--', alpha=0.5,
               label='Chance (binary)')
    ax.grid(axis='y', ls='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved bar plot: {output_path}")


def plot_side_by_side_bars(binary, threeclass, duration, output_path):
    """
    Create a grouped bar plot comparing binary vs 3-class:
    For each model, 6 bars: Binary Accuracy, Binary F1, Binary AUC,
                             3class Accuracy, 3class F1, 3class AUC
    """
    models = list(binary.keys())
    x = np.arange(len(models))
    width = 0.13  # Width of each bar (narrower to fit 6 bars)

    fig, ax = plt.subplots(figsize=(20, 10))

    # Bar positions
    positions = {
        'Binary Accuracy': x - 2.5 * width,
        'Binary F1':       x - 1.5 * width,
        'Binary AUC':      x - 0.5 * width,
        '3-Class Accuracy': x + 0.5 * width,
        '3-Class F1':      x + 1.5 * width,
        '3-Class AUC':     x + 2.5 * width,
    }

    colors = {
        'Binary Accuracy': '#1976D2',      # Blue
        'Binary F1':       '#388E3C',      # Green
        'Binary AUC':      '#7B1FA2',      # Purple
        '3-Class Accuracy': '#FFA000',     # Amber
        '3-Class F1':      '#D32F2F',      # Red
        '3-Class AUC':     '#00897B',      # Teal
    }

    for label, pos in positions.items():
        if '3-Class' in label:
            metric = label.split(' ')[-1].lower()  # 'accuracy', 'f1', or 'auc'
            values = [threeclass[m][metric] for m in models]
        else:
            metric = label.split(' ')[-1].lower()  # 'accuracy', 'f1', or 'auc'
            values = [binary[m][metric] for m in models]

        bars = ax.bar(pos, values, width, label=label,
                      color=colors[label], edgecolor='black', linewidth=0.6,
                      alpha=0.88)

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=15, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n').title() for m in models],
                       fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_title('Model Performance Comparison: Binary vs 3-Class\n'
                 f'({duration}s Window Duration, Mean over CV Folds)',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right', ncol=2)
    ax.axhline(0.5, color='grey', lw=0.8, ls='--', alpha=0.5,
               label='Chance (binary)')
    ax.grid(axis='y', ls='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved bar plot: {output_path}")


def main():
    args = parse_arguments()
    label_mode = args.labels
    durations = args.dataset

    print("=" * 60)
    print("BAR PLOT: Accuracy & F1 for ML Models")
    print("=" * 60)
    print(f"📋 Label mode: {label_mode}")
    print(f"⏱️  Durations: {durations}s")

    for duration in durations:
        if label_mode in ('binary', 'both'):
            binary_path = RESULTS_DIR / f"ml_results_{duration}s_binary.csv"
            binary_data = load_results(binary_path)
            if binary_data is None:
                print(f"⚠️  Binary results not found for {duration}s")
                binary_data = None
            else:
                print(f"📂 Binary results ({duration}s): {len(binary_data)} models")
        else:
            binary_data = None

        if label_mode in ('3class', 'both'):
            threeclass_path = RESULTS_DIR / f"ml_results_{duration}s_3class.csv"
            threeclass_data = load_results(threeclass_path)
            if threeclass_data is None:
                print(f"⚠️  3-Class results not found for {duration}s")
                threeclass_data = None
            else:
                print(f"📂 3-Class results ({duration}s): {len(threeclass_data)} models")
        else:
            threeclass_data = None

        if binary_data is None and threeclass_data is None:
            print(f"❌ No results found for {duration}s duration. Skipping...")
            print("   Make sure to run:")
            print("   py src/main.py -m -d <duration> -l binary")
            print("   py src/main.py -m -d <duration> -l 3class")
            continue

        if label_mode == 'both' and binary_data is not None and threeclass_data is not None:
            output_path = RESULTS_DIR / f"ml_barplot_accuracy_f1_{duration}s.png"
            plot_side_by_side_bars(binary_data, threeclass_data, duration, output_path)
        elif label_mode == 'binary' and binary_data is not None:
            output_path = RESULTS_DIR / f"ml_barplot_accuracy_f1_{duration}s_binary.png"
            plot_single_mode(binary_data, 'binary', duration, output_path)
        elif label_mode == '3class' and threeclass_data is not None:
            output_path = RESULTS_DIR / f"ml_barplot_accuracy_f1_{duration}s_3class.png"
            plot_single_mode(threeclass_data, '3class', duration, output_path)
        elif label_mode == 'both' and binary_data is not None:
            print("⚠️  Only binary data available — plotting single mode")
            output_path = RESULTS_DIR / f"ml_barplot_accuracy_f1_{duration}s_binary.png"
            plot_single_mode(binary_data, 'binary', duration, output_path)
        elif label_mode == 'both' and threeclass_data is not None:
            print("⚠️  Only 3-class data available — plotting single mode")
            output_path = RESULTS_DIR / f"ml_barplot_accuracy_f1_{duration}s_3class.png"
            plot_single_mode(threeclass_data, '3class', duration, output_path)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()