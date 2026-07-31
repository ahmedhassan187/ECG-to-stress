"""
Generate a side-by-side bar plot comparing Accuracy and F1 scores
for binary and 3-class classification (30s window duration).

Reads results from:
    ../results/ml_results/ml_results_30s_binary.csv
    ../results/ml_results/ml_results_30s_3class.csv

Usage:
    py src/figures/bar_plot_accuracy_f1.py

Output:
    ../results/ml_results/ml_barplot_accuracy_f1.png
"""

import sys
import os
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


def plot_side_by_side_bars(binary, threeclass, output_path):
    """
    Create a grouped bar plot:
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
                 '(30s Window Duration, Mean over CV Folds)',
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
    print("=" * 60)
    print("BAR PLOT: Accuracy & F1 for Binary vs 3-Class")
    print("=" * 60)

    # Load results from CSV files
    binary_path = RESULTS_DIR / "ml_results_30s_binary.csv"
    threeclass_path = RESULTS_DIR / "ml_results_30s_3class.csv"

    binary_data = load_results(binary_path)
    threeclass_data = load_results(threeclass_path)

    if binary_data is None or threeclass_data is None:
        print("❌ Could not load both result files. Make sure to run:")
        print("   py src/main.py -m -d 30 -l binary")
        print("   py src/main.py -m -d 30 -l 3class")
        return

    print(f"📂 Binary results: {len(binary_data)} models")
    print(f"📂 3-Class results: {len(threeclass_data)} models")

    output_path = RESULTS_DIR / "ml_barplot_accuracy_f1.png"
    plot_side_by_side_bars(binary_data, threeclass_data, output_path)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()