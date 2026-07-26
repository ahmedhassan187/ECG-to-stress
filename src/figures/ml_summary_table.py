"""
Generate a styled summary table figure showing best model per duration
with Accuracy, F1, and AUC metrics.

Usage:
    py src/figures/ml_summary_table.py

Output:
    ../results/ml_results/ml_summary_table.png
    ../results/ml_results/ml_summary_table.csv
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Ensure src is on path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Change to project root
project_root = src_dir.parent
os.chdir(str(project_root))

RESULTS_DIR = Path("../results/ml_results")
OUTPUT_DIR = RESULTS_DIR


def load_results():
    """Load all ml_results_<duration>s.csv files."""
    data = {}
    for csv_path in sorted(RESULTS_DIR.glob("ml_results_*s.csv")):
        dur_str = csv_path.stem.replace("ml_results_", "").replace("s", "")
        try:
            duration = int(dur_str)
        except ValueError:
            continue
        df = pd.read_csv(csv_path, index_col="model")
        data[duration] = df
    return data


def find_best_model(df, metric="f1_mean"):
    """Find the model with the highest value of the given metric."""
    if metric not in df.columns:
        return None, np.nan
    best_idx = df[metric].idxmax()
    return best_idx, df.loc[best_idx, metric]


def compute_auc_from_saved_models(durations, feature_names=None):
    """
    Load saved models and compute AUC on held-out data.
    
    We do LOSO for each saved model to get y_pred_proba and compute AUC.
    This requires the full WESAD dataset.
    """
    from data import Data
    from features import Features
    from label_config import LabelConfig
    from ml import ML

    # Try loading models and computing AUC
    model_dir = RESULTS_DIR / "saved_models"
    if not model_dir.exists():
        return {}

    # Load WESAD data
    try:
        data_loader = Data(fs=700)
        feature_extractor = Features(fs=700)
        label_cfg = LabelConfig("binary")
        dataset_path = project_root / "data" / "WESAD"
        ecgs, labels = data_loader.read_dataset(str(dataset_path))
    except Exception as e:
        print(f"  ⚠️ Could not load WESAD dataset for AUC: {e}")
        return {}

    ml_evaluator = ML(random_state=42)
    auc_results = {}

    for duration in durations:
        auc_results[duration] = {}
        for model_path in sorted(model_dir.glob(f"*_{duration}s.pkl")):
            # Skip meta files
            if ".meta." in model_path.name:
                continue
            model_name = model_path.stem.replace(f"_{duration}s", "")
            try:
                import joblib
                model = joblib.load(model_path)
                if not hasattr(model, "predict_proba"):
                    continue

                # Run LOSO to get predictions with probabilities
                chunk_size = duration * 700
                all_chunks = []
                all_labels = []
                all_subject_ids = []

                for subj_id, (ecg, label) in enumerate(zip(ecgs, labels)):
                    valid_mask = np.isin(label, label_cfg.VALID_LABELS)
                    valid_ecg = ecg[valid_mask]
                    valid_label = label[valid_mask]

                    for i in range(0, len(valid_ecg) - chunk_size + 1, chunk_size):
                        chunk = valid_ecg[i:i + chunk_size]
                        chunk_label = valid_label[i]

                        all_chunks.append(chunk)
                        target_label = label_cfg.map_label(int(chunk_label))
                        all_labels.append(target_label)
                        all_subject_ids.append(subj_id)

                # Extract features
                feature_list = []
                for chunk in all_chunks:
                    try:
                        features_dict = feature_extractor.get_hrv_features(chunk)
                        feature_values = [
                            features_dict.get("mean_rr", np.nan),
                            features_dict.get("mean_hr", np.nan),
                            features_dict.get("sdnn", np.nan),
                            features_dict.get("rmssd", np.nan),
                            features_dict.get("pnn50", np.nan),
                            features_dict.get("lf_power", np.nan),
                            features_dict.get("hf_power", np.nan),
                            features_dict.get("lf_hf_ratio", np.nan),
                        ]
                        feature_list.append(feature_values)
                    except Exception:
                        feature_list.append([np.nan] * 8)

                feature_names_list = [
                    "mean_rr", "mean_hr", "sdnn", "rmssd",
                    "pnn50", "lf_power", "hf_power", "lf_hf_ratio"
                ]
                df = pd.DataFrame(feature_list, columns=feature_names_list)
                df["label"] = all_labels
                df["subject_id"] = all_subject_ids

                # Impute NaN
                nan_count = df[feature_names_list].isna().sum().sum()
                if nan_count > 0:
                    fill_values = df[feature_names_list].median()
                    fill_values = fill_values.fillna(0)
                    df[feature_names_list] = df[feature_names_list].fillna(fill_values)

                # LOSO evaluate with probabilities
                from sklearn.base import clone
                subjects = sorted(df["subject_id"].unique())
                all_y_true = []
                all_y_pred_proba = []

                for test_subj in subjects:
                    train_df = df[df["subject_id"] != test_subj]
                    test_df = df[df["subject_id"] == test_subj]

                    X_train = train_df[feature_names_list].values
                    y_train = train_df["label"].values
                    X_test = test_df[feature_names_list].values

                    model_clone = clone(model)
                    model_clone.fit(X_train, y_train)

                    y_proba = model_clone.predict_proba(X_test)
                    all_y_pred_proba.append(y_proba)
                    all_y_true.extend(test_df["label"].values)

                y_true = np.array(all_y_true)
                y_pred_proba = np.vstack(all_y_pred_proba)

                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                except Exception:
                    auc = np.nan

                auc_results[duration][model_name] = auc
                print(f"  AUC {model_name} ({duration}s): {auc:.4f}")

            except Exception as e:
                print(f"  Skipping {model_path.name}: {e}")
                continue

    return auc_results


def build_summary_table(data):
    """
    Build the summary table (Duration | Best Model | Accuracy | F1 | AUC).
    
    Best model is chosen by F1 score.
    """
    durations = sorted(data.keys())
    rows = []

    # Try to compute AUC from saved models
    print("Computing AUC from saved models (LOSO CV)...")
    auc_data = compute_auc_from_saved_models(durations)
    print("Done.\n")

    for duration in durations:
        df = data[duration]
        best_model, best_f1 = find_best_model(df, "f1_mean")
        if best_model is None:
            continue

        best_acc = df.loc[best_model, "accuracy_mean"] if best_model in df.index else np.nan

        # Get AUC for best model at this duration
        best_auc = np.nan
        if duration in auc_data and best_model in auc_data[duration]:
            best_auc = auc_data[duration][best_model]

        rows.append({
            "Duration": f"{duration} s",
            "Best Model": best_model.replace("_", " ").title(),
            "Accuracy": best_acc,
            "F1": best_f1,
            "AUC": best_auc,
        })

    return pd.DataFrame(rows)


def plot_table(df, output_path):
    """Render the summary table as a styled PNG figure."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")

    col_labels = list(df.columns)
    cell_text = []
    for _, row in df.iterrows():
        cell_text.append([
            row["Duration"],
            row["Best Model"],
            f'{row["Accuracy"]:.4f}',
            f'{row["F1"]:.4f}',
            f'{row["AUC"]:.4f}' if not np.isnan(row["AUC"]) else "N/A",
        ])

    n_rows = len(cell_text)
    n_cols = len(col_labels)

    # Color code the metric cells (columns 2,3,4)
    cmap = plt.cm.RdYlGn

    cell_colors = []
    for r in range(n_rows):
        row_colors = []
        for c in range(n_cols):
            val = df.iloc[r, c]
            if c >= 2:  # Metric columns
                if isinstance(val, (int, float)) and not np.isnan(val):
                    # Normalize to [0, 0.5, 1.0] range for cmap
                    norm = np.clip((val - 0.3) / 0.7, 0, 1)
                    row_colors.append(cmap(norm))
                else:
                    row_colors.append((0.9, 0.9, 0.9, 1.0))
            else:
                row_colors.append((0.95, 0.95, 0.95, 1.0))
        cell_colors.append(row_colors)

    # Create table
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.7, 2.4)

    # Style header row
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold", fontsize=13)
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=13)
        cell.set_edgecolor("#bdc3c7")
        cell.set_linewidth(1.5)

    # Center header cells
    for c in range(n_cols):
        cell = table[0, c]
        cell.set_text_props(ha="center", va="center")

    ax.set_title(
        "Best Model Performance by Window Duration",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved summary table: {output_path}")


def main():
    print("=" * 60)
    print("ML SUMMARY TABLE GENERATOR")
    print("=" * 60)

    data = load_results()
    if not data:
        print("❌ No ML results CSV files found.")
        print(f"   Looked in: {RESULTS_DIR.resolve()}")
        return

    durations = sorted(data.keys())
    print(f"📂 Found results for durations: {durations} s")
    for d in durations:
        print(f"   {d}s: {list(data[d].index)}")

    summary_df = build_summary_table(data)
    print("\n📋 Summary table:")
    print(summary_df.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = OUTPUT_DIR / "ml_summary_table.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")

    # Plot table figure
    png_path = OUTPUT_DIR / "ml_summary_table.png"
    plot_table(summary_df, png_path)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()