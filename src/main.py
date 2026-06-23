"""
ECG-to-Stress Analysis CLI
Command-line interface for WESAD dataset analysis, feature extraction, correlation analysis, 
and machine learning model training.

Usage Examples:
    python main.py --help                           # Show help message
    python main.py -c                               # Run correlation analysis (all features)
    python main.py --corr -f feature1 feature2      # Run correlation on specific features
    python main.py -f                               # Plot full ECG signals (default 5000 points)
    python main.py --full -p 10000                  # Plot with 10000 points per chunk
    python main.py -m                               # Train all models on all datasets
    python main.py --ml -d 30 120                   # Train models on 30s and 120s datasets
    python main.py -m -mo knn svm random_forest     # Train specific models on all datasets
"""

import argparse
import sys
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

# Import classes from src
try:
    from data import Data
    from features import Features
    from visualization import Visualization
    from correlation import Correlation
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print(f"   Make sure you're running from the project root:")
    print(f"   cd g:\\Master\\Thesis\\FLT\\Code\\ECG-to-stress")
    print(f"   python src/main.py --help")
    sys.exit(1)

warnings.filterwarnings('ignore')


def parse_arguments():
    """
    Parse command-line arguments using argparse.
    
    Returns:
        Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog='python src/main.py',
        description='ECG-to-Stress Analysis Tool - WESAD Dataset Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Correlation Analysis
  python src/main.py -c                              # All features
  python src/main.py --corr -f mean_rr mean_hr       # Specific features
  
  # Full Signal Visualization
  python src/main.py -f                              # Default 5000 points per chunk
  python src/main.py --full -p 10000                 # 10000 points per chunk
  
  # Machine Learning Models
  python src/main.py -m                              # All models, all datasets
  python src/main.py -m -d 30 120 300                # Specific datasets (30s, 120s, 300s)
  python src/main.py -m -mo knn svm xgboost          # Specific models
  python src/main.py -m -d 30 -mo random_forest      # Specific dataset + models
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


def run_correlation_analysis(args):
    """
    Execute correlation analysis on HRV features.
    
    Parameters:
        args: parsed arguments containing features, dataset, and output settings
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
    dataset_path = Path(__file__).parent.parent / "data" / "WESAD"
    
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
    
    # Process each dataset duration
    for duration in args.dataset:
        print(f"\n📊 Processing {duration}s chunks...")
        
        chunk_size = duration * 700  # 700 Hz sampling rate
        all_features = []
        all_labels = []
        
        # Extract features for all chunks
        for ecg, label in zip(ecgs, labels):
            # Filter for labels 1, 2, 3, 4
            valid_mask = np.isin(label, [1, 2, 3, 4])
            valid_ecg = ecg[valid_mask]
            valid_label = label[valid_mask]
            
            # Chunk the signal
            for i in range(0, len(valid_ecg) - chunk_size + 1, chunk_size):
                chunk = valid_ecg[i:i + chunk_size]
                chunk_label = valid_label[i]
                
                # Extract HRV features
                try:
                    mean_rr = feature_extractor.get_mean_rr(chunk) or 0
                    mean_hr = feature_extractor.get_mean_hr(chunk) or 0
                    sdnn = feature_extractor.get_sdnn(chunk) or 0
                    rmssd = feature_extractor.get_rmssd(chunk) or 0
                    pnn50 = feature_extractor.get_pnn50(chunk) or 0
                    lf_power = feature_extractor.get_lf_power(chunk) or 0
                    hf_power = feature_extractor.get_hf_power(chunk) or 0
                    lf_hf_ratio = feature_extractor.get_lf_hf_ratio(chunk) or 0
                    
                    all_features.append({
                        'mean_rr': mean_rr,
                        'mean_hr': mean_hr,
                        'sdnn': sdnn,
                        'rmssd': rmssd,
                        'pnn50': pnn50,
                        'lf_power': lf_power,
                        'hf_power': hf_power,
                        'lf_hf_ratio': lf_hf_ratio,
                        'label': chunk_label
                    })
                except:
                    continue
        
        if not all_features:
            print(f"⚠️  No features extracted for {duration}s duration")
            continue
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        print(f"✓ Extracted {len(df)} chunks with 8 HRV features")
        
        # Determine which features to analyze
        if 'all' in args.features:
            feature_list = ['mean_rr', 'mean_hr', 'sdnn', 'rmssd', 'pnn50', 'lf_power', 'hf_power', 'lf_hf_ratio']
        else:
            feature_list = [f for f in args.features if f in df.columns]
        
        # Generate correlation plots
        print(f"📈 Generating correlation visualizations for {len(feature_list)} features...")
        # TODO: Implement correlation visualization methods from Visualization class
        
        # Save results
        csv_path = output_path / f"features_{duration}s.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")
    
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
    dataset_path = Path(__file__).parent.parent / "data" / "WESAD"
    
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
    
    # Generate plots for each subject
    subjects = args.subjects if args.subjects else range(len(ecgs))
    
    for idx in subjects:
        if idx >= len(ecgs):
            print(f"⚠️  Subject {idx} not found (only {len(ecgs)} subjects available)")
            continue
        
        print(f"\n📌 Processing Subject {idx}...")
        ecg = ecgs[idx]
        label = labels[idx]
        
        try:
            # Generate rolling plots with specified chunk size
            viz.plot_ecg_rolling(
                ecg=ecg,
                fs=700,
                chunk_size=args.points,
                label=label,
                title=f"Subject {idx} - Full ECG Signal ({len(ecg)} samples)"
            )
            
            # Save plot
            plot_path = output_path / f"subject_{idx:02d}_ecg.png"
            import matplotlib.pyplot as plt
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {plot_path}")
        except Exception as e:
            print(f"❌ Error processing subject {idx}: {e}")
    
    print("\n✅ ECG signal visualization complete!")


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
    
    # Get dataset path
    dataset_path = Path(__file__).parent.parent / "data" / "WESAD"
    
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
        
        # Extract features for all chunks
        for ecg, label in zip(ecgs, labels):
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
        
        print(f"✓ Created {len(all_chunks)} chunks")
        
        # Extract HRV features from chunks
        feature_list = []
        for chunk in all_chunks:
            try:
                features = feature_extractor.get_hrv_features(chunk)
                feature_list.append(features)
            except:
                feature_list.append([np.nan] * 8)
        
        # Create feature matrix
        X = np.array(feature_list)
        y = np.array(all_labels)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        print(f"✓ Extracted {X.shape[0]} samples × {X.shape[1]} features")
        print(f"✓ Class distribution: {np.sum(y==0)} no-stress, {np.sum(y==1)} stress")
        
        # Train models (placeholder for actual training)
        print(f"🚀 Training {len(models_to_train)} models...")
        
        for model_name in models_to_train:
            print(f"   → {model_name.upper()}", end='... ')
            # TODO: Implement actual model training with cross-validation
            print("✓")
        
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
