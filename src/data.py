import numpy as np
import pandas as pd
import os
import pickle
import warnings

from pathlib import Path


class Data:
    """
    Data class for reading and processing ECG data from WESAD dataset.
    """
    
    def __init__(self, fs=700):
        """
        Initialize Data class.
        
        Parameters:
        - fs: sampling frequency (default 700 Hz for WESAD)
        """
        self.fs = fs
    
    # Data file functions
    def read_subject(self, file_path):
        """
        Read a single subject's pickle file and return ECG and labels.
        """
        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        
        ecg = data["signal"]["chest"]["ECG"].flatten()
        label = data["label"].flatten()
        
        return ecg, label
    
    def read_dataset(self, folder_path):
        """
        Read all subjects from a folder and return lists of ECGs and labels.
        """
        ecgs = []
        labels = []
        
        for subject_folder in sorted(os.listdir(folder_path)):
            subject_path = os.path.join(folder_path, subject_folder)
            
            # skip non-folders
            if not os.path.isdir(subject_path):
                continue
            
            # find .pkl file inside subject folder
            for file in os.listdir(subject_path):
                if file.endswith(".pkl"):
                    file_path = os.path.join(subject_path, file)
                    ecg, label = self.read_subject(file_path)
                    ecgs.append(ecg)
                    labels.append(label)
        
        return ecgs, labels
    
    def get_label_sub(self, ecg, label, target_label=1):
        """
        Filter ECG segments matching a specific label for a single subject.
        """
        mask = (label == target_label)
        return ecg[mask]
    
    def get_label_dataset(self, ecgs, labels, target_label=1):
        """
        Filter ECG segments matching a specific label for all subjects.
        """
        filtered_ecgs = []
        
        for ecg, label in zip(ecgs, labels):
            filtered_ecg = self.get_label_sub(ecg, label, target_label)
            filtered_ecgs.append(filtered_ecg)
        
        return filtered_ecgs
    
    def get_random_chunk(self, long_ecg, time_in_sec=30):
        """
        Returns a random chunk of the ECG signal of specified duration.
        """
        chunk_size = self.fs * time_in_sec
        
        if len(long_ecg) < chunk_size:
            raise ValueError("ECG signal is shorter than the requested chunk size.")
        
        random_int = np.random.randint(0, len(long_ecg) - chunk_size + 1)
        return long_ecg[random_int:random_int + chunk_size]
    
    def get_chunked_ecg(self, long_ecg, time_in_sec=30):
        """
        Returns sequential non-overlapping chunks of the ECG signal.
        """
        chunk_size = self.fs * time_in_sec
        
        chunks = [
            long_ecg[i:i+chunk_size]
            for i in range(0, len(long_ecg)-chunk_size+1, chunk_size)
        ]
        
        return chunks
    
    def get_majority_label_chunks(self, ecg, label, time_in_sec=30, threshold=0.9):
        """
        Split the full ECG into chunks and assign a label to each chunk
        based on majority vote. Chunks where no single label reaches
        'threshold' proportion are discarded.
        
        Parameters:
        - ecg: 1D array of ECG signal
        - label: 1D array of label values (same length as ecg)
        - time_in_sec: chunk duration in seconds (default: 30)
        - threshold: minimum fraction needed to keep a chunk (default: 0.9)
        
        Returns:
        - keep_ecg: list of ECG chunks that passed the threshold
        - keep_labels: list of majority-label values for kept chunks
        - discarded: dict with info about discarded chunks
            {'count': int, 'labels': list, 'purities': list}
        """
        chunk_size = self.fs * time_in_sec
        n_total = len(ecg)
        n_chunks = n_total // chunk_size
        
        keep_ecg = []
        keep_labels = []
        discarded_labels = []
        discarded_purities = []
        
        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size
            lbl_chunk = label[start:end]
            
            # Count label proportions
            unique, counts = np.unique(lbl_chunk, return_counts=True)
            fractions = counts / chunk_size
            max_idx = np.argmax(fractions)
            max_label = unique[max_idx]
            max_frac = fractions[max_idx]
            
            if max_frac >= threshold:
                keep_ecg.append(ecg[start:end])
                keep_labels.append(max_label)
            else:
                discarded_labels.append(max_label)
                discarded_purities.append(max_frac)
        
        discarded = {
            'count': len(discarded_labels),
            'labels': discarded_labels,
            'purities': discarded_purities
        }
        
        return keep_ecg, keep_labels, discarded

    # ── Utility ──────────────────────────────────────────────────────────────

    @staticmethod
    def get_dataset_path(args, project_root=None):
        """
        Resolve the WESAD dataset path from CLI arguments or default.

        Parameters:
            args: parsed argparse namespace (may have .input)
            project_root: Path to project root (used for default fallback)

        Returns:
            Path to the WESAD dataset directory
        """
        if args.input:
            return Path(args.input)
        if project_root:
            return project_root / 'data' / 'WESAD'
        return Path('data/WESAD')

    @staticmethod
    def load_pavia_data(data_dir=None, features_path=None, labels_path=None,
                        project_root=None):
        """
        Load Pavia HRV data from CSV files and align to the 8 standard features.

        Parameters:
            data_dir: path to the data directory (default: project_root / 'data')
            features_path: direct path to the features CSV (overrides data_dir)
            labels_path:   direct path to the labels CSV (overrides data_dir)
            project_root:  project root Path (used for default fallback)

        Returns:
            X: np.ndarray of shape (n_samples, 8) with standard feature ordering
            y: np.ndarray of shape (n_samples,) with binary labels (0/1)
            feature_names: list of standard feature column names
        """
        # Resolve feature/label paths
        if features_path and labels_path:
            features_path = Path(features_path)
            labels_path = Path(labels_path)
        else:
            data_dir = Path(data_dir) if data_dir else (
                project_root / 'data' if project_root else Path('data')
            )
            features_path = data_dir / 'pavia_features.csv'
            labels_path = data_dir / 'pavia_labels.csv'

        if not features_path.exists():
            print(f"   ❌ Pavia features not found: {features_path}")
            return None, None, None
        if not labels_path.exists():
            print(f"   ❌ Pavia labels not found: {labels_path}")
            return None, None, None

        raw_df = pd.read_csv(features_path)
        raw_labels = pd.read_csv(labels_path)
        print(f"   ✓ Loaded {features_path.name} - shape {raw_df.shape}")
        print(f"   ✓ Loaded {labels_path.name}   - shape {raw_labels.shape}")

        # Remove all-NaN rows
        all_nan = raw_df.isna().all(axis=1)
        n_empty = int(all_nan.sum())
        if n_empty > 0:
            print(f"   Removing {n_empty} empty row(s) from features")
            valid_df = raw_df.dropna(how='all').reset_index(drop=True)
            valid_labels = raw_labels.loc[~all_nan].reset_index(drop=True)
        else:
            valid_df = raw_df
            valid_labels = raw_labels

        # Map Pavia column names to standard names
        pavia_to_standard = {
            'HR': 'mean_hr', 'SDNN': 'sdnn', 'rMSSD': 'rmssd',
            'pNN50': 'pnn50', 'SE': None, 'LF': 'lf_power',
            'HF': 'hf_power', 'LFHF': 'lf_hf_ratio',
        }
        mapped = valid_df.rename(
            columns={k: v for k, v in pavia_to_standard.items() if v is not None}
        )
        cols_to_drop = [
            c for c in pavia_to_standard
            if pavia_to_standard[c] is None and c in mapped.columns
        ]
        if cols_to_drop:
            mapped = mapped.drop(columns=cols_to_drop)

        # Compute mean_rr from HR
        if 'mean_hr' in mapped.columns:
            mapped['mean_rr'] = 60000.0 / mapped['mean_hr']

        # Select only the 8 standard features
        standard_features = [
            'mean_rr', 'mean_hr', 'sdnn', 'rmssd', 'pnn50',
            'lf_power', 'hf_power', 'lf_hf_ratio'
        ]
        available_features = [f for f in standard_features if f in mapped.columns]
        missing = set(standard_features) - set(available_features)
        if missing:
            print(f"   ⚠️  Missing standard features: {missing}")

        X = mapped[available_features].values.astype(np.float64)
        y = valid_labels.values.ravel().astype(int)

        # Handle NaN in features
        nan_mask = ~np.isnan(X).any(axis=1)
        n_nan_rows = (~nan_mask).sum()
        if n_nan_rows > 0:
            print(f"   Removing {n_nan_rows} row(s) with NaN values")
            X = X[nan_mask]
            y = y[nan_mask]

        print(f"   ✓ Pavia data ready: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, available_features
