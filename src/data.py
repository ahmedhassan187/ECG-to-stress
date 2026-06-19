import numpy as np
import pandas as pd
import os
import pickle
import warnings

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
