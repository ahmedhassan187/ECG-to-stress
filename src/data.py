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