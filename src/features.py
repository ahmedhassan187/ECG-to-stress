import numpy as np
import pandas as pd
import neurokit2 as nk
import warnings

class Features:
    """
    Features class for extracting HRV features from ECG signals.
    """
    
    def __init__(self, fs=700):
        """
        Initialize Features class.
        
        Parameters:
        - fs: sampling frequency (default 700 Hz for WESAD)
        """
        self.fs = fs
    
    def _get_rr_intervals(self, ecg):
        """
        Helper method to extract RR intervals from ECG signal.
        
        Parameters:
        - ecg: ECG signal array
        
        Returns:
        - rr: RR intervals in milliseconds
        - rpeaks: R peak indices
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            cleaned = nk.ecg_clean(ecg, sampling_rate=self.fs)
            _, peaks = nk.ecg_peaks(cleaned, sampling_rate=self.fs)
            rpeaks = peaks["ECG_R_Peaks"]
            
            if len(rpeaks) < 3:
                return None, None
            
            rr = np.diff(rpeaks) / self.fs * 1000  # Convert to milliseconds
            return rr, rpeaks
    
    def get_mean_rr(self, ecg):
        """
        Calculate mean RR interval.
        
        Parameters:
        - ecg: ECG signal array
        
        Returns:
        - mean_rr: mean RR interval in milliseconds
        """
        rr, _ = self._get_rr_intervals(ecg)
        if rr is None or len(rr) == 0:
            return np.nan
        return np.mean(rr)
    
    def get_mean_hr(self, ecg):
        """
        Calculate mean heart rate from RR intervals.
        
        Parameters:
        - ecg: ECG signal array
        
        Returns:
        - mean_hr: mean heart rate in beats per minute (BPM)
        """
        rr, _ = self._get_rr_intervals(ecg)
        if rr is None or len(rr) == 0:
            return np.nan
        # HR = 60,000 / RR (ms)
        hr = 60000 / rr
        return np.mean(hr)
    
    def get_rmssd(self, ecg):
        """
        Calculate RMSSD (Root Mean Square of Successive Differences).
        
        Parameters:
        - ecg: ECG signal array
        
        Returns:
        - rmssd: RMSSD value in milliseconds
        """
        rr, _ = self._get_rr_intervals(ecg)
        if rr is None or len(rr) < 2:
            return np.nan
        diff_rr = np.diff(rr)
        rmssd = np.sqrt(np.mean(diff_rr**2))
        return rmssd
    
    def get_sdnn(self, ecg):
        """
        Calculate SDNN (Standard Deviation of NN intervals).
        
        Parameters:
        - ecg: ECG signal array
        
        Returns:
        - sdnn: SDNN value in milliseconds
        """
        rr, _ = self._get_rr_intervals(ecg)
        if rr is None or len(rr) == 0:
            return np.nan
        sdnn = np.std(rr, ddof=1)
        return sdnn
    
    def get_pnn50(self, ecg):
        """
        Calculate pNN50 (percentage of successive RR differences > 50ms).
        
        Parameters:
        - ecg: ECG signal array
        
        Returns:
        - pnn50: pNN50 percentage (0-100)
        """
        rr, _ = self._get_rr_intervals(ecg)
        if rr is None or len(rr) < 2:
            return np.nan
        diff_rr = np.abs(np.diff(rr))
        pnn50 = np.sum(diff_rr > 50) / len(diff_rr) * 100
        return pnn50
    
    def get_lf_power(self, ecg):
        """
        Calculate LF (Low Frequency) power from RR intervals.
        Range: 0.04 - 0.15 Hz
        
        Parameters:
        - ecg: ECG signal array
        
        Returns:
        - lf_power: LF power in ms²
        """
        rr, _ = self._get_rr_intervals(ecg)
        if rr is None or len(rr) < 10:
            return np.nan
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # Interpolate RR intervals for even sampling
                rr_interpolated = np.interp(
                    np.linspace(0, len(rr), len(rr) * 4),
                    np.arange(len(rr)),
                    rr
                )
                
                # Calculate power spectral density
                from scipy import signal
                freqs, psd = signal.welch(rr_interpolated, fs=4, nperseg=min(256, len(rr_interpolated)))
                
                # Find LF band (0.04 - 0.15 Hz)
                lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
                lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
                
                return lf_power
        except:
            return np.nan
    
    def get_hf_power(self, ecg):
        """
        Calculate HF (High Frequency) power from RR intervals.
        Range: 0.15 - 0.4 Hz
        
        Parameters:
        - ecg: ECG signal array
        
        Returns:
        - hf_power: HF power in ms²
        """
        rr, _ = self._get_rr_intervals(ecg)
        if rr is None or len(rr) < 10:
            return np.nan
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # Interpolate RR intervals for even sampling
                rr_interpolated = np.interp(
                    np.linspace(0, len(rr), len(rr) * 4),
                    np.arange(len(rr)),
                    rr
                )
                
                # Calculate power spectral density
                from scipy import signal
                freqs, psd = signal.welch(rr_interpolated, fs=4, nperseg=min(256, len(rr_interpolated)))
                
                # Find HF band (0.15 - 0.4 Hz)
                hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
                hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])
                
                return hf_power
        except:
            return np.nan
    
    def get_lf_hf_ratio(self, ecg):
        """
        Calculate LF/HF ratio.
        
        Parameters:
        - ecg: ECG signal array
        
        Returns:
        - lf_hf_ratio: LF/HF ratio
        """
        lf_power = self.get_lf_power(ecg)
        hf_power = self.get_hf_power(ecg)
        
        if np.isnan(lf_power) or np.isnan(hf_power) or hf_power == 0:
            return np.nan
        
        return lf_power / hf_power
    
    def get_hrv_features(self, ecg):
        """
        Extract all HRV features from a single ECG signal.
        
        Parameters:
        - ecg: ECG signal array
        
        Returns:
        - features_dict: dictionary containing all HRV features
        """
        features_dict = {
            'mean_rr': self.get_mean_rr(ecg),
            'mean_hr': self.get_mean_hr(ecg),
            'sdnn': self.get_sdnn(ecg),
            'rmssd': self.get_rmssd(ecg),
            'pnn50': self.get_pnn50(ecg),
            'lf_power': self.get_lf_power(ecg),
            'hf_power': self.get_hf_power(ecg),
            'lf_hf_ratio': self.get_lf_hf_ratio(ecg)
        }
        
        return features_dict
    
    def get_hrv_features_batch(self, ecg_list, verbose=True):
        """
        Extract all HRV features from a list of ECG signals.
        
        Parameters:
        - ecg_list: list of ECG signal arrays
        - verbose: whether to print progress
        
        Returns:
        - features_list: list of feature dictionaries
        """
        features_list = []
        
        for i, ecg in enumerate(ecg_list):
            if verbose and i % 10 == 0:
                print(f"Processing ECG {i+1}/{len(ecg_list)}...")
            
            features_dict = self.get_hrv_features(ecg)
            features_list.append(features_dict)
        
        return features_list
    
    def get_dataframe(self, features_list, labels=None):
        """
        Convert features to DataFrame.
        
        Parameters:
        - features_list: list of feature dictionaries or 2D array
        - labels: optional list of labels for each sample
        
        Returns:
        - df: DataFrame with features and optional labels
        """
        # If features_list is already a list of dictionaries
        if isinstance(features_list[0], dict):
            df = pd.DataFrame(features_list)
        else:
            # Assume it's a 2D array with columns in standard order
            feature_names = ['mean_rr', 'mean_hr', 'sdnn', 'rmssd', 
                           'pnn50', 'lf_power', 'hf_power', 'lf_hf_ratio']
            df = pd.DataFrame(features_list, columns=feature_names)
        
        # Add labels if provided
        if labels is not None:
            if len(labels) == len(df):
                df['label'] = labels
            else:
                raise ValueError("Length of labels must match number of samples in features_list")
        
        return df
    
    # Convenience methods for single feature extraction on batches
    def get_mean_rr_batch(self, ecg_list):
        """Extract mean RR for a list of ECG signals."""
        return [self.get_mean_rr(ecg) for ecg in ecg_list]
    
    def get_mean_hr_batch(self, ecg_list):
        """Extract mean HR for a list of ECG signals."""
        return [self.get_mean_hr(ecg) for ecg in ecg_list]
    
    def get_sdnn_batch(self, ecg_list):
        """Extract SDNN for a list of ECG signals."""
        return [self.get_sdnn(ecg) for ecg in ecg_list]
    
    def get_rmssd_batch(self, ecg_list):
        """Extract RMSSD for a list of ECG signals."""
        return [self.get_rmssd(ecg) for ecg in ecg_list]
    
    def get_pnn50_batch(self, ecg_list):
        """Extract pNN50 for a list of ECG signals."""
        return [self.get_pnn50(ecg) for ecg in ecg_list]
    
    def get_lf_power_batch(self, ecg_list):
        """Extract LF power for a list of ECG signals."""
        return [self.get_lf_power(ecg) for ecg in ecg_list]
    
    def get_hf_power_batch(self, ecg_list):
        """Extract HF power for a list of ECG signals."""
        return [self.get_hf_power(ecg) for ecg in ecg_list]
    
    def get_lf_hf_ratio_batch(self, ecg_list):
        """Extract LF/HF ratio for a list of ECG signals."""
        return [self.get_lf_hf_ratio(ecg) for ecg in ecg_list]

    # ---------------- FFT helpers ----------------
    def compute_fft(self, ecg):
        """
        Compute the FFT of a single ECG chunk.

        Steps:
            1. Clean the ECG with neurokit2.
            2. Remove the DC component (mean subtraction).
            3. Apply a Hann window to reduce spectral leakage.
            4. Compute the real FFT and return the magnitude spectrum
               and the matching frequency axis (one-sided, Hz).

        Parameters:
        - ecg: 1D array-like ECG signal (one chunk).

        Returns:
        - freqs: 1D np.ndarray of positive frequencies in Hz.
        - magnitude: 1D np.ndarray of FFT magnitudes (same length as freqs).
        """
        # Convert to numpy array (handle list / pandas input)
        ecg = np.asarray(ecg, dtype=np.float64).flatten()

        if ecg.size < 2:
            return np.array([]), np.array([])

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            cleaned = nk.ecg_clean(ecg, sampling_rate=self.fs)

        n = cleaned.size

        # DC removal + Hann window to reduce spectral leakage
        centered = cleaned - np.mean(cleaned)
        window = np.hanning(n)
        windowed = centered * window

        # Real FFT (rfft) → one-sided spectrum
        fft_vals = np.fft.rfft(windowed)
        magnitude = np.abs(fft_vals) * (2.0 / np.sum(window))

        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)

        return freqs, magnitude

    def compute_fft_batch(self, ecg_list, verbose=True):
        """
        Compute the FFT for every ECG chunk in a list.

        Parameters:
        - ecg_list: list of ECG signal arrays (chunks).
        - verbose: print progress every 10 chunks.

        Returns:
        - results: list of tuples (freqs, magnitude) in the same order
                   as ecg_list. Chunks that fail return empty arrays.
        """
        results = []

        for i, ecg in enumerate(ecg_list):
            if verbose and i % 10 == 0:
                print(f"Computing FFT {i+1}/{len(ecg_list)}...")

            try:
                freqs, magnitude = self.compute_fft(ecg)
                results.append((freqs, magnitude))
            except Exception as e:
                # Keep alignment with the input list using empty arrays
                if verbose:
                    print(f"  ⚠️ FFT failed on chunk {i+1}: {e}")
                results.append((np.array([]), np.array([])))

        return results