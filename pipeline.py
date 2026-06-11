import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import pingouin as pg
import os
import pickle
import warnings


class Pipeline:
    def __init__(self, label, big_chunck_size, small_chunck_size, chunk_method="random"):
        self.label = label
        self.big_chunck_size = big_chunck_size
        self.small_chunck_size = small_chunck_size
        self.chunk_method = chunk_method  # Default chunk method
# Reading WESAD dataset
    def read_wesad(self,folder_path):
        ecg_list = []
        label_list = []

        for subject_folder in sorted(os.listdir(folder_path)):

            subject_path = os.path.join(folder_path, subject_folder)

            # skip non-folders
            if not os.path.isdir(subject_path):
                print(f"Skipping non-folder: {subject_path}")
                continue

            # find .pkl file inside subject folder
            for file in os.listdir(subject_path):

                if file.endswith(".pkl"):
                    print(f"Processing file: {file} in {subject_path}")

                    file_path = os.path.join(subject_path, file)

                    with open(file_path, "rb") as f:
                        data = pickle.load(f, encoding="latin1")

                    ecg = data["signal"]["chest"]["ECG"].flatten()
                    labels = data["label"].flatten()

                    ecg_list.append(ecg)
                    label_list.append(labels)

        return ecg_list, label_list
    def get_ecg_by_label(self,ecgs_list, labels_list, label=1):
        """
        Returns ECG segments matching a specific label
        """
        filtered = []

        for ecg, labels in zip(ecgs_list, labels_list):

            mask = (labels == label)

            filtered.append(ecg[mask])

        return filtered
# Cut data into chunks
    def random_chunk(self, ecg_subjetcs, fs=700, time_in_sec=30):
        """
        Returns a random chunk of the ECG signal of specified duration
        """
        chunk_size = fs * time_in_sec
        ecg_random_subjects = []
        for ecg_big_chuncks in ecg_subjetcs:
            random_chunks = []
            for ecg in ecg_big_chuncks:
                if len(ecg) < chunk_size:
                    raise ValueError("ECG signal is shorter than the requested chunk size.")
                random_int = np.random.randint(0, len(ecg) - chunk_size + 1)
                random_chunks.append(ecg[random_int:random_int + chunk_size])
            ecg_random_subjects.append(random_chunks)      
        return ecg_random_subjects

    def chunk_ecg(self, ecg_list, fs=700, time_in_sec=30):
        
        all_chunks = []
    
        win = fs * time_in_sec
    
        for ecg in ecg_list:
        
            chunks = [
                ecg[i:i+win]
                for i in range(0, len(ecg)-win+1, win)
            ]
    
            all_chunks.append(chunks)
    
        return all_chunks
    def get_first_chunck(self,ecg_subjetcs, fs=700, time_in_sec=30):
        """
        Returns the first chunk of the ECG signal of specified duration
        """
        ecg_small_subjects = []
        chunk_size = fs * time_in_sec
        # print(f"Chunk size in samples: {chunk_size}")
        for ecg_big_chuncks in ecg_subjetcs:
            small_chunks = []
            for ecg in ecg_big_chuncks:
                # print(f"Processing ECG of length: {len(ecg)} samples")
                small_chunks.append(ecg[:chunk_size])
            ecg_small_subjects.append(small_chunks)  
        return ecg_small_subjects
# Parameters and then statistics
    def get_hrv_parameters(self, ecg_chunks_list, fs=700):
        
        rmssd_all = []
        sdnn_all = []
    
        for subject_chunks in ecg_chunks_list:
        
            rmssd_sub = []
            sdnn_sub = []
    
            for chunk in subject_chunks:
            
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        cleaned = nk.ecg_clean(chunk, sampling_rate=fs)
        
                        _, peaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
        
                        rpeaks = peaks["ECG_R_Peaks"]
        
                        if len(rpeaks) < 3:
                            continue
                        
                        rr = np.diff(rpeaks) / fs * 1000
        
                        sdnn = np.std(rr, ddof=1)
                        rmssd = np.sqrt(np.mean(np.diff(rr)**2))
        
                        rmssd_sub.append(rmssd)
                        sdnn_sub.append(sdnn)
    
                except:
                    continue
                
            rmssd_all.append(rmssd_sub)
            sdnn_all.append(sdnn_sub)
    
        return rmssd_all, sdnn_all
    def get_statistics(self,feature1_list, feature2_list):
        """
        Calculate statistics (R, MAE, ICC) for each subject.

        Parameters:
        - feature1_list: list of features for each subject from big chunks [[1,2,3], [4,5,6]]
        - feature2_list: list of features for each subject from small chunks [[1,2,3], [4,5,6]]

        Returns:
        - Lists of r, mae, icc for each subject
        """
        r_list = []
        mae_list = []
        icc_list = []

        # Iterate over each subject
        for feature1, feature2 in zip(feature1_list, feature2_list):
            feature1 = np.array(feature1).flatten()
            feature2 = np.array(feature2).flatten()

            # Match lengths
            min_len = min(len(feature1), len(feature2))
            feature1 = feature1[:min_len]
            feature2 = feature2[:min_len]

            # Skip empty or singleton arrays (Pearson R and ICC require at least 2 samples)
            if min_len < 2:
                r_list.append(np.nan)
                mae_list.append(np.nan)
                icc_list.append(np.nan)
                continue

            # Calculate Pearson R and MAE
            r, _ = pearsonr(feature1, feature2)
            mae = mean_absolute_error(feature1, feature2)

            # Restructure data for ICC in long format
            data = pd.DataFrame({
                'target': list(range(min_len)) * 2,  # measurement indices
                'rater': ['big'] * min_len + ['small'] * min_len,
                'rating': np.concatenate([feature1, feature2])
            })

            # Compute ICC (suppress pingouin's divide-by-zero RuntimeWarning
            # which fires when a chunk has near-zero variance)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                icc_table = pg.intraclass_corr(data=data, targets='target', raters='rater', ratings='rating')
            icc = icc_table.loc[2, 'ICC']

            r_list.append(r)
            mae_list.append(mae)
            icc_list.append(icc)

        return r_list, mae_list, icc_list
# results
    def show_results(self, r_list, mae_list, icc_list, feature_name, chunk_label=""):
        label = f" {chunk_label}" if chunk_label else ""
        print(f"Results for {feature_name}{label}:")
        for i, (r, mae, icc) in enumerate(zip(r_list, mae_list, icc_list)):
            print(f"Subject {i+1}: R={r:.4f}, MAE={mae:.4f}, ICC={icc:.4f}")
        return
    def show_results_table(self, r_list, mae_list, icc_list, feature_name, chunk_label):
        """
        Plot a color-coded table of results.
        - High R and ICC values: green
        - Medium R and ICC values: yellow  
        - Low R and ICC values: red
        - For MAE: low values (good) are green, medium are yellow, high are red
        """
        # Create DataFrame
        df = pd.DataFrame({
            'Subject': [f'Subject {i+1}' for i in range(len(r_list))],
            'R': r_list,
            'MAE': mae_list,
            'ICC': icc_list
        })
        df = df.set_index('Subject')

        # Define color mapping functions with better contrast
        def color_r_icc(val):
            """Color for R and ICC values (higher is better)"""
            if val >= 0.8:
                return 'background-color: #2d7a3e; color: white; font-weight: bold'  # Dark green with white text
            elif val >= 0.5:
                return 'background-color: #f4a460; color: black; font-weight: bold'  # Dark orange with black text
            else:
                return 'background-color: #c41e3a; color: white; font-weight: bold'  # Dark red with white text

        def color_mae(val):
            """Color for MAE values (lower is better)"""
            if val <= 5:
                return 'background-color: #2d7a3e; color: white; font-weight: bold'  # Dark green with white text
            elif val <= 15:
                return 'background-color: #f4a460; color: black; font-weight: bold'  # Dark orange with black text
            else:
                return 'background-color: #c41e3a; color: white; font-weight: bold'  # Dark red with white text

        # Build a title that identifies the feature (RMSSD/SDNN) and chunk type (Random/First)
        title = f"Results for {feature_name} ({chunk_label})"

        # Apply styling and add the title as a caption so it shows on the table itself
        styled_df = df.style.map(lambda x: color_r_icc(x), subset=['R', 'ICC']) \
                            .map(lambda x: color_mae(x), subset=['MAE']) \
                            .format({'R': '{:.4f}', 'MAE': '{:.4f}', 'ICC': '{:.4f}'}) \
                            .set_caption(title) \
                            .set_table_styles([{
                                'selector': 'caption',
                                'props': [('font-size', '16px'),
                                          ('font-weight', 'bold'),
                                          ('text-align', 'left'),
                                          ('padding-bottom', '8px')]
                            }])

        # Display the styled table only (no duplicate text version)
        display(styled_df)
        return None

    def run_pipeline(self, folder_path):
        # 1. Read data
        ecgs_list, labels_list = self.read_wesad(folder_path)

        # 2. Filter by label
        ecg_big_chuncks = self.get_ecg_by_label(ecgs_list, labels_list, self.label)

        # 3. Chunk ECG into big chuncks
        ecg_big_chuncks = self.chunk_ecg(ecg_big_chuncks, time_in_sec=self.big_chunck_size)

        # 4. Get random small chuncks from big chuncks
        if self.chunk_method == "random":
            ecg_small_chuncks = self.random_chunk(ecg_big_chuncks, time_in_sec=self.small_chunck_size)
        elif self.chunk_method == "first":
            ecg_small_chuncks = self.get_first_chunck(ecg_big_chuncks, time_in_sec=self.small_chunck_size)
        else:
            raise ValueError("Invalid chunk method. Use 'random' or 'first'.")
        # 5. Get HRV parameters for big and small chuncks
        rmssd_big, sdnn_big = self.get_hrv_parameters(ecg_big_chuncks)
        rmssd_small, sdnn_small = self.get_hrv_parameters(ecg_small_chuncks)

        # 6. Get statistics for RMSSD and SDNN
        r_rmssd, mae_rmssd, icc_rmssd = self.get_statistics(rmssd_big, rmssd_small)
        r_sdnn, mae_sdnn, icc_sdnn = self.get_statistics(sdnn_big, sdnn_small)
        self.show_results_table(r_rmssd, mae_rmssd, icc_rmssd, "RMSSD", f"{self.chunk_method} chunks")

        self.show_results_table(r_sdnn, mae_sdnn, icc_sdnn, "SDNN", f"{self.chunk_method} chunks")
        # self.show_results([r_rmssd], [mae_rmssd], [icc_rmssd], "RMSSD")
        # self.show_results([r_sdnn], [mae_sdnn], [icc_sdnn], "SDNN")

        return {
            'r_rmssd': r_rmssd,
            'mae_rmssd': mae_rmssd,
            'icc_rmssd': icc_rmssd,
            'r_sdnn': r_sdnn,
            'mae_sdnn': mae_sdnn,
            'icc_sdnn': icc_sdnn
        }