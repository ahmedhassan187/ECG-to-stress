import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pingouin as pg
import warnings

class Correlation:
    """
    Correlation class for analyzing relationships between features.
    Provides methods for Pearson R, ICC, MAE, and other statistical measures.
    """
    
    def __init__(self):
        """
        Initialize Correlation class.
        """
        pass
    
    def get_r(self, features1, features2, method='pearson'):
        """
        Calculate correlation coefficient between two sets of features.
        
        Parameters:
        - features1: array-like, first set of features
        - features2: array-like, second set of features
        - method: correlation method ('pearson' or 'spearman')
        
        Returns:
        - r: correlation coefficient
        - p_value: p-value (optional, returned if return_p=True)
        """
        # Convert to numpy arrays and flatten
        features1 = np.array(features1).flatten()
        features2 = np.array(features2).flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(features1) | np.isnan(features2))
        features1 = features1[mask]
        features2 = features2[mask]
        
        # Check if we have enough samples
        if len(features1) < 2 or len(features2) < 2:
            return np.nan
        
        # Calculate correlation
        if method == 'pearson':
            r, p_value = pearsonr(features1, features2)
        elif method == 'spearman':
            r, p_value = spearmanr(features1, features2)
        else:
            raise ValueError("method must be 'pearson' or 'spearman'")
        
        return r, p_value
    
    def get_r_batch(self, feature_list1, feature_list2, method='pearson'):
        """
        Calculate correlation coefficients for multiple subject pairs.
        
        Parameters:
        - feature_list1: list of arrays, first set of features for each subject
        - feature_list2: list of arrays, second set of features for each subject
        - method: correlation method ('pearson' or 'spearman')
        
        Returns:
        - r_list: list of correlation coefficients for each subject
        - p_list: list of p-values for each subject
        """
        r_list = []
        p_list = []
        
        for f1, f2 in zip(feature_list1, feature_list2):
            r, p = self.get_r(f1, f2, method)
            r_list.append(r)
            p_list.append(p)
        
        return r_list, p_list
    
    def get_icc(self, features1, features2, icc_type='ICC2'):
        """
        Calculate Intraclass Correlation Coefficient (ICC) between two sets of features.
        
        Parameters:
        - features1: array-like, first set of features (e.g., from big chunks)
        - features2: array-like, second set of features (e.g., from small chunks)
        - icc_type: type of ICC to calculate
            - 'ICC1': one-way random effects, single measurement
            - 'ICC2': two-way random effects, single measurement (default)
            - 'ICC3': two-way mixed effects, single measurement
            - 'ICC1k', 'ICC2k', 'ICC3k': for average of k measurements
        
        Returns:
        - icc: ICC value
        - conf_int: confidence interval (if available)
        """
        # Convert to numpy arrays and flatten
        features1 = np.array(features1).flatten()
        features2 = np.array(features2).flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(features1) | np.isnan(features2))
        features1 = features1[mask]
        features2 = features2[mask]
        
        # Check if we have enough samples
        n = len(features1)
        if n < 2:
            return np.nan, (np.nan, np.nan)
        
        # If all values are identical (variance=0), ICC = 1
        if (np.all(features1 == features1[0]) and 
            np.all(features2 == features2[0]) and 
            features1[0] == features2[0]):
            return 1.0, (1.0, 1.0)
        
        try:
            k = 2  # number of raters / methods
            # Grand mean of all observations
            grand_mean = np.mean(np.concatenate([features1, features2]))
            
            # Subject (target) means — average across raters
            target_means = (features1 + features2) / 2
            
            # Rater means — average across subjects
            rater_means = np.array([np.mean(features1), np.mean(features2)])
            
            # --- Sums of squares ---
            # Between subjects (targets)
            ss_between = k * np.sum((target_means - grand_mean)**2)
            df_between = n - 1
            ms_between = ss_between / df_between if df_between > 0 else 0
            
            # Between raters
            ss_rater = n * np.sum((rater_means - grand_mean)**2)
            df_rater = k - 1
            ms_rater = ss_rater / df_rater if df_rater > 0 else 0
            
            # Residual / error
            # For each observation: x_{ij} = target_mean_i + rater_mean_j - grand_mean + residual
            residuals = np.concatenate([
                features1 - target_means - rater_means[0] + grand_mean,
                features2 - target_means - rater_means[1] + grand_mean
            ])
            ss_error = np.sum(residuals**2)
            df_error = (n - 1) * (k - 1)
            ms_error = ss_error / df_error if df_error > 0 else 0
            
            # --- ICC formulas (Shrout & Fleiss) ---
            if icc_type == 'ICC1':
                # One-way random, single measure
                # Re-use ms_error but note: ICC1 ignores rater effects,
                # so ms_within = ss_error/df_error from two-way isn't correct for ICC1.
                # For ICC1: ss_within = ss_rater + ss_error, df_within = df_rater + df_error
                ss_within_total = ss_rater + ss_error
                df_within_total = df_rater + df_error
                ms_within_total = ss_within_total / df_within_total if df_within_total > 0 else 0
                if ms_within_total == 0:
                    return 1.0, (np.nan, np.nan)
                icc = (ms_between - ms_within_total) / (ms_between + (k - 1) * ms_within_total)
            elif icc_type == 'ICC2':
                # Two-way random, single measure, absolute agreement
                numerator = ms_between - ms_error
                denominator = ms_between + (k - 1) * ms_error + k * (ms_rater - ms_error) / n
                if denominator == 0:
                    return 1.0, (np.nan, np.nan)
                icc = numerator / denominator
            elif icc_type == 'ICC3':
                # Two-way mixed, single measure, consistency
                numerator = ms_between - ms_error
                denominator = ms_between + (k - 1) * ms_error
                if denominator == 0:
                    return 1.0, (np.nan, np.nan)
                icc = numerator / denominator
            elif icc_type == 'ICC2k':
                # Two-way random, average measure (k raters)
                numerator = ms_between - ms_error
                denominator = ms_between + (ms_rater - ms_error) / n
                if denominator == 0:
                    return 1.0, (np.nan, np.nan)
                icc = numerator / denominator
            elif icc_type == 'ICC3k':
                # Two-way mixed, average measure (k raters)
                numerator = ms_between - ms_error
                denominator = ms_between
                if denominator == 0:
                    return 1.0, (np.nan, np.nan)
                icc = numerator / denominator
            else:
                return np.nan, (np.nan, np.nan)
            
            icc = max(-1.0, min(1.0, icc))
            return icc, (np.nan, np.nan)
        
        except Exception as e:
            # Fallback: pingouin
            try:
                data = pd.DataFrame({
                    'target': list(range(n)) * 2,
                    'rater': ['method1'] * n + ['method2'] * n,
                    'rating': np.concatenate([features1, features2])
                })
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    icc_table = pg.intraclass_corr(data=data, targets='target', 
                                                    raters='rater', ratings='rating')
                    match = icc_table[icc_table['Type'] == icc_type]
                    if len(match) == 1:
                        icc_val = match['ICC'].values[0]
                        ci_low = match['CI2.5%'].values[0]
                        ci_high = match['CI97.5%'].values[0]
                        return icc_val, (ci_low, ci_high)
                    else:
                        return icc_table['ICC'].iloc[0], (icc_table['CI2.5%'].iloc[0], icc_table['CI97.5%'].iloc[0])
            except:
                return np.nan, (np.nan, np.nan)
    
    def get_icc_batch(self, feature_list1, feature_list2, icc_type='ICC2'):
        """
        Calculate ICC for multiple subject pairs.
        
        Parameters:
        - feature_list1: list of arrays, first set of features for each subject
        - feature_list2: list of arrays, second set of features for each subject
        - icc_type: type of ICC to calculate
        
        Returns:
        - icc_list: list of ICC values for each subject
        - conf_int_list: list of confidence intervals
        """
        icc_list = []
        conf_int_list = []
        
        for f1, f2 in zip(feature_list1, feature_list2):
            icc, conf_int = self.get_icc(f1, f2, icc_type)
            icc_list.append(icc)
            conf_int_list.append(conf_int)
        
        return icc_list, conf_int_list
    
    def get_mae(self, features1, features2):
        """
        Calculate Mean Absolute Error between two sets of features.
        
        Parameters:
        - features1: array-like, first set of features (ground truth/predicted)
        - features2: array-like, second set of features (predicted/ground truth)
        
        Returns:
        - mae: Mean Absolute Error
        """
        # Convert to numpy arrays and flatten
        features1 = np.array(features1).flatten()
        features2 = np.array(features2).flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(features1) | np.isnan(features2))
        features1 = features1[mask]
        features2 = features2[mask]
        
        # Check if we have samples
        if len(features1) == 0 or len(features2) == 0:
            return np.nan
        
        # Match lengths
        min_len = min(len(features1), len(features2))
        features1 = features1[:min_len]
        features2 = features2[:min_len]
        
        return mean_absolute_error(features1, features2)
    
    def get_mae_batch(self, feature_list1, feature_list2):
        """
        Calculate MAE for multiple subject pairs.
        
        Parameters:
        - feature_list1: list of arrays, first set of features for each subject
        - feature_list2: list of arrays, second set of features for each subject
        
        Returns:
        - mae_list: list of MAE values for each subject
        """
        mae_list = []
        
        for f1, f2 in zip(feature_list1, feature_list2):
            mae = self.get_mae(f1, f2)
            mae_list.append(mae)
        
        return mae_list
    
    def get_rmse(self, features1, features2):
        """
        Calculate Root Mean Square Error between two sets of features.
        
        Parameters:
        - features1: array-like, first set of features
        - features2: array-like, second set of features
        
        Returns:
        - rmse: Root Mean Square Error
        """
        # Convert to numpy arrays and flatten
        features1 = np.array(features1).flatten()
        features2 = np.array(features2).flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(features1) | np.isnan(features2))
        features1 = features1[mask]
        features2 = features2[mask]
        
        # Check if we have samples
        if len(features1) == 0 or len(features2) == 0:
            return np.nan
        
        # Match lengths
        min_len = min(len(features1), len(features2))
        features1 = features1[:min_len]
        features2 = features2[:min_len]
        
        return np.sqrt(mean_squared_error(features1, features2))
    
    def get_all_metrics(self, features1, features2):
        """
        Calculate all metrics (R, ICC, MAE, RMSE) at once.
        
        Parameters:
        - features1: array-like, first set of features
        - features2: array-like, second set of features
        
        Returns:
        - metrics_dict: dictionary containing all metrics
        """
        r, p_value = self.get_r(features1, features2)
        icc, icc_conf_int = self.get_icc(features1, features2)
        mae = self.get_mae(features1, features2)
        rmse = self.get_rmse(features1, features2)
        
        metrics_dict = {
            'pearson_r': r,
            'p_value': p_value,
            'icc': icc,
            'icc_ci_lower': icc_conf_int[0],
            'icc_ci_upper': icc_conf_int[1],
            'mae': mae,
            'rmse': rmse
        }
        
        return metrics_dict
    
    def get_all_metrics_batch(self, feature_list1, feature_list2):
        """
        Calculate all metrics for multiple subject pairs.
        
        Parameters:
        - feature_list1: list of arrays, first set of features for each subject
        - feature_list2: list of arrays, second set of features for each subject
        
        Returns:
        - metrics_df: DataFrame with all metrics for each subject
        """
        results = []
        
        for i, (f1, f2) in enumerate(zip(feature_list1, feature_list2)):
            metrics = self.get_all_metrics(f1, f2)
            metrics['subject'] = i + 1
            results.append(metrics)
        
        df = pd.DataFrame(results)
        # Reorder columns
        cols = ['subject', 'pearson_r', 'p_value', 'icc', 'mae', 'rmse', 
                'icc_ci_lower', 'icc_ci_upper']
        df = df[[c for c in cols if c in df.columns]]
        
        return df
    
    def interpret_correlation(self, r_value):
        """
        Interpret correlation strength based on Pearson R value.
        
        Parameters:
        - r_value: correlation coefficient
        
        Returns:
        - interpretation: string describing correlation strength
        """
        abs_r = abs(r_value)
        
        if abs_r >= 0.9:
            strength = "Very strong"
        elif abs_r >= 0.7:
            strength = "Strong"
        elif abs_r >= 0.5:
            strength = "Moderate"
        elif abs_r >= 0.3:
            strength = "Weak"
        else:
            strength = "Very weak or no"
        
        direction = "positive" if r_value > 0 else "negative"
        
        return f"{strength} {direction} correlation (r = {r_value:.3f})"
    
    def interpret_icc(self, icc_value):
        """
        Interpret ICC value based on common guidelines.
        
        Parameters:
        - icc_value: ICC coefficient
        
        Returns:
        - interpretation: string describing reliability
        """
        if icc_value < 0.5:
            reliability = "Poor"
        elif icc_value < 0.75:
            reliability = "Moderate"
        elif icc_value < 0.9:
            reliability = "Good"
        else:
            reliability = "Excellent"
        
        return f"{reliability} reliability (ICC = {icc_value:.3f})"
    
    def interpret_mae(self, mae_value, feature_range=None):
        """
        Interpret MAE value.
        
        Parameters:
        - mae_value: MAE value
        - feature_range: tuple (min, max) of feature values for context
        
        Returns:
        - interpretation: string describing MAE
        """
        if feature_range:
            range_width = feature_range[1] - feature_range[0]
            percentage = (mae_value / range_width) * 100
            return f"MAE = {mae_value:.3f} ({percentage:.1f}% of feature range)"
        else:
            return f"MAE = {mae_value:.3f}"