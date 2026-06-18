import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from IPython.display import display

class Visualization:
    """
    Visualization class for ECG/HRV feature analysis and model evaluation.
    """
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """
        Initialize Visualization class.
        
        Parameters:
        - style: matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_features_dist(self, df, feature_cols=None, figsize=(16, 7)):
        """
        Plot feature distributions by class (stress vs non-stress).
        
        Parameters:
        - df: DataFrame with features and 'label' column
        - feature_cols: list of feature column names (if None, uses all numeric columns except 'label')
        - figsize: figure size (width, height)
        
        Returns:
        - fig, axes: matplotlib figure and axes objects
        """
        if feature_cols is None:
            # Use all numeric columns except 'label'
            feature_cols = [col for col in df.columns if col != 'label' and np.issubdtype(df[col].dtype, np.number)]
        
        # Calculate number of rows and columns for subplots
        n_features = len(feature_cols)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, feat in enumerate(feature_cols):
            for cls, color, lbl in [(0, '#2d7a3e', 'Non-stress'), (1, '#c41e3a', 'Stress')]:
                axes[i].hist(df[df['label'] == cls][feat], bins=30, alpha=0.6,
                           color=color, label=lbl, density=True)
            axes[i].set_title(feat, fontsize=10)
            axes[i].legend(fontsize=8)
            axes[i].set_xlabel('Value', fontsize=8)
            axes[i].set_ylabel('Density', fontsize=8)
            axes[i].tick_params(labelsize=8)
        
        # Hide unused subplots
        for idx in range(len(feature_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Feature Distributions: Stress vs Non-stress', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig, axes
    
    def plot_features_corr_mat(self, df, feature_cols=None, figsize=(9, 7), 
                               cmap='coolwarm', annot=True, fmt='.2f'):
        """
        Plot feature correlation matrix heatmap.
        
        Parameters:
        - df: DataFrame with features
        - feature_cols: list of feature column names (if None, uses all numeric columns)
        - figsize: figure size (width, height)
        - cmap: colormap for heatmap
        - annot: whether to show correlation values
        - fmt: format string for annotations
        
        Returns:
        - fig, ax: matplotlib figure and axes objects
        """
        if feature_cols is None:
            # Use all numeric columns
            feature_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(df[feature_cols].corr(), annot=annot, fmt=fmt, 
                   cmap=cmap, center=0, ax=ax)
        
        ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        return fig, ax
    
    def plot_results(self, df, metrics=None, figsize=(10, 6)):
        """
        Plot results table with color-coded metrics.
        
        Parameters:
        - df: DataFrame with results (should contain accuracy, f1, precision, recall columns)
        - metrics: list of metric columns to display (if None, uses common metrics)
        - figsize: figure size (width, height) - affects text size only
        
        Returns:
        - styled_df: styled DataFrame for display
        """
        if metrics is None:
            metrics = ['accuracy', 'f1', 'precision', 'recall']
        
        # Check which metrics exist in DataFrame
        available_metrics = [m for m in metrics if m in df.columns]
        
        if not available_metrics:
            print("No matching metrics found in DataFrame")
            return None
        
        def color_metric(val):
            """Color metrics based on value (higher is better)"""
            if val >= 0.8:
                return 'background-color: #2d7a3e; color: white; font-weight: bold'
            elif val >= 0.6:
                return 'background-color: #f4a460; color: black; font-weight: bold'
            else:
                return 'background-color: #c41e3a; color: white; font-weight: bold'
        
        styled_df = (df[available_metrics].style
                    .map(color_metric, subset=available_metrics)
                    .format('{:.4f}', subset=available_metrics))
        
        return styled_df
    
    def plot_conf_mat(self, conf_mat, labels=None, title='Confusion Matrix', 
                      figsize=(6, 5), cmap='Blues'):
        """
        Plot confusion matrix heatmap.
        
        Parameters:
        - conf_mat: confusion matrix (2x2 array) or tuple of (y_true, y_pred)
        - labels: list of class labels (default: ['Non-stress', 'Stress'])
        - title: plot title
        - figsize: figure size (width, height)
        - cmap: colormap for heatmap
        
        Returns:
        - fig, ax: matplotlib figure and axes objects
        """
        if labels is None:
            labels = ['Non-stress', 'Stress']
        
        # If conf_mat is a tuple of (y_true, y_pred), compute confusion matrix
        if isinstance(conf_mat, tuple) and len(conf_mat) == 2:
            y_true, y_pred = conf_mat
            conf_mat = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap=cmap, ax=ax,
                   xticklabels=labels, yticklabels=labels)
        
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        
        plt.tight_layout()
        
        return fig, ax
    
    def plot_multiple_conf_mats(self, all_predictions, ncols=3, figsize_per_plot=(5, 4),
                                labels=None, title='Confusion Matrices'):
        """
        Plot multiple confusion matrices for different models.
        
        Parameters:
        - all_predictions: dict with model names as keys and (y_true, y_pred) tuples as values
        - ncols: number of columns in subplot grid
        - figsize_per_plot: size of each individual plot (width, height)
        - labels: list of class labels
        - title: overall title for the figure
        
        Returns:
        - fig, axes: matplotlib figure and axes objects
        """
        if labels is None:
            labels = ['Non-stress', 'Stress']
        
        n_models = len(all_predictions)
        nrows = (n_models + ncols - 1) // ncols
        
        fig_width = ncols * figsize_per_plot[0]
        fig_height = nrows * figsize_per_plot[1]
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
        
        # Flatten axes if needed
        if nrows == 1 and ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, (y_true, y_pred)) in enumerate(all_predictions.items()):
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=labels, yticklabels=labels)
            axes[idx].set_title(model_name, fontweight='bold', fontsize=10)
            axes[idx].set_xlabel('Predicted', fontsize=9)
            axes[idx].set_ylabel('True', fontsize=9)
            axes[idx].tick_params(labelsize=8)
        
        # Hide unused subplots
        for idx in range(len(all_predictions), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig, axes
    
    def plot_features_importance(self, importances, feature_cols=None, figsize=(8, 5),
                                 title='Feature Importance', color_threshold='median'):
        """
        Plot feature importance bar chart.
        
        Parameters:
        - importances: array-like of importance values OR trained model with feature_importances_ attribute
        - feature_cols: list of feature column names (required if importances is array-like)
        - figsize: figure size (width, height)
        - title: plot title
        - color_threshold: threshold for coloring bars ('median', 'mean', or numeric value)
        
        Returns:
        - fig, ax: matplotlib figure and axes objects
        - importances_series: pandas Series with feature importances
        """
        # If importances is a model with feature_importances_ attribute
        if hasattr(importances, 'feature_importances_'):
            importances_values = importances.feature_importances_
            if feature_cols is None:
                # Try to get feature names from model if available
                if hasattr(importances, 'feature_names_in_'):
                    feature_cols = importances.feature_names_in_
                else:
                    feature_cols = [f'feature_{i}' for i in range(len(importances_values))]
        else:
            importances_values = importances
        
        # Create Series and sort
        if feature_cols is not None:
            importances_series = pd.Series(importances_values, index=feature_cols)
        else:
            importances_series = pd.Series(importances_values)
        
        importances_series = importances_series.sort_values(ascending=True)
        
        # Determine color threshold
        if color_threshold == 'median':
            threshold = importances_series.median()
        elif color_threshold == 'mean':
            threshold = importances_series.mean()
        else:
            threshold = color_threshold
        
        # Color bars
        colors = ['#2d7a3e' if v >= threshold else '#f4a460' for v in importances_series]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        importances_series.plot(kind='barh', ax=ax, color=colors)
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontweight='bold')
        ax.axvline(threshold, color='gray', linestyle='--', alpha=0.7, 
                  label=f'Threshold: {threshold:.3f}')
        ax.legend(fontsize=9)
        
        plt.tight_layout()
        
        return fig, ax, importances_series
    
    def show_descriptive_stats(self, df, feature_cols=None, label_map=None):
        """
        Display descriptive statistics by class.
        
        Parameters:
        - df: DataFrame with features and 'label' column
        - feature_cols: list of feature column names
        - label_map: dict mapping label values to names (e.g., {0: 'Non-stress', 1: 'Stress'})
        
        Returns:
        - stats_df: DataFrame with grouped statistics
        """
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != 'label' and np.issubdtype(df[col].dtype, np.number)]
        
        if label_map is None:
            label_map = {0: 'Non-stress', 1: 'Stress'}
        
        stats_df = df.groupby('label')[feature_cols].mean()
        
        # Rename index if mapping exists
        if label_map:
            stats_df = stats_df.rename(index=label_map)
        
        print("=== Descriptive statistics by class ===")
        display(stats_df.round(3))
        
        return stats_df