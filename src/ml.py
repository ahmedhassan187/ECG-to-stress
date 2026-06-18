import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings

class ML:
    """
    Machine Learning class for model evaluation and validation.
    Provides methods for LOSO CV, model evaluation, and confusion matrices.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize ML class.
        
        Parameters:
        - random_state: random seed for reproducibility
        """
        self.random_state = random_state
    
    def loso_evaluate(self, df, model, feature_cols, subject_col='subject_id', 
                      label_col='label', average_metrics=True):
        """
        Run Leave-One-Subject-Out Cross Validation and return per-fold and aggregate metrics.
        
        Parameters:
        - df: DataFrame with features, labels, and subject IDs
        - model: scikit-learn compatible model (will be cloned for each fold)
        - feature_cols: list of feature column names
        - subject_col: name of column containing subject IDs
        - label_col: name of column containing labels
        - average_metrics: whether to calculate overall metrics (average across all predictions)
        
        Returns:
        - results_df: DataFrame with per-subject metrics
        - overall: dict with overall metrics (if average_metrics=True)
        - all_y_true: list of all true labels
        - all_y_pred: list of all predicted labels
        """
        from sklearn.base import clone
        
        subjects = sorted(df[subject_col].unique())
        results = []
        all_y_true = []
        all_y_pred = []
        
        for test_subj in subjects:
            train_df = df[df[subject_col] != test_subj]
            test_df = df[df[subject_col] == test_subj]
            
            X_train = train_df[feature_cols].values
            y_train = train_df[label_col].values
            X_test = test_df[feature_cols].values
            y_test = test_df[label_col].values
            
            # Clone model for each fold to avoid overfitting
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)
            
            # Store predictions for overall metrics
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            
            # Per-subject metrics
            results.append({
                'subject': test_subj,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'n_test': len(y_test),
            })
        
        results_df = pd.DataFrame(results).set_index('subject')
        
        if average_metrics:
            overall = {
                'accuracy': accuracy_score(all_y_true, all_y_pred),
                'f1': f1_score(all_y_true, all_y_pred, zero_division=0),
                'precision': precision_score(all_y_true, all_y_pred, zero_division=0),
                'recall': recall_score(all_y_true, all_y_pred, zero_division=0),
            }
            return results_df, overall, all_y_true, all_y_pred
        else:
            return results_df, all_y_true, all_y_pred
    
    def eval_all(self, df, models, feature_cols, subject_col='subject_id',
                 label_col='label', cv_method='loso'):
        """
        Evaluate multiple models using specified cross-validation method.
        
        Parameters:
        - df: DataFrame with features, labels, and subject IDs
        - models: dict of model names to model objects
        - feature_cols: list of feature column names
        - subject_col: name of column containing subject IDs
        - label_col: name of column containing labels
        - cv_method: cross-validation method ('loso' or 'kfold')
        
        Returns:
        - results_dict: dict containing results for each model
        """
        results_dict = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            if cv_method == 'loso':
                results_df, overall, y_true, y_pred = self.loso_evaluate(
                    df, model, feature_cols, subject_col, label_col, average_metrics=True
                )
                
                results_dict[model_name] = {
                    'per_subject': results_df,
                    'overall': overall,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'model': model
                }
                
                print(f"  Overall Accuracy: {overall['accuracy']:.4f}")
                print(f"  Overall F1: {overall['f1']:.4f}")
                
            elif cv_method == 'kfold':
                X = df[feature_cols].values
                y = df[label_col].values
                
                # Perform k-fold cross-validation
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                
                accuracies = []
                f1_scores = []
                precisions = []
                recalls = []
                all_y_true = []
                all_y_pred = []
                
                for train_idx, test_idx in skf.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    all_y_true.extend(y_test)
                    all_y_pred.extend(y_pred)
                    
                    accuracies.append(accuracy_score(y_test, y_pred))
                    f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
                    precisions.append(precision_score(y_test, y_pred, zero_division=0))
                    recalls.append(recall_score(y_test, y_pred, zero_division=0))
                
                results_dict[model_name] = {
                    'cv_scores': {
                        'accuracy': accuracies,
                        'f1': f1_scores,
                        'precision': precisions,
                        'recall': recalls
                    },
                    'overall': {
                        'accuracy_mean': np.mean(accuracies),
                        'accuracy_std': np.std(accuracies),
                        'f1_mean': np.mean(f1_scores),
                        'f1_std': np.std(f1_scores),
                        'precision_mean': np.mean(precisions),
                        'precision_std': np.std(precisions),
                        'recall_mean': np.mean(recalls),
                        'recall_std': np.std(recalls)
                    },
                    'y_true': all_y_true,
                    'y_pred': all_y_pred,
                    'model': model
                }
                
                print(f"  Mean Accuracy: {results_dict[model_name]['overall']['accuracy_mean']:.4f} (+/- {results_dict[model_name]['overall']['accuracy_std']:.4f})")
            
            else:
                raise ValueError("cv_method must be 'loso' or 'kfold'")
        
        return results_dict
    
    def make_conf_mat(self, y_true, y_pred, normalize=False):
        """
        Generate confusion matrix from true and predicted labels.
        
        Parameters:
        - y_true: array-like, true labels
        - y_pred: array-like, predicted labels
        - normalize: whether to normalize the confusion matrix
        
        Returns:
        - conf_mat: confusion matrix as numpy array
        """
        conf_mat = confusion_matrix(y_true, y_pred)
        
        if normalize:
            conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        
        return conf_mat
    
    def make_conf_mat_from_df(self, df, model, feature_cols, label_col='label', normalize=False):
        """
        Train model on full dataset and generate confusion matrix.
        
        Parameters:
        - df: DataFrame with features and labels
        - model: scikit-learn compatible model
        - feature_cols: list of feature column names
        - label_col: name of column containing labels
        - normalize: whether to normalize the confusion matrix
        
        Returns:
        - conf_mat: confusion matrix
        - y_true: true labels
        - y_pred: predicted labels
        """
        X = df[feature_cols].values
        y_true = df[label_col].values
        
        model.fit(X, y_true)
        y_pred = model.predict(X)
        
        conf_mat = self.make_conf_mat(y_true, y_pred, normalize)
        
        return conf_mat, y_true, y_pred
    
    def get_classification_report(self, y_true, y_pred, target_names=None):
        """
        Generate detailed classification report.
        
        Parameters:
        - y_true: array-like, true labels
        - y_pred: array-like, predicted labels
        - target_names: list of class names (e.g., ['Non-stress', 'Stress'])
        
        Returns:
        - report: classification report as string
        - report_dict: classification report as dictionary
        """
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=False)
        report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        
        return report, report_dict
    
    def get_roc_auc(self, y_true, y_pred_proba):
        """
        Calculate ROC-AUC score.
        
        Parameters:
        - y_true: array-like, true labels
        - y_pred_proba: array-like, predicted probabilities (for positive class)
        
        Returns:
        - auc: ROC-AUC score
        - fpr: false positive rates
        - tpr: true positive rates
        """
        auc = roc_auc_score(y_true, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        
        return auc, fpr, tpr
    
    def compare_models(self, results_dict, metric='accuracy'):
        """
        Compare multiple models based on a specific metric.
        
        Parameters:
        - results_dict: output from eval_all method
        - metric: metric to compare ('accuracy', 'f1', 'precision', 'recall')
        
        Returns:
        - comparison_df: DataFrame with model comparisons
        """
        comparison = []
        
        for model_name, results in results_dict.items():
            if 'overall' in results:
                if metric in results['overall']:
                    comparison.append({
                        'model': model_name,
                        metric: results['overall'][metric]
                    })
                elif f'{metric}_mean' in results['overall']:
                    comparison.append({
                        'model': model_name,
                        f'{metric}_mean': results['overall'][f'{metric}_mean'],
                        f'{metric}_std': results['overall'][f'{metric}_std']
                    })
        
        comparison_df = pd.DataFrame(comparison)
        
        if 'model' in comparison_df.columns:
            comparison_df = comparison_df.set_index('model')
        
        return comparison_df
    
    def get_feature_importance(self, model, feature_cols, top_n=None):
        """
        Extract feature importance from trained model.
        
        Parameters:
        - model: trained model with feature_importances_ or coef_ attribute
        - feature_cols: list of feature column names
        - top_n: number of top features to return (if None, return all)
        
        Returns:
        - importance_df: DataFrame with feature names and importance scores
        """
        # Check if model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        # Check if model has coef_ (for linear models)
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) > 1:
                importances = np.abs(model.coef_[0])
            else:
                importances = np.abs(model.coef_)
        else:
            raise ValueError("Model does not have feature_importances_ or coef_ attribute")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def cross_validate_model(self, df, model, feature_cols, label_col='label', 
                             cv=5, scoring='accuracy'):
        """
        Perform cross-validation on a model.
        
        Parameters:
        - df: DataFrame with features and labels
        - model: scikit-learn compatible model
        - feature_cols: list of feature column names
        - label_col: name of column containing labels
        - cv: number of cross-validation folds
        - scoring: scoring metric
        
        Returns:
        - scores: list of scores for each fold
        - mean_score: mean of scores
        - std_score: standard deviation of scores
        """
        X = df[feature_cols].values
        y = df[label_col].values
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return scores, scores.mean(), scores.std()