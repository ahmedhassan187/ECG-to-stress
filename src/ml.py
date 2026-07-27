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
    Supports binary and multi-class classification.
    """

    def __init__(self, random_state=42, average='weighted'):
        """
        Initialize ML class.

        Parameters:
        - random_state: random seed for reproducibility
        - average: averaging strategy for multi-class metrics
                   ('weighted', 'macro', 'micro', 'binary').
                   Default 'weighted' works for both binary and multi-class.
        """
        self.random_state = random_state
        self.average = average

    def loso_evaluate(self, df, model, feature_cols, subject_col='subject_id',
                      label_col='label', average_metrics=True, average=None):
        """
        Run Leave-One-Subject-Out Cross Validation and return per-fold and aggregate metrics.

        Parameters:
        - df: DataFrame with features, labels, and subject IDs
        - model: scikit-learn compatible model (will be cloned for each fold)
        - feature_cols: list of feature column names
        - subject_col: name of column containing subject IDs
        - label_col: name of column containing labels
        - average_metrics: whether to calculate overall metrics (average across all predictions)
        - average: averaging strategy for multi-class metrics (overrides self.average)

        Returns:
        - results_df: DataFrame with per-subject metrics
        - overall: dict with overall metrics (if average_metrics=True)
        - all_y_true: list of all true labels
        - all_y_pred: list of all predicted labels
        """
        from sklearn.base import clone

        avg = average if average is not None else self.average

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
                'f1': f1_score(y_test, y_pred, average=avg, zero_division=0),
                'precision': precision_score(y_test, y_pred, average=avg, zero_division=0),
                'recall': recall_score(y_test, y_pred, average=avg, zero_division=0),
                'n_test': len(y_test),
            })

        results_df = pd.DataFrame(results).set_index('subject')

        if average_metrics:
            overall = {
                'accuracy': accuracy_score(all_y_true, all_y_pred),
                'f1': f1_score(all_y_true, all_y_pred, average=avg, zero_division=0),
                'precision': precision_score(all_y_true, all_y_pred, average=avg, zero_division=0),
                'recall': recall_score(all_y_true, all_y_pred, average=avg, zero_division=0),
            }
            return results_df, overall, all_y_true, all_y_pred
        else:
            return results_df, all_y_true, all_y_pred

    def eval_all(self, df, models, feature_cols, subject_col='subject_id',
                 label_col='label', cv_method='loso', average=None):
        """
        Evaluate multiple models using specified cross-validation method.

        Parameters:
        - df: DataFrame with features, labels, and subject IDs
        - models: dict of model names to model objects
        - feature_cols: list of feature column names
        - subject_col: name of column containing subject IDs
        - label_col: name of column containing labels
        - cv_method: cross-validation method ('loso' or 'kfold')
        - average: averaging strategy for multi-class metrics (overrides self.average)

        Returns:
        - results_dict: dict containing results for each model
        """
        avg = average if average is not None else self.average
        results_dict = {}

        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")

            if cv_method == 'loso':
                results_df, overall, y_true, y_pred = self.loso_evaluate(
                    df, model, feature_cols, subject_col, label_col,
                    average_metrics=True, average=avg
                )

                results_dict[model_name] = {
                    'per_subject': results_df,
                    'overall': overall,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'model': model
                }

                print(f"  Overall Accuracy: {overall['accuracy']:.4f}")
                f1_info = f" (avg={avg})" if avg != 'binary' else ""
                print(f"  Overall F1: {overall['f1']:.4f}{f1_info}")

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
                all_y_pred_proba = []

                for train_idx, test_idx in skf.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    model_clone = model.__class__(**model.get_params())
                    model_clone.fit(X_train, y_train)
                    y_pred = model_clone.predict(X_test)

                    all_y_true.extend(y_test)
                    all_y_pred.extend(y_pred)

                    # Collect predicted probabilities for AUC
                    if hasattr(model_clone, 'predict_proba'):
                        y_proba = model_clone.predict_proba(X_test)
                        all_y_pred_proba.append(y_proba)
                    else:
                        all_y_pred_proba.append(None)

                    accuracies.append(accuracy_score(y_test, y_pred))
                    f1_scores.append(f1_score(y_test, y_pred, average=avg, zero_division=0))
                    precisions.append(precision_score(y_test, y_pred, average=avg, zero_division=0))
                    recalls.append(recall_score(y_test, y_pred, average=avg, zero_division=0))

                # Compute AUC from aggregated predictions
                n_unique = len(np.unique(all_y_true))
                auc_mean = np.nan
                auc_std = np.nan
                auc_scores = []
                if all_y_pred_proba[0] is not None:
                    # Stack all fold probabilities
                    try:
                        y_pred_proba_full = np.vstack(all_y_pred_proba)
                        if n_unique == 2:
                            # Binary: use probability of positive class
                            y_score = y_pred_proba_full[:, 1]
                            auc_val = roc_auc_score(all_y_true, y_score)
                            auc_scores = [auc_val]
                            auc_mean = auc_val
                            auc_std = 0.0
                        else:
                            # Multi-class: use full probability matrix
                            auc_val = roc_auc_score(all_y_true, y_pred_proba_full,
                                                    multi_class='ovr')
                            auc_scores = [auc_val]
                            auc_mean = auc_val
                            auc_std = 0.0
                    except Exception as e:
                        print(f"   ⚠️ AUC computation failed: {e}")

                results_dict[model_name] = {
                    'cv_scores': {
                        'accuracy': accuracies,
                        'f1': f1_scores,
                        'precision': precisions,
                        'recall': recalls,
                        'auc': auc_scores,
                    },
                    'overall': {
                        'accuracy_mean': np.mean(accuracies),
                        'accuracy_std': np.std(accuracies),
                        'f1_mean': np.mean(f1_scores),
                        'f1_std': np.std(f1_scores),
                        'precision_mean': np.mean(precisions),
                        'precision_std': np.std(precisions),
                        'recall_mean': np.mean(recalls),
                        'recall_std': np.std(recalls),
                        'auc_mean': auc_mean,
                        'auc_std': auc_std,
                    },
                    'y_true': all_y_true,
                    'y_pred': all_y_pred,
                    'model': model
                }

                f1_info = f" (avg={avg})" if avg != 'binary' else ""
                print(f"  Mean Accuracy: {results_dict[model_name]['overall']['accuracy_mean']:.4f} (+/- {results_dict[model_name]['overall']['accuracy_std']:.4f})")
                print(f"  Mean F1: {results_dict[model_name]['overall']['f1_mean']:.4f}{f1_info}")

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
        report = classification_report(y_true, y_pred, target_names=target_names,
                                       output_dict=False, zero_division=0)
        report_dict = classification_report(y_true, y_pred, target_names=target_names,
                                            output_dict=True, zero_division=0)

        return report, report_dict

    def get_roc_auc(self, y_true, y_pred_proba, multi_class=None, n_classes=2):
        """
        Calculate ROC-AUC score.

        For binary classification: uses standard ROC-AUC.
        For multi-class: uses 'ovr' (One-vs-Rest) by default.

        Parameters:
        - y_true: array-like, true labels
        - y_pred_proba: array-like, predicted probabilities
                        For binary: probability of positive class (1D) or full matrix (n_samples, 2)
                        For multi-class: full probability matrix (n_samples, n_classes)
        - multi_class: 'ovr', 'ovo', or None (auto-detected from n_classes)
        - n_classes: number of classes (2=default for binary)

        Returns:
        - auc: ROC-AUC score
        - fpr: false positive rates (binary only, None for multi-class)
        - tpr: true positive rates (binary only, None for multi-class)
        """
        y_pred_proba = np.asarray(y_pred_proba)
        n_unique = len(np.unique(y_true))

        if n_unique == 2:
            # Binary classification
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] >= 2:
                y_pred_proba = y_pred_proba[:, 1]
            auc = roc_auc_score(y_true, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            return auc, fpr, tpr
        else:
            # Multi-class classification
            if y_pred_proba.ndim == 1:
                print("   ⚠️ ROC-AUC requires probability matrix for multi-class. Skipping.")
                return np.nan, None, None
            mc = multi_class if multi_class else 'ovr'
            try:
                auc = roc_auc_score(y_true, y_pred_proba, multi_class=mc)
                return auc, None, None
            except Exception as e:
                print(f"   ⚠️ ROC-AUC multi-class failed: {e}")
                return np.nan, None, None

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

    # ── Model factory ───────────────────────────────────────────────────────────

    @staticmethod
    def get_default_models():
        """Return list of available model names (excluding optional deps)."""
        return [
            'knn', 'svm', 'decision_tree', 'random_forest',
            'gradient_boosting', 'logistic_regression', 'xgboost'
        ]

    @staticmethod
    def get_model(model_name, xgboost_available=False,
                  XGBClassifier=None, random_state=42):
        """
        Create a model instance by name.

        Parameters:
            model_name: one of 'knn', 'svm', 'decision_tree', 'random_forest',
                        'gradient_boosting', 'logistic_regression', 'xgboost'
            xgboost_available: whether xgboost is installed
            XGBClassifier: the XGBClassifier class (or None)
            random_state: random seed

        Returns:
            model instance, or None if xgboost requested but not available
        """
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression

        models = {
            'knn': KNeighborsClassifier(n_neighbors=5),
            'svm': SVC(kernel='rbf', probability=True, random_state=random_state),
            'decision_tree': DecisionTreeClassifier(random_state=random_state),
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=random_state
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=random_state
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000, random_state=random_state
            ),
        }
        if model_name == 'xgboost':
            if xgboost_available and XGBClassifier is not None:
                return XGBClassifier(
                    n_estimators=100, random_state=random_state, verbosity=0
                )
            return None
        return models.get(model_name)

    @staticmethod
    def get_available_models(requested_models, available_models,
                             xgboost_available=False):
        """
        Filter requested models to only those that are available.

        Parameters:
            requested_models: list of model name strings
            available_models: list of all valid model name strings
            xgboost_available: whether xgboost is installed

        Returns:
            filtered list of model names
        """
        result = []
        for m in requested_models:
            if m not in available_models:
                continue
            if m == 'xgboost' and not xgboost_available:
                continue
            result.append(m)
        return result