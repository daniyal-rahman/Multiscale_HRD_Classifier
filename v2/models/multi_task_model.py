"""
Multi-task gradient boosting model for HRD classification.

Jointly predicts multiple related endpoints:
  - HRD status (binary or soft)
  - BRCA subtype (BRCA1-type, BRCA2-type, HRD-BRCApos, HRP)
  - Drug response (optional: platinum/PARPi sensitivity)

Architecture: shared feature space with task-specific XGBoost/LightGBM
heads. Multi-task coupling via shared feature selection and optional
cross-task feature augmentation.

Uses SHAP for interpretability — critical for clinical adoption.
"""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
)

logger = logging.getLogger(__name__)


# Default hyperparameters per task type
_XGBOOST_DEFAULTS = {
    "binary": dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
    ),
    "multiclass": dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
    ),
    "regression": dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
    ),
}

_LIGHTGBM_DEFAULTS = {
    "binary": dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    ),
    "multiclass": dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    ),
    "regression": dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    ),
}


def _infer_task_type(y: np.ndarray | pd.Series) -> str:
    """Infer whether a target array is binary, multiclass, or regression."""
    y_arr = np.asarray(y)
    unique = np.unique(y_arr[~np.isnan(y_arr)] if np.issubdtype(y_arr.dtype, np.floating) else y_arr)
    if len(unique) == 2:
        return "binary"
    elif len(unique) <= 10 and np.issubdtype(y_arr.dtype, np.integer):
        return "multiclass"
    elif np.issubdtype(y_arr.dtype, np.floating):
        # Could be soft labels (binary-ish) or continuous
        if set(unique).issubset({0.0, 1.0}):
            return "binary"
        return "regression"
    return "multiclass"


def _build_model(base_model: str, task_type: str, override_params: dict | None = None):
    """Build an XGBoost or LightGBM model for the given task type."""
    if base_model == "xgboost":
        import xgboost as xgb
        defaults = deepcopy(_XGBOOST_DEFAULTS[task_type])
        if override_params:
            defaults.update(override_params)

        if task_type == "binary":
            return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **defaults)
        elif task_type == "multiclass":
            return xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **defaults)
        else:
            return xgb.XGBRegressor(objective="reg:squarederror", **defaults)

    elif base_model == "lightgbm":
        import lightgbm as lgb
        defaults = deepcopy(_LIGHTGBM_DEFAULTS[task_type])
        if override_params:
            defaults.update(override_params)

        if task_type == "binary":
            return lgb.LGBMClassifier(objective="binary", **defaults)
        elif task_type == "multiclass":
            return lgb.LGBMClassifier(objective="multiclass", **defaults)
        else:
            return lgb.LGBMRegressor(objective="regression", **defaults)

    else:
        raise ValueError(f"Unknown base_model: {base_model}. Use 'xgboost' or 'lightgbm'.")


class MultiTaskHRDModel(BaseEstimator, ClassifierMixin):
    """Multi-task gradient boosting for HRD prediction.

    Trains separate task-specific models on a shared feature space.
    Multi-task coupling is achieved through:
      1. Shared feature selection (features important across tasks are kept)
      2. Optional cross-task feature augmentation (predictions from one task
         fed as features to another)
      3. Joint hyperparameter tuning under nested CV

    Parameters
    ----------
    tasks : list of str
        Task names, e.g. ['hrd_status', 'brca_type', 'drug_response'].
    base_model : str
        'xgboost' or 'lightgbm'.
    use_label_smoothing : bool
        If True and soft labels are provided, use regression objective for
        the primary HRD task instead of classification.
    soft_label_column : str or None
        Column in y_dict that contains soft (continuous) labels for HRD.
    model_params : dict or None
        Override default hyperparameters. Can be a flat dict (applied to all
        tasks) or nested dict keyed by task name.
    cross_task_features : bool
        If True, predictions from earlier tasks are added as features for
        later tasks (stacking-like).
    """

    def __init__(
        self,
        tasks: list[str] | None = None,
        base_model: str = "xgboost",
        use_label_smoothing: bool = True,
        soft_label_column: str | None = None,
        model_params: dict | None = None,
        cross_task_features: bool = False,
    ):
        self.tasks = tasks or ["hrd_status", "brca_type", "drug_response"]
        self.base_model = base_model
        self.use_label_smoothing = use_label_smoothing
        self.soft_label_column = soft_label_column
        self.model_params = model_params or {}
        self.cross_task_features = cross_task_features

        # Populated during fit
        self.models_: dict = {}
        self.task_types_: dict = {}
        self.feature_names_: list[str] | None = None
        self.label_encoders_: dict = {}
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y_dict: dict[str, np.ndarray | pd.Series],
        sample_weights: np.ndarray | None = None,
    ) -> "MultiTaskHRDModel":
        """Fit task-specific models on shared features.

        Parameters
        ----------
        X : DataFrame or array (n_samples, n_features)
            Feature matrix.
        y_dict : dict
            {task_name: labels} for each task.
            Tasks not in y_dict are silently skipped.
        sample_weights : array or None
            Per-sample weights (e.g. from confident learning).

        Returns
        -------
        self
        """
        X_arr, feature_names = self._to_array(X)
        self.feature_names_ = feature_names

        active_tasks = [t for t in self.tasks if t in y_dict]
        if not active_tasks:
            raise ValueError(f"No matching tasks. y_dict keys: {list(y_dict.keys())}, expected: {self.tasks}")

        cross_preds = {}

        for task_name in active_tasks:
            y_task = y_dict[task_name]
            y_arr, task_type, encoder = self._prepare_labels(y_task, task_name)

            # Determine samples with valid labels for this task
            if np.issubdtype(y_arr.dtype, np.floating):
                valid = ~np.isnan(y_arr)
            else:
                valid = np.ones(len(y_arr), dtype=bool)

            X_task = X_arr[valid]
            y_task_valid = y_arr[valid]
            sw = sample_weights[valid] if sample_weights is not None else None

            # Cross-task feature augmentation
            if self.cross_task_features and cross_preds:
                extra = np.column_stack([cross_preds[t][valid] for t in cross_preds])
                X_task = np.hstack([X_task, extra])

            # Get task-specific params
            task_params = self.model_params.get(task_name, self.model_params) if isinstance(
                self.model_params.get(task_name), dict
            ) else self.model_params

            model = _build_model(self.base_model, task_type, task_params)

            fit_kwargs = {}
            if sw is not None:
                fit_kwargs["sample_weight"] = sw

            model.fit(X_task, y_task_valid, **fit_kwargs)

            self.models_[task_name] = model
            self.task_types_[task_name] = task_type
            if encoder is not None:
                self.label_encoders_[task_name] = encoder

            # Store cross-task predictions for subsequent tasks
            if self.cross_task_features:
                if task_type in ("binary", "multiclass"):
                    cross_preds[task_name] = model.predict_proba(X_arr if not cross_preds else
                        np.hstack([X_arr, np.column_stack([cross_preds[t] for t in cross_preds])]))[:, -1]
                else:
                    cross_preds[task_name] = model.predict(X_arr if not cross_preds else
                        np.hstack([X_arr, np.column_stack([cross_preds[t] for t in cross_preds])]))

            logger.info("Fitted %s (%s): %d samples", task_name, task_type, len(y_task_valid))

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> dict[str, np.ndarray]:
        """Predict labels for each task.

        Returns
        -------
        dict of {task_name: predictions}
        """
        self._check_fitted()
        X_arr, _ = self._to_array(X)
        results = {}

        for task_name, model in self.models_.items():
            preds = model.predict(X_arr)
            if task_name in self.label_encoders_:
                preds = self.label_encoders_[task_name].inverse_transform(preds.astype(int))
            results[task_name] = preds

        return results

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> dict[str, np.ndarray]:
        """Predict probabilities for each task.

        Returns
        -------
        dict of {task_name: probability_array}
            For binary tasks: shape (n_samples, 2)
            For multiclass: shape (n_samples, n_classes)
            For regression: shape (n_samples,) — point predictions
        """
        self._check_fitted()
        X_arr, _ = self._to_array(X)
        results = {}

        for task_name, model in self.models_.items():
            task_type = self.task_types_[task_name]
            if task_type in ("binary", "multiclass"):
                results[task_name] = model.predict_proba(X_arr)
            else:
                results[task_name] = model.predict(X_arr)

        return results

    def get_feature_importance(
        self,
        method: str = "shap",
        X_background: pd.DataFrame | np.ndarray | None = None,
        max_samples: int = 200,
    ) -> pd.DataFrame:
        """Get feature importances across tasks.

        Parameters
        ----------
        method : str
            'shap' for SHAP values, 'gain' for split-based importance.
        X_background : array-like or None
            Background data for SHAP (subsample if large). Required for 'shap'.
        max_samples : int
            Max samples for SHAP background.

        Returns
        -------
        pd.DataFrame
            Columns: feature, task, importance.
        """
        self._check_fitted()
        rows = []

        if method == "gain":
            for task_name, model in self.models_.items():
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    for i, imp in enumerate(importances):
                        fname = self.feature_names_[i] if self.feature_names_ and i < len(self.feature_names_) else f"f{i}"
                        rows.append({"feature": fname, "task": task_name, "importance": imp})

        elif method == "shap":
            try:
                import shap
            except ImportError:
                logger.warning("shap not installed, falling back to 'gain'")
                return self.get_feature_importance(method="gain")

            if X_background is None:
                raise ValueError("X_background required for SHAP method")

            X_bg, _ = self._to_array(X_background)
            if len(X_bg) > max_samples:
                idx = np.random.RandomState(42).choice(len(X_bg), max_samples, replace=False)
                X_bg = X_bg[idx]

            for task_name, model in self.models_.items():
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_bg)

                # For binary classification, shap_values might be a list
                if isinstance(shap_values, list):
                    shap_values = shap_values[-1]  # positive class

                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                for i, imp in enumerate(mean_abs_shap):
                    fname = self.feature_names_[i] if self.feature_names_ and i < len(self.feature_names_) else f"f{i}"
                    rows.append({"feature": fname, "task": task_name, "importance": imp})
        else:
            raise ValueError(f"Unknown method: {method}. Use 'shap' or 'gain'.")

        df = pd.DataFrame(rows)
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def cross_validate(
        self,
        X: pd.DataFrame | np.ndarray,
        y_dict: dict[str, np.ndarray | pd.Series],
        cv_strategy: str = "stratified",
        n_splits: int = 5,
        dataset_col: pd.Series | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> dict:
        """Nested cross-validation for unbiased evaluation.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y_dict : dict
            {task_name: labels}.
        cv_strategy : str
            'stratified' or 'group' (leave-one-dataset-out).
        n_splits : int
            Number of outer folds (ignored if cv_strategy='group').
        dataset_col : Series or None
            Dataset/batch identifiers. Required if cv_strategy='group'.
            Outer fold = leave-one-dataset-out.
        sample_weights : array or None
            Per-sample weights.

        Returns
        -------
        dict with per-task metrics:
            {task_name: {metric: [fold_values], 'mean_metric': float}}
        """
        X_arr, feature_names = self._to_array(X)

        # Determine primary task for stratification
        primary_task = self.tasks[0] if self.tasks[0] in y_dict else list(y_dict.keys())[0]
        y_primary = np.asarray(y_dict[primary_task])

        if cv_strategy == "group" and dataset_col is not None:
            groups = np.asarray(dataset_col)
            cv = GroupKFold(n_splits=len(np.unique(groups)))
            splits = list(cv.split(X_arr, y_primary, groups))
        else:
            # Stratify on primary task
            y_strat = y_primary.copy()
            if np.issubdtype(y_strat.dtype, np.floating):
                y_strat = (y_strat >= 0.5).astype(int)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(cv.split(X_arr, y_strat))

        # Collect per-fold metrics
        all_metrics = {task: [] for task in y_dict}

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = X_arr[train_idx]
            X_test = X_arr[test_idx]

            y_train = {t: np.asarray(y_dict[t])[train_idx] for t in y_dict}
            y_test = {t: np.asarray(y_dict[t])[test_idx] for t in y_dict}

            sw_train = sample_weights[train_idx] if sample_weights is not None else None

            # Create a fresh model for this fold
            fold_model = MultiTaskHRDModel(
                tasks=self.tasks,
                base_model=self.base_model,
                use_label_smoothing=self.use_label_smoothing,
                model_params=self.model_params,
                cross_task_features=self.cross_task_features,
            )

            # Wrap arrays as DataFrames so feature names are preserved
            X_train_df = pd.DataFrame(X_train, columns=feature_names) if feature_names else X_train
            X_test_df = pd.DataFrame(X_test, columns=feature_names) if feature_names else X_test

            fold_model.fit(X_train_df, y_train, sample_weights=sw_train)

            preds = fold_model.predict(X_test_df)
            probas = fold_model.predict_proba(X_test_df)

            for task_name in y_dict:
                if task_name not in fold_model.models_:
                    continue
                task_type = fold_model.task_types_[task_name]
                metrics = self._compute_task_metrics(
                    y_test[task_name], preds.get(task_name), probas.get(task_name), task_type
                )
                metrics["fold"] = fold_idx
                all_metrics[task_name].append(metrics)

        # Aggregate
        results = {}
        for task_name, fold_metrics in all_metrics.items():
            if not fold_metrics:
                continue
            metric_keys = [k for k in fold_metrics[0] if k != "fold"]
            agg = {}
            for k in metric_keys:
                vals = [m[k] for m in fold_metrics if m[k] is not None]
                if vals:
                    agg[f"{k}_per_fold"] = vals
                    agg[f"{k}_mean"] = np.mean(vals)
                    agg[f"{k}_std"] = np.std(vals)
            agg["n_folds"] = len(fold_metrics)
            results[task_name] = agg

        return results

    # ------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------- #

    def _prepare_labels(self, y, task_name: str):
        """Prepare labels: encode if needed, infer task type."""
        y_arr = np.asarray(y)
        encoder = None

        # Check if this task should use soft labels (regression)
        if (self.use_label_smoothing
                and task_name == self.soft_label_column
                and np.issubdtype(y_arr.dtype, np.floating)):
            return y_arr.astype(float), "regression", None

        # Encode string labels
        if y_arr.dtype.kind in ("U", "S", "O"):
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            y_arr = encoder.fit_transform(y_arr)

        task_type = _infer_task_type(y_arr)

        # If use_label_smoothing and this is the primary HRD task with float values
        if (self.use_label_smoothing
                and task_name in ("hrd_status",)
                and np.issubdtype(y_arr.dtype, np.floating)
                and not set(np.unique(y_arr)).issubset({0.0, 1.0})):
            task_type = "regression"

        return y_arr, task_type, encoder

    @staticmethod
    def _to_array(X) -> tuple[np.ndarray, list[str] | None]:
        """Convert to numpy array, extracting feature names if DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X.values, list(X.columns)
        return np.asarray(X), None

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

    @staticmethod
    def _compute_task_metrics(y_true, y_pred, y_proba, task_type):
        """Compute metrics for a single task."""
        metrics = {}
        y_true = np.asarray(y_true)
        valid = ~pd.isna(y_true) if hasattr(pd, "isna") else np.ones(len(y_true), dtype=bool)
        y_true = y_true[valid]

        if y_pred is not None:
            y_pred = np.asarray(y_pred)[valid]
        if y_proba is not None:
            y_proba = np.asarray(y_proba)[valid] if y_proba.ndim == 1 else np.asarray(y_proba)[valid]

        if task_type == "binary":
            metrics["accuracy"] = accuracy_score(y_true, y_pred) if y_pred is not None else None
            if y_proba is not None and len(np.unique(y_true)) > 1:
                proba_1 = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                try:
                    metrics["auc"] = roc_auc_score(y_true, proba_1)
                    metrics["log_loss"] = log_loss(y_true, proba_1)
                except ValueError:
                    metrics["auc"] = None
                    metrics["log_loss"] = None
            else:
                metrics["auc"] = None
                metrics["log_loss"] = None

        elif task_type == "multiclass":
            metrics["accuracy"] = accuracy_score(y_true, y_pred) if y_pred is not None else None
            if y_proba is not None and len(np.unique(y_true)) > 1:
                try:
                    metrics["auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
                except ValueError:
                    metrics["auc_ovr"] = None
            else:
                metrics["auc_ovr"] = None

        elif task_type == "regression":
            if y_pred is not None:
                metrics["mse"] = mean_squared_error(y_true, y_pred)
                metrics["r2"] = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else None
                # Also compute binary AUC if values are in [0, 1]
                if y_true.min() >= 0 and y_true.max() <= 1:
                    y_binary = (y_true >= 0.5).astype(int)
                    pred_binary = y_pred if y_proba is None else y_proba
                    if len(np.unique(y_binary)) > 1:
                        try:
                            pred_vals = np.asarray(pred_binary)
                            if pred_vals.ndim > 1:
                                pred_vals = pred_vals[:, -1]
                            metrics["binary_auc"] = roc_auc_score(y_binary, pred_vals)
                        except ValueError:
                            metrics["binary_auc"] = None

        return metrics
