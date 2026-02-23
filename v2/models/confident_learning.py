"""
Confident learning for label noise detection in HRD datasets.

Implements a Cleanlab-inspired approach to identify likely mislabeled samples
BEFORE model training. This is critical for HRD because:
  - Binary HRD labels from threshold-based scoring (HRD-sum >= 42) are noisy
  - Samples near the threshold boundary are often mislabeled
  - Epigenetic/methylation-driven HRD may have low GIS scores

Uses cross-validated predicted probabilities to estimate the joint distribution
of true vs. observed labels, then flags samples where the model consistently
disagrees with the provided label.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class ConfidentLearner:
    """Cross-validated confident learning for label noise detection.

    Parameters
    ----------
    n_folds : int
        Number of CV folds for out-of-fold probability estimation.
    calibration_method : str
        Probability calibration method: 'isotonic' or 'sigmoid'.
    confidence_threshold : float
        Threshold on label quality score below which samples are flagged.
        If None, uses adaptive threshold (mean - 1 std of quality scores).
    """

    def __init__(
        self,
        n_folds: int = 5,
        calibration_method: str = "isotonic",
        confidence_threshold: Optional[float] = None,
    ):
        self.n_folds = n_folds
        self.calibration_method = calibration_method
        self.confidence_threshold = confidence_threshold

        # Populated after find_noisy_labels()
        self.noise_mask_: Optional[np.ndarray] = None
        self.confidence_scores_: Optional[np.ndarray] = None
        self.label_quality_: Optional[np.ndarray] = None
        self.predicted_probs_: Optional[np.ndarray] = None
        self.thresholds_: Optional[dict] = None

    def find_noisy_labels(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        model=None,
    ) -> dict:
        """Identify likely mislabeled samples via cross-validated confident learning.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Observed binary labels (0/1).
        model : sklearn estimator or None
            Base classifier. If None, uses GradientBoostingClassifier.
            Must support predict_proba or be wrappable with calibration.

        Returns
        -------
        dict with keys:
            noise_mask : bool array (True = suspected noisy)
            confidence_scores : float array (per-sample out-of-fold P(predicted class))
            label_quality : float array (estimated P(label is correct))
            suggested_labels : int array (what the model thinks the label should be)
            predicted_probs : float array (out-of-fold P(class=1))
            threshold_used : float (the noise threshold applied)
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=int)
        n_samples = len(y_arr)

        if model is None:
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )

        # Step 1: Get out-of-fold predicted probabilities
        oof_probs = self._get_oof_probabilities(X_arr, y_arr, model)

        # Step 2: Compute confident joint (estimated joint distribution)
        thresholds = self._compute_thresholds(oof_probs, y_arr)

        # Step 3: Estimate label quality
        label_quality = self._compute_label_quality(oof_probs, y_arr)

        # Step 4: Determine noise threshold
        if self.confidence_threshold is not None:
            threshold = self.confidence_threshold
        else:
            # Adaptive: mean - 1 std
            threshold = max(
                label_quality.mean() - label_quality.std(),
                0.1,  # floor to avoid flagging everything
            )

        # Step 5: Flag noisy samples
        noise_mask = label_quality < threshold

        # Step 6: Suggested labels from model
        suggested_labels = (oof_probs >= 0.5).astype(int)

        # Confidence scores: P(predicted class)
        confidence_scores = np.where(
            suggested_labels == 1, oof_probs, 1.0 - oof_probs
        )

        # Store for later use
        self.noise_mask_ = noise_mask
        self.confidence_scores_ = confidence_scores
        self.label_quality_ = label_quality
        self.predicted_probs_ = oof_probs
        self.thresholds_ = {
            "per_class": thresholds,
            "noise_threshold": threshold,
        }

        n_noisy = noise_mask.sum()
        logger.info(
            "ConfidentLearner: flagged %d/%d samples (%.1f%%) as potentially noisy",
            n_noisy, n_samples, 100 * n_noisy / n_samples,
        )

        return {
            "noise_mask": noise_mask,
            "confidence_scores": confidence_scores,
            "label_quality": label_quality,
            "suggested_labels": suggested_labels,
            "predicted_probs": oof_probs,
            "threshold_used": threshold,
        }

    def clean_dataset(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        strategy: str = "prune",
    ) -> tuple:
        """Clean the dataset based on noise detection results.

        Must call ``find_noisy_labels`` first.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Observed labels.
        strategy : str
            'prune'   — remove samples flagged as noisy
            'relabel' — replace noisy labels with model suggestions
            'weight'  — return sample weights (low weight for noisy)

        Returns
        -------
        tuple of (X_clean, y_clean) for 'prune' and 'relabel',
        or (X, y, sample_weights) for 'weight'.
        """
        if self.noise_mask_ is None:
            raise RuntimeError("Must call find_noisy_labels() first")

        X_arr = np.asarray(X) if isinstance(X, pd.DataFrame) else X
        y_arr = np.asarray(y, dtype=int)
        is_df = isinstance(X, pd.DataFrame)

        if strategy == "prune":
            clean_mask = ~self.noise_mask_
            X_clean = X.loc[clean_mask] if is_df else X_arr[clean_mask]
            y_clean = y[clean_mask] if isinstance(y, pd.Series) else y_arr[clean_mask]
            logger.info(
                "Pruned %d noisy samples, %d remain",
                self.noise_mask_.sum(), clean_mask.sum(),
            )
            return X_clean, y_clean

        elif strategy == "relabel":
            suggested = (self.predicted_probs_ >= 0.5).astype(int)
            y_clean = y_arr.copy()
            y_clean[self.noise_mask_] = suggested[self.noise_mask_]
            n_changed = (y_arr != y_clean).sum()
            logger.info("Relabeled %d samples", n_changed)
            return X, y_clean if not isinstance(y, pd.Series) else pd.Series(y_clean, index=y.index)

        elif strategy == "weight":
            weights = self.label_quality_.copy()
            # Floor weights so noisy samples still contribute a little
            weights = np.clip(weights, 0.1, 1.0)
            return X, y, weights

        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'prune', 'relabel', or 'weight'.")

    def _get_oof_probabilities(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,
    ) -> np.ndarray:
        """Get out-of-fold predicted probabilities via cross-validation."""
        oof_probs = np.zeros(len(y), dtype=float)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            fold_model = clone(model)

            # Calibrate probabilities
            if self.calibration_method and len(np.unique(y_train)) > 1:
                cal_model = CalibratedClassifierCV(
                    fold_model,
                    method=self.calibration_method,
                    cv=3,
                )
                cal_model.fit(X_train, y_train)
                probs = cal_model.predict_proba(X_val)[:, 1]
            else:
                fold_model.fit(X_train, y_train)
                probs = fold_model.predict_proba(X_val)[:, 1]

            oof_probs[val_idx] = probs

        return oof_probs

    @staticmethod
    def _compute_thresholds(
        probs: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """Compute per-class average predicted probability thresholds.

        For each class k, the threshold t_k is the average predicted
        probability of class k among samples labeled as class k.
        """
        classes = np.unique(y)
        thresholds = {}
        for k in classes:
            mask = y == k
            if k == 1:
                thresholds[k] = probs[mask].mean() if mask.any() else 0.5
            else:
                thresholds[k] = (1 - probs[mask]).mean() if mask.any() else 0.5
        return thresholds

    @staticmethod
    def _compute_label_quality(
        probs: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Estimate per-sample label quality.

        Label quality = P(observed label | x), estimated from out-of-fold
        predicted probabilities. High quality ≈ model agrees with label.
        """
        # P(label=1|x) = probs, P(label=0|x) = 1 - probs
        quality = np.where(y == 1, probs, 1.0 - probs)
        return quality

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of noise detection results.

        Returns
        -------
        pd.DataFrame with columns: label_quality, predicted_prob, noise_flag, suggested_label
        """
        if self.noise_mask_ is None:
            raise RuntimeError("Must call find_noisy_labels() first")

        return pd.DataFrame({
            "label_quality": self.label_quality_,
            "predicted_prob": self.predicted_probs_,
            "confidence": self.confidence_scores_,
            "noise_flag": self.noise_mask_,
            "suggested_label": (self.predicted_probs_ >= 0.5).astype(int),
        })
