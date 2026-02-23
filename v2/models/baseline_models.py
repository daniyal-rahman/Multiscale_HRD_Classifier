"""
Baseline models for fair comparison against the multi-task approach.

Implements:
  1. ElasticNetBaseline — reproduces the softHRD v1 pipeline
  2. CentroidBaseline  — reproduces the Multiscale/Jacobson centroid approach
  3. SingleGeneBaseline — POLQ, BRCA1, BRCA2 as single-gene predictors

All follow the sklearn fit/predict API for easy plugging into CV loops.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class ElasticNetBaseline(BaseEstimator, ClassifierMixin):
    """Reproduce the softHRD v1 approach: DESeq2 genes -> ElasticNet.

    Two modes:
      - classification: LogisticRegression with elasticnet penalty (binary HRD)
      - regression: ElasticNet for continuous HRD score prediction

    Parameters
    ----------
    mode : str
        'classification' or 'regression'.
    alpha : float
        Regularization strength. For classification, this is C=1/alpha.
    l1_ratio : float
        ElasticNet mixing (0=Ridge, 1=Lasso).
    max_iter : int
        Maximum iterations for solver.
    """

    def __init__(
        self,
        mode: str = "classification",
        alpha: float = 0.1,
        l1_ratio: float = 0.5,
        max_iter: int = 2000,
    ):
        self.mode = mode
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter

        self.scaler_ = None
        self.model_ = None
        self.label_encoder_ = None

    def fit(self, X, y, sample_weight=None):
        """Fit the ElasticNet model.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : array-like (n_samples,)
            Binary labels (classification) or continuous scores (regression).
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        # Encode string labels
        if y_arr.dtype.kind in ("U", "S", "O"):
            self.label_encoder_ = LabelEncoder()
            y_arr = self.label_encoder_.fit_transform(y_arr)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)

        if self.mode == "classification":
            self.model_ = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=1.0 / max(self.alpha, 1e-8),
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                random_state=42,
            )
            self.model_.fit(X_scaled, y_arr.astype(int), sample_weight=sample_weight)
        else:
            self.model_ = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                random_state=42,
            )
            self.model_.fit(X_scaled, y_arr.astype(float), sample_weight=sample_weight)

        return self

    def predict(self, X):
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=float))
        preds = self.model_.predict(X_scaled)
        if self.label_encoder_ is not None:
            preds = self.label_encoder_.inverse_transform(preds.astype(int))
        return preds

    def predict_proba(self, X):
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=float))
        if self.mode == "classification" and hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X_scaled)
        else:
            # For regression, return point prediction clipped to [0, 1]
            preds = self.model_.predict(X_scaled)
            proba = np.clip(preds, 0, 1)
            return np.column_stack([1 - proba, proba])

    def get_top_features(self, feature_names=None, n_top=50):
        """Get top features by absolute coefficient magnitude."""
        if self.mode == "classification":
            coefs = self.model_.coef_[0]
        else:
            coefs = self.model_.coef_

        abs_coefs = np.abs(coefs)
        top_idx = np.argsort(abs_coefs)[-n_top:][::-1]

        if feature_names is not None:
            names = np.asarray(feature_names)
            return list(zip(names[top_idx], coefs[top_idx]))
        return list(zip(top_idx, coefs[top_idx]))


class CentroidBaseline(BaseEstimator, ClassifierMixin):
    """Centroid correlation classifier (Multiscale/Jacobson approach).

    Computes class centroids (mean expression per class) and classifies
    new samples by Pearson correlation to each centroid.

    Parameters
    ----------
    n_genes : int or None
        Number of top variable genes to use. None = all genes.
    """

    def __init__(self, n_genes: Optional[int] = None):
        self.n_genes = n_genes
        self.centroids_ = None
        self.classes_ = None
        self.gene_mask_ = None
        self.scaler_ = None

    def fit(self, X, y):
        """Compute class centroids from training data."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        if y_arr.dtype.kind in ("U", "S", "O"):
            self._le = LabelEncoder()
            y_arr = self._le.fit_transform(y_arr)
            self.classes_ = self._le.classes_
        else:
            self._le = None
            self.classes_ = np.unique(y_arr)

        # Select top variable genes
        if self.n_genes is not None and self.n_genes < X_arr.shape[1]:
            gene_var = np.var(X_arr, axis=0)
            self.gene_mask_ = np.argsort(gene_var)[-self.n_genes:]
            X_arr = X_arr[:, self.gene_mask_]
        else:
            self.gene_mask_ = np.arange(X_arr.shape[1])

        # Scale
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)

        # Compute centroids
        self.centroids_ = {}
        for cls in np.unique(y_arr):
            mask = y_arr == cls
            self.centroids_[cls] = X_scaled[mask].mean(axis=0)

        return self

    def predict(self, X):
        """Classify by nearest centroid (Pearson correlation)."""
        proba = self.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        if self._le is not None:
            return self._le.inverse_transform(pred_idx)
        return np.array(list(self.centroids_.keys()))[pred_idx]

    def predict_proba(self, X):
        """Compute centroid correlations as pseudo-probabilities."""
        X_arr = np.asarray(X, dtype=float)
        if self.gene_mask_ is not None:
            X_arr = X_arr[:, self.gene_mask_]
        X_scaled = self.scaler_.transform(X_arr)

        corrs = np.zeros((len(X_arr), len(self.centroids_)))
        for i, (cls, centroid) in enumerate(sorted(self.centroids_.items())):
            for j in range(len(X_arr)):
                r = np.corrcoef(X_scaled[j], centroid)[0, 1]
                corrs[j, i] = r

        # Convert correlations to pseudo-probabilities via softmax
        exp_corrs = np.exp(corrs - corrs.max(axis=1, keepdims=True))
        proba = exp_corrs / exp_corrs.sum(axis=1, keepdims=True)
        return proba


class SingleGeneBaseline(BaseEstimator, ClassifierMixin):
    """Single-gene predictors for HRD status.

    Uses individual gene expression levels (e.g. POLQ, BRCA1, BRCA2)
    as univariate classifiers via optimal threshold selection.

    Parameters
    ----------
    gene_names : list of str
        Genes to evaluate as individual predictors.
    """

    def __init__(self, gene_names: list[str] | None = None):
        self.gene_names = gene_names or ["POLQ", "BRCA1", "BRCA2"]
        self.thresholds_: dict = {}
        self.directions_: dict = {}  # 'higher' or 'lower' predicts HRD
        self.aucs_: dict = {}
        self.best_gene_: Optional[str] = None
        self._scaler = None
        self._gene_indices = None

    def fit(self, X, y, feature_names=None):
        """Find optimal thresholds for each gene.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : array-like of {0, 1}
        feature_names : list of str or None
            Column names. If X is a DataFrame, extracted automatically.
        """
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X, dtype=float)

        y_arr = np.asarray(y)
        if y_arr.dtype.kind in ("U", "S", "O"):
            le = LabelEncoder()
            y_arr = le.fit_transform(y_arr)

        if feature_names is None:
            logger.warning("No feature names provided; using column indices.")
            feature_names = [str(i) for i in range(X_arr.shape[1])]

        self._gene_indices = {}
        for gene in self.gene_names:
            if gene in feature_names:
                self._gene_indices[gene] = feature_names.index(gene)

        best_auc = 0.0
        for gene, idx in self._gene_indices.items():
            values = X_arr[:, idx]

            # Try both directions
            try:
                auc_high = roc_auc_score(y_arr, values)
            except ValueError:
                auc_high = 0.5

            auc_low = 1.0 - auc_high

            if auc_high >= auc_low:
                self.directions_[gene] = "higher"
                self.aucs_[gene] = auc_high
                scored = values
            else:
                self.directions_[gene] = "lower"
                self.aucs_[gene] = auc_low
                scored = -values

            # Find optimal threshold (Youden's J)
            unique_vals = np.unique(scored)
            best_j, best_t = 0.0, np.median(scored)
            for t in unique_vals:
                pred = (scored >= t).astype(int)
                tp = ((pred == 1) & (y_arr == 1)).sum()
                tn = ((pred == 0) & (y_arr == 0)).sum()
                fn = ((pred == 0) & (y_arr == 1)).sum()
                fp = ((pred == 1) & (y_arr == 0)).sum()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                j = sens + spec - 1
                if j > best_j:
                    best_j = j
                    best_t = t

            self.thresholds_[gene] = best_t
            if self.aucs_[gene] > best_auc:
                best_auc = self.aucs_[gene]
                self.best_gene_ = gene

        return self

    def predict(self, X, gene=None):
        """Predict using a single gene (default: best gene from fit)."""
        gene = gene or self.best_gene_
        if gene is None:
            raise RuntimeError("No gene selected. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            values = X[gene].values if gene in X.columns else np.zeros(len(X))
        else:
            idx = self._gene_indices.get(gene)
            if idx is None:
                raise ValueError(f"Gene {gene} not found")
            values = np.asarray(X)[:, idx]

        if self.directions_[gene] == "lower":
            values = -values

        return (values >= self.thresholds_[gene]).astype(int)

    def predict_proba(self, X, gene=None):
        """Return pseudo-probabilities based on gene expression z-score."""
        gene = gene or self.best_gene_
        if isinstance(X, pd.DataFrame):
            values = X[gene].values.astype(float) if gene in X.columns else np.zeros(len(X))
        else:
            idx = self._gene_indices.get(gene)
            values = np.asarray(X, dtype=float)[:, idx]

        # Z-score → sigmoid as pseudo-probability
        z = (values - values.mean()) / (values.std() + 1e-8)
        if self.directions_[gene] == "lower":
            z = -z
        proba = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - proba, proba])

    def summary(self) -> pd.DataFrame:
        """Return a summary of per-gene AUCs."""
        rows = []
        for gene in self.gene_names:
            if gene in self.aucs_:
                rows.append({
                    "gene": gene,
                    "auc": self.aucs_[gene],
                    "direction": self.directions_[gene],
                    "threshold": self.thresholds_[gene],
                    "is_best": gene == self.best_gene_,
                })
        return pd.DataFrame(rows)
