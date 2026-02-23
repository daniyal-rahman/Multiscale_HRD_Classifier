"""
Rank-based pairwise gene comparison features (k-TSP / 16-GPS style).

The key insight: if gene A is consistently more highly expressed than gene B in
HRD tumors but not in HRP tumors, the binary feature (A > B) is platform-agnostic
because it depends only on *relative ordering*, not absolute expression levels.

This module:
  1. Discovers discriminative gene pairs from labeled training data
  2. Generates binary features from any expression matrix using fitted pairs
  3. Is robust to batch effects, normalization differences, and platform changes

Reference: Paquet & Bhatt (2019) "Absolute Assignment of Breast Cancer Intrinsic
Molecular Subtype" (k-TSP); Li et al. (2010) 16-GPS for chemo response.
"""

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class RankPairFeatures:
    """Discover and apply rank-based gene pair features for HRD classification.

    Parameters
    ----------
    n_pairs : int
        Maximum number of gene pairs to select.
    min_delta : float
        Minimum difference in pair-ordering frequency between classes.
        A pair (A, B) is kept if |P(A>B | HRD) - P(A>B | HRP)| > min_delta.
    candidate_genes : list[str] or None
        Restrict pair search to these genes. If None, uses all genes.
    """

    def __init__(self, n_pairs=50, min_delta=0.15, candidate_genes=None):
        self.n_pairs = n_pairs
        self.min_delta = min_delta
        self.candidate_genes = candidate_genes
        self.pairs_ = None  # fitted list of (geneA, geneB, direction)

    def fit(self, expr, labels):
        """Discover discriminative gene pairs from training data.

        Parameters
        ----------
        expr : pd.DataFrame
            Gene expression matrix (genes x samples) or (samples x genes).
            Will be auto-oriented so that index = genes.
        labels : pd.Series
            Binary labels aligned to samples. Values should be 0/1 or
            'HRD'/'HRP' (any truthy/falsy).

        Returns
        -------
        self
        """
        expr, labels = self._orient_and_align(expr, labels)

        # Restrict to candidate genes if specified
        if self.candidate_genes is not None:
            available = expr.index.intersection(self.candidate_genes)
            expr = expr.loc[available]
            logger.info("Restricted to %d/%d candidate genes",
                        len(available), len(self.candidate_genes))

        genes = expr.index.tolist()
        n_genes = len(genes)
        logger.info("Searching gene pairs among %d genes (%d possible pairs)",
                     n_genes, n_genes * (n_genes - 1) // 2)

        # Convert labels to boolean (True = HRD)
        # Use pd.api.types to handle both object and StringDtype (pandas 3.x)
        if pd.api.types.is_string_dtype(labels):
            hrd_mask = labels.isin(["HRD", "hrd", 1, "1", True])
        else:
            hrd_mask = labels.astype(bool)

        hrd_expr = expr.loc[:, hrd_mask].values  # (n_genes, n_hrd)
        hrp_expr = expr.loc[:, ~hrd_mask].values  # (n_genes, n_hrp)

        # For efficiency with many genes, use a pre-screening step:
        # only consider gene pairs where at least one gene is differentially expressed
        if n_genes > 500:
            gene_scores = self._prescreen_genes(expr, hrd_mask, top_k=200)
            top_idx = gene_scores.argsort()[-200:]
            genes = [genes[i] for i in top_idx]
            hrd_expr = hrd_expr[top_idx]
            hrp_expr = hrp_expr[top_idx]
            n_genes = len(genes)
            logger.info("Pre-screened to top %d differentially expressed genes", n_genes)

        # Score all pairs: P(A > B | class)
        scored_pairs = []
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                # Frequency of gene_i > gene_j in each class
                p_hrd = np.mean(hrd_expr[i] > hrd_expr[j])
                p_hrp = np.mean(hrp_expr[i] > hrp_expr[j])
                delta = p_hrd - p_hrp

                if abs(delta) > self.min_delta:
                    scored_pairs.append((genes[i], genes[j], delta))

        # Sort by absolute delta (discriminative power) and take top n
        scored_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        self.pairs_ = scored_pairs[: self.n_pairs]

        logger.info("Selected %d gene pairs (min_delta=%.2f)",
                     len(self.pairs_), self.min_delta)
        return self

    def transform(self, expr):
        """Generate binary rank-pair features for new samples.

        Parameters
        ----------
        expr : pd.DataFrame
            Expression matrix (genes x samples) or (samples x genes).

        Returns
        -------
        pd.DataFrame
            Binary feature matrix (samples x pairs), with columns like
            "BRCA1_gt_RAD51C" indicating BRCA1 > RAD51C.
        """
        if self.pairs_ is None:
            raise RuntimeError("Must call fit() before transform()")

        expr = self._orient_to_genes_x_samples(expr)
        available_genes = set(expr.index)

        features = {}
        for gene_a, gene_b, delta in self.pairs_:
            if gene_a not in available_genes or gene_b not in available_genes:
                continue

            col_name = f"{gene_a}_gt_{gene_b}"
            if delta > 0:
                # In HRD: A > B more often → feature = 1 when A > B
                features[col_name] = (expr.loc[gene_a] > expr.loc[gene_b]).astype(int)
            else:
                # In HRD: A > B less often → flip: feature = 1 when B > A
                features[col_name] = (expr.loc[gene_b] > expr.loc[gene_a]).astype(int)

        result = pd.DataFrame(features, index=expr.columns)
        logger.info("Generated %d rank-pair features for %d samples",
                     result.shape[1], result.shape[0])
        return result

    def fit_transform(self, expr, labels):
        """Fit on training data and return features."""
        return self.fit(expr, labels).transform(expr)

    def get_pair_info(self):
        """Return a DataFrame describing the fitted pairs."""
        if self.pairs_ is None:
            return pd.DataFrame()
        return pd.DataFrame(self.pairs_, columns=["gene_A", "gene_B", "delta"])

    # -- internal helpers ------------------------------------------------------

    @staticmethod
    def _prescreen_genes(expr, hrd_mask, top_k=200):
        """Quick t-test pre-screening to reduce pair search space."""
        hrd_expr = expr.loc[:, hrd_mask].values
        hrp_expr = expr.loc[:, ~hrd_mask].values
        t_stats, _ = stats.ttest_ind(hrd_expr, hrp_expr, axis=1, equal_var=False)
        return np.abs(np.nan_to_num(t_stats))

    @staticmethod
    def _orient_to_genes_x_samples(expr):
        """Ensure expression is genes (rows) x samples (columns).

        Heuristic: if the number of rows >> columns, it's already genes x samples.
        If rows << columns, transpose.
        """
        if expr.shape[0] < expr.shape[1]:
            return expr
        # Check if index looks like gene names (contains letters)
        sample_idx = str(expr.index[0])
        if any(c.isdigit() and "-" in sample_idx for c in sample_idx):
            # Index looks like barcodes → samples x genes → transpose
            return expr.T
        return expr

    @staticmethod
    def _orient_and_align(expr, labels):
        """Orient expression and align labels to columns (samples)."""
        expr = RankPairFeatures._orient_to_genes_x_samples(expr)

        # Align labels with expression columns
        common = expr.columns.intersection(labels.index)
        if len(common) == 0:
            # Try truncating barcodes to 12 chars (TCGA convention)
            short_cols = pd.Index([str(c)[:12] for c in expr.columns])
            short_labels = pd.Index([str(c)[:12] for c in labels.index])
            common = short_cols.intersection(short_labels)
            if len(common) == 0:
                raise ValueError(
                    "No overlapping sample IDs between expression and labels. "
                    f"Expression columns: {list(expr.columns[:3])}..., "
                    f"Label index: {list(labels.index[:3])}..."
                )
            expr.columns = short_cols
            labels.index = short_labels

        expr = expr[common]
        labels = labels[common]
        return expr, labels
