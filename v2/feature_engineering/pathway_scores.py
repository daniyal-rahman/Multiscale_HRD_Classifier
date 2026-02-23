"""
Pathway-level activity scoring: ssGSEA and rank-based enrichment.

Computes per-sample pathway activity scores that summarize expression of
gene sets into single values. These are more robust to batch effects than
individual gene expression because they capture coordinated pathway activity.

Implements:
  1. ssGSEA (single-sample GSEA) — rank-based enrichment per sample
  2. Mean-rank scoring — simple average of within-sample gene ranks
  3. Z-score aggregation — mean z-score of gene set members

These are pure Python/numpy implementations with no R dependency.
"""

import logging

import numpy as np
import pandas as pd

from .gene_sets import GENE_SETS

logger = logging.getLogger(__name__)


class PathwayScorer:
    """Compute pathway activity scores for expression matrices.

    Parameters
    ----------
    gene_sets : dict[str, list[str]] or None
        Dictionary of {pathway_name: [gene1, gene2, ...]}. If None, uses
        the built-in HRD-relevant gene sets from gene_sets.py.
    method : str
        Scoring method: 'ssgsea', 'mean_rank', or 'zscore'.
    min_genes : int
        Minimum number of overlapping genes required to score a pathway.
    """

    def __init__(self, gene_sets=None, method="ssgsea", min_genes=5):
        self.gene_sets = gene_sets or GENE_SETS
        self.method = method
        self.min_genes = min_genes

    def transform(self, expr):
        """Compute pathway scores for each sample.

        Parameters
        ----------
        expr : pd.DataFrame
            Expression matrix. Can be genes x samples or samples x genes.
            Auto-detected by shape heuristic.

        Returns
        -------
        pd.DataFrame
            Pathway scores (samples x pathways).
        """
        expr = self._orient_genes_x_samples(expr)
        available_genes = set(expr.index)

        scores = {}
        for pathway_name, gene_list in self.gene_sets.items():
            overlap = [g for g in gene_list if g in available_genes]
            if len(overlap) < self.min_genes:
                logger.debug(
                    "Skipping %s: only %d/%d genes available (min=%d)",
                    pathway_name, len(overlap), len(gene_list), self.min_genes,
                )
                continue

            if self.method == "ssgsea":
                scores[pathway_name] = self._ssgsea(expr, overlap)
            elif self.method == "mean_rank":
                scores[pathway_name] = self._mean_rank(expr, overlap)
            elif self.method == "zscore":
                scores[pathway_name] = self._zscore(expr, overlap)
            else:
                raise ValueError(f"Unknown method: {self.method}")

        result = pd.DataFrame(scores, index=expr.columns)
        logger.info("Computed %d pathway scores (%s) for %d samples",
                     result.shape[1], self.method, result.shape[0])
        return result

    # -- ssGSEA ----------------------------------------------------------------

    @staticmethod
    def _ssgsea(expr, gene_set, alpha=0.25):
        """Single-sample GSEA (Barbie et al. 2009).

        For each sample, genes are ranked by expression. The enrichment score
        is computed as the difference between a weighted ECDF of genes in the
        set vs genes not in the set, walking down the ranked list.

        Parameters
        ----------
        expr : pd.DataFrame
            genes x samples expression matrix.
        gene_set : list[str]
            Genes in the pathway.
        alpha : float
            Weighting exponent for the running sum. 0.25 is standard for ssGSEA.
        """
        n_genes = expr.shape[0]
        gene_set_mask = np.asarray(expr.index.isin(gene_set))
        n_in_set = gene_set_mask.sum()
        n_not_in_set = n_genes - n_in_set

        scores = np.zeros(expr.shape[1])

        for j in range(expr.shape[1]):
            # Rank genes by expression (descending)
            sample_vals = expr.iloc[:, j].values
            rank_order = np.argsort(-sample_vals)
            sorted_in_set = gene_set_mask[rank_order]

            # Weighted cumulative sum for genes in set
            abs_expr_ranked = np.abs(sample_vals[rank_order])
            weights_in = np.where(sorted_in_set, abs_expr_ranked ** alpha, 0.0)
            sum_weights_in = weights_in.sum()
            if sum_weights_in == 0:
                scores[j] = 0.0
                continue

            # Running sums
            p_hit = np.cumsum(weights_in) / sum_weights_in
            p_miss = np.cumsum(~sorted_in_set) / n_not_in_set

            # Enrichment score = max deviation
            es = p_hit - p_miss
            scores[j] = es.sum()  # ssGSEA uses sum, not max

        return scores

    # -- Mean rank scoring -----------------------------------------------------

    @staticmethod
    def _mean_rank(expr, gene_set):
        """Average within-sample rank of gene set members.

        Higher score = gene set members are more highly expressed in that sample.
        Completely platform-agnostic (rank-based).
        """
        # Rank each sample (column) in ascending order
        ranks = expr.rank(axis=0, method="average")
        # Normalize to [0, 1]
        ranks = ranks / ranks.max(axis=0)
        # Average rank of gene set members
        gene_set_mask = expr.index.isin(gene_set)
        return ranks.loc[gene_set_mask].mean(axis=0).values

    # -- Z-score aggregation ---------------------------------------------------

    @staticmethod
    def _zscore(expr, gene_set):
        """Mean z-score of gene set members per sample.

        Each gene is z-scored across samples, then the mean across gene set
        members gives the pathway activity per sample.
        """
        # Z-score each gene across samples
        z = (expr - expr.mean(axis=1).values[:, None]) / (expr.std(axis=1).values[:, None] + 1e-8)
        gene_set_mask = expr.index.isin(gene_set)
        return z.loc[gene_set_mask].mean(axis=0).values

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _orient_genes_x_samples(expr):
        """Orient so rows = genes, columns = samples."""
        if expr.shape[0] < expr.shape[1]:
            return expr
        sample_idx = str(expr.index[0])
        if any(c.isdigit() for c in sample_idx) and "-" in sample_idx:
            return expr.T
        return expr

    def available_pathways(self, expr):
        """List pathways that have enough genes in the expression matrix."""
        expr = self._orient_genes_x_samples(expr)
        genes = set(expr.index)
        result = {}
        for name, gene_list in self.gene_sets.items():
            overlap = [g for g in gene_list if g in genes]
            result[name] = {"total_genes": len(gene_list), "available": len(overlap)}
        return pd.DataFrame(result).T
