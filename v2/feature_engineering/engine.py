"""
FeatureEngine — unified interface for all feature engineering steps.

Orchestrates rank-pair features, pathway scores, PROGENy activities, and
CNV-expression features into a single feature matrix ready for model training.
"""

import logging

import numpy as np
import pandas as pd

from .rank_pairs import RankPairFeatures
from .pathway_scores import PathwayScorer
from .progeny import PROGENyScorer
from .cnv_expression import CNVExpressionFeatures

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Unified feature engineering pipeline for HRD classification.

    Transforms a raw expression matrix (and optionally CNV data) into a
    multi-scale feature set combining:
      - Rank-based pairwise gene comparisons (platform-agnostic binary features)
      - ssGSEA pathway activity scores (coordinated gene set behavior)
      - PROGENy signaling pathway activities (perturbation-based footprints)
      - CNV-expression features (dosage sensitivity, CN-adjusted residuals)

    Parameters
    ----------
    rank_pairs : bool
        Include rank-based pair features. Requires fit() with labels.
    pathway_scores : bool
        Include ssGSEA/pathway activity scores.
    pathway_method : str
        Method for pathway scoring: 'ssgsea', 'mean_rank', or 'zscore'.
    progeny : bool
        Include PROGENy pathway activities.
    cnv_features : bool
        Include CNV-expression features. Requires CNV data.
    n_pairs : int
        Number of rank-based gene pairs to discover.
    custom_gene_sets : dict or None
        Additional gene sets for pathway scoring.
    """

    def __init__(
        self,
        rank_pairs=True,
        pathway_scores=True,
        pathway_method="ssgsea",
        progeny=True,
        cnv_features=True,
        n_pairs=50,
        custom_gene_sets=None,
    ):
        self.rank_pairs = rank_pairs
        self.pathway_scores = pathway_scores
        self.pathway_method = pathway_method
        self.progeny = progeny
        self.cnv_features = cnv_features

        # Initialize components
        self._rank_pair_model = RankPairFeatures(n_pairs=n_pairs) if rank_pairs else None
        self._pathway_scorer = PathwayScorer(
            gene_sets=custom_gene_sets, method=pathway_method
        ) if pathway_scores else None
        self._progeny_scorer = PROGENyScorer() if progeny else None
        self._cnv_model = CNVExpressionFeatures() if cnv_features else None

        self._is_fitted = False

    def fit(self, expr, labels=None, cnv=None):
        """Fit feature engineering components on training data.

        Parameters
        ----------
        expr : pd.DataFrame
            Expression matrix (genes x samples or samples x genes).
        labels : pd.Series or None
            Binary HRD labels. Required if rank_pairs=True.
        cnv : pd.DataFrame or None
            Copy number data. Required if cnv_features=True.

        Returns
        -------
        self
        """
        if self.rank_pairs:
            if labels is None:
                raise ValueError("labels required when rank_pairs=True")
            self._rank_pair_model.fit(expr, labels)

        if self.cnv_features and cnv is not None:
            self._cnv_model.fit(expr, cnv)

        self._is_fitted = True
        return self

    def transform(self, expr, cnv=None):
        """Transform expression data into the engineered feature matrix.

        Parameters
        ----------
        expr : pd.DataFrame
            Expression matrix.
        cnv : pd.DataFrame or None
            Copy number data (optional).

        Returns
        -------
        pd.DataFrame
            Combined feature matrix (samples x features).
        """
        feature_blocks = []
        block_names = []

        # 1. Rank-based pairwise features
        if self.rank_pairs and self._rank_pair_model is not None and self._rank_pair_model.pairs_ is not None:
            rp = self._rank_pair_model.transform(expr)
            feature_blocks.append(rp)
            block_names.append(f"rank_pairs ({rp.shape[1]})")

        # 2. Pathway activity scores
        if self.pathway_scores and self._pathway_scorer is not None:
            pw = self._pathway_scorer.transform(expr)
            # Prefix column names
            pw.columns = [f"pathway_{c}" for c in pw.columns]
            feature_blocks.append(pw)
            block_names.append(f"pathway_scores ({pw.shape[1]})")

        # 3. PROGENy signaling activities
        if self.progeny and self._progeny_scorer is not None:
            pg = self._progeny_scorer.transform(expr)
            pg.columns = [f"progeny_{c}" for c in pg.columns]
            feature_blocks.append(pg)
            block_names.append(f"progeny ({pg.shape[1]})")

        # 4. CNV-expression features
        if self.cnv_features and self._cnv_model is not None:
            cv = self._cnv_model.transform(expr, cnv)
            feature_blocks.append(cv)
            block_names.append(f"cnv_expr ({cv.shape[1]})")

        if not feature_blocks:
            raise RuntimeError("No feature blocks generated. Check configuration.")

        # Align all blocks on their sample index and concatenate
        combined = self._align_and_concat(feature_blocks)

        logger.info(
            "Feature matrix: %d samples x %d features [%s]",
            combined.shape[0], combined.shape[1], ", ".join(block_names),
        )
        return combined

    def fit_transform(self, expr, labels=None, cnv=None):
        """Fit and transform in one step."""
        return self.fit(expr, labels, cnv).transform(expr, cnv)

    def feature_summary(self):
        """Return a summary of the feature blocks and their sizes."""
        info = {}
        if self.rank_pairs and self._rank_pair_model is not None:
            n = len(self._rank_pair_model.pairs_) if self._rank_pair_model.pairs_ else 0
            info["rank_pairs"] = n
        if self.pathway_scores:
            info["pathway_scores"] = len(self._pathway_scorer.gene_sets) if self._pathway_scorer else 0
        if self.progeny:
            info["progeny_pathways"] = 14  # number of PROGENy pathways
        if self.cnv_features:
            info["cnv_dosage_genes"] = len(self._cnv_model.dosage_genes) if self._cnv_model else 0
        return info

    @staticmethod
    def _align_and_concat(blocks):
        """Align feature blocks on sample index and concatenate."""
        if len(blocks) == 1:
            return blocks[0]

        # Find common samples
        common_idx = blocks[0].index
        for block in blocks[1:]:
            common_idx = common_idx.intersection(block.index)

        if len(common_idx) == 0:
            raise ValueError("No common samples across feature blocks")

        aligned = [block.loc[common_idx] for block in blocks]
        return pd.concat(aligned, axis=1)
