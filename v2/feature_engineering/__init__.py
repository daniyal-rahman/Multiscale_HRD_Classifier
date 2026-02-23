"""
Feature engineering module for HRD classification.

Provides platform-agnostic feature transformations that are robust to batch effects:
  - Rank-based pairwise gene comparisons (like 16-GPS / k-TSP)
  - Pathway-level activity scores (ssGSEA, GSVA-like, PROGENy)
  - Expression ratio features between functionally related genes
  - Combined HRD feature pipeline
  - Copy-number-derived expression effects
  - Centroid correlation features (from v1 signature)

Main entry point: HRDFeaturePipeline.fit_transform(expression_matrix, labels)
"""

from .rank_pairs import RankPairFeatures
from .pathway_scores import PathwayScorer
from .progeny import PROGENyScorer
from .feature_transforms import (
    RankBasedFeatures,
    PathwayScores,
    ExpressionRatios,
    HRDFeaturePipeline,
)

from .cnv_expression import CNVExpressionFeatures
from .engine import FeatureEngine
