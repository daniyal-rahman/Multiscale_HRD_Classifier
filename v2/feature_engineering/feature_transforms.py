"""
Unified feature engineering pipeline for HRD classification.

Provides platform-agnostic feature transformations that survive batch effects
across RNA-seq (TCGA), microarray (I-SPY2), and cell-line platforms:

  1. RankBasedFeatures — pairwise gene comparisons (k-TSP / 16-GPS style)
  2. PathwayScores     — ssGSEA / mean-rank / z-score pathway activity
  3. ExpressionRatios  — log-ratios between functionally related genes
  4. HRDFeaturePipeline — chains all of the above with sklearn-like API

All transformers follow the sklearn fit/transform convention and operate on
pandas DataFrames (genes x samples or samples x genes, auto-detected).
"""

import logging
from itertools import combinations

import numpy as np
import pandas as pd

from .gene_sets import GENE_SETS, HR_REPAIR, DDR_BROAD, ALT_EJ, NHEJ
from .rank_pairs import RankPairFeatures
from .pathway_scores import PathwayScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curated gene sets for rank-pair candidate generation
# ---------------------------------------------------------------------------
# Merge pathway gene lists into a broad candidate set for pair discovery.
# The union covers HRR, DDR, alt-EJ, NHEJ, and key cell-cycle/replication genes.
_DEFAULT_CANDIDATE_GENES = sorted(set(
    HR_REPAIR + DDR_BROAD + ALT_EJ + NHEJ
    + GENE_SETS.get("cell_cycle", [])
    + GENE_SETS.get("replication_stress", [])
))

# Default expression-ratio pairs (numerator, denominator, biological rationale)
_DEFAULT_RATIO_PAIRS = [
    ("BRCA1", "BRCA2", "HR sub-pathway balance"),
    ("RAD51", "BRCA2", "RAD51 loading relative to BRCA2"),
    ("POLQ", "RAD51", "alt-EJ vs HR repair balance — POLQ upregulated in HRD"),
    ("PARP1", "BRCA1", "PARP dependency vs BRCA1 status"),
    ("POLQ", "BRCA1", "alt-EJ compensation for BRCA1 loss"),
    ("POLQ", "BRCA2", "alt-EJ compensation for BRCA2 loss"),
    ("CHEK1", "CHEK2", "ATR-CHEK1 vs ATM-CHEK2 axis preference"),
    ("ATR", "ATM", "replication-stress vs DSB signaling"),
    ("EXO1", "TP53BP1", "resection vs end-protection balance"),
    ("CCNE1", "RB1", "cell-cycle deregulation / replication stress"),
    ("E2F1", "RB1", "E2F activation"),
    ("MKI67", "CDKN1A", "proliferation vs p21 arrest"),
    ("CDK1", "CDKN1A", "CDK activity vs p21"),
    ("PARP1", "PARP2", "PARP isoform balance"),
    ("FANCD2", "FANCA", "FA pathway activation vs upstream"),
]


# ===================================================================
# 1. Rank-Based Gene Pair Features
# ===================================================================
class RankBasedFeatures:
    """Discover and apply platform-agnostic rank-based gene pair features.

    Wraps :class:`RankPairFeatures` from ``rank_pairs.py`` with the API
    requested by the project spec (gene_sets dict, max_pairs in fit).

    Parameters
    ----------
    gene_sets : dict[str, list[str]] or None
        Named gene sets whose union forms the candidate gene pool.
        If None, uses the default HRD-relevant candidate set.
    n_pairs : int
        Maximum pairs to retain (can be overridden in ``fit``).
    min_delta : float
        Minimum |P(A>B|HRD) - P(A>B|HRP)| to keep a pair.
    """

    def __init__(self, gene_sets: dict = None, n_pairs: int = 500,
                 min_delta: float = 0.15):
        if gene_sets is not None:
            candidates = sorted(set(g for gl in gene_sets.values() for g in gl))
        else:
            candidates = _DEFAULT_CANDIDATE_GENES
        self.gene_sets = gene_sets
        self.candidates = candidates
        self.n_pairs = n_pairs
        self.min_delta = min_delta
        self._rpf = None  # fitted RankPairFeatures instance

    def fit(self, X: pd.DataFrame, y: pd.Series,
            max_pairs: int = None) -> "RankBasedFeatures":
        """Identify the most discriminative gene pairs from labeled data.

        Parameters
        ----------
        X : pd.DataFrame
            Expression matrix (genes x samples or samples x genes).
        y : pd.Series
            Binary HRD labels aligned to sample IDs.
        max_pairs : int or None
            Override ``n_pairs`` for this call.

        Returns
        -------
        self
        """
        n = max_pairs or self.n_pairs
        self._rpf = RankPairFeatures(
            n_pairs=n,
            min_delta=self.min_delta,
            candidate_genes=self.candidates,
        )
        self._rpf.fit(X, y)
        logger.info("RankBasedFeatures.fit: selected %d pairs", len(self._rpf.pairs_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform expression matrix into binary gene-pair features."""
        if self._rpf is None:
            raise RuntimeError("Must call fit() before transform()")
        return self._rpf.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                      max_pairs: int = None) -> pd.DataFrame:
        return self.fit(X, y, max_pairs=max_pairs).transform(X)

    def get_pair_info(self) -> pd.DataFrame:
        """Return a DataFrame describing fitted pairs and their deltas."""
        if self._rpf is None:
            return pd.DataFrame()
        return self._rpf.get_pair_info()

    @property
    def n_features_(self) -> int:
        if self._rpf is None or self._rpf.pairs_ is None:
            return 0
        return len(self._rpf.pairs_)


# ===================================================================
# 2. Pathway Activity Scores
# ===================================================================
class PathwayScores:
    """Compute per-sample pathway activity scores.

    Wraps :class:`PathwayScorer` from ``pathway_scores.py`` and adds
    convenience methods for HRD-relevant pathway subsets.

    Parameters
    ----------
    gene_set_db : str
        Which built-in gene sets to use: ``'hallmark'`` (default) uses the
        curated HRD-relevant sets from ``gene_sets.py``.  ``'custom'``
        requires passing ``custom_gene_sets``.
    method : str
        Scoring method: ``'ssgsea'``, ``'mean_rank'``, or ``'zscore'``.
    min_genes : int
        Minimum overlap with expression matrix to score a pathway.
    custom_gene_sets : dict or None
        Custom gene sets (only used when ``gene_set_db='custom'``).
    """

    # Pathways most directly relevant to HRD biology
    HRD_PATHWAYS = [
        "HR_repair", "NHEJ", "alt_EJ", "fanconi_anemia",
        "DDR_broad", "cell_cycle", "replication_stress",
    ]

    def __init__(self, gene_set_db: str = "hallmark", method: str = "ssgsea",
                 min_genes: int = 5, custom_gene_sets: dict = None):
        self.gene_set_db = gene_set_db
        self.method = method
        self.min_genes = min_genes

        if gene_set_db == "custom":
            if custom_gene_sets is None:
                raise ValueError("custom_gene_sets required when gene_set_db='custom'")
            gs = custom_gene_sets
        else:
            gs = GENE_SETS  # built-in HRD-relevant sets
        self._scorer = PathwayScorer(gene_sets=gs, method=method,
                                     min_genes=min_genes)

    def compute_ssgsea(self, expression_df: pd.DataFrame,
                       gene_sets: dict = None) -> pd.DataFrame:
        """Compute single-sample GSEA scores.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix.
        gene_sets : dict or None
            Override gene sets for this call.  If None, uses the instance sets.

        Returns
        -------
        pd.DataFrame
            Pathway scores (samples x pathways).
        """
        if gene_sets is not None:
            scorer = PathwayScorer(gene_sets=gene_sets, method="ssgsea",
                                   min_genes=self.min_genes)
        else:
            scorer = PathwayScorer(gene_sets=self._scorer.gene_sets,
                                   method="ssgsea", min_genes=self.min_genes)
        return scorer.transform(expression_df)

    def compute_hrd_relevant_scores(self, expression_df: pd.DataFrame) -> pd.DataFrame:
        """Compute scores for HRD-relevant pathways only.

        Returns scores for HR repair, NHEJ, alt-EJ, Fanconi anemia, DDR,
        cell cycle, and replication stress pathways.
        """
        hrd_sets = {k: v for k, v in GENE_SETS.items() if k in self.HRD_PATHWAYS}
        scorer = PathwayScorer(gene_sets=hrd_sets, method=self.method,
                               min_genes=self.min_genes)
        return scorer.transform(expression_df)

    def transform(self, expression_df: pd.DataFrame) -> pd.DataFrame:
        """Compute all pathway scores (sklearn-compatible)."""
        return self._scorer.transform(expression_df)

    def fit(self, X: pd.DataFrame, y=None) -> "PathwayScores":
        """No-op fit for sklearn pipeline compatibility."""
        return self

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# ===================================================================
# 3. Expression Ratio Features
# ===================================================================
class ExpressionRatios:
    """Compute log-ratios between functionally related gene pairs.

    Instead of absolute expression levels, ratios capture relative pathway
    activity and are more robust to global normalization differences.

    Parameters
    ----------
    ratio_pairs : list of (str, str) or (str, str, str) or None
        Gene pairs as (numerator, denominator) or
        (numerator, denominator, description).
        If None, uses the default HRD-relevant ratio pairs.
    pseudocount : float
        Added to expression values before log-ratio to avoid log(0).
        Set to 0 if data is already log-transformed.
    log_transform : bool
        If True, compute log2(num / denom). If False, compute raw ratio.
        Log-ratios are preferred for symmetry around zero.
    """

    def __init__(self, ratio_pairs: list = None, pseudocount: float = 1.0,
                 log_transform: bool = True):
        if ratio_pairs is not None:
            # Normalize to (num, denom, desc) tuples
            self.ratio_pairs = []
            for p in ratio_pairs:
                if len(p) == 2:
                    self.ratio_pairs.append((p[0], p[1], f"{p[0]}/{p[1]}"))
                else:
                    self.ratio_pairs.append(tuple(p[:3]))
        else:
            self.ratio_pairs = [(n, d, desc) for n, d, desc in _DEFAULT_RATIO_PAIRS]
        self.pseudocount = pseudocount
        self.log_transform = log_transform

    def compute_ratios(self, expression_df: pd.DataFrame) -> pd.DataFrame:
        """Compute expression ratios for all configured gene pairs.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix (genes x samples or samples x genes).

        Returns
        -------
        pd.DataFrame
            Ratio features (samples x ratio_pairs).
        """
        expr = self._orient_genes_x_samples(expression_df)
        available = set(expr.index)

        ratios = {}
        skipped = []
        for num, denom, desc in self.ratio_pairs:
            if num not in available or denom not in available:
                skipped.append(f"{num}/{denom}")
                continue

            col_name = f"ratio_{num}_over_{denom}"
            num_vals = expr.loc[num].values.astype(float)
            denom_vals = expr.loc[denom].values.astype(float)

            if self.log_transform:
                # log2((num + pc) / (denom + pc))
                ratios[col_name] = np.log2(
                    (num_vals + self.pseudocount) /
                    (denom_vals + self.pseudocount + 1e-10)
                )
            else:
                ratios[col_name] = (
                    (num_vals + self.pseudocount) /
                    (denom_vals + self.pseudocount + 1e-10)
                )

        if skipped:
            logger.info("ExpressionRatios: skipped %d pairs (missing genes): %s",
                        len(skipped), ", ".join(skipped[:5]))

        result = pd.DataFrame(ratios, index=expr.columns)
        logger.info("Computed %d expression ratio features for %d samples",
                     result.shape[1], result.shape[0])
        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sklearn-compatible transform alias."""
        return self.compute_ratios(X)

    def fit(self, X: pd.DataFrame, y=None) -> "ExpressionRatios":
        """No-op fit for sklearn pipeline compatibility.

        Ratios are predefined, so no fitting is needed. However, we could
        optionally use ``y`` to select the most discriminative ratios
        (not implemented yet — use ``select_top_ratios`` instead).
        """
        return self

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def select_top_ratios(self, X: pd.DataFrame, y: pd.Series,
                          top_k: int = 10) -> list:
        """Rank ratio features by discriminative power (AUC) and return top k.

        Parameters
        ----------
        X : pd.DataFrame
            Expression matrix.
        y : pd.Series
            Binary labels.
        top_k : int
            Number of top ratios to return.

        Returns
        -------
        list of (col_name, auc) tuples, sorted by descending AUC distance from 0.5.
        """
        ratios = self.compute_ratios(X)
        expr = self._orient_genes_x_samples(X)

        # Align labels
        common = ratios.index.intersection(y.index)
        ratios = ratios.loc[common]
        y_aligned = y.loc[common]

        if pd.api.types.is_string_dtype(y_aligned):
            mask = y_aligned.isin(["HRD", "hrd", 1, "1", True])
        else:
            mask = y_aligned.astype(bool)

        scored = []
        for col in ratios.columns:
            vals = ratios[col].values
            hrd_vals = vals[mask.values]
            hrp_vals = vals[~mask.values]
            # Mann-Whitney U → AUC approximation
            n1, n2 = len(hrd_vals), len(hrp_vals)
            if n1 == 0 or n2 == 0:
                continue
            # Simple AUC: fraction of (hrd, hrp) pairs where hrd > hrp
            auc = np.mean(hrd_vals[:, None] > hrp_vals[None, :])
            scored.append((col, auc))

        scored.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)
        return scored[:top_k]

    @staticmethod
    def _orient_genes_x_samples(expr):
        """Orient so rows = genes, columns = samples."""
        if expr.shape[0] < expr.shape[1]:
            return expr
        sample_idx = str(expr.index[0])
        if any(c.isdigit() for c in sample_idx) and "-" in sample_idx:
            return expr.T
        return expr

    @property
    def n_features_(self) -> int:
        return len(self.ratio_pairs)


# ===================================================================
# 4. Combined Feature Pipeline
# ===================================================================
class HRDFeaturePipeline:
    """Chains rank-pair, pathway-score, and ratio features into one transform.

    Follows the sklearn fit/transform API so it can be plugged directly
    into an sklearn Pipeline or cross-validation loop.

    Parameters
    ----------
    use_rank_pairs : bool
        Include binary gene-pair features.
    use_pathway_scores : bool
        Include ssGSEA/rank pathway activity scores.
    use_ratios : bool
        Include expression log-ratio features.
    use_raw_expression : bool
        Include raw (z-scored) expression for a set of key genes.
    rank_kwargs : dict
        Extra kwargs for :class:`RankBasedFeatures`.
    pathway_kwargs : dict
        Extra kwargs for :class:`PathwayScores`.
    ratio_kwargs : dict
        Extra kwargs for :class:`ExpressionRatios`.
    raw_genes : list[str] or None
        Genes to include when ``use_raw_expression=True``.
        Defaults to the HR_REPAIR + ALT_EJ gene set.
    """

    def __init__(self, use_rank_pairs: bool = True,
                 use_pathway_scores: bool = True,
                 use_ratios: bool = True,
                 use_raw_expression: bool = False,
                 rank_kwargs: dict = None,
                 pathway_kwargs: dict = None,
                 ratio_kwargs: dict = None,
                 raw_genes: list = None):
        self.use_rank_pairs = use_rank_pairs
        self.use_pathway_scores = use_pathway_scores
        self.use_ratios = use_ratios
        self.use_raw_expression = use_raw_expression

        self._rank = RankBasedFeatures(**(rank_kwargs or {})) if use_rank_pairs else None
        self._pathway = PathwayScores(**(pathway_kwargs or {})) if use_pathway_scores else None
        self._ratios = ExpressionRatios(**(ratio_kwargs or {})) if use_ratios else None
        self._raw_genes = raw_genes or (HR_REPAIR + ALT_EJ)

        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "HRDFeaturePipeline":
        """Fit supervised components (rank pairs require labels).

        Parameters
        ----------
        X : pd.DataFrame
            Expression matrix.
        y : pd.Series or None
            Binary HRD labels. Required if ``use_rank_pairs=True``.
        """
        if self.use_rank_pairs:
            if y is None:
                raise ValueError("Labels (y) required to fit rank-pair features")
            self._rank.fit(X, y)
        if self._pathway is not None:
            self._pathway.fit(X, y)
        if self._ratios is not None:
            self._ratios.fit(X, y)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate all feature types and concatenate column-wise.

        Returns
        -------
        pd.DataFrame
            Combined features (samples x total_features). Column names
            are prefixed by type: ``rp_``, ``pw_``, ``rt_``, ``raw_``.
        """
        parts = []

        if self.use_rank_pairs and self._rank is not None:
            rp = self._rank.transform(X)
            rp = rp.add_prefix("rp_")
            parts.append(rp)

        if self.use_pathway_scores and self._pathway is not None:
            pw = self._pathway.transform(X)
            pw = pw.add_prefix("pw_")
            parts.append(pw)

        if self.use_ratios and self._ratios is not None:
            rt = self._ratios.transform(X)
            rt = rt.add_prefix("rt_")
            parts.append(rt)

        if self.use_raw_expression:
            raw = self._extract_raw(X)
            raw = raw.add_prefix("raw_")
            parts.append(raw)

        if not parts:
            raise ValueError("No feature types enabled")

        # Align sample indices across feature types
        common_idx = parts[0].index
        for p in parts[1:]:
            common_idx = common_idx.intersection(p.index)
        result = pd.concat([p.loc[common_idx] for p in parts], axis=1)

        logger.info(
            "HRDFeaturePipeline: %d total features (%s) for %d samples",
            result.shape[1],
            " + ".join(
                f"{name}={p.shape[1]}"
                for name, p in zip(
                    ["rank_pairs", "pathway", "ratios", "raw"],
                    parts,
                )
            ),
            result.shape[0],
        )
        return result

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _extract_raw(self, X: pd.DataFrame) -> pd.DataFrame:
        """Z-score a set of key genes across samples."""
        expr = self._orient_genes_x_samples(X)
        available = [g for g in self._raw_genes if g in expr.index]
        raw = expr.loc[available].T  # samples x genes
        # Z-score each gene across samples
        raw = (raw - raw.mean()) / (raw.std() + 1e-8)
        return raw

    @staticmethod
    def _orient_genes_x_samples(expr):
        if expr.shape[0] < expr.shape[1]:
            return expr
        sample_idx = str(expr.index[0])
        if any(c.isdigit() for c in sample_idx) and "-" in sample_idx:
            return expr.T
        return expr

    def feature_summary(self) -> dict:
        """Return a summary of feature counts by type."""
        summary = {}
        if self.use_rank_pairs and self._rank is not None:
            summary["rank_pairs"] = self._rank.n_features_
        if self.use_pathway_scores:
            summary["pathway_scores"] = len(GENE_SETS)
        if self.use_ratios and self._ratios is not None:
            summary["ratios"] = self._ratios.n_features_
        if self.use_raw_expression:
            summary["raw_expression"] = len(self._raw_genes)
        return summary
