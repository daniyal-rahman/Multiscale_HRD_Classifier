"""
Unified scoring interface for HRD signatures.

Supports:
  - softHRD v2 (trained model with predict_proba)
  - softHRD v1 / centroid-correlation approach (Jacobson et al.)
  - Published signatures: Severson, PARPi7, CIN70, Peng, POLQ single-gene
  - Any sklearn-compatible model with predict_proba() or decision_function()
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Published gene signatures (mean-expression based)
# ---------------------------------------------------------------------------

# Severson et al. 2017 — 14-gene HR-deficiency signature
SEVERSON_GENES = {
    "up_in_HRD": ["CXCL10", "STAT1", "CCL5", "AIM2", "APOBEC3G",
                   "ADAR", "IDO1"],
    "down_in_HRD": ["MAPT", "MYB", "MLPH", "NAT1", "SLC39A6",
                     "THSD4", "TTC36"],
}

# PARPi7 — 7-gene PARP-inhibitor response signature (Daemen et al.)
PARPI7_GENES = {
    "up_in_sensitive": ["BRCA1", "CHEK2", "MAPKAPK2"],
    "down_in_sensitive": ["SLFN11", "XPA", "E2F2", "XRCC1"],
}

# CIN70 — chromosomal instability signature (Carter et al.)
CIN70_GENES = [
    "TPX2", "PRC1", "FOXM1", "CDC2", "TGIF2", "MCM2", "H2AFZ", "TOP2A",
    "PCNA", "UBE2C", "MELK", "TRIP13", "RNASEH2A", "RAD51AP1", "KIF20A",
    "CDC45", "MAD2L1", "ESPL1", "CCNB2", "FEN1", "TTK", "CCT5", "RFC4",
    "ATAD2", "ch_TOF1", "CENPA", "NUP205", "CDC20", "CKS2", "RRM2",
    "ELAVL1", "CCNA2", "EZH2", "AURKA", "AURKB", "BUB1", "BUB1B",
    "CENPF", "KIF4A", "CENPE", "KIF2C", "MCM7", "ZWINT", "BIRC5",
    "PTTG1", "CDCA8", "NEK2", "ASF1B", "ECT2", "CEP55", "NDC80",
    "KIF11", "DLGAP5", "CDCA3", "OIP5", "HJURP", "DTL", "CCNB1",
    "NUSAP1", "UBE2T", "KIF23", "MCM10", "CDKN3", "NUF2", "SKA1",
    "SKA3", "NCAPH", "SGOL2", "CENPN", "KIF18A", "MKI67",
]

# Peng et al. — DNA repair deficiency signature genes (top HR genes)
PENG_GENES = {
    "up_in_HRD": ["POLQ", "CENPF", "EXO1", "FANCI", "FANCD2"],
    "down_in_HRD": ["BRCA1", "RAD51", "BRCA2", "PALB2", "RAD51C"],
}


class SignatureScorer:
    """Score samples using any HRD signature method.

    Parameters
    ----------
    method : str
        One of: 'softHRD_v2', 'softHRD_v1', 'severson', 'parpi7',
                'cin70', 'peng', 'polq', 'single_gene', 'custom_model'.
    model : object or None
        For 'softHRD_v2' or 'custom_model': a fitted model with
        predict_proba() or decision_function(). For 'softHRD_v1': ignored.
    feature_engine : object or None
        For 'softHRD_v2': a fitted FeatureEngine for transforming raw
        expression into model features. For gene-set methods: ignored.
    centroid_df : pd.DataFrame or None
        For 'softHRD_v1': centroid DataFrame (genes x conditions).
    gene : str
        For 'single_gene' method: which gene to use (default 'POLQ').
    """

    SUPPORTED_METHODS = {
        "softHRD_v2", "softHRD_v1", "custom_model",
        "severson", "parpi7", "cin70", "peng", "polq", "single_gene",
    }

    def __init__(self, method="softHRD_v2", model=None, feature_engine=None,
                 centroid_df=None, gene="POLQ"):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Supported: {self.SUPPORTED_METHODS}"
            )
        self.method = method
        self.model = model
        self.feature_engine = feature_engine
        self.centroid_df = centroid_df
        self.gene = gene

    def score(self, expression_df: pd.DataFrame) -> pd.Series:
        """Return continuous HRD score for each sample.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix. Accepted layouts:
              - samples x genes  (index = sample IDs, columns = genes)
              - genes x samples  (auto-detected if nrows >> ncols)

        Returns
        -------
        pd.Series
            Continuous HRD score indexed by sample ID.
        """
        expr = self._orient_samples_as_rows(expression_df)
        dispatcher = {
            "softHRD_v2": self._score_v2,
            "softHRD_v1": self._score_v1_centroid,
            "custom_model": self._score_custom_model,
            "severson": self._score_severson,
            "parpi7": self._score_parpi7,
            "cin70": self._score_cin70,
            "peng": self._score_peng,
            "polq": self._score_single_gene,
            "single_gene": self._score_single_gene,
        }
        return dispatcher[self.method](expr)

    def classify(self, expression_df: pd.DataFrame, threshold=0.5) -> pd.Series:
        """Return binary HRD/HRP classification.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix (see `score` for layout).
        threshold : float
            Score threshold above which a sample is classified HRD.

        Returns
        -------
        pd.Series
            Binary labels: 1 = HRD, 0 = HRP.
        """
        scores = self.score(expression_df)
        return (scores >= threshold).astype(int).rename("HRD_class")

    # ----- Model-based scoring ------------------------------------------------

    def _score_v2(self, expr):
        """softHRD v2: FeatureEngine → trained model → P(HRD)."""
        if self.model is None:
            raise RuntimeError("softHRD_v2 requires a fitted model (pass model=...)")
        if self.feature_engine is None:
            raise RuntimeError("softHRD_v2 requires a fitted FeatureEngine")

        features = self.feature_engine.transform(expr)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)
            # Handle multi-class or binary
            if probs.ndim == 2:
                score = probs[:, 1] if probs.shape[1] == 2 else probs.max(axis=1)
            else:
                score = probs
        elif hasattr(self.model, "decision_function"):
            score = self.model.decision_function(features)
        else:
            raise RuntimeError("Model must have predict_proba() or decision_function()")

        return pd.Series(score, index=features.index, name="HRD_score")

    def _score_custom_model(self, expr):
        """Score with any sklearn-compatible model (no feature engine)."""
        if self.model is None:
            raise RuntimeError("custom_model requires a fitted model (pass model=...)")
        if self.feature_engine is not None:
            expr = self.feature_engine.transform(expr)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(expr)
            score = probs[:, 1] if probs.ndim == 2 and probs.shape[1] == 2 else probs.ravel()
        elif hasattr(self.model, "decision_function"):
            score = self.model.decision_function(expr)
        else:
            score = self.model.predict(expr)
        return pd.Series(np.asarray(score).ravel(), index=expr.index, name="HRD_score")

    # ----- Centroid-based scoring (v1) ----------------------------------------

    def _score_v1_centroid(self, expr):
        """softHRD v1: Pearson correlation to HRD/HRP centroids."""
        if self.centroid_df is None:
            raise RuntimeError(
                "softHRD_v1 requires centroid_df (pass centroid_df=...)"
            )
        scorer = CentroidScorer(self.centroid_df)
        corr_df = scorer.score(expr)

        if "HRD" in corr_df.columns and "HR_proficient" in corr_df.columns:
            score = corr_df["HRD"] - corr_df["HR_proficient"]
        elif "HRD" in corr_df.columns:
            score = corr_df["HRD"]
        else:
            score = corr_df.iloc[:, 0]

        return score.rename("HRD_score")

    # ----- Gene-set-based scoring ---------------------------------------------

    def _score_severson(self, expr):
        """Severson et al. 14-gene signature: mean(up) - mean(down)."""
        return self._mean_diff_score(
            expr, SEVERSON_GENES["up_in_HRD"], SEVERSON_GENES["down_in_HRD"],
            name="severson_score",
        )

    def _score_parpi7(self, expr):
        """PARPi7 7-gene signature: mean(up_sensitive) - mean(down_sensitive)."""
        return self._mean_diff_score(
            expr,
            PARPI7_GENES["up_in_sensitive"],
            PARPI7_GENES["down_in_sensitive"],
            name="parpi7_score",
        )

    def _score_cin70(self, expr):
        """CIN70: mean z-scored expression of 70 CIN genes."""
        avail = [g for g in CIN70_GENES if g in expr.columns]
        if not avail:
            raise ValueError("No CIN70 genes found in expression data")
        logger.info("CIN70: using %d / %d genes", len(avail), len(CIN70_GENES))
        subset = expr[avail]
        z = (subset - subset.mean()) / subset.std().replace(0, 1)
        return z.mean(axis=1).rename("cin70_score")

    def _score_peng(self, expr):
        """Peng et al. DNA repair signature: mean(up) - mean(down)."""
        return self._mean_diff_score(
            expr, PENG_GENES["up_in_HRD"], PENG_GENES["down_in_HRD"],
            name="peng_score",
        )

    def _score_single_gene(self, expr):
        """Single gene marker (default: POLQ)."""
        gene = self.gene
        if gene not in expr.columns:
            raise ValueError(f"Gene '{gene}' not found in expression data")
        vals = expr[gene]
        z = (vals - vals.mean()) / (vals.std() or 1.0)
        return z.rename(f"{gene}_zscore")

    # ----- Helpers ------------------------------------------------------------

    @staticmethod
    def _mean_diff_score(expr, up_genes, down_genes, name="score"):
        """Compute mean(up_genes) - mean(down_genes) per sample."""
        up = [g for g in up_genes if g in expr.columns]
        down = [g for g in down_genes if g in expr.columns]
        if not up and not down:
            raise ValueError(f"No signature genes found in expression data")
        logger.info("%s: %d up / %d down genes available", name, len(up), len(down))

        up_mean = expr[up].mean(axis=1) if up else 0.0
        down_mean = expr[down].mean(axis=1) if down else 0.0
        return (up_mean - down_mean).rename(name)

    @staticmethod
    def _orient_samples_as_rows(df):
        """Ensure expression DataFrame has samples as rows, genes as columns.

        Heuristic: if nrows >> ncols (ratio > 10x), assume genes are rows.
        Typical gene counts: ~20,000; typical sample counts: 50-500.
        Only transposes when the layout is unambiguously genes×samples.
        """
        if df.shape[0] > df.shape[1] * 10 and df.shape[0] > 5000:
            logger.info(
                "Expression looks like genes×samples (%d×%d), transposing",
                df.shape[0], df.shape[1],
            )
            return df.T
        return df


class CentroidScorer:
    """Centroid-correlation scoring from Jacobson et al. / softHRD v1.

    Computes per-sample Pearson correlation to each class centroid, producing
    a correlation profile that separates HRD subtypes.

    Parameters
    ----------
    centroid_df : pd.DataFrame
        Centroid expression matrix: genes (index) x conditions (columns).
        Typical columns: 'HRD', 'HR_proficient', 'BRCA1', 'BRCA2',
        'HRD_BRCApos', 'HR_BRCA_proficient'.
    """

    def __init__(self, centroid_df: pd.DataFrame):
        if centroid_df is None or centroid_df.empty:
            raise ValueError("centroid_df must be a non-empty DataFrame")
        self.centroid_df = centroid_df

    def score(self, expression_df: pd.DataFrame) -> pd.DataFrame:
        """Compute Pearson correlation of each sample to each centroid.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix (samples x genes) or (genes x samples).

        Returns
        -------
        pd.DataFrame
            Correlation matrix: samples x centroid conditions.
        """
        expr = self._orient(expression_df)

        # Intersect genes
        common_genes = expr.columns.intersection(self.centroid_df.index)
        if len(common_genes) < 10:
            raise ValueError(
                f"Only {len(common_genes)} genes overlap between expression "
                f"data and centroid. Need at least 10."
            )
        logger.info(
            "CentroidScorer: %d / %d centroid genes found in data",
            len(common_genes), len(self.centroid_df),
        )

        centroid = self.centroid_df.loc[common_genes]
        expr_sub = expr[common_genes]

        results = {}
        for condition in centroid.columns:
            ref = centroid[condition].values
            corrs = expr_sub.apply(
                lambda row: stats.pearsonr(row.values, ref)[0], axis=1
            )
            results[condition] = corrs

        return pd.DataFrame(results, index=expr_sub.index)

    def hrd_score(self, expression_df: pd.DataFrame) -> pd.Series:
        """Convenience: return corr(HRD) - corr(HR_proficient)."""
        corr_df = self.score(expression_df)
        hrd_col = next(
            (c for c in corr_df.columns if "HRD" in c and "proficient" not in c.lower()),
            corr_df.columns[0],
        )
        hrp_col = next(
            (c for c in corr_df.columns if "proficient" in c.lower()),
            corr_df.columns[1] if len(corr_df.columns) > 1 else None,
        )
        if hrp_col:
            return (corr_df[hrd_col] - corr_df[hrp_col]).rename("HRD_score")
        return corr_df[hrd_col].rename("HRD_score")

    @staticmethod
    def _orient(df):
        """Ensure samples as rows."""
        if df.shape[0] > df.shape[1] * 10 and df.shape[0] > 5000:
            return df.T
        return df
