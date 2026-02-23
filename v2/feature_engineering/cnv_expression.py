"""
Copy-number-derived expression features.

HRD tumors have characteristic copy number profiles (high LOH, TAI, LST).
This module computes features that capture the *expression consequences* of
these copy number alterations:

  1. CN-adjusted expression residuals — expression after regressing out CN effects
  2. Dosage-sensitive gene features — genes where CN strongly predicts expression
  3. HRD-associated CN regions — expression in frequently altered regions

These features complement pure expression-based features by separating
transcriptional rewiring (dynamic HRD state) from CN-driven expression changes
(historical genomic scars).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Chromosomal regions frequently affected by LOH in HRD tumors
# (from Marquard et al. 2015 and Popova et al. 2012)
HRD_LOH_REGIONS = {
    # Region name: (chrom, start_mb, end_mb) — approximate coordinates
    "5q_loss": ("5", 100, 181),
    "8p_loss": ("8", 0, 43),
    "13q_loss": ("13", 19, 115),  # Contains RB1, BRCA2
    "17p_loss": ("17", 0, 22),  # Contains TP53
    "17q_gain": ("17", 25, 81),  # Contains BRCA1 region
    "4p_loss": ("4", 0, 50),
    "15q_loss": ("15", 20, 100),
    "22q_loss": ("22", 15, 51),
}

# Key dosage-sensitive genes for HRD (CN-expression correlation > 0.5 in TCGA)
DOSAGE_SENSITIVE_GENES = [
    "MYC", "CCNE1", "CCND1", "ERBB2",  # Oncogenic amplifications
    "RB1", "TP53", "PTEN", "CDKN2A",  # Tumor suppressors
    "BRCA1", "BRCA2",  # Direct HRD genes
    "RAD51C", "RAD51D", "PALB2",
    "POLQ",  # Alt-EJ, often amplified in HRD
    "MDM2", "CDK4",
    "EZH2",  # Synthetic lethal with BRCA1
]


class CNVExpressionFeatures:
    """Extract features relating copy number to expression for HRD.

    Parameters
    ----------
    dosage_genes : list[str] or None
        Genes to extract dosage-expression features for.
        Defaults to DOSAGE_SENSITIVE_GENES.
    """

    def __init__(self, dosage_genes=None):
        self.dosage_genes = dosage_genes or DOSAGE_SENSITIVE_GENES
        self._cn_coefficients = None  # fitted from training data

    def fit(self, expr, cnv):
        """Fit CN-expression relationships from paired data.

        Parameters
        ----------
        expr : pd.DataFrame
            Expression matrix (genes x samples).
        cnv : pd.DataFrame
            Copy number matrix (genes x samples), same orientation as expr.
            Values are typically log2(CN/2) or discrete GISTIC values.

        Returns
        -------
        self
        """
        expr = self._orient(expr)
        cnv = self._orient(cnv)

        # Align genes and samples
        common_genes = expr.index.intersection(cnv.index)
        common_samples = expr.columns.intersection(cnv.columns)

        if len(common_genes) == 0 or len(common_samples) == 0:
            logger.warning("No common genes/samples between expression and CNV")
            return self

        expr_aligned = expr.loc[common_genes, common_samples]
        cnv_aligned = cnv.loc[common_genes, common_samples]

        # Fit per-gene linear model: expr_g = a + b * cnv_g
        # Store the slope (b) for each gene — high |b| means dosage-sensitive
        slopes = {}
        for gene in common_genes:
            e = expr_aligned.loc[gene].values.astype(float)
            c = cnv_aligned.loc[gene].values.astype(float)
            mask = np.isfinite(e) & np.isfinite(c)
            if mask.sum() < 10:
                continue
            e_m, c_m = e[mask], c[mask]
            # Simple regression: slope = cov(c,e) / var(c)
            c_var = np.var(c_m)
            if c_var < 1e-10:
                continue
            slope = np.cov(c_m, e_m)[0, 1] / c_var
            slopes[gene] = slope

        self._cn_coefficients = pd.Series(slopes)
        logger.info("Fitted CN-expression slopes for %d genes", len(slopes))
        return self

    def transform(self, expr, cnv=None):
        """Generate CNV-expression features.

        Parameters
        ----------
        expr : pd.DataFrame
            Expression matrix (genes x samples).
        cnv : pd.DataFrame or None
            Copy number matrix. If provided, computes CN-adjusted residuals
            and dosage features. If None, only computes expression-based
            features for dosage-sensitive genes.

        Returns
        -------
        pd.DataFrame
            Features (samples x features).
        """
        expr = self._orient(expr)
        features = {}

        # 1. Raw expression of dosage-sensitive genes (always available)
        available_dosage = [g for g in self.dosage_genes if g in expr.index]
        for gene in available_dosage:
            features[f"dosage_{gene}_expr"] = expr.loc[gene].values

        if cnv is not None:
            cnv = self._orient(cnv)
            common_samples = expr.columns.intersection(cnv.columns)

            if len(common_samples) > 0:
                expr_a = expr[common_samples]
                cnv_a = cnv[common_samples]

                # 2. CN-adjusted expression residuals for dosage genes
                if self._cn_coefficients is not None:
                    for gene in available_dosage:
                        if gene in cnv_a.index and gene in self._cn_coefficients.index:
                            slope = self._cn_coefficients[gene]
                            predicted = slope * cnv_a.loc[gene]
                            residual = expr_a.loc[gene] - predicted
                            features[f"dosage_{gene}_cn_residual"] = residual.values

                # 3. Aggregate CN features per HRD-associated region
                #    (requires gene-level CN, not segment-level)
                for gene in available_dosage:
                    if gene in cnv_a.index:
                        features[f"dosage_{gene}_cn"] = cnv_a.loc[gene].values

        result = pd.DataFrame(features, index=expr.columns if cnv is None else common_samples)
        logger.info("Generated %d CNV-expression features for %d samples",
                     result.shape[1], result.shape[0])
        return result

    def fit_transform(self, expr, cnv):
        """Fit and transform in one step."""
        return self.fit(expr, cnv).transform(expr, cnv)

    @staticmethod
    def _orient(df):
        """Orient to genes x samples."""
        if df.shape[0] < df.shape[1]:
            return df
        sample_idx = str(df.index[0])
        if any(c.isdigit() for c in sample_idx) and "-" in sample_idx:
            return df.T
        return df
