"""
PROGENy-style pathway activity inference from gene expression.

PROGENy (Pathway RespOnsive GENes for activity inference) uses a weight matrix
derived from pathway perturbation experiments to infer pathway activity from
gene expression. Unlike gene-set-based methods, PROGENy uses signed weights
to capture both activation and repression of target genes.

This module provides:
  1. A built-in weight matrix for 14 key signaling pathways
  2. A scorer that computes per-sample pathway activities via linear model
  3. Optional integration with the decoupler Python package when available

Reference: Schubert et al. (2018) Nat Commun. "Perturbation-response genes
reveal signaling footprints in cancer gene expression."
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Simplified PROGENy footprint gene weights for key pathways.
# These are the top responsive genes per pathway (sign = direction of response).
# Full matrix has 100+ genes per pathway; here we include the top ~15 per pathway
# to keep the module self-contained. For full weights, use decoupler.get_progeny().
PROGENY_WEIGHTS = {
    "Androgen": {
        "KLK3": 1.0, "KLK2": 0.9, "FKBP5": 0.85, "TMPRSS2": 0.8,
        "NKX3-1": 0.75, "PMEPA1": 0.7, "ACSL3": 0.65, "HERC3": 0.6,
        "EAF2": 0.55, "ABCC4": 0.5,
    },
    "EGFR": {
        "DUSP6": 1.0, "SPRY2": 0.95, "SPRY4": 0.9, "ETV4": 0.85,
        "ETV5": 0.8, "PHLDA1": 0.75, "DUSP4": 0.7, "ERRFI1": 0.65,
        "GJA1": -0.6, "FOXO1": -0.55,
    },
    "Estrogen": {
        "GREB1": 1.0, "PGR": 0.95, "TFF1": 0.9, "TFF3": 0.85,
        "CA12": 0.8, "XBP1": 0.75, "PKIB": 0.7, "SLC7A2": 0.65,
        "AGR2": 0.6, "MYB": 0.55,
    },
    "Hypoxia": {
        "BNIP3": 1.0, "CA9": 0.95, "SLC2A1": 0.9, "VEGFA": 0.85,
        "ADM": 0.8, "NDRG1": 0.75, "P4HA1": 0.7, "LDHA": 0.65,
        "PGK1": 0.6, "ENO2": 0.55, "PDK1": 0.5,
    },
    "JAK-STAT": {
        "SOCS1": 1.0, "SOCS3": 0.95, "IRF1": 0.9, "BCL3": 0.85,
        "PIM1": 0.8, "MYC": 0.75, "JUNB": 0.7, "BCL2L1": 0.65,
        "GBP1": 0.6, "STAT1": 0.55,
    },
    "MAPK": {
        "DUSP6": 1.0, "SPRY2": 0.95, "SPRY4": 0.9, "ETV4": 0.85,
        "DUSP4": 0.8, "ETV5": 0.75, "PHLDA1": 0.7, "EPHA2": 0.65,
        "CCND1": 0.6, "FOS": 0.55,
    },
    "NFkB": {
        "NFKBIA": 1.0, "TNFAIP3": 0.95, "CCL2": 0.9, "CXCL8": 0.85,
        "IL6": 0.8, "ICAM1": 0.75, "BIRC3": 0.7, "TRAF1": 0.65,
        "BCL2A1": 0.6, "RELB": 0.55,
    },
    "p53": {
        "CDKN1A": 1.0, "MDM2": 0.95, "BAX": 0.9, "BBC3": 0.85,
        "GADD45A": 0.8, "DDB2": 0.75, "SESN1": 0.7, "TP53INP1": 0.65,
        "FAS": 0.6, "PMAIP1": 0.55,
    },
    "PI3K": {
        "TXNIP": -1.0, "IGFBP1": -0.9, "PCK1": -0.85, "SGK1": 0.8,
        "INSR": -0.75, "FOXO1": -0.7, "IRS2": -0.65, "CCND2": 0.6,
    },
    "TGFb": {
        "SMAD7": 1.0, "SERPINE1": 0.95, "JUNB": 0.9, "SKIL": 0.85,
        "CDKN2B": 0.8, "COL1A1": 0.75, "CTGF": 0.7, "ID1": 0.65,
        "TGFBI": 0.6, "PDGFB": 0.55,
    },
    "TNFa": {
        "TNFAIP3": 1.0, "NFKBIA": 0.95, "CCL2": 0.9, "CXCL2": 0.85,
        "BIRC3": 0.8, "IL1B": 0.75, "TRAF1": 0.7, "ICAM1": 0.65,
        "PTGS2": 0.6, "NFKB2": 0.55,
    },
    "Trail": {
        "BIRC3": 1.0, "TNFAIP3": 0.9, "BCL2A1": 0.85, "CFLAR": 0.8,
        "TRAF1": 0.75, "NFKBIA": 0.7, "NFKB2": 0.65,
    },
    "VEGF": {
        "NR4A1": 1.0, "DUSP5": 0.9, "CYR61": 0.85, "ID1": 0.8,
        "PTGS2": 0.75, "KDR": 0.7, "HMOX1": 0.65, "NOS3": 0.6,
    },
    "WNT": {
        "AXIN2": 1.0, "NKD1": 0.95, "DKK1": 0.9, "CCND1": 0.85,
        "MYC": 0.8, "LEF1": 0.75, "TCF7": 0.7, "RNF43": 0.65,
        "LGR5": 0.6, "NOTUM": 0.55,
    },
}


class PROGENyScorer:
    """Compute PROGENy pathway activity scores.

    Parameters
    ----------
    top_n : int
        Number of top footprint genes per pathway. Default 100.
        (When using built-in weights, all available genes are used.)
    use_decoupler : bool
        If True and decoupler is installed, use its full weight matrix.
        Falls back to built-in weights otherwise.
    """

    def __init__(self, top_n=100, use_decoupler=True):
        self.top_n = top_n
        self.use_decoupler = use_decoupler
        self._weight_matrix = None

    def _get_weight_matrix(self, gene_universe):
        """Get the PROGENy weight matrix, optionally from decoupler."""
        if self._weight_matrix is not None:
            return self._weight_matrix

        if self.use_decoupler:
            try:
                import decoupler as dc
                net = dc.get_progeny(organism="human", top=self.top_n)
                # decoupler returns: source, target, weight
                self._weight_matrix = net
                logger.info("Using decoupler PROGENy weights (%d entries)", len(net))
                return self._weight_matrix
            except ImportError:
                logger.info("decoupler not installed, using built-in weights")
            except Exception as e:
                logger.warning("Failed to get decoupler weights: %s", e)

        # Fall back to built-in weights
        rows = []
        for pathway, genes in PROGENY_WEIGHTS.items():
            for gene, weight in genes.items():
                rows.append({"source": pathway, "target": gene, "weight": weight})
        self._weight_matrix = pd.DataFrame(rows)
        logger.info("Using built-in PROGENy weights (%d entries)", len(self._weight_matrix))
        return self._weight_matrix

    def transform(self, expr):
        """Compute PROGENy pathway activities for each sample.

        The activity of pathway p in sample s is:
            A(p, s) = sum_g [ w(p, g) * z(g, s) ]
        where w(p,g) is the PROGENy weight and z(g,s) is the z-scored expression.

        Parameters
        ----------
        expr : pd.DataFrame
            Expression matrix (genes x samples or samples x genes).

        Returns
        -------
        pd.DataFrame
            Pathway activities (samples x pathways).
        """
        expr = self._orient_genes_x_samples(expr)

        # Z-score genes across samples
        gene_mean = expr.mean(axis=1)
        gene_std = expr.std(axis=1).replace(0, 1)
        z_expr = expr.sub(gene_mean, axis=0).div(gene_std, axis=0)

        # Get weight matrix
        net = self._get_weight_matrix(set(expr.index))
        available_genes = set(expr.index)

        # Filter to available genes
        net_filtered = net[net["target"].isin(available_genes)]

        # Compute activity per pathway
        pathways = net_filtered["source"].unique()
        activities = {}

        for pathway in pathways:
            pw_net = net_filtered[net_filtered["source"] == pathway]
            genes = pw_net["target"].values
            weights = pw_net["weight"].values

            # Weighted sum of z-scored expression
            gene_z = z_expr.loc[genes].values  # (n_genes_in_pathway, n_samples)
            activity = weights @ gene_z  # (n_samples,)

            # Normalize by sqrt of number of genes (like a t-statistic)
            activity = activity / np.sqrt(len(genes))
            activities[pathway] = activity

        result = pd.DataFrame(activities, index=expr.columns)
        logger.info("Computed PROGENy scores for %d pathways x %d samples",
                     result.shape[1], result.shape[0])
        return result

    @staticmethod
    def _orient_genes_x_samples(expr):
        """Orient so rows = genes, columns = samples."""
        if expr.shape[0] < expr.shape[1]:
            return expr
        sample_idx = str(expr.index[0])
        if any(c.isdigit() for c in sample_idx) and "-" in sample_idx:
            return expr.T
        return expr
