"""
Dataset-specific validation classes.

Each validator encapsulates data loading, label extraction, scoring,
and evaluation for a specific external cohort. They share a common
interface so they can be orchestrated by ValidationReport.

Cohorts:
  - TCGA-OV:  Platinum-free interval (PFI) classification
  - I-SPY2:   pCR prediction on durvalumab/olaparib arm
  - GEO ovarian: Platinum response from GSE9891, GSE26712, etc.
  - Cell lines: GDSC/CCLE olaparib/platinum IC50 correlation
  - Pan-cancer: Cross-cancer generalization (prostate, pancreatic)
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import ValidationMetrics

logger = logging.getLogger(__name__)

# Default data root (can be overridden)
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
V2_DATA_ROOT = Path(__file__).resolve().parents[1] / "data_acquisition" / "processed"


class BaseCohortValidator(ABC):
    """Base class for cohort-specific validators.

    Subclasses must implement:
      - load_data() -> (expression_df, labels_df)
      - endpoint_col -> str (column name for primary endpoint)
      - cohort_name -> str
    """

    @property
    @abstractmethod
    def cohort_name(self) -> str:
        ...

    @property
    @abstractmethod
    def endpoint_col(self) -> str:
        ...

    @abstractmethod
    def load_data(self) -> tuple:
        """Return (expression_df, labels_df).

        expression_df: samples × genes
        labels_df: samples × {endpoint_col, ...}
        """
        ...

    def validate(self, scorer, label_col=None, **metric_kwargs) -> dict:
        """Run full validation pipeline with a given scorer.

        Parameters
        ----------
        scorer : SignatureScorer
            Any scorer with a .score(expression_df) method.
        label_col : str or None
            Override default endpoint column.

        Returns
        -------
        dict
            {cohort_name, n_samples, metrics: {...}, scores: pd.Series}
        """
        expr, labels = self.load_data()
        col = label_col or self.endpoint_col

        # Align
        common = expr.index.intersection(labels.index)
        if len(common) == 0:
            raise ValueError(f"{self.cohort_name}: no overlapping samples")
        logger.info("%s: %d samples available", self.cohort_name, len(common))

        expr = expr.loc[common]
        labels = labels.loc[common]
        y_true = labels[col].values

        # Score
        scores = scorer.score(expr)
        scores = scores.loc[common]

        # Metrics
        result = {
            "cohort_name": self.cohort_name,
            "n_samples": len(common),
            "endpoint": col,
        }

        # Binary metrics
        y_true_binary = np.asarray(y_true, dtype=float)
        valid = ~np.isnan(y_true_binary)
        if valid.sum() > 0:
            result["binary_metrics"] = ValidationMetrics.binary_metrics(
                y_true_binary[valid], scores.values[valid]
            )

        # Drug response metrics (if response endpoint)
        if col in ("pCR", "response", "platinum_response", "sensitive"):
            result["drug_response_metrics"] = ValidationMetrics.drug_response_metrics(
                y_true_binary[valid], scores.values[valid]
            )

        result["scores"] = scores
        result["labels"] = labels

        return result


class TCGAOVValidator(BaseCohortValidator):
    """Validate on TCGA-OV using platinum-free interval (PFI).

    Endpoint: PFI <= 6 months = resistant (0), PFI > 6 months = sensitive (1).
    Expression: TCGA-OV RNA-seq (bulk, not deconvoluted).
    """

    cohort_name = "TCGA-OV"
    endpoint_col = "platinum_response"

    def __init__(self, expression_path=None, clinical_path=None,
                 pfi_threshold_months=6):
        self.expression_path = expression_path
        self.clinical_path = clinical_path
        self.pfi_threshold_months = pfi_threshold_months

    def load_data(self):
        """Load TCGA-OV expression and derive platinum labels from PFI."""
        expr_path = self.expression_path or (
            V2_DATA_ROOT / "tcga" / "TCGA-OV_expression.parquet"
        )
        clin_path = self.clinical_path or (
            V2_DATA_ROOT / "tcga" / "TCGA-OV_clinical.parquet"
        )

        expr = _load_expression(expr_path)
        clinical = _load_table(clin_path)

        # Derive platinum response from PFI
        pfi_col = _find_column(clinical, ["PFI.time", "PFI_time", "pfi_time",
                                           "OS.time", "os_time"])
        event_col = _find_column(clinical, ["PFI", "PFI.event", "pfi_event"])

        if pfi_col is None:
            logger.warning("TCGA-OV: PFI column not found, using placeholder labels")
            clinical[self.endpoint_col] = np.nan
        else:
            pfi_months = pd.to_numeric(clinical[pfi_col], errors="coerce") / 30.44
            clinical[self.endpoint_col] = np.where(
                pfi_months > self.pfi_threshold_months, 1, 0
            )
            # Mark NaN where PFI is missing
            clinical.loc[pfi_months.isna(), self.endpoint_col] = np.nan

        if event_col:
            clinical["pfi_event"] = pd.to_numeric(
                clinical[event_col], errors="coerce"
            )
        if pfi_col:
            clinical["pfi_time_months"] = (
                pd.to_numeric(clinical[pfi_col], errors="coerce") / 30.44
            )

        return expr, clinical

    def validate(self, scorer, **kwargs):
        """Override to add survival metrics."""
        result = super().validate(scorer, **kwargs)

        labels = result["labels"]
        scores = result["scores"]
        common = scores.index

        # Survival analysis if PFI available
        if "pfi_time_months" in labels.columns and "pfi_event" in labels.columns:
            valid = labels.loc[common, ["pfi_time_months", "pfi_event"]].notna().all(axis=1)
            if valid.sum() > 10:
                result["survival_metrics"] = ValidationMetrics.survival_metrics(
                    labels.loc[common[valid], "pfi_time_months"].values,
                    labels.loc[common[valid], "pfi_event"].values,
                    scores.loc[common[valid]].values,
                )

        return result


class ISPY2Validator(BaseCohortValidator):
    """Validate on I-SPY2 using pCR as endpoint.

    Focuses on durvalumab/olaparib arm (n~80).
    Expression: Agilent microarray (FFPE, gene-level).
    """

    cohort_name = "I-SPY2"
    endpoint_col = "pCR"

    def __init__(self, expression_path=None, biomarker_path=None, arm=None):
        self.expression_path = expression_path
        self.biomarker_path = biomarker_path
        self.arm = arm or "durvalumab/olaparib"

    def load_data(self):
        """Load I-SPY2 expression and pCR labels."""
        expr_path = self.expression_path or (
            V2_DATA_ROOT / "ispy2" / "expression_matched.parquet"
        )
        bio_path = self.biomarker_path or (
            V2_DATA_ROOT / "ispy2" / "biomarkers_matched.parquet"
        )

        # Fallback: try legacy repo data paths
        if not Path(expr_path).exists():
            legacy = DATA_ROOT / "validation"
            candidates = list(legacy.glob("*ISPY2*Exp*"))
            if candidates:
                expr_path = candidates[0]
                logger.info("I-SPY2: using legacy path %s", expr_path)

        expr = _load_expression(expr_path)
        biomarkers = _load_table(bio_path)

        # Filter arm
        arm_col = _find_column(biomarkers, ["Arm", "arm", "treatment_arm"])
        if arm_col and self.arm:
            biomarkers = biomarkers[
                biomarkers[arm_col].str.contains(self.arm, case=False, na=False)
            ]

        # Normalize pCR column
        pcr_col = _find_column(biomarkers, ["pCR.status", "pCR", "pcr_status",
                                             "pathologic_complete_response"])
        if pcr_col:
            biomarkers[self.endpoint_col] = pd.to_numeric(
                biomarkers[pcr_col], errors="coerce"
            )
            # Recode -1 → 0 (I-SPY2 convention)
            biomarkers.loc[biomarkers[self.endpoint_col] == -1, self.endpoint_col] = 0
        else:
            logger.warning("I-SPY2: pCR column not found")
            biomarkers[self.endpoint_col] = np.nan

        # Keep PARPi7 for correlation analysis
        parpi7_col = _find_column(biomarkers, ["PARPi7_sig.", "PARPi7", "parpi7"])
        if parpi7_col:
            biomarkers["PARPi7_sig"] = pd.to_numeric(
                biomarkers[parpi7_col], errors="coerce"
            )

        return expr, biomarkers

    def validate(self, scorer, **kwargs):
        """Override to add PARPi7 correlation."""
        result = super().validate(scorer, **kwargs)

        labels = result["labels"]
        scores = result["scores"]
        common = scores.index

        if "PARPi7_sig" in labels.columns:
            valid = labels.loc[common, "PARPi7_sig"].notna()
            if valid.sum() > 5:
                r, p = stats.pearsonr(
                    scores.loc[common[valid]].values,
                    labels.loc[common[valid], "PARPi7_sig"].values,
                )
                result["parpi7_correlation"] = {"pearson_r": float(r), "p_value": float(p)}

        return result


class GEOOvarianValidator(BaseCohortValidator):
    """Validate on GEO ovarian cohorts with platinum response labels.

    Supports: GSE9891, GSE26712, GSE51373, GSE63885, and others.
    """

    cohort_name = "GEO-Ovarian"
    endpoint_col = "platinum_response"

    def __init__(self, geo_id="GSE9891", expression_path=None,
                 clinical_path=None):
        self.geo_id = geo_id
        self.expression_path = expression_path
        self.clinical_path = clinical_path

    def load_data(self):
        """Load GEO expression and clinical data."""
        expr_path = self.expression_path or (
            V2_DATA_ROOT / "geo" / self.geo_id / "expression.parquet"
        )
        clin_path = self.clinical_path or (
            V2_DATA_ROOT / "geo" / self.geo_id / "clinical.parquet"
        )

        expr = _load_expression(expr_path)
        clinical = _load_table(clin_path)

        # Look for response column
        resp_col = _find_column(clinical, [
            "platinum_response", "response", "chemo_response",
            "platinum_sensitivity", "sensitive", "resistant",
        ])
        if resp_col:
            vals = clinical[resp_col].str.lower().str.strip()
            clinical[self.endpoint_col] = vals.map(
                lambda x: 1 if x in ("sensitive", "responder", "complete", "1", "yes")
                else (0 if x in ("resistant", "nonresponder", "non-responder",
                                  "refractory", "0", "no")
                      else np.nan)
            )
        else:
            logger.warning("%s: platinum response column not found", self.geo_id)
            clinical[self.endpoint_col] = np.nan

        return expr, clinical


class CellLineValidator(BaseCohortValidator):
    """Validate on GDSC/CCLE using drug IC50 as endpoint.

    Correlates HRD score with olaparib/platinum IC50 values.
    Lower IC50 = more sensitive = expected to be higher HRD score.
    """

    cohort_name = "CellLine-GDSC"
    endpoint_col = "sensitive"

    def __init__(self, expression_path=None, drug_path=None,
                 drug_name="Olaparib", ic50_threshold=None):
        self.expression_path = expression_path
        self.drug_path = drug_path
        self.drug_name = drug_name
        self.ic50_threshold = ic50_threshold

    def load_data(self):
        """Load cell line expression and drug sensitivity data."""
        expr_path = self.expression_path or (
            V2_DATA_ROOT / "gdsc_ccle" / "expression.parquet"
        )
        drug_path = self.drug_path or (
            V2_DATA_ROOT / "gdsc_ccle" / "drug_response.parquet"
        )

        expr = _load_expression(expr_path)
        drug = _load_table(drug_path)

        # Filter to target drug
        drug_col = _find_column(drug, ["drug_name", "Drug", "compound"])
        if drug_col:
            drug = drug[drug[drug_col].str.contains(self.drug_name, case=False, na=False)]

        # Get IC50
        ic50_col = _find_column(drug, ["IC50", "ic50", "LN_IC50", "ln_ic50",
                                        "AUC", "auc"])
        if ic50_col:
            drug["ic50"] = pd.to_numeric(drug[ic50_col], errors="coerce")

            # Binary: sensitive if IC50 below threshold (or median)
            if self.ic50_threshold is not None:
                thresh = self.ic50_threshold
            else:
                thresh = drug["ic50"].median()

            drug[self.endpoint_col] = (drug["ic50"] <= thresh).astype(int)
        else:
            drug[self.endpoint_col] = np.nan

        return expr, drug

    def validate(self, scorer, **kwargs):
        """Override to add IC50 correlation."""
        result = super().validate(scorer, **kwargs)

        labels = result["labels"]
        scores = result["scores"]
        common = scores.index

        if "ic50" in labels.columns:
            valid = labels.loc[common, "ic50"].notna()
            if valid.sum() > 5:
                r, p = stats.spearmanr(
                    scores.loc[common[valid]].values,
                    labels.loc[common[valid], "ic50"].values,
                )
                result["ic50_correlation"] = {
                    "spearman_r": float(r),
                    "p_value": float(p),
                    "drug": self.drug_name,
                    "note": "Negative r expected (higher HRD → lower IC50)",
                }

        return result


class PanCancerValidator(BaseCohortValidator):
    """Test signature generalization across cancer types.

    Uses TCGA pan-cancer RNA-seq with available HRD labels (e.g.
    prostate, pancreatic, ovarian, GI). Labels derived from genomic
    scars or known BRCA/HRR mutations.
    """

    cohort_name = "PanCancer"
    endpoint_col = "HRD_label"

    def __init__(self, cancer_types=None, expression_path=None,
                 labels_path=None):
        self.cancer_types = cancer_types or ["PRAD", "PAAD", "OV", "BRCA"]
        self.expression_path = expression_path
        self.labels_path = labels_path

    def load_data(self):
        """Load pan-cancer expression and HRD labels."""
        expr_path = self.expression_path or (
            V2_DATA_ROOT / "tcga" / "TCGA_pancancer_expression.parquet"
        )
        labels_path = self.labels_path or (
            V2_DATA_ROOT / "tcga" / "TCGA_pancancer_hrd_labels.parquet"
        )

        expr = _load_expression(expr_path)
        labels = _load_table(labels_path)

        # Filter to requested cancer types
        type_col = _find_column(labels, ["cancer_type", "project", "disease",
                                          "tumor_type", "TCGA_project"])
        if type_col:
            labels = labels[labels[type_col].isin(self.cancer_types)]
            labels["cancer_type"] = labels[type_col]

        return expr, labels

    def validate(self, scorer, **kwargs):
        """Override to provide per-cancer-type breakdown."""
        result = super().validate(scorer, **kwargs)

        labels = result["labels"]
        scores = result["scores"]
        common = scores.index

        if "cancer_type" in labels.columns:
            per_type = {}
            for ct in labels.loc[common, "cancer_type"].unique():
                mask = labels.loc[common, "cancer_type"] == ct
                ct_idx = common[mask]
                if len(ct_idx) < 5:
                    continue
                y_true = labels.loc[ct_idx, self.endpoint_col].values.astype(float)
                y_score = scores.loc[ct_idx].values
                valid = ~np.isnan(y_true)
                if valid.sum() >= 5 and len(np.unique(y_true[valid])) > 1:
                    per_type[ct] = ValidationMetrics.binary_metrics(
                        y_true[valid], y_score[valid]
                    )
                    per_type[ct]["n"] = int(valid.sum())

            result["per_cancer_type"] = per_type

        return result


# ======================================================================
# Internal helpers
# ======================================================================

def _load_expression(path):
    """Load expression matrix with fallback to multiple formats."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expression file not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t", index_col=0)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, index_col=0)
    elif path.suffix == ".pkl":
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    # Orient: ensure samples as rows
    if df.shape[0] > df.shape[1] * 10 and df.shape[0] > 5000:
        logger.info("Transposing %s (genes×samples → samples×genes)", path.name)
        df = df.T

    return df


def _load_table(path):
    """Load tabular data with format auto-detection."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t", index_col=0)
    elif path.suffix == ".csv":
        return pd.read_csv(path, index_col=0)
    elif path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, index_col=0)
    elif path.suffix == ".pkl":
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def _find_column(df, candidates):
    """Find the first matching column name from a list of candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    # Case-insensitive fallback
    col_lower = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in col_lower:
            return col_lower[col.lower()]
    return None
