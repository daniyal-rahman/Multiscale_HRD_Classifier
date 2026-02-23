#!/usr/bin/env python3
"""
Tiered gold-standard labeling system for HRD status.

Uses multi-modal concordance to produce high-confidence training labels:
  Tier 1 (HRD-positive):  biallelic HRR loss + high genomic scar + optional SBS3
  Tier 2 (HRD-negative):  no HRR events + low genomic scar
  Tier 3 (ambiguous):     excluded from training (monoallelic, intermediate scar, etc.)

Designed to consume TCGA-BRCA data formats from:
  - Knijnenburg et al. 2018 BRCA status annotations (toga.breast.brca.status.txt)
  - Marquard et al. HRD scores (tcga.hrdscore.xlsx)

Can also work with v2 data-acquisition outputs (parquet files).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# -- Full HRR gene panel (for negative screening) ---------------------------
HRR_GENES = [
    "BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D",
    "ATM", "ATR", "CHEK1", "CHEK2", "BARD1", "BRIP1",
    "FANCA", "FANCC", "NBN", "MRE11", "RAD50", "RAD51B",
]

# Genes where biallelic loss is strong evidence for HRD
BIALLELIC_HRD_GENES = ["BRCA1", "BRCA2", "RAD51C", "RAD51D", "PALB2"]

# CDK12 — tandem-duplication phenotype, distinct from HRD
CDK12_GENE = "CDK12"


@dataclass
class SampleLabel:
    """Result of tiered labeling for a single sample."""
    sample_id: str
    tier: int                    # 1, 2, or 3
    label: str                   # "HRD", "HRP", or "ambiguous"
    confidence: float            # 0.0–1.0
    reasons: list[str] = field(default_factory=list)


class TieredHRDLabeler:
    """Multi-modal concordance labeler for HRD status.

    Parameters
    ----------
    gis_positive_threshold : int
        HRD-sum (LOH + TAI + LST) above which genomic scar is "high".
    gis_negative_threshold : int
        HRD-sum below which genomic scar is "low".
    gloh_threshold : float
        Fraction genome LOH above which gLOH is "high" (0–1 scale).
    sbs3_threshold : float
        SBS3 signature weight above which mutational-signature evidence is "high".
    brca1_methylation_beta_threshold : float
        Mean BRCA1-promoter beta above which methylation is considered positive.
    """

    def __init__(
        self,
        gis_positive_threshold: int = 42,
        gis_negative_threshold: int = 20,
        gloh_threshold: float = 0.16,
        sbs3_threshold: float = 0.06,
        brca1_methylation_beta_threshold: float = 0.3,
    ):
        self.gis_pos = gis_positive_threshold
        self.gis_neg = gis_negative_threshold
        self.gloh_thr = gloh_threshold
        self.sbs3_thr = sbs3_threshold
        self.meth_thr = brca1_methylation_beta_threshold

    # --------------------------------------------------------------------- #
    #  Core single-sample labeler                                            #
    # --------------------------------------------------------------------- #

    def label_sample(
        self,
        mutation_data: dict,
        cnv_data: dict,
        methylation_data: Optional[dict] = None,
        sbs3_score: Optional[float] = None,
        clinical_data: Optional[dict] = None,
    ) -> dict:
        """Label a single sample.

        Parameters
        ----------
        mutation_data : dict
            Keys expected:
              biallelic_genes   – set/list of genes with biallelic loss
              monoallelic_genes – set/list of genes with monoallelic hit
              has_reversion     – bool
              atm_only          – bool (True if ATM is the only HRR hit)
              has_cdk12         – bool
              any_hrr_mutation  – bool
        cnv_data : dict
            Keys expected:
              hrd_sum  – int  (LOH + TAI + LST)
              gloh     – float (fraction, 0–1) or None
        methylation_data : dict | None
            Keys expected:
              brca1_methylated      – bool
              brca1_methylation_loh – bool (LOH at BRCA1 locus)
        sbs3_score : float | None
            SBS3 signature weight (0–1).
        clinical_data : dict | None
            Keys expected:
              prior_platinum – bool
              prior_parpi    – bool

        Returns
        -------
        dict with keys: tier, label, confidence, reasons
        """
        reasons: list[str] = []

        biallelic = set(mutation_data.get("biallelic_genes", []))
        monoallelic = set(mutation_data.get("monoallelic_genes", []))
        has_reversion = mutation_data.get("has_reversion", False)
        atm_only = mutation_data.get("atm_only", False)
        has_cdk12 = mutation_data.get("has_cdk12", False)
        any_hrr = mutation_data.get("any_hrr_mutation", False)

        hrd_sum = cnv_data.get("hrd_sum", None)
        gloh = cnv_data.get("gloh", None)

        brca1_meth = False
        brca1_meth_loh = False
        if methylation_data:
            brca1_meth = methylation_data.get("brca1_methylated", False)
            brca1_meth_loh = methylation_data.get("brca1_methylation_loh", False)

        prior_treatment = False
        if clinical_data:
            prior_treatment = (
                clinical_data.get("prior_platinum", False)
                or clinical_data.get("prior_parpi", False)
            )

        # ---- Exclusion checks (Tier 3) FIRST ----

        if has_reversion:
            reasons.append("reversion mutation detected")
            return _result(3, "ambiguous", 0.0, reasons)

        if has_cdk12:
            reasons.append("CDK12 alteration (tandem-dup phenotype, not HRD)")
            return _result(3, "ambiguous", 0.0, reasons)

        if prior_treatment:
            reasons.append("prior platinum/PARPi treatment")
            return _result(3, "ambiguous", 0.0, reasons)

        if atm_only:
            reasons.append("ATM-only mutation (contested HRD driver)")
            return _result(3, "ambiguous", 0.0, reasons)

        # ---- Attempt Tier 1 (gold-standard HRD-positive) ----
        biallelic_hrd = biallelic & set(BIALLELIC_HRD_GENES)

        # BRCA1 methylation can count as biallelic if confirmed by LOH
        brca1_meth_biallelic = False
        if brca1_meth and brca1_meth_loh:
            brca1_meth_biallelic = True
            biallelic_hrd.add("BRCA1")
        elif brca1_meth and not brca1_meth_loh:
            reasons.append("BRCA1 methylation without LOH confirmation")
            return _result(3, "ambiguous", 0.0, reasons)

        high_scar = False
        if hrd_sum is not None and hrd_sum >= self.gis_pos:
            high_scar = True
        if gloh is not None and gloh >= self.gloh_thr:
            high_scar = True

        sbs3_concordant = True  # default: not contradictory
        sbs3_supportive = False
        if sbs3_score is not None:
            if sbs3_score >= self.sbs3_thr:
                sbs3_supportive = True
            else:
                sbs3_concordant = False

        if biallelic_hrd and high_scar and sbs3_concordant:
            genes_str = ", ".join(sorted(biallelic_hrd))
            reasons.append(f"biallelic loss in {genes_str}")
            if brca1_meth_biallelic:
                reasons.append("BRCA1 methylation confirmed by LOH")
            reasons.append(f"high genomic scar (HRD-sum={hrd_sum}, gLOH={gloh})")
            if sbs3_supportive:
                reasons.append(f"high SBS3 ({sbs3_score:.3f})")

            confidence = 1.0
            if not sbs3_supportive and sbs3_score is not None:
                # SBS3 available but low — slight uncertainty
                confidence = 0.85
            return _result(1, "HRD", confidence, reasons)

        # ---- Attempt Tier 2 (gold-standard HRP) ----
        low_scar = False
        if hrd_sum is not None and hrd_sum < self.gis_neg:
            low_scar = True

        no_hrr = not any_hrr and not biallelic_hrd and not monoallelic
        no_meth = not brca1_meth

        sbs3_low = True
        if sbs3_score is not None and sbs3_score >= self.sbs3_thr:
            sbs3_low = False

        if no_hrr and no_meth and low_scar and sbs3_low:
            reasons.append("no HRR gene mutations")
            reasons.append("no BRCA1 methylation")
            reasons.append(f"low genomic scar (HRD-sum={hrd_sum})")
            if sbs3_score is not None:
                reasons.append(f"low SBS3 ({sbs3_score:.3f})")
            return _result(2, "HRP", 1.0, reasons)

        # ---- Tier 3 (ambiguous) ----
        if monoallelic and not biallelic_hrd:
            mono_str = ", ".join(sorted(monoallelic))
            reasons.append(f"monoallelic hit(s) only: {mono_str}")

        if hrd_sum is not None and self.gis_neg <= hrd_sum < self.gis_pos:
            reasons.append(
                f"intermediate genomic scar (HRD-sum={hrd_sum}, "
                f"range {self.gis_neg}–{self.gis_pos})"
            )

        if biallelic_hrd and not high_scar:
            reasons.append("biallelic loss but low/intermediate scar")

        if not biallelic_hrd and high_scar:
            reasons.append("high scar but no biallelic HRR loss identified")

        if any_hrr and not biallelic_hrd and not monoallelic:
            reasons.append("HRR pathway mutation without clear allele status")

        if not reasons:
            reasons.append("does not meet Tier-1 or Tier-2 criteria")

        return _result(3, "ambiguous", 0.0, reasons)

    # --------------------------------------------------------------------- #
    #  TCGA-BRCA cohort labeler (legacy format)                              #
    # --------------------------------------------------------------------- #

    def label_tcga_cohort(
        self,
        brca_status_df: pd.DataFrame,
        hrd_scores_df: pd.DataFrame,
        methylation_df: Optional[pd.DataFrame] = None,
        sbs3_df: Optional[pd.DataFrame] = None,
        clinical_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Label an entire TCGA-BRCA cohort using the legacy data files.

        Parameters
        ----------
        brca_status_df : DataFrame
            Index = sample ID.  Columns include `event.BRCA1`, `event.BRCA2`,
            `event.RAD51C`, `event.PALB2`, plus detailed alteration columns
            (BRCA1_somatic_null, BRCA1_germ_bi_allelic, etc.).
        hrd_scores_df : DataFrame
            Must contain columns: ``sample``, ``HRD-sum``.
            Optional: ``HRD``, ``Telomeric AI``, ``LST``.
        methylation_df : DataFrame | None
            Index = sample, column ``brca1_methylation_beta`` (mean beta).
        sbs3_df : DataFrame | None
            Index = sample, column ``SBS3``.
        clinical_df : DataFrame | None
            Index = sample, column ``prior_platinum``, ``prior_parpi`` (bool).

        Returns
        -------
        DataFrame indexed by sample with columns:
          tier, label, confidence, reasons, HRD-sum, event.BRCA1, …
        """
        # Merge BRCA status and HRD scores on sample ID
        hrd = hrd_scores_df.copy()
        if "sample" in hrd.columns:
            hrd = hrd.set_index("sample")

        brca = brca_status_df.copy()
        # Ensure indices match format (dots → hyphens)
        brca.index = brca.index.astype(str).str.replace(".", "-", regex=False)
        hrd.index = hrd.index.astype(str).str.replace(".", "-", regex=False)

        common = brca.index.intersection(hrd.index)
        brca = brca.loc[common]
        hrd = hrd.loc[common]

        rows = []
        for sample_id in common:
            b = brca.loc[sample_id]
            h = hrd.loc[sample_id]

            mutation_data = self._parse_tcga_mutations(b)
            cnv_data = {"hrd_sum": _safe_int(h.get("HRD-sum")), "gloh": None}
            methylation_data = self._parse_tcga_methylation(b, sample_id, methylation_df)

            sbs3 = None
            if sbs3_df is not None and sample_id in sbs3_df.index:
                sbs3 = float(sbs3_df.loc[sample_id, "SBS3"])

            clinical = None
            if clinical_df is not None and sample_id in clinical_df.index:
                clinical = {
                    "prior_platinum": bool(clinical_df.loc[sample_id].get("prior_platinum", False)),
                    "prior_parpi": bool(clinical_df.loc[sample_id].get("prior_parpi", False)),
                }

            result = self.label_sample(mutation_data, cnv_data, methylation_data, sbs3, clinical)
            result["sample"] = sample_id
            result["HRD-sum"] = cnv_data["hrd_sum"]
            result["event.BRCA1"] = str(b.get("event.BRCA1", ""))
            result["event.BRCA2"] = str(b.get("event.BRCA2", ""))
            result["event.RAD51C"] = str(b.get("event.RAD51C", ""))
            result["event.PALB2"] = str(b.get("event.PALB2", ""))
            rows.append(result)

        df = pd.DataFrame(rows).set_index("sample")
        df["reasons"] = df["reasons"].apply(lambda x: "; ".join(x))
        return df

    # --------------------------------------------------------------------- #
    #  Comparison with existing labels                                       #
    # --------------------------------------------------------------------- #

    @staticmethod
    def compare_with_existing_labels(
        tiered_labels: pd.DataFrame,
        existing_labels: pd.Series | pd.DataFrame,
        existing_col: str = "HRD_status_base",
    ) -> dict:
        """Compare tiered labels against a simple binary threshold.

        Parameters
        ----------
        tiered_labels : DataFrame
            Output of ``label_tcga_cohort`` (must have 'tier' and 'label' columns).
        existing_labels : Series or DataFrame
            Binary labels. If DataFrame, ``existing_col`` is used.

        Returns
        -------
        dict with keys:
          tier_counts, concordance, discordance, reclassified samples
        """
        if isinstance(existing_labels, pd.DataFrame):
            existing = existing_labels[existing_col]
        else:
            existing = existing_labels

        # Align
        common = tiered_labels.index.intersection(existing.index)
        t = tiered_labels.loc[common]
        e = existing.loc[common]

        # Map existing to HRD/HRP
        existing_binary = e.map(lambda x: "HRD" if str(x).upper() in ("HRD", "1") else "HRP")

        tier_counts = t["tier"].value_counts().to_dict()

        # Only compare Tier 1 and 2 (Tier 3 has no corresponding label)
        tier12 = t[t["tier"].isin([1, 2])]
        e12 = existing_binary.loc[tier12.index]

        concordant = (tier12["label"] == e12).sum()
        discordant = (tier12["label"] != e12).sum()
        total12 = len(tier12)

        reclassified = tier12[tier12["label"] != e12].copy()
        reclassified["existing_label"] = e12.loc[reclassified.index]

        # Samples moved to Tier 3 that had a definite existing label
        moved_ambiguous = t[t["tier"] == 3].index.intersection(common)
        moved_ambiguous_df = t.loc[moved_ambiguous].copy()
        moved_ambiguous_df["existing_label"] = existing_binary.loc[moved_ambiguous]

        return {
            "tier_counts": tier_counts,
            "total_samples": len(common),
            "tier12_samples": total12,
            "concordant": int(concordant),
            "discordant": int(discordant),
            "concordance_rate": concordant / total12 if total12 else 0.0,
            "reclassified": reclassified,
            "moved_to_ambiguous": moved_ambiguous_df,
        }

    # --------------------------------------------------------------------- #
    #  Internal helpers for TCGA format                                      #
    # --------------------------------------------------------------------- #

    def _parse_tcga_mutations(self, row: pd.Series) -> dict:
        """Parse a single row of the Knijnenburg-style BRCA status table."""
        biallelic = set()
        monoallelic = set()
        any_hrr = False

        # BRCA1
        event_brca1 = str(row.get("event.BRCA1", "0"))
        if event_brca1 == "Bi-allelic-inactivation":
            biallelic.add("BRCA1")
            any_hrr = True
        elif event_brca1 == "1":
            monoallelic.add("BRCA1")
            any_hrr = True
        else:
            # Check detailed columns
            if _truthy(row.get("BRCA1_germ_bi_allelic")):
                biallelic.add("BRCA1")
                any_hrr = True
            elif _truthy(row.get("BRCA1_somatic_null")) and _truthy(row.get("BRCA1_deletion")):
                biallelic.add("BRCA1")
                any_hrr = True
            elif _truthy(row.get("BRCA1_germ_mono_allelic")):
                monoallelic.add("BRCA1")
                any_hrr = True
            elif _truthy(row.get("BRCA1_somatic_null")):
                monoallelic.add("BRCA1")
                any_hrr = True

        # BRCA2
        event_brca2 = str(row.get("event.BRCA2", "0"))
        if event_brca2 == "Bi-allelic-inactivation":
            biallelic.add("BRCA2")
            any_hrr = True
        elif event_brca2 in ("Bi-allelic-undetermined", "1"):
            # Undetermined biallelic — not confident enough for Tier 1
            monoallelic.add("BRCA2")
            any_hrr = True
        else:
            if _truthy(row.get("BRCA2_germ_bi_allelic")):
                biallelic.add("BRCA2")
                any_hrr = True
            elif _truthy(row.get("BRCA2_germ_mono_allelic")):
                monoallelic.add("BRCA2")
                any_hrr = True
            elif _truthy(row.get("BRCA2_somatic_null")):
                monoallelic.add("BRCA2")
                any_hrr = True
            elif _truthy(row.get("BRCA2_germ_undetermined")):
                monoallelic.add("BRCA2")
                any_hrr = True

        # RAD51C
        event_rad51c = str(row.get("event.RAD51C", "0"))
        if event_rad51c == "Bi-allelic-inactivation":
            biallelic.add("RAD51C")
            any_hrr = True
        elif event_rad51c not in ("0", "nan", ""):
            monoallelic.add("RAD51C")
            any_hrr = True
        else:
            if _truthy(row.get("RAD51C_germ")) and _truthy(row.get("RAD51C_deletion")):
                biallelic.add("RAD51C")
                any_hrr = True
            elif _truthy(row.get("RAD51C_germ")):
                monoallelic.add("RAD51C")
                any_hrr = True

        # PALB2
        event_palb2 = str(row.get("event.PALB2", "0"))
        if event_palb2 == "Bi-allelic-inactivation":
            biallelic.add("PALB2")
            any_hrr = True
        elif event_palb2 not in ("0", "nan", ""):
            monoallelic.add("PALB2")
            any_hrr = True
        else:
            if _truthy(row.get("PALB2_germ")) and _truthy(row.get("PALB2_somatic_null")):
                biallelic.add("PALB2")
                any_hrr = True
            elif _truthy(row.get("PALB2_germ")) or _truthy(row.get("PALB2_somatic_null")):
                monoallelic.add("PALB2")
                any_hrr = True

        # Epigenetic silencing of BRCA1 → handled via methylation_data
        # (not counted as a mutation here to avoid double-counting)

        # ATM-only flag: ATM mutated and nothing else
        # (not available in legacy TCGA BRCA status file — only BRCA1/2/RAD51C/PALB2)
        atm_only = False

        return {
            "biallelic_genes": biallelic,
            "monoallelic_genes": monoallelic,
            "has_reversion": False,   # not annotated in legacy TCGA data
            "atm_only": atm_only,
            "has_cdk12": False,       # not annotated in legacy TCGA data
            "any_hrr_mutation": any_hrr,
        }

    def _parse_tcga_methylation(
        self,
        row: pd.Series,
        sample_id: str,
        methylation_df: Optional[pd.DataFrame],
    ) -> Optional[dict]:
        """Determine BRCA1 methylation status from TCGA data."""
        epigenetic = _truthy(row.get("BRCA1_epigenetic_silencing"))
        deletion = _truthy(row.get("BRCA1_deletion"))

        if methylation_df is not None and sample_id in methylation_df.index:
            beta = float(methylation_df.loc[sample_id, "brca1_methylation_beta"])
            methylated = beta >= self.meth_thr
            return {
                "brca1_methylated": methylated,
                "brca1_methylation_loh": deletion,
            }

        if epigenetic:
            return {
                "brca1_methylated": True,
                "brca1_methylation_loh": deletion,
            }

        return None


# --------------------------------------------------------------------------- #
#  Module-level helpers                                                       #
# --------------------------------------------------------------------------- #

def _truthy(val) -> bool:
    """Check if a value from a messy dataframe is truthy (non-zero, non-NaN)."""
    if val is None:
        return False
    if isinstance(val, float) and np.isnan(val):
        return False
    try:
        return bool(int(float(val)))
    except (ValueError, TypeError):
        # Probably a string like 'NaN' or ''
        return False


def _safe_int(val) -> Optional[int]:
    """Convert to int, returning None on failure."""
    if val is None:
        return None
    try:
        v = float(val)
        if np.isnan(v):
            return None
        return int(v)
    except (ValueError, TypeError):
        return None


def _result(tier: int, label: str, confidence: float, reasons: list[str]) -> dict:
    return {
        "tier": tier,
        "label": label,
        "confidence": confidence,
        "reasons": reasons,
    }
