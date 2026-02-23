#!/usr/bin/env python3
"""
Download GEO ovarian cancer cohorts with platinum response data.

Accessions:
  - GSE9891   (285 ovarian, subtype & platinum response)
  - GSE26712  (185 serous OvCa, chemo response)
  - GSE51088  (172 HGSOC, platinum response)
  - GSE63885  (101 OvCa, platinum response)
  - GSE30161  (58 advanced serous, platinum response)
  - GSE32062  (260 Japanese HGSOC, survival data)

Uses GEOparse for downloading and processing.

Usage:
    srun --mem=8G --time=02:00:00 python download_geo.py
"""

import logging
import re
from pathlib import Path

import GEOparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from _manifest import update_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "geo"
PROCESSED_DIR = BASE_DIR / "processed" / "geo"

# Define datasets with metadata extraction instructions
GEO_DATASETS = {
    "GSE9891": {
        "description": "285 ovarian tumors (Tothill et al. 2008)",
        "n_samples": 285,
        "platform": "GPL570",  # Affymetrix HG-U133 Plus 2.0
        "response_fields": ["characteristics_ch1"],
        "notes": "Subtype classification + chemo response. Look for 'response' in characteristics.",
    },
    "GSE26712": {
        "description": "185 serous OvCa (Bonome et al. 2008)",
        "n_samples": 185,
        "platform": "GPL96",  # Affymetrix HG-U133A
        "response_fields": ["characteristics_ch1"],
        "notes": "Late-stage serous OvCa with survival data.",
    },
    "GSE51088": {
        "description": "172 HGSOC with platinum response (Ferriss et al.)",
        "n_samples": 172,
        "platform": "GPL570",
        "response_fields": ["characteristics_ch1"],
        "notes": "HGSOC samples. Platinum sensitivity in clinical metadata.",
    },
    "GSE63885": {
        "description": "101 OvCa with platinum response (Lisowska et al.)",
        "n_samples": 101,
        "platform": "GPL570",
        "response_fields": ["characteristics_ch1"],
        "notes": "Ovarian cancer cohort from Poland with treatment response.",
    },
    "GSE30161": {
        "description": "58 advanced serous OvCa (Ferriss et al. 2012)",
        "n_samples": 58,
        "platform": "GPL570",
        "response_fields": ["characteristics_ch1"],
        "notes": "Advanced stage serous OvCa with platinum response.",
    },
    "GSE32062": {
        "description": "260 Japanese HGSOC (Yoshihara et al. 2012)",
        "n_samples": 260,
        "platform": "GPL570",
        "response_fields": ["characteristics_ch1"],
        "notes": "Japanese HGSOC cohort with survival data. Paired with GSE32063 (validation).",
    },
}


def _standardize_response(raw_value):
    """Map diverse platinum response annotations to standard labels.

    Returns one of: 'sensitive', 'resistant', 'refractory', 'partial', or None.
    """
    if raw_value is None or pd.isna(raw_value):
        return None

    val = str(raw_value).lower().strip()

    # Sensitive / complete response
    if any(kw in val for kw in [
        "sensitive", "complete response", "complete_response", "cr",
        "platinum_sensitive", "responder",
    ]):
        return "sensitive"

    # Resistant
    if any(kw in val for kw in [
        "resistant", "platinum_resistant", "non-responder", "nonresponder",
        "progressive", "pd",
    ]):
        return "resistant"

    # Refractory (subset of resistant — progressed on treatment)
    if "refractory" in val:
        return "refractory"

    # Partial response
    if any(kw in val for kw in ["partial response", "partial_response", "pr"]):
        return "partial"

    return None


def _extract_clinical_metadata(gsm_dict):
    """Extract all clinical metadata from GEO sample annotations."""
    rows = []
    for gsm_name, gsm in gsm_dict.items():
        row = {"sample_id": gsm_name}

        # Extract title
        row["title"] = gsm.metadata.get("title", [""])[0]

        # Extract all characteristics
        chars = gsm.metadata.get("characteristics_ch1", [])
        for char in chars:
            # GEO characteristics are typically "key: value"
            if ":" in char:
                key, value = char.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                row[key] = value
            else:
                row[f"char_{chars.index(char)}"] = char.strip()

        # Also check description field
        desc = gsm.metadata.get("description", [""])[0]
        if desc:
            row["description"] = desc

        # Source name
        row["source"] = gsm.metadata.get("source_name_ch1", [""])[0]

        rows.append(row)

    return pd.DataFrame(rows)


def _find_response_column(clinical_df):
    """Try to identify which column contains platinum response information."""
    response_keywords = [
        "response", "platinum", "chemo", "sensitive", "resistant",
        "recurrence", "status", "outcome",
    ]

    for col in clinical_df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in response_keywords):
            unique_vals = clinical_df[col].dropna().unique()
            if len(unique_vals) > 0:
                logger.info("  Candidate response column: '%s' with values: %s",
                            col, unique_vals[:10])
                return col

    return None


def download_geo_dataset(accession, info):
    """Download and process a single GEO dataset."""
    logger.info("Downloading %s: %s", accession, info["description"])

    raw_dir = RAW_DIR / accession
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download using GEOparse (caches in raw_dir)
    gse = GEOparse.get_GEO(
        geo=accession,
        destdir=str(raw_dir),
        silent=True,
    )

    # Extract expression matrix
    logger.info("  Extracting expression data for %s", accession)
    expr_frames = {}
    for gsm_name, gsm in tqdm(gse.gsms.items(), desc=f"  {accession} samples"):
        table = gsm.table
        if table is not None and not table.empty:
            if "VALUE" in table.columns and "ID_REF" in table.columns:
                series = table.set_index("ID_REF")["VALUE"]
                series.name = gsm_name
                expr_frames[gsm_name] = series

    if not expr_frames:
        logger.warning("  No expression data found for %s", accession)
        return

    expr_df = pd.DataFrame(expr_frames)

    # Convert to numeric, coerce errors
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce")

    logger.info("  Expression matrix: %d probes x %d samples", *expr_df.shape)

    # Extract clinical metadata
    clinical_df = _extract_clinical_metadata(gse.gsms)
    logger.info("  Clinical metadata: %d samples x %d fields", *clinical_df.shape)

    # Try to map probe IDs to gene symbols using platform annotation
    gene_mapping = _get_probe_to_gene_mapping(gse, info.get("platform"))
    if gene_mapping is not None:
        expr_genes = _map_probes_to_genes(expr_df, gene_mapping)
    else:
        expr_genes = expr_df  # keep probe-level if mapping unavailable

    # Standardize response labels
    response_col = _find_response_column(clinical_df)
    if response_col:
        clinical_df["platinum_response_std"] = clinical_df[response_col].apply(
            _standardize_response
        )
        n_labeled = clinical_df["platinum_response_std"].notna().sum()
        logger.info("  Standardized %d/%d samples with response labels",
                     n_labeled, len(clinical_df))

    # Save
    out_dir = PROCESSED_DIR / accession
    out_dir.mkdir(parents=True, exist_ok=True)

    expr_df.to_parquet(out_dir / "expression_probes.parquet")
    expr_genes.to_parquet(out_dir / "expression_genes.parquet")
    clinical_df.to_parquet(out_dir / "clinical.parquet", index=False)

    logger.info("  Saved processed data to %s", out_dir)
    return {
        "n_samples": expr_df.shape[1],
        "n_probes": expr_df.shape[0],
        "n_genes": expr_genes.shape[0],
        "n_response_labels": int(clinical_df.get("platinum_response_std", pd.Series()).notna().sum()),
    }


def _get_probe_to_gene_mapping(gse, platform_id):
    """Extract probe-to-gene-symbol mapping from the GEO platform annotation."""
    try:
        if platform_id and platform_id in gse.gpls:
            gpl = gse.gpls[platform_id]
        elif gse.gpls:
            gpl = list(gse.gpls.values())[0]
        else:
            return None

        table = gpl.table
        if table is None or table.empty:
            return None

        # Look for gene symbol column
        gene_cols = [c for c in table.columns if "gene" in c.lower() and "symbol" in c.lower()]
        if not gene_cols:
            gene_cols = [c for c in table.columns if c.lower() in ["gene_symbol", "gene symbol", "symbol"]]
        if not gene_cols:
            gene_cols = [c for c in table.columns if "gene" in c.lower()]

        if not gene_cols:
            return None

        mapping = table.set_index("ID")[gene_cols[0]].dropna()
        # Some entries have multiple genes separated by ///
        mapping = mapping.apply(lambda x: str(x).split("///")[0].strip() if pd.notna(x) else x)
        return mapping

    except Exception as e:
        logger.warning("Failed to get probe-to-gene mapping: %s", e)
        return None


def _map_probes_to_genes(expr_df, gene_mapping):
    """Map probe-level expression to gene-level by taking the max-mean probe."""
    common_probes = expr_df.index.intersection(gene_mapping.index)
    expr_mapped = expr_df.loc[common_probes].copy()
    expr_mapped["gene_symbol"] = gene_mapping.loc[common_probes]

    # Remove unmapped / empty gene symbols
    expr_mapped = expr_mapped[
        expr_mapped["gene_symbol"].notna()
        & (expr_mapped["gene_symbol"] != "")
        & (expr_mapped["gene_symbol"] != "---")
    ]

    # For duplicate genes, keep the probe with highest mean expression
    expr_mapped["mean_expr"] = expr_mapped.drop(columns=["gene_symbol"]).mean(axis=1)
    expr_mapped = expr_mapped.sort_values("mean_expr", ascending=False)
    expr_mapped = expr_mapped.drop_duplicates(subset="gene_symbol", keep="first")
    expr_mapped = expr_mapped.set_index("gene_symbol").drop(columns=["mean_expr"])

    return expr_mapped


def main():
    """Download and process all GEO ovarian cancer datasets."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for accession, info in GEO_DATASETS.items():
        logger.info("=" * 60)
        try:
            stats = download_geo_dataset(accession, info)
            results[accession] = stats or {}
        except Exception as e:
            logger.error("Failed to process %s: %s", accession, e)
            results[accession] = {"error": str(e)}

    # Update manifest
    update_manifest("geo", {
        "accessions": list(GEO_DATASETS.keys()),
        "descriptions": {k: v["description"] for k, v in GEO_DATASETS.items()},
        "source": "NCBI GEO via GEOparse",
        "results": results,
    })

    logger.info("GEO download complete.")


if __name__ == "__main__":
    main()
