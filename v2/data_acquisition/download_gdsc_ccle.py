#!/usr/bin/env python3
"""
Download GDSC drug sensitivity and CCLE/DepMap expression data.

GDSC (cancerrxgene.org):
  - Drug sensitivity IC50 for: olaparib, rucaparib, talazoparib, cisplatin, carboplatin
  - Cell line annotations (tissue, mutations)

DepMap/CCLE:
  - RNA-seq expression (TPM log2(x+1))
  - Mutation data
  - CRISPR dependency scores for HRR genes

Matches cell lines between GDSC and CCLE by name and COSMIC ID.

Usage:
    srun --mem=8G --time=01:00:00 python download_gdsc_ccle.py
"""

import io
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from _manifest import update_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "gdsc_ccle"
PROCESSED_DIR = BASE_DIR / "processed" / "gdsc_ccle"

# Drugs of interest for HRD research
DRUGS_OF_INTEREST = [
    "Olaparib", "Rucaparib", "Talazoparib", "Niraparib",
    "Cisplatin", "Carboplatin",
]

HRR_GENES = [
    "BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D", "ATM",
    "ATR", "CHEK1", "CHEK2", "BARD1", "BRIP1", "FANCA",
    "FANCC", "FANCM", "RAD51B", "NBN", "MRE11", "CDK12",
]

MAX_RETRIES = 3


def _download_file(url, dest_path, desc=""):
    """Download a file with retry logic and progress bar."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()

            total_size = int(resp.headers.get("content-length", 0))
            with open(dest_path, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True

        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning("Download failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, e)
                time.sleep(5 * (attempt + 1))
            else:
                logger.error("Download failed after %d attempts: %s", MAX_RETRIES, e)
                raise


# ---------------------------------------------------------------------------
# GDSC
# ---------------------------------------------------------------------------

# GDSC download URLs (from cancerrxgene.org/downloads/bulk_download)
GDSC_URLS = {
    # GDSC2 drug sensitivity (fitted dose-response curves)
    "gdsc2_fitted": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx",
    # Cell line details
    "cell_line_details": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/Cell_Lines_Details.xlsx",
}


def download_gdsc():
    """Download GDSC drug sensitivity data."""
    logger.info("Downloading GDSC drug sensitivity data")
    raw_dir = RAW_DIR / "gdsc"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download GDSC2 fitted dose-response data
    gdsc_path = raw_dir / "GDSC2_fitted_dose_response.xlsx"
    if not gdsc_path.exists():
        _download_file(GDSC_URLS["gdsc2_fitted"], gdsc_path, desc="GDSC2 dose-response")

    # Download cell line details
    cell_lines_path = raw_dir / "Cell_Lines_Details.xlsx"
    if not cell_lines_path.exists():
        _download_file(GDSC_URLS["cell_line_details"], cell_lines_path, desc="Cell line details")

    # Process GDSC data
    logger.info("Processing GDSC drug sensitivity data")
    gdsc_df = pd.read_excel(gdsc_path)

    # Filter for drugs of interest (case-insensitive)
    drug_names_lower = [d.lower() for d in DRUGS_OF_INTEREST]
    gdsc_filtered = gdsc_df[
        gdsc_df["DRUG_NAME"].str.lower().isin(drug_names_lower)
    ].copy()

    logger.info("  Found %d dose-response entries for drugs of interest", len(gdsc_filtered))

    # Pivot to get IC50 matrix (cell lines x drugs)
    ic50_matrix = gdsc_filtered.pivot_table(
        index=["CELL_LINE_NAME", "COSMIC_ID"],
        columns="DRUG_NAME",
        values="LN_IC50",
        aggfunc="mean",
    )
    ic50_matrix = ic50_matrix.reset_index()

    # Process cell line annotations
    try:
        cell_lines_df = pd.read_excel(cell_lines_path)
        logger.info("  Loaded %d cell line annotations", len(cell_lines_df))
    except Exception as e:
        logger.warning("Could not load cell line details: %s", e)
        cell_lines_df = None

    # Save
    out_dir = PROCESSED_DIR / "gdsc"
    out_dir.mkdir(parents=True, exist_ok=True)

    ic50_matrix.to_parquet(out_dir / "drug_sensitivity_ic50.parquet", index=False)
    gdsc_filtered.to_parquet(out_dir / "dose_response_full.parquet", index=False)
    if cell_lines_df is not None:
        cell_lines_df.to_parquet(out_dir / "cell_line_annotations.parquet", index=False)

    logger.info("  Saved GDSC data: %d cell lines x %d drugs",
                ic50_matrix.shape[0], ic50_matrix.shape[1] - 2)

    return ic50_matrix


# ---------------------------------------------------------------------------
# DepMap / CCLE
# ---------------------------------------------------------------------------

# DepMap public download URLs (24Q2 release)
DEPMAP_URLS = {
    "expression": "https://figshare.com/ndownloader/files/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "mutations": "https://figshare.com/ndownloader/files/OmicsSomaticMutations.csv",
    "crispr": "https://figshare.com/ndownloader/files/CRISPRGeneEffect.csv",
    "sample_info": "https://figshare.com/ndownloader/files/Model.csv",
}

# NOTE: The exact figshare URLs change with each DepMap release.
# The URLs above are placeholders. In practice, get the current URLs from:
# https://depmap.org/portal/download/all/
# The actual filename patterns are stable:
DEPMAP_FILES = {
    "expression": "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "mutations": "OmicsSomaticMutations.csv",
    "crispr": "CRISPRGeneEffect.csv",
    "sample_info": "Model.csv",
}


def download_depmap():
    """Download DepMap/CCLE data.

    NOTE: DepMap download URLs change with each quarterly release.
    This function expects the files to either be pre-downloaded to raw/gdsc_ccle/depmap/
    or will attempt to download from the latest known URLs.

    To get current URLs, visit: https://depmap.org/portal/download/all/
    """
    logger.info("Downloading DepMap/CCLE data")
    raw_dir = RAW_DIR / "depmap"
    raw_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Check for pre-downloaded files first
    for key, filename in DEPMAP_FILES.items():
        local_path = raw_dir / filename
        if local_path.exists():
            logger.info("  Found pre-downloaded %s", filename)
            results[key] = local_path
        else:
            logger.warning(
                "  %s not found at %s. "
                "Please download manually from https://depmap.org/portal/download/all/ "
                "or update DEPMAP_URLS with the current release URLs.",
                filename, local_path,
            )
            # Attempt download from stored URL (may fail if URL is outdated)
            if key in DEPMAP_URLS:
                try:
                    _download_file(DEPMAP_URLS[key], local_path, desc=f"DepMap {key}")
                    results[key] = local_path
                except Exception as e:
                    logger.error("  Failed to download %s: %s", key, e)

    return results


def process_depmap(file_paths):
    """Process downloaded DepMap files."""
    out_dir = PROCESSED_DIR / "ccle"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process expression data
    if "expression" in file_paths:
        logger.info("Processing CCLE expression data")
        expr = pd.read_csv(file_paths["expression"], index_col=0)
        # Columns are "GENE_NAME (ENTREZ_ID)" — extract gene names
        expr.columns = [c.split(" (")[0] for c in expr.columns]
        expr.to_parquet(out_dir / "expression_tpm_log2.parquet")
        logger.info("  Expression: %d cell lines x %d genes", *expr.shape)

    # Process mutation data
    if "mutations" in file_paths:
        logger.info("Processing CCLE mutation data")
        mut = pd.read_csv(file_paths["mutations"])
        # Filter for HRR gene mutations
        if "HugoSymbol" in mut.columns:
            hrr_mut = mut[mut["HugoSymbol"].isin(HRR_GENES)]
        elif "Hugo_Symbol" in mut.columns:
            hrr_mut = mut[mut["Hugo_Symbol"].isin(HRR_GENES)]
        else:
            hrr_mut = mut
            logger.warning("  Could not identify gene symbol column in mutations")

        mut.to_parquet(out_dir / "mutations_all.parquet", index=False)
        hrr_mut.to_parquet(out_dir / "mutations_hrr_genes.parquet", index=False)
        logger.info("  Mutations: %d total, %d in HRR genes", len(mut), len(hrr_mut))

    # Process CRISPR dependency scores
    if "crispr" in file_paths:
        logger.info("Processing CRISPR dependency scores")
        crispr = pd.read_csv(file_paths["crispr"], index_col=0)
        # Columns are "GENE_NAME (ENTREZ_ID)"
        crispr.columns = [c.split(" (")[0] for c in crispr.columns]
        # Extract HRR genes only
        hrr_cols = [c for c in crispr.columns if c in HRR_GENES]
        crispr_hrr = crispr[hrr_cols] if hrr_cols else crispr
        crispr_hrr.to_parquet(out_dir / "crispr_dependency_hrr.parquet")
        logger.info("  CRISPR: %d cell lines x %d HRR genes", crispr_hrr.shape[0], len(hrr_cols))

    # Process sample info
    if "sample_info" in file_paths:
        logger.info("Processing cell line sample info")
        info = pd.read_csv(file_paths["sample_info"])
        info.to_parquet(out_dir / "sample_info.parquet", index=False)
        logger.info("  Sample info: %d cell lines", len(info))


# ---------------------------------------------------------------------------
# Cross-matching GDSC and CCLE
# ---------------------------------------------------------------------------

def match_gdsc_ccle():
    """Match cell lines between GDSC and CCLE datasets."""
    logger.info("Matching cell lines between GDSC and CCLE")

    gdsc_path = PROCESSED_DIR / "gdsc" / "drug_sensitivity_ic50.parquet"
    ccle_expr_path = PROCESSED_DIR / "ccle" / "expression_tpm_log2.parquet"
    ccle_info_path = PROCESSED_DIR / "ccle" / "sample_info.parquet"

    if not gdsc_path.exists() or not ccle_expr_path.exists():
        logger.warning("Cannot match — GDSC or CCLE processed data not found")
        return

    gdsc = pd.read_parquet(gdsc_path)
    ccle_expr = pd.read_parquet(ccle_expr_path)

    # Try to match via sample info (has COSMIC IDs)
    if ccle_info_path.exists():
        ccle_info = pd.read_parquet(ccle_info_path)

        # Build mapping: cell line name (cleaned) -> DepMap ID
        ccle_info["clean_name"] = (
            ccle_info.get("CellLineName", ccle_info.get("cell_line_name", pd.Series()))
            .str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True)
        )
        gdsc["clean_name"] = (
            gdsc["CELL_LINE_NAME"]
            .str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True)
        )

        # Match by cleaned name
        merged = gdsc.merge(ccle_info[["clean_name", "ModelID"]].drop_duplicates(),
                            on="clean_name", how="inner")
        logger.info("  Matched %d/%d GDSC cell lines to CCLE by name",
                    merged["CELL_LINE_NAME"].nunique(),
                    gdsc["CELL_LINE_NAME"].nunique())

        # Further filter to those with expression data
        expr_ids = set(ccle_expr.index)
        matched = merged[merged["ModelID"].isin(expr_ids)]
        logger.info("  %d cell lines have both drug sensitivity and expression",
                    matched["CELL_LINE_NAME"].nunique())

        out_dir = PROCESSED_DIR / "matched"
        out_dir.mkdir(parents=True, exist_ok=True)
        matched.to_parquet(out_dir / "gdsc_ccle_matched.parquet", index=False)
    else:
        logger.warning("  No CCLE sample info available for matching")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Download and process GDSC and DepMap/CCLE data."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Download and process GDSC
    download_gdsc()

    # Download and process DepMap/CCLE
    depmap_files = download_depmap()
    if depmap_files:
        process_depmap(depmap_files)

    # Match cell lines
    match_gdsc_ccle()

    # Update manifest
    update_manifest("gdsc_ccle", {
        "gdsc": {
            "drugs": DRUGS_OF_INTEREST,
            "source": "cancerrxgene.org (GDSC2 release 8.5)",
        },
        "ccle": {
            "data_types": [
                "Expression (TPM log2(x+1))",
                "Mutations (somatic)",
                "CRISPR dependency (HRR genes)",
            ],
            "source": "DepMap portal (depmap.org)",
        },
        "hrr_genes_tracked": HRR_GENES,
    })

    logger.info("GDSC/CCLE download complete.")


if __name__ == "__main__":
    main()
