#!/usr/bin/env python3
"""
Download and process I-SPY2 validation data.

Data sources:
  - GSE173839: I-SPY2 durvalumab/olaparib arm biomarkers (GEO)
  - Pusztai et al. 2021 expression data (supplementary / pre-existing in repo)
  - Existing processed data in the repo's data/validation/ directory

Some I-SPY2 data may already be present at:
  data/validation/
  ~/Data/ClinicalData/

This script processes existing files and downloads additional public data.

Usage:
    srun --mem=4G --time=00:30:00 python download_ispy2.py
"""

import logging
from pathlib import Path

import GEOparse
import numpy as np
import pandas as pd

import sys; sys.path.insert(0, str(Path(__file__).parent))
from _manifest import update_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "ispy2"
PROCESSED_DIR = BASE_DIR / "processed" / "ispy2"
REPO_ROOT = BASE_DIR.parent.parent

# Known locations of existing I-SPY2 data in the repo / filesystem
EXISTING_DATA_PATHS = {
    "expression": Path.home() / "Data" / "ClinicalData" / "ISPY2_Puzstai2021_expression.txt",
    "biomarkers": Path.home() / "Data" / "ClinicalData" / "GSE173839_ISPY2_DurvalumabOlaparibArm_biomarkers.csv",
    "validation_dir": REPO_ROOT / "data" / "validation",
}

# GEO accession for I-SPY2 durvalumab/olaparib arm
ISPY2_GEO = "GSE173839"


def check_existing_data():
    """Check what I-SPY2 data already exists locally."""
    logger.info("Checking for existing I-SPY2 data")
    found = {}

    for name, path in EXISTING_DATA_PATHS.items():
        if path.exists():
            if path.is_dir():
                files = list(path.glob("*"))
                logger.info("  Found %s: %d files in %s", name, len(files), path)
                found[name] = [str(f) for f in files]
            else:
                size_mb = path.stat().st_size / 1e6
                logger.info("  Found %s: %.1f MB at %s", name, size_mb, path)
                found[name] = str(path)
        else:
            logger.info("  Not found: %s (%s)", name, path)

    return found


def process_existing_expression(expr_path):
    """Process existing I-SPY2 expression data from Pusztai et al. 2021."""
    logger.info("Processing I-SPY2 expression data: %s", expr_path)

    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    logger.info("  Expression matrix: %d genes x %d samples", *expr.shape)

    out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    expr.to_parquet(out_dir / "expression.parquet")

    return expr


def process_existing_biomarkers(bio_path):
    """Process I-SPY2 durvalumab/olaparib biomarker data."""
    logger.info("Processing I-SPY2 biomarker data: %s", bio_path)

    bio = pd.read_csv(bio_path)
    logger.info("  Biomarkers: %d samples x %d columns", *bio.shape)

    # Filter for durvalumab/olaparib arm
    dvo_arm = bio[bio["Arm"] == "durvalumab/olaparib"].copy()
    logger.info("  Durvalumab/olaparib arm: %d samples", len(dvo_arm))

    # Standardize pCR labels: -1 -> 0 (as in original repo code)
    if "pCR.status" in dvo_arm.columns:
        dvo_arm.loc[dvo_arm["pCR.status"] == -1, "pCR.status"] = 0
        n_pcr = dvo_arm["pCR.status"].sum()
        logger.info("  pCR rate: %d/%d (%.1f%%)",
                     n_pcr, len(dvo_arm), 100 * n_pcr / len(dvo_arm))

    out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    bio.to_parquet(out_dir / "biomarkers_all_arms.parquet", index=False)
    dvo_arm.to_parquet(out_dir / "biomarkers_dvo_arm.parquet", index=False)

    return dvo_arm


def download_geo_ispy2():
    """Download supplementary data from GSE173839 via GEO."""
    logger.info("Downloading GSE173839 from GEO")

    raw_dir = RAW_DIR / ISPY2_GEO
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        gse = GEOparse.get_GEO(
            geo=ISPY2_GEO,
            destdir=str(raw_dir),
            silent=True,
        )

        # Extract sample metadata
        rows = []
        for gsm_name, gsm in gse.gsms.items():
            row = {"sample_id": gsm_name}
            row["title"] = gsm.metadata.get("title", [""])[0]
            chars = gsm.metadata.get("characteristics_ch1", [])
            for char in chars:
                if ":" in char:
                    key, val = char.split(":", 1)
                    row[key.strip().lower().replace(" ", "_")] = val.strip()
            rows.append(row)

        meta_df = pd.DataFrame(rows)
        out_dir = PROCESSED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_df.to_parquet(out_dir / "geo_metadata.parquet", index=False)
        logger.info("  Downloaded metadata for %d samples from %s", len(meta_df), ISPY2_GEO)

    except Exception as e:
        logger.error("Failed to download %s: %s", ISPY2_GEO, e)


def create_combined_dataset():
    """Create a combined I-SPY2 dataset with expression + clinical annotations."""
    logger.info("Creating combined I-SPY2 dataset")

    expr_path = PROCESSED_DIR / "expression.parquet"
    bio_path = PROCESSED_DIR / "biomarkers_dvo_arm.parquet"

    if not expr_path.exists() or not bio_path.exists():
        logger.warning("Cannot create combined dataset — missing expression or biomarker data")
        return

    expr = pd.read_parquet(expr_path)
    bio = pd.read_parquet(bio_path)

    # Match sample IDs (expression has X-prefixed IDs)
    bio["ResearchID_expr"] = "X" + bio["ResearchID"].astype(str)
    matched_ids = set(bio["ResearchID_expr"]) & set(expr.columns)

    if not matched_ids:
        logger.warning("No matching sample IDs between expression and biomarkers")
        return

    expr_matched = expr[list(matched_ids)]
    bio_matched = bio[bio["ResearchID_expr"].isin(matched_ids)]

    logger.info("  Combined: %d genes x %d matched samples", expr_matched.shape[0], len(matched_ids))

    expr_matched.to_parquet(PROCESSED_DIR / "expression_matched.parquet")
    bio_matched.to_parquet(PROCESSED_DIR / "biomarkers_matched.parquet", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Process I-SPY2 validation data."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing data
    existing = check_existing_data()

    # Process existing expression data
    if "expression" in existing:
        process_existing_expression(Path(existing["expression"]))

    # Process existing biomarker data
    if "biomarkers" in existing:
        process_existing_biomarkers(Path(existing["biomarkers"]))

    # Download from GEO
    download_geo_ispy2()

    # Create combined dataset
    create_combined_dataset()

    # Update manifest
    update_manifest("ispy2", {
        "data_available": {
            "expression": "Pusztai et al. 2021 (pre-treatment gene expression)",
            "biomarkers": "GSE173839 durvalumab/olaparib arm biomarkers",
            "pCR_endpoint": "Pathological complete response (pCR)",
        },
        "existing_data_checked": list(existing.keys()),
        "geo_accession": ISPY2_GEO,
        "notes": (
            "I-SPY2 is an adaptive platform trial. "
            "Primary HRD validation uses durvalumab/olaparib arm. "
            "PARPi7 signature score available as comparator. "
            "Additional arms and timepoints may be available through "
            "the I-SPY2 data sharing portal."
        ),
        "access": {
            "public": "GSE173839 (GEO), selected supplementary tables",
            "restricted": "Full trial data requires DUA with I-SPY2 consortium",
            "portal": "https://www.ispytrials.org/results/data-access",
        },
    })

    logger.info("I-SPY2 processing complete.")


if __name__ == "__main__":
    main()
