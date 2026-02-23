#!/usr/bin/env python3
"""
Download METABRIC data from cBioPortal.

Downloads via the cBioPortal API:
  - Expression data (microarray)
  - CNV data (copy number alterations)
  - Clinical data
  - Mutation data for HRR genes

Study ID: brca_metabric

Usage:
    srun --mem=4G --time=00:30:00 python download_metabric.py
"""

import logging
import time
from pathlib import Path

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
RAW_DIR = BASE_DIR / "raw" / "metabric"
PROCESSED_DIR = BASE_DIR / "processed" / "metabric"

CBIOPORTAL_API = "https://www.cbioportal.org/api"
STUDY_ID = "brca_metabric"

HRR_GENES = [
    "BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D", "ATM",
    "ATR", "CHEK1", "CHEK2", "BARD1", "BRIP1", "FANCA",
    "FANCC", "FANCM", "RAD51B", "NBN", "MRE11", "CDK12",
]

MAX_RETRIES = 3
RETRY_DELAY = 5


def _api_get(endpoint, params=None):
    """Make a GET request to cBioPortal API with retries."""
    url = f"{CBIOPORTAL_API}/{endpoint}"
    headers = {"Accept": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning("API request failed (attempt %d/%d): %s",
                               attempt + 1, MAX_RETRIES, e)
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


def _api_post(endpoint, json_body):
    """Make a POST request to cBioPortal API with retries."""
    url = f"{CBIOPORTAL_API}/{endpoint}"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, json=json_body, headers=headers, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning("API request failed (attempt %d/%d): %s",
                               attempt + 1, MAX_RETRIES, e)
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


# ---------------------------------------------------------------------------
# Clinical data
# ---------------------------------------------------------------------------

def download_clinical():
    """Download clinical data for METABRIC."""
    logger.info("Downloading METABRIC clinical data")

    # Get all clinical data for the study
    clinical = _api_get(f"studies/{STUDY_ID}/clinical-data", params={"clinicalDataType": "PATIENT"})

    # Pivot to wide format
    rows = {}
    for entry in clinical:
        pid = entry["patientId"]
        attr = entry["clinicalAttributeId"]
        val = entry["value"]
        if pid not in rows:
            rows[pid] = {"patient_id": pid}
        rows[pid][attr] = val

    clinical_df = pd.DataFrame(list(rows.values()))

    # Also get sample-level clinical data
    sample_clinical = _api_get(f"studies/{STUDY_ID}/clinical-data", params={"clinicalDataType": "SAMPLE"})

    sample_rows = {}
    for entry in sample_clinical:
        sid = entry["sampleId"]
        attr = entry["clinicalAttributeId"]
        val = entry["value"]
        if sid not in sample_rows:
            sample_rows[sid] = {"sample_id": sid, "patient_id": entry.get("patientId", "")}
        sample_rows[sid][attr] = val

    sample_df = pd.DataFrame(list(sample_rows.values()))

    out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    clinical_df.to_parquet(out_dir / "clinical_patient.parquet", index=False)
    sample_df.to_parquet(out_dir / "clinical_sample.parquet", index=False)

    logger.info("  Clinical: %d patients, %d samples", len(clinical_df), len(sample_df))
    return sample_df


# ---------------------------------------------------------------------------
# Expression data
# ---------------------------------------------------------------------------

def download_expression():
    """Download expression data for METABRIC."""
    logger.info("Downloading METABRIC expression data")

    # Get molecular profiles
    profiles = _api_get(f"studies/{STUDY_ID}/molecular-profiles")
    expr_profiles = [p for p in profiles if p["molecularAlterationType"] == "MRNA_EXPRESSION"]

    if not expr_profiles:
        logger.warning("No expression profiles found for METABRIC")
        return

    profile_id = expr_profiles[0]["molecularProfileId"]
    logger.info("  Using profile: %s", profile_id)

    # Get all sample IDs
    samples = _api_get(f"studies/{STUDY_ID}/samples")
    sample_ids = [s["sampleId"] for s in samples]
    logger.info("  Total samples: %d", len(sample_ids))

    # Fetch expression data in batches (API has limits)
    batch_size = 100
    all_data = []

    for i in tqdm(range(0, len(sample_ids), batch_size), desc="  Expression batches"):
        batch = sample_ids[i : i + batch_size]
        body = {
            "sampleIds": batch,
            "molecularProfileId": profile_id,
        }
        try:
            data = _api_post("molecular-data/fetch", body)
            all_data.extend(data)
        except Exception as e:
            logger.warning("  Batch %d failed: %s", i // batch_size, e)
        time.sleep(0.5)  # be polite to the API

    if not all_data:
        logger.warning("  No expression data retrieved")
        return

    # Pivot into a matrix (genes x samples)
    expr_df = pd.DataFrame(all_data)
    expr_matrix = expr_df.pivot_table(
        index="entrezGeneId",
        columns="sampleId",
        values="value",
        aggfunc="first",
    )

    # Map Entrez IDs to gene symbols
    gene_info = expr_df[["entrezGeneId", "gene"]].drop_duplicates()
    gene_info = gene_info.dropna(subset=["gene"])
    gene_map = dict(zip(gene_info["entrezGeneId"], gene_info["gene"].apply(
        lambda x: x.get("hugoGeneSymbol", "") if isinstance(x, dict) else str(x)
    )))
    expr_matrix.index = expr_matrix.index.map(lambda x: gene_map.get(x, str(x)))

    out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    expr_matrix.to_parquet(out_dir / "expression.parquet")
    logger.info("  Expression matrix: %d genes x %d samples", *expr_matrix.shape)


# ---------------------------------------------------------------------------
# CNV data
# ---------------------------------------------------------------------------

def download_cnv():
    """Download copy number data for METABRIC."""
    logger.info("Downloading METABRIC CNV data")

    profiles = _api_get(f"studies/{STUDY_ID}/molecular-profiles")
    cnv_profiles = [
        p for p in profiles
        if p["molecularAlterationType"] in ("COPY_NUMBER_ALTERATION",)
    ]

    if not cnv_profiles:
        logger.warning("No CNV profiles found for METABRIC")
        return

    # Prefer discrete (GISTIC) over continuous
    discrete = [p for p in cnv_profiles if "gistic" in p["molecularProfileId"].lower()
                or "discrete" in p.get("name", "").lower()]
    profile = discrete[0] if discrete else cnv_profiles[0]
    profile_id = profile["molecularProfileId"]
    logger.info("  Using CNV profile: %s", profile_id)

    samples = _api_get(f"studies/{STUDY_ID}/samples")
    sample_ids = [s["sampleId"] for s in samples]

    batch_size = 100
    all_data = []
    for i in tqdm(range(0, len(sample_ids), batch_size), desc="  CNV batches"):
        batch = sample_ids[i : i + batch_size]
        body = {
            "sampleIds": batch,
            "molecularProfileId": profile_id,
        }
        try:
            data = _api_post("molecular-data/fetch", body)
            all_data.extend(data)
        except Exception as e:
            logger.warning("  Batch %d failed: %s", i // batch_size, e)
        time.sleep(0.5)

    if not all_data:
        logger.warning("  No CNV data retrieved")
        return

    cnv_df = pd.DataFrame(all_data)
    cnv_matrix = cnv_df.pivot_table(
        index="entrezGeneId",
        columns="sampleId",
        values="value",
        aggfunc="first",
    )

    out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cnv_matrix.to_parquet(out_dir / "cnv.parquet")
    logger.info("  CNV matrix: %d genes x %d samples", *cnv_matrix.shape)


# ---------------------------------------------------------------------------
# Mutation data
# ---------------------------------------------------------------------------

def download_mutations():
    """Download mutation data for METABRIC, focusing on HRR genes."""
    logger.info("Downloading METABRIC mutation data")

    profiles = _api_get(f"studies/{STUDY_ID}/molecular-profiles")
    mut_profiles = [p for p in profiles if p["molecularAlterationType"] == "MUTATION_EXTENDED"]

    if not mut_profiles:
        logger.warning("No mutation profiles found for METABRIC")
        return

    profile_id = mut_profiles[0]["molecularProfileId"]
    logger.info("  Using mutation profile: %s", profile_id)

    samples = _api_get(f"studies/{STUDY_ID}/samples")
    sample_ids = [s["sampleId"] for s in samples]

    # Fetch mutations for HRR genes
    # cBioPortal mutation fetch requires entrezGeneIds or gene panels
    # First get Entrez IDs for our genes of interest
    all_mutations = []
    batch_size = 100

    for i in tqdm(range(0, len(sample_ids), batch_size), desc="  Mutation batches"):
        batch = sample_ids[i : i + batch_size]
        body = {
            "sampleIds": batch,
            "molecularProfileId": profile_id,
        }
        try:
            data = _api_post("mutations/fetch", body)
            all_mutations.extend(data)
        except Exception as e:
            logger.warning("  Batch %d failed: %s", i // batch_size, e)
        time.sleep(0.5)

    if not all_mutations:
        logger.warning("  No mutation data retrieved")
        return

    mut_df = pd.DataFrame(all_mutations)

    # Extract gene symbol from the nested gene object
    if "gene" in mut_df.columns:
        mut_df["hugo_symbol"] = mut_df["gene"].apply(
            lambda x: x.get("hugoGeneSymbol", "") if isinstance(x, dict) else ""
        )

    # Filter for HRR genes
    hrr_mutations = mut_df[mut_df["hugo_symbol"].isin(HRR_GENES)].copy()

    out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mut_df.to_parquet(out_dir / "mutations_all.parquet", index=False)
    hrr_mutations.to_parquet(out_dir / "mutations_hrr_genes.parquet", index=False)

    logger.info("  Mutations: %d total, %d in HRR genes (%d unique samples)",
                len(mut_df), len(hrr_mutations),
                hrr_mutations["sampleId"].nunique() if "sampleId" in hrr_mutations.columns else 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Download METABRIC data from cBioPortal."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    download_clinical()
    download_expression()
    download_cnv()
    download_mutations()

    update_manifest("metabric", {
        "study_id": STUDY_ID,
        "data_types": [
            "Clinical (patient + sample level)",
            "Expression (microarray)",
            "CNV (GISTIC discrete)",
            "Mutations (all + HRR gene subset)",
        ],
        "source": f"cBioPortal API ({CBIOPORTAL_API})",
        "hrr_genes_tracked": HRR_GENES,
    })

    logger.info("METABRIC download complete.")


if __name__ == "__main__":
    main()
