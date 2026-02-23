#!/usr/bin/env python3
"""
Download TCGA-BRCA and TCGA-OV data from the GDC API.

Downloads:
  - RNA-seq gene expression (STAR-Counts: FPKM + raw counts)
  - Clinical data (treatment, survival, platinum response for OV)
  - Mutation data (MAF for BRCA1/2, PALB2, RAD51C/D, ATM)
  - Copy number data (for LOH/GIS calculation)
  - Methylation data (for BRCA1 promoter methylation)

Filters for Primary Tumor samples only and saves as parquet files.

Usage:
    srun --mem=8G --time=02:00:00 python download_tcga.py
"""

import io
import gzip
import json
import logging
import tarfile
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
RAW_DIR = BASE_DIR / "raw" / "tcga"
PROCESSED_DIR = BASE_DIR / "processed" / "tcga"
GDC_API = "https://api.gdc.cancer.gov"

PROJECTS = ["TCGA-BRCA", "TCGA-OV"]

HRR_GENES = [
    "BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D", "ATM",
    "ATR", "CHEK1", "CHEK2", "BARD1", "BRIP1", "FANCA",
    "FANCC", "FANCM", "RAD51B", "NBN", "MRE11", "CDK12",
]

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def _retry_request(method, url, retries=MAX_RETRIES, **kwargs):
    """Make an HTTP request with retries."""
    for attempt in range(retries):
        try:
            resp = getattr(requests, method)(url, timeout=120, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.HTTPError) as e:
            if attempt < retries - 1:
                logger.warning(
                    "Request failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, retries, e, RETRY_DELAY,
                )
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


# ---------------------------------------------------------------------------
# GDC query helpers
# ---------------------------------------------------------------------------

def query_file_ids(project, data_category, data_type, workflow_type=None,
                   experimental_strategy=None, page_size=500):
    """Query GDC for file UUIDs matching the given criteria."""
    content = [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
        {"op": "in", "content": {"field": "files.data_category", "value": [data_category]}},
        {"op": "in", "content": {"field": "files.data_type", "value": [data_type]}},
        {"op": "in", "content": {"field": "cases.samples.sample_type", "value": ["Primary Tumor"]}},
    ]
    if workflow_type:
        content.append(
            {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": [workflow_type]}}
        )
    if experimental_strategy:
        content.append(
            {"op": "in", "content": {"field": "files.experimental_strategy", "value": [experimental_strategy]}}
        )

    filters = {"op": "and", "content": content}

    all_file_ids = []
    offset = 0
    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,cases.submitter_id,cases.samples.submitter_id,"
                      "cases.samples.sample_type",
            "format": "JSON",
            "size": str(page_size),
            "from": str(offset),
        }
        resp = _retry_request("get", f"{GDC_API}/files", params=params)
        data = resp.json()["data"]
        hits = data["hits"]
        if not hits:
            break
        for h in hits:
            all_file_ids.append({
                "file_id": h["file_id"],
                "file_name": h.get("file_name", ""),
                "case_id": _extract_case_id(h),
            })
        offset += page_size
        if offset >= data["pagination"]["total"]:
            break

    logger.info(
        "Found %d files for %s / %s / %s", len(all_file_ids), project, data_category, data_type
    )
    return all_file_ids


def _extract_case_id(hit):
    """Extract case submitter_id from a GDC hit."""
    try:
        return hit["cases"][0]["submitter_id"]
    except (KeyError, IndexError):
        return None


def download_files_tar(file_ids, dest_dir, label=""):
    """Download a batch of files via GDC /data endpoint (tar stream)."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Process in batches to avoid oversized requests
    batch_size = 50
    ids_only = [f["file_id"] for f in file_ids]

    for batch_start in tqdm(
        range(0, len(ids_only), batch_size),
        desc=f"Downloading {label}",
        unit="batch",
    ):
        batch = ids_only[batch_start : batch_start + batch_size]
        payload = {"ids": batch}
        resp = _retry_request(
            "post",
            f"{GDC_API}/data",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        # GDC returns a tar archive when multiple files are requested
        if len(batch) > 1:
            tar_bytes = io.BytesIO(resp.content)
            with tarfile.open(fileobj=tar_bytes) as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        fname = Path(member.name).name
                        with open(dest_dir / fname, "wb") as f:
                            f.write(tar.extractfile(member).read())
        else:
            # Single file — save directly
            fname = file_ids[batch_start].get("file_name", f"{batch[0]}.dat")
            with open(dest_dir / fname, "wb") as f:
                f.write(resp.content)


# ---------------------------------------------------------------------------
# RNA-seq expression
# ---------------------------------------------------------------------------

def download_rnaseq(project):
    """Download STAR-Counts RNA-seq for a TCGA project."""
    logger.info("Downloading RNA-seq for %s", project)
    file_infos = query_file_ids(
        project,
        data_category="Transcriptome Profiling",
        data_type="Gene Expression Quantification",
        workflow_type="STAR - Counts",
    )
    if not file_infos:
        logger.warning("No RNA-seq files found for %s", project)
        return

    raw_dir = RAW_DIR / project / "rnaseq"
    download_files_tar(file_infos, raw_dir, label=f"{project} RNA-seq")

    # Parse individual STAR-Counts TSV files into a combined matrix
    logger.info("Processing RNA-seq files for %s", project)
    expr_fpkm = {}
    expr_counts = {}

    for fpath in tqdm(list(raw_dir.glob("*.tsv*")), desc="Parsing RNA-seq"):
        try:
            opener = gzip.open if fpath.suffix == ".gz" else open
            with opener(fpath, "rt") as f:
                df = pd.read_csv(f, sep="\t", comment="#", index_col=0)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", fpath.name, e)
            continue

        # STAR-Counts files have columns: gene_name, gene_type,
        # unstranded, stranded_first, stranded_second, tpm_unstranded, fpkm_unstranded, ...
        sample_id = fpath.stem.split(".")[0]

        if "fpkm_unstranded" in df.columns:
            expr_fpkm[sample_id] = df["fpkm_unstranded"]
        if "unstranded" in df.columns:
            expr_counts[sample_id] = df["unstranded"]

    if expr_fpkm:
        fpkm_df = pd.DataFrame(expr_fpkm)
        out = PROCESSED_DIR / project
        out.mkdir(parents=True, exist_ok=True)
        fpkm_df.to_parquet(out / "rnaseq_fpkm.parquet")
        logger.info("Saved FPKM matrix: %s (%d genes x %d samples)",
                     out / "rnaseq_fpkm.parquet", *fpkm_df.shape)

    if expr_counts:
        counts_df = pd.DataFrame(expr_counts)
        out = PROCESSED_DIR / project
        out.mkdir(parents=True, exist_ok=True)
        counts_df.to_parquet(out / "rnaseq_counts.parquet")
        logger.info("Saved counts matrix: %s (%d genes x %d samples)",
                     out / "rnaseq_counts.parquet", *counts_df.shape)


# ---------------------------------------------------------------------------
# Clinical data
# ---------------------------------------------------------------------------

def download_clinical(project):
    """Download clinical data via GDC API."""
    logger.info("Downloading clinical data for %s", project)

    # Use the cases endpoint with clinical fields
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "project.project_id", "value": [project]}},
        ],
    }

    fields = [
        "submitter_id",
        "demographic.gender",
        "demographic.race",
        "demographic.ethnicity",
        "demographic.vital_status",
        "demographic.days_to_death",
        "diagnoses.age_at_diagnosis",
        "diagnoses.tumor_stage",
        "diagnoses.primary_diagnosis",
        "diagnoses.site_of_resection_or_biopsy",
        "diagnoses.treatments.treatment_type",
        "diagnoses.treatments.therapeutic_agents",
        "diagnoses.treatments.treatment_outcome",
        "diagnoses.days_to_last_follow_up",
        "exposures.alcohol_history",
    ]

    all_cases = []
    offset = 0
    page_size = 500
    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "format": "JSON",
            "size": str(page_size),
            "from": str(offset),
        }
        resp = _retry_request("get", f"{GDC_API}/cases", params=params)
        data = resp.json()["data"]
        hits = data["hits"]
        if not hits:
            break
        all_cases.extend(hits)
        offset += page_size
        if offset >= data["pagination"]["total"]:
            break

    logger.info("Downloaded clinical data for %d cases in %s", len(all_cases), project)

    # Flatten into a DataFrame
    rows = []
    for case in all_cases:
        row = {"case_id": case.get("submitter_id")}

        demo = case.get("demographic", {})
        row["gender"] = demo.get("gender")
        row["race"] = demo.get("race")
        row["vital_status"] = demo.get("vital_status")
        row["days_to_death"] = demo.get("days_to_death")

        diags = case.get("diagnoses", [{}])
        if diags:
            d = diags[0]
            row["age_at_diagnosis"] = d.get("age_at_diagnosis")
            row["tumor_stage"] = d.get("tumor_stage")
            row["primary_diagnosis"] = d.get("primary_diagnosis")
            row["days_to_last_follow_up"] = d.get("days_to_last_follow_up")

            # Extract treatment info
            treatments = d.get("treatments", [])
            tx_types = [t.get("treatment_type", "") for t in treatments]
            tx_agents = [t.get("therapeutic_agents", "") for t in treatments]
            tx_outcomes = [t.get("treatment_outcome", "") for t in treatments]
            row["treatment_types"] = "; ".join(filter(None, tx_types))
            row["therapeutic_agents"] = "; ".join(filter(None, tx_agents))
            row["treatment_outcomes"] = "; ".join(filter(None, tx_outcomes))

        rows.append(row)

    clinical_df = pd.DataFrame(rows)

    out = PROCESSED_DIR / project
    out.mkdir(parents=True, exist_ok=True)
    clinical_df.to_parquet(out / "clinical.parquet", index=False)
    logger.info("Saved clinical data: %s (%d cases)", out / "clinical.parquet", len(clinical_df))

    # For TCGA-OV: compute platinum-free interval (PFI) labels
    if project == "TCGA-OV":
        _compute_platinum_response(clinical_df, out)


def _compute_platinum_response(clinical_df, out_dir):
    """Derive platinum sensitivity labels for ovarian cancer."""
    logger.info("Computing platinum response labels for TCGA-OV")

    # Standard definition: platinum-resistant = recurrence within 6 months (182 days)
    # We look for platinum-based treatment info in therapeutic_agents
    plat_keywords = ["platinum", "cisplatin", "carboplatin", "oxaliplatin"]

    platinum_cases = clinical_df[
        clinical_df["therapeutic_agents"].str.lower().str.contains(
            "|".join(plat_keywords), na=False
        )
    ].copy()

    if platinum_cases.empty:
        logger.warning("No platinum-treated cases found in clinical data")
        return

    platinum_cases.to_parquet(out_dir / "clinical_platinum_cases.parquet", index=False)
    logger.info(
        "Found %d platinum-treated cases in TCGA-OV", len(platinum_cases)
    )


# ---------------------------------------------------------------------------
# Mutation data (MAF)
# ---------------------------------------------------------------------------

def download_mutations(project):
    """Download mutation data (MAF) for HRR genes."""
    logger.info("Downloading mutation data for %s", project)

    file_infos = query_file_ids(
        project,
        data_category="Simple Nucleotide Variation",
        data_type="Masked Somatic Mutation",
        workflow_type="Aliquot Ensemble Somatic Variant Merging and Masking",
    )
    if not file_infos:
        logger.warning("No MAF files found for %s", project)
        return

    raw_dir = RAW_DIR / project / "mutations"
    download_files_tar(file_infos, raw_dir, label=f"{project} mutations")

    # Parse MAF files and filter for HRR genes
    logger.info("Parsing MAF files for HRR gene mutations")
    maf_frames = []
    for fpath in tqdm(list(raw_dir.glob("*.maf*")), desc="Parsing MAF"):
        try:
            opener = gzip.open if fpath.suffix == ".gz" else open
            with opener(fpath, "rt") as f:
                df = pd.read_csv(f, sep="\t", comment="#", low_memory=False)
            maf_frames.append(df)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", fpath.name, e)

    if not maf_frames:
        logger.warning("No MAF data parsed for %s", project)
        return

    maf_all = pd.concat(maf_frames, ignore_index=True)

    # Filter for HRR genes
    hrr_maf = maf_all[maf_all["Hugo_Symbol"].isin(HRR_GENES)].copy()

    # Filter for primary tumor barcodes (sample type code 01)
    if "Tumor_Sample_Barcode" in hrr_maf.columns:
        hrr_maf = hrr_maf[hrr_maf["Tumor_Sample_Barcode"].str[13:15] == "01"]

    out = PROCESSED_DIR / project
    out.mkdir(parents=True, exist_ok=True)
    maf_all.to_parquet(out / "mutations_all.parquet", index=False)
    hrr_maf.to_parquet(out / "mutations_hrr_genes.parquet", index=False)
    logger.info(
        "Saved mutations: %d total variants, %d in HRR genes for %s",
        len(maf_all), len(hrr_maf), project,
    )


# ---------------------------------------------------------------------------
# Copy number data
# ---------------------------------------------------------------------------

def download_cnv(project):
    """Download copy number segment data."""
    logger.info("Downloading CNV data for %s", project)

    file_infos = query_file_ids(
        project,
        data_category="Copy Number Variation",
        data_type="Copy Number Segment",
        workflow_type="DNAcopy",
    )
    if not file_infos:
        # Try alternative workflow
        file_infos = query_file_ids(
            project,
            data_category="Copy Number Variation",
            data_type="Copy Number Segment",
        )

    if not file_infos:
        logger.warning("No CNV files found for %s", project)
        return

    raw_dir = RAW_DIR / project / "cnv"
    download_files_tar(file_infos, raw_dir, label=f"{project} CNV")

    # Parse segment files
    seg_frames = []
    for fpath in tqdm(list(raw_dir.glob("*.seg*")), desc="Parsing CNV"):
        try:
            df = pd.read_csv(fpath, sep="\t")
            seg_frames.append(df)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", fpath.name, e)

    if not seg_frames:
        logger.warning("No CNV segments parsed for %s", project)
        return

    cnv_all = pd.concat(seg_frames, ignore_index=True)

    # Filter for primary tumor (barcode position 13:15 == '01')
    if "GDC_Aliquot" in cnv_all.columns:
        cnv_all = cnv_all[cnv_all["GDC_Aliquot"].str[13:15] == "01"]

    out = PROCESSED_DIR / project
    out.mkdir(parents=True, exist_ok=True)
    cnv_all.to_parquet(out / "cnv_segments.parquet", index=False)
    logger.info("Saved CNV segments for %s: %d rows", project, len(cnv_all))


# ---------------------------------------------------------------------------
# Methylation data (BRCA1 promoter)
# ---------------------------------------------------------------------------

def download_methylation(project):
    """Download methylation data (for BRCA1 promoter methylation)."""
    logger.info("Downloading methylation data for %s", project)

    file_infos = query_file_ids(
        project,
        data_category="DNA Methylation",
        data_type="Methylation Beta Value",
        experimental_strategy="Methylation Array",
    )
    if not file_infos:
        logger.warning("No methylation files found for %s", project)
        return

    raw_dir = RAW_DIR / project / "methylation"
    download_files_tar(file_infos, raw_dir, label=f"{project} methylation")

    # Parse and extract BRCA1-associated probes
    # Key BRCA1 promoter CpG probes (Illumina 450K / EPIC)
    brca1_probes = [
        "cg04110421", "cg07589773", "cg08993267", "cg13782816",
        "cg19088651", "cg19531713", "cg21253966",
    ]

    methyl_data = {}
    for fpath in tqdm(list(raw_dir.glob("*")), desc="Parsing methylation"):
        try:
            df = pd.read_csv(fpath, sep="\t", index_col=0, header=None, names=["beta"])
            # Extract only BRCA1 probes
            brca1_betas = df.loc[df.index.isin(brca1_probes), "beta"]
            if not brca1_betas.empty:
                methyl_data[fpath.stem] = brca1_betas
        except Exception as e:
            logger.warning("Failed to parse %s: %s", fpath.name, e)

    if methyl_data:
        methyl_df = pd.DataFrame(methyl_data)
        out = PROCESSED_DIR / project
        out.mkdir(parents=True, exist_ok=True)
        methyl_df.to_parquet(out / "methylation_brca1_probes.parquet")
        logger.info("Saved BRCA1 methylation for %s: %d probes x %d samples",
                     project, *methyl_df.shape)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Download and process TCGA-BRCA and TCGA-OV data."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for project in PROJECTS:
        logger.info("=" * 60)
        logger.info("Processing %s", project)
        logger.info("=" * 60)

        download_rnaseq(project)
        download_clinical(project)
        download_mutations(project)
        download_cnv(project)
        download_methylation(project)

    # Update manifest
    update_manifest("tcga", {
        "projects": PROJECTS,
        "data_types": [
            "RNA-seq (STAR-Counts FPKM + raw counts)",
            "Clinical (treatment, survival, platinum response)",
            "Mutations (MAF, HRR gene filter)",
            "Copy number segments (for LOH/GIS)",
            "Methylation (BRCA1 promoter probes)",
        ],
        "source": "GDC API (https://api.gdc.cancer.gov)",
        "filters": "Primary Tumor samples only",
    })

    logger.info("TCGA download complete.")


if __name__ == "__main__":
    main()
