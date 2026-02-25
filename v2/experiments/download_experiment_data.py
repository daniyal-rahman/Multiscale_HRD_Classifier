#!/usr/bin/env python3
"""
Download all data needed for Experiment 1 (tiered vs simple labels).

Downloads:
  1. TCGA-BRCA RNA-seq FPKM matrix (GDC API)
  2. HRD scores (Marquard et al. supplementary - via Knijnenburg)
  3. BRCA mutation status (Knijnenburg et al. 2018 supplementary)
  4. I-SPY2 expression + biomarkers (GEO GSE173839)

Usage:
    srun --mem=8G --time=01:00:00 /home/dani/miniconda3/envs/ML/bin/python download_experiment_data.py
"""

import gzip
import io
import json
import logging
import os
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
VALIDATION_DIR = DATA_DIR / "validation"
GDC_API = "https://api.gdc.cancer.gov"


# ---------------------------------------------------------------------------
# 1. TCGA-BRCA RNA-seq (GDC STAR-Counts)
# ---------------------------------------------------------------------------

def download_tcga_rnaseq():
    """Download TCGA-BRCA RNA-seq FPKM from GDC API and assemble matrix."""
    outfile = DATA_DIR / "tcga.brca.rnaseq.unstranded.fpkm.counts.matrix.txt"
    if outfile.exists():
        logger.info("RNA-seq matrix already exists: %s", outfile)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Querying GDC for TCGA-BRCA RNA-seq files...")

    # Query for STAR-Counts gene expression files
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
            {"op": "in", "content": {"field": "files.data_category", "value": ["Transcriptome Profiling"]}},
            {"op": "in", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": ["STAR - Counts"]}},
        ],
    }

    all_file_ids = []
    offset = 0
    page_size = 500
    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,cases.submitter_id,cases.samples.sample_type",
            "format": "JSON",
            "size": str(page_size),
            "from": str(offset),
        }
        resp = requests.get(f"{GDC_API}/files", params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()["data"]
        hits = data["hits"]
        if not hits:
            break
        for h in hits:
            case_id = None
            sample_type = None
            try:
                case_id = h["cases"][0]["submitter_id"]
                samples = h["cases"][0].get("samples", [])
                if samples:
                    sample_type = samples[0].get("sample_type", "")
            except (KeyError, IndexError):
                pass
            all_file_ids.append({
                "file_id": h["file_id"],
                "file_name": h.get("file_name", ""),
                "case_id": case_id,
                "sample_type": sample_type,
            })
        offset += page_size
        if offset >= data["pagination"]["total"]:
            break

    logger.info("Found %d RNA-seq files", len(all_file_ids))

    # Download and parse in batches
    expr_data = {}  # case_id -> {gene: fpkm}
    sample_info = {}  # case_id -> {Sample ID, Sample Type}
    batch_size = 50
    ids_only = [f["file_id"] for f in all_file_ids]

    for batch_start in tqdm(range(0, len(ids_only), batch_size), desc="Downloading RNA-seq"):
        batch = ids_only[batch_start:batch_start + batch_size]
        payload = {"ids": batch}

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{GDC_API}/data",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=300,
                )
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt < 2:
                    logger.warning("Retry %d: %s", attempt + 1, e)
                    time.sleep(5 * (attempt + 1))
                else:
                    raise

        # Parse tar archive
        try:
            tar_bytes = io.BytesIO(resp.content)
            with tarfile.open(fileobj=tar_bytes) as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    fname = Path(member.name).name
                    if not fname.endswith(".tsv") and not fname.endswith(".tsv.gz"):
                        continue

                    content = tar.extractfile(member).read()
                    try:
                        if fname.endswith(".gz"):
                            content = gzip.decompress(content)
                        df = pd.read_csv(io.BytesIO(content), sep="\t", comment="#", index_col=0)
                    except Exception:
                        continue

                    if "fpkm_unstranded" not in df.columns:
                        continue

                    # Find which file_info this corresponds to
                    file_id = member.name.split("/")[0] if "/" in member.name else fname.split(".")[0]
                    matching = [f for f in all_file_ids[batch_start:batch_start + batch_size]
                                if f["file_id"] == file_id]
                    if not matching:
                        # Try finding by any file in batch
                        matching = [f for f in all_file_ids if f["file_id"] == file_id]
                    if not matching:
                        continue

                    case_id = matching[0]["case_id"]
                    stype = matching[0].get("sample_type", "")

                    if case_id and case_id not in expr_data:
                        # Build gene labels: ENSG|GeneName|gene_type
                        gene_labels = []
                        for idx in df.index:
                            gene_name = df.loc[idx, "gene_name"] if "gene_name" in df.columns else idx
                            gene_type = df.loc[idx, "gene_type"] if "gene_type" in df.columns else ""
                            gene_labels.append(f"{idx}|{gene_name}|{gene_type}")
                        expr_data[case_id] = pd.Series(
                            df["fpkm_unstranded"].values,
                            index=gene_labels,
                        )
                        sample_info[case_id] = {"Sample ID": case_id, "Sample Type": stype}
        except tarfile.TarError:
            # Single file response
            pass

    logger.info("Parsed expression for %d samples", len(expr_data))

    if not expr_data:
        logger.error("No expression data parsed!")
        return

    # Build matrix: rows = samples, columns = genes
    expr_df = pd.DataFrame(expr_data).T
    expr_df.index.name = "Case ID"

    # Add sample info columns
    info_df = pd.DataFrame(sample_info).T
    info_df.index.name = "Case ID"
    expr_df = info_df.join(expr_df)

    expr_df.to_csv(outfile, sep="\t")
    logger.info("Saved RNA-seq matrix: %s (%d x %d)", outfile, *expr_df.shape)


# ---------------------------------------------------------------------------
# 2. HRD scores (Marquard et al.)
# ---------------------------------------------------------------------------

def download_hrd_scores():
    """Download HRD scores from public supplementary data."""
    outfile = DATA_DIR / "tcga.hrdscore.xlsx"
    if outfile.exists():
        logger.info("HRD scores already exist: %s", outfile)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # The HRD scores come from Marquard et al. (doi:10.1158/1541-7786.MCR-15-0418)
    # Available as supplementary table from the Cell Reports paper by
    # Knijnenburg et al. 2018 (doi:10.1016/j.celrep.2018.03.076)
    # Direct link to the processed table:
    url = "https://www.cell.com/cms/10.1016/j.celrep.2018.03.076/attachment/0bd4bad6-e607-4618-b7ab-eb8d2868e946/mmc2.xlsx"

    logger.info("Downloading HRD scores from %s", url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    with open(outfile, "wb") as f:
        f.write(resp.content)
    logger.info("Saved HRD scores: %s", outfile)


# ---------------------------------------------------------------------------
# 3. BRCA mutation status (Knijnenburg et al.)
# ---------------------------------------------------------------------------

def download_brca_status():
    """Download BRCA status annotations from Knijnenburg et al. 2018."""
    outfile = DATA_DIR / "toga.breast.brca.status.txt"
    if outfile.exists():
        logger.info("BRCA status already exists: %s", outfile)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Knijnenburg et al. 2018 supplementary Table S5 (BRCA-specific annotations)
    url = "https://www.cell.com/cms/10.1016/j.celrep.2018.03.076/attachment/5bebfca5-30e6-41b7-96c0-3ab35f4e4c8e/mmc6.xlsx"

    logger.info("Downloading BRCA status from %s", url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    # Parse the Excel file and save as TSV (matching expected format)
    xlsx_path = DATA_DIR / "knijnenburg_brca_status_raw.xlsx"
    with open(xlsx_path, "wb") as f:
        f.write(resp.content)

    try:
        df = pd.read_excel(xlsx_path, sheet_name=0)
        # The Knijnenburg table has sample IDs with dots; the pipeline expects them as-is
        df.to_csv(outfile, sep="\t", index=True)
        logger.info("Saved BRCA status: %s (%d samples)", outfile, len(df))
    except Exception as e:
        logger.error("Failed to parse BRCA status Excel: %s", e)
        # Save raw Excel as fallback
        import shutil
        shutil.copy(xlsx_path, outfile.with_suffix(".xlsx"))


# ---------------------------------------------------------------------------
# 4. I-SPY2 validation data (GEO GSE173839)
# ---------------------------------------------------------------------------

def download_ispy2():
    """Download I-SPY2 validation data from GEO."""
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    expr_file = VALIDATION_DIR / "GSE173839_ISPY2_AgilentGeneExp_durvaPlusCtr_FFPE_meanCol_geneLevel_n105.txt"
    bio_file = VALIDATION_DIR / "GSE173839_ISPY2_DurvalumabOlaparibArm_biomarkers.csv"

    base_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE173nnn/GSE173839/suppl/"

    if not expr_file.exists():
        fname = "GSE173839_ISPY2_AgilentGeneExp_durvaPlusCtr_FFPE_meanCol_geneLevel_n105.txt.gz"
        url = base_url + fname
        logger.info("Downloading I-SPY2 expression from %s", url)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        content = gzip.decompress(resp.content)
        with open(expr_file, "wb") as f:
            f.write(content)
        logger.info("Saved: %s", expr_file)
    else:
        logger.info("I-SPY2 expression already exists: %s", expr_file)

    if not bio_file.exists():
        fname = "GSE173839_ISPY2_DurvalumabOlaparibArm_biomarkers.csv.gz"
        url = base_url + fname
        logger.info("Downloading I-SPY2 biomarkers from %s", url)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        content = gzip.decompress(resp.content)
        with open(bio_file, "wb") as f:
            f.write(content)
        logger.info("Saved: %s", bio_file)
    else:
        logger.info("I-SPY2 biomarkers already exist: %s", bio_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Downloading experiment data")
    logger.info("=" * 60)

    # Download in order of importance
    download_hrd_scores()
    download_brca_status()
    download_ispy2()
    download_tcga_rnaseq()

    logger.info("=" * 60)
    logger.info("Data download complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
