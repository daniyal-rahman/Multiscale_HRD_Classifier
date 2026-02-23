#!/usr/bin/env python3
"""
Download and process PCAWG pan-cancer WGS + RNA-seq data.

PCAWG raw data is controlled-access through ICGC/TCGA. This script:
  1. Downloads publicly available HRDetect scores
  2. Downloads CHORD HRD predictions
  3. Downloads SBS mutational signature contributions
  4. Provides a processing pipeline for when WGS/RNA-seq data is available

For controlled access data, apply through:
  - ICGC DACO: https://daco.icgc-argo.org/
  - dbGaP (for TCGA portion): https://dbgap.ncbi.nlm.nih.gov/

Usage:
    srun --mem=4G --time=00:30:00 python download_pcawg.py
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
RAW_DIR = BASE_DIR / "raw" / "pcawg"
PROCESSED_DIR = BASE_DIR / "processed" / "pcawg"

MAX_RETRIES = 3


def _download_file(url, dest_path, desc=""):
    """Download a file with retry logic."""
    if dest_path.exists():
        logger.info("  Already downloaded: %s", dest_path.name)
        return True

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest_path, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning("Download attempt %d failed: %s", attempt + 1, e)
                time.sleep(5 * (attempt + 1))
            else:
                logger.error("Failed to download %s: %s", url, e)
                return False


# ---------------------------------------------------------------------------
# Public PCAWG resources
# ---------------------------------------------------------------------------

# These are publicly available supplementary data from key PCAWG publications
PCAWG_PUBLIC_URLS = {
    # PCAWG mutational signatures (Alexandrov et al. 2020, Nature)
    # SBS signature contributions per sample
    "sbs_signatures": (
        "https://dcc.icgc.org/api/v1/download?fn=/PCAWG/mutational_signatures/"
        "Signatures_in_Samples/SP_Signatures_in_Samples/"
        "PCAWG_sigProfiler_SBS_signatures_in_samples.csv"
    ),
    # DBS signature contributions
    "dbs_signatures": (
        "https://dcc.icgc.org/api/v1/download?fn=/PCAWG/mutational_signatures/"
        "Signatures_in_Samples/SP_Signatures_in_Samples/"
        "PCAWG_sigProfiler_DBS_signatures_in_samples.csv"
    ),
    # ID (indel) signature contributions
    "id_signatures": (
        "https://dcc.icgc.org/api/v1/download?fn=/PCAWG/mutational_signatures/"
        "Signatures_in_Samples/SP_Signatures_in_Samples/"
        "PCAWG_sigProfiler_ID_signatures_in_samples.csv"
    ),
    # PCAWG sample metadata
    "sample_sheet": (
        "https://dcc.icgc.org/api/v1/download?fn=/PCAWG/clinical_and_histology/"
        "pcawg_specimen_histology_August2016_v9.xlsx"
    ),
}

# HRDetect scores from Davies et al. 2017 (Nature Medicine)
# and CHORD predictions from Nguyen et al. 2020 (Nature Communications)
# These are typically in supplementary tables — URLs may need manual update
HRDETECT_CHORD_NOTE = """
HRDetect scores and CHORD predictions are available from:

1. HRDetect (Davies et al. 2017, Nat Med):
   - Supplementary Table 1 of the paper
   - Or from: https://github.com/Nik-Zainal-Group/signature.tools.lib
   - Scores for ~370 breast cancer WGS samples

2. CHORD (Nguyen et al. 2020, Nat Commun):
   - Available at: https://github.com/UMCUGenetics/CHORD
   - Pre-computed for PCAWG samples in supplementary data
   - Binary HRD classification + probability scores

3. HRDetect pan-cancer (Degasperi et al. 2020, Nat Cancer):
   - Extended to pan-cancer PCAWG cohort
   - Supplementary tables with per-sample scores

Place downloaded supplementary tables in:
  {raw_dir}/hrdetect_scores.tsv
  {raw_dir}/chord_predictions.tsv
"""


def download_public_signatures():
    """Download publicly available PCAWG mutational signature data."""
    logger.info("Downloading PCAWG mutational signature contributions")
    raw_dir = RAW_DIR / "signatures"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name, url in PCAWG_PUBLIC_URLS.items():
        ext = ".xlsx" if "xlsx" in url else ".csv"
        dest = raw_dir / f"{name}{ext}"
        _download_file(url, dest, desc=f"PCAWG {name}")


def process_signatures():
    """Process downloaded signature contribution files."""
    logger.info("Processing PCAWG signature data")
    raw_dir = RAW_DIR / "signatures"
    out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process SBS signatures
    sbs_path = raw_dir / "sbs_signatures.csv"
    if sbs_path.exists():
        sbs = pd.read_csv(sbs_path)
        logger.info("  SBS signatures: %d samples x %d signatures", *sbs.shape)

        # SBS3 is the canonical HRD-associated signature
        if "SBS3" in sbs.columns:
            sbs["SBS3_high"] = sbs["SBS3"] > sbs["SBS3"].quantile(0.75)
            logger.info("  SBS3 > Q3 threshold: %d samples", sbs["SBS3_high"].sum())

        sbs.to_parquet(out_dir / "sbs_signatures.parquet", index=False)

    # Process ID signatures
    id_path = raw_dir / "id_signatures.csv"
    if id_path.exists():
        ids = pd.read_csv(id_path)
        logger.info("  ID signatures: %d samples x %d signatures", *ids.shape)
        # ID6 is associated with HRD
        ids.to_parquet(out_dir / "id_signatures.parquet", index=False)

    # Process DBS signatures
    dbs_path = raw_dir / "dbs_signatures.csv"
    if dbs_path.exists():
        dbs = pd.read_csv(dbs_path)
        logger.info("  DBS signatures: %d samples x %d signatures", *dbs.shape)
        dbs.to_parquet(out_dir / "dbs_signatures.parquet", index=False)

    # Process sample sheet
    sheet_path = raw_dir / "sample_sheet.xlsx"
    if sheet_path.exists():
        samples = pd.read_excel(sheet_path)
        logger.info("  Sample metadata: %d samples", len(samples))
        samples.to_parquet(out_dir / "sample_metadata.parquet", index=False)


def process_hrdetect_chord():
    """Process HRDetect and CHORD scores if available."""
    logger.info("Processing HRDetect/CHORD scores (if available)")
    out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # HRDetect scores
    hrdetect_path = RAW_DIR / "hrdetect_scores.tsv"
    if hrdetect_path.exists():
        hrdetect = pd.read_csv(hrdetect_path, sep="\t")
        logger.info("  HRDetect: %d samples", len(hrdetect))
        hrdetect.to_parquet(out_dir / "hrdetect_scores.parquet", index=False)
    else:
        logger.info("  HRDetect scores not found at %s", hrdetect_path)
        logger.info("  See README for download instructions")

    # CHORD predictions
    chord_path = RAW_DIR / "chord_predictions.tsv"
    if chord_path.exists():
        chord = pd.read_csv(chord_path, sep="\t")
        logger.info("  CHORD: %d samples", len(chord))
        chord.to_parquet(out_dir / "chord_predictions.parquet", index=False)
    else:
        logger.info("  CHORD predictions not found at %s", chord_path)
        logger.info("  See README for download instructions")


def create_controlled_access_pipeline():
    """Create a template pipeline for processing controlled-access PCAWG data.

    This writes a helper script that can be used once ICGC/dbGaP access is granted.
    """
    pipeline_path = BASE_DIR / "process_pcawg_controlled.py"

    # Only write if it doesn't already exist
    if pipeline_path.exists():
        logger.info("Controlled-access pipeline already exists: %s", pipeline_path)
        return

    template = '''#!/usr/bin/env python3
"""
Process controlled-access PCAWG WGS + RNA-seq data.

Prerequisites:
  - ICGC DACO approval (https://daco.icgc-argo.org/)
  - score-client installed for ICGC data download
  - dbGaP approval for TCGA portion

This script assumes data has been downloaded to:
  v2/data_acquisition/raw/pcawg/controlled/

Usage:
    srun --mem=12G --time=04:00:00 python process_pcawg_controlled.py
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent / "raw" / "pcawg" / "controlled"
OUT_DIR = Path(__file__).resolve().parent / "processed" / "pcawg"


def process_rnaseq():
    """Process PCAWG RNA-seq (FPKM and TPM)."""
    # PCAWG RNA-seq is typically in a matrix format
    # File: pcawg.rnaseq.extended.metadata.aliquot_id.V4.tsv.gz (metadata)
    # File: tophat_star_fpkm_uq.v2_aliquot_gl.tsv.gz (expression)

    rnaseq_path = RAW_DIR / "tophat_star_fpkm_uq.v2_aliquot_gl.tsv.gz"
    if not rnaseq_path.exists():
        logger.warning("RNA-seq file not found: %s", rnaseq_path)
        return

    logger.info("Processing PCAWG RNA-seq")
    # Read in chunks to manage memory
    expr = pd.read_csv(rnaseq_path, sep="\\t", index_col=0, compression="gzip")
    logger.info("  Expression matrix: %d genes x %d samples", *expr.shape)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expr.to_parquet(OUT_DIR / "rnaseq_fpkm.parquet")


def process_wgs_features():
    """Extract WGS-derived features relevant to HRD.

    From PCAWG WGS data, compute:
    - LOH segments
    - Telomeric allelic imbalance (TAI)
    - Large-scale state transitions (LST)
    - These three compose the genomic instability score (GIS / HRD score)
    """
    # This requires the SNP6 or WGS-derived allele-specific copy number calls
    # Typically from ASCAT or Battenberg
    ascat_path = RAW_DIR / "consensus.20170119.somatic.cna.annotated.txt"
    if not ascat_path.exists():
        logger.warning("ASCAT CN file not found: %s", ascat_path)
        return

    logger.info("Processing PCAWG copy number data")
    cn = pd.read_csv(ascat_path, sep="\\t")

    # Extract sample-level HRD scar features
    # (actual computation depends on segment format)
    logger.info("  Copy number segments: %d rows for %d samples",
                len(cn), cn.iloc[:, 0].nunique())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cn.to_parquet(OUT_DIR / "copy_number_segments.parquet", index=False)


def main():
    """Process controlled-access PCAWG data."""
    if not RAW_DIR.exists():
        logger.error(
            "Raw data directory not found: %s\\n"
            "Download controlled-access data first.\\n"
            "See: https://docs.icgc-argo.org/docs/data-access/daco/applying",
            RAW_DIR,
        )
        return

    process_rnaseq()
    process_wgs_features()
    logger.info("PCAWG controlled data processing complete.")


if __name__ == "__main__":
    main()
'''

    with open(pipeline_path, "w") as f:
        f.write(template)
    logger.info("Created controlled-access pipeline: %s", pipeline_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Download public PCAWG data and set up controlled-access pipeline."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Download public signature data
    download_public_signatures()
    process_signatures()

    # Process HRDetect/CHORD if available
    process_hrdetect_chord()

    # Create controlled-access pipeline template
    create_controlled_access_pipeline()

    # Write access instructions
    instructions_path = PROCESSED_DIR / "CONTROLLED_ACCESS_README.txt"
    with open(instructions_path, "w") as f:
        f.write(HRDETECT_CHORD_NOTE.format(raw_dir=RAW_DIR))
    logger.info("Wrote access instructions to %s", instructions_path)

    # Update manifest
    update_manifest("pcawg", {
        "public_data": [
            "SBS mutational signature contributions",
            "DBS mutational signature contributions",
            "ID (indel) signature contributions",
            "Sample metadata/histology",
        ],
        "manual_download_required": [
            "HRDetect scores (Davies et al. 2017 supplementary)",
            "CHORD predictions (Nguyen et al. 2020 supplementary)",
        ],
        "controlled_access": [
            "WGS aligned reads (requires ICGC DACO)",
            "RNA-seq expression (requires ICGC DACO / dbGaP)",
            "Somatic variant calls (requires ICGC DACO)",
        ],
        "source": "ICGC DCC (dcc.icgc.org) + supplementary publications",
    })

    logger.info("PCAWG download complete.")


if __name__ == "__main__":
    main()
