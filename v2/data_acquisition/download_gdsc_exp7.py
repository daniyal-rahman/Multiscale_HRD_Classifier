#!/usr/bin/env python3
"""
Experiment 7: GDSC Olaparib IC50 Validation Data Acquisition

Downloads and processes:
  1. GDSC2 drug sensitivity data (olaparib, rucaparib, talazoparib IC50/AUC)
  2. DepMap/CCLE RNA-seq expression (TPM log2(x+1))
  3. DepMap/CCLE somatic mutations (for BRCA1/2 status)
  4. Cell line metadata and cross-mapping

Raw files saved to:   raw/gdsc/
Processed files to:   processed/gdsc/

Usage:
    srun --mem=4G --time=00:30:00 python download_gdsc_exp7.py
"""

import io
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _manifest import update_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "gdsc"
PROCESSED_DIR = BASE_DIR / "processed" / "gdsc"

# Also check the old location for already-downloaded GDSC files
OLD_RAW_DIR = BASE_DIR / "raw" / "gdsc_ccle" / "gdsc"

MAX_RETRIES = 3

# ---- URLs ----

# GDSC2 drug sensitivity
GDSC_URLS = {
    "gdsc2_fitted": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx",
    "cell_line_details": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/Cell_Lines_Details.xlsx",
}

# DepMap 24Q4 from figshare (stable URLs)
DEPMAP_URLS = {
    "expression": "https://ndownloader.figshare.com/files/51065489",  # OmicsExpressionProteinCodingGenesTPMLogp1.csv
    "mutations": "https://ndownloader.figshare.com/files/51065732",   # OmicsSomaticMutations.csv
    "model_info": "https://ndownloader.figshare.com/files/51065297",  # Model.csv
}

DEPMAP_FILENAMES = {
    "expression": "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "mutations": "OmicsSomaticMutations.csv",
    "model_info": "Model.csv",
}

# PARPi drugs of interest
PARPI_DRUGS = ["Olaparib", "Rucaparib", "Talazoparib"]

# HRR pathway genes for BRCA status
BRCA_GENES = ["BRCA1", "BRCA2"]
HRR_GENES = [
    "BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D", "ATM",
    "ATR", "CHEK1", "CHEK2", "BARD1", "BRIP1", "FANCA",
    "FANCC", "FANCM", "RAD51B", "NBN", "MRE11", "CDK12",
]

# Track what we download for the report
download_log = []


def log_download(name, source, dest, size_bytes, shape=None, columns=None, notes=""):
    """Record a download for the report."""
    entry = {
        "name": name,
        "source": source,
        "destination": str(dest),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / 1024 / 1024, 2),
        "shape": str(shape) if shape else None,
        "columns": columns,
        "notes": notes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    download_log.append(entry)
    return entry


def _download_file(url, dest_path, desc=""):
    """Download a file with retry logic, streaming, and progress bar."""
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

            actual_size = os.path.getsize(dest_path)
            logger.info("  Downloaded %s (%.2f MB)", dest_path.name, actual_size / 1024 / 1024)
            return actual_size

        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = 5 * (attempt + 1)
                logger.warning("Download failed (attempt %d/%d): %s. Retrying in %ds...",
                               attempt + 1, MAX_RETRIES, e, wait)
                time.sleep(wait)
            else:
                logger.error("Download failed after %d attempts: %s", MAX_RETRIES, e)
                raise


def _safe_to_parquet(df, path):
    """Save DataFrame to parquet, handling mixed-type columns.

    Fixes ArrowTypeError: Expected bytes, got a 'int' object
    by converting problematic columns to string.
    """
    for col in df.columns:
        if df[col].dtype == object:
            # Check for mixed types
            types = df[col].dropna().apply(type).unique()
            if len(types) > 1:
                logger.info("  Converting mixed-type column '%s' to string (types: %s)",
                            col, [t.__name__ for t in types])
                df[col] = df[col].astype(str)
    df.to_parquet(path, index=False)
    logger.info("  Saved %s (%d rows x %d cols, %.2f MB)",
                path.name, df.shape[0], df.shape[1],
                os.path.getsize(path) / 1024 / 1024)


# ===========================================================================
# 1. GDSC2 Drug Sensitivity Data
# ===========================================================================

def download_gdsc_sensitivity():
    """Download GDSC2 dose-response data and filter for PARPi drugs."""
    logger.info("=" * 60)
    logger.info("STEP 1: GDSC2 Drug Sensitivity Data")
    logger.info("=" * 60)

    # --- Download raw GDSC2 dose-response ---
    gdsc_raw_path = RAW_DIR / "GDSC2_dose_response.xlsx"

    if gdsc_raw_path.exists():
        logger.info("  GDSC2 dose-response already exists at %s", gdsc_raw_path)
    elif (OLD_RAW_DIR / "GDSC2_fitted_dose_response.xlsx").exists():
        # Copy from old location
        src = OLD_RAW_DIR / "GDSC2_fitted_dose_response.xlsx"
        shutil.copy2(src, gdsc_raw_path)
        logger.info("  Copied GDSC2 dose-response from previous download: %s", src)
    else:
        _download_file(GDSC_URLS["gdsc2_fitted"], gdsc_raw_path, desc="GDSC2 dose-response")

    size = os.path.getsize(gdsc_raw_path)
    log_download(
        "GDSC2 Fitted Dose Response",
        GDSC_URLS["gdsc2_fitted"],
        gdsc_raw_path, size,
        notes="GDSC2 release 8.5 (27 Oct 2023). All drug-cell line IC50/AUC pairs."
    )

    # --- Download cell line details ---
    cell_lines_raw_path = RAW_DIR / "Cell_Lines_Details.xlsx"

    if cell_lines_raw_path.exists():
        logger.info("  Cell line details already exists at %s", cell_lines_raw_path)
    elif (OLD_RAW_DIR / "Cell_Lines_Details.xlsx").exists():
        src = OLD_RAW_DIR / "Cell_Lines_Details.xlsx"
        shutil.copy2(src, cell_lines_raw_path)
        logger.info("  Copied Cell_Lines_Details from previous download: %s", src)
    else:
        _download_file(GDSC_URLS["cell_line_details"], cell_lines_raw_path, desc="Cell line details")

    size = os.path.getsize(cell_lines_raw_path)
    log_download(
        "GDSC Cell Line Details",
        GDSC_URLS["cell_line_details"],
        cell_lines_raw_path, size,
        notes="Cell line annotations (tissue, cancer type, COSMIC ID)."
    )

    # --- Process: read full dose-response ---
    logger.info("  Reading GDSC2 dose-response Excel file...")
    gdsc_df = pd.read_excel(gdsc_raw_path)
    logger.info("  Full GDSC2 dataset: %d rows x %d cols", gdsc_df.shape[0], gdsc_df.shape[1])
    logger.info("  Columns: %s", list(gdsc_df.columns))

    # Check which drug name column exists
    drug_col = None
    for candidate in ["DRUG_NAME", "Drug Name", "drug_name"]:
        if candidate in gdsc_df.columns:
            drug_col = candidate
            break
    if drug_col is None:
        # Try to find a column containing drug info
        logger.warning("  No standard drug name column found. Columns: %s", list(gdsc_df.columns))
        drug_col = gdsc_df.columns[4]  # heuristic fallback
        logger.warning("  Using column '%s' as drug name", drug_col)

    # Show all unique drugs containing 'olaparib', 'rucaparib', 'talazoparib'
    all_drugs = gdsc_df[drug_col].unique()
    for drug in PARPI_DRUGS:
        matches = [d for d in all_drugs if drug.lower() in str(d).lower()]
        logger.info("  Drug matches for '%s': %s", drug, matches)

    # Filter for PARPi drugs
    parpi_mask = gdsc_df[drug_col].str.lower().isin([d.lower() for d in PARPI_DRUGS])
    parpi_df = gdsc_df[parpi_mask].copy()
    logger.info("  PARPi entries: %d rows", len(parpi_df))

    if len(parpi_df) == 0:
        logger.warning("  No PARPi entries found! Checking drug list...")
        logger.info("  Available drugs (sample): %s", sorted(all_drugs)[:50])

    # --- Save olaparib IC50 ---
    olaparib_mask = gdsc_df[drug_col].str.lower() == "olaparib"
    olaparib_df = gdsc_df[olaparib_mask].copy()
    logger.info("  Olaparib entries: %d rows", len(olaparib_df))

    _safe_to_parquet(olaparib_df, PROCESSED_DIR / "olaparib_ic50.parquet")
    log_download(
        "Olaparib IC50 (filtered)",
        "Filtered from GDSC2_dose_response.xlsx",
        PROCESSED_DIR / "olaparib_ic50.parquet",
        os.path.getsize(PROCESSED_DIR / "olaparib_ic50.parquet"),
        shape=olaparib_df.shape,
        columns=list(olaparib_df.columns),
        notes=f"Olaparib only. {len(olaparib_df)} cell lines."
    )

    # Also save full PARPi data for reference
    if len(parpi_df) > 0:
        _safe_to_parquet(parpi_df, PROCESSED_DIR / "parpi_ic50.parquet")

    return gdsc_df, olaparib_df


# ===========================================================================
# 2. DepMap Expression Data
# ===========================================================================

def download_expression():
    """Download CCLE/DepMap RNA-seq expression data."""
    logger.info("=" * 60)
    logger.info("STEP 2: DepMap Expression Data (RNA-seq TPM)")
    logger.info("=" * 60)

    raw_path = RAW_DIR / DEPMAP_FILENAMES["expression"]

    if raw_path.exists():
        logger.info("  Expression file already exists: %s", raw_path)
    else:
        logger.info("  Downloading DepMap 24Q4 expression data (~300 MB)...")
        _download_file(DEPMAP_URLS["expression"], raw_path,
                       desc="DepMap expression (TPM log2(x+1))")

    size = os.path.getsize(raw_path)
    log_download(
        "DepMap Expression (RNA-seq TPM)",
        DEPMAP_URLS["expression"],
        raw_path, size,
        notes="OmicsExpressionProteinCodingGenesTPMLogp1.csv from DepMap 24Q4. log2(TPM+1) values."
    )

    # Process expression data
    logger.info("  Reading expression CSV...")
    expr = pd.read_csv(raw_path, index_col=0)
    logger.info("  Expression matrix: %d cell lines x %d genes", *expr.shape)

    # Columns are "GENE_NAME (ENTREZ_ID)" - extract gene names
    original_cols = expr.columns.tolist()
    gene_names = [c.split(" (")[0] for c in expr.columns]
    expr.columns = gene_names
    logger.info("  Gene name examples: %s", gene_names[:5])
    logger.info("  Index (ModelID) examples: %s", expr.index[:5].tolist())

    # Save as parquet
    expr.to_parquet(PROCESSED_DIR / "expression.parquet")
    size = os.path.getsize(PROCESSED_DIR / "expression.parquet")
    log_download(
        "Expression matrix (processed)",
        "Processed from OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        PROCESSED_DIR / "expression.parquet", size,
        shape=expr.shape,
        notes="Index: ModelID (DepMap). Columns: gene symbols. Values: log2(TPM+1)."
    )

    return expr


# ===========================================================================
# 3. DepMap Mutation Data + BRCA Status
# ===========================================================================

def download_mutations():
    """Download CCLE/DepMap somatic mutation data and extract BRCA1/2 status."""
    logger.info("=" * 60)
    logger.info("STEP 3: DepMap Somatic Mutations + BRCA Status")
    logger.info("=" * 60)

    raw_path = RAW_DIR / DEPMAP_FILENAMES["mutations"]

    if raw_path.exists():
        logger.info("  Mutations file already exists: %s", raw_path)
    else:
        logger.info("  Downloading DepMap 24Q4 somatic mutations (~600 MB)...")
        _download_file(DEPMAP_URLS["mutations"], raw_path,
                       desc="DepMap somatic mutations")

    size = os.path.getsize(raw_path)
    log_download(
        "DepMap Somatic Mutations",
        DEPMAP_URLS["mutations"],
        raw_path, size,
        notes="OmicsSomaticMutations.csv from DepMap 24Q4. All somatic variants."
    )

    # Process mutations
    logger.info("  Reading mutations CSV (this may take a while for large files)...")
    mut = pd.read_csv(raw_path)
    logger.info("  Mutations: %d rows x %d cols", mut.shape[0], mut.shape[1])
    logger.info("  Columns: %s", list(mut.columns))

    # Find the gene symbol column
    gene_col = None
    for candidate in ["HugoSymbol", "Hugo_Symbol", "gene", "Gene"]:
        if candidate in mut.columns:
            gene_col = candidate
            break
    if gene_col is None:
        logger.warning("  Could not find gene symbol column. Columns: %s", list(mut.columns))
        # Try first column that contains "hugo" or "gene" (case-insensitive)
        for col in mut.columns:
            if "hugo" in col.lower() or "gene" in col.lower():
                gene_col = col
                break

    logger.info("  Using gene column: %s", gene_col)

    # Find the model ID column
    model_col = None
    for candidate in ["ModelID", "DepMap_ID", "model_id"]:
        if candidate in mut.columns:
            model_col = candidate
            break
    if model_col is None:
        for col in mut.columns:
            if "model" in col.lower() or "depmap" in col.lower():
                model_col = col
                break

    logger.info("  Using model column: %s", model_col)

    # Filter for HRR genes
    if gene_col:
        hrr_mut = mut[mut[gene_col].isin(HRR_GENES)].copy()
        logger.info("  HRR gene mutations: %d rows", len(hrr_mut))

        brca_mut = mut[mut[gene_col].isin(BRCA_GENES)].copy()
        logger.info("  BRCA1/2 mutations: %d rows", len(brca_mut))
    else:
        hrr_mut = mut.copy()
        brca_mut = mut.copy()
        logger.warning("  Cannot filter by gene - saving all mutations")

    # Save full mutations
    _safe_to_parquet(mut, PROCESSED_DIR / "mutations.parquet")
    log_download(
        "Mutations (processed)",
        "Processed from OmicsSomaticMutations.csv",
        PROCESSED_DIR / "mutations.parquet",
        os.path.getsize(PROCESSED_DIR / "mutations.parquet"),
        shape=mut.shape,
        notes="All somatic mutations."
    )

    # --- Create BRCA status table ---
    if gene_col and model_col:
        logger.info("  Building BRCA1/2 mutation status per cell line...")

        # Get all unique cell lines
        all_models = mut[model_col].unique()

        # For each cell line, check if BRCA1 or BRCA2 is mutated
        brca_status_rows = []
        for model_id in all_models:
            model_muts = mut[mut[model_col] == model_id]
            if gene_col:
                model_genes = set(model_muts[gene_col].values)
            else:
                model_genes = set()

            brca_status_rows.append({
                "ModelID": model_id,
                "BRCA1_mutated": "BRCA1" in model_genes,
                "BRCA2_mutated": "BRCA2" in model_genes,
                "BRCA_any_mutated": "BRCA1" in model_genes or "BRCA2" in model_genes,
            })

        brca_status = pd.DataFrame(brca_status_rows)
        logger.info("  BRCA status: %d cell lines", len(brca_status))
        logger.info("  BRCA1 mutated: %d", brca_status["BRCA1_mutated"].sum())
        logger.info("  BRCA2 mutated: %d", brca_status["BRCA2_mutated"].sum())
        logger.info("  Any BRCA mutated: %d", brca_status["BRCA_any_mutated"].sum())

        brca_status.to_parquet(PROCESSED_DIR / "brca_status.parquet", index=False)
        log_download(
            "BRCA1/2 Status (derived)",
            "Derived from OmicsSomaticMutations.csv",
            PROCESSED_DIR / "brca_status.parquet",
            os.path.getsize(PROCESSED_DIR / "brca_status.parquet"),
            shape=brca_status.shape,
            columns=list(brca_status.columns),
            notes=f"BRCA1 mut: {brca_status['BRCA1_mutated'].sum()}, "
                  f"BRCA2 mut: {brca_status['BRCA2_mutated'].sum()}, "
                  f"Any: {brca_status['BRCA_any_mutated'].sum()}"
        )
    else:
        logger.warning("  Cannot build BRCA status - missing gene or model column")
        brca_status = None

    return mut, brca_status


# ===========================================================================
# 4. Cell Line Metadata
# ===========================================================================

def download_metadata():
    """Download DepMap Model.csv and GDSC cell line details for metadata."""
    logger.info("=" * 60)
    logger.info("STEP 4: Cell Line Metadata")
    logger.info("=" * 60)

    # --- DepMap Model.csv ---
    raw_path = RAW_DIR / DEPMAP_FILENAMES["model_info"]

    if raw_path.exists():
        logger.info("  Model.csv already exists: %s", raw_path)
    else:
        logger.info("  Downloading DepMap 24Q4 Model.csv...")
        _download_file(DEPMAP_URLS["model_info"], raw_path, desc="DepMap Model.csv")

    size = os.path.getsize(raw_path)
    log_download(
        "DepMap Model Info",
        DEPMAP_URLS["model_info"],
        raw_path, size,
        notes="Model.csv from DepMap 24Q4. Cell line metadata incl. tissue, cancer type."
    )

    # Process
    logger.info("  Reading Model.csv...")
    model_df = pd.read_csv(raw_path)
    logger.info("  Model info: %d cell lines x %d columns", *model_df.shape)
    logger.info("  Columns: %s", list(model_df.columns))

    # --- GDSC cell line details ---
    gdsc_cl_path = RAW_DIR / "Cell_Lines_Details.xlsx"
    gdsc_cl_df = None
    if gdsc_cl_path.exists():
        logger.info("  Reading GDSC Cell_Lines_Details.xlsx...")
        try:
            gdsc_cl_df = pd.read_excel(gdsc_cl_path)
            logger.info("  GDSC cell lines: %d rows x %d cols", *gdsc_cl_df.shape)
            logger.info("  Columns: %s", list(gdsc_cl_df.columns))
        except Exception as e:
            logger.warning("  Failed to read Cell_Lines_Details.xlsx: %s", e)

    # Build combined metadata
    # Key columns from Model.csv
    keep_cols_model = []
    for col_name in ["ModelID", "CellLineName", "StrippedCellLineName",
                     "OncotreeCode", "OncotreeSubtype", "OncotreePrimaryDisease",
                     "OncotreeLineage", "CatalogNumber", "PlateCoating",
                     "COSMICID", "LegacySubSubtype",
                     "PrimaryOrMetastasis", "SampleCollectionSite",
                     "Sex", "Age", "GrowthPattern"]:
        if col_name in model_df.columns:
            keep_cols_model.append(col_name)

    metadata = model_df[keep_cols_model].copy()
    logger.info("  Metadata subset: %d rows x %d cols", *metadata.shape)

    # Save
    _safe_to_parquet(metadata, PROCESSED_DIR / "cell_line_metadata.parquet")
    log_download(
        "Cell Line Metadata (processed)",
        "Processed from Model.csv + Cell_Lines_Details.xlsx",
        PROCESSED_DIR / "cell_line_metadata.parquet",
        os.path.getsize(PROCESSED_DIR / "cell_line_metadata.parquet"),
        shape=metadata.shape,
        columns=list(metadata.columns),
        notes="Combined DepMap + GDSC cell line annotations."
    )

    if gdsc_cl_df is not None:
        _safe_to_parquet(gdsc_cl_df, PROCESSED_DIR / "gdsc_cell_line_details.parquet")

    return metadata, model_df


# ===========================================================================
# 5. Cross-mapping and Summary
# ===========================================================================

def cross_map_gdsc_depmap(olaparib_df, model_df, brca_status):
    """Map GDSC cell line names to DepMap ModelIDs."""
    logger.info("=" * 60)
    logger.info("STEP 5: Cross-mapping GDSC <-> DepMap")
    logger.info("=" * 60)

    if olaparib_df is None or model_df is None:
        logger.warning("  Missing data for cross-mapping")
        return

    # Find the cell line name column in GDSC
    cl_col = None
    for candidate in ["CELL_LINE_NAME", "Cell Line Name", "cell_line_name"]:
        if candidate in olaparib_df.columns:
            cl_col = candidate
            break

    if cl_col is None:
        logger.warning("  Cannot find cell line name column in GDSC data")
        return

    # Clean names for matching
    def clean_name(s):
        return str(s).upper().replace("-", "").replace("_", "").replace(" ", "").replace(".", "")

    # Build DepMap lookup
    depmap_name_col = None
    for candidate in ["StrippedCellLineName", "CellLineName"]:
        if candidate in model_df.columns:
            depmap_name_col = candidate
            break

    if depmap_name_col is None:
        logger.warning("  Cannot find cell line name in DepMap Model.csv")
        return

    model_df["_clean_name"] = model_df[depmap_name_col].apply(clean_name)
    olaparib_df = olaparib_df.copy()
    olaparib_df["_clean_name"] = olaparib_df[cl_col].apply(clean_name)

    # Merge
    merged = olaparib_df.merge(
        model_df[["ModelID", "_clean_name"]].drop_duplicates(subset="_clean_name"),
        on="_clean_name",
        how="left",
    )

    n_matched = merged["ModelID"].notna().sum()
    n_total = len(merged)
    logger.info("  Matched %d/%d olaparib cell lines to DepMap ModelIDs (%.1f%%)",
                n_matched, n_total, 100 * n_matched / max(n_total, 1))

    # Add BRCA status if available
    if brca_status is not None:
        merged = merged.merge(brca_status, on="ModelID", how="left")
        n_brca = merged["BRCA_any_mutated"].sum()
        logger.info("  Of matched: %d have BRCA1/2 mutations", int(n_brca) if pd.notna(n_brca) else 0)

    # Drop temporary column
    merged = merged.drop(columns=["_clean_name"], errors="ignore")

    # Save enriched olaparib data
    _safe_to_parquet(merged, PROCESSED_DIR / "olaparib_ic50_enriched.parquet")
    logger.info("  Saved enriched olaparib data with DepMap IDs and BRCA status")

    return merged


# ===========================================================================
# 6. Download Report
# ===========================================================================

def write_download_report():
    """Write a markdown report of everything downloaded."""
    logger.info("=" * 60)
    logger.info("STEP 6: Writing Download Report")
    logger.info("=" * 60)

    report_path = PROCESSED_DIR / "download_report.md"

    lines = [
        "# GDSC Olaparib IC50 Validation - Data Acquisition Report",
        "",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Experiment**: 7 (GDSC Olaparib IC50 Validation)",
        "",
        "## Downloads",
        "",
    ]

    total_size = 0
    for entry in download_log:
        total_size += entry["size_bytes"]
        lines.append(f"### {entry['name']}")
        lines.append(f"- **Source**: `{entry['source']}`")
        lines.append(f"- **File**: `{entry['destination']}`")
        lines.append(f"- **Size**: {entry['size_mb']} MB")
        if entry.get("shape"):
            lines.append(f"- **Shape**: {entry['shape']}")
        if entry.get("columns"):
            cols_str = ", ".join(entry["columns"][:15])
            if len(entry["columns"]) > 15:
                cols_str += f", ... ({len(entry['columns'])} total)"
            lines.append(f"- **Columns**: {cols_str}")
        if entry.get("notes"):
            lines.append(f"- **Notes**: {entry['notes']}")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total files**: {len(download_log)}")
    lines.append(f"- **Total size**: {round(total_size / 1024 / 1024, 2)} MB")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")

    # List all files in raw and processed dirs
    for label, dirpath in [("Raw", RAW_DIR), ("Processed", PROCESSED_DIR)]:
        lines.append(f"### {label} (`{dirpath}`)")
        if dirpath.exists():
            for f in sorted(dirpath.iterdir()):
                if f.is_file():
                    sz = os.path.getsize(f) / 1024 / 1024
                    lines.append(f"- `{f.name}` ({sz:.2f} MB)")
        lines.append("")

    lines.append("## Data Sources")
    lines.append("")
    lines.append("- GDSC2: https://www.cancerrxgene.org/downloads/bulk_download (Release 8.5)")
    lines.append("- DepMap: https://depmap.org/portal/download/all/ (24Q4 release)")
    lines.append("- Figshare: https://plus.figshare.com/articles/dataset/DepMap_24Q4_Public/27993248")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("  Report written to %s", report_path)


# ===========================================================================
# Main
# ===========================================================================

def main():
    """Download and process all data for Experiment 7."""
    logger.info("Starting Experiment 7 data acquisition: GDSC olaparib IC50 validation")
    logger.info("Raw dir: %s", RAW_DIR)
    logger.info("Processed dir: %s", PROCESSED_DIR)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. GDSC drug sensitivity
    gdsc_df, olaparib_df = download_gdsc_sensitivity()

    # 2. Expression
    expr = download_expression()

    # 3. Mutations + BRCA status
    mut, brca_status = download_mutations()

    # 4. Cell line metadata
    metadata, model_df = download_metadata()

    # 5. Cross-map
    cross_map_gdsc_depmap(olaparib_df, model_df, brca_status)

    # 6. Report
    write_download_report()

    # Update manifest
    update_manifest("gdsc_exp7", {
        "description": "Experiment 7: GDSC olaparib IC50 validation data",
        "drugs": PARPI_DRUGS,
        "gdsc_source": "cancerrxgene.org GDSC2 release 8.5 (27 Oct 2023)",
        "depmap_source": "DepMap 24Q4 (figshare article 27993248)",
        "olaparib_cell_lines": len(olaparib_df) if olaparib_df is not None else 0,
        "expression_shape": list(expr.shape) if expr is not None else None,
        "brca_mutated_lines": int(brca_status["BRCA_any_mutated"].sum()) if brca_status is not None else None,
    })

    logger.info("=" * 60)
    logger.info("Experiment 7 data acquisition COMPLETE")
    logger.info("=" * 60)

    # Final summary
    logger.info("Output files:")
    for dirpath in [RAW_DIR, PROCESSED_DIR]:
        if dirpath.exists():
            for f in sorted(dirpath.iterdir()):
                if f.is_file():
                    sz = os.path.getsize(f) / 1024 / 1024
                    logger.info("  %s (%.2f MB)", f, sz)


if __name__ == "__main__":
    main()
