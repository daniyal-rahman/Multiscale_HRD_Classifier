# Data Acquisition

Scripts to download and process publicly available datasets for HRD research.

## Directory Structure

```
data_acquisition/
├── download_tcga.py       # TCGA-BRCA and TCGA-OV from GDC
├── download_geo.py        # GEO ovarian cohorts with platinum response
├── download_gdsc_ccle.py  # GDSC drug sensitivity + CCLE/DepMap expression
├── download_pcawg.py      # PCAWG pan-cancer signatures + HRDetect/CHORD
├── download_metabric.py   # METABRIC from cBioPortal
├── download_ispy2.py      # I-SPY2 validation data
├── _manifest.py           # Shared manifest tracking
├── manifest.json          # Auto-generated download log
├── raw/                   # Downloaded files (not committed)
└── processed/             # Standardized parquet outputs
```

## Scripts

### download_tcga.py — TCGA-BRCA and TCGA-OV

Downloads from the GDC API (no R dependencies):
- **RNA-seq**: STAR-Counts (FPKM + raw counts)
- **Clinical**: treatment, survival, platinum response (OV)
- **Mutations**: MAF files filtered for HRR genes (BRCA1/2, PALB2, RAD51C/D, ATM, etc.)
- **Copy number**: segment data for LOH/GIS calculation
- **Methylation**: BRCA1 promoter CpG probes

Filters for Primary Tumor samples only.

```bash
srun --mem=8G --time=02:00:00 python download_tcga.py
```

### download_geo.py — GEO Ovarian Cohorts

Downloads six ovarian cancer cohorts with platinum response annotations:

| Accession | N | Description |
|-----------|---|-------------|
| GSE9891 | 285 | Tothill et al. — subtypes + chemo response |
| GSE26712 | 185 | Bonome et al. — late-stage serous OvCa |
| GSE51088 | 172 | HGSOC with platinum response |
| GSE63885 | 101 | OvCa with platinum response (Poland) |
| GSE30161 | 58 | Advanced serous OvCa + platinum response |
| GSE32062 | 260 | Japanese HGSOC + survival data |

Extracts expression matrices, maps probes to gene symbols, and standardizes platinum response labels (sensitive/resistant/refractory/partial).

```bash
srun --mem=8G --time=02:00:00 python download_geo.py
```

### download_gdsc_ccle.py — Cell Line Drug Sensitivity

**GDSC** (cancerrxgene.org):
- IC50 values for olaparib, rucaparib, talazoparib, niraparib, cisplatin, carboplatin
- Cell line annotations

**DepMap/CCLE** (depmap.org):
- RNA-seq expression (TPM, log2(x+1))
- Somatic mutations
- CRISPR dependency scores for HRR genes

Cross-matches cell lines between GDSC and CCLE by name.

```bash
srun --mem=8G --time=01:00:00 python download_gdsc_ccle.py
```

**Note**: DepMap download URLs change quarterly. If downloads fail, get current URLs from https://depmap.org/portal/download/all/ and place files in `raw/gdsc_ccle/depmap/`.

### download_pcawg.py — PCAWG Pan-Cancer

**Public data** (downloaded automatically):
- SBS mutational signature contributions (SBS3 = HRD-associated)
- DBS and ID signature contributions
- Sample metadata

**Manual download required**:
- HRDetect scores (Davies et al. 2017 supplementary)
- CHORD predictions (Nguyen et al. 2020 supplementary)

**Controlled access** (requires ICGC DACO / dbGaP approval):
- WGS aligned reads and somatic variant calls
- RNA-seq expression

A template pipeline (`process_pcawg_controlled.py`) is generated for processing controlled-access data once approved.

```bash
srun --mem=4G --time=00:30:00 python download_pcawg.py
```

### download_metabric.py — METABRIC

Downloads from cBioPortal API (study: `brca_metabric`):
- Clinical data (patient + sample level)
- Expression data (microarray)
- CNV data (GISTIC discrete)
- Mutations for HRR genes

```bash
srun --mem=4G --time=00:30:00 python download_metabric.py
```

### download_ispy2.py — I-SPY2 Validation

Processes existing I-SPY2 data and downloads additional metadata from GEO:
- Pusztai et al. 2021 expression data
- GSE173839 durvalumab/olaparib arm biomarkers
- pCR (pathological complete response) endpoint

```bash
srun --mem=4G --time=00:30:00 python download_ispy2.py
```

**Note**: Full I-SPY2 trial data requires a DUA with the I-SPY2 consortium. See https://www.ispytrials.org/results/data-access.

## Output Format

All processed data is saved as **parquet** files in `processed/{dataset}/`. Common outputs:
- `expression.parquet` — gene x sample expression matrix
- `clinical.parquet` — sample-level clinical annotations
- `mutations_hrr_genes.parquet` — mutations in HRR pathway genes
- `cnv.parquet` or `cnv_segments.parquet` — copy number data

## Manifest

`manifest.json` is auto-updated by each script and tracks:
- Which datasets have been downloaded
- Timestamps
- Data type summaries

## Dependencies

```
pandas
requests
tqdm
GEOparse
openpyxl    # for reading GDSC .xlsx files
pyarrow     # for parquet I/O
```
