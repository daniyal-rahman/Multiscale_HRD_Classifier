#!/usr/bin/env python3
"""
Runner script for Experiment 4: POLQ as a simple HRD biomarker.
Tests data loading and core pipeline before full notebook execution.
"""
import os
import sys
import json
import gzip
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')

REPO_ROOT = Path('/home/dani/repos2/Multiscale_HRD_Classifier')
PROCESSED = REPO_ROOT / 'v2' / 'data_acquisition' / 'processed'
RAW_DIR = REPO_ROOT / 'v2' / 'data_acquisition' / 'raw'
GDC_API = 'https://api.gdc.cancer.gov'

print("="*60)
print("Experiment 4: POLQ Simple Biomarker - Data Validation")
print("="*60)

# --- 1. Gene annotation ---
print("\n[1] Building gene annotation...")
raw_rnaseq_dir = RAW_DIR / 'tcga' / 'TCGA-BRCA' / 'rnaseq'
tsv_files = sorted([f for f in os.listdir(raw_rnaseq_dir) if '.tsv' in f])
sample_file = raw_rnaseq_dir / tsv_files[0]

opener = gzip.open if str(sample_file).endswith('.gz') else open
with opener(sample_file, 'rt') as fh:
    gene_anno = pd.read_csv(fh, sep='\t', comment='#', index_col=0,
                            usecols=['gene_id', 'gene_name', 'gene_type'])

gene_anno = gene_anno.dropna(subset=['gene_name'])
protein_coding = gene_anno[gene_anno['gene_type'] == 'protein_coding']
ensembl_to_symbol = protein_coding['gene_name'].to_dict()
print(f'  {len(protein_coding)} protein-coding genes')

# Check target genes
target_genes = ['POLQ', 'BRCA1', 'BRCA2', 'RAD51', 'MRE11', 'NBN',
                'TDG', 'XPA', 'CHEK2', 'MAPKAPK2']
for g in target_genes:
    matches = [k for k, v in ensembl_to_symbol.items() if v == g]
    print(f'  {g}: {"FOUND" if matches else "MISSING"} ({matches})')

# --- 2. UUID mapping ---
print("\n[2] UUID to barcode mapping...")
uuid_cache = PROCESSED / 'tcga' / 'TCGA-BRCA' / 'uuid_to_barcode.parquet'

if uuid_cache.exists():
    uuid_map_df = pd.read_parquet(uuid_cache)
    print(f'  Loaded cached: {len(uuid_map_df)} entries')
else:
    print('  Querying GDC API...')
    filters = {
        'op': 'and',
        'content': [
            {'op': 'in', 'content': {'field': 'cases.project.project_id', 'value': ['TCGA-BRCA']}},
            {'op': 'in', 'content': {'field': 'files.data_category', 'value': ['Transcriptome Profiling']}},
            {'op': 'in', 'content': {'field': 'files.data_type', 'value': ['Gene Expression Quantification']}},
            {'op': 'in', 'content': {'field': 'files.analysis.workflow_type', 'value': ['STAR - Counts']}},
            {'op': 'in', 'content': {'field': 'cases.samples.sample_type', 'value': ['Primary Tumor']}},
        ]
    }

    all_hits = []
    offset = 0
    page_size = 500
    while True:
        params = {
            'filters': json.dumps(filters),
            'fields': 'file_id,file_name,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type',
            'format': 'JSON',
            'size': str(page_size),
            'from': str(offset),
        }
        resp = requests.get(f'{GDC_API}/files', params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()['data']
        hits = data['hits']
        if not hits:
            break
        all_hits.extend(hits)
        offset += page_size
        if offset >= data['pagination']['total']:
            break

    print(f'  Retrieved {len(all_hits)} file records from GDC')

    rows = []
    for h in all_hits:
        file_id = h['file_id']
        file_name = h.get('file_name', '')
        case_id = None
        sample_barcode = None
        try:
            case_id = h['cases'][0]['submitter_id']
            samples = h['cases'][0].get('samples', [])
            for s in samples:
                if s.get('sample_type') == 'Primary Tumor':
                    sample_barcode = s.get('submitter_id', '')
                    break
            if sample_barcode is None and samples:
                sample_barcode = samples[0].get('submitter_id', '')
        except (KeyError, IndexError):
            pass

        col_uuid = file_name.split('.')[0] if file_name else file_id
        rows.append({
            'file_id': file_id,
            'file_uuid_col': col_uuid,
            'case_id': case_id,
            'sample_barcode': sample_barcode,
        })

    uuid_map_df = pd.DataFrame(rows)
    uuid_map_df.to_parquet(uuid_cache, index=False)
    print(f'  Saved: {len(uuid_map_df)} entries')

# --- 3. FPKM matrix ---
print("\n[3] Loading FPKM matrix...")
fpkm_raw = pd.read_parquet(PROCESSED / 'tcga' / 'TCGA-BRCA' / 'rnaseq_fpkm.parquet')
print(f'  Raw: {fpkm_raw.shape}')

# Filter to protein-coding
fpkm_pc = fpkm_raw.loc[fpkm_raw.index.isin(ensembl_to_symbol.keys())].copy()
fpkm_pc.index = fpkm_pc.index.map(ensembl_to_symbol)

# Handle duplicates
fpkm_pc['_mean'] = fpkm_pc.mean(axis=1)
fpkm_pc = fpkm_pc.sort_values('_mean', ascending=False)
fpkm_pc = fpkm_pc[~fpkm_pc.index.duplicated(keep='first')]
fpkm_pc = fpkm_pc.drop(columns=['_mean'])
print(f'  Protein-coding: {fpkm_pc.shape}')

# Map columns (handle different column naming conventions in cached file)
uuid_col_1 = 'filename_uuid' if 'filename_uuid' in uuid_map_df.columns else 'file_uuid_col'
uuid_col_2 = 'gdc_file_id' if 'gdc_file_id' in uuid_map_df.columns else 'file_id'
col_to_case = dict(zip(uuid_map_df[uuid_col_1], uuid_map_df['case_id']))
col_to_case_2 = dict(zip(uuid_map_df[uuid_col_2], uuid_map_df['case_id']))

new_cols = []
for c in fpkm_pc.columns:
    mapped = col_to_case.get(c) or col_to_case_2.get(c)
    new_cols.append(mapped if mapped else c)

fpkm_pc.columns = new_cols
n_mapped = sum(1 for c in fpkm_pc.columns if str(c).startswith('TCGA-'))
print(f'  Mapped {n_mapped}/{fpkm_pc.shape[1]} columns to TCGA barcodes')

fpkm_pc = fpkm_pc.loc[:, fpkm_pc.columns.astype(str).str.startswith('TCGA-')]
fpkm_pc = fpkm_pc.loc[:, ~fpkm_pc.columns.duplicated(keep='first')]
print(f'  Final: {fpkm_pc.shape}')

# Check target genes in expression
for g in target_genes:
    if g in fpkm_pc.index:
        vals = fpkm_pc.loc[g]
        print(f'  {g}: median={vals.median():.2f}, range=[{vals.min():.2f}, {vals.max():.2f}]')
    else:
        print(f'  {g}: NOT IN EXPRESSION')

# --- 4. HRD scores ---
print("\n[4] Getting HRD scores...")
# Try downloading from PanCanAtlas / Marquard
hrd_cache = PROCESSED / 'tcga' / 'TCGA-BRCA' / 'hrd_scores.parquet'

if hrd_cache.exists():
    hrd_df = pd.read_parquet(hrd_cache)
    print(f'  Loaded cached: {hrd_df.shape}')
    print(f'  Columns: {hrd_df.columns.tolist()[:15]}')
else:
    # Try multiple sources for HRD scores
    hrd_df = None

    # Source 1: TCGA PanCanAtlas from GDC
    urls = [
        'https://api.gdc.cancer.gov/data/5e41b26d-9843-4f40-b5bd-787e03e318c7',
        'https://api.gdc.cancer.gov/data/40e0a5f0-09e7-4ce7-a9c7-30e8eca2a8a7',
        'https://api.gdc.cancer.gov/data/b57e5e89-7f72-4ae0-a703-45e2e42b411f',
    ]

    for url in urls:
        try:
            print(f'  Trying {url}...')
            resp = requests.get(url, timeout=120, allow_redirects=True)
            resp.raise_for_status()
            from io import StringIO, BytesIO
            content = resp.content
            text = resp.text

            # Try TSV
            try:
                df = pd.read_csv(StringIO(text), sep='\t')
                if len(df) > 10 and len(df.columns) > 2:
                    hrd_df = df
                    print(f'  Downloaded TSV: {hrd_df.shape}')
                    print(f'  Columns: {hrd_df.columns.tolist()[:15]}')
                    break
            except:
                pass

            # Try Excel
            try:
                df = pd.read_excel(BytesIO(content))
                if len(df) > 10:
                    hrd_df = df
                    print(f'  Downloaded Excel: {hrd_df.shape}')
                    print(f'  Columns: {hrd_df.columns.tolist()[:15]}')
                    break
            except:
                pass

            # Try CSV
            try:
                df = pd.read_csv(StringIO(text))
                if len(df) > 10 and len(df.columns) > 2:
                    hrd_df = df
                    print(f'  Downloaded CSV: {hrd_df.shape}')
                    print(f'  Columns: {hrd_df.columns.tolist()[:15]}')
                    break
            except:
                pass

        except Exception as e:
            print(f'  Failed: {e}')

    if hrd_df is None:
        print('  WARNING: Could not download HRD scores from GDC.')
        print('  Trying alternative approach...')

        # Source 2: Use cBioPortal clinical data
        try:
            print('  Trying cBioPortal...')
            cbio_url = 'https://www.cbioportal.org/api/studies/brca_tcga_pan_can_atlas_2018/clinical-data?clinicalDataType=SAMPLE&projection=DETAILED'
            resp = requests.get(cbio_url, timeout=60,
                              headers={'Accept': 'application/json'})
            resp.raise_for_status()
            cbio_data = resp.json()

            # Parse clinical data - look for HRD-related attributes
            from collections import defaultdict
            sample_data = defaultdict(dict)
            for entry in cbio_data:
                sid = entry.get('sampleId', '')
                attr = entry.get('clinicalAttributeId', '')
                val = entry.get('value', '')
                if attr in ['FRACTION_GENOME_ALTERED', 'ANEUPLOIDY_SCORE', 'MSI_SCORE_MANTIS',
                           'TMB_NONSYNONYMOUS', 'CANCER_TYPE_DETAILED']:
                    sample_data[sid][attr] = val

            if sample_data:
                hrd_df = pd.DataFrame.from_dict(sample_data, orient='index')
                hrd_df.index.name = 'sample'
                hrd_df = hrd_df.reset_index()
                print(f'  cBioPortal clinical data: {hrd_df.shape}')
                print(f'  Columns: {hrd_df.columns.tolist()}')
            else:
                print('  No data from cBioPortal.')
        except Exception as e:
            print(f'  cBioPortal failed: {e}')

    if hrd_df is not None and len(hrd_df) > 0:
        hrd_df.to_parquet(hrd_cache, index=False)

# Analyze HRD score availability
if hrd_df is not None:
    cols_lower = {c: c.lower() for c in hrd_df.columns}
    hrd_related = [c for c in hrd_df.columns if any(p in c.lower() for p in
                   ['hrd', 'scar', 'loh', 'tai', 'lst', 'telomeric', 'hrdeficiency'])]
    print(f'  HRD-related columns: {hrd_related}')

    sample_cols = [c for c in hrd_df.columns if any(p in c.lower() for p in
                   ['sample', 'barcode', 'patient', 'case', 'tcga'])]
    print(f'  Sample ID columns: {sample_cols}')

print("\n[DONE] Data validation complete.")
print("The notebook is ready to execute.")
