#!/usr/bin/env python3
"""
TCGA-OV WES Analysis: Genomic Features for Drug Response Prediction

Downloads TCGA-OV mutation and copy number data from cBioPortal,
computes WES-derived features, and tests whether they predict
platinum response — alone and combined with our RNA model scores.

Steps:
  1. Download somatic mutations from cBioPortal (all three OV studies)
  2. Download copy number alterations (GISTIC)
  3. Compute per-patient WES features: TMB, BRCA status, key gene muts, CCNE1 amp
  4. Cross-reference with RNA model predictions (LODO-CV scores)
  5. Test WES features → response (logistic regression)
  6. Test WES + RNA combined → response
  7. Report results

Output: addendum/tcga_wes_results.json, addendum/tcga_wes_analysis.md
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import json
import requests
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, mannwhitneyu, fisher_exact
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────
EXP = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
DATA = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
OUT = EXP / "addendum"
OUT.mkdir(parents=True, exist_ok=True)

CBIO_BASE = "https://www.cbioportal.org/api"
STUDY_IDS = ["ov_tcga_pub", "ov_tcga_pan_can_atlas_2018", "ov_tcga"]

# Key genes for ovarian cancer WES features
HRR_GENES = ["BRCA1", "BRCA2", "ATM", "ATR", "PALB2", "RAD51B", "RAD51C",
             "RAD51D", "BRIP1", "FANCA", "FANCC", "FANCG", "CDK12",
             "CHEK1", "CHEK2", "BARD1", "NBN", "RAD50", "MRE11"]
RESISTANCE_GENES = ["NF1", "RB1", "CCNE1", "KRAS", "MYC", "PIK3CA", "PTEN"]
ALL_KEY_GENES = list(set(HRR_GENES + RESISTANCE_GENES + ["TP53", "CSMD3", "FAT3", "USH2A"]))

print("=" * 72)
print("TCGA-OV WES Analysis: Genomic Features for Drug Response Prediction")
print("=" * 72)

# ══════════════════════════════════════════════════════════════════
# 1. Load our RNA model data (response labels + predictions)
# ══════════════════════════════════════════════════════════════════
print("\n[1/7] Loading RNA model data...")

pooled = pd.read_parquet(EXP / 'validation_fixed' / 'pooled_rank_matrix_fixed.parquet')
meta_cols = ['response_binary', 'response_continuous', 'drug', 'dataset', 'category', 'cancer_type']
gene_cols = [c for c in pooled.columns if c not in meta_cols]

# Model A filter
ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (pooled['category'] != 'cell_line') &
    ~((pooled['category'] == 'clinical_ispy2') & (~pooled['drug'].isin(ispy2_parpi_drugs)))
)
model_a = pooled[model_a_mask].copy()

# Run LODO-CV to get TCGA-OV predictions
X_all = model_a[gene_cols].values
y_all = model_a['response_binary'].values.astype(int)
datasets_all = model_a['dataset'].values
sample_ids_all = model_a.index.values
X_all = np.nan_to_num(X_all, nan=0.5)

# Just need TCGA-OV fold
tcga_mask = datasets_all == 'TCGA-OV'
train_mask = ~tcga_mask

clf_rna = LogisticRegression(
    penalty='l2', C=0.01, solver='lbfgs',
    max_iter=5000, class_weight='balanced', random_state=42
)
clf_rna.fit(X_all[train_mask], y_all[train_mask])
rna_scores = clf_rna.predict_proba(X_all[tcga_mask])[:, 1]

tcga_ov = model_a[tcga_mask].copy()
tcga_ov['rna_score'] = rna_scores
rna_auc = roc_auc_score(tcga_ov['response_binary'].values, rna_scores)

print(f"  TCGA-OV: {len(tcga_ov)} patients, RNA model AUC = {rna_auc:.4f}")
print(f"  Response: {tcga_ov['response_binary'].sum()} sensitive, {(~tcga_ov['response_binary'].astype(bool)).sum()} resistant")

# ══════════════════════════════════════════════════════════════════
# 2. Download TCGA-OV mutations from cBioPortal
# ══════════════════════════════════════════════════════════════════
print("\n[2/7] Downloading TCGA-OV mutation data from cBioPortal...")

def fetch_cbio(endpoint, params=None):
    """Fetch from cBioPortal API with retry."""
    url = f"{CBIO_BASE}/{endpoint}"
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"    Retry {attempt+1}/3: {e}")
            time.sleep(2)
    return None

# Use PanCancer Atlas (most comprehensive)
study_id = "ov_tcga_pan_can_atlas_2018"
profiles = fetch_cbio(f"studies/{study_id}/molecular-profiles")

mut_profile_id = None
cn_profile_id = None
for p in profiles:
    alt_type = p.get('molecularAlterationType', '')
    if alt_type == 'MUTATION_EXTENDED':
        mut_profile_id = p['molecularProfileId']
    elif alt_type == 'COPY_NUMBER_ALTERATION' and 'gistic' in p['molecularProfileId'].lower():
        cn_profile_id = p['molecularProfileId']
    elif alt_type == 'COPY_NUMBER_ALTERATION' and cn_profile_id is None:
        cn_profile_id = p['molecularProfileId']

print(f"  Mutation profile: {mut_profile_id}")
print(f"  CN profile: {cn_profile_id}")

# Get sample list
sample_list = fetch_cbio(f"sample-lists/{study_id}_all")
n_total_samples = sample_list.get('sampleCount', 'unknown') if sample_list else 'unknown'
print(f"  Total samples in study: {n_total_samples}")

# Fetch mutations for all key genes via POST
print("  Fetching mutations for key genes via POST...")
all_muts_list = []

# Use the mutations endpoint with gene filtering
for gene in ALL_KEY_GENES:
    url = f"{CBIO_BASE}/molecular-profiles/{mut_profile_id}/mutations"
    params = {
        'sampleListId': f"{study_id}_all",
    }
    # Try fetching by gene hugo symbol
    try:
        # First get entrezGeneId
        gene_info = fetch_cbio(f"genes/{gene}")
        if gene_info and 'entrezGeneId' in gene_info:
            entrez_id = gene_info['entrezGeneId']
            muts = fetch_cbio(
                f"molecular-profiles/{mut_profile_id}/mutations",
                params={'sampleListId': f"{study_id}_all", 'entrezGeneId': entrez_id}
            )
            if muts:
                for m in muts:
                    patient_id = m.get('patientId', '')
                    all_muts_list.append({
                        'patient_id': patient_id,
                        'gene': gene,
                        'mutation_type': m.get('mutationType', ''),
                        'variant_type': m.get('variantType', ''),
                        'protein_change': m.get('proteinChange', ''),
                        'mutation_status': m.get('mutationStatus', ''),
                        'variant_allele_freq': m.get('tumorAltCount', 0) / max(m.get('tumorRefCount', 0) + m.get('tumorAltCount', 1), 1),
                        'functional_impact': m.get('functionalImpactScore', ''),
                    })
                if len(muts) > 0:
                    print(f"    {gene}: {len(muts)} mutations")
    except Exception as e:
        print(f"    {gene}: error - {e}")

muts_df = pd.DataFrame(all_muts_list)
print(f"  Total key-gene mutations: {len(muts_df)}")

# ══════════════════════════════════════════════════════════════════
# 3. Download copy number data from cBioPortal
# ══════════════════════════════════════════════════════════════════
print("\n[3/7] Downloading TCGA-OV copy number data...")

cn_data = {}
cn_genes = ["CCNE1", "KRAS", "MYC", "BRCA1", "BRCA2", "RB1", "NF1", "PTEN", "PIK3CA", "CDK12"]
for gene in cn_genes:
    try:
        gene_info = fetch_cbio(f"genes/{gene}")
        if gene_info and cn_profile_id:
            entrez_id = gene_info['entrezGeneId']
            cn_vals = fetch_cbio(
                f"molecular-profiles/{cn_profile_id}/molecular-data",
                params={'sampleListId': f"{study_id}_all", 'entrezGeneId': entrez_id}
            )
            if cn_vals:
                for v in cn_vals:
                    pid = v.get('patientId', '')
                    val = v.get('value', 0)
                    if pid not in cn_data:
                        cn_data[pid] = {}
                    cn_data[pid][gene] = val
                print(f"    {gene}: {len(cn_vals)} CN values")
    except Exception as e:
        print(f"    {gene}: error - {e}")

# Also try to get TMB data - fetch all mutations (not just key genes)
print("\n  Fetching total mutation counts for TMB...")

# Get clinical data which may include TMB
clinical_data = fetch_cbio(f"studies/{study_id}/clinical-data", params={'clinicalDataType': 'SAMPLE'})
tmb_data = {}
if clinical_data:
    for item in clinical_data:
        pid = item.get('patientId', '')
        attr_id = item.get('clinicalAttributeId', '')
        if attr_id in ['MUTATION_COUNT', 'TMB_NONSYNONYMOUS']:
            try:
                tmb_data[pid] = float(item.get('value', 0))
            except (ValueError, TypeError):
                pass
    print(f"  TMB data from clinical attributes: {len(tmb_data)} patients")

# If TMB not in clinical data, count from mutation data
if len(tmb_data) < 100:
    print("  TMB not in clinical data, will compute from mutation counts...")
    # Fetch ALL mutations (not just key genes) - use sample-level mutation count
    # This is expensive, so let's use a more efficient approach
    # Fetch sample mutation counts
    try:
        url = f"{CBIO_BASE}/studies/{study_id}/clinical-data"
        params = {'clinicalDataType': 'SAMPLE', 'projection': 'SUMMARY'}
        all_clinical = fetch_cbio(f"studies/{study_id}/clinical-data",
                                  params={'clinicalDataType': 'SAMPLE'})
        if all_clinical:
            for item in all_clinical:
                pid = item.get('patientId', '')
                attr_id = item.get('clinicalAttributeId', '')
                if 'MUTATION' in attr_id.upper() or 'TMB' in attr_id.upper():
                    try:
                        tmb_data[pid] = float(item.get('value', 0))
                    except (ValueError, TypeError):
                        pass
            print(f"  Updated TMB data: {len(tmb_data)} patients")
    except Exception as e:
        print(f"  TMB fetch error: {e}")

# ══════════════════════════════════════════════════════════════════
# 4. Build WES feature matrix
# ══════════════════════════════════════════════════════════════════
print("\n[4/7] Building WES feature matrix...")

# Get our TCGA-OV patient IDs
our_patients = list(tcga_ov.index.astype(str))
print(f"  Our TCGA-OV patients: {len(our_patients)}")

# Build mutation features
wes_features = pd.DataFrame(index=our_patients)

# BRCA1/2 status
if len(muts_df) > 0:
    brca1_patients = set(muts_df[muts_df['gene'] == 'BRCA1']['patient_id'].unique())
    brca2_patients = set(muts_df[muts_df['gene'] == 'BRCA2']['patient_id'].unique())
    brca_any_patients = brca1_patients | brca2_patients

    wes_features['brca1_mut'] = [1 if p in brca1_patients else 0 for p in our_patients]
    wes_features['brca2_mut'] = [1 if p in brca2_patients else 0 for p in our_patients]
    wes_features['brca_any'] = [1 if p in brca_any_patients else 0 for p in our_patients]

    print(f"  BRCA1 mutated: {wes_features['brca1_mut'].sum()}")
    print(f"  BRCA2 mutated: {wes_features['brca2_mut'].sum()}")
    print(f"  BRCA any: {wes_features['brca_any'].sum()}")

    # HRR pathway mutations (any gene in HRR_GENES)
    hrr_patients = set(muts_df[muts_df['gene'].isin(HRR_GENES)]['patient_id'].unique())
    wes_features['hrr_mutated'] = [1 if p in hrr_patients else 0 for p in our_patients]
    print(f"  HRR pathway mutated: {wes_features['hrr_mutated'].sum()}")

    # Individual key gene mutations
    for gene in ['TP53', 'NF1', 'RB1', 'CDK12', 'PTEN', 'PIK3CA']:
        gene_patients = set(muts_df[muts_df['gene'] == gene]['patient_id'].unique())
        wes_features[f'{gene.lower()}_mut'] = [1 if p in gene_patients else 0 for p in our_patients]

    # Damaging mutation types
    damaging_types = ['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'Splice_Site']
    damaging_muts = muts_df[muts_df['mutation_type'].isin(damaging_types)]
    brca_damaging = set(damaging_muts[damaging_muts['gene'].isin(['BRCA1', 'BRCA2'])]['patient_id'].unique())
    wes_features['brca_damaging'] = [1 if p in brca_damaging else 0 for p in our_patients]
    print(f"  BRCA damaging (truncating): {wes_features['brca_damaging'].sum()}")

# TMB
if tmb_data:
    wes_features['tmb'] = [tmb_data.get(p, np.nan) for p in our_patients]
    n_tmb = wes_features['tmb'].notna().sum()
    print(f"  TMB available: {n_tmb} patients, median = {wes_features['tmb'].median():.1f}")

# Copy number features
if cn_data:
    for gene in cn_genes:
        wes_features[f'{gene.lower()}_cn'] = [cn_data.get(p, {}).get(gene, np.nan) for p in our_patients]

    # GISTIC values: -2=deep del, -1=shallow del, 0=diploid, 1=gain, 2=amplification
    if 'ccne1_cn' in wes_features.columns:
        wes_features['ccne1_amp'] = (wes_features['ccne1_cn'] == 2).astype(int)
        print(f"  CCNE1 amplified: {wes_features['ccne1_amp'].sum()}")

    if 'myc_cn' in wes_features.columns:
        wes_features['myc_amp'] = (wes_features['myc_cn'] == 2).astype(int)
        print(f"  MYC amplified: {wes_features['myc_amp'].sum()}")

    if 'brca1_cn' in wes_features.columns:
        wes_features['brca1_del'] = (wes_features['brca1_cn'] <= -1).astype(int)
        print(f"  BRCA1 deleted: {wes_features['brca1_del'].sum()}")

    if 'rb1_cn' in wes_features.columns:
        wes_features['rb1_del'] = (wes_features['rb1_cn'] <= -1).astype(int)
        print(f"  RB1 deleted: {wes_features['rb1_del'].sum()}")

    if 'pten_cn' in wes_features.columns:
        wes_features['pten_del'] = (wes_features['pten_cn'] <= -1).astype(int)
        print(f"  PTEN deleted: {wes_features['pten_del'].sum()}")

# Also load existing BRCA data we already have
with open(DATA / 'ov_tcga_combined_brca.json') as f:
    brca_existing = json.load(f)
wes_features['brca_existing'] = [1 if p in brca_existing else 0 for p in our_patients]
print(f"  BRCA any (existing data): {wes_features['brca_existing'].sum()}")

# Add response and RNA score
wes_features['response'] = tcga_ov['response_binary'].values.astype(int)
wes_features['rna_score'] = tcga_ov['rna_score'].values

print(f"\n  WES feature matrix: {wes_features.shape}")
print(f"  Columns: {list(wes_features.columns)}")

# ══════════════════════════════════════════════════════════════════
# 5. Univariate associations: WES features vs response
# ══════════════════════════════════════════════════════════════════
print("\n[5/7] Testing univariate WES feature associations with response...")

results = {
    'n_patients': len(wes_features),
    'n_sensitive': int(wes_features['response'].sum()),
    'n_resistant': int((wes_features['response'] == 0).sum()),
    'rna_model_auc': round(rna_auc, 4),
}

univariate = {}
binary_features = [c for c in wes_features.columns
                   if c not in ['response', 'rna_score', 'tmb'] and
                   not c.endswith('_cn') and
                   wes_features[c].nunique() <= 2]

for feat in binary_features:
    col = wes_features[feat]
    resp = wes_features['response']

    n_pos = int(col.sum())
    n_neg = int((col == 0).sum())
    if n_pos < 3 or n_neg < 3:
        continue

    # Response rate in mutated vs wildtype
    resp_mut = resp[col == 1].mean()
    resp_wt = resp[col == 0].mean()

    # Fisher's exact test
    table = pd.crosstab(col, resp)
    if table.shape == (2, 2):
        odds_ratio, fisher_p = fisher_exact(table)
    else:
        odds_ratio, fisher_p = np.nan, np.nan

    # AUC (can this feature alone predict response?)
    try:
        auc = roc_auc_score(resp, col)
    except ValueError:
        auc = np.nan

    univariate[feat] = {
        'n_positive': n_pos,
        'n_negative': n_neg,
        'response_rate_positive': round(resp_mut, 3),
        'response_rate_negative': round(resp_wt, 3),
        'odds_ratio': round(odds_ratio, 3) if not np.isnan(odds_ratio) else None,
        'fisher_p': round(fisher_p, 4) if not np.isnan(fisher_p) else None,
        'auc': round(auc, 4) if not np.isnan(auc) else None,
    }
    print(f"  {feat}: n={n_pos}, resp_rate={resp_mut:.3f} vs {resp_wt:.3f}, OR={odds_ratio:.2f}, p={fisher_p:.4f}, AUC={auc:.3f}")

# TMB if available
if 'tmb' in wes_features.columns and wes_features['tmb'].notna().sum() > 50:
    tmb_valid = wes_features.dropna(subset=['tmb'])
    tmb_rho, tmb_p = spearmanr(tmb_valid['tmb'], tmb_valid['response'])
    try:
        tmb_auc = roc_auc_score(tmb_valid['response'], tmb_valid['tmb'])
    except ValueError:
        tmb_auc = np.nan
    univariate['tmb'] = {
        'n_available': len(tmb_valid),
        'spearman_rho': round(tmb_rho, 4),
        'spearman_p': round(tmb_p, 4),
        'auc': round(tmb_auc, 4) if not np.isnan(tmb_auc) else None,
        'median': round(tmb_valid['tmb'].median(), 1),
    }
    print(f"  TMB: rho={tmb_rho:.3f}, p={tmb_p:.4f}, AUC={tmb_auc:.3f}")

results['univariate_associations'] = univariate

# ══════════════════════════════════════════════════════════════════
# 6. Multivariate: WES features → response
# ══════════════════════════════════════════════════════════════════
print("\n[6/7] Multivariate WES models for response prediction...")

# Select WES features for modeling (binary + TMB, no CN raw values)
model_features = [c for c in binary_features if univariate.get(c, {}).get('n_positive', 0) >= 3]
if 'tmb' in wes_features.columns and wes_features['tmb'].notna().sum() > 100:
    model_features.append('tmb')

print(f"  WES model features ({len(model_features)}): {model_features}")

# Prepare data
wes_model_data = wes_features[model_features + ['response', 'rna_score']].dropna()
X_wes = wes_model_data[model_features].values
y = wes_model_data['response'].values.astype(int)
rna = wes_model_data['rna_score'].values

print(f"  Samples with all features: {len(wes_model_data)}")

# Bootstrap evaluation (since we have one dataset, use bootstrap)
n_bootstrap = 2000
np.random.seed(42)

def bootstrap_auc(y_true, y_score, n_boot=2000):
    """Bootstrap AUC with 95% CI."""
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        y_b = y_true[idx]
        s_b = y_score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, s_b))
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

# 5-fold CV for WES model
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model A: RNA score only
rna_cv_preds = np.zeros(len(y))
# Model B: WES features only
wes_cv_preds = np.zeros(len(y))
# Model C: WES + RNA combined
combined_cv_preds = np.zeros(len(y))
# Model D: BRCA only (single feature)
brca_col_idx = model_features.index('brca_any') if 'brca_any' in model_features else None

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_wes, y)):
    # RNA only (already have scores, but re-fit logistic for calibration)
    clf_r = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=3000,
                                class_weight='balanced', random_state=42)
    clf_r.fit(rna[train_idx].reshape(-1, 1), y[train_idx])
    rna_cv_preds[test_idx] = clf_r.predict_proba(rna[test_idx].reshape(-1, 1))[:, 1]

    # WES only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_wes[train_idx])
    X_test_scaled = scaler.transform(X_wes[test_idx])

    clf_w = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', max_iter=3000,
                                class_weight='balanced', random_state=42)
    clf_w.fit(X_train_scaled, y[train_idx])
    wes_cv_preds[test_idx] = clf_w.predict_proba(X_test_scaled)[:, 1]

    # Combined: WES features + RNA score
    X_combined_train = np.column_stack([X_train_scaled, rna[train_idx]])
    X_combined_test = np.column_stack([X_test_scaled, rna[test_idx]])

    clf_c = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', max_iter=3000,
                                class_weight='balanced', random_state=42)
    clf_c.fit(X_combined_train, y[train_idx])
    combined_cv_preds[test_idx] = clf_c.predict_proba(X_combined_test)[:, 1]

# Also compute AUC for raw RNA score (no re-fitting)
rna_raw_auc = roc_auc_score(y, rna)
rna_raw_boot = bootstrap_auc(y, rna)

# CV AUCs
rna_cv_auc = roc_auc_score(y, rna_cv_preds)
wes_cv_auc = roc_auc_score(y, wes_cv_preds)
combined_cv_auc = roc_auc_score(y, combined_cv_preds)

# Bootstrap CIs
rna_cv_boot = bootstrap_auc(y, rna_cv_preds)
wes_cv_boot = bootstrap_auc(y, wes_cv_preds)
combined_cv_boot = bootstrap_auc(y, combined_cv_preds)

# BRCA-only AUC (raw feature, no training needed)
if brca_col_idx is not None:
    brca_auc = roc_auc_score(y, X_wes[:, brca_col_idx])
    brca_boot = bootstrap_auc(y, X_wes[:, brca_col_idx])
else:
    brca_auc = np.nan
    brca_boot = (np.nan, np.nan, np.nan)

print(f"\n  Results (5-fold CV on TCGA-OV):")
print(f"    RNA score (LODO, raw):     AUC = {rna_raw_auc:.4f} [{rna_raw_boot[1]:.3f}, {rna_raw_boot[2]:.3f}]")
print(f"    RNA score (5-fold CV):     AUC = {rna_cv_auc:.4f} [{rna_cv_boot[1]:.3f}, {rna_cv_boot[2]:.3f}]")
print(f"    WES features (5-fold CV):  AUC = {wes_cv_auc:.4f} [{wes_cv_boot[1]:.3f}, {wes_cv_boot[2]:.3f}]")
print(f"    WES+RNA combined (5f CV):  AUC = {combined_cv_auc:.4f} [{combined_cv_boot[1]:.3f}, {combined_cv_boot[2]:.3f}]")
print(f"    BRCA-any (raw feature):    AUC = {brca_auc:.4f} [{brca_boot[1]:.3f}, {brca_boot[2]:.3f}]")

results['multivariate_models'] = {
    'rna_lodo_raw': {
        'auc': round(rna_raw_auc, 4),
        'ci_low': round(rna_raw_boot[1], 4),
        'ci_high': round(rna_raw_boot[2], 4),
        'note': 'LODO-CV predicted score (trained on 8 other datasets)'
    },
    'rna_5fold_cv': {
        'auc': round(rna_cv_auc, 4),
        'ci_low': round(rna_cv_boot[1], 4),
        'ci_high': round(rna_cv_boot[2], 4),
        'note': 'RNA score passed through logistic regression (5-fold CV within TCGA-OV)'
    },
    'wes_5fold_cv': {
        'auc': round(wes_cv_auc, 4),
        'ci_low': round(wes_cv_boot[1], 4),
        'ci_high': round(wes_cv_boot[2], 4),
        'n_features': len(model_features),
        'features': model_features,
        'note': 'L2 logistic regression on WES features (5-fold CV within TCGA-OV)'
    },
    'combined_5fold_cv': {
        'auc': round(combined_cv_auc, 4),
        'ci_low': round(combined_cv_boot[1], 4),
        'ci_high': round(combined_cv_boot[2], 4),
        'note': 'WES features + RNA score combined (5-fold CV)'
    },
    'brca_raw': {
        'auc': round(brca_auc, 4) if not np.isnan(brca_auc) else None,
        'ci_low': round(brca_boot[1], 4) if not np.isnan(brca_boot[1]) else None,
        'ci_high': round(brca_boot[2], 4) if not np.isnan(brca_boot[2]) else None,
        'note': 'Raw BRCA-any mutation as single predictor'
    }
}

# ══════════════════════════════════════════════════════════════════
# 7. Correlation between RNA and WES scores
# ══════════════════════════════════════════════════════════════════
print("\n[7/7] Checking RNA-WES score correlation (complementarity)...")

rna_wes_rho, rna_wes_p = spearmanr(rna, wes_cv_preds)
print(f"  RNA score vs WES score: Spearman rho = {rna_wes_rho:.3f}, p = {rna_wes_p:.4f}")

# BRCA status vs RNA score
if brca_col_idx is not None:
    brca_vals = X_wes[:, brca_col_idx]
    rna_brca_rho, rna_brca_p = spearmanr(rna, brca_vals)
    mw_stat, mw_p = mannwhitneyu(rna[brca_vals == 1], rna[brca_vals == 0], alternative='two-sided')
    print(f"  RNA score vs BRCA status: rho = {rna_brca_rho:.3f}, p = {rna_brca_p:.4f}")
    print(f"  RNA score in BRCA-mut: {rna[brca_vals == 1].mean():.3f} vs BRCA-wt: {rna[brca_vals == 0].mean():.3f} (MW p={mw_p:.4f})")

results['complementarity'] = {
    'rna_vs_wes_score': {
        'spearman_rho': round(rna_wes_rho, 4),
        'p': round(rna_wes_p, 4),
    },
}

if brca_col_idx is not None:
    results['complementarity']['rna_vs_brca'] = {
        'spearman_rho': round(rna_brca_rho, 4),
        'p': round(rna_brca_p, 4),
        'mean_rna_brca_mut': round(rna[brca_vals == 1].mean(), 4),
        'mean_rna_brca_wt': round(rna[brca_vals == 0].mean(), 4),
        'mannwhitney_p': round(mw_p, 4),
    }

# ══════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("Saving results...")

results['analysis_metadata'] = {
    'script': 'addendum/tcga_wes_analysis.py',
    'date': '2026-02-25',
    'data_source': 'cBioPortal API (ov_tcga_pan_can_atlas_2018)',
    'rna_model': 'L2 LogReg C=0.01, LODO-CV on fixed 11089-gene matrix',
    'wes_model': 'L2 LogReg C=0.1, 5-fold stratified CV within TCGA-OV',
    'bootstrap_n': n_bootstrap,
}

with open(OUT / 'tcga_wes_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Results saved to {OUT / 'tcga_wes_results.json'}")

# Save WES feature matrix
wes_features.to_csv(OUT / 'tcga_wes_features.csv')
print(f"  Features saved to {OUT / 'tcga_wes_features.csv'}")

print("\nDone!")
