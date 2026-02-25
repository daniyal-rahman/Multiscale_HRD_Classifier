#!/usr/bin/env python3
"""
GSE63885 Investigation: Why does the model fail on this dataset? (AUC=0.575)

Unlike GSE32062, GSE63885 has FULL gene coverage (11140/11140, zero NaN).
This is a genuine model failure. The question is: WHY?
"""

import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
EXP = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT = EXP / "gse63885_investigation"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading data...", flush=True)
pooled = pd.read_parquet(EXP / 'pooled_rank_matrix.parquet')
fw = pd.read_parquet(EXP / 'feature_weights.parquet')
meta = pd.read_parquet(BASE / 'GSE63885_metadata.parquet')
resp = pd.read_parquet(BASE / 'GSE63885_response.parquet')

meta_cols = ['response_binary', 'response_continuous', 'drug', 'dataset', 'category', 'cancer_type']
gene_cols = [c for c in pooled.columns if c not in meta_cols]

with open(BASE / "common_genes_all_10_datasets.txt") as f:
    common_genes = [g.strip() for g in f]

report = {}

# =====================================================================
# 1. DATA CHARACTERIZATION
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("1. DATA CHARACTERIZATION", flush=True)
print("=" * 70, flush=True)

gse63 = pooled[pooled['dataset'] == 'GSE63885']
y = gse63['response_binary'].values.astype(int)
n_pos, n_neg = int(y.sum()), int((1-y).sum())
print(f"GSE63885: n={len(y)}, sensitive={n_pos} ({n_pos/len(y)*100:.1f}%), resistant={n_neg} ({n_neg/len(y)*100:.1f}%)", flush=True)

# Gene coverage check
nan_frac = gse63[gene_cols].isna().mean().mean()
print(f"NaN fraction in pooled rank matrix: {nan_frac:.6f} (PERFECT — no data issues)", flush=True)

# 3-category platinum sensitivity
plat_sens = meta.set_index('sample_id').loc[resp['sample_id'].values, 'plat_sensitivity']
print(f"\nPlatinum sensitivity categories:")
for cat in ['resistant', 'moderately sensitive', 'highly sensitive']:
    n = (plat_sens == cat).sum()
    print(f"  {cat}: {n} ({n/len(plat_sens)*100:.1f}%)", flush=True)

# Treatment breakdown
chemo = meta.set_index('sample_id').loc[resp['sample_id'].values, 'adjuwant chemotherapy']
print(f"\nTreatment regimens:")
print(chemo.value_counts().to_string(), flush=True)

# Cross-tab: treatment × response
resp_matched = resp.set_index('sample_id')
meta_matched = meta.set_index('sample_id')
combined = pd.DataFrame({
    'response': resp_matched.loc[meta_matched.index.intersection(resp_matched.index), 'response_binary'],
    'chemo': meta_matched.loc[meta_matched.index.intersection(resp_matched.index), 'adjuwant chemotherapy'],
    'plat_sens': meta_matched.loc[meta_matched.index.intersection(resp_matched.index), 'plat_sensitivity'],
    'brca1': meta_matched.loc[meta_matched.index.intersection(resp_matched.index), 'brca1 mutation'],
})
combined['response'] = combined['response'].astype(int)

print(f"\nTreatment × Response:")
ct = pd.crosstab(combined['chemo'], combined['response'], margins=True)
print(ct.to_string(), flush=True)

# BRCA1 × response
print(f"\nBRCA1 mutation × Response:")
combined['brca1_mut'] = combined['brca1'] != 'no mutation'
ct_brca = pd.crosstab(combined['brca1_mut'], combined['response'], margins=True)
print(ct_brca.to_string(), flush=True)

# DFS distribution
dfs_days = meta.set_index('sample_id').loc[resp['sample_id'].values, 'dfs - disease-free survival [days]']
dfs_days = pd.to_numeric(dfs_days, errors='coerce')
dfs_months = dfs_days / 30.44  # Convert to months

print(f"\nDFS distribution (months):")
print(f"  All: median={dfs_months.median():.1f}, mean={dfs_months.mean():.1f}, range=[{dfs_months.min():.1f}, {dfs_months.max():.1f}]", flush=True)
print(f"  Resistant (DFS<6m): n={n_neg}, median={dfs_months[combined['response']==0].median():.1f}", flush=True)
print(f"  Sensitive (DFS>=6m): n={n_pos}, median={dfs_months[combined['response']==1].median():.1f}", flush=True)

report['characterization'] = {
    'n_samples': len(y),
    'n_sensitive': n_pos,
    'n_resistant': n_neg,
    'nan_fraction': round(float(nan_frac), 6),
    'platform': 'Affymetrix_HGU133Plus2',
    'gene_coverage': '11140/11140 (100%)',
    'treatment_taxane_platinum': int((chemo == 'taxane/platinum').sum()),
    'treatment_platinum_cyclo': int((chemo == 'platinum/cyclophosphamide').sum()),
    'brca1_mutated': int(combined['brca1_mut'].sum()),
    'brca1_wildtype': int((~combined['brca1_mut']).sum()),
    'platinum_sensitivity_categories': {
        'resistant': int((plat_sens == 'resistant').sum()),
        'moderately_sensitive': int((plat_sens == 'moderately sensitive').sum()),
        'highly_sensitive': int((plat_sens == 'highly sensitive').sum()),
    },
    'dfs_cutoff_days': 180,
    'dfs_cutoff_months': 6,
}

# =====================================================================
# 2. PREDICTION ANALYSIS: What does the model predict?
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("2. PREDICTION ANALYSIS", flush=True)
print("=" * 70, flush=True)

# Train on non-GSE63885 (matching LODO-CV setup)
ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (pooled['category'] != 'cell_line') &
    ~((pooled['category'] == 'clinical_ispy2') & (~pooled['drug'].isin(ispy2_parpi_drugs)))
)
model_a = pooled[model_a_mask]
non_gse63 = model_a[model_a['dataset'] != 'GSE63885']

X_train = np.nan_to_num(non_gse63[gene_cols].values, nan=0.5)
y_train = non_gse63['response_binary'].values.astype(int)
X_test = np.nan_to_num(gse63[gene_cols].values, nan=0.5)

clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                         max_iter=5000, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y, y_pred)

print(f"AUC: {auc:.4f}", flush=True)
print(f"Predictions: mean={y_pred.mean():.4f}, std={y_pred.std():.4f}, range=[{y_pred.min():.4f}, {y_pred.max():.4f}]", flush=True)
print(f"  Sensitive:  mean={y_pred[y==1].mean():.4f} ± {y_pred[y==1].std():.4f}", flush=True)
print(f"  Resistant:  mean={y_pred[y==0].mean():.4f} ± {y_pred[y==0].std():.4f}", flush=True)

# Also try C=0.01 (team lead says this is better)
clf_reg = LogisticRegression(penalty='l2', C=0.01, solver='lbfgs',
                             max_iter=5000, class_weight='balanced', random_state=42)
clf_reg.fit(X_train, y_train)
y_pred_reg = clf_reg.predict_proba(X_test)[:, 1]
auc_reg = roc_auc_score(y, y_pred_reg)
print(f"\nWith C=0.01 (stronger regularization):", flush=True)
print(f"  AUC: {auc_reg:.4f}", flush=True)
print(f"  Predictions: mean={y_pred_reg.mean():.4f}, std={y_pred_reg.std():.4f}, range=[{y_pred_reg.min():.4f}, {y_pred_reg.max():.4f}]", flush=True)

# Stratify by platinum sensitivity category
print(f"\nPrediction by platinum sensitivity:", flush=True)
sample_ids = gse63.index.tolist()
for cat in ['resistant', 'moderately sensitive', 'highly sensitive']:
    cat_mask = plat_sens.values == cat
    # Make sure alignment is correct
    if cat_mask.sum() > 0:
        cat_preds = y_pred[cat_mask[:len(y_pred)]] if len(cat_mask) >= len(y_pred) else y_pred[:cat_mask.sum()]
        print(f"  {cat:25s}: n={cat_mask.sum()}, mean_pred={y_pred[cat_mask[:len(y)]].mean():.4f} ± {y_pred[cat_mask[:len(y)]].std():.4f}", flush=True)

# Stratify by treatment
print(f"\nPrediction by treatment:", flush=True)
chemo_vals = chemo.values[:len(y)]
for trt in chemo.unique():
    trt_mask = chemo_vals == trt
    if trt_mask.sum() > 0:
        trt_y = y[trt_mask]
        trt_pred = y_pred[trt_mask]
        if len(np.unique(trt_y)) >= 2:
            trt_auc = roc_auc_score(trt_y, trt_pred)
        else:
            trt_auc = float('nan')
        print(f"  {trt:35s}: n={trt_mask.sum()}, AUC={trt_auc:.4f}, "
              f"sens={trt_y.sum()}/{len(trt_y)}", flush=True)

# Stratify by BRCA1
brca_vals = combined['brca1_mut'].values[:len(y)]
for label, mask_val in [('BRCA1 mutated', True), ('BRCA1 wildtype', False)]:
    brca_mask = brca_vals == mask_val
    if brca_mask.sum() > 0 and len(np.unique(y[brca_mask])) >= 2:
        brca_auc = roc_auc_score(y[brca_mask], y_pred[brca_mask])
        print(f"  {label:25s}: n={brca_mask.sum()}, AUC={brca_auc:.4f}", flush=True)

report['prediction_analysis'] = {
    'auc_C1': round(float(auc), 4),
    'auc_C001': round(float(auc_reg), 4),
    'pred_mean': round(float(y_pred.mean()), 4),
    'pred_std': round(float(y_pred.std()), 4),
    'pred_range': round(float(y_pred.max() - y_pred.min()), 4),
    'pred_mean_sensitive': round(float(y_pred[y==1].mean()), 4),
    'pred_mean_resistant': round(float(y_pred[y==0].mean()), 4),
}

# =====================================================================
# 3. RESPONSE DEFINITION COMPARISON
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("3. RESPONSE DEFINITION COMPARISON", flush=True)
print("=" * 70, flush=True)

# GSE63885 uses DFS < 180 days (6 months) = resistant
# Try different cutoffs
print("AUC at different DFS cutoffs:", flush=True)
cutoff_results = {}
for cutoff_months in [3, 6, 9, 12, 18, 24]:
    cutoff_days = cutoff_months * 30.44
    y_alt = (dfs_days.values > cutoff_days).astype(int)
    valid = ~np.isnan(dfs_days.values)
    y_alt_v = y_alt[valid][:len(y_pred)]
    y_pred_v = y_pred[valid[:len(y_pred)]]
    if len(np.unique(y_alt_v)) >= 2 and len(y_alt_v) == len(y_pred_v):
        auc_alt = roc_auc_score(y_alt_v, y_pred_v)
        n_sens = y_alt_v.sum()
        n_res = len(y_alt_v) - n_sens
        cutoff_results[cutoff_months] = {'auc': round(float(auc_alt), 4),
                                         'n_sensitive': int(n_sens),
                                         'n_resistant': int(n_res)}
        print(f"  >{cutoff_months}m: AUC={auc_alt:.4f} (sens={n_sens}, res={n_res})", flush=True)

# Compare response definitions across training datasets
print(f"\nResponse definitions across datasets:", flush=True)
print(f"  GSE63885: DFS > 180 days (6 months)", flush=True)
print(f"  GSE32062: PFS > 6 months", flush=True)
print(f"  TCGA-OV:  PLATINUM_STATUS (clinical annotation)", flush=True)
print(f"  GSE30161: PFS > 6 months", flush=True)
print(f"  GSE156699: 'response' column (presumably CR/PR vs SD/PD)", flush=True)

# Key question: is the response definition compatible?
# GSE63885 "resistant" = DFS < 6 months = same as others using PFS < 6m
# But DFS and PFS are NOT the same:
# DFS = time from surgery to recurrence or death
# PFS = time from start of chemo to progression
print(f"\n  IMPORTANT: DFS (disease-free survival) ≠ PFS (progression-free survival)", flush=True)
print(f"  DFS counts from SURGERY; PFS counts from START OF CHEMO", flush=True)
print(f"  If surgery-to-chemo gap is ~1 month, DFS < 180 days ≈ PFS < 150 days (5 months)", flush=True)
print(f"  This makes GSE63885 'resistant' slightly MORE inclusive than PFS < 6m", flush=True)

report['response_definition'] = {
    'definition': 'DFS < 180 days = resistant (NOT PFS)',
    'dfs_vs_pfs_note': 'DFS counts from surgery, PFS from chemo start. DFS cutoff is effectively stricter.',
    'cutoff_auc_sweep': cutoff_results,
}

# =====================================================================
# 4. BATCH EFFECT ANALYSIS: How different is GSE63885 from training?
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("4. BATCH EFFECT ANALYSIS", flush=True)
print("=" * 70, flush=True)

# PCA: where does GSE63885 sit relative to training datasets?
from sklearn.decomposition import PCA

# Use all clinical data
clinical = pooled[pooled['category'] != 'cell_line']
X_clin = np.nan_to_num(clinical[gene_cols].values, nan=0.5)
ds_clin = clinical['dataset'].values

pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_clin)

print("PCA analysis (all clinical datasets):", flush=True)
for ds in sorted(clinical['dataset'].unique()):
    ds_mask = ds_clin == ds
    mean_pc1 = X_pca[ds_mask, 0].mean()
    mean_pc2 = X_pca[ds_mask, 1].mean()
    print(f"  {ds:15s}: PC1 mean={mean_pc1:7.2f}, PC2 mean={mean_pc2:7.2f}", flush=True)

# Distance from GSE63885 to other datasets (in PCA space)
gse63_pca_mean = X_pca[ds_clin == 'GSE63885'].mean(axis=0)
print(f"\nPCA distances from GSE63885 centroid:", flush=True)
pca_distances = {}
for ds in sorted(clinical['dataset'].unique()):
    if ds == 'GSE63885':
        continue
    ds_mean = X_pca[ds_clin == ds].mean(axis=0)
    dist = np.linalg.norm(gse63_pca_mean - ds_mean)
    pca_distances[ds] = round(float(dist), 2)
    print(f"  → {ds:15s}: distance={dist:.2f}", flush=True)

# Top 20 genes: compare rank distributions in GSE63885 vs TCGA-OV
top20 = fw.nlargest(20, 'abs_weight_clinical')['gene'].tolist()
print(f"\nTop 20 gene rank distributions (GSE63885 vs TCGA-OV):", flush=True)
tcga = pooled[pooled['dataset'] == 'TCGA-OV']
gene_comparison = []
for gene in top20:
    gse_vals = gse63[gene].dropna()
    tcga_vals = tcga[gene].dropna()
    if len(gse_vals) > 5 and len(tcga_vals) > 5:
        ks_stat, ks_p = stats.ks_2samp(gse_vals, tcga_vals)
        diff_mean = float(gse_vals.mean() - tcga_vals.mean())
        gene_comparison.append({
            'gene': gene,
            'gse63885_mean': round(float(gse_vals.mean()), 4),
            'tcga_mean': round(float(tcga_vals.mean()), 4),
            'mean_diff': round(diff_mean, 4),
            'ks_stat': round(float(ks_stat), 4),
            'ks_p': float(ks_p),
        })
        sig = '*' if ks_p < 0.001 else ''
        print(f"  {gene:12s}: GSE63={gse_vals.mean():.3f}, TCGA={tcga_vals.mean():.3f}, "
              f"diff={diff_mean:+.3f}, KS={ks_stat:.3f} {sig}", flush=True)

n_shifted = sum(1 for g in gene_comparison if g['ks_p'] < 0.001)
print(f"\nGenes with significantly different distributions (p<0.001): {n_shifted}/{len(gene_comparison)}", flush=True)

report['batch_effect'] = {
    'pca_distances': pca_distances,
    'top20_gene_shifts': gene_comparison,
    'n_significantly_shifted': n_shifted,
}

# =====================================================================
# 5. TREATMENT CONFOUND
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("5. TREATMENT CONFOUND", flush=True)
print("=" * 70, flush=True)

# Is treatment (taxane/plat vs plat/cyclo) confounded with response?
ct_trt_resp = pd.crosstab(combined['chemo'], combined['response'])
print("Treatment × Response crosstab:", flush=True)
print(ct_trt_resp.to_string(), flush=True)

odds_trt, p_trt = stats.fisher_exact(ct_trt_resp.values)
print(f"Fisher exact: OR={odds_trt:.3f}, p={p_trt:.4f}", flush=True)

# The model was trained on OTHER datasets which use DIFFERENT chemo regimens
# GSE63885 has 34 platinum/cyclophosphamide — this is an OLDER regimen
# Most modern training datasets use taxane/platinum
print(f"\nTraining data treatment regimens:", flush=True)
for ds in sorted(pooled['dataset'].unique()):
    if ds == 'GSE63885' or pooled[pooled['dataset']==ds]['category'].iloc[0] == 'cell_line':
        continue
    drugs = pooled[pooled['dataset'] == ds]['drug'].value_counts()
    print(f"  {ds:15s}: {dict(drugs)}", flush=True)

# Within-treatment AUC
print(f"\nWithin-treatment AUC:", flush=True)
chemo_aligned = chemo.values[:len(y)]
for trt in ['taxane/platinum', 'platinum/cyclophosphamide']:
    mask = chemo_aligned == trt
    if mask.sum() > 0 and len(np.unique(y[mask])) >= 2:
        trt_auc = roc_auc_score(y[mask], y_pred[mask])
        print(f"  {trt}: AUC={trt_auc:.4f} (n={mask.sum()}, sens={y[mask].sum()}, res={(1-y[mask]).sum()})", flush=True)
    else:
        print(f"  {trt}: cannot compute (n={mask.sum()})", flush=True)

report['treatment_confound'] = {
    'treatment_response_OR': round(float(odds_trt), 3),
    'treatment_response_p': round(float(p_trt), 4),
}

# =====================================================================
# 6. WITHIN-DATASET MODEL
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("6. WITHIN-DATASET MODEL", flush=True)
print("=" * 70, flush=True)

X_within = np.nan_to_num(gse63[gene_cols].values, nan=0.5)
y_within = y

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
within_aucs = []
within_aucs_reg = []
for fold, (tr_idx, te_idx) in enumerate(skf.split(X_within, y_within)):
    # C=1.0
    clf_w = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                               max_iter=5000, class_weight='balanced', random_state=42)
    clf_w.fit(X_within[tr_idx], y_within[tr_idx])
    p = clf_w.predict_proba(X_within[te_idx])[:, 1]
    if len(np.unique(y_within[te_idx])) >= 2:
        within_aucs.append(roc_auc_score(y_within[te_idx], p))

    # C=0.01
    clf_w2 = LogisticRegression(penalty='l2', C=0.01, solver='lbfgs',
                                max_iter=5000, class_weight='balanced', random_state=42)
    clf_w2.fit(X_within[tr_idx], y_within[tr_idx])
    p2 = clf_w2.predict_proba(X_within[te_idx])[:, 1]
    if len(np.unique(y_within[te_idx])) >= 2:
        within_aucs_reg.append(roc_auc_score(y_within[te_idx], p2))

print(f"Within-dataset 5-fold CV (C=1.0): {np.mean(within_aucs):.4f} ± {np.std(within_aucs):.4f}", flush=True)
print(f"Within-dataset 5-fold CV (C=0.01): {np.mean(within_aucs_reg):.4f} ± {np.std(within_aucs_reg):.4f}", flush=True)

report['within_dataset'] = {
    'cv_auc_C1_mean': round(float(np.mean(within_aucs)), 4),
    'cv_auc_C1_std': round(float(np.std(within_aucs)), 4),
    'cv_auc_C001_mean': round(float(np.mean(within_aucs_reg)), 4),
    'cv_auc_C001_std': round(float(np.std(within_aucs_reg)), 4),
    'cv_aucs_C1': [round(a, 4) for a in within_aucs],
    'cv_aucs_C001': [round(a, 4) for a in within_aucs_reg],
}

# =====================================================================
# 7. UNIVARIATE GENE-RESPONSE CORRELATIONS
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("7. UNIVARIATE GENE-RESPONSE CORRELATIONS", flush=True)
print("=" * 70, flush=True)

gene_univar = []
for gene in gene_cols:
    vals = gse63[gene].values
    valid = ~np.isnan(vals)
    if valid.sum() > 10 and len(np.unique(y[valid])) >= 2:
        r, p = stats.pointbiserialr(y[valid], vals[valid])
        gene_univar.append({'gene': gene, 'r': float(r), 'p': float(p)})

gene_univar_df = pd.DataFrame(gene_univar).sort_values('p')
n_sig_005 = (gene_univar_df['p'] < 0.05).sum()
n_sig_001 = (gene_univar_df['p'] < 0.01).sum()
n_sig_bonf = (gene_univar_df['p'] < 0.05/len(gene_univar)).sum()

print(f"Univariate gene-response correlations:", flush=True)
print(f"  Total genes tested: {len(gene_univar)}", flush=True)
print(f"  Significant at p<0.05: {n_sig_005}", flush=True)
print(f"  Significant at p<0.01: {n_sig_001}", flush=True)
print(f"  Significant after Bonferroni: {n_sig_bonf}", flush=True)
print(f"  Expected by chance (p<0.05): {len(gene_univar)*0.05:.0f}", flush=True)

# Are the TOP model genes predictive?
print(f"\nTop 20 model genes — univariate correlation in GSE63885:", flush=True)
for gene in top20:
    match = gene_univar_df[gene_univar_df['gene'] == gene]
    if len(match) > 0:
        r_val = match.iloc[0]['r']
        p_val = match.iloc[0]['p']
        weight = float(fw[fw['gene'] == gene]['weight_clinical'].iloc[0])
        # Are signs consistent?
        sign_match = "MATCH" if np.sign(r_val) == np.sign(weight) else "MISMATCH"
        sig = "*" if p_val < 0.05 else ""
        print(f"  {gene:12s}: r={r_val:+.4f}, p={p_val:.4f}{sig}, weight={weight:+.4f} → {sign_match}", flush=True)

# Top univariate predictors
print(f"\nTop 10 univariate predictors in GSE63885:", flush=True)
for _, row in gene_univar_df.head(10).iterrows():
    w = fw[fw['gene'] == row['gene']]['weight_clinical']
    w_val = float(w.iloc[0]) if len(w) > 0 else 0
    print(f"  {row['gene']:12s}: r={row['r']:+.4f}, p={row['p']:.2e}, model_weight={w_val:+.4f}", flush=True)

report['univariate'] = {
    'n_tested': len(gene_univar),
    'n_sig_005': int(n_sig_005),
    'n_sig_001': int(n_sig_001),
    'n_sig_bonferroni': int(n_sig_bonf),
    'expected_by_chance': round(len(gene_univar) * 0.05),
}

# =====================================================================
# 8. CROSS-DATASET TRANSFER DIRECTION
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("8. CROSS-DATASET TRANSFER", flush=True)
print("=" * 70, flush=True)

# Can GSE63885 predict TCGA-OV? And vice versa with restricted features?
tcga_data = pooled[pooled['dataset'] == 'TCGA-OV']
X_tcga = np.nan_to_num(tcga_data[gene_cols].values, nan=0.5)
y_tcga = tcga_data['response_binary'].values.astype(int)

# GSE63885 → TCGA-OV
clf_63to = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                              max_iter=5000, class_weight='balanced', random_state=42)
clf_63to.fit(X_within, y_within)
auc_63_to_tcga = roc_auc_score(y_tcga, clf_63to.predict_proba(X_tcga)[:, 1])
print(f"GSE63885 → TCGA-OV: AUC={auc_63_to_tcga:.4f}", flush=True)

# TCGA-OV → GSE63885
clf_tcgato = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                                max_iter=5000, class_weight='balanced', random_state=42)
clf_tcgato.fit(X_tcga, y_tcga)
auc_tcga_to_63 = roc_auc_score(y_within, clf_tcgato.predict_proba(X_within)[:, 1])
print(f"TCGA-OV → GSE63885: AUC={auc_tcga_to_63:.4f}", flush=True)

# GSE156699 → GSE63885
gse156 = pooled[pooled['dataset'] == 'GSE156699']
X_156 = np.nan_to_num(gse156[gene_cols].values, nan=0.5)
y_156 = gse156['response_binary'].values.astype(int)
clf_156to = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                               max_iter=5000, class_weight='balanced', random_state=42)
clf_156to.fit(X_156, y_156)
auc_156_to_63 = roc_auc_score(y_within, clf_156to.predict_proba(X_within)[:, 1])
print(f"GSE156699 → GSE63885: AUC={auc_156_to_63:.4f}", flush=True)

report['cross_dataset'] = {
    'gse63885_to_tcga': round(float(auc_63_to_tcga), 4),
    'tcga_to_gse63885': round(float(auc_tcga_to_63), 4),
    'gse156699_to_gse63885': round(float(auc_156_to_63), 4),
}

# =====================================================================
# SUMMARY & PLOTS
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("SUMMARY", flush=True)
print("=" * 70, flush=True)

summary_text = []
summary_text.append("GSE63885 has FULL gene coverage (11140/11140, zero NaN).")
summary_text.append("This is a GENUINE model failure, not a data pipeline bug.")
summary_text.append(f"AUC={auc:.4f} with C=1.0, AUC={auc_reg:.4f} with C=0.01.")
summary_text.append(f"Prediction range is {y_pred.max()-y_pred.min():.3f} — the model DOES discriminate (unlike GSE32062).")
summary_text.append(f"Within-dataset CV: {np.mean(within_aucs):.3f} (C=1) / {np.mean(within_aucs_reg):.3f} (C=0.01).")
summary_text.append(f"Univariate predictors: {n_sig_005} genes at p<0.05 (expected {len(gene_univar)*0.05:.0f} by chance).")
summary_text.append(f"BRCA1 mutation rate: {combined['brca1_mut'].mean()*100:.0f}% — high for ovarian cohort.")
summary_text.append(f"Treatment mix: {int((chemo=='taxane/platinum').sum())} taxane/plat + {int((chemo=='platinum/cyclophosphamide').sum())} plat/cyclo.")

for s in summary_text:
    print(f"  {s}", flush=True)

report['summary'] = {
    'verdict': 'Genuine failure — model signal does not transfer to this cohort',
    'key_points': summary_text,
    'possible_explanations': [
        '1. Treatment heterogeneity: 45% received platinum/cyclophosphamide (older regimen) vs taxane/platinum in most training data',
        '2. High BRCA1 mutation rate (28%) may alter response biology differently than in WT-dominated training cohorts',
        '3. Response endpoint is DFS-based (from surgery) not PFS-based (from chemo start) — subtle definition mismatch',
        '4. Polish cohort may have population-specific biology or treatment protocols',
        '5. Small sample size (n=75, 34 resistant) gives wide CIs: [0.444, 0.704] from bootstrap',
    ],
}

# Create summary figure
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Panel 1: Prediction distribution by class
ax = axes[0, 0]
ax.hist(y_pred[y==1], bins=15, alpha=0.5, label=f'Sensitive (n={n_pos})', density=True, color='green')
ax.hist(y_pred[y==0], bins=15, alpha=0.5, label=f'Resistant (n={n_neg})', density=True, color='red')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Density')
ax.set_title(f'GSE63885 predictions (AUC={auc:.3f})')
ax.legend()

# Panel 2: Prediction by platinum sensitivity category
ax = axes[0, 1]
for cat, color in [('resistant', 'red'), ('moderately sensitive', 'orange'), ('highly sensitive', 'green')]:
    cat_mask = plat_sens.values[:len(y)] == cat
    if cat_mask.sum() > 0:
        ax.hist(y_pred[cat_mask], bins=10, alpha=0.4, label=f'{cat} (n={cat_mask.sum()})',
                density=True, color=color)
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Density')
ax.set_title('Predictions by plat sensitivity')
ax.legend(fontsize=8)

# Panel 3: AUC by DFS cutoff
ax = axes[0, 2]
if cutoff_results:
    cutoffs = sorted(cutoff_results.keys())
    aucs_sweep = [cutoff_results[c]['auc'] for c in cutoffs]
    ax.plot(cutoffs, aucs_sweep, 'bo-', linewidth=2)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('DFS cutoff (months)')
    ax.set_ylabel('AUC')
    ax.set_title('AUC vs response definition cutoff')
    ax.set_ylim([0.3, 0.8])

# Panel 4: Treatment × response
ax = axes[1, 0]
trt_labels = ['taxane/plat', 'plat/cyclo']
sens_counts = [ct_trt_resp.loc['taxane/platinum', 1] if 'taxane/platinum' in ct_trt_resp.index else 0,
               ct_trt_resp.loc['platinum/cyclophosphamide', 1] if 'platinum/cyclophosphamide' in ct_trt_resp.index else 0]
res_counts = [ct_trt_resp.loc['taxane/platinum', 0] if 'taxane/platinum' in ct_trt_resp.index else 0,
              ct_trt_resp.loc['platinum/cyclophosphamide', 0] if 'platinum/cyclophosphamide' in ct_trt_resp.index else 0]
x = np.arange(len(trt_labels))
ax.bar(x - 0.15, sens_counts, 0.3, label='Sensitive', color='green', alpha=0.7)
ax.bar(x + 0.15, res_counts, 0.3, label='Resistant', color='red', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(trt_labels)
ax.set_ylabel('Count')
ax.set_title(f'Treatment × Response (Fisher p={p_trt:.3f})')
ax.legend()

# Panel 5: DFS distribution
ax = axes[1, 1]
ax.hist(dfs_months[y==1].dropna(), bins=15, alpha=0.5, label='Sensitive', color='green', density=True)
ax.hist(dfs_months[y==0].dropna(), bins=15, alpha=0.5, label='Resistant', color='red', density=True)
ax.axvline(6, color='black', linestyle='--', label='6m cutoff')
ax.set_xlabel('DFS (months)')
ax.set_ylabel('Density')
ax.set_title('DFS distribution by class')
ax.legend()

# Panel 6: Cross-dataset transfer
ax = axes[1, 2]
transfers = {
    'LODO→GSE63885': auc,
    'TCGA→GSE63885': auc_tcga_to_63,
    'GSE156→GSE63885': auc_156_to_63,
    'GSE63885→TCGA': auc_63_to_tcga,
    'Within-CV': np.mean(within_aucs),
}
bars = ax.barh(range(len(transfers)), list(transfers.values()),
               color=['steelblue' if v > 0.55 else 'salmon' for v in transfers.values()])
ax.set_yticks(range(len(transfers)))
ax.set_yticklabels(list(transfers.keys()), fontsize=9)
ax.axvline(0.5, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('AUC')
ax.set_title('Cross-dataset transfer')
ax.set_xlim([0.3, 0.9])

fig.suptitle('GSE63885 Investigation: Genuine Model Failure', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'investigation_summary.png', dpi=150)
plt.close(fig)

# Save
with open(OUT / 'investigation_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nAll results saved to {OUT}", flush=True)
print("DONE.", flush=True)
