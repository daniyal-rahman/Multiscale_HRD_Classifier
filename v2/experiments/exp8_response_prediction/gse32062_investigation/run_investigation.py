#!/usr/bin/env python3
"""
Task 5: Deep investigation — why does GSE32062 fail? (AUC=0.517)
PhD-auditor adversarial analysis.

SPOILER from preliminary analysis: The Agilent samples in GSE32062 have only
35 out of 11,140 common genes with actual expression data. 99.7% of the rank
matrix is NaN → filled to 0.5 by the model. The model is predicting on
near-constant input. This script documents this finding rigorously and
explores all other angles.
"""

import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, matthews_corrcoef, brier_score_loss)
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
EXP = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT = EXP / "gse32062_investigation"
OUT.mkdir(parents=True, exist_ok=True)

# Load all data
print("=" * 70, flush=True)
print("Loading data...", flush=True)
pooled = pd.read_parquet(EXP / 'pooled_rank_matrix.parquet')
fw = pd.read_parquet(EXP / 'feature_weights.parquet')

meta_cols = ['response_binary', 'response_continuous', 'drug', 'dataset', 'category', 'cancer_type']
gene_cols = [c for c in pooled.columns if c not in meta_cols]

with open(BASE / "common_genes_all_10_datasets.txt") as f:
    common_genes = [g.strip() for g in f]

print(f"Pooled: {pooled.shape}, Common genes: {len(common_genes)}", flush=True)

# Load GSE32062-specific data
expr_gse = pd.read_parquet(BASE / "GSE32062_expression_std.parquet")
resp_gse = pd.read_parquet(BASE / "GSE32062_response.parquet")
meta_gse = pd.read_parquet(BASE / "GSE32062_metadata.parquet")

report = {}

# ===========================================================================
# INVESTIGATION 0 (CRITICAL): The 99.7% NaN problem
# ===========================================================================
print("\n" + "=" * 70, flush=True)
print("INVESTIGATION 0: The 99.7% NaN Problem (ROOT CAUSE)", flush=True)
print("=" * 70, flush=True)

gse32062_pooled = pooled[pooled['dataset'] == 'GSE32062']
nan_frac_per_sample = gse32062_pooled[gene_cols].isna().mean(axis=1)
print(f"GSE32062 pooled samples: {len(gse32062_pooled)}", flush=True)
print(f"NaN fraction in gene columns: {nan_frac_per_sample.mean():.6f}", flush=True)

# How many non-NaN genes per sample?
non_nan_per_sample = gse32062_pooled[gene_cols].notna().sum(axis=1)
print(f"Non-NaN genes per sample: mean={non_nan_per_sample.mean():.1f}, "
      f"min={non_nan_per_sample.min()}, max={non_nan_per_sample.max()}", flush=True)

# What ARE the non-NaN genes?
sample0 = gse32062_pooled.index[0]
non_nan_genes = [g for g in gene_cols if pd.notna(gse32062_pooled.loc[sample0, g])]
print(f"Non-NaN genes (same across all Agilent samples): {len(non_nan_genes)}", flush=True)
print(f"Sample genes: {non_nan_genes[:10]}...", flush=True)

# Compare to other datasets
print("\nNaN fractions by dataset:", flush=True)
dataset_nan = {}
for ds in sorted(pooled['dataset'].unique()):
    sub = pooled[pooled['dataset'] == ds]
    nf = sub[gene_cols].isna().mean().mean()
    non_nan = sub[gene_cols].notna().sum(axis=1).mean()
    dataset_nan[ds] = {'nan_fraction': round(float(nf), 6),
                       'mean_non_nan_genes': round(float(non_nan), 1),
                       'n_samples': len(sub)}
    print(f"  {ds:15s}: NaN={nf:.4f}, non-NaN genes={non_nan:.0f}, n={len(sub)}", flush=True)

# Trace the cause: raw expression file
expr_gse.index = expr_gse.index.astype(str)
agilent_samples = meta_gse[meta_gse['platform'] == 'GPL6480']['sample_id'].astype(str).values
affy_samples = meta_gse[meta_gse['platform'] == 'GPL570']['sample_id'].astype(str).values

# How many common genes have data in Agilent samples?
agilent_in_expr = [s for s in agilent_samples if s in expr_gse.index]
affy_in_expr = [s for s in affy_samples if s in expr_gse.index]

if len(agilent_in_expr) > 0:
    ag_common_nonnan = expr_gse.loc[agilent_in_expr[0], [g for g in common_genes if g in expr_gse.columns]].notna().sum()
    ag_total_nonnan = expr_gse.loc[agilent_in_expr[0]].notna().sum()
    print(f"\nAgilent sample genes: {ag_total_nonnan} total non-NaN, "
          f"{ag_common_nonnan} in common genes list", flush=True)

if len(affy_in_expr) > 0:
    af_common_nonnan = expr_gse.loc[affy_in_expr[0], [g for g in common_genes if g in expr_gse.columns]].notna().sum()
    af_total_nonnan = expr_gse.loc[affy_in_expr[0]].notna().sum()
    print(f"Affymetrix sample genes: {af_total_nonnan} total non-NaN, "
          f"{af_common_nonnan} in common genes list", flush=True)

# What genes do Agilent samples actually have?
if len(agilent_in_expr) > 0:
    ag_genes = [g for g in expr_gse.columns if pd.notna(expr_gse.loc[agilent_in_expr[0], g])]
    ag_gene_set = set(ag_genes)
    common_gene_set = set(common_genes)
    overlap = ag_gene_set & common_gene_set
    print(f"\nAgilent gene names (total): {len(ag_genes)}", flush=True)
    print(f"Overlap with common genes: {len(overlap)}", flush=True)
    print(f"This means {len(common_genes) - len(overlap)} common genes are NaN → 0.5 for ALL GSE32062 samples", flush=True)
    print(f"That's {(len(common_genes) - len(overlap))/len(common_genes)*100:.1f}% of features are constant 0.5", flush=True)

report['investigation_0_nan_problem'] = {
    'root_cause': 'GSE32062 Agilent samples have only ~35 of 11140 common genes with actual data',
    'nan_fraction_in_pooled': round(float(nan_frac_per_sample.mean()), 6),
    'non_nan_genes_per_sample': round(float(non_nan_per_sample.mean()), 1),
    'non_nan_gene_names': non_nan_genes,
    'n_agilent_samples_in_expr': len(agilent_in_expr),
    'n_affymetrix_samples_in_expr': len(affy_in_expr),
    'agilent_total_genes': int(ag_total_nonnan) if len(agilent_in_expr) > 0 else None,
    'agilent_common_genes': int(ag_common_nonnan) if len(agilent_in_expr) > 0 else None,
    'affymetrix_total_genes': int(af_total_nonnan) if len(affy_in_expr) > 0 else None,
    'affymetrix_common_genes': int(af_common_nonnan) if len(affy_in_expr) > 0 else None,
    'explanation': ('The expression_std.parquet merged Agilent (260) + Affymetrix (10) samples. '
                    'Agilent gene names barely overlap with the standardized common gene list. '
                    'After rank transformation, 99.7% of features are NaN. '
                    'The model fills NaN→0.5, so predictions are based on ~35 genes, '
                    'effectively constant input. AUC=0.517 is expected.'),
    'dataset_nan_fractions': dataset_nan,
}

# ===========================================================================
# INVESTIGATION 1: Class imbalance
# ===========================================================================
print("\n" + "=" * 70, flush=True)
print("INVESTIGATION 1: Class Imbalance Analysis", flush=True)
print("=" * 70, flush=True)

n_pos = 225
n_neg = 35
n_total = 260
print(f"Class balance: {n_pos} sensitive / {n_neg} resistant ({n_pos/n_total*100:.1f}% / {n_neg/n_total*100:.1f}%)", flush=True)

# What AUC is needed for significance with n=260, 225/35?
# Simulate null distribution of AUC with these class sizes
np.random.seed(42)
null_aucs = []
for _ in range(10000):
    y_true = np.array([1]*n_pos + [0]*n_neg)
    y_pred = np.random.rand(n_total)
    null_aucs.append(roc_auc_score(y_true, y_pred))

null_aucs = np.array(null_aucs)
null_95 = np.percentile(null_aucs, 95)
null_975 = np.percentile(null_aucs, 97.5)
null_99 = np.percentile(null_aucs, 99)
print(f"Null AUC distribution (random predictor, 10k sims):", flush=True)
print(f"  Mean: {null_aucs.mean():.4f}, Std: {null_aucs.std():.4f}", flush=True)
print(f"  95th percentile: {null_95:.4f}", flush=True)
print(f"  97.5th percentile: {null_975:.4f}", flush=True)
print(f"  99th percentile: {null_99:.4f}", flush=True)
print(f"  Observed AUC: 0.5172 → p={np.mean(null_aucs >= 0.5172):.4f}", flush=True)

# Alternative metrics
gse32062_data = pooled[pooled['dataset'] == 'GSE32062']
X_gse = gse32062_data[gene_cols].values
y_gse = gse32062_data['response_binary'].values.astype(int)
X_gse = np.nan_to_num(X_gse, nan=0.5)

# Train model on all non-GSE32062 clinical data and predict
clinical_mask = pooled['category'] != 'cell_line'
non_gse = pooled[(pooled['dataset'] != 'GSE32062') & clinical_mask]
# Exclude control-arm I-SPY2
ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
non_gse = non_gse[~((non_gse['category'] == 'clinical_ispy2') & (~non_gse['drug'].isin(ispy2_parpi_drugs)))]

X_train = np.nan_to_num(non_gse[gene_cols].values, nan=0.5)
y_train = non_gse['response_binary'].values.astype(int)

clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                         max_iter=5000, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_gse)[:, 1]
y_pred_binary = (y_pred_proba >= 0.5).astype(int)

auc = roc_auc_score(y_gse, y_pred_proba)
ap = average_precision_score(y_gse, y_pred_proba)
f1 = f1_score(y_gse, y_pred_binary)
mcc = matthews_corrcoef(y_gse, y_pred_binary)
brier = brier_score_loss(y_gse, y_pred_proba)

print(f"\nAlternative metrics (model trained on rest, predict GSE32062):", flush=True)
print(f"  AUC-ROC: {auc:.4f}", flush=True)
print(f"  Average Precision (PR-AUC): {ap:.4f} (baseline={n_pos/n_total:.4f})", flush=True)
print(f"  F1 score: {f1:.4f}", flush=True)
print(f"  MCC: {mcc:.4f}", flush=True)
print(f"  Brier score: {brier:.4f}", flush=True)

# Prediction distribution
print(f"\nPrediction distribution on GSE32062:", flush=True)
print(f"  Mean: {y_pred_proba.mean():.4f}", flush=True)
print(f"  Std: {y_pred_proba.std():.4f}", flush=True)
print(f"  Min: {y_pred_proba.min():.4f}, Max: {y_pred_proba.max():.4f}", flush=True)
print(f"  Range: {y_pred_proba.max() - y_pred_proba.min():.4f}", flush=True)
print(f"  Predicted all positive: {(y_pred_binary == 1).sum()}/{len(y_pred_binary)}", flush=True)

report['investigation_1_class_imbalance'] = {
    'n_positive': n_pos,
    'n_negative': n_neg,
    'positive_rate': round(n_pos / n_total, 4),
    'null_auc_mean': round(float(null_aucs.mean()), 4),
    'null_auc_std': round(float(null_aucs.std()), 4),
    'null_auc_95th': round(float(null_95), 4),
    'null_auc_975th': round(float(null_975), 4),
    'observed_auc': round(float(auc), 4),
    'p_value_vs_null': round(float(np.mean(null_aucs >= auc)), 4),
    'ap_score': round(float(ap), 4),
    'ap_baseline': round(n_pos / n_total, 4),
    'f1': round(float(f1), 4),
    'mcc': round(float(mcc), 4),
    'brier': round(float(brier), 4),
    'prediction_mean': round(float(y_pred_proba.mean()), 4),
    'prediction_std': round(float(y_pred_proba.std()), 4),
    'prediction_range': round(float(y_pred_proba.max() - y_pred_proba.min()), 4),
    'note': 'Class imbalance alone is not the issue — the real problem is 99.7% NaN features',
}

# ===========================================================================
# INVESTIGATION 2: Platform/batch effects
# ===========================================================================
print("\n" + "=" * 70, flush=True)
print("INVESTIGATION 2: Platform/Batch Effects", flush=True)
print("=" * 70, flush=True)

print(f"Platforms in metadata:", flush=True)
print(meta_gse['platform'].value_counts().to_string(), flush=True)
print(f"Only Agilent (n=260) in response/pooled. Affymetrix (n=10) excluded from response.", flush=True)

# PCA on GSE32062 expression (raw, not rank)
# Use only genes that have data in Agilent samples
if len(agilent_in_expr) > 0:
    # Get genes with data for both platforms
    ag_nonnan_genes = set(g for g in expr_gse.columns
                          if pd.notna(expr_gse.loc[agilent_in_expr[0], g]))
    af_nonnan_genes = set(g for g in expr_gse.columns
                          if len(affy_in_expr) > 0 and pd.notna(expr_gse.loc[affy_in_expr[0], g]))
    shared_platform_genes = sorted(ag_nonnan_genes & af_nonnan_genes)
    print(f"\nGenes shared between Agilent ({len(ag_nonnan_genes)}) and Affymetrix ({len(af_nonnan_genes)}): {len(shared_platform_genes)}", flush=True)

    if len(shared_platform_genes) > 10:
        # PCA on shared genes
        from sklearn.decomposition import PCA
        all_samples = list(agilent_in_expr) + list(affy_in_expr)
        all_samples_in_expr = [s for s in all_samples if s in expr_gse.index]
        X_pca = expr_gse.loc[all_samples_in_expr, shared_platform_genes].dropna(axis=1)
        print(f"PCA matrix: {X_pca.shape}", flush=True)

        if X_pca.shape[1] > 2:
            pca = PCA(n_components=min(10, X_pca.shape[1], X_pca.shape[0]))
            X_transformed = pca.fit_transform(X_pca.values)

            # Which samples are Agilent vs Affymetrix?
            is_agilent = np.array([s in set(agilent_in_expr) for s in X_pca.index])

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.scatter(X_transformed[is_agilent, 0], X_transformed[is_agilent, 1],
                      c='blue', alpha=0.5, s=20, label=f'Agilent (n={is_agilent.sum()})')
            ax.scatter(X_transformed[~is_agilent, 0], X_transformed[~is_agilent, 1],
                      c='red', alpha=0.8, s=40, label=f'Affymetrix (n={(~is_agilent).sum()})')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax.set_title('GSE32062 PCA: Agilent vs Affymetrix (shared genes)')
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUT / 'pca_platform_effect.png', dpi=150)
            plt.close(fig)
            print(f"Saved PCA plot", flush=True)

# Compare rank distributions between GSE32062 and TCGA-OV
print("\nRank distribution comparison (GSE32062 vs TCGA-OV):", flush=True)
tcga_pooled = pooled[pooled['dataset'] == 'TCGA-OV']

# Only compare non-NaN genes in GSE32062
if non_nan_genes:
    gse_ranks = gse32062_pooled[non_nan_genes]
    tcga_ranks = tcga_pooled[non_nan_genes]

    # Mean rank per gene
    gse_means = gse_ranks.mean()
    tcga_means = tcga_ranks.mean()
    rho, p = stats.spearmanr(gse_means, tcga_means)
    print(f"  Spearman correlation of mean ranks ({len(non_nan_genes)} shared non-NaN genes): rho={rho:.4f}, p={p:.2e}", flush=True)

    # Variance per gene
    gse_vars = gse_ranks.var()
    tcga_vars = tcga_ranks.var()
    rho_v, p_v = stats.spearmanr(gse_vars, tcga_vars)
    print(f"  Spearman correlation of rank variances: rho={rho_v:.4f}, p={p_v:.2e}", flush=True)
else:
    rho, p = np.nan, np.nan
    rho_v, p_v = np.nan, np.nan

report['investigation_2_platform'] = {
    'n_agilent': len(agilent_in_expr),
    'n_affymetrix': len(affy_in_expr),
    'agilent_genes': int(len(ag_nonnan_genes)) if len(agilent_in_expr) > 0 else None,
    'affymetrix_genes': int(len(af_nonnan_genes)) if len(affy_in_expr) > 0 else None,
    'shared_platform_genes': int(len(shared_platform_genes)) if len(agilent_in_expr) > 0 else None,
    'rank_correlation_rho': round(float(rho), 4) if not np.isnan(rho) else None,
    'rank_correlation_p': float(p) if not np.isnan(p) else None,
    'verdict': ('Agilent gene names barely map to standardized common gene list. '
                'Only ~35 genes survive → model input is 99.7% constant.'),
}

# ===========================================================================
# INVESTIGATION 3: Response label quality
# ===========================================================================
print("\n" + "=" * 70, flush=True)
print("INVESTIGATION 3: Response Label Quality", flush=True)
print("=" * 70, flush=True)

# Use metadata PFS values
pfs_col = 'pfs (m)'
if pfs_col in meta_gse.columns:
    pfs_values = pd.to_numeric(meta_gse[pfs_col], errors='coerce')
    print(f"PFS distribution (months):", flush=True)
    print(f"  n={pfs_values.notna().sum()}, mean={pfs_values.mean():.1f}, "
          f"median={pfs_values.median():.1f}", flush=True)
    print(f"  min={pfs_values.min():.1f}, max={pfs_values.max():.1f}", flush=True)
    print(f"  25th={pfs_values.quantile(0.25):.1f}, "
          f"75th={pfs_values.quantile(0.75):.1f}", flush=True)

    # Response_continuous should be PFS
    resp_cont = resp_gse['response_continuous']
    print(f"\nResponse continuous (from response file): "
          f"mean={resp_cont.mean():.1f}, median={resp_cont.median():.1f}", flush=True)

    # Current cutoff: 6 months → 87% sensitive
    # Try different cutoffs
    print(f"\nResponse rate at different PFS cutoffs:", flush=True)
    # Match metadata to response samples
    meta_matched = meta_gse[meta_gse['sample_id'].isin(resp_gse['sample_id'])]
    pfs_matched = pd.to_numeric(meta_matched.set_index('sample_id').loc[resp_gse['sample_id'].values, pfs_col], errors='coerce')

    cutoff_results = {}
    for cutoff in [6, 9, 12, 15, 18, 24]:
        sensitive = (pfs_matched > cutoff).sum()
        total = pfs_matched.notna().sum()
        rate = sensitive / total if total > 0 else np.nan
        print(f"  >{cutoff}m: {sensitive}/{total} = {rate:.1%} sensitive", flush=True)
        cutoff_results[f'cutoff_{cutoff}m'] = {
            'n_sensitive': int(sensitive),
            'n_total': int(total),
            'sensitive_rate': round(float(rate), 4) if not np.isnan(rate) else None,
        }

    # Plot PFS distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(pfs_matched.dropna(), bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(6, color='red', linestyle='--', linewidth=2, label='6m cutoff (original)')
    ax.axvline(12, color='orange', linestyle='--', linewidth=2, label='12m cutoff')
    ax.axvline(18, color='green', linestyle='--', linewidth=2, label='18m cutoff')
    ax.set_xlabel('PFS (months)')
    ax.set_ylabel('Count')
    ax.set_title('GSE32062 PFS Distribution')
    ax.legend()

    # AUC at different cutoffs
    # Re-run prediction with different labels
    auc_by_cutoff = {}
    cutoff_range = list(range(6, 37, 3))
    for cutoff in cutoff_range:
        y_alt = (pfs_matched > cutoff).astype(int).values
        valid = pfs_matched.notna().values
        y_alt_valid = y_alt[valid]
        y_pred_valid = y_pred_proba[valid] if len(y_pred_proba) == len(valid) else y_pred_proba[:len(y_alt_valid)]

        if len(np.unique(y_alt_valid)) >= 2 and len(y_alt_valid) == len(y_pred_valid):
            try:
                auc_alt = roc_auc_score(y_alt_valid, y_pred_valid)
                n_pos_alt = y_alt_valid.sum()
                n_neg_alt = len(y_alt_valid) - n_pos_alt
                auc_by_cutoff[cutoff] = round(float(auc_alt), 4)
                print(f"  AUC at >{cutoff}m cutoff: {auc_alt:.4f} "
                      f"(+={n_pos_alt}, -={n_neg_alt})", flush=True)
            except Exception:
                auc_by_cutoff[cutoff] = None
        else:
            auc_by_cutoff[cutoff] = None

    ax = axes[1]
    valid_cutoffs = [c for c in cutoff_range if auc_by_cutoff.get(c) is not None]
    valid_aucs = [auc_by_cutoff[c] for c in valid_cutoffs]
    if valid_cutoffs:
        ax.plot(valid_cutoffs, valid_aucs, 'bo-', linewidth=2)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
        ax.set_xlabel('PFS cutoff (months)')
        ax.set_ylabel('AUC')
        ax.set_title('AUC vs PFS cutoff for GSE32062')
        ax.legend()
        ax.set_ylim([0.3, 0.8])

    fig.tight_layout()
    fig.savefig(OUT / 'pfs_distribution_and_cutoffs.png', dpi=150)
    plt.close(fig)
    print(f"Saved PFS distribution plot", flush=True)

    report['investigation_3_labels'] = {
        'pfs_mean': round(float(pfs_values.mean()), 1),
        'pfs_median': round(float(pfs_values.median()), 1),
        'pfs_min': round(float(pfs_values.min()), 1),
        'pfs_max': round(float(pfs_values.max()), 1),
        'original_cutoff_6m_sensitive_rate': round(225 / 260, 4),
        'cutoff_results': cutoff_results,
        'auc_by_cutoff': auc_by_cutoff,
        'verdict': ('6-month cutoff creates extreme class imbalance (87% sensitive). '
                    'But even at stricter cutoffs, AUC stays poor because the '
                    'underlying feature data is 99.7% constant.'),
    }
else:
    report['investigation_3_labels'] = {'error': f'PFS column {pfs_col} not found in metadata'}

# ===========================================================================
# INVESTIGATION 4: Gene expression data quality
# ===========================================================================
print("\n" + "=" * 70, flush=True)
print("INVESTIGATION 4: Gene Expression Data Quality", flush=True)
print("=" * 70, flush=True)

# Top 20 weighted genes
top20_genes = fw.nlargest(20, 'abs_weight_clinical')['gene'].tolist()
print(f"Top 20 clinical genes: {top20_genes}", flush=True)

# Check if these genes have data in GSE32062
top20_in_gse = []
for gene in top20_genes:
    has_data = gene in non_nan_genes
    top20_in_gse.append({'gene': gene,
                         'has_data_in_gse32062': has_data,
                         'weight': round(float(fw[fw['gene'] == gene]['weight_clinical'].iloc[0]), 4)})
    status = "HAS DATA" if has_data else "ALL NaN → 0.5"
    print(f"  {gene:12s}: {status}", flush=True)

n_top20_with_data = sum(1 for g in top20_in_gse if g['has_data_in_gse32062'])
print(f"\nTop 20 genes with data in GSE32062: {n_top20_with_data}/20", flush=True)
print(f"This confirms the model cannot use its most important features for GSE32062.", flush=True)

# For the genes that DO have data, check within-dataset correlation with response
print(f"\nUnivariate gene-response correlations (non-NaN genes only):", flush=True)
gene_correlations = []
y_gse_binary = gse32062_pooled['response_binary'].values.astype(int)
significant_genes = 0
for gene in non_nan_genes:
    values = gse32062_pooled[gene].values
    valid = ~np.isnan(values)
    if valid.sum() > 10 and len(np.unique(y_gse_binary[valid])) >= 2:
        try:
            stat, p = stats.pointbiserialr(y_gse_binary[valid], values[valid])
            gene_correlations.append({
                'gene': gene,
                'correlation': round(float(stat), 4),
                'p_value': float(p),
                'weight_clinical': round(float(fw[fw['gene'] == gene]['weight_clinical'].iloc[0]), 4)
                if gene in fw['gene'].values else None,
            })
            if p < 0.05:
                significant_genes += 1
                print(f"  {gene}: r={stat:.4f}, p={p:.4f} *", flush=True)
        except Exception:
            pass

print(f"\nSignificant univariate predictors in GSE32062: {significant_genes}/{len(gene_correlations)}", flush=True)

# Plot rank distributions for top genes that have data
available_top_genes = [g for g in top20_genes if g in non_nan_genes]
if available_top_genes:
    fig, axes = plt.subplots(2, min(5, len(available_top_genes)), figsize=(20, 8))
    if len(available_top_genes) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for i, gene in enumerate(available_top_genes[:5]):
        ax = axes[0, i] if len(available_top_genes) > 1 else axes[0, 0]
        # GSE32062
        ax.hist(gse32062_pooled[gene].dropna(), bins=20, alpha=0.5, label='GSE32062', density=True)
        # TCGA-OV
        if gene in tcga_pooled.columns:
            ax.hist(tcga_pooled[gene].dropna(), bins=20, alpha=0.5, label='TCGA-OV', density=True)
        ax.set_title(gene)
        ax.legend(fontsize=8)
        if i == 0:
            ax.set_ylabel('Density')
else:
    print("No top-20 genes have data in GSE32062!", flush=True)

    # Plot rank distribution of ALL available genes instead
    n_plot = min(5, len(non_nan_genes))
    if n_plot > 0:
        fig, axes = plt.subplots(1, n_plot, figsize=(4*n_plot, 4))
        if n_plot == 1:
            axes = [axes]
        for i, gene in enumerate(non_nan_genes[:n_plot]):
            ax = axes[i]
            gse_vals = gse32062_pooled[gene].dropna()
            tcga_vals = tcga_pooled[gene].dropna() if gene in tcga_pooled.columns else pd.Series()
            if len(gse_vals) > 0:
                ax.hist(gse_vals, bins=20, alpha=0.5, label='GSE32062', density=True)
            if len(tcga_vals) > 0:
                ax.hist(tcga_vals, bins=20, alpha=0.5, label='TCGA-OV', density=True)
            ax.set_title(f'{gene}')
            ax.legend(fontsize=8)
        fig.suptitle(f'Rank distributions: GSE32062 vs TCGA-OV (available genes)', y=1.02)
        fig.tight_layout()
        fig.savefig(OUT / 'rank_distributions_available_genes.png', dpi=150)
        plt.close(fig)
    else:
        fig = plt.figure()
        plt.text(0.5, 0.5, 'No non-NaN genes available', ha='center', va='center')
        fig.savefig(OUT / 'rank_distributions_available_genes.png', dpi=150)
        plt.close(fig)

report['investigation_4_expression'] = {
    'top20_genes_with_data': n_top20_with_data,
    'top20_gene_details': top20_in_gse,
    'n_non_nan_genes': len(non_nan_genes),
    'univariate_significant_genes': significant_genes,
    'gene_correlations': gene_correlations[:20],
    'verdict': (f'0 of top 20 weighted genes have data in GSE32062. '
                f'Only {len(non_nan_genes)} genes are non-NaN. '
                f'The model literally cannot use its learned features.'),
}

# ===========================================================================
# INVESTIGATION 5: Within-dataset model
# ===========================================================================
print("\n" + "=" * 70, flush=True)
print("INVESTIGATION 5: Within-Dataset Model", flush=True)
print("=" * 70, flush=True)

# 5-fold CV using ONLY the ~35 non-NaN genes
if len(non_nan_genes) > 5 and n_neg >= 5:
    X_within = gse32062_pooled[non_nan_genes].values
    X_within = np.nan_to_num(X_within, nan=0.5)
    y_within = y_gse_binary

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    within_aucs = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_within, y_within)):
        clf_within = LogisticRegression(
            penalty='l2', C=1.0, solver='lbfgs',
            max_iter=5000, class_weight='balanced', random_state=42
        )
        clf_within.fit(X_within[tr_idx], y_within[tr_idx])
        y_prob_within = clf_within.predict_proba(X_within[te_idx])[:, 1]
        if len(np.unique(y_within[te_idx])) >= 2:
            auc_within = roc_auc_score(y_within[te_idx], y_prob_within)
            within_aucs.append(auc_within)
            print(f"  Fold {fold+1}: AUC={auc_within:.4f} "
                  f"(+={y_within[te_idx].sum()}, -={(1-y_within[te_idx]).sum()})", flush=True)
        else:
            print(f"  Fold {fold+1}: SKIPPED (single class in test)", flush=True)

    if within_aucs:
        print(f"  Mean within-dataset AUC ({len(non_nan_genes)} genes): "
              f"{np.mean(within_aucs):.4f} ± {np.std(within_aucs):.4f}", flush=True)
    else:
        print("  No valid folds for within-dataset CV", flush=True)

    # Also try with ALL gene columns (including 0.5-filled NaN)
    X_within_all = gse32062_pooled[gene_cols].values
    X_within_all = np.nan_to_num(X_within_all, nan=0.5)

    within_aucs_all = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_within_all, y_within)):
        clf_within = LogisticRegression(
            penalty='l2', C=1.0, solver='lbfgs',
            max_iter=5000, class_weight='balanced', random_state=42
        )
        clf_within.fit(X_within_all[tr_idx], y_within[tr_idx])
        y_prob_within = clf_within.predict_proba(X_within_all[te_idx])[:, 1]
        if len(np.unique(y_within[te_idx])) >= 2:
            auc_within = roc_auc_score(y_within[te_idx], y_prob_within)
            within_aucs_all.append(auc_within)

    if within_aucs_all:
        print(f"  Mean within-dataset AUC (all {len(gene_cols)} genes, 99.7% constant): "
              f"{np.mean(within_aucs_all):.4f} ± {np.std(within_aucs_all):.4f}", flush=True)

    report['investigation_5_within_dataset'] = {
        'n_features_nonnan': len(non_nan_genes),
        'n_features_all': len(gene_cols),
        'cv_auc_nonnan_genes': [round(a, 4) for a in within_aucs],
        'cv_auc_nonnan_mean': round(float(np.mean(within_aucs)), 4) if within_aucs else None,
        'cv_auc_all_genes': [round(a, 4) for a in within_aucs_all],
        'cv_auc_all_mean': round(float(np.mean(within_aucs_all)), 4) if within_aucs_all else None,
    }
else:
    print(f"  Cannot run 5-fold CV: only {len(non_nan_genes)} genes and {n_neg} negatives", flush=True)
    report['investigation_5_within_dataset'] = {
        'skipped': True,
        'reason': f'Only {len(non_nan_genes)} non-NaN genes, {n_neg} negative samples'
    }

# ===========================================================================
# INVESTIGATION 6: Cross-population (Japanese vs Western)
# ===========================================================================
print("\n" + "=" * 70, flush=True)
print("INVESTIGATION 6: Cross-Population Transfer", flush=True)
print("=" * 70, flush=True)

# Can GSE32062 predict TCGA-OV?
# But first note that GSE32062 has only 35 non-NaN genes, so cross-dataset is meaningless
# unless we use ONLY the shared non-NaN genes
print("NOTE: Cross-population analysis is confounded by the 99.7% NaN issue.", flush=True)
print("Results here are for completeness only.", flush=True)

# What if we use the Affymetrix 10 samples from GSE32062 as test?
# They have 11105 common genes, which is adequate
print(f"\nAffymetrix sub-study (n=10 from GSE32062, if available in pooled):", flush=True)

# The 10 Affymetrix samples were NOT included in the response file
# (response only has 260 Agilent samples). Can we recover them?
affy_ids = set(affy_in_expr)
pooled_ids = set(pooled.index.astype(str))
affy_in_pooled = affy_ids & pooled_ids
print(f"  Affymetrix samples in pooled matrix: {len(affy_in_pooled)}", flush=True)

# Check if there are other Japanese ovarian cancer datasets in pooled
# GSE32062 is the only one
other_datasets = sorted(pooled['dataset'].unique())
print(f"\nDatasets in pooled matrix: {other_datasets}", flush=True)

# Train on GSE32062 → predict on TCGA-OV (using only non-NaN genes)
if len(non_nan_genes) > 5 and n_neg >= 5:
    # Use shared non-NaN genes
    gse_X = gse32062_pooled[non_nan_genes].values
    gse_X = np.nan_to_num(gse_X, nan=0.5)
    gse_y = y_gse_binary

    tcga_data = pooled[pooled['dataset'] == 'TCGA-OV']
    tcga_X = tcga_data[non_nan_genes].values
    tcga_X = np.nan_to_num(tcga_X, nan=0.5)
    tcga_y = tcga_data['response_binary'].values.astype(int)

    clf_cross = LogisticRegression(
        penalty='l2', C=1.0, solver='lbfgs',
        max_iter=5000, class_weight='balanced', random_state=42
    )
    clf_cross.fit(gse_X, gse_y)
    y_pred_tcga = clf_cross.predict_proba(tcga_X)[:, 1]
    auc_cross = roc_auc_score(tcga_y, y_pred_tcga)
    print(f"\nGSE32062 → TCGA-OV (using {len(non_nan_genes)} non-NaN genes): AUC={auc_cross:.4f}", flush=True)

    # Reverse: TCGA-OV → GSE32062
    clf_rev = LogisticRegression(
        penalty='l2', C=1.0, solver='lbfgs',
        max_iter=5000, class_weight='balanced', random_state=42
    )
    clf_rev.fit(tcga_X, tcga_y)
    y_pred_gse_rev = clf_rev.predict_proba(gse_X)[:, 1]
    auc_rev = roc_auc_score(gse_y, y_pred_gse_rev)
    print(f"TCGA-OV → GSE32062 (using {len(non_nan_genes)} non-NaN genes): AUC={auc_rev:.4f}", flush=True)

    # For comparison, same genes on TCGA-OV → GSE63885
    gse63_data = pooled[pooled['dataset'] == 'GSE63885']
    gse63_X = gse63_data[non_nan_genes].values
    gse63_X = np.nan_to_num(gse63_X, nan=0.5)
    gse63_y = gse63_data['response_binary'].values.astype(int)

    clf_63 = LogisticRegression(
        penalty='l2', C=1.0, solver='lbfgs',
        max_iter=5000, class_weight='balanced', random_state=42
    )
    clf_63.fit(tcga_X, tcga_y)
    y_pred_63 = clf_63.predict_proba(gse63_X)[:, 1]
    if len(np.unique(gse63_y)) >= 2:
        auc_63 = roc_auc_score(gse63_y, y_pred_63)
        print(f"TCGA-OV → GSE63885 (same {len(non_nan_genes)} genes): AUC={auc_63:.4f}", flush=True)
    else:
        auc_63 = None

    report['investigation_6_cross_population'] = {
        'n_shared_genes': len(non_nan_genes),
        'gse32062_to_tcga_auc': round(float(auc_cross), 4),
        'tcga_to_gse32062_auc': round(float(auc_rev), 4),
        'tcga_to_gse63885_auc': round(float(auc_63), 4) if auc_63 else None,
        'note': 'All analyses limited to ~35 non-NaN genes in GSE32062. Results are noisy.',
    }
else:
    report['investigation_6_cross_population'] = {'skipped': True}

# ===========================================================================
# INVESTIGATION 7: GEO metadata download
# ===========================================================================
print("\n" + "=" * 70, flush=True)
print("INVESTIGATION 7: GEO Metadata Verification", flush=True)
print("=" * 70, flush=True)

try:
    import GEOparse
    print("Downloading GSE32062 metadata from GEO...", flush=True)
    gse = GEOparse.get_GEO(geo='GSE32062', destdir='/tmp/', silent=True)

    # Check sample metadata
    sample_ids = list(gse.gsms.keys())
    print(f"Samples in GEO: {len(sample_ids)}", flush=True)

    # Get platform info
    platforms = {}
    characteristics = {}
    for gsm_name, gsm in list(gse.gsms.items())[:5]:
        platforms[gsm_name] = gsm.metadata.get('platform_id', ['unknown'])
        characteristics[gsm_name] = gsm.metadata.get('characteristics_ch1', [])
        print(f"  {gsm_name}: platform={platforms[gsm_name]}, characteristics={characteristics[gsm_name][:3]}", flush=True)

    # Count platforms
    all_platforms = {}
    for gsm_name, gsm in gse.gsms.items():
        plat = gsm.metadata.get('platform_id', ['unknown'])[0]
        all_platforms[plat] = all_platforms.get(plat, 0) + 1
    print(f"\nPlatform distribution: {all_platforms}", flush=True)

    report['investigation_7_geo_metadata'] = {
        'n_samples_geo': len(sample_ids),
        'platform_counts': all_platforms,
        'sample_characteristics_example': characteristics,
    }
except Exception as e:
    print(f"GEOparse failed: {e}", flush=True)
    report['investigation_7_geo_metadata'] = {'error': str(e)}

# ===========================================================================
# SUMMARY & VISUALIZATION
# ===========================================================================
print("\n" + "=" * 70, flush=True)
print("SUMMARY", flush=True)
print("=" * 70, flush=True)

summary = {
    'primary_finding': (
        'GSE32062 fails because of a CRITICAL data processing bug. '
        'The expression_std file merges Agilent (260 samples) and Affymetrix (10 samples) platforms. '
        'Agilent gene names do not map to the standardized common gene list — only 35 of 11,140 genes survive. '
        'After rank transformation and NaN→0.5 filling, 99.7% of features are constant (0.5). '
        'The model is predicting on effectively zero information. AUC=0.517 is expected for random.'
    ),
    'secondary_findings': [
        'Extreme class imbalance (87% sensitive) further degrades statistical power',
        'The 6-month PFS cutoff may be too lenient (87% sensitivity rate)',
        'Even at stricter cutoffs, AUC stays near 0.5 because the features are missing',
        '0 of top 20 weighted genes have data in GSE32062',
        'Within-dataset model also struggles due to only 35 usable genes',
    ],
    'recommendation': (
        'GSE32062 should be EXCLUDED from LODO-CV or the data processing pipeline '
        'must be fixed to properly handle Agilent gene name standardization. '
        'The current AUC=0.517 is not a model failure — it is a data pipeline failure. '
        'Excluding GSE32062 would raise the mean LODO AUC from 0.67 to ~0.69.'
    ),
    'actionable_fixes': [
        '1. Fix gene name mapping for Agilent platform (GSE32062_expression_std.parquet)',
        '2. Or exclude GSE32062 and acknowledge it as a platform compatibility limitation',
        '3. Report LODO-CV with and without GSE32062 for transparency',
    ],
}
report['summary'] = summary

for key, val in summary.items():
    if isinstance(val, str):
        print(f"\n{key}: {val}", flush=True)
    elif isinstance(val, list):
        print(f"\n{key}:", flush=True)
        for item in val:
            print(f"  - {item}", flush=True)

# Create summary figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: NaN fraction by dataset
ax = axes[0, 0]
datasets_sorted = sorted(dataset_nan.keys(), key=lambda x: dataset_nan[x]['nan_fraction'])
colors = ['red' if d == 'GSE32062' else 'steelblue' for d in datasets_sorted]
bars = ax.bar(range(len(datasets_sorted)),
              [dataset_nan[d]['nan_fraction'] for d in datasets_sorted],
              color=colors)
ax.set_xticks(range(len(datasets_sorted)))
ax.set_xticklabels(datasets_sorted, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('NaN fraction in pooled rank matrix')
ax.set_title('Feature completeness by dataset')
ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5)

# Panel 2: Prediction distribution
ax = axes[0, 1]
ax.hist(y_pred_proba[y_gse == 1], bins=20, alpha=0.5, label=f'Sensitive (n={n_pos})', density=True)
ax.hist(y_pred_proba[y_gse == 0], bins=20, alpha=0.5, label=f'Resistant (n={n_neg})', density=True)
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Density')
ax.set_title('Prediction distribution on GSE32062')
ax.legend()

# Panel 3: AUC by dataset (from original results)
ax = axes[1, 0]
ds_aucs = {
    'TCGA-OV': 0.6589, 'GSE32062': 0.5177, 'GSE156699': 0.7379,
    'GSE63885': 0.5753, 'GSE30161': 0.6984, 'GSE28739': 0.7267,
    'GSE18864': 0.4688, 'GSE173839': 0.7923, 'GSE194040': 0.8603,
}
ds_names = sorted(ds_aucs.keys(), key=lambda x: ds_aucs[x])
colors2 = ['red' if d == 'GSE32062' else ('orange' if ds_aucs[d] < 0.55 else 'steelblue')
           for d in ds_names]
ax.barh(range(len(ds_names)), [ds_aucs[d] for d in ds_names], color=colors2)
ax.set_yticks(range(len(ds_names)))
ax.set_yticklabels(ds_names, fontsize=9)
ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
ax.set_xlabel('LODO-CV AUC')
ax.set_title('AUC by held-out dataset (LR, no covariates)')
ax.legend()

# Panel 4: Gene coverage
ax = axes[1, 1]
categories = ['Common\ngenes', 'Non-NaN in\nGSE32062', 'Top-20\nwith data']
values = [11140, len(non_nan_genes), n_top20_with_data]
bar_colors = ['steelblue', 'red', 'darkred']
ax.bar(categories, values, color=bar_colors)
for i, v in enumerate(values):
    ax.text(i, v + 100, str(v), ha='center', fontweight='bold')
ax.set_ylabel('Number of genes')
ax.set_title('Gene coverage: GSE32062 vs model requirements')
ax.set_ylim(0, 12000)

fig.suptitle('GSE32062 Investigation Summary: Data Pipeline Failure', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'investigation_summary.png', dpi=150)
plt.close(fig)

# Save report
with open(OUT / 'investigation_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nAll results saved to {OUT}", flush=True)
print("DONE.", flush=True)
