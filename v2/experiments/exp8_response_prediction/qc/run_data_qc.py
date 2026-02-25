#!/usr/bin/env python3
"""
Exp8 Data QC: Thorough quality control of the pooled rank-transformed expression matrix.

Checks:
1. PCA colored by dataset — batch effects after rank transform?
2. UMAP colored by dataset — same
3. Per-dataset rank distributions for random genes — should be ~Uniform[0,1]
4. Outlier detection via within-dataset sample correlation
5. GSE32062 platform check (Agilent vs Affymetrix)
6. Class balance analysis with 95% CI on AUC under null
7. Response label verification

Output: plots to qc/, summary to qc/data_qc_report.json
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
RESP_DIR = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
QC_DIR = BASE / "qc"
QC_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────
print("Loading pooled rank matrix...", flush=True)
df = pd.read_parquet(BASE / "pooled_rank_matrix.parquet")
meta_cols = ['response_binary', 'response_continuous', 'drug', 'dataset', 'category', 'cancer_type']
gene_cols = [c for c in df.columns if c not in meta_cols]
X = df[gene_cols].values
datasets = df['dataset'].values
response = df['response_binary'].values

print(f"  Shape: {df.shape[0]} samples x {len(gene_cols)} genes + {len(meta_cols)} metadata cols", flush=True)
print(f"  Datasets: {df['dataset'].value_counts().to_dict()}", flush=True)

report = {}

# ═══════════════════════════════════════════════════════════════════
# 1. PCA colored by dataset
# ═══════════════════════════════════════════════════════════════════
print("\n[1/7] PCA analysis...", flush=True)

# Replace NaN with 0.5 for PCA (median rank)
X_clean = np.nan_to_num(X, nan=0.5)

pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_clean)
explained = pca.explained_variance_ratio_

print(f"  Variance explained by first 10 PCs: {[f'{v:.3f}' for v in explained]}", flush=True)

# PCA plot: PC1 vs PC2
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
unique_ds = sorted(df['dataset'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_ds)))
ds_color_map = {ds: colors[i] for i, ds in enumerate(unique_ds)}

for ax, (pcx, pcy) in zip(axes, [(0, 1), (2, 3)]):
    for ds in unique_ds:
        mask = datasets == ds
        ax.scatter(X_pca[mask, pcx], X_pca[mask, pcy],
                   c=[ds_color_map[ds]], label=ds, alpha=0.5, s=15)
    ax.set_xlabel(f"PC{pcx+1} ({explained[pcx]:.1%})")
    ax.set_ylabel(f"PC{pcy+1} ({explained[pcy]:.1%})")
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title(f"PCA: PC{pcx+1} vs PC{pcy+1}")

fig.suptitle("PCA of Rank-Transformed Pooled Matrix (colored by dataset)", fontsize=13)
plt.tight_layout()
plt.savefig(QC_DIR / "pca_by_dataset.png", dpi=150)
plt.close()

# Check if datasets cluster: compute silhouette-like metric
# Average distance to own-dataset centroid vs other-dataset centroid
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
ds_labels = le.fit_transform(datasets)
sil_score = silhouette_score(X_pca[:, :5], ds_labels, sample_size=min(1000, len(ds_labels)), random_state=42)
print(f"  Silhouette score (dataset labels, PCA 5D): {sil_score:.3f}", flush=True)
print(f"    (>0.3 = strong clustering by dataset = batch effect)", flush=True)
print(f"    (~0 = well-mixed = good harmonization)", flush=True)

report['pca'] = {
    'variance_explained_10pc': [round(float(v), 4) for v in explained],
    'total_var_explained_5pc': round(float(explained[:5].sum()), 4),
    'silhouette_score_5pc': round(float(sil_score), 4),
    'interpretation': 'strong_batch_effect' if sil_score > 0.3 else ('moderate' if sil_score > 0.1 else 'well_mixed')
}

# ═══════════════════════════════════════════════════════════════════
# 2. UMAP colored by dataset
# ═══════════════════════════════════════════════════════════════════
print("\n[2/7] UMAP analysis...", flush=True)

try:
    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='euclidean', random_state=42)
    X_umap = reducer.fit_transform(X_pca[:, :20])  # Use top 20 PCs as input for speed

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Color by dataset
    for ds in unique_ds:
        mask = datasets == ds
        axes[0].scatter(X_umap[mask, 0], X_umap[mask, 1],
                       c=[ds_color_map[ds]], label=ds, alpha=0.5, s=15)
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")
    axes[0].legend(fontsize=7, markerscale=2)
    axes[0].set_title("UMAP colored by dataset")

    # Color by response
    for resp_val, label, color in [(1, 'Responder', 'tab:blue'), (0, 'Non-responder', 'tab:red')]:
        mask = response == resp_val
        axes[1].scatter(X_umap[mask, 0], X_umap[mask, 1],
                       c=color, label=label, alpha=0.4, s=15)
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")
    axes[1].legend(fontsize=9, markerscale=2)
    axes[1].set_title("UMAP colored by response")

    fig.suptitle("UMAP of Rank-Transformed Pooled Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(QC_DIR / "umap_by_dataset.png", dpi=150)
    plt.close()

    # UMAP silhouette
    sil_umap = silhouette_score(X_umap, ds_labels, sample_size=min(1000, len(ds_labels)), random_state=42)
    print(f"  UMAP silhouette score (dataset labels): {sil_umap:.3f}", flush=True)
    report['umap'] = {
        'silhouette_score': round(float(sil_umap), 4),
        'status': 'completed'
    }
except ImportError:
    print("  umap-learn not installed, skipping UMAP", flush=True)
    report['umap'] = {'status': 'skipped_no_umap_package'}

# ═══════════════════════════════════════════════════════════════════
# 3. Per-dataset rank distribution (5 random genes)
# ═══════════════════════════════════════════════════════════════════
print("\n[3/7] Per-dataset rank distribution check...", flush=True)

np.random.seed(42)
sample_genes = np.random.choice(gene_cols, size=5, replace=False)
print(f"  Selected genes: {list(sample_genes)}", flush=True)

fig, axes = plt.subplots(1, 5, figsize=(25, 5))
distribution_stats = {}

for i, gene in enumerate(sample_genes):
    gene_data = []
    ds_labels_violin = []
    for ds in unique_ds:
        vals = df.loc[df['dataset'] == ds, gene].dropna().values
        if len(vals) == 0:
            # Violin plot can't handle empty arrays — use a single 0.5 placeholder
            vals = np.array([0.5])
        gene_data.append(vals)
        ds_labels_violin.append(ds)

    parts = axes[i].violinplot(gene_data, showmeans=True, showmedians=True)
    axes[i].set_xticks(range(1, len(unique_ds) + 1))
    axes[i].set_xticklabels(unique_ds, rotation=45, ha='right', fontsize=7)
    axes[i].set_title(gene, fontsize=10)
    axes[i].set_ylabel("Rank value [0,1]")
    axes[i].set_ylim(-0.05, 1.05)
    axes[i].axhline(0.5, color='red', ls='--', alpha=0.3, label='expected median')

    # Check: for rank-transformed data, mean should be ~0.5 and distribution should be ~uniform
    for j, ds in enumerate(unique_ds):
        vals = df.loc[df['dataset'] == ds, gene].dropna().values
        if len(vals) > 0:
            distribution_stats[f"{gene}_{ds}"] = {
                'mean': round(float(np.mean(vals)), 4),
                'std': round(float(np.std(vals)), 4),
                'min': round(float(np.min(vals)), 4),
                'max': round(float(np.max(vals)), 4),
            }

fig.suptitle("Rank Distribution per Dataset (5 random genes)\nExpected: ~Uniform[0,1] for all", fontsize=12)
plt.tight_layout()
plt.savefig(QC_DIR / "rank_distributions_per_dataset.png", dpi=150)
plt.close()

# Summary: check if all means are near 0.5
means_by_ds = {}
for ds in unique_ds:
    ds_vals = df.loc[df['dataset'] == ds, gene_cols].values
    means_by_ds[ds] = {
        'global_mean': round(float(np.nanmean(ds_vals)), 4),
        'global_std': round(float(np.nanstd(ds_vals)), 4),
    }
    print(f"  {ds}: mean={means_by_ds[ds]['global_mean']:.4f}, std={means_by_ds[ds]['global_std']:.4f}", flush=True)

# Expected for Uniform[0,1]: mean ≈ 0.5, std ≈ 0.2887
expected_std = 1.0 / np.sqrt(12)
print(f"  Expected for Uniform[0,1]: mean=0.5000, std={expected_std:.4f}", flush=True)

report['rank_distributions'] = {
    'sample_genes': list(sample_genes),
    'per_dataset_global_stats': means_by_ds,
    'expected_uniform_mean': 0.5,
    'expected_uniform_std': round(float(expected_std), 4),
}

# ═══════════════════════════════════════════════════════════════════
# 4. Outlier detection
# ═══════════════════════════════════════════════════════════════════
print("\n[4/7] Outlier detection (within-dataset correlation)...", flush=True)

outlier_results = {}
all_flagged = []

for ds in unique_ds:
    mask = df['dataset'] == ds
    ds_data = df.loc[mask, gene_cols].values
    n_samples = ds_data.shape[0]

    if n_samples < 5:
        print(f"  {ds}: too few samples ({n_samples}), skipping", flush=True)
        continue

    # For efficiency, use a random subset of genes for correlation
    n_genes_for_corr = min(2000, len(gene_cols))
    gene_idx = np.random.choice(len(gene_cols), size=n_genes_for_corr, replace=False)
    ds_sub = np.nan_to_num(ds_data[:, gene_idx], nan=0.5)

    # Compute pairwise Pearson correlations
    corr_matrix = np.corrcoef(ds_sub)

    # Mean correlation with other samples (excluding self)
    np.fill_diagonal(corr_matrix, np.nan)
    mean_corrs = np.nanmean(corr_matrix, axis=1)

    flagged_mask = mean_corrs < 0.5
    n_flagged = int(flagged_mask.sum())

    sample_ids = df.index[mask].tolist()
    flagged_ids = [sample_ids[j] for j in range(n_samples) if flagged_mask[j]]

    print(f"  {ds}: n={n_samples}, mean_corr={np.mean(mean_corrs):.3f} "
          f"[{np.min(mean_corrs):.3f}, {np.max(mean_corrs):.3f}], "
          f"flagged={n_flagged}", flush=True)

    outlier_results[ds] = {
        'n_samples': n_samples,
        'mean_within_corr': round(float(np.mean(mean_corrs)), 4),
        'min_within_corr': round(float(np.min(mean_corrs)), 4),
        'max_within_corr': round(float(np.max(mean_corrs)), 4),
        'n_flagged_below_0.5': n_flagged,
        'flagged_sample_ids': flagged_ids[:20],  # Cap at 20
    }
    all_flagged.extend(flagged_ids)

report['outlier_detection'] = {
    'method': 'mean_pairwise_pearson_correlation_within_dataset',
    'threshold': 0.5,
    'n_genes_used': n_genes_for_corr,
    'total_flagged': len(all_flagged),
    'per_dataset': outlier_results,
}

# Plot outlier detection summary
fig, ax = plt.subplots(figsize=(12, 5))
ds_names = []
ds_corrs = []
for ds in unique_ds:
    if ds in outlier_results:
        ds_names.append(ds)
        # Get individual correlation values
        mask = df['dataset'] == ds
        ds_data = df.loc[mask, gene_cols].values
        gene_idx = np.random.choice(len(gene_cols), size=n_genes_for_corr, replace=False)
        ds_sub = np.nan_to_num(ds_data[:, gene_idx], nan=0.5)
        corr_matrix = np.corrcoef(ds_sub)
        np.fill_diagonal(corr_matrix, np.nan)
        mean_corrs = np.nanmean(corr_matrix, axis=1)
        ds_corrs.append(mean_corrs)

parts = ax.violinplot(ds_corrs, showmeans=True, showmedians=True)
ax.set_xticks(range(1, len(ds_names) + 1))
ax.set_xticklabels(ds_names, rotation=45, ha='right')
ax.axhline(0.5, color='red', ls='--', alpha=0.5, label='Outlier threshold')
ax.set_ylabel("Mean pairwise correlation with dataset peers")
ax.set_title("Within-Dataset Sample Correlation (outlier detection)")
ax.legend()
plt.tight_layout()
plt.savefig(QC_DIR / "outlier_detection_correlations.png", dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════
# 5. GSE32062 platform check
# ═══════════════════════════════════════════════════════════════════
print("\n[5/7] GSE32062 platform check...", flush=True)

# Load GSE32062 metadata to identify Agilent vs Affymetrix samples
gse32062_meta_path = RESP_DIR / "GSE32062_metadata.parquet"
gse32062_resp_path = RESP_DIR / "GSE32062_response.parquet"

platform_check = {}

if gse32062_meta_path.exists():
    gse32062_meta = pd.read_parquet(gse32062_meta_path)
    print(f"  GSE32062 metadata shape: {gse32062_meta.shape}", flush=True)
    print(f"  Columns: {list(gse32062_meta.columns)}", flush=True)

    # Check for platform info
    for col in gse32062_meta.columns:
        unique_vals = gse32062_meta[col].nunique()
        if unique_vals < 20:
            print(f"    {col}: {gse32062_meta[col].value_counts().to_dict()}", flush=True)

    platform_check['metadata_found'] = True
    platform_check['metadata_columns'] = list(gse32062_meta.columns)
else:
    print("  GSE32062 metadata file not found", flush=True)
    platform_check['metadata_found'] = False

# Regardless of metadata, check if samples separate in PCA
gse32062_mask = df['dataset'] == 'GSE32062'
gse32062_data = df.loc[gse32062_mask, gene_cols].values
gse32062_clean = np.nan_to_num(gse32062_data, nan=0.5)
gse32062_ids = df.index[gse32062_mask].tolist()

if gse32062_clean.shape[0] > 10:
    pca_gse = PCA(n_components=5, random_state=42)
    X_pca_gse = pca_gse.fit_transform(gse32062_clean)

    # Check for a natural cluster of ~10 samples (putative Affymetrix)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_km = km.fit_predict(X_pca_gse[:, :3])

    cluster_sizes = np.bincount(labels_km)
    smaller_cluster = int(np.min(cluster_sizes))
    print(f"  KMeans(k=2) on GSE32062 PCA: cluster sizes = {list(cluster_sizes)}", flush=True)

    # If one cluster has ~10 samples, it's suspicious
    if smaller_cluster <= 15:
        smaller_idx = np.argmin(cluster_sizes)
        smaller_samples = [gse32062_ids[j] for j in range(len(labels_km)) if labels_km[j] == smaller_idx]
        print(f"  Smaller cluster ({smaller_cluster} samples): {smaller_samples[:10]}...", flush=True)

        # Check mean correlation between clusters
        c0 = gse32062_clean[labels_km == 0]
        c1 = gse32062_clean[labels_km == 1]
        # Compute mean correlation between cluster members and within
        within0 = np.corrcoef(c0)
        np.fill_diagonal(within0, np.nan)
        within1 = np.corrcoef(c1)
        np.fill_diagonal(within1, np.nan)

        mean_within0 = float(np.nanmean(within0))
        mean_within1 = float(np.nanmean(within1))

        # Cross-cluster correlation
        cross_corr = np.corrcoef(np.vstack([c0, c1]))[:len(c0), len(c0):]
        mean_cross = float(np.mean(cross_corr))

        print(f"  Within-cluster correlation: cluster0={mean_within0:.3f}, cluster1={mean_within1:.3f}", flush=True)
        print(f"  Cross-cluster correlation: {mean_cross:.3f}", flush=True)

        platform_check['cluster_analysis'] = {
            'cluster_sizes': [int(x) for x in cluster_sizes],
            'smaller_cluster_samples': smaller_samples,
            'mean_within_corr_cluster0': round(mean_within0, 4),
            'mean_within_corr_cluster1': round(mean_within1, 4),
            'mean_cross_cluster_corr': round(mean_cross, 4),
            'interpretation': 'potential_platform_effect' if (mean_cross < 0.8 * min(mean_within0, mean_within1)) else 'no_strong_platform_effect'
        }
    else:
        platform_check['cluster_analysis'] = {
            'cluster_sizes': [int(x) for x in cluster_sizes],
            'interpretation': 'no_obvious_small_cluster',
        }

    # Plot GSE32062 PCA
    fig, ax = plt.subplots(figsize=(8, 6))
    for cl in range(2):
        mask = labels_km == cl
        ax.scatter(X_pca_gse[mask, 0], X_pca_gse[mask, 1], alpha=0.6, s=30,
                  label=f"Cluster {cl} (n={cluster_sizes[cl]})")
    ax.set_xlabel(f"PC1 ({pca_gse.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_gse.explained_variance_ratio_[1]:.1%})")
    ax.set_title("GSE32062: PCA with K-means clustering (platform check)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(QC_DIR / "gse32062_platform_check.png", dpi=150)
    plt.close()

report['gse32062_platform_check'] = platform_check

# ═══════════════════════════════════════════════════════════════════
# 6. Class balance analysis with AUC confidence intervals
# ═══════════════════════════════════════════════════════════════════
print("\n[6/7] Class balance analysis...", flush=True)

def auc_null_ci(n_pos, n_neg, alpha=0.05):
    """
    Compute 95% CI for AUC under the null hypothesis (AUC=0.5).
    Under H0, AUC ~ N(0.5, sigma^2) where sigma^2 is given by the
    Bamber (1975) / Hanley-McNeil formula for the variance of AUC under null.

    Var(AUC|H0) = (n_pos + n_neg + 1) / (12 * n_pos * n_neg)
    """
    from scipy.stats import norm

    if n_pos == 0 or n_neg == 0:
        return None, None, None

    var_auc = (n_pos + n_neg + 1) / (12 * n_pos * n_neg)
    se_auc = np.sqrt(var_auc)
    z = norm.ppf(1 - alpha / 2)
    ci_low = 0.5 - z * se_auc
    ci_high = 0.5 + z * se_auc
    return round(float(se_auc), 4), round(float(ci_low), 4), round(float(ci_high), 4)

class_balance = {}
for ds in unique_ds:
    mask = df['dataset'] == ds
    y_ds = df.loc[mask, 'response_binary'].values
    n = int(mask.sum())
    n_pos = int(y_ds.sum())
    n_neg = n - n_pos
    prevalence = round(float(n_pos / n), 4) if n > 0 else None

    se, ci_low, ci_high = auc_null_ci(n_pos, n_neg)

    class_balance[ds] = {
        'n': n,
        'n_positive': n_pos,
        'n_negative': n_neg,
        'prevalence': prevalence,
        'majority_class': 'positive' if n_pos > n_neg else 'negative',
        'majority_pct': round(float(max(n_pos, n_neg) / n * 100), 1),
        'auc_null_se': se,
        'auc_null_95ci': [ci_low, ci_high] if ci_low is not None else None,
    }

    imbalanced = max(n_pos, n_neg) / n > 0.8
    print(f"  {ds}: n={n}, pos={n_pos}, neg={n_neg}, prevalence={prevalence:.3f}, "
          f"AUC_null_CI=[{ci_low:.3f}, {ci_high:.3f}]"
          f"{' **IMBALANCED**' if imbalanced else ''}", flush=True)

report['class_balance'] = class_balance

# Plot class balance
fig, ax = plt.subplots(figsize=(12, 5))
ds_order = sorted(unique_ds, key=lambda d: class_balance[d]['n'], reverse=True)
x_pos = range(len(ds_order))
pos_counts = [class_balance[d]['n_positive'] for d in ds_order]
neg_counts = [class_balance[d]['n_negative'] for d in ds_order]

bars1 = ax.bar(x_pos, pos_counts, color='tab:blue', label='Positive (responder)')
bars2 = ax.bar(x_pos, neg_counts, bottom=pos_counts, color='tab:red', label='Negative (non-responder)')

for i, ds in enumerate(ds_order):
    total = class_balance[ds]['n']
    prev = class_balance[ds]['prevalence']
    ax.text(i, total + 5, f"{prev:.0%}", ha='center', fontsize=8)

ax.set_xticks(x_pos)
ax.set_xticklabels(ds_order, rotation=45, ha='right')
ax.set_ylabel("Sample count")
ax.set_title("Class Balance per Dataset (prevalence shown above bars)")
ax.legend()
plt.tight_layout()
plt.savefig(QC_DIR / "class_balance.png", dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════
# 7. Response label verification
# ═══════════════════════════════════════════════════════════════════
print("\n[7/7] Response label verification...", flush=True)

label_verification = {}

for ds in unique_ds:
    resp_path = RESP_DIR / f"{ds}_response.parquet"
    if not resp_path.exists():
        print(f"  {ds}: response file not found at {resp_path}", flush=True)
        label_verification[ds] = {'status': 'response_file_not_found'}
        continue

    resp_df = pd.read_parquet(resp_path)
    print(f"\n  {ds}:", flush=True)
    print(f"    Response columns: {list(resp_df.columns)}", flush=True)
    print(f"    Shape: {resp_df.shape}", flush=True)

    # Show unique values for key columns
    info = {'columns': list(resp_df.columns), 'n_samples': len(resp_df)}

    if 'response_binary' in resp_df.columns:
        info['response_binary_dist'] = resp_df['response_binary'].value_counts().to_dict()
        info['response_binary_dist'] = {str(k): int(v) for k, v in info['response_binary_dist'].items()}

    if 'response_continuous' in resp_df.columns:
        info['response_continuous_stats'] = {
            'mean': round(float(resp_df['response_continuous'].mean()), 4),
            'median': round(float(resp_df['response_continuous'].median()), 4),
            'min': round(float(resp_df['response_continuous'].min()), 4),
            'max': round(float(resp_df['response_continuous'].max()), 4),
        }
        print(f"    response_continuous: {info['response_continuous_stats']}", flush=True)

    if 'drug' in resp_df.columns:
        info['drugs'] = resp_df['drug'].value_counts().to_dict()
        info['drugs'] = {str(k): int(v) for k, v in info['drugs'].items()}
        print(f"    drugs: {info['drugs']}", flush=True)

    # Cross-check with what's in the pooled matrix
    pooled_ds = df[df['dataset'] == ds]
    info['n_in_pooled'] = len(pooled_ds)
    info['pooled_response_dist'] = {str(k): int(v) for k, v in pooled_ds['response_binary'].value_counts().to_dict().items()}

    print(f"    In pooled matrix: {info['n_in_pooled']} samples, "
          f"response_binary dist: {info['pooled_response_dist']}", flush=True)

    label_verification[ds] = info

# Specific checks for known datasets
print("\n  --- Specific label checks ---", flush=True)

# TCGA-OV: response_binary=1 should = platinum-sensitive (PFI > 6m)
tcga_resp = RESP_DIR / "TCGA-OV_response.parquet"
if tcga_resp.exists():
    tr = pd.read_parquet(tcga_resp)
    if 'response_continuous' in tr.columns:
        # response_continuous should be PFI in months
        sens = tr[tr['response_binary'] == 1]['response_continuous']
        resist = tr[tr['response_binary'] == 0]['response_continuous']
        print(f"\n  TCGA-OV verification:", flush=True)
        print(f"    Sensitive (binary=1): median PFI = {sens.median():.1f} months, range [{sens.min():.1f}, {sens.max():.1f}]", flush=True)
        print(f"    Resistant (binary=0): median PFI = {resist.median():.1f} months, range [{resist.min():.1f}, {resist.max():.1f}]", flush=True)
        # Check: all sensitive should have PFI > 6m, all resistant PFI <= 6m
        sens_correct = (sens > 6).all() if len(sens) > 0 else None
        resist_correct = (resist <= 6).all() if len(resist) > 0 else None
        print(f"    All sensitive PFI > 6m? {sens_correct}", flush=True)
        print(f"    All resistant PFI <= 6m? {resist_correct}", flush=True)
        label_verification['TCGA-OV']['pfi_check'] = {
            'sensitive_median_pfi': round(float(sens.median()), 1),
            'resistant_median_pfi': round(float(resist.median()), 1),
            'all_sens_above_6m': bool(sens_correct) if sens_correct is not None else None,
            'all_resist_below_6m': bool(resist_correct) if resist_correct is not None else None,
        }

# GSE32062: check response definition
gse32062_resp = RESP_DIR / "GSE32062_response.parquet"
if gse32062_resp.exists():
    gr = pd.read_parquet(gse32062_resp)
    print(f"\n  GSE32062 verification:", flush=True)
    print(f"    Columns: {list(gr.columns)}", flush=True)
    if 'response_continuous' in gr.columns:
        sens = gr[gr['response_binary'] == 1]['response_continuous']
        resist = gr[gr['response_binary'] == 0]['response_continuous']
        print(f"    Sensitive (binary=1): n={len(sens)}, median={sens.median():.1f}", flush=True)
        print(f"    Resistant (binary=0): n={len(resist)}, median={resist.median():.1f}", flush=True)

# GDSC: check IC50-based binarization
gdsc_resp = RESP_DIR / "GDSC_response.parquet"
if gdsc_resp.exists():
    gdr = pd.read_parquet(gdsc_resp)
    print(f"\n  GDSC verification:", flush=True)
    print(f"    Columns: {list(gdr.columns)}", flush=True)
    if 'drug' in gdr.columns:
        for drug in gdr['drug'].unique()[:5]:
            drug_sub = gdr[gdr['drug'] == drug]
            if 'response_continuous' in drug_sub.columns:
                sens = drug_sub[drug_sub['response_binary'] == 1]['response_continuous']
                resist = drug_sub[drug_sub['response_binary'] == 0]['response_continuous']
                print(f"    {drug}: sens median IC50={sens.median():.2f}, resist median IC50={resist.median():.2f}", flush=True)

report['response_label_verification'] = label_verification

# ═══════════════════════════════════════════════════════════════════
# Save report
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("Saving QC report...", flush=True)

# Convert any numpy types for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

report = convert_numpy(report)

with open(QC_DIR / "data_qc_report.json", 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nAll QC outputs saved to {QC_DIR}", flush=True)
print(f"  - pca_by_dataset.png", flush=True)
print(f"  - umap_by_dataset.png", flush=True)
print(f"  - rank_distributions_per_dataset.png", flush=True)
print(f"  - outlier_detection_correlations.png", flush=True)
print(f"  - gse32062_platform_check.png", flush=True)
print(f"  - class_balance.png", flush=True)
print(f"  - data_qc_report.json", flush=True)
print("\nDone!", flush=True)
