#!/usr/bin/env python3
"""
Immune Baseline Benchmarks for Drug Response Prediction

Tests whether simple immune scores predict platinum/PARPi response
as well as the full 11,140-gene L2 model.

Baselines:
  1. Single gene: CXCL9 (raw rank as score)
  2. Single gene: CD8A (raw rank as score)
  3. Single gene: IFNG (raw rank as score; check availability)
  4. 5-gene immune score: mean rank of [CXCL9, GBP5, HLA-DQA1, SLC15A3, GBP3]
  5. 5-gene L2 model: same 5 genes, fit L2 C=0.01
  6. 18-gene TIS score: mean rank of NanoString TIS genes (available subset)
  7. TIS L2 model: same TIS genes, fit L2 C=0.01
  8. Full model (reference): L2 C=0.01 on all 11,140 genes

All use same LODO-CV framework and Model A mask.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

EXP = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT = EXP / "immune_baselines"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load data
# ============================================================
print("=" * 70)
print("Loading pooled rank matrix...")
sys.stdout.flush()

pooled = pd.read_parquet(EXP / 'pooled_rank_matrix.parquet')
meta_cols = ['response_binary', 'response_continuous', 'drug', 'dataset', 'category', 'cancer_type']
gene_cols = [c for c in pooled.columns if c not in meta_cols]

print(f"  Shape: {pooled.shape[0]} samples x {len(gene_cols)} genes")
sys.stdout.flush()

# ============================================================
# Apply Model A mask
# ============================================================
ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (pooled['category'] != 'cell_line') &
    ~((pooled['category'] == 'clinical_ispy2') & (~pooled['drug'].isin(ispy2_parpi_drugs)))
)
model_a = pooled[model_a_mask].copy()

print(f"  Model A: {len(model_a)} samples (excluded cell_line + I-SPY2 non-PARPi)")
sys.stdout.flush()

y_all = model_a['response_binary'].values.astype(int)
datasets_all = model_a['dataset'].values
unique_datasets = sorted(model_a['dataset'].unique())

# Define clean-7 exclusions
exclude_datasets = {'GSE28739', 'GSE32062'}

print(f"  Datasets: {unique_datasets}")
print(f"  Clean-7 excludes: {sorted(exclude_datasets)}")
sys.stdout.flush()

# ============================================================
# Check gene availability
# ============================================================
immune5_genes = ['CXCL9', 'GBP5', 'HLA-DQA1', 'SLC15A3', 'GBP3']
tis_genes_full = ['CCL5', 'CD27', 'CD274', 'CD276', 'CD8A', 'CMKLR1', 'CXCL9',
                  'CXCR6', 'HLA-DQA1', 'HLA-DRB1', 'HLA-E', 'IDO1', 'LAG3',
                  'KLRK1', 'PDCD1LG2', 'PSMB10', 'STAT1', 'TIGIT']

# Check availability
print("\n--- Gene availability check ---")
sys.stdout.flush()

for g in ['CXCL9', 'CD8A', 'IFNG']:
    status = "YES" if g in gene_cols else "NO"
    print(f"  {g}: {status}")

tis_available = [g for g in tis_genes_full if g in gene_cols]
tis_missing = [g for g in tis_genes_full if g not in gene_cols]
print(f"\n  TIS genes available: {len(tis_available)}/{len(tis_genes_full)}")
print(f"  TIS missing: {tis_missing}")

for g in immune5_genes:
    assert g in gene_cols, f"Immune-5 gene {g} not found!"
print(f"  Immune-5 genes: all 5 present")
sys.stdout.flush()

# IFNG is not in the gene list. Check alternatives.
ifng_gene = None
if 'IFNG' in gene_cols:
    ifng_gene = 'IFNG'
else:
    # Try IFNG-AS1, or use IRF1 (downstream target) as fallback
    for alt in ['IFNG-AS1', 'IRF1', 'STAT1']:
        if alt in gene_cols:
            ifng_gene = alt
            print(f"  IFNG not available; using {alt} as substitute")
            break
if ifng_gene is None:
    print("  WARNING: No IFNG or suitable substitute found")

sys.stdout.flush()

# ============================================================
# LODO-CV helper
# ============================================================
def run_lodo_score(scores, y, datasets, label):
    """LODO-CV using pre-computed scores (no model fitting).
    For single-gene / average-score baselines: just compute AUC of score vs response.
    """
    results = {}
    for held_out in unique_datasets:
        test_mask = datasets == held_out
        y_test = y[test_mask]
        s_test = scores[test_mask]

        if len(np.unique(y_test)) < 2:
            results[held_out] = {'auc': float('nan'), 'n_test': int(test_mask.sum()), 'reason': 'single_class'}
            continue

        # Handle NaN in scores
        s_test = np.nan_to_num(s_test, nan=0.5)
        auc = roc_auc_score(y_test, s_test)
        results[held_out] = {
            'auc': round(float(auc), 4),
            'n_test': int(test_mask.sum()),
            'n_pos': int(y_test.sum()),
            'n_neg': int((1 - y_test).sum()),
        }

    return results


def run_lodo_model(X, y, datasets, C=0.01, label=''):
    """LODO-CV with L2 LogReg fitting."""
    results = {}
    for held_out in unique_datasets:
        train_mask = datasets != held_out
        test_mask = datasets == held_out

        X_train = np.nan_to_num(X[train_mask], nan=0.5)
        X_test = np.nan_to_num(X[test_mask], nan=0.5)
        y_train = y[train_mask]
        y_test = y[test_mask]

        if len(np.unique(y_test)) < 2:
            results[held_out] = {'auc': float('nan'), 'n_test': int(test_mask.sum()), 'reason': 'single_class'}
            continue

        clf = LogisticRegression(
            penalty='l2', C=C, solver='lbfgs',
            max_iter=3000, class_weight='balanced', random_state=42
        )
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        results[held_out] = {
            'auc': round(float(auc), 4),
            'n_test': int(test_mask.sum()),
            'n_pos': int(y_test.sum()),
            'n_neg': int((1 - y_test).sum()),
        }

    return results


def summarize(results, label):
    """Print and add summary stats to results dict."""
    valid_aucs = [v['auc'] for k, v in results.items()
                  if not k.startswith('_') and not np.isnan(v.get('auc', float('nan')))]
    clean7_aucs = [v['auc'] for k, v in results.items()
                   if not k.startswith('_') and k not in exclude_datasets
                   and not np.isnan(v.get('auc', float('nan')))]

    mean_all = np.mean(valid_aucs) if valid_aucs else float('nan')
    mean_clean7 = np.mean(clean7_aucs) if clean7_aucs else float('nan')

    results['_mean_auc_all9'] = round(float(mean_all), 4)
    results['_mean_auc_clean7'] = round(float(mean_clean7), 4)
    results['_n_datasets'] = len(valid_aucs)
    results['_n_clean7'] = len(clean7_aucs)

    print(f"\n  {label}")
    for ds in unique_datasets:
        if ds in results:
            auc_str = f"{results[ds]['auc']:.4f}" if not np.isnan(results[ds].get('auc', float('nan'))) else "N/A"
            marker = " *" if ds in exclude_datasets else ""
            print(f"    {ds:15s}: AUC = {auc_str}{marker}")
    print(f"    {'Mean (all 9)':15s}: {mean_all:.4f}")
    print(f"    {'Mean (clean 7)':15s}: {mean_clean7:.4f}")
    sys.stdout.flush()
    return results


# ============================================================
# Run all baselines
# ============================================================
all_results = {}

# ---- 1. Single gene: CXCL9 ----
print("\n" + "=" * 70)
print("Baseline 1: Single gene CXCL9 (raw rank)")
print("=" * 70)
sys.stdout.flush()

scores_cxcl9 = model_a['CXCL9'].values
res = run_lodo_score(scores_cxcl9, y_all, datasets_all, 'CXCL9')
all_results['1_CXCL9_raw'] = summarize(res, 'CXCL9 (raw rank)')

# ---- 2. Single gene: CD8A ----
print("\n" + "=" * 70)
print("Baseline 2: Single gene CD8A (raw rank)")
print("=" * 70)
sys.stdout.flush()

scores_cd8a = model_a['CD8A'].values
res = run_lodo_score(scores_cd8a, y_all, datasets_all, 'CD8A')
all_results['2_CD8A_raw'] = summarize(res, 'CD8A (raw rank)')

# ---- 3. Single gene: IFNG (or substitute) ----
print("\n" + "=" * 70)
if ifng_gene:
    print(f"Baseline 3: Single gene {ifng_gene} (raw rank)")
else:
    print("Baseline 3: IFNG — SKIPPED (gene not available)")
print("=" * 70)
sys.stdout.flush()

if ifng_gene:
    scores_ifng = model_a[ifng_gene].values
    res = run_lodo_score(scores_ifng, y_all, datasets_all, ifng_gene)
    all_results[f'3_{ifng_gene}_raw'] = summarize(res, f'{ifng_gene} (raw rank)')
else:
    all_results['3_IFNG_raw'] = {'_skipped': True, '_reason': 'IFNG not in 11140 gene set'}

# ---- 4. 5-gene immune score (mean rank) ----
print("\n" + "=" * 70)
print("Baseline 4: 5-gene immune score (mean rank)")
print(f"  Genes: {immune5_genes}")
print("=" * 70)
sys.stdout.flush()

scores_immune5 = model_a[immune5_genes].mean(axis=1).values
res = run_lodo_score(scores_immune5, y_all, datasets_all, 'Immune-5 avg')
all_results['4_immune5_avg'] = summarize(res, '5-gene immune avg')

# ---- 5. 5-gene L2 model ----
print("\n" + "=" * 70)
print("Baseline 5: 5-gene L2 model (C=0.01)")
print(f"  Genes: {immune5_genes}")
print("=" * 70)
sys.stdout.flush()

X_immune5 = model_a[immune5_genes].values
res = run_lodo_model(X_immune5, y_all, datasets_all, C=0.01, label='Immune-5 L2')
all_results['5_immune5_L2'] = summarize(res, '5-gene immune L2 (C=0.01)')

# ---- 6. TIS score (mean rank) ----
print("\n" + "=" * 70)
print(f"Baseline 6: TIS score (mean rank, {len(tis_available)}/{len(tis_genes_full)} genes)")
print(f"  Available: {tis_available}")
print(f"  Missing: {tis_missing}")
print("=" * 70)
sys.stdout.flush()

scores_tis = model_a[tis_available].mean(axis=1).values
res = run_lodo_score(scores_tis, y_all, datasets_all, 'TIS avg')
all_results['6_TIS_avg'] = summarize(res, f'TIS avg ({len(tis_available)} genes)')

# ---- 7. TIS L2 model ----
print("\n" + "=" * 70)
print(f"Baseline 7: TIS L2 model (C=0.01, {len(tis_available)} genes)")
print("=" * 70)
sys.stdout.flush()

X_tis = model_a[tis_available].values
res = run_lodo_model(X_tis, y_all, datasets_all, C=0.01, label='TIS L2')
all_results['7_TIS_L2'] = summarize(res, f'TIS L2 (C=0.01, {len(tis_available)} genes)')

# ---- 8. Full model (reference) ----
print("\n" + "=" * 70)
print("Baseline 8: Full L2 model (C=0.01, 11,140 genes) — reference")
print("=" * 70)
sys.stdout.flush()

X_full = model_a[gene_cols].values
res = run_lodo_model(X_full, y_all, datasets_all, C=0.01, label='Full L2')
all_results['8_full_L2'] = summarize(res, 'Full L2 (C=0.01, 11140 genes)')


# ============================================================
# Summary comparison table
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY COMPARISON")
print("=" * 70)

header = f"{'Baseline':<30s}"
for ds in unique_datasets:
    header += f" {ds:>10s}"
header += f" {'Mean9':>8s} {'Mean7':>8s}"
print(header)
print("-" * len(header))
sys.stdout.flush()

for bname, bres in all_results.items():
    if bres.get('_skipped'):
        continue
    row = f"{bname:<30s}"
    for ds in unique_datasets:
        if ds in bres and not np.isnan(bres[ds].get('auc', float('nan'))):
            row += f" {bres[ds]['auc']:>10.4f}"
        else:
            row += f" {'N/A':>10s}"
    row += f" {bres.get('_mean_auc_all9', float('nan')):>8.4f}"
    row += f" {bres.get('_mean_auc_clean7', float('nan')):>8.4f}"
    print(row)
sys.stdout.flush()


# ============================================================
# Save JSON results
# ============================================================
print("\n" + "=" * 70)
print("Saving results...")
print("=" * 70)
sys.stdout.flush()

# Build clean output
output = {
    'experiment': 'Immune Baseline Benchmarks for Drug Response Prediction',
    'description': 'Tests whether simple immune scores predict platinum/PARPi response as well as the full 11140-gene L2 model',
    'model_a_n_samples': int(len(model_a)),
    'n_datasets': len(unique_datasets),
    'datasets': unique_datasets,
    'exclude_from_clean7': sorted(exclude_datasets),
    'gene_availability': {
        'CXCL9': 'CXCL9' in gene_cols,
        'CD8A': 'CD8A' in gene_cols,
        'IFNG': 'IFNG' in gene_cols,
        'IFNG_substitute': ifng_gene if ifng_gene != 'IFNG' else None,
        'immune5_genes': immune5_genes,
        'tis_available': tis_available,
        'tis_missing': tis_missing,
    },
    'baselines': all_results,
    'summary': {}
}

# Compact summary
for bname, bres in all_results.items():
    if bres.get('_skipped'):
        output['summary'][bname] = 'SKIPPED'
        continue
    output['summary'][bname] = {
        'mean_auc_all9': bres.get('_mean_auc_all9'),
        'mean_auc_clean7': bres.get('_mean_auc_clean7'),
    }

with open(OUT / 'immune_baseline_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"  Saved: {OUT / 'immune_baseline_results.json'}")
sys.stdout.flush()


# ============================================================
# Comparison bar chart
# ============================================================
print("Generating comparison bar chart...")
sys.stdout.flush()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Collect data for plotting
labels = []
mean9_vals = []
mean7_vals = []
for bname, bres in all_results.items():
    if bres.get('_skipped'):
        continue
    # Clean up label
    label = bname.split('_', 1)[1]  # Remove number prefix
    labels.append(label)
    mean9_vals.append(bres.get('_mean_auc_all9', 0))
    mean7_vals.append(bres.get('_mean_auc_clean7', 0))

x = np.arange(len(labels))
width = 0.35

# Plot 1: Mean AUC comparison
ax = axes[0]
bars1 = ax.bar(x - width/2, mean9_vals, width, label='All 9 datasets', color='steelblue', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, mean7_vals, width, label='Clean 7 datasets', color='coral', edgecolor='black', linewidth=0.5)
ax.set_ylabel('Mean AUC', fontsize=12)
ax.set_title('Immune Baselines vs Full Model', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=9)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
ax.set_ylim(0.4, max(max(mean9_vals), max(mean7_vals)) + 0.05)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{height:.3f}', ha='center', va='bottom', fontsize=7)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{height:.3f}', ha='center', va='bottom', fontsize=7)

# Plot 2: Per-dataset heatmap-style comparison
ax2 = axes[1]
# Build matrix: baselines x datasets
baseline_names = []
auc_matrix = []
for bname, bres in all_results.items():
    if bres.get('_skipped'):
        continue
    label = bname.split('_', 1)[1]
    baseline_names.append(label)
    row = []
    for ds in unique_datasets:
        if ds in bres and not np.isnan(bres[ds].get('auc', float('nan'))):
            row.append(bres[ds]['auc'])
        else:
            row.append(np.nan)
    auc_matrix.append(row)

auc_arr = np.array(auc_matrix)
im = ax2.imshow(auc_arr, cmap='RdYlGn', aspect='auto', vmin=0.35, vmax=0.9)
ax2.set_xticks(range(len(unique_datasets)))
ax2.set_xticklabels(unique_datasets, rotation=45, ha='right', fontsize=8)
ax2.set_yticks(range(len(baseline_names)))
ax2.set_yticklabels(baseline_names, fontsize=9)
ax2.set_title('Per-Dataset AUC', fontsize=13, fontweight='bold')

# Add text annotations
for i in range(len(baseline_names)):
    for j in range(len(unique_datasets)):
        val = auc_arr[i, j]
        if not np.isnan(val):
            color = 'white' if val < 0.5 else 'black'
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

plt.colorbar(im, ax=ax2, shrink=0.8, label='AUC')
plt.tight_layout()
plt.savefig(OUT / 'baseline_comparison.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUT / 'baseline_comparison.png'}")
sys.stdout.flush()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
