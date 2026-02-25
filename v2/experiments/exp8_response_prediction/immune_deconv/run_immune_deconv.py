#!/usr/bin/env python3
"""
Immune cell deconvolution analysis for Exp8 drug response model.

Goal: Confirm that the drug response model (L2 C=0.01) is learning immune
contexture by correlating predicted scores with immune cell type signature
scores derived from marker gene mean ranks.

Approach (gene signature scoring):
- Define immune cell type gene sets from literature markers
- For each cell type, compute mean rank of its marker genes from the
  pooled rank matrix
- Train L2 C=0.01 LODO-CV model to get per-sample predictions
- Spearman correlate: predicted score vs each immune signature
- Also correlate immune signatures with actual response_binary
- Repeat for all datasets

Output:
- immune_deconv/deconvolution_results.json
- immune_deconv/immune_correlation_heatmap.png
"""
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pointbiserialr, mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
EXP = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT = EXP / "immune_deconv"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70, flush=True)
print("Immune Cell Deconvolution — Gene Signature Scoring", flush=True)
print("=" * 70, flush=True)

# ── Load data ──────────────────────────────────────────────────────
print("\nLoading pooled rank matrix...", flush=True)
pooled = pd.read_parquet(EXP / 'pooled_rank_matrix.parquet')
common_genes = [g.strip() for g in open(BASE / "common_genes_all_10_datasets.txt")]
meta_cols = ['response_binary', 'response_continuous', 'drug', 'dataset', 'category', 'cancer_type']
gene_cols = [c for c in pooled.columns if c not in meta_cols]

print(f"Pooled matrix: {pooled.shape[0]} samples x {len(gene_cols)} genes", flush=True)

# ── Define immune cell type gene signatures ────────────────────────
# Using canonical markers from literature (Bindea 2013, Newman 2015, Ayers 2017)
# Only include genes present in the 11,140 common gene set
immune_signatures_raw = {
    'CD8+ T cells':        ['CD8A', 'CD8B', 'GZMA', 'GZMB', 'PRF1', 'IFNG', 'GZMK', 'CD3D', 'CD3E'],
    'M1 macrophages':      ['CD68', 'NOS2', 'IL1B', 'TNF', 'CXCL10', 'CXCL9', 'IRF1', 'STAT1'],
    'NK cells':            ['NCAM1', 'KLRD1', 'NKG7', 'GNLY', 'GZMB', 'PRF1'],
    'B cells':             ['CD19', 'MS4A1', 'CD79A', 'CD79B'],
    'Tregs':               ['FOXP3', 'IL2RA', 'CTLA4', 'TIGIT'],
    'Myeloid/MDSC':        ['CD33', 'ITGAM', 'ARG1', 'CD14'],
    'Stroma/fibroblasts':  ['FAP', 'ACTA2', 'COL1A1', 'COL3A1', 'COL5A1', 'FN1'],
    # Additional signatures relevant to drug response
    'IFN-gamma signaling': ['STAT1', 'IRF1', 'GBP1', 'GBP2', 'GBP3', 'GBP5', 'CXCL9', 'CXCL10', 'CXCL11', 'IDO1'],
    'MHC class I':         ['HLA-A', 'HLA-B', 'HLA-C', 'B2M'],
    'MHC class II':        ['HLA-DRA', 'HLA-DRB1', 'HLA-DQA1', 'HLA-DPA1', 'HLA-DPB1'],
    'Immune checkpoint':   ['PDCD1', 'CD274', 'LAG3', 'CTLA4', 'TIGIT', 'IDO1'],
    'Cytotoxic effector':  ['GZMA', 'GZMB', 'GZMK', 'PRF1', 'GNLY', 'NKG7', 'CCL5'],
    'T cell inflammation (Ayers)': ['CD8A', 'CXCL9', 'CXCL10', 'IDO1', 'STAT1', 'HLA-DRA',
                                     'CCL5', 'LAG3', 'PDCD1', 'CD274', 'GZMA', 'GZMB', 'PRF1',
                                     'HLA-DQA1', 'NKG7', 'CD3D', 'CD3E'],
    'Exp8 top immune genes': ['CXCL9', 'GBP3', 'GBP5', 'HLA-DQA1', 'SLC15A3'],
}

# Filter to available genes
immune_signatures = {}
print("\nImmune cell type gene signatures (filtered to available genes):", flush=True)
for ct, genes in immune_signatures_raw.items():
    available = [g for g in genes if g in gene_cols]
    missing = [g for g in genes if g not in gene_cols]
    immune_signatures[ct] = available
    print(f"  {ct}: {len(available)}/{len(genes)} genes. "
          f"Available: {available}"
          f"{f' | Missing: {missing}' if missing else ''}", flush=True)

# ── Compute signature scores ──────────────────────────────────────
print("\nComputing immune signature scores (mean rank per cell type)...", flush=True)
sig_scores = pd.DataFrame(index=pooled.index)
for ct, genes in immune_signatures.items():
    if len(genes) >= 1:
        sig_scores[ct] = pooled[genes].mean(axis=1)
    else:
        sig_scores[ct] = np.nan

print(f"Signature score matrix: {sig_scores.shape}", flush=True)
print(f"Score ranges:", flush=True)
for ct in sig_scores.columns:
    vals = sig_scores[ct].dropna()
    print(f"  {ct}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
          f"range=[{vals.min():.4f}, {vals.max():.4f}]", flush=True)

# ── LODO-CV predictions with tuned C=0.01 ─────────────────────────
print("\n" + "=" * 70, flush=True)
print("Generating LODO-CV predictions (L2 C=0.01, tuned model)", flush=True)
print("=" * 70, flush=True)

ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (pooled['category'] != 'cell_line') &
    ~((pooled['category'] == 'clinical_ispy2') & (~pooled['drug'].isin(ispy2_parpi_drugs)))
)
model_a = pooled[model_a_mask].copy()

X_all = model_a[gene_cols].values
y_all = model_a['response_binary'].values.astype(int)
datasets_all = model_a['dataset'].values
sample_ids_all = model_a.index.values

X_all = np.nan_to_num(X_all, nan=0.5)

unique_datasets = sorted(model_a['dataset'].unique())

all_predictions = {}
for held_out in unique_datasets:
    train_mask = datasets_all != held_out
    test_mask = datasets_all == held_out

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    clf = LogisticRegression(
        penalty='l2', C=0.01, solver='lbfgs',
        max_iter=5000, class_weight='balanced', random_state=42
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]

    test_ids = sample_ids_all[test_mask]
    for sid, prob in zip(test_ids, y_prob):
        all_predictions[sid] = prob

    # Report AUC
    if len(np.unique(y_test)) >= 2:
        auc = roc_auc_score(y_test, y_prob)
        print(f"  {held_out}: AUC={auc:.3f} (n={len(y_test)})", flush=True)
    else:
        print(f"  {held_out}: SKIP (single class, n={len(y_test)})", flush=True)

print(f"\nGenerated predictions for {len(all_predictions)} samples", flush=True)

# Add predictions to model_a
model_a['predicted_score'] = [all_predictions[sid] for sid in model_a.index]

# ── Correlate: per dataset ─────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("Correlation Analysis: Immune Signatures vs Model Score & Response", flush=True)
print("=" * 70, flush=True)

results = {
    'experiment': 'Immune Cell Deconvolution via Gene Signature Scoring',
    'date': '2026-02-24',
    'method': 'Mean rank of marker genes from pooled rank matrix',
    'model': 'L2 LogReg C=0.01, LODO-CV predictions',
    'immune_signatures': {ct: genes for ct, genes in immune_signatures.items()},
    'n_signatures': len(immune_signatures),
    'correlations_per_dataset': {},
    'correlations_pooled': {},
}

# All datasets to analyze
all_datasets = unique_datasets

cell_types = list(immune_signatures.keys())

# ── Per-dataset correlations ───────────────────────────────────────
for ds in all_datasets:
    ds_mask = model_a['dataset'] == ds
    ds_data = model_a[ds_mask]
    ds_sig = sig_scores.loc[ds_data.index]

    print(f"\n--- {ds} (n={len(ds_data)}) ---", flush=True)

    ds_results = {'n': len(ds_data)}
    for ct in cell_types:
        if ds_sig[ct].isna().all() or len(immune_signatures[ct]) == 0:
            continue

        scores_ct = ds_sig[ct].values
        pred_scores = ds_data['predicted_score'].values
        response = ds_data['response_binary'].values

        # Drop NaN entries
        valid = ~np.isnan(scores_ct)
        if valid.sum() < 10:
            continue
        scores_ct_v = scores_ct[valid]
        pred_v = pred_scores[valid]
        resp_v = response[valid]

        # Spearman: immune sig vs predicted score
        rho_pred, p_pred = spearmanr(scores_ct_v, pred_v)

        # Point-biserial: immune sig vs response_binary
        if len(np.unique(resp_v)) >= 2:
            rho_resp, p_resp = pointbiserialr(resp_v, scores_ct_v)
        else:
            rho_resp, p_resp = np.nan, np.nan

        ds_results[ct] = {
            'rho_vs_pred_score': round(float(rho_pred), 4),
            'p_vs_pred_score': round(float(p_pred), 6),
            'rho_vs_response': round(float(rho_resp), 4) if not np.isnan(rho_resp) else None,
            'p_vs_response': round(float(p_resp), 6) if not np.isnan(p_resp) else None,
        }

        sig_marker = "*" if p_pred < 0.05 else " "
        resp_marker = "*" if (not np.isnan(p_resp)) and p_resp < 0.05 else " "
        print(f"  {ct:30s}  vs_score: rho={rho_pred:+.3f} p={p_pred:.4f}{sig_marker}  "
              f"vs_response: rho={rho_resp:+.3f} p={p_resp:.4f}{resp_marker}", flush=True)

    results['correlations_per_dataset'][ds] = ds_results

# ── Pooled correlations (all clinical samples) ─────────────────────
print(f"\n--- POOLED (all clinical, n={len(model_a)}) ---", flush=True)
pooled_sig = sig_scores.loc[model_a.index]
pooled_results = {'n': len(model_a)}

for ct in cell_types:
    if pooled_sig[ct].isna().all():
        continue

    scores_ct = pooled_sig[ct].values
    pred_scores = model_a['predicted_score'].values
    response = model_a['response_binary'].values

    # Drop NaN entries for clean correlation
    valid_mask = ~np.isnan(scores_ct)
    scores_ct_clean = scores_ct[valid_mask]
    pred_clean = pred_scores[valid_mask]
    resp_clean = response[valid_mask]

    rho_pred, p_pred = spearmanr(scores_ct_clean, pred_clean)

    if len(np.unique(resp_clean)) >= 2:
        rho_resp, p_resp = pointbiserialr(resp_clean, scores_ct_clean)
    else:
        rho_resp, p_resp = np.nan, np.nan

    pooled_results[ct] = {
        'rho_vs_pred_score': round(float(rho_pred), 4),
        'p_vs_pred_score': round(float(p_pred), 6),
        'rho_vs_response': round(float(rho_resp), 4) if not np.isnan(rho_resp) else None,
        'p_vs_response': round(float(p_resp), 6) if not np.isnan(p_resp) else None,
        'n_valid': int(valid_mask.sum()),
    }

    sig_marker = "***" if p_pred < 0.001 else ("**" if p_pred < 0.01 else ("*" if p_pred < 0.05 else ""))
    print(f"  {ct:30s}  vs_score: rho={rho_pred:+.3f} p={p_pred:.2e}{sig_marker}  "
          f"vs_response: rho={rho_resp:+.3f} p={p_resp:.2e}  (n={valid_mask.sum()})", flush=True)

results['correlations_pooled'] = pooled_results

# ── Additional: TCGA-OV focused analysis ───────────────────────────
print(f"\n--- TCGA-OV Deep Dive ---", flush=True)
tcga_mask = model_a['dataset'] == 'TCGA-OV'
tcga_data = model_a[tcga_mask]
tcga_sig = sig_scores.loc[tcga_data.index]
tcga_pred = tcga_data['predicted_score'].values
tcga_resp = tcga_data['response_binary'].values

tcga_results = {'n': len(tcga_data)}

# Median split by predicted score
tcga_median = np.median(tcga_pred)
high_mask = tcga_pred >= tcga_median
low_mask = tcga_pred < tcga_median

print(f"  High-score group: n={high_mask.sum()}, response rate={tcga_resp[high_mask].mean():.3f}", flush=True)
print(f"  Low-score group:  n={low_mask.sum()}, response rate={tcga_resp[low_mask].mean():.3f}", flush=True)

for ct in cell_types:
    if tcga_sig[ct].isna().all():
        continue
    scores_ct = tcga_sig[ct].values
    high_scores = scores_ct[high_mask]
    low_scores = scores_ct[low_mask]

    # Mann-Whitney: high vs low predicted score groups
    stat, p_mw = mannwhitneyu(high_scores, low_scores, alternative='two-sided')
    diff = np.mean(high_scores) - np.mean(low_scores)

    tcga_results[ct] = {
        'mean_high_group': round(float(np.mean(high_scores)), 4),
        'mean_low_group': round(float(np.mean(low_scores)), 4),
        'diff': round(float(diff), 4),
        'mannwhitney_p': round(float(p_mw), 6),
    }

    sig_marker = "***" if p_mw < 0.001 else ("**" if p_mw < 0.01 else ("*" if p_mw < 0.05 else ""))
    print(f"  {ct:30s}  high={np.mean(high_scores):.4f} low={np.mean(low_scores):.4f} "
          f"diff={diff:+.4f} p={p_mw:.4f}{sig_marker}", flush=True)

results['tcga_ov_median_split'] = tcga_results

# ── Correlation heatmap ────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("Creating correlation heatmap", flush=True)
print("=" * 70, flush=True)

# Build matrices for heatmap
# Rows = cell types, Columns = datasets + pooled
# Values = Spearman rho (score vs predicted score)
datasets_for_hm = all_datasets + ['POOLED']
n_ct = len(cell_types)
n_ds = len(datasets_for_hm)

rho_matrix_pred = np.full((n_ct, n_ds), np.nan)
pval_matrix_pred = np.full((n_ct, n_ds), np.nan)
rho_matrix_resp = np.full((n_ct, n_ds), np.nan)
pval_matrix_resp = np.full((n_ct, n_ds), np.nan)

for j, ds in enumerate(datasets_for_hm):
    if ds == 'POOLED':
        src = results['correlations_pooled']
    else:
        src = results['correlations_per_dataset'].get(ds, {})

    for i, ct in enumerate(cell_types):
        if ct in src and isinstance(src[ct], dict):
            rho_matrix_pred[i, j] = src[ct].get('rho_vs_pred_score', np.nan)
            pval_matrix_pred[i, j] = src[ct].get('p_vs_pred_score', np.nan)
            rho_matrix_resp[i, j] = src[ct].get('rho_vs_response', np.nan)
            pval_matrix_resp[i, j] = src[ct].get('p_vs_response', np.nan)

# Create figure with two heatmaps side by side
fig = plt.figure(figsize=(22, 12))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

for panel_idx, (rho_mat, pval_mat, title_suffix) in enumerate([
    (rho_matrix_pred, pval_matrix_pred, 'vs Predicted Drug Response Score'),
    (rho_matrix_resp, pval_matrix_resp, 'vs Actual Response (binary)'),
]):
    ax = fig.add_subplot(gs[panel_idx])

    # Use diverging colormap
    vmax = max(abs(np.nanmin(rho_mat)), abs(np.nanmax(rho_mat)), 0.3)
    im = ax.imshow(rho_mat, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

    # Add text annotations
    for i in range(n_ct):
        for j in range(n_ds):
            rho_val = rho_mat[i, j]
            p_val = pval_mat[i, j]
            if np.isnan(rho_val):
                continue
            stars = ''
            if not np.isnan(p_val):
                if p_val < 0.001:
                    stars = '***'
                elif p_val < 0.01:
                    stars = '**'
                elif p_val < 0.05:
                    stars = '*'

            color = 'white' if abs(rho_val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{rho_val:.2f}{stars}', ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold' if stars else 'normal')

    ax.set_xticks(range(n_ds))
    ax.set_xticklabels(datasets_for_hm, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_ct))
    ax.set_yticklabels(cell_types, fontsize=9)
    ax.set_title(f'Immune Signature Correlation\n{title_suffix}', fontsize=12, fontweight='bold')

    cb = plt.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label('Spearman rho', fontsize=10)

fig.suptitle('Exp8 Drug Response Model: Immune Contexture Analysis\n'
             'Gene signature scores (mean rank) correlated with model predictions and actual response\n'
             'Model: L2 LogReg C=0.01, LODO-CV | * p<0.05, ** p<0.01, *** p<0.001',
             fontsize=13, fontweight='bold', y=0.99)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / 'immune_correlation_heatmap.png', dpi=150, bbox_inches='tight')
print(f"Saved heatmap to {OUT / 'immune_correlation_heatmap.png'}", flush=True)
plt.close()

# ── Summary statistics ─────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("SUMMARY", flush=True)
print("=" * 70, flush=True)

# Count significant correlations per cell type (pooled)
print("\nPooled correlations (all clinical samples):", flush=True)
print(f"{'Cell Type':35s} {'rho(score)':>10s} {'p(score)':>12s} {'rho(resp)':>10s} {'p(resp)':>12s}", flush=True)
print("-" * 82, flush=True)

sig_count_pred = 0
sig_count_resp = 0
for ct in cell_types:
    if ct in pooled_results and isinstance(pooled_results[ct], dict):
        rp = pooled_results[ct]['rho_vs_pred_score']
        pp = pooled_results[ct]['p_vs_pred_score']
        rr = pooled_results[ct].get('rho_vs_response')
        pr = pooled_results[ct].get('p_vs_response')
        if rp is None or pp is None:
            continue
        sp = "***" if pp < 0.001 else ("**" if pp < 0.01 else ("*" if pp < 0.05 else ""))
        if pp < 0.05:
            sig_count_pred += 1
        sr_str = ""
        if pr is not None:
            sr_str = "***" if pr < 0.001 else ("**" if pr < 0.01 else ("*" if pr < 0.05 else ""))
            if pr < 0.05:
                sig_count_resp += 1
            print(f"  {ct:33s} {rp:+.4f}{sp:3s} {pp:12.2e}  {rr:+.4f}{sr_str:3s} {pr:12.2e}", flush=True)
        else:
            print(f"  {ct:33s} {rp:+.4f}{sp:3s} {pp:12.2e}  {'N/A':>10s}", flush=True)

print(f"\nSignificant (p<0.05) with predicted score: {sig_count_pred}/{len(cell_types)}", flush=True)
print(f"Significant (p<0.05) with actual response:  {sig_count_resp}/{len(cell_types)}", flush=True)

# TCGA-OV consistency check
print(f"\nTCGA-OV correlations (n={len(tcga_data)}):", flush=True)
tcga_corr = results['correlations_per_dataset'].get('TCGA-OV', {})
for ct in cell_types:
    if ct in tcga_corr and isinstance(tcga_corr[ct], dict):
        rp = tcga_corr[ct]['rho_vs_pred_score']
        pp = tcga_corr[ct]['p_vs_pred_score']
        sp = "***" if pp < 0.001 else ("**" if pp < 0.01 else ("*" if pp < 0.05 else ""))
        print(f"  {ct:33s} rho={rp:+.4f} p={pp:.4f}{sp}", flush=True)

# Cross-dataset consistency
print(f"\nCross-dataset consistency for key signatures:", flush=True)
key_sigs = ['CD8+ T cells', 'IFN-gamma signaling', 'Cytotoxic effector',
            'T cell inflammation (Ayers)', 'Exp8 top immune genes', 'Stroma/fibroblasts']
for ct in key_sigs:
    rhos = []
    for ds in all_datasets:
        ds_res = results['correlations_per_dataset'].get(ds, {})
        if ct in ds_res and isinstance(ds_res[ct], dict):
            r = ds_res[ct].get('rho_vs_pred_score')
            if r is not None and not np.isnan(r):
                rhos.append(r)
    if rhos:
        n_pos = sum(1 for r in rhos if r > 0)
        print(f"  {ct:33s}  median rho={np.median(rhos):+.3f}  "
              f"range=[{min(rhos):+.3f}, {max(rhos):+.3f}]  "
              f"positive: {n_pos}/{len(rhos)} datasets", flush=True)

results['summary'] = {
    'n_sig_pred_score_pooled': sig_count_pred,
    'n_sig_response_pooled': sig_count_resp,
    'n_total_signatures': len(cell_types),
}

# ── Save results ───────────────────────────────────────────────────
with open(OUT / 'deconvolution_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUT / 'deconvolution_results.json'}", flush=True)

print("\nDone!", flush=True)
