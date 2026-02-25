#!/usr/bin/env python3
"""
Training Data Composition Ablation.

Best model: L2 LogReg, C=0.01 (from Task 2).

Ablations:
  1. Ovarian-only training (exclude breast data)
  2. Ovarian + breast PARPi (add I-SPY2 PARPi arms)
  3. Large datasets only (n >= 75)
  4. Remove GSE32062 from training
  5. GDSC pre-training (cell line features as additional input)
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import json
import warnings
import time
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT = BASE / "ablation"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────
print("Loading data...", flush=True)
df = pd.read_parquet(BASE / "pooled_rank_matrix.parquet")
meta_cols = ['response_binary', 'response_continuous', 'drug', 'dataset', 'category', 'cancer_type']
gene_cols = [c for c in df.columns if c not in meta_cols]

# Full model A: same as Task 2
ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (df['category'] != 'cell_line') &
    ~((df['category'] == 'clinical_ispy2') & (~df['drug'].isin(ispy2_parpi_drugs)))
)
model_a = df[model_a_mask].copy()
print(f"Full Model A: {len(model_a)} samples", flush=True)

# Also load GDSC for ablation 5
gdsc = df[df['dataset'] == 'GDSC'].copy()
print(f"GDSC: {len(gdsc)} samples", flush=True)

X_full = np.nan_to_num(model_a[gene_cols].values, nan=0.5).astype(np.float32)
y_full = model_a['response_binary'].values.astype(int)
ds_full = model_a['dataset'].values

# GDSC
X_gdsc = np.nan_to_num(gdsc[gene_cols].values, nan=0.5).astype(np.float32)
y_gdsc = gdsc['response_binary'].values.astype(int)

# Dataset category mapping
ovarian_datasets = ['TCGA-OV', 'GSE32062', 'GSE156699', 'GSE63885', 'GSE30161', 'GSE28739']
breast_datasets = ['GSE173839', 'GSE194040']  # I-SPY2 PARPi arms
cisplatin_datasets = ['GSE18864']
large_datasets = ['TCGA-OV', 'GSE32062', 'GSE156699', 'GSE63885']  # n >= 75

unique_datasets = sorted(model_a['dataset'].unique())


def run_lodo(X, y, ds_arr, train_ds_filter=None, test_datasets=None, label=""):
    """Run LODO-CV with optional training data filter.

    train_ds_filter: if provided, only use these datasets for training
    test_datasets: if provided, only test on these datasets
    """
    if test_datasets is None:
        test_datasets = unique_datasets

    results = {}
    for held_out in test_datasets:
        # Test set: held_out dataset
        test_mask = ds_arr == held_out
        if test_mask.sum() == 0:
            continue

        # Training set: everything except held_out, filtered if needed
        if train_ds_filter is not None:
            train_mask = np.array([d in train_ds_filter and d != held_out for d in ds_arr])
        else:
            train_mask = ds_arr != held_out

        if train_mask.sum() < 10:
            results[held_out] = {'auc': None, 'reason': 'too_few_train', 'n_train': int(train_mask.sum())}
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(np.unique(y_test)) < 2:
            results[held_out] = {'auc': None, 'reason': 'single_class'}
            continue

        clf = LogisticRegression(penalty='l2', C=0.01, solver='lbfgs',
                                 max_iter=2000, class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        results[held_out] = {
            'auc': round(float(auc), 4),
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
        }

    valid = [v['auc'] for v in results.values() if isinstance(v, dict) and v.get('auc') is not None]
    mean_auc = round(float(np.mean(valid)), 4) if valid else None
    return results, mean_auc


all_results = {}
t0 = time.time()

# ── Baseline: Full Model A ─────────────────────────────────────────
print("\n=== Baseline: Full Model A (all clinical data) ===", flush=True)
baseline_results, baseline_mean = run_lodo(X_full, y_full, ds_full, label="baseline")
print(f"  Mean AUC: {baseline_mean}", flush=True)
for ds in unique_datasets:
    if ds in baseline_results and baseline_results[ds].get('auc') is not None:
        print(f"  {ds}: AUC={baseline_results[ds]['auc']:.3f} (n_train={baseline_results[ds]['n_train']})", flush=True)
all_results['baseline'] = {'per_dataset': baseline_results, 'mean_auc': baseline_mean}

# ── Ablation 1: Ovarian-only ──────────────────────────────────────
print("\n=== Ablation 1: Ovarian-only training ===", flush=True)
abl1_results, abl1_mean = run_lodo(X_full, y_full, ds_full,
                                    train_ds_filter=ovarian_datasets,
                                    test_datasets=ovarian_datasets)
print(f"  Mean AUC: {abl1_mean}", flush=True)
for ds in ovarian_datasets:
    if ds in abl1_results and abl1_results[ds].get('auc') is not None:
        print(f"  {ds}: AUC={abl1_results[ds]['auc']:.3f} (n_train={abl1_results[ds]['n_train']})", flush=True)
all_results['ablation1_ovarian_only'] = {'per_dataset': abl1_results, 'mean_auc': abl1_mean}

# ── Ablation 2: Ovarian + I-SPY2 PARPi ───────────────────────────
print("\n=== Ablation 2: Ovarian + I-SPY2 PARPi ===", flush=True)
abl2_train_ds = ovarian_datasets + breast_datasets
abl2_results, abl2_mean = run_lodo(X_full, y_full, ds_full,
                                    train_ds_filter=abl2_train_ds,
                                    test_datasets=ovarian_datasets)
print(f"  Mean AUC: {abl2_mean}", flush=True)
for ds in ovarian_datasets:
    if ds in abl2_results and abl2_results[ds].get('auc') is not None:
        print(f"  {ds}: AUC={abl2_results[ds]['auc']:.3f} (n_train={abl2_results[ds]['n_train']})", flush=True)
all_results['ablation2_ovarian_plus_ispy2'] = {'per_dataset': abl2_results, 'mean_auc': abl2_mean}

# ── Ablation 3: Large datasets only ──────────────────────────────
print("\n=== Ablation 3: Large datasets only (n>=75) ===", flush=True)
abl3_results, abl3_mean = run_lodo(X_full, y_full, ds_full,
                                    train_ds_filter=large_datasets,
                                    test_datasets=large_datasets)
print(f"  Mean AUC: {abl3_mean}", flush=True)
for ds in large_datasets:
    if ds in abl3_results and abl3_results[ds].get('auc') is not None:
        print(f"  {ds}: AUC={abl3_results[ds]['auc']:.3f} (n_train={abl3_results[ds]['n_train']})", flush=True)
all_results['ablation3_large_only'] = {'per_dataset': abl3_results, 'mean_auc': abl3_mean}

# Also test on ALL datasets (train on large, test on all)
print("  --- Also testing on all datasets ---", flush=True)
abl3b_results, abl3b_mean = run_lodo(X_full, y_full, ds_full,
                                      train_ds_filter=large_datasets,
                                      test_datasets=unique_datasets)
print(f"  Mean AUC (all held-out): {abl3b_mean}", flush=True)
all_results['ablation3b_large_train_all_test'] = {'per_dataset': abl3b_results, 'mean_auc': abl3b_mean}

# ── Ablation 4: Remove GSE32062 ──────────────────────────────────
print("\n=== Ablation 4: Remove GSE32062 from training ===", flush=True)
no_gse32062 = [ds for ds in unique_datasets if ds != 'GSE32062']
abl4_results, abl4_mean = run_lodo(X_full, y_full, ds_full,
                                    train_ds_filter=no_gse32062,
                                    test_datasets=no_gse32062)
print(f"  Mean AUC: {abl4_mean}", flush=True)
for ds in no_gse32062:
    if ds in abl4_results and abl4_results[ds].get('auc') is not None:
        base_auc = baseline_results.get(ds, {}).get('auc', 'N/A')
        print(f"  {ds}: AUC={abl4_results[ds]['auc']:.3f} (baseline={base_auc})", flush=True)
all_results['ablation4_no_gse32062'] = {'per_dataset': abl4_results, 'mean_auc': abl4_mean}

# ── Ablation 5: GDSC pre-training ────────────────────────────────
print("\n=== Ablation 5: GDSC pre-training ===", flush=True)
# Stage 1: Train on GDSC, get prediction scores
clf_gdsc = LogisticRegression(penalty='l2', C=0.01, solver='lbfgs',
                               max_iter=2000, class_weight='balanced', random_state=42)
clf_gdsc.fit(X_gdsc, y_gdsc)
gdsc_scores = clf_gdsc.predict_proba(X_full)[:, 1]  # GDSC-based scores for clinical samples

# Stage 2: Add GDSC score as an extra feature
X_full_with_gdsc = np.column_stack([X_full, gdsc_scores])

abl5_results = {}
for held_out in unique_datasets:
    train_mask = ds_full != held_out
    test_mask = ds_full == held_out

    X_train = X_full_with_gdsc[train_mask]
    X_test = X_full_with_gdsc[test_mask]
    y_train = y_full[train_mask]
    y_test = y_full[test_mask]

    if len(np.unique(y_test)) < 2:
        abl5_results[held_out] = {'auc': None, 'reason': 'single_class'}
        continue

    clf = LogisticRegression(penalty='l2', C=0.01, solver='lbfgs',
                             max_iter=2000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    abl5_results[held_out] = {
        'auc': round(float(auc), 4),
        'gdsc_feature_weight': round(float(clf.coef_[0][-1]), 6),
    }

valid5 = [v['auc'] for v in abl5_results.values() if isinstance(v, dict) and v.get('auc') is not None]
abl5_mean = round(float(np.mean(valid5)), 4) if valid5 else None
print(f"  Mean AUC: {abl5_mean}", flush=True)
for ds in unique_datasets:
    if ds in abl5_results and abl5_results[ds].get('auc') is not None:
        gdsc_w = abl5_results[ds].get('gdsc_feature_weight', 'N/A')
        base_auc = baseline_results.get(ds, {}).get('auc', 'N/A')
        print(f"  {ds}: AUC={abl5_results[ds]['auc']:.3f} (baseline={base_auc}, gdsc_weight={gdsc_w})", flush=True)
all_results['ablation5_gdsc_pretrain'] = {'per_dataset': abl5_results, 'mean_auc': abl5_mean}

elapsed = time.time() - t0
print(f"\nTotal: {elapsed:.0f}s", flush=True)

# ── Summary comparison ─────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("COMPARISON SUMMARY (ovarian datasets only for fair comparison)", flush=True)
print("=" * 70, flush=True)

ablation_names = [
    ('baseline', 'All clinical (baseline)'),
    ('ablation1_ovarian_only', 'Ovarian-only'),
    ('ablation2_ovarian_plus_ispy2', 'Ovarian + I-SPY2 PARPi'),
    ('ablation4_no_gse32062', 'No GSE32062'),
    ('ablation5_gdsc_pretrain', 'GDSC pre-training'),
]

comp_rows = []
for key, label in ablation_names:
    row = {'ablation': label}
    for ds in ovarian_datasets:
        auc = all_results.get(key, {}).get('per_dataset', {}).get(ds, {})
        if isinstance(auc, dict):
            row[ds] = auc.get('auc')
        else:
            row[ds] = None
    comp_rows.append(row)

comp_df = pd.DataFrame(comp_rows).set_index('ablation')
print(comp_df.to_string(), flush=True)
comp_df.to_csv(OUT / "ablation_comparison.csv")

# ── Save ───────────────────────────────────────────────────────────
all_results['elapsed_seconds'] = round(elapsed, 1)
all_results['summary'] = {
    'baseline_mean': baseline_mean,
    'ovarian_only_mean': abl1_mean,
    'ovarian_plus_ispy2_mean': abl2_mean,
    'large_only_mean': abl3_mean,
    'no_gse32062_mean': abl4_mean,
    'gdsc_pretrain_mean': abl5_mean,
}

with open(OUT / "training_composition_results.json", 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\nSaved to {OUT}", flush=True)
print("Done!", flush=True)
