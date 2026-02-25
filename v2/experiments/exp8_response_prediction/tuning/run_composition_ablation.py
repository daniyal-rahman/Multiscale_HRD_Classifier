#!/usr/bin/env python3
"""
Task #6: Training data composition ablation.
Tests how training set composition affects LODO-CV performance:
1. All clean datasets (baseline, excl GSE28739/GSE32062)
2. Ovarian-only (no I-SPY2 breast)
3. Large datasets only (n >= 50)
4. Clinical-only (no I-SPY2 trial data)
5. I-SPY2 PARPi + TCGA-OV only (core PARPi datasets)

Uses L2 LogReg with C=0.01, class_weight='balanced'.
"""
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
OUT = BASE / "tuning"
OUT.mkdir(parents=True, exist_ok=True)
C_OPT = 0.01

# ── Load data ──────────────────────────────────────────────────────
print("Loading data...", flush=True)
df = pd.read_parquet(BASE / "pooled_rank_matrix.parquet")
meta_cols = ['response_binary', 'response_continuous', 'drug', 'dataset', 'category', 'cancer_type']
gene_cols = [c for c in df.columns if c not in meta_cols]

# Full Model A (clinical + I-SPY2 PARPi)
ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (df['category'] != 'cell_line') &
    ~((df['category'] == 'clinical_ispy2') & (~df['drug'].isin(ispy2_parpi_drugs)))
)
model_a_full = df[model_a_mask].copy()

EXCLUDE = ['GSE28739', 'GSE32062']
model_a_clean = model_a_full[~model_a_full['dataset'].isin(EXCLUDE)]

print(f"Full Model A: {len(model_a_full)}", flush=True)
print(f"Clean Model A (excl {EXCLUDE}): {len(model_a_clean)}", flush=True)

# Define composition configurations
# Each config: name, description, which datasets to include in training, which to test on
# Testing is always LODO on the included datasets
configs = {}

# 1. All clean (baseline)
configs['all_clean'] = {
    'desc': 'All clean datasets (excl GSE28739, GSE32062)',
    'data': model_a_clean,
}

# 2. Ovarian only (no breast I-SPY2)
ov_mask = model_a_clean['cancer_type'] != 'breast'
configs['ovarian_only'] = {
    'desc': 'Ovarian datasets only (no I-SPY2 breast)',
    'data': model_a_clean[ov_mask],
}

# 3. Large datasets only (n >= 50)
ds_sizes = model_a_clean.groupby('dataset').size()
large_ds = ds_sizes[ds_sizes >= 50].index.tolist()
configs['large_only'] = {
    'desc': f'Large datasets only (n>=50): {large_ds}',
    'data': model_a_clean[model_a_clean['dataset'].isin(large_ds)],
}

# 4. Non-trial clinical only (no I-SPY2)
non_ispy2_mask = model_a_clean['category'] != 'clinical_ispy2'
configs['no_ispy2_train'] = {
    'desc': 'Non-trial clinical only (no I-SPY2 in training)',
    'data': model_a_clean[non_ispy2_mask],
}

# 5. Core PARPi: I-SPY2 PARPi + TCGA-OV
core_mask = model_a_clean['dataset'].isin(['I-SPY2', 'TCGA-OV'])
if core_mask.sum() > 0:
    configs['core_parpi'] = {
        'desc': 'I-SPY2 PARPi + TCGA-OV only',
        'data': model_a_clean[core_mask],
    }


def run_lodo(name, data, gene_cols):
    print(f"\n{'='*60}", flush=True)
    print(f"Config: {name} ({len(data)} samples)", flush=True)
    ds_list = sorted(data['dataset'].unique())
    print(f"Datasets: {ds_list}", flush=True)
    for ds in ds_list:
        m = data['dataset'] == ds
        print(f"  {ds}: n={m.sum()}, pos={data.loc[m, 'response_binary'].sum():.0f}", flush=True)
    print(f"{'='*60}", flush=True)

    X = np.nan_to_num(data[gene_cols].values, nan=0.5).astype(np.float32)
    y = data['response_binary'].values.astype(int)
    datasets = data['dataset'].values

    fold_results = {}
    for held_out in ds_list:
        train_mask = datasets != held_out
        test_mask = datasets == held_out

        if train_mask.sum() < 30:
            fold_results[held_out] = {'auc': None, 'reason': 'train_too_small'}
            continue
        if len(np.unique(y[test_mask])) < 2:
            fold_results[held_out] = {'auc': None, 'reason': 'single_class_test'}
            continue

        clf = LogisticRegression(penalty='l2', C=C_OPT, solver='lbfgs',
                                  max_iter=3000, class_weight='balanced', random_state=42)
        clf.fit(X[train_mask], y[train_mask])
        y_prob = clf.predict_proba(X[test_mask])[:, 1]
        auc = roc_auc_score(y[test_mask], y_prob)
        fold_results[held_out] = {
            'auc': round(float(auc), 4),
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
        }
        print(f"  {held_out}: AUC={auc:.3f} (train={train_mask.sum()}, test={test_mask.sum()})", flush=True)

    valid_aucs = [v['auc'] for v in fold_results.values() if v.get('auc') is not None]
    return {
        'n_samples': len(data),
        'n_datasets': len(ds_list),
        'datasets': ds_list,
        'mean_auc': round(float(np.mean(valid_aucs)), 4) if valid_aucs else None,
        'median_auc': round(float(np.median(valid_aucs)), 4) if valid_aucs else None,
        'per_dataset': fold_results,
    }


results = {}
for name, cfg in configs.items():
    results[name] = run_lodo(name, cfg['data'], gene_cols)
    results[name]['description'] = cfg['desc']

# Also test: train on everything except target, predict target
# (cross-dataset rather than LODO within composition)
print("\n" + "="*60, flush=True)
print("Cross-composition: train on all_clean, test each dataset", flush=True)
print("="*60, flush=True)
X_clean = np.nan_to_num(model_a_clean[gene_cols].values, nan=0.5).astype(np.float32)
y_clean = model_a_clean['response_binary'].values.astype(int)
ds_clean = model_a_clean['dataset'].values

cross_results = {}
for held_out in sorted(model_a_clean['dataset'].unique()):
    train_mask = ds_clean != held_out
    test_mask = ds_clean == held_out
    if len(np.unique(y_clean[test_mask])) < 2:
        continue
    clf = LogisticRegression(penalty='l2', C=C_OPT, solver='lbfgs',
                              max_iter=3000, class_weight='balanced', random_state=42)
    clf.fit(X_clean[train_mask], y_clean[train_mask])
    auc = roc_auc_score(y_clean[test_mask], clf.predict_proba(X_clean[test_mask])[:, 1])
    cross_results[held_out] = round(float(auc), 4)
    print(f"  {held_out}: AUC={auc:.3f}", flush=True)

valid_cross = list(cross_results.values())
results['cross_dataset_baseline'] = {
    'description': 'LODO on all clean with C=0.01 (reference)',
    'per_dataset': cross_results,
    'mean_auc': round(float(np.mean(valid_cross)), 4),
}

with open(OUT / "composition_ablation_results.json", 'w') as f:
    json.dump(results, f, indent=2)

# Summary
print("\n" + "="*60, flush=True)
print("COMPOSITION ABLATION SUMMARY", flush=True)
print("="*60, flush=True)
for name, r in results.items():
    print(f"  {name}: mean_auc={r.get('mean_auc')}, n={r.get('n_samples', '?')}, "
          f"datasets={r.get('n_datasets', '?')}", flush=True)

print(f"\nSaved to {OUT / 'composition_ablation_results.json'}", flush=True)
print("Done!", flush=True)
