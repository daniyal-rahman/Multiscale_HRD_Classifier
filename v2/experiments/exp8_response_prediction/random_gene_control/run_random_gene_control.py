#!/usr/bin/env python3
"""
Random Gene Control Experiment
==============================
Tests whether the specific genes selected by Model A matter, or whether
any random subset of K genes would achieve similar LODO-CV AUC.

For each K in {100, 500, 1000, 5000, 11140}:
  - "Curated": top-K genes by |weight| from L2 LogReg trained on training fold
  - "Random":  10 random draws (seeds 0-9) of K genes from all 11,140

For K=11140, random == curated == full model (sanity check).

Model: L2 LogReg, C=0.01, solver=lbfgs, class_weight=balanced
Evaluation: LODO-CV across 9 clinical datasets (Model A filter)
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT = BASE / "random_gene_control"

# ── Load data ──────────────────────────────────────────────────────
print("Loading data...", flush=True)
df = pd.read_parquet(BASE / "pooled_rank_matrix.parquet")
meta_cols = ['response_binary', 'response_continuous', 'drug', 'dataset', 'category', 'cancer_type']
gene_cols = [c for c in df.columns if c not in meta_cols]

ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (df['category'] != 'cell_line') &
    ~((df['category'] == 'clinical_ispy2') & (~df['drug'].isin(ispy2_parpi_drugs)))
)
model_a = df[model_a_mask].copy()

X_all = np.nan_to_num(model_a[gene_cols].values, nan=0.5).astype(np.float32)
y_all = model_a['response_binary'].values.astype(int)
datasets_all = model_a['dataset'].values
unique_datasets = sorted(model_a['dataset'].unique())
n_genes = len(gene_cols)

print(f"Model A: {len(model_a)} samples, {n_genes} genes, {len(unique_datasets)} datasets", flush=True)

K_VALUES = [100, 500, 1000, 5000, n_genes]
N_RANDOM_DRAWS = 10  # seeds 0-9


# ── LODO-CV helpers ────────────────────────────────────────────────
def lodo_cv(X, y, datasets, feature_idx=None):
    """Run LODO-CV with optional feature subsetting. Returns {dataset: auc}."""
    results = {}
    for held_out in unique_datasets:
        train_mask = datasets != held_out
        test_mask = datasets == held_out

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        if feature_idx is not None:
            X_train = X_train[:, feature_idx]
            X_test = X_test[:, feature_idx]

        if len(np.unique(y_test)) < 2:
            results[held_out] = None
            continue

        clf = LogisticRegression(
            penalty='l2', C=0.01, solver='lbfgs',
            max_iter=2000, class_weight='balanced', random_state=42
        )
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        results[held_out] = round(float(auc), 4)

    valid = [v for v in results.values() if v is not None]
    mean_auc = round(float(np.mean(valid)), 4) if valid else None
    return results, mean_auc


def select_top_k_by_weight(X_train, y_train, k):
    """Select top-k features by |weight| from L2 LogReg trained on training data."""
    clf = LogisticRegression(
        penalty='l2', C=0.01, solver='lbfgs',
        max_iter=2000, class_weight='balanced', random_state=42
    )
    clf.fit(X_train, y_train)
    abs_weights = np.abs(clf.coef_[0])
    return np.argsort(abs_weights)[::-1][:k]


def lodo_cv_curated(k):
    """LODO-CV with curated top-k feature selection (re-selected each fold)."""
    results = {}
    for held_out in unique_datasets:
        train_mask = datasets_all != held_out
        test_mask = datasets_all == held_out

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        if len(np.unique(y_test)) < 2:
            results[held_out] = None
            continue

        if k >= n_genes:
            idx = np.arange(n_genes)
        else:
            idx = select_top_k_by_weight(X_train, y_train, k)

        clf = LogisticRegression(
            penalty='l2', C=0.01, solver='lbfgs',
            max_iter=2000, class_weight='balanced', random_state=42
        )
        clf.fit(X_train[:, idx], y_train)
        y_prob = clf.predict_proba(X_test[:, idx])[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        results[held_out] = round(float(auc), 4)

    valid = [v for v in results.values() if v is not None]
    mean_auc = round(float(np.mean(valid)), 4) if valid else None
    return results, mean_auc


# ── Run experiment ─────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("Random Gene Control Experiment", flush=True)
print("=" * 70, flush=True)

all_results = {}
t0 = time.time()

for k in K_VALUES:
    print(f"\n--- K = {k} ---", flush=True)
    k_results = {}

    # 1) Curated: top-k by model weight (re-selected per fold)
    t1 = time.time()
    curated_per_ds, curated_mean = lodo_cv_curated(k)
    dt = time.time() - t1
    print(f"  Curated top-{k}: mean_AUC = {curated_mean:.4f}  ({dt:.1f}s)", flush=True)
    k_results['curated'] = {
        'per_dataset': curated_per_ds,
        'mean_auc': curated_mean,
    }

    # 2) Random draws (seeds 0-9)
    random_means = []
    random_per_ds_all = []
    for seed in range(N_RANDOM_DRAWS):
        t1 = time.time()
        rng = np.random.RandomState(seed)
        if k >= n_genes:
            # All genes = same as full model regardless of draw
            random_idx = np.arange(n_genes)
        else:
            random_idx = rng.choice(n_genes, size=k, replace=False)

        per_ds, mean_auc = lodo_cv(X_all, y_all, datasets_all, feature_idx=random_idx)
        dt = time.time() - t1
        random_means.append(mean_auc)
        random_per_ds_all.append(per_ds)
        print(f"  Random seed={seed}: mean_AUC = {mean_auc:.4f}  ({dt:.1f}s)", flush=True)

    avg_random = round(float(np.mean(random_means)), 4)
    std_random = round(float(np.std(random_means)), 4)
    print(f"  Random avg: {avg_random:.4f} +/- {std_random:.4f}", flush=True)
    print(f"  Curated advantage: {curated_mean - avg_random:+.4f}", flush=True)

    k_results['random'] = {
        'per_seed': {
            str(s): {'per_dataset': random_per_ds_all[s], 'mean_auc': random_means[s]}
            for s in range(N_RANDOM_DRAWS)
        },
        'mean_of_means': avg_random,
        'std_of_means': std_random,
        'all_means': [round(float(m), 4) for m in random_means],
    }

    k_results['curated_minus_random'] = round(float(curated_mean - avg_random), 4)
    all_results[str(k)] = k_results

elapsed = time.time() - t0

# ── Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"{'K':>6s}  {'Curated':>8s}  {'Random_avg':>10s}  {'Random_std':>10s}  {'Delta':>8s}", flush=True)
print("-" * 50, flush=True)
for k in K_VALUES:
    kr = all_results[str(k)]
    print(f"{k:>6d}  {kr['curated']['mean_auc']:>8.4f}  "
          f"{kr['random']['mean_of_means']:>10.4f}  "
          f"{kr['random']['std_of_means']:>10.4f}  "
          f"{kr['curated_minus_random']:>+8.4f}", flush=True)

print(f"\nTotal elapsed: {elapsed:.0f}s", flush=True)

# ── Save ───────────────────────────────────────────────────────────
output = {
    'experiment': 'Random Gene Control',
    'description': (
        'Tests whether curated top-K genes (by model weight) outperform '
        'random K-gene subsets in LODO-CV. For each K, curated selection '
        'is re-done inside each LODO fold (no data leakage). Random draws '
        'use seeds 0-9.'
    ),
    'model': 'L2 LogReg, C=0.01, lbfgs, class_weight=balanced',
    'n_samples': int(len(model_a)),
    'n_genes_total': int(n_genes),
    'n_datasets': int(len(unique_datasets)),
    'datasets': unique_datasets,
    'k_values': K_VALUES,
    'n_random_draws': N_RANDOM_DRAWS,
    'results': all_results,
    'summary': {
        str(k): {
            'curated_mean_auc': all_results[str(k)]['curated']['mean_auc'],
            'random_mean_auc': all_results[str(k)]['random']['mean_of_means'],
            'random_std_auc': all_results[str(k)]['random']['std_of_means'],
            'curated_advantage': all_results[str(k)]['curated_minus_random'],
        }
        for k in K_VALUES
    },
    'elapsed_seconds': round(elapsed, 1),
}

with open(OUT / "random_gene_control_results.json", 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {OUT / 'random_gene_control_results.json'}", flush=True)
print("Done!", flush=True)
