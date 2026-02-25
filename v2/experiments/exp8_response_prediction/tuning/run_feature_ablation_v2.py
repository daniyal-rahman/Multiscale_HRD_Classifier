#!/usr/bin/env python3
"""
Feature Selection Ablation: Test whether using fewer features improves LODO-CV.

Two strategies:
  A. Model-weight based: Rank genes by |weight| from L2 LogReg trained on training fold
  B. Univariate: Rank genes by Mann-Whitney U p-value between response groups in training fold

Best model from Task 2: L2 LogReg, C=0.01, lbfgs, class_weight='balanced'

k values: [25, 50, 100, 200, 500, 1000, 2000, 5000, 11140]
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import json
import warnings
import time
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

warnings.filterwarnings('ignore')

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT = BASE / "tuning"

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

print(f"Model A: {len(model_a)} samples, {len(gene_cols)} genes", flush=True)

k_values = [25, 50, 100, 200, 500, 1000, 2000, 5000, len(gene_cols)]


def select_by_model_weight(X_train, y_train, k):
    clf = LogisticRegression(penalty='l2', C=0.01, solver='lbfgs',
                             max_iter=2000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    abs_weights = np.abs(clf.coef_[0])
    return np.argsort(abs_weights)[::-1][:k]


def select_by_univariate(X_train, y_train, k):
    pos_mask = y_train == 1
    neg_mask = y_train == 0
    p_vals = np.ones(X_train.shape[1])
    for j in range(X_train.shape[1]):
        try:
            _, p = mannwhitneyu(X_train[pos_mask, j], X_train[neg_mask, j], alternative='two-sided')
            p_vals[j] = p
        except Exception:
            p_vals[j] = 1.0
    return np.argsort(p_vals)[:k]


def run_lodo_with_selection(select_fn, k):
    results = {}
    for held_out in unique_datasets:
        train_mask = datasets_all != held_out
        test_mask = datasets_all == held_out
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        if len(np.unique(y_test)) < 2:
            results[held_out] = None
            continue

        if k >= X_train.shape[1]:
            selected_idx = np.arange(X_train.shape[1])
        else:
            selected_idx = select_fn(X_train, y_train, k)

        clf = LogisticRegression(penalty='l2', C=0.01, solver='lbfgs',
                                 max_iter=2000, class_weight='balanced', random_state=42)
        clf.fit(X_train[:, selected_idx], y_train)
        y_prob = clf.predict_proba(X_test[:, selected_idx])[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        results[held_out] = round(float(auc), 4)

    valid = [v for v in results.values() if v is not None]
    mean_auc = round(float(np.mean(valid)), 4) if valid else None
    return results, mean_auc


# ── Run ────────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("Feature Selection Ablation", flush=True)
print("=" * 70, flush=True)

all_results = {'model_weight_selection': {}, 'univariate_selection': {}}
t0 = time.time()

print("\n--- A: Model-Weight Feature Selection ---", flush=True)
for k in k_values:
    t1 = time.time()
    per_ds, mean_auc = run_lodo_with_selection(select_by_model_weight, k)
    dt = time.time() - t1
    print(f"  k={k:>5}: mean_AUC={mean_auc:.3f} ({dt:.0f}s)", flush=True)
    all_results['model_weight_selection'][str(k)] = {'per_dataset': per_ds, 'mean_auc': mean_auc}

print("\n--- B: Univariate (Mann-Whitney) ---", flush=True)
for k in k_values:
    t1 = time.time()
    per_ds, mean_auc = run_lodo_with_selection(select_by_univariate, k)
    dt = time.time() - t1
    print(f"  k={k:>5}: mean_AUC={mean_auc:.3f} ({dt:.0f}s)", flush=True)
    all_results['univariate_selection'][str(k)] = {'per_dataset': per_ds, 'mean_auc': mean_auc}

elapsed = time.time() - t0
print(f"\nTotal: {elapsed:.0f}s", flush=True)

# ── Plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (strategy, label) in zip(axes, [
    ('model_weight_selection', 'Model-Weight Selection'),
    ('univariate_selection', 'Univariate (Mann-Whitney)')
]):
    mean_aucs = [all_results[strategy][str(k)]['mean_auc'] for k in k_values]
    for ds in unique_datasets:
        ds_aucs = [all_results[strategy][str(k)]['per_dataset'].get(ds) for k in k_values]
        ax.plot(k_values, ds_aucs, '--', alpha=0.4, linewidth=1, label=ds)
    ax.plot(k_values, mean_aucs, 'k-o', linewidth=2.5, markersize=6, label='Mean AUC')
    ax.set_xscale('log')
    ax.set_xlabel('Number of features (k)')
    ax.set_ylabel('LODO-CV AUC')
    ax.set_title(label)
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0.5, color='red', ls='--', alpha=0.3)
    ax.set_ylim(0.3, 1.0)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "feature_selection_ablation.png", dpi=150)
plt.close()

# Comparison plot
fig, ax = plt.subplots(figsize=(10, 6))
wm = [all_results['model_weight_selection'][str(k)]['mean_auc'] for k in k_values]
um = [all_results['univariate_selection'][str(k)]['mean_auc'] for k in k_values]
ax.plot(k_values, wm, 'b-o', linewidth=2, label='Model-weight')
ax.plot(k_values, um, 'r-s', linewidth=2, label='Univariate (MW)')
ax.axhline(0.5, color='gray', ls='--', alpha=0.3)
ax.set_xscale('log')
ax.set_xlabel('Number of features (k)', fontsize=12)
ax.set_ylabel('Mean LODO-CV AUC', fontsize=12)
ax.set_title('Feature Selection Ablation', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

best_kw = k_values[np.argmax(wm)]
best_ku = k_values[np.argmax(um)]
ax.annotate(f'Best k={best_kw}\nAUC={max(wm):.3f}',
            xy=(best_kw, max(wm)), xytext=(best_kw*2, max(wm)+0.02),
            arrowprops=dict(arrowstyle='->', color='blue'), color='blue', fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "feature_selection_comparison.png", dpi=150)
plt.close()

# ── Save ───────────────────────────────────────────────────────────
all_results['summary'] = {
    'model_weight_best_k': int(best_kw),
    'model_weight_best_auc': max(wm),
    'univariate_best_k': int(best_ku),
    'univariate_best_auc': max(um),
    'all_features_auc': wm[-1],
}
all_results['elapsed_seconds'] = round(elapsed, 1)

with open(OUT / "feature_selection_results.json", 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\nSaved. Best model-weight k={best_kw} (AUC={max(wm):.3f}), univariate k={best_ku} (AUC={max(um):.3f})", flush=True)
print("Done!", flush=True)
