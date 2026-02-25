#!/usr/bin/env python3
"""
Exp8 Nested CV: Leave-One-Dataset-Out with inner 5-fold stratified CV.

Models compared (with checkpointing after each model):
  1. L1 Logistic Regression (liblinear)
  2. L2 Logistic Regression (lbfgs)
  3. ElasticNet l1_ratio=0.5 (saga, low tol for speed)
  4. ElasticNet l1_ratio=0.9 (saga, low tol for speed)
  5. LightGBM
  6. SVM (RBF kernel)
  7. Baseline: L2 default C=1.0

Checkpoints after each model to nested_cv_results.json so partial results are preserved.
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import json
import warnings
import time
import os
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT = BASE / "tuning"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUT / "nested_cv_results.json"

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
sample_ids = model_a.index.tolist()
unique_datasets = sorted(model_a['dataset'].unique())

print(f"Model A: {len(model_a)} samples, {len(gene_cols)} genes, {len(unique_datasets)} datasets", flush=True)
for ds in unique_datasets:
    m = datasets_all == ds
    print(f"  {ds}: n={m.sum()}, pos={y_all[m].sum()}, neg={(1-y_all[m]).sum()}", flush=True)

# ── Resume from checkpoint if exists ───────────────────────────────
if RESULTS_FILE.exists():
    with open(RESULTS_FILE) as f:
        all_results = json.load(f)
    print(f"\nResuming from checkpoint: {list(all_results.keys())} already done", flush=True)
else:
    all_results = {}

all_predictions = []

# ── Model definitions ──────────────────────────────────────────────
C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

MODELS = [
    ('L1_LogReg', [{'C': c} for c in C_values], False),
    ('L2_LogReg', [{'C': c} for c in C_values], False),
    ('ElasticNet_0.5', [{'C': c} for c in C_values], False),
    ('ElasticNet_0.9', [{'C': c} for c in C_values], False),
    ('LightGBM', [
        {'n_estimators': ne, 'max_depth': md, 'learning_rate': 0.05}
        for ne in [50, 100, 200] for md in [3, 5, 7]
    ], True),  # 9 configs (fixed lr=0.05 to reduce grid)
    ('SVM_RBF', [
        {'C': c, 'gamma': g}
        for c in [0.01, 0.1, 1.0, 10.0] for g in ['scale', 'auto']
    ], False),
]


def build_model(model_name, params, y_train=None):
    if model_name == 'L1_LogReg':
        return LogisticRegression(penalty='l1', C=params['C'], solver='liblinear',
                                  max_iter=2000, class_weight='balanced', random_state=42)
    elif model_name == 'L2_LogReg':
        return LogisticRegression(penalty='l2', C=params['C'], solver='lbfgs',
                                  max_iter=2000, class_weight='balanced', random_state=42)
    elif model_name == 'ElasticNet_0.5':
        return LogisticRegression(penalty='elasticnet', C=params['C'], solver='saga',
                                  l1_ratio=0.5, max_iter=500, tol=1e-2,
                                  class_weight='balanced', random_state=42)
    elif model_name == 'ElasticNet_0.9':
        return LogisticRegression(penalty='elasticnet', C=params['C'], solver='saga',
                                  l1_ratio=0.9, max_iter=500, tol=1e-2,
                                  class_weight='balanced', random_state=42)
    elif model_name == 'LightGBM':
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        return lgb.LGBMClassifier(
            n_estimators=params['n_estimators'], max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=0.8, colsample_bytree=0.3,
            scale_pos_weight=n_neg / max(n_pos, 1),
            random_state=42, verbose=-1)
    elif model_name == 'SVM_RBF':
        return SVC(kernel='rbf', C=params['C'], gamma=params['gamma'],
                   class_weight='balanced', probability=True, random_state=42)


def get_proba(clf, X):
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def inner_cv_select(X_train, y_train, model_name, param_list, needs_y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_auc, best_params = -1, None
    all_inner = []

    for params in param_list:
        fold_aucs = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            if len(np.unique(y_val)) < 2:
                continue
            try:
                clf = build_model(model_name, params, y_tr if needs_y else None)
                clf.fit(X_tr, y_tr)
                auc = roc_auc_score(y_val, get_proba(clf, X_val))
                fold_aucs.append(auc)
            except Exception:
                pass

        mean_auc = np.mean(fold_aucs) if fold_aucs else 0.0
        all_inner.append({
            'params': {k: str(v) for k, v in params.items()},
            'mean_inner_auc': round(float(mean_auc), 4),
            'n_valid_folds': len(fold_aucs),
        })
        if mean_auc > best_auc:
            best_auc, best_params = mean_auc, params

    return best_params, best_auc, all_inner


def checkpoint(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)


# ── Main loop ──────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("Nested LODO-CV", flush=True)
print("=" * 70, flush=True)

for model_name, param_list, needs_y in MODELS:
    if model_name in all_results:
        print(f"\n{model_name}: already done (checkpoint), skipping", flush=True)
        continue

    print(f"\n{'─'*50}", flush=True)
    print(f"Model: {model_name} ({len(param_list)} configs)", flush=True)
    print(f"{'─'*50}", flush=True)
    model_results = {}
    t0 = time.time()

    for ds_idx, held_out in enumerate(unique_datasets):
        train_mask = datasets_all != held_out
        test_mask = datasets_all == held_out
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        test_ids = [sample_ids[i] for i in range(len(sample_ids)) if test_mask[i]]

        if len(np.unique(y_test)) < 2:
            print(f"  [{ds_idx+1}/9] {held_out}: SKIP (single class)", flush=True)
            model_results[held_out] = {'auc': None, 'reason': 'single_class_test',
                                       'n_test': int(test_mask.sum())}
            continue

        t1 = time.time()
        best_params, best_inner_auc, inner_results = inner_cv_select(
            X_train, y_train, model_name, param_list, needs_y)

        if best_params is None:
            print(f"  [{ds_idx+1}/9] {held_out}: NO VALID CONFIG", flush=True)
            model_results[held_out] = {'auc': None, 'reason': 'no_valid_config'}
            continue

        try:
            clf = build_model(model_name, best_params, y_train if needs_y else None)
            clf.fit(X_train, y_train)
            y_prob = get_proba(clf, X_test)
            auc = roc_auc_score(y_test, y_prob)
            dt = time.time() - t1

            model_results[held_out] = {
                'auc': round(float(auc), 4),
                'best_params': {k: str(v) for k, v in best_params.items()},
                'best_inner_auc': round(float(best_inner_auc), 4),
                'n_train': int(train_mask.sum()),
                'n_test': int(test_mask.sum()),
                'n_pos_test': int(y_test.sum()),
                'n_neg_test': int((1 - y_test).sum()),
                'inner_cv_results': inner_results,
            }

            for sid, prob, ty in zip(test_ids, y_prob, y_test):
                all_predictions.append({
                    'sample_id': sid, 'model': model_name,
                    'held_out_dataset': held_out,
                    'predicted_prob': float(prob), 'true_label': int(ty),
                })

            print(f"  [{ds_idx+1}/9] {held_out}: AUC={auc:.3f} "
                  f"(params={best_params}, inner={best_inner_auc:.3f}, {dt:.0f}s)", flush=True)

        except Exception as e:
            print(f"  [{ds_idx+1}/9] {held_out}: ERROR - {e}", flush=True)
            model_results[held_out] = {'auc': None, 'reason': str(e)}

    valid = [v['auc'] for v in model_results.values() if v.get('auc') is not None]
    elapsed = time.time() - t0
    mean_auc = np.mean(valid) if valid else None
    median_auc = np.median(valid) if valid else None

    print(f"\n  >>> {model_name}: mean={mean_auc:.3f}, median={median_auc:.3f}, "
          f"valid={len(valid)}/9, time={elapsed:.0f}s", flush=True)

    model_results['_mean_auc'] = round(float(mean_auc), 4) if mean_auc else None
    model_results['_median_auc'] = round(float(median_auc), 4) if median_auc else None
    model_results['_n_valid_folds'] = len(valid)
    model_results['_elapsed_seconds'] = round(elapsed, 1)
    all_results[model_name] = model_results
    checkpoint(all_results)  # Save after each model!


# ── Baseline: Default L2 C=1.0 ────────────────────────────────────
if 'L2_LogReg_default_C1' not in all_results:
    print("\n" + "=" * 70, flush=True)
    print("Baseline: L2 Default C=1.0 (no tuning)", flush=True)
    print("=" * 70, flush=True)

    default_results = {}
    for held_out in unique_datasets:
        train_mask = datasets_all != held_out
        test_mask = datasets_all == held_out
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        if len(np.unique(y_test)) < 2:
            default_results[held_out] = {'auc': None, 'reason': 'single_class'}
            continue

        clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                                 max_iter=2000, class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        default_results[held_out] = {'auc': round(float(auc), 4)}
        print(f"  {held_out}: AUC={auc:.3f}", flush=True)

    valid_d = [v['auc'] for v in default_results.values() if v.get('auc') is not None]
    default_results['_mean_auc'] = round(float(np.mean(valid_d)), 4) if valid_d else None
    default_results['_median_auc'] = round(float(np.median(valid_d)), 4) if valid_d else None
    print(f"  Default L2 mean: {default_results['_mean_auc']}", flush=True)
    all_results['L2_LogReg_default_C1'] = default_results
    checkpoint(all_results)


# ── Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("MODEL COMPARISON SUMMARY", flush=True)
print("=" * 70, flush=True)

model_names = [m[0] for m in MODELS] + ['L2_LogReg_default_C1']
rows = []
for ds in unique_datasets:
    row = {'dataset': ds}
    for mn in model_names:
        row[mn] = all_results.get(mn, {}).get(ds, {}).get('auc')
    rows.append(row)

for agg, fn in [('MEAN', '_mean_auc'), ('MEDIAN', '_median_auc')]:
    row = {'dataset': agg}
    for mn in model_names:
        row[mn] = all_results.get(mn, {}).get(fn)
    rows.append(row)

summary_df = pd.DataFrame(rows).set_index('dataset')
print(summary_df.to_string(), flush=True)
summary_df.to_csv(OUT / "model_comparison_summary.csv")

# Tuning impact
print("\n--- Tuning Impact (L2 tuned vs default C=1.0) ---", flush=True)
for ds in unique_datasets:
    tuned = all_results.get('L2_LogReg', {}).get(ds, {}).get('auc')
    default = all_results.get('L2_LogReg_default_C1', {}).get(ds, {}).get('auc')
    if tuned is not None and default is not None:
        diff = tuned - default
        sel = all_results['L2_LogReg'].get(ds, {}).get('best_params', {}).get('C', '?')
        print(f"  {ds}: tuned={tuned:.3f} default={default:.3f} diff={diff:+.3f} (C={sel})", flush=True)

# Save predictions
pred_df = pd.DataFrame(all_predictions)
if len(pred_df) > 0:
    pred_df.to_parquet(OUT / "tuning_lodo_predictions.parquet", index=False)

print(f"\nAll saved to {OUT}", flush=True)
print("Done!", flush=True)
