#!/usr/bin/env python3
"""
Complete the remaining nested CV models: finish LightGBM (2 remaining folds),
run SVM (reduced grid), and baseline L2.
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
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT = BASE / "tuning"
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

print(f"Model A: {len(model_a)} samples", flush=True)

# Load checkpoint
with open(RESULTS_FILE) as f:
    all_results = json.load(f)
print(f"Checkpoint has: {list(all_results.keys())}", flush=True)

all_predictions = []


def get_proba(clf, X):
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def inner_cv_select(X_train, y_train, model_name, param_list, build_fn, n_splits=5):
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
                clf = build_fn(params, y_tr)
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


def run_model(model_name, param_list, build_fn, held_out_datasets=None):
    """Run nested CV for a single model. Optionally limit to specific held-out datasets."""
    print(f"\n{'─'*50}", flush=True)
    print(f"Model: {model_name} ({len(param_list)} configs)", flush=True)
    print(f"{'─'*50}", flush=True)
    model_results = {}
    t0 = time.time()

    for ds_idx, held_out in enumerate(unique_datasets):
        if held_out_datasets and held_out not in held_out_datasets:
            continue

        train_mask = datasets_all != held_out
        test_mask = datasets_all == held_out
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        test_ids = [sample_ids[i] for i in range(len(sample_ids)) if test_mask[i]]

        if len(np.unique(y_test)) < 2:
            model_results[held_out] = {'auc': None, 'reason': 'single_class'}
            continue

        t1 = time.time()
        best_params, best_inner, inner_results = inner_cv_select(
            X_train, y_train, model_name, param_list, build_fn)

        if best_params is None:
            model_results[held_out] = {'auc': None, 'reason': 'no_valid_config'}
            continue

        try:
            clf = build_fn(best_params, y_train)
            clf.fit(X_train, y_train)
            y_prob = get_proba(clf, X_test)
            auc = roc_auc_score(y_test, y_prob)
            dt = time.time() - t1

            model_results[held_out] = {
                'auc': round(float(auc), 4),
                'best_params': {k: str(v) for k, v in best_params.items()},
                'best_inner_auc': round(float(best_inner), 4),
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

            print(f"  {held_out}: AUC={auc:.3f} (params={best_params}, {dt:.0f}s)", flush=True)
        except Exception as e:
            print(f"  {held_out}: ERROR - {e}", flush=True)
            model_results[held_out] = {'auc': None, 'reason': str(e)}

    return model_results


def checkpoint():
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════
# 1. Complete LightGBM (remaining 2 folds)
# ═══════════════════════════════════════════════════════════════════
# From partial results we know 7/9 folds completed
lgb_partial = {
    'GSE156699': {'auc': 0.716, 'best_params': {'n_estimators': '100', 'max_depth': '3', 'learning_rate': '0.05'}},
    'GSE173839': {'auc': 0.736, 'best_params': {'n_estimators': '50', 'max_depth': '3', 'learning_rate': '0.05'}},
    'GSE18864': {'auc': 0.594, 'best_params': {'n_estimators': '100', 'max_depth': '3', 'learning_rate': '0.05'}},
    'GSE194040': {'auc': 0.829, 'best_params': {'n_estimators': '100', 'max_depth': '3', 'learning_rate': '0.05'}},
    'GSE28739': {'auc': 0.727, 'best_params': {'n_estimators': '200', 'max_depth': '3', 'learning_rate': '0.05'}},
    'GSE30161': {'auc': 0.569, 'best_params': {'n_estimators': '100', 'max_depth': '3', 'learning_rate': '0.05'}},
    'GSE32062': {'auc': 0.500, 'best_params': {'n_estimators': '200', 'max_depth': '5', 'learning_rate': '0.05'}},
}
remaining_lgb = [ds for ds in unique_datasets if ds not in lgb_partial]
print(f"\nLightGBM: {len(lgb_partial)} folds from partial run, need {remaining_lgb}", flush=True)

lgb_params = [
    {'n_estimators': ne, 'max_depth': md, 'learning_rate': 0.05}
    for ne in [50, 100, 200] for md in [3, 5, 7]
]

def build_lgb(params, y_train):
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    return lgb.LGBMClassifier(
        n_estimators=params['n_estimators'], max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=0.8, colsample_bytree=0.3,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, verbose=-1)

lgb_remaining = run_model('LightGBM', lgb_params, build_lgb, remaining_lgb)

# Merge with partial results
lgb_all = {**lgb_partial, **lgb_remaining}
valid = [v['auc'] for v in lgb_all.values() if isinstance(v, dict) and v.get('auc') is not None]
lgb_all['_mean_auc'] = round(float(np.mean(valid)), 4) if valid else None
lgb_all['_median_auc'] = round(float(np.median(valid)), 4) if valid else None
lgb_all['_n_valid_folds'] = len(valid)
all_results['LightGBM'] = lgb_all
print(f"\n  LightGBM complete: mean={lgb_all['_mean_auc']}, median={lgb_all['_median_auc']}", flush=True)
checkpoint()


# ═══════════════════════════════════════════════════════════════════
# 2. SVM RBF (reduced grid: 4 configs)
# ═══════════════════════════════════════════════════════════════════
svm_params = [
    {'C': c, 'gamma': g}
    for c in [0.1, 1.0, 10.0]
    for g in ['scale']
]

def build_svm(params, y_train):
    return SVC(kernel='rbf', C=params['C'], gamma=params['gamma'],
               class_weight='balanced', probability=True, random_state=42)

svm_results = run_model('SVM_RBF', svm_params, build_svm)
valid = [v['auc'] for v in svm_results.values() if isinstance(v, dict) and v.get('auc') is not None]
svm_results['_mean_auc'] = round(float(np.mean(valid)), 4) if valid else None
svm_results['_median_auc'] = round(float(np.median(valid)), 4) if valid else None
svm_results['_n_valid_folds'] = len(valid)
all_results['SVM_RBF'] = svm_results
print(f"\n  SVM_RBF complete: mean={svm_results['_mean_auc']}", flush=True)
checkpoint()


# ═══════════════════════════════════════════════════════════════════
# 3. Baseline: L2 default C=1.0
# ═══════════════════════════════════════════════════════════════════
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
all_results['L2_LogReg_default_C1'] = default_results
print(f"  Default L2 mean: {default_results['_mean_auc']}", flush=True)
checkpoint()


# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("MODEL COMPARISON SUMMARY", flush=True)
print("=" * 70, flush=True)

model_names = ['L1_LogReg', 'L2_LogReg', 'ElasticNet_0.5', 'ElasticNet_0.9',
               'LightGBM', 'SVM_RBF', 'L2_LogReg_default_C1']
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
