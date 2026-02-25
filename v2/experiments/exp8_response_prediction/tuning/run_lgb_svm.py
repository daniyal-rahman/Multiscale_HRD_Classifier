#!/usr/bin/env python3
"""
Quick LightGBM + SVM-RBF comparison.
Reduced configs to fit within time limit.
Merges results into nested_cv_results.json.
"""
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

# Load data
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

print(f"Model A: {len(model_a)} samples, {len(gene_cols)} genes", flush=True)

# Load existing results
with open(RESULTS_FILE) as f:
    all_results = json.load(f)
print(f"Existing models: {list(all_results.keys())}", flush=True)

# Reduced LightGBM configs: 12 instead of 27
LGB_CONFIGS = [
    {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.05},
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05},
    {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05},
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
    {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.01},
    {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05},
    {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.01},
    {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
]

SVM_CONFIGS = [
    {'C': 0.01, 'gamma': 'scale'},
    {'C': 0.1, 'gamma': 'scale'},
    {'C': 1.0, 'gamma': 'scale'},
    {'C': 10.0, 'gamma': 'scale'},
    {'C': 0.1, 'gamma': 'auto'},
    {'C': 1.0, 'gamma': 'auto'},
]


def inner_cv(X_train, y_train, model_name, configs, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_auc, best_params = -1, None
    all_inner = []

    for params in configs:
        fold_aucs = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            if len(np.unique(y_val)) < 2:
                continue
            try:
                if model_name == 'LightGBM':
                    n_pos = y_tr.sum()
                    n_neg = len(y_tr) - n_pos
                    clf = lgb.LGBMClassifier(
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'],
                        learning_rate=params['learning_rate'],
                        subsample=0.8, colsample_bytree=0.3,
                        scale_pos_weight=n_neg / max(n_pos, 1),
                        random_state=42, verbose=-1, n_jobs=1)
                else:
                    clf = SVC(kernel='rbf', C=params['C'], gamma=params['gamma'],
                              class_weight='balanced', probability=True, random_state=42)
                clf.fit(X_tr, y_tr)
                prob = clf.predict_proba(X_val)[:, 1]
                fold_aucs.append(roc_auc_score(y_val, prob))
            except Exception:
                pass

        mean_auc = np.mean(fold_aucs) if fold_aucs else 0.0
        all_inner.append({
            'params': {k: str(v) for k, v in params.items()},
            'mean_inner_auc': round(float(mean_auc), 4),
        })
        if mean_auc > best_auc:
            best_auc, best_params = mean_auc, params

    return best_params, best_auc, all_inner


for model_name, configs in [('LightGBM', LGB_CONFIGS), ('SVM_RBF', SVM_CONFIGS)]:
    if model_name in all_results:
        print(f"\n{model_name}: already done, skipping", flush=True)
        continue

    print(f"\n{'='*50}", flush=True)
    print(f"{model_name} ({len(configs)} configs)", flush=True)
    print(f"{'='*50}", flush=True)
    model_results = {}
    t0 = time.time()

    for ds_idx, held_out in enumerate(unique_datasets):
        train_mask = datasets_all != held_out
        test_mask = datasets_all == held_out
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        if len(np.unique(y_test)) < 2:
            model_results[held_out] = {'auc': None, 'reason': 'single_class'}
            continue

        t1 = time.time()
        best_params, best_inner_auc, inner = inner_cv(X_train, y_train, model_name, configs)

        if best_params is None:
            model_results[held_out] = {'auc': None, 'reason': 'no_valid'}
            continue

        if model_name == 'LightGBM':
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            clf = lgb.LGBMClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                learning_rate=best_params['learning_rate'],
                subsample=0.8, colsample_bytree=0.3,
                scale_pos_weight=n_neg / max(n_pos, 1),
                random_state=42, verbose=-1, n_jobs=1)
        else:
            clf = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'],
                      class_weight='balanced', probability=True, random_state=42)

        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
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
        }
        print(f"  [{ds_idx+1}/9] {held_out}: AUC={auc:.3f} (params={best_params}, {dt:.0f}s)", flush=True)

    valid = [v['auc'] for v in model_results.values() if v.get('auc') is not None]
    elapsed = time.time() - t0
    mean_auc = np.mean(valid) if valid else None
    model_results['_mean_auc'] = round(float(mean_auc), 4) if mean_auc else None
    model_results['_median_auc'] = round(float(np.median(valid)), 4) if valid else None
    model_results['_n_valid_folds'] = len(valid)
    model_results['_elapsed_seconds'] = round(elapsed, 1)
    all_results[model_name] = model_results

    print(f"\n  >>> {model_name}: mean={mean_auc:.3f}, time={elapsed:.0f}s", flush=True)

    # Checkpoint
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

# Baseline
if 'L2_LogReg_default_C1' not in all_results:
    print("\nBaseline: L2 default C=1.0", flush=True)
    default_results = {}
    for held_out in unique_datasets:
        train_mask = datasets_all != held_out
        test_mask = datasets_all == held_out
        if len(np.unique(y_all[test_mask])) < 2:
            default_results[held_out] = {'auc': None}
            continue
        clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                                  max_iter=3000, class_weight='balanced', random_state=42)
        clf.fit(X_all[train_mask], y_all[train_mask])
        auc = roc_auc_score(y_all[test_mask], clf.predict_proba(X_all[test_mask])[:, 1])
        default_results[held_out] = {'auc': round(float(auc), 4)}
        print(f"  {held_out}: AUC={auc:.3f}", flush=True)
    valid_d = [v['auc'] for v in default_results.values() if v.get('auc') is not None]
    default_results['_mean_auc'] = round(float(np.mean(valid_d)), 4)
    default_results['_median_auc'] = round(float(np.median(valid_d)), 4)
    all_results['L2_LogReg_default_C1'] = default_results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

# Summary
print("\n" + "="*60, flush=True)
print("ALL MODELS SUMMARY", flush=True)
print("="*60, flush=True)
for mn in sorted(all_results.keys()):
    print(f"  {mn}: mean={all_results[mn].get('_mean_auc')}", flush=True)
print("Done!", flush=True)
