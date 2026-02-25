#!/usr/bin/env python3
"""
Nested CV v2: Corrected run with:
1. GSE28739 excluded (dye-swap incompatible)
2. GSE32062 excluded (99.7% NaN from Agilent gene mapping bug)
3. Batch residualization comparison (raw ranks vs dataset-residualized)
4. Models: L2 LogReg (tuned), LightGBM, SVM-RBF, plus L2 default baseline
5. Feature selection at best hyperparams (for Task #3)

Checkpoints after each model/config to nested_cv_v2_results.json.
"""
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import json
import warnings
import time
import sys
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
RESULTS_FILE = OUT / "nested_cv_v2_results.json"

EXCLUDE_DATASETS = ['GSE28739', 'GSE32062']

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

# Exclude problematic datasets
model_a = model_a[~model_a['dataset'].isin(EXCLUDE_DATASETS)]
print(f"Excluded {EXCLUDE_DATASETS}", flush=True)

X_raw = model_a[gene_cols].values.astype(np.float32)
X_raw = np.nan_to_num(X_raw, nan=0.5)
y_all = model_a['response_binary'].values.astype(int)
datasets_all = model_a['dataset'].values
sample_ids = model_a.index.tolist()
unique_datasets = sorted(model_a['dataset'].unique())

print(f"Model A (clean): {len(model_a)} samples, {len(gene_cols)} genes, {len(unique_datasets)} datasets", flush=True)
for ds in unique_datasets:
    m = datasets_all == ds
    print(f"  {ds}: n={m.sum()}, pos={y_all[m].sum()}, neg={(1-y_all[m]).sum()}", flush=True)


# ── Batch residualization ──────────────────────────────────────────
def residualize_by_dataset(X, datasets):
    """Remove dataset-level mean from each feature (per-gene centering by batch)."""
    X_res = X.copy()
    for ds in np.unique(datasets):
        mask = datasets == ds
        ds_mean = X[mask].mean(axis=0)
        X_res[mask] -= ds_mean
        X_res[mask] += 0.5  # Re-center around 0.5 (rank midpoint)
    return X_res

X_resid = residualize_by_dataset(X_raw, datasets_all)
print(f"Residualized features: shape={X_resid.shape}", flush=True)


# ── Resume from checkpoint ─────────────────────────────────────────
if RESULTS_FILE.exists():
    with open(RESULTS_FILE) as f:
        all_results = json.load(f)
    print(f"Resuming from checkpoint: {list(all_results.keys())} done", flush=True)
else:
    all_results = {}

all_predictions = []


# ── Model definitions ──────────────────────────────────────────────
# L2: fine-grained C grid (v1 showed C=0.01 is optimal)
C_values_fine = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]

MODELS = [
    ('L2_LogReg', [{'C': c} for c in C_values_fine], False),
    ('LightGBM', [
        {'n_estimators': ne, 'max_depth': md, 'learning_rate': lr}
        for ne in [50, 100, 200] for md in [3, 5] for lr in [0.01, 0.05, 0.1]
    ], True),
    ('SVM_RBF', [
        {'C': c, 'gamma': g}
        for c in [0.01, 0.1, 1.0, 10.0] for g in ['scale', 'auto']
    ], False),
]


def build_model(model_name, params, y_train=None):
    if model_name == 'L2_LogReg':
        return LogisticRegression(penalty='l2', C=params['C'], solver='lbfgs',
                                  max_iter=3000, class_weight='balanced', random_state=42)
    elif model_name == 'LightGBM':
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        return lgb.LGBMClassifier(
            n_estimators=params['n_estimators'], max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=0.8, colsample_bytree=0.3,
            scale_pos_weight=n_neg / max(n_pos, 1),
            random_state=42, verbose=-1, n_jobs=1)
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


def run_lodo(X, y, datasets, model_name, param_list, needs_y, label):
    """Run full LODO-CV for one model on one feature set."""
    key = f"{model_name}_{label}"
    if key in all_results:
        print(f"\n{key}: already done (checkpoint), skipping", flush=True)
        return

    print(f"\n{'─'*50}", flush=True)
    print(f"Model: {key} ({len(param_list)} configs)", flush=True)
    print(f"{'─'*50}", flush=True)
    model_results = {}
    t0 = time.time()

    for ds_idx, held_out in enumerate(unique_datasets):
        train_mask = datasets == held_out
        train_mask = ~train_mask
        test_mask = datasets == held_out
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        test_ids = [sample_ids[i] for i in range(len(sample_ids)) if test_mask[i]]

        if len(np.unique(y_test)) < 2:
            print(f"  [{ds_idx+1}/{len(unique_datasets)}] {held_out}: SKIP (single class)", flush=True)
            model_results[held_out] = {'auc': None, 'reason': 'single_class_test'}
            continue

        t1 = time.time()
        best_params, best_inner_auc, inner_results = inner_cv_select(
            X_train, y_train, model_name, param_list, needs_y)

        if best_params is None:
            print(f"  [{ds_idx+1}/{len(unique_datasets)}] {held_out}: NO VALID CONFIG", flush=True)
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
            }

            for sid, prob, ty in zip(test_ids, y_prob, y_test):
                all_predictions.append({
                    'sample_id': sid, 'model': key,
                    'held_out_dataset': held_out,
                    'predicted_prob': float(prob), 'true_label': int(ty),
                })

            print(f"  [{ds_idx+1}/{len(unique_datasets)}] {held_out}: AUC={auc:.3f} "
                  f"(params={best_params}, inner={best_inner_auc:.3f}, {dt:.0f}s)", flush=True)

        except Exception as e:
            print(f"  [{ds_idx+1}/{len(unique_datasets)}] {held_out}: ERROR - {e}", flush=True)
            model_results[held_out] = {'auc': None, 'reason': str(e)}

    valid = [v['auc'] for v in model_results.values() if v.get('auc') is not None]
    elapsed = time.time() - t0
    mean_auc = np.mean(valid) if valid else None
    median_auc = np.median(valid) if valid else None

    print(f"\n  >>> {key}: mean={mean_auc:.3f}, median={median_auc:.3f}, "
          f"valid={len(valid)}/{len(unique_datasets)}, time={elapsed:.0f}s", flush=True)

    model_results['_mean_auc'] = round(float(mean_auc), 4) if mean_auc else None
    model_results['_median_auc'] = round(float(median_auc), 4) if median_auc else None
    model_results['_n_valid_folds'] = len(valid)
    model_results['_elapsed_seconds'] = round(elapsed, 1)
    all_results[key] = model_results
    checkpoint(all_results)


# ── Run all models on both feature sets ────────────────────────────
print("\n" + "=" * 70, flush=True)
print("Nested LODO-CV v2 (clean datasets, raw + residualized)", flush=True)
print(f"Excluded: {EXCLUDE_DATASETS}", flush=True)
print("=" * 70, flush=True)

# L2 baseline (no tuning) on both feature sets
for label, X in [('raw', X_raw), ('resid', X_resid)]:
    key = f"L2_default_C1_{label}"
    if key not in all_results:
        print(f"\nBaseline: {key}", flush=True)
        default_results = {}
        for held_out in unique_datasets:
            train_mask = datasets_all != held_out
            test_mask = datasets_all == held_out
            X_train, y_train = X[train_mask], y_all[train_mask]
            X_test, y_test = X[test_mask], y_all[test_mask]
            if len(np.unique(y_test)) < 2:
                default_results[held_out] = {'auc': None, 'reason': 'single_class'}
                continue
            clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                                     max_iter=3000, class_weight='balanced', random_state=42)
            clf.fit(X_train, y_train)
            auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            default_results[held_out] = {'auc': round(float(auc), 4)}
            print(f"  {held_out}: AUC={auc:.3f}", flush=True)
        valid_d = [v['auc'] for v in default_results.values() if v.get('auc') is not None]
        default_results['_mean_auc'] = round(float(np.mean(valid_d)), 4) if valid_d else None
        default_results['_median_auc'] = round(float(np.median(valid_d)), 4) if valid_d else None
        all_results[key] = default_results
        checkpoint(all_results)
        print(f"  {key} mean: {default_results['_mean_auc']}", flush=True)

# Tuned models on both feature sets
for model_name, param_list, needs_y in MODELS:
    for label, X in [('raw', X_raw), ('resid', X_resid)]:
        run_lodo(X, y_all, datasets_all, model_name, param_list, needs_y, label)


# ── Summary comparison ─────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("MODEL COMPARISON SUMMARY (v2, clean datasets)", flush=True)
print("=" * 70, flush=True)

model_keys = sorted(all_results.keys())
rows = []
for ds in unique_datasets:
    row = {'dataset': ds}
    for mk in model_keys:
        row[mk] = all_results.get(mk, {}).get(ds, {}).get('auc')
    rows.append(row)

for agg, fn in [('MEAN', '_mean_auc'), ('MEDIAN', '_median_auc')]:
    row = {'dataset': agg}
    for mk in model_keys:
        row[mk] = all_results.get(mk, {}).get(fn)
    rows.append(row)

summary_df = pd.DataFrame(rows).set_index('dataset')
print(summary_df.to_string(), flush=True)
summary_df.to_csv(OUT / "model_comparison_v2_summary.csv")

# Residualization impact
print("\n--- Residualization Impact ---", flush=True)
for model_name in ['L2_default_C1', 'L2_LogReg', 'LightGBM', 'SVM_RBF']:
    raw_key = f"{model_name}_raw"
    res_key = f"{model_name}_resid"
    if raw_key in all_results and res_key in all_results:
        raw_mean = all_results[raw_key].get('_mean_auc')
        res_mean = all_results[res_key].get('_mean_auc')
        if raw_mean and res_mean:
            print(f"  {model_name}: raw={raw_mean:.3f}, resid={res_mean:.3f}, diff={res_mean - raw_mean:+.3f}", flush=True)

# Save predictions
pred_df = pd.DataFrame(all_predictions)
if len(pred_df) > 0:
    pred_df.to_parquet(OUT / "tuning_lodo_predictions_v2.parquet", index=False)

print(f"\nAll saved to {OUT}", flush=True)
print("Done!", flush=True)
