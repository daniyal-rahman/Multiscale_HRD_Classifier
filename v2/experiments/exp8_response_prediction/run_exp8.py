#!/usr/bin/env python3
"""
Exp8: Drug Response Prediction — Rank-Transformed Pooled Clinical Model

Phase 1: Data harmonization
  - Rank-transform each dataset independently (within-sample gene ranks → [0,1])
  - Intersect to 11,140 common genes across all 10 datasets
  - Binarize response labels per dataset definitions
  - Output single pooled matrix

Phase 2: Model A — L2 logistic regression + LightGBM, LODO-CV
  - ~900 clinical samples (738 ovarian platinum + 142 PARPi breast + 24 cisplatin TNBC)
  - Exclude GDSC cell lines and paclitaxel-only controls
  - Dataset ID as categorical covariate
  - Leave-one-dataset-out cross-validation, report AUC per held-out dataset

Phase 3: I-SPY2 interaction test
  - Train on all non-I-SPY2 clinical data
  - Predict on all I-SPY2 (both PARPi and control arms)
  - Test: logit(pCR) ~ score + treatment_arm + score:treatment_arm
  - If interaction significant → learned PARPi-specific signal
  - If not → learned general chemo-sensitivity
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not available, skipping GBT models")

try:
    import statsmodels.formula.api as smf
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("statsmodels not available, using manual interaction test")

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
OUT = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# PHASE 1: Data Harmonization
# ============================================================
print("=" * 70)
print("PHASE 1: Data Harmonization")
print("=" * 70)

# Load common genes
common_genes = [g.strip() for g in open(BASE / "common_genes_all_10_datasets.txt")]
print(f"Common genes: {len(common_genes)}")

# Dataset configurations
datasets_config = {
    'TCGA-OV':   {'expr': 'TCGA-OV_expression_std.parquet',   'resp': 'TCGA-OV_response.parquet',   'category': 'clinical_platinum'},
    'GSE32062':  {'expr': 'GSE32062_expression_std.parquet',  'resp': 'GSE32062_response.parquet',  'category': 'clinical_platinum'},
    'GSE156699': {'expr': 'GSE156699_expression_std.parquet', 'resp': 'GSE156699_response.parquet', 'category': 'clinical_platinum'},
    'GSE63885':  {'expr': 'GSE63885_expression_std.parquet',  'resp': 'GSE63885_response.parquet',  'category': 'clinical_platinum'},
    'GSE30161':  {'expr': 'GSE30161_expression_std.parquet',  'resp': 'GSE30161_response.parquet',  'category': 'clinical_platinum'},
    'GSE28739':  {'expr': 'GSE28739_expression_std.parquet',  'resp': 'GSE28739_response.parquet',  'category': 'clinical_platinum'},
    'GSE18864':  {'expr': 'GSE18864_expression_std.parquet',  'resp': 'GSE18864_response.parquet',  'category': 'clinical_cisplatin'},
    'GSE173839': {'expr': 'GSE173839_expression_std.parquet', 'resp': 'GSE173839_response.parquet', 'category': 'clinical_ispy2'},
    'GSE194040': {'expr': 'GSE194040_expression_std.parquet', 'resp': 'GSE194040_response.parquet', 'category': 'clinical_ispy2'},
    'GDSC':      {'expr': 'GDSC_expression_std.parquet',      'resp': 'GDSC_response.parquet',      'category': 'cell_line'},
}

pooled_dfs = []
for name, cfg in datasets_config.items():
    print(f"\nProcessing {name}...")

    # Load expression (samples x genes)
    expr = pd.read_parquet(BASE / cfg['expr'])

    # Subset to common genes
    available = [g for g in common_genes if g in expr.columns]
    print(f"  Shape: {expr.shape}, common genes available: {len(available)}/{len(common_genes)}")
    expr = expr[available]

    # Fill any missing common genes with NaN
    for g in common_genes:
        if g not in expr.columns:
            expr[g] = np.nan
    expr = expr[common_genes]

    # Rank transform within each sample (row) independently → [0, 1]
    ranks = expr.rank(axis=1, pct=True)

    # Load response labels
    resp = pd.read_parquet(BASE / cfg['resp'])

    # GDSC has multiple drugs per cell line — use olaparib only for pooling
    if name == 'GDSC':
        resp = resp[resp['drug'] == 'olaparib']

    # Match sample IDs
    resp = resp.set_index('sample_id')
    # Ensure string index for matching
    ranks.index = ranks.index.astype(str)
    resp.index = resp.index.astype(str)
    common_samples = ranks.index.intersection(resp.index)

    print(f"  Matched samples: {len(common_samples)}")

    ranks_matched = ranks.loc[common_samples].copy()
    resp_matched = resp.loc[common_samples]

    # Add metadata columns
    ranks_matched['response_binary'] = resp_matched['response_binary'].values
    if 'response_continuous' in resp_matched.columns:
        ranks_matched['response_continuous'] = resp_matched['response_continuous'].values
    else:
        ranks_matched['response_continuous'] = np.nan
    ranks_matched['drug'] = resp_matched['drug'].values if 'drug' in resp_matched.columns else 'unknown'
    ranks_matched['dataset'] = name
    ranks_matched['category'] = cfg['category']
    if 'cancer_type' in resp_matched.columns:
        ranks_matched['cancer_type'] = resp_matched['cancer_type'].values
    else:
        ranks_matched['cancer_type'] = 'unknown'

    pooled_dfs.append(ranks_matched)

# Combine all datasets
pooled = pd.concat(pooled_dfs)
print(f"\n{'='*50}")
print(f"Pooled matrix: {pooled.shape[0]} samples x {len(common_genes)} genes + metadata")
print(f"\nResponse rate by dataset:")
for ds in pooled['dataset'].unique():
    sub = pooled[pooled['dataset'] == ds]
    print(f"  {ds}: {sub['response_binary'].mean():.3f} ({int(sub['response_binary'].sum())}/{len(sub)})")

pooled.to_parquet(OUT / 'pooled_rank_matrix.parquet')
print(f"\nSaved pooled matrix")


# ============================================================
# PHASE 2: Model A — LODO-CV on clinical samples
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: Model A — Leave-One-Dataset-Out Cross-Validation")
print("=" * 70)

# Model A includes:
#   - All platinum/PARPi response clinical samples
#   - Excludes GDSC cell lines
#   - Excludes paclitaxel-only controls from I-SPY2
ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']

model_a_mask = (
    (pooled['category'] != 'cell_line') &
    ~(
        (pooled['category'] == 'clinical_ispy2') &
        (~pooled['drug'].isin(ispy2_parpi_drugs))
    )
)
model_a = pooled[model_a_mask].copy()

print(f"\nModel A: {len(model_a)} clinical samples")
print(f"Response rate: {model_a['response_binary'].mean():.3f}")
print(f"\nPer-dataset breakdown:")
for ds in sorted(model_a['dataset'].unique()):
    sub = model_a[model_a['dataset'] == ds]
    print(f"  {ds}: n={len(sub)}, response_rate={sub['response_binary'].mean():.3f} "
          f"(pos={int(sub['response_binary'].sum())}, neg={len(sub)-int(sub['response_binary'].sum())})")

# Prepare arrays
gene_cols = common_genes
X_all = model_a[gene_cols].values
y_all = model_a['response_binary'].values.astype(int)
datasets_all = model_a['dataset'].values

unique_datasets = sorted(model_a['dataset'].unique())
dataset_to_idx = {d: i for i, d in enumerate(unique_datasets)}


def run_lodo_cv(X, y, datasets, model_type='lr', use_covariates=True, label=''):
    """Run leave-one-dataset-out CV."""
    results = {}
    for held_out in unique_datasets:
        train_mask = datasets != held_out
        test_mask = datasets == held_out

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if use_covariates:
            # One-hot encode dataset ID
            ds_train = datasets[train_mask]
            ds_test_arr = datasets[test_mask]

            train_dummies = np.zeros((len(X_train), len(unique_datasets)))
            for i, d in enumerate(ds_train):
                train_dummies[i, dataset_to_idx[d]] = 1

            test_dummies = np.zeros((len(X_test), len(unique_datasets)))
            for i, d in enumerate(ds_test_arr):
                test_dummies[i, dataset_to_idx[d]] = 1

            X_train = np.hstack([X_train, train_dummies])
            X_test = np.hstack([X_test, test_dummies])

        # Handle NaN (rank of missing gene → 0.5 = median rank)
        X_train = np.nan_to_num(X_train, nan=0.5)
        X_test = np.nan_to_num(X_test, nan=0.5)

        # Skip single-class test sets
        if len(np.unique(y_test)) < 2:
            results[held_out] = {
                'auc': float('nan'), 'n_test': int(test_mask.sum()),
                'n_pos': int(y_test.sum()), 'reason': 'single_class'
            }
            print(f"  {held_out}: SKIPPED (only one class)")
            continue

        if model_type == 'lr':
            clf = LogisticRegression(
                penalty='l2', C=1.0, solver='lbfgs',
                max_iter=5000, class_weight='balanced', random_state=42
            )
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]

        elif model_type == 'lgb' and HAS_LGB:
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            clf = lgb.LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.3,
                scale_pos_weight=n_neg / max(n_pos, 1),
                random_state=42, verbose=-1,
            )
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:
            continue

        auc = roc_auc_score(y_test, y_prob)
        results[held_out] = {
            'auc': round(float(auc), 4),
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
            'n_pos_test': int(y_test.sum()),
            'n_neg_test': int((1 - y_test).sum()),
        }
        print(f"  {held_out}: AUC={auc:.3f} (n={len(y_test)}, "
              f"+={int(y_test.sum())}, -={int((1-y_test).sum())})")

    # Summary
    valid = [v['auc'] for v in results.values() if not np.isnan(v.get('auc', float('nan')))]
    if valid:
        print(f"  --- Mean AUC: {np.mean(valid):.3f} (n={len(valid)} datasets) ---")
        results['_mean_auc'] = round(float(np.mean(valid)), 4)
        results['_median_auc'] = round(float(np.median(valid)), 4)
    return results


# Run all model variants
print("\n--- L2 Logistic Regression WITH dataset covariates ---")
results_lr_cov = run_lodo_cv(X_all, y_all, datasets_all, 'lr', True)

print("\n--- L2 Logistic Regression WITHOUT dataset covariates ---")
results_lr_nocov = run_lodo_cv(X_all, y_all, datasets_all, 'lr', False)

results_lgb_cov = {}
if HAS_LGB:
    print("\n--- LightGBM WITH dataset covariates ---")
    results_lgb_cov = run_lodo_cv(X_all, y_all, datasets_all, 'lgb', True)


# ============================================================
# PHASE 3: I-SPY2 Treatment × Score Interaction Test
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: I-SPY2 Treatment × Score Interaction Test")
print("=" * 70)

# Training: all non-I-SPY2 clinical data (no GDSC, no I-SPY2)
non_ispy2_mask = (pooled['category'] != 'cell_line') & (pooled['category'] != 'clinical_ispy2')
# Test: ALL I-SPY2 (both PARPi and control arms)
ispy2_mask = pooled['category'] == 'clinical_ispy2'

X_train_int = pooled.loc[non_ispy2_mask, gene_cols].values
y_train_int = pooled.loc[non_ispy2_mask, 'response_binary'].values.astype(int)

X_ispy2 = pooled.loc[ispy2_mask, gene_cols].values
y_ispy2 = pooled.loc[ispy2_mask, 'response_binary'].values.astype(int)
drug_ispy2 = pooled.loc[ispy2_mask, 'drug'].values
dataset_ispy2 = pooled.loc[ispy2_mask, 'dataset'].values

X_train_int = np.nan_to_num(X_train_int, nan=0.5)
X_ispy2 = np.nan_to_num(X_ispy2, nan=0.5)

n_parpi = sum(d in ispy2_parpi_drugs for d in drug_ispy2)
n_control = sum(d not in ispy2_parpi_drugs for d in drug_ispy2)
print(f"\nTraining: {len(X_train_int)} non-I-SPY2 clinical samples")
print(f"Testing:  {len(X_ispy2)} I-SPY2 samples ({n_parpi} PARPi, {n_control} control)")

# Train model on non-I-SPY2 data
lr_int = LogisticRegression(
    penalty='l2', C=1.0, solver='lbfgs',
    max_iter=5000, class_weight='balanced', random_state=42
)
lr_int.fit(X_train_int, y_train_int)

# Predict on I-SPY2
scores_ispy2 = lr_int.predict_proba(X_ispy2)[:, 1]

# Build test dataframe
ispy2_df = pd.DataFrame({
    'score': scores_ispy2,
    'pCR': y_ispy2,
    'is_parpi_arm': [1 if d in ispy2_parpi_drugs else 0 for d in drug_ispy2],
    'drug': drug_ispy2,
    'dataset': dataset_ispy2,
})

print(f"\nPrediction statistics:")
parpi_data = ispy2_df[ispy2_df['is_parpi_arm'] == 1]
ctrl_data = ispy2_df[ispy2_df['is_parpi_arm'] == 0]
print(f"  Score mean (PARPi arm):   {parpi_data['score'].mean():.3f} ± {parpi_data['score'].std():.3f}")
print(f"  Score mean (Control arm): {ctrl_data['score'].mean():.3f} ± {ctrl_data['score'].std():.3f}")
print(f"  pCR rate  (PARPi arm):    {parpi_data['pCR'].mean():.3f} ({int(parpi_data['pCR'].sum())}/{len(parpi_data)})")
print(f"  pCR rate  (Control arm):  {ctrl_data['pCR'].mean():.3f} ({int(ctrl_data['pCR'].sum())}/{len(ctrl_data)})")

# Stratified AUCs
interaction_results = {}
for arm_label, arm_val in [('PARPi', 1), ('Control', 0)]:
    arm_d = ispy2_df[ispy2_df['is_parpi_arm'] == arm_val]
    if len(arm_d['pCR'].unique()) >= 2:
        auc = roc_auc_score(arm_d['pCR'], arm_d['score'])
        print(f"  AUC ({arm_label} arm): {auc:.3f} (n={len(arm_d)})")
        interaction_results[f'auc_{arm_label.lower()}'] = round(float(auc), 4)
    else:
        print(f"  AUC ({arm_label} arm): CANNOT COMPUTE (single class)")

# Overall AUC on all I-SPY2
if len(ispy2_df['pCR'].unique()) >= 2:
    auc_all = roc_auc_score(ispy2_df['pCR'], ispy2_df['score'])
    print(f"  AUC (all I-SPY2):  {auc_all:.3f} (n={len(ispy2_df)})")
    interaction_results['auc_all_ispy2'] = round(float(auc_all), 4)

# Interaction test: logit(pCR) ~ score + is_parpi_arm + score:is_parpi_arm
print(f"\n--- Interaction Test ---")
if HAS_SM:
    try:
        int_model = smf.logit('pCR ~ score * is_parpi_arm', data=ispy2_df).fit(disp=0)
        print(int_model.summary2().tables[1].to_string())

        interaction_coef = float(int_model.params.get('score:is_parpi_arm', np.nan))
        interaction_pval = float(int_model.pvalues.get('score:is_parpi_arm', np.nan))
        score_coef = float(int_model.params.get('score', np.nan))
        score_pval = float(int_model.pvalues.get('score', np.nan))

        print(f"\n*** INTERACTION TERM: coef={interaction_coef:.4f}, p={interaction_pval:.4f} ***")
        if interaction_pval < 0.05:
            print(">>> SIGNIFICANT: Model captures PARPi-SPECIFIC signal!")
        else:
            print(">>> NOT SIGNIFICANT: Model likely captures general chemo-sensitivity")

        interaction_results.update({
            'interaction_coef': round(interaction_coef, 4),
            'interaction_pval': round(interaction_pval, 6),
            'score_coef': round(score_coef, 4),
            'score_pval': round(score_pval, 6),
            'arm_coef': round(float(int_model.params.get('is_parpi_arm', np.nan)), 4),
            'arm_pval': round(float(int_model.pvalues.get('is_parpi_arm', np.nan)), 6),
        })

        # Per-study interaction tests
        for study in ['GSE173839', 'GSE194040']:
            study_data = ispy2_df[ispy2_df['dataset'] == study]
            if len(study_data) > 10 and len(study_data['pCR'].unique()) >= 2:
                try:
                    m = smf.logit('pCR ~ score * is_parpi_arm', data=study_data).fit(disp=0)
                    ip = float(m.pvalues.get('score:is_parpi_arm', np.nan))
                    ic = float(m.params.get('score:is_parpi_arm', np.nan))
                    print(f"\n  {study} interaction: coef={ic:.4f}, p={ip:.4f}")
                    interaction_results[f'{study}_interaction_coef'] = round(ic, 4)
                    interaction_results[f'{study}_interaction_pval'] = round(ip, 6)

                    for arm_label, arm_val in [('PARPi', 1), ('Control', 0)]:
                        arm_d = study_data[study_data['is_parpi_arm'] == arm_val]
                        if len(arm_d['pCR'].unique()) >= 2:
                            auc = roc_auc_score(arm_d['pCR'], arm_d['score'])
                            print(f"    AUC on {arm_label} arm: {auc:.3f} (n={len(arm_d)})")
                            interaction_results[f'{study}_auc_{arm_label.lower()}'] = round(float(auc), 4)
                except Exception as e:
                    print(f"  {study} interaction failed: {e}")

    except Exception as e:
        print(f"Interaction model failed: {e}")
        interaction_results['error'] = str(e)
else:
    print("statsmodels not available — skipping formal interaction test")

# Specificity check: paclitaxel-control pCR patients
print(f"\n--- Specificity Check ---")
paclitaxel_pcr = ispy2_df[(ispy2_df['is_parpi_arm'] == 0) & (ispy2_df['pCR'] == 1)]
paclitaxel_nopcr = ispy2_df[(ispy2_df['is_parpi_arm'] == 0) & (ispy2_df['pCR'] == 0)]
parpi_pcr = ispy2_df[(ispy2_df['is_parpi_arm'] == 1) & (ispy2_df['pCR'] == 1)]
parpi_nopcr = ispy2_df[(ispy2_df['is_parpi_arm'] == 1) & (ispy2_df['pCR'] == 0)]

print(f"  Mean score — PARPi pCR:       {parpi_pcr['score'].mean():.3f} (n={len(parpi_pcr)})")
print(f"  Mean score — PARPi no-pCR:    {parpi_nopcr['score'].mean():.3f} (n={len(parpi_nopcr)})")
print(f"  Mean score — Control pCR:     {paclitaxel_pcr['score'].mean():.3f} (n={len(paclitaxel_pcr)})")
print(f"  Mean score — Control no-pCR:  {paclitaxel_nopcr['score'].mean():.3f} (n={len(paclitaxel_nopcr)})")
print(f"\n  If PARPi-specific: PARPi pCR >> Control pCR scores")
print(f"  If general chemo:  PARPi pCR ≈ Control pCR scores")

if len(parpi_pcr) > 0 and len(paclitaxel_pcr) > 0:
    stat, p = mannwhitneyu(parpi_pcr['score'], paclitaxel_pcr['score'], alternative='greater')
    print(f"\n  Mann-Whitney (PARPi pCR > Control pCR scores): U={stat:.0f}, p={p:.4f}")
    interaction_results['specificity_mw_U'] = float(stat)
    interaction_results['specificity_mw_p'] = round(float(p), 6)

interaction_results['specificity'] = {
    'mean_score_parpi_pcr': round(float(parpi_pcr['score'].mean()), 4) if len(parpi_pcr) > 0 else None,
    'mean_score_parpi_nopcr': round(float(parpi_nopcr['score'].mean()), 4) if len(parpi_nopcr) > 0 else None,
    'mean_score_ctrl_pcr': round(float(paclitaxel_pcr['score'].mean()), 4) if len(paclitaxel_pcr) > 0 else None,
    'mean_score_ctrl_nopcr': round(float(paclitaxel_nopcr['score'].mean()), 4) if len(paclitaxel_nopcr) > 0 else None,
    'n_parpi_pcr': len(parpi_pcr),
    'n_ctrl_pcr': len(paclitaxel_pcr),
}


# ============================================================
# PHASE 4: Model B — GDSC-only (mechanistic probe)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: Model B — GDSC Cell Line Drug Sensitivity (mechanistic)")
print("=" * 70)

gdsc = pooled[pooled['category'] == 'cell_line'].copy()
# Use only olaparib for direct comparison
gdsc_ola = gdsc[gdsc['drug'] == 'olaparib']
print(f"GDSC olaparib samples: {len(gdsc_ola)}")

X_gdsc = gdsc_ola[gene_cols].values
y_gdsc = gdsc_ola['response_binary'].values.astype(int)
X_gdsc = np.nan_to_num(X_gdsc, nan=0.5)

# 5-fold CV on GDSC
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gdsc_aucs = []
for fold, (tr_idx, te_idx) in enumerate(skf.split(X_gdsc, y_gdsc)):
    clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                             max_iter=5000, class_weight='balanced', random_state=42)
    clf.fit(X_gdsc[tr_idx], y_gdsc[tr_idx])
    y_prob = clf.predict_proba(X_gdsc[te_idx])[:, 1]
    auc = roc_auc_score(y_gdsc[te_idx], y_prob)
    gdsc_aucs.append(auc)
    print(f"  Fold {fold+1}: AUC={auc:.3f}")
print(f"  Mean AUC: {np.mean(gdsc_aucs):.3f} ± {np.std(gdsc_aucs):.3f}")

# Train full GDSC model to get feature weights
clf_gdsc = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                               max_iter=5000, class_weight='balanced', random_state=42)
clf_gdsc.fit(X_gdsc, y_gdsc)

# Train full clinical Model A to get feature weights
X_clinical = np.nan_to_num(X_all, nan=0.5)
clf_clinical = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                                   max_iter=5000, class_weight='balanced', random_state=42)
clf_clinical.fit(X_clinical, y_all)

# Compare feature weights between Model A and Model B
weights_clinical = clf_clinical.coef_[0]
weights_gdsc = clf_gdsc.coef_[0]

from scipy.stats import spearmanr
rho, p_corr = spearmanr(weights_clinical, weights_gdsc)
print(f"\nFeature weight correlation (Model A vs Model B):")
print(f"  Spearman rho={rho:.3f}, p={p_corr:.2e}")
print(f"  If high: signal is cell-intrinsic (DNA repair biology)")
print(f"  If low:  clinical model uses TME/context-dependent features")

# Top 50 genes by absolute weight in each model
top_clinical = pd.Series(np.abs(weights_clinical), index=common_genes).nlargest(50)
top_gdsc = pd.Series(np.abs(weights_gdsc), index=common_genes).nlargest(50)
overlap = set(top_clinical.index) & set(top_gdsc.index)
print(f"\n  Top-50 gene overlap: {len(overlap)}/50")
if overlap:
    print(f"  Shared genes: {sorted(overlap)[:20]}{'...' if len(overlap)>20 else ''}")

model_b_results = {
    'gdsc_cv_aucs': [round(a, 4) for a in gdsc_aucs],
    'gdsc_cv_mean': round(float(np.mean(gdsc_aucs)), 4),
    'gdsc_cv_std': round(float(np.std(gdsc_aucs)), 4),
    'weight_correlation_rho': round(float(rho), 4),
    'weight_correlation_p': float(p_corr),
    'top50_overlap': len(overlap),
    'shared_top_genes': sorted(list(overlap))[:30],
}


# ============================================================
# Save all results
# ============================================================
print("\n" + "=" * 70)
print("Saving results")
print("=" * 70)

all_results = {
    'experiment': 'Exp8: Drug Response Prediction — Rank-Pooled Clinical Model',
    'date': '2026-02-23',
    'data': {
        'n_total_pooled': int(len(pooled)),
        'n_clinical_model_a': int(model_a_mask.sum()),
        'n_common_genes': len(common_genes),
        'n_ispy2_total': int(ispy2_mask.sum()),
        'n_ispy2_parpi': n_parpi,
        'n_ispy2_control': n_control,
        'n_gdsc_olaparib': len(gdsc_ola),
    },
    'phase2_lodo_cv': {
        'lr_with_covariates': results_lr_cov,
        'lr_no_covariates': results_lr_nocov,
        'lgb_with_covariates': results_lgb_cov,
    },
    'phase3_ispy2_interaction': interaction_results,
    'phase4_model_b_gdsc': model_b_results,
}

with open(OUT / 'exp8_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

ispy2_df.to_parquet(OUT / 'ispy2_predictions.parquet')

# Save feature weights for downstream analysis
weights_df = pd.DataFrame({
    'gene': common_genes,
    'weight_clinical': weights_clinical,
    'weight_gdsc': weights_gdsc,
    'abs_weight_clinical': np.abs(weights_clinical),
    'abs_weight_gdsc': np.abs(weights_gdsc),
})
weights_df.to_parquet(OUT / 'feature_weights.parquet', index=False)

print(f"\nAll results saved to {OUT}")
print("\nDone!")
