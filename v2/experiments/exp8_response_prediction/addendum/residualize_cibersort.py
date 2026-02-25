#!/usr/bin/env python3
"""
Addendum Analysis: Residualize LODO-CV Model Scores Against CIBERSORT Fractions

Reviewer concern: "Are the model's predictions simply recapitulating immune
cell composition (which correlates with platinum response)?"

Approach:
  1. Run LODO-CV to get predicted scores for all Model A samples
  2. For TCGA-OV: regress scores on 22 CIBERSORT LM22 cell-type fractions
  3. Extract residuals (the part of scores NOT explained by immune composition)
  4. Test whether residuals still discriminate sensitive vs resistant
  5. Compare original vs residualized AUC, Mann-Whitney, logistic regression

If residuals retain predictive value → model captures biology beyond immune infiltrate.
If residuals lose predictive value → model is primarily an immune proxy.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, mannwhitneyu, pearsonr
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────
BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
EXP  = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT  = EXP / "addendum"
OUT.mkdir(parents=True, exist_ok=True)

CIBERSORT_FILE = EXP / "immune_deconv" / "thorsson_data" / "TCGA_cibersort_relative.tsv"
ESTIMATE_FILE  = EXP / "immune_deconv" / "thorsson_data" / "TCGA_all_leuk_estimate.tsv"

print("=" * 72)
print("Residualization: LODO-CV Scores ~ CIBERSORT 22 Cell-Type Fractions")
print("=" * 72)

# ══════════════════════════════════════════════════════════════════
# 1. Load pooled data and run LODO-CV (matching run_exp8.py exactly)
# ══════════════════════════════════════════════════════════════════
print("\n[1/6] Loading pooled rank matrix and running LODO-CV...")

pooled = pd.read_parquet(EXP / 'validation_fixed' / 'pooled_rank_matrix_fixed.parquet')
meta_cols = ['response_binary', 'response_continuous', 'drug', 'dataset', 'category', 'cancer_type']
gene_cols = [c for c in pooled.columns if c not in meta_cols]

# Model A filter (same as run_exp8.py and run_cell_type_deconv.py)
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
print(f"  Model A: {len(model_a)} samples, {len(gene_cols)} genes, {len(unique_datasets)} datasets")

# LODO-CV with C=0.01 (matching the immune deconv analysis)
all_predictions = {}
for held_out in unique_datasets:
    train_mask = datasets_all != held_out
    test_mask = datasets_all == held_out
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test = X_all[test_mask]

    clf = LogisticRegression(
        penalty='l2', C=0.01, solver='lbfgs',
        max_iter=5000, class_weight='balanced', random_state=42
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    test_ids = sample_ids_all[test_mask]
    for sid, prob in zip(test_ids, y_prob):
        all_predictions[sid] = float(prob)
    print(f"    {held_out}: n={test_mask.sum()}, score range=[{y_prob.min():.3f}, {y_prob.max():.3f}]")

model_a['predicted_score'] = [all_predictions[sid] for sid in model_a.index]

# ══════════════════════════════════════════════════════════════════
# 2. Load CIBERSORT fractions for TCGA-OV
# ══════════════════════════════════════════════════════════════════
print("\n[2/6] Loading CIBERSORT LM22 fractions...")

cibersort = pd.read_csv(CIBERSORT_FILE, sep='\t')
cibersort_ov = cibersort[cibersort['CancerType'] == 'OV'].copy()

# Convert sample IDs: TCGA.XX.XXXX.01A... -> TCGA-XX-XXXX
cibersort_ov['short_id'] = (
    cibersort_ov['SampleID']
    .str.replace('.', '-', regex=False)
    .str[:12]
)
# Deduplicate (keep highest correlation aliquot)
cibersort_ov = cibersort_ov.sort_values('Correlation', ascending=False)
cibersort_ov = cibersort_ov.drop_duplicates(subset='short_id', keep='first')
cibersort_ov = cibersort_ov.set_index('short_id')

cell_type_cols = [c for c in cibersort_ov.columns
                  if c not in ['SampleID', 'CancerType', 'P.value', 'Correlation', 'RMSE']]
print(f"  OV samples in CIBERSORT: {len(cibersort_ov)}, cell types: {len(cell_type_cols)}")

# ══════════════════════════════════════════════════════════════════
# 3. Load ESTIMATE purity for TCGA-OV
# ══════════════════════════════════════════════════════════════════
print("\n[3/6] Loading ESTIMATE/leukocyte fraction data...")

estimate = pd.read_csv(ESTIMATE_FILE, sep='\t', header=None, names=['CancerType', 'SampleID', 'LeukFraction'])
estimate_ov = estimate[estimate['CancerType'] == 'OV'].copy()
# Truncate to patient barcode (TCGA-XX-XXXX)
estimate_ov['short_id'] = estimate_ov['SampleID'].str[:12]
estimate_ov = estimate_ov.drop_duplicates(subset='short_id', keep='first')
estimate_ov = estimate_ov.set_index('short_id')
print(f"  OV samples in ESTIMATE: {len(estimate_ov)}")

# ══════════════════════════════════════════════════════════════════
# 4. Match samples: model predictions ∩ CIBERSORT ∩ ESTIMATE
# ══════════════════════════════════════════════════════════════════
print("\n[4/6] Matching samples across data sources...")

tcga_ov = model_a[model_a['dataset'] == 'TCGA-OV'].copy()
common = sorted(
    set(tcga_ov.index) & set(cibersort_ov.index) & set(estimate_ov.index)
)
print(f"  TCGA-OV in model:     {len(tcga_ov)}")
print(f"  TCGA-OV in CIBERSORT: {len(cibersort_ov)}")
print(f"  TCGA-OV in ESTIMATE:  {len(estimate_ov)}")
print(f"  Three-way overlap:    {len(common)}")

# Also get CIBERSORT-only overlap (for main residualization, not gated on ESTIMATE)
common_cib = sorted(set(tcga_ov.index) & set(cibersort_ov.index))
print(f"  Model ∩ CIBERSORT:    {len(common_cib)}")

scores      = tcga_ov.loc[common_cib, 'predicted_score'].values
response    = tcga_ov.loc[common_cib, 'response_binary'].values.astype(int)
fractions   = cibersort_ov.loc[common_cib, cell_type_cols].values  # n x 22
frac_df     = cibersort_ov.loc[common_cib, cell_type_cols]

# ESTIMATE subset (3-way overlap)
scores_3way   = tcga_ov.loc[common, 'predicted_score'].values
response_3way = tcga_ov.loc[common, 'response_binary'].values.astype(int)
frac_3way     = cibersort_ov.loc[common, cell_type_cols].values
leuk_frac     = estimate_ov.loc[common, 'LeukFraction'].values

# ══════════════════════════════════════════════════════════════════
# 5. Residualization analyses
# ══════════════════════════════════════════════════════════════════
print("\n[5/6] Running residualization analyses...")
results = {}

# ----- 5a. OLS: score ~ 22 CIBERSORT fractions -----
print("\n--- 5a. OLS regression: score ~ 22 CIBERSORT fractions ---")
reg = LinearRegression()
reg.fit(fractions, scores)
predicted_by_immune = reg.predict(fractions)
residuals = scores - predicted_by_immune
r2 = reg.score(fractions, scores)
print(f"  R² (score ~ 22 cell types): {r2:.4f}")
print(f"  → {r2*100:.1f}% of score variance explained by immune composition")

# Test residuals vs response
auc_original = roc_auc_score(response, scores)
auc_residual = roc_auc_score(response, residuals)
print(f"  AUC original scores:     {auc_original:.4f}")
print(f"  AUC residualized scores: {auc_residual:.4f}")
print(f"  AUC retention:           {auc_residual/auc_original*100:.1f}%")

# Mann-Whitney on residuals
sens_mask = response == 1
resist_mask = response == 0
mw_orig_stat, mw_orig_p = mannwhitneyu(
    scores[sens_mask], scores[resist_mask], alternative='greater'
)
mw_resid_stat, mw_resid_p = mannwhitneyu(
    residuals[sens_mask], residuals[resist_mask], alternative='greater'
)
print(f"  MW original:     U={mw_orig_stat:.0f}, p={mw_orig_p:.2e}")
print(f"  MW residualized: U={mw_resid_stat:.0f}, p={mw_resid_p:.2e}")

# Spearman: residuals vs response
rho_orig, p_orig = spearmanr(response, scores)
rho_resid, p_resid = spearmanr(response, residuals)
print(f"  Spearman (response ~ score):    rho={rho_orig:.4f}, p={p_orig:.2e}")
print(f"  Spearman (response ~ residual): rho={rho_resid:.4f}, p={p_resid:.2e}")

# Mean score separation
print(f"  Mean original score — sensitive:  {scores[sens_mask].mean():.4f}")
print(f"  Mean original score — resistant:  {scores[resist_mask].mean():.4f}")
print(f"  Mean residual — sensitive:        {residuals[sens_mask].mean():.4f}")
print(f"  Mean residual — resistant:        {residuals[resist_mask].mean():.4f}")

results['cibersort_22_residualization'] = {
    'n_samples': len(common_cib),
    'n_sensitive': int(sens_mask.sum()),
    'n_resistant': int(resist_mask.sum()),
    'n_cell_types': len(cell_type_cols),
    'R2_score_from_immune': round(r2, 4),
    'pct_variance_explained': round(r2 * 100, 1),
    'auc_original': round(auc_original, 4),
    'auc_residualized': round(auc_residual, 4),
    'auc_retention_pct': round(auc_residual / auc_original * 100, 1),
    'mannwhitney_original_p': float(f"{mw_orig_p:.2e}"),
    'mannwhitney_residual_p': float(f"{mw_resid_p:.2e}"),
    'spearman_original': {'rho': round(float(rho_orig), 4), 'p': float(f"{p_orig:.2e}")},
    'spearman_residual': {'rho': round(float(rho_resid), 4), 'p': float(f"{p_resid:.2e}")},
    'mean_score_sensitive': round(float(scores[sens_mask].mean()), 4),
    'mean_score_resistant': round(float(scores[resist_mask].mean()), 4),
    'mean_residual_sensitive': round(float(residuals[sens_mask].mean()), 4),
    'mean_residual_resistant': round(float(residuals[resist_mask].mean()), 4),
}

# ----- 5b. CIBERSORT coefficient details -----
print("\n--- 5b. CIBERSORT regression coefficients ---")
coef_df = pd.DataFrame({
    'cell_type': cell_type_cols,
    'coefficient': reg.coef_,
    'abs_coef': np.abs(reg.coef_),
}).sort_values('abs_coef', ascending=False)
print(coef_df.to_string(index=False))

results['cibersort_regression_coefficients'] = {
    row['cell_type']: round(row['coefficient'], 6)
    for _, row in coef_df.iterrows()
}
results['cibersort_regression_intercept'] = round(float(reg.intercept_), 6)

# ----- 5c. Logistic regression: response ~ residual (single predictor) -----
print("\n--- 5c. Logistic regression: response ~ residualized score ---")
from sklearn.metrics import log_loss

# Original score as sole predictor
lr_orig = LogisticRegression(solver='lbfgs', max_iter=5000)
lr_orig.fit(scores.reshape(-1, 1), response)
auc_lr_orig = roc_auc_score(response, lr_orig.predict_proba(scores.reshape(-1, 1))[:, 1])

# Residualized score as sole predictor
lr_resid = LogisticRegression(solver='lbfgs', max_iter=5000)
lr_resid.fit(residuals.reshape(-1, 1), response)
auc_lr_resid = roc_auc_score(response, lr_resid.predict_proba(residuals.reshape(-1, 1))[:, 1])

print(f"  Logistic AUC (original score):     {auc_lr_orig:.4f}")
print(f"  Logistic AUC (residualized score): {auc_lr_resid:.4f}")

results['logistic_regression_single_predictor'] = {
    'auc_original_score': round(auc_lr_orig, 4),
    'auc_residualized_score': round(auc_lr_resid, 4),
}

# ----- 5d. Add leukocyte fraction / tumor purity as covariate -----
print("\n--- 5d. Tumor purity confound (ESTIMATE leukocyte fraction) ---")
print(f"  Leukocyte fraction range: [{np.nanmin(leuk_frac):.4f}, {np.nanmax(leuk_frac):.4f}]")
print(f"  Leukocyte fraction mean:  {np.nanmean(leuk_frac):.4f}")

# Correlation: score vs leukocyte fraction
valid_leuk = ~np.isnan(leuk_frac)
rho_leuk, p_leuk = spearmanr(scores_3way[valid_leuk], leuk_frac[valid_leuk])
print(f"  Spearman (score ~ leuk_frac):    rho={rho_leuk:.4f}, p={p_leuk:.2e}")

# Correlation: response vs leukocyte fraction
rho_leuk_resp, p_leuk_resp = spearmanr(response_3way[valid_leuk], leuk_frac[valid_leuk])
print(f"  Spearman (response ~ leuk_frac): rho={rho_leuk_resp:.4f}, p={p_leuk_resp:.2e}")

# OLS: score ~ 22 CIBERSORT + leukocyte fraction
X_extended = np.column_stack([frac_3way, leuk_frac])
valid_mask = ~np.isnan(X_extended).any(axis=1)
X_ext_valid = X_extended[valid_mask]
scores_3valid = scores_3way[valid_mask]
resp_3valid = response_3way[valid_mask]

reg_ext = LinearRegression()
reg_ext.fit(X_ext_valid, scores_3valid)
r2_ext = reg_ext.score(X_ext_valid, scores_3valid)
residuals_ext = scores_3valid - reg_ext.predict(X_ext_valid)

auc_resid_ext = roc_auc_score(resp_3valid, residuals_ext)
auc_orig_3way = roc_auc_score(resp_3valid, scores_3valid)

print(f"  R² (score ~ 22 cell types + leuk_frac): {r2_ext:.4f}")
print(f"  AUC original (3-way):     {auc_orig_3way:.4f}")
print(f"  AUC residualized (ext):   {auc_resid_ext:.4f}")
print(f"  AUC retention:            {auc_resid_ext/auc_orig_3way*100:.1f}%")

mw_ext_stat, mw_ext_p = mannwhitneyu(
    residuals_ext[resp_3valid == 1], residuals_ext[resp_3valid == 0], alternative='greater'
)
print(f"  MW residualized (ext): U={mw_ext_stat:.0f}, p={mw_ext_p:.2e}")

results['purity_confound'] = {
    'n_samples_3way': int(valid_mask.sum()),
    'leuk_frac_range': [round(float(np.nanmin(leuk_frac)), 4), round(float(np.nanmax(leuk_frac)), 4)],
    'leuk_frac_mean': round(float(np.nanmean(leuk_frac)), 4),
    'spearman_score_vs_leukfrac': {'rho': round(float(rho_leuk), 4), 'p': float(f"{p_leuk:.2e}")},
    'spearman_response_vs_leukfrac': {'rho': round(float(rho_leuk_resp), 4), 'p': float(f"{p_leuk_resp:.2e}")},
    'R2_score_from_immune_plus_purity': round(r2_ext, 4),
    'auc_original_3way': round(auc_orig_3way, 4),
    'auc_residualized_extended': round(auc_resid_ext, 4),
    'auc_retention_pct': round(auc_resid_ext / auc_orig_3way * 100, 1),
    'mannwhitney_residual_extended_p': float(f"{mw_ext_p:.2e}"),
}

# ----- 5e. Partial correlation: score vs response, controlling for immune -----
print("\n--- 5e. Partial correlation analysis ---")
# Residualize both score AND response against immune, then correlate residuals
reg_score = LinearRegression().fit(fractions, scores)
score_resid = scores - reg_score.predict(fractions)

# For response (binary), use logistic regression residuals (deviance residuals)
from sklearn.linear_model import LogisticRegression as LR
lr_immune = LR(solver='lbfgs', max_iter=5000, C=1e6)  # high C = no regularization
lr_immune.fit(fractions, response)
response_pred_prob = lr_immune.predict_proba(fractions)[:, 1]
# Pearson residuals for logistic: (y - p) / sqrt(p*(1-p))
response_resid = (response - response_pred_prob) / np.sqrt(response_pred_prob * (1 - response_pred_prob) + 1e-10)

rho_partial, p_partial = spearmanr(score_resid, response_resid)
r_partial, p_partial_pearson = pearsonr(score_resid, response_resid)
print(f"  Partial Spearman (score ⊥ immune, response ⊥ immune): rho={rho_partial:.4f}, p={p_partial:.2e}")
print(f"  Partial Pearson:                                       r={r_partial:.4f},   p={p_partial_pearson:.2e}")

results['partial_correlation'] = {
    'method': 'Residualize both score and response against 22 CIBERSORT fractions, then correlate',
    'partial_spearman': {'rho': round(float(rho_partial), 4), 'p': float(f"{p_partial:.2e}")},
    'partial_pearson': {'r': round(float(r_partial), 4), 'p': float(f"{p_partial_pearson:.2e}")},
}

# ----- 5f. Permutation test: are residualized AUCs above chance? -----
print("\n--- 5f. Permutation test (1000 permutations) ---")
np.random.seed(42)
n_perm = 1000
perm_aucs_orig = np.zeros(n_perm)
perm_aucs_resid = np.zeros(n_perm)

for i in range(n_perm):
    perm_idx = np.random.permutation(len(response))
    resp_perm = response[perm_idx]
    perm_aucs_orig[i] = roc_auc_score(resp_perm, scores)
    perm_aucs_resid[i] = roc_auc_score(resp_perm, residuals)

p_perm_orig = (perm_aucs_orig >= auc_original).mean()
p_perm_resid = (perm_aucs_resid >= auc_residual).mean()
print(f"  Permutation p-value (original AUC ≥ {auc_original:.4f}):     p={p_perm_orig:.4f}")
print(f"  Permutation p-value (residual AUC ≥ {auc_residual:.4f}):     p={p_perm_resid:.4f}")

results['permutation_test'] = {
    'n_permutations': n_perm,
    'perm_p_original': round(float(p_perm_orig), 4),
    'perm_p_residualized': round(float(p_perm_resid), 4),
    'perm_null_auc_mean': round(float(perm_aucs_resid.mean()), 4),
    'perm_null_auc_std': round(float(perm_aucs_resid.std()), 4),
}

# ----- 5g. Tertile analysis on residualized scores -----
print("\n--- 5g. Tertile analysis on residualized scores ---")
t33_r = np.percentile(residuals, 33.33)
t67_r = np.percentile(residuals, 66.67)
bottom_r = residuals <= t33_r
top_r = residuals >= t67_r

resp_rate_bottom_r = response[bottom_r].mean()
resp_rate_top_r = response[top_r].mean()

print(f"  Residual tertile thresholds: [{t33_r:.4f}, {t67_r:.4f}]")
print(f"  Bottom tertile: n={bottom_r.sum()}, response rate={resp_rate_bottom_r:.3f}")
print(f"  Top tertile:    n={top_r.sum()}, response rate={resp_rate_top_r:.3f}")
print(f"  Fold enrichment (top/bottom): {resp_rate_top_r / max(resp_rate_bottom_r, 0.001):.2f}x")

if bottom_r.sum() > 0 and top_r.sum() > 0:
    _, mw_tert_p = mannwhitneyu(
        response[top_r], response[bottom_r], alternative='greater'
    )
    print(f"  MW tertile test: p={mw_tert_p:.2e}")
else:
    mw_tert_p = float('nan')

results['tertile_analysis_residualized'] = {
    'threshold_33': round(float(t33_r), 4),
    'threshold_67': round(float(t67_r), 4),
    'n_bottom': int(bottom_r.sum()),
    'n_top': int(top_r.sum()),
    'response_rate_bottom': round(float(resp_rate_bottom_r), 4),
    'response_rate_top': round(float(resp_rate_top_r), 4),
    'fold_enrichment': round(float(resp_rate_top_r / max(resp_rate_bottom_r, 0.001)), 2),
    'mannwhitney_p': float(f"{mw_tert_p:.2e}") if not np.isnan(mw_tert_p) else None,
}

# Compare with original tertiles
t33_o = np.percentile(scores, 33.33)
t67_o = np.percentile(scores, 66.67)
resp_rate_bottom_o = response[scores <= t33_o].mean()
resp_rate_top_o = response[scores >= t67_o].mean()
results['tertile_analysis_original'] = {
    'response_rate_bottom': round(float(resp_rate_bottom_o), 4),
    'response_rate_top': round(float(resp_rate_top_o), 4),
    'fold_enrichment': round(float(resp_rate_top_o / max(resp_rate_bottom_o, 0.001)), 2),
}

# ══════════════════════════════════════════════════════════════════
# 6. Summary and save
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

r2_pct = results['cibersort_22_residualization']['pct_variance_explained']
auc_o = results['cibersort_22_residualization']['auc_original']
auc_r = results['cibersort_22_residualization']['auc_residualized']
ret = results['cibersort_22_residualization']['auc_retention_pct']
mw_o = results['cibersort_22_residualization']['mannwhitney_original_p']
mw_r = results['cibersort_22_residualization']['mannwhitney_residual_p']

print(f"\n  Immune composition explains {r2_pct}% of score variance (R²={r2_pct/100:.4f})")
print(f"  Original AUC:     {auc_o:.4f}   (MW p={mw_o:.2e})")
print(f"  Residualized AUC: {auc_r:.4f}   (MW p={mw_r:.2e})")
print(f"  AUC retention:    {ret:.1f}%")

if auc_r > 0.55 and mw_r < 0.05:
    conclusion = (
        f"POSITIVE: After removing immune composition signal, residualized scores "
        f"still discriminate sensitive vs resistant (AUC={auc_r:.3f}, p={mw_r:.2e}). "
        f"The model captures biology beyond immune infiltrate."
    )
elif auc_r > 0.55:
    conclusion = (
        f"MODERATE: Residualized AUC ({auc_r:.3f}) suggests some non-immune signal, "
        f"but the effect is not statistically significant (MW p={mw_r:.2e})."
    )
else:
    conclusion = (
        f"NEGATIVE: After residualization, scores no longer discriminate well "
        f"(AUC={auc_r:.3f}). The model may primarily capture immune composition."
    )

print(f"\n  Conclusion: {conclusion}")

results['conclusion'] = conclusion
results['analysis_metadata'] = {
    'script': 'addendum/residualize_cibersort.py',
    'date': '2026-02-25',
    'model': 'L2 LogReg C=0.01, class_weight=balanced, LODO-CV',
    'residualization_method': 'OLS linear regression of predicted scores on 22 CIBERSORT LM22 relative fractions',
    'dataset': 'TCGA-OV',
}

# Save results
out_path = OUT / 'residualize_cibersort_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("Done!")
