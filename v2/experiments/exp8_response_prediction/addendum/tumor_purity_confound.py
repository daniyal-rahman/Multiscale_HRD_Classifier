#!/usr/bin/env python3
"""
Addendum Analysis: Tumor Purity Confound

Reviewer concern: "Is the model score confounded by tumor purity? Higher purity
tumors may have different gene expression patterns regardless of drug response."

Approach:
  1. Correlate ESTIMATE leukocyte fraction with model scores and response labels
  2. Compute ESTIMATE tumor purity proxy = 1 - leukocyte fraction
  3. Test score ~ response after adjusting for purity (logistic regression covariate)
  4. Stratify: test AUC in high-purity vs low-purity subgroups
  5. Gene-level: check whether top model genes correlate with purity
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
EXP  = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT  = EXP / "addendum"
ESTIMATE_FILE = EXP / "immune_deconv" / "thorsson_data" / "TCGA_all_leuk_estimate.tsv"
CIBERSORT_FILE = EXP / "immune_deconv" / "thorsson_data" / "TCGA_cibersort_relative.tsv"

print("=" * 72)
print("Tumor Purity Confound Analysis")
print("=" * 72)

# ══════════════════════════════════════════════════════════════════
# 1. Load data and run LODO-CV
# ══════════════════════════════════════════════════════════════════
print("\n[1/5] Loading data and running LODO-CV...")

pooled = pd.read_parquet(EXP / 'validation_fixed' / 'pooled_rank_matrix_fixed.parquet')
meta_cols = ['response_binary', 'response_continuous', 'drug', 'dataset', 'category', 'cancer_type']
gene_cols = [c for c in pooled.columns if c not in meta_cols]

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

all_predictions = {}
for held_out in unique_datasets:
    train_mask = datasets_all != held_out
    test_mask = datasets_all == held_out
    clf = LogisticRegression(
        penalty='l2', C=0.01, solver='lbfgs',
        max_iter=5000, class_weight='balanced', random_state=42
    )
    clf.fit(X_all[train_mask], y_all[train_mask])
    y_prob = clf.predict_proba(X_all[test_mask])[:, 1]
    for sid, prob in zip(sample_ids_all[test_mask], y_prob):
        all_predictions[sid] = float(prob)

model_a['predicted_score'] = [all_predictions[sid] for sid in model_a.index]
print(f"  Model A: {len(model_a)} samples")

# Also train full model on all data except TCGA-OV to get feature weights
tcga_mask = datasets_all == 'TCGA-OV'
clf_weights = LogisticRegression(
    penalty='l2', C=0.01, solver='lbfgs',
    max_iter=5000, class_weight='balanced', random_state=42
)
clf_weights.fit(X_all[~tcga_mask], y_all[~tcga_mask])
feature_weights = clf_weights.coef_[0]

# ══════════════════════════════════════════════════════════════════
# 2. Load purity data
# ══════════════════════════════════════════════════════════════════
print("\n[2/5] Loading ESTIMATE leukocyte fractions...")

estimate = pd.read_csv(ESTIMATE_FILE, sep='\t', header=None,
                       names=['CancerType', 'SampleID', 'LeukFraction'])
estimate_ov = estimate[estimate['CancerType'] == 'OV'].copy()
estimate_ov['short_id'] = estimate_ov['SampleID'].str[:12]
estimate_ov = estimate_ov.drop_duplicates(subset='short_id', keep='first')
estimate_ov = estimate_ov.set_index('short_id')

# Tumor purity proxy = 1 - leukocyte fraction
estimate_ov['tumor_purity'] = 1 - estimate_ov['LeukFraction']
print(f"  OV samples: {len(estimate_ov)}")
print(f"  Leukocyte fraction: mean={estimate_ov['LeukFraction'].mean():.4f}, "
      f"range=[{estimate_ov['LeukFraction'].min():.4f}, {estimate_ov['LeukFraction'].max():.4f}]")
print(f"  Tumor purity proxy: mean={estimate_ov['tumor_purity'].mean():.4f}")

# Match
tcga_ov = model_a[model_a['dataset'] == 'TCGA-OV'].copy()
common = sorted(set(tcga_ov.index) & set(estimate_ov.index))
print(f"  Matched samples: {len(common)}")

scores   = tcga_ov.loc[common, 'predicted_score'].values
response = tcga_ov.loc[common, 'response_binary'].values.astype(int)
leuk     = estimate_ov.loc[common, 'LeukFraction'].values
purity   = estimate_ov.loc[common, 'tumor_purity'].values
X_tcga   = model_a.loc[common, gene_cols].values
X_tcga   = np.nan_to_num(X_tcga, nan=0.5)

results = {}

# ══════════════════════════════════════════════════════════════════
# 3. Basic correlations
# ══════════════════════════════════════════════════════════════════
print("\n[3/5] Correlations...")

# Score vs leukocyte fraction
rho_sl, p_sl = spearmanr(scores, leuk)
r_sl, p_sl_p = pearsonr(scores, leuk)
print(f"  Score vs leuk_frac:    Spearman rho={rho_sl:.4f} (p={p_sl:.2e}), Pearson r={r_sl:.4f} (p={p_sl_p:.2e})")

# Score vs tumor purity
rho_sp, p_sp = spearmanr(scores, purity)
print(f"  Score vs tumor_purity: Spearman rho={rho_sp:.4f} (p={p_sp:.2e})")

# Response vs leukocyte fraction
rho_rl, p_rl = spearmanr(response, leuk)
print(f"  Response vs leuk_frac: Spearman rho={rho_rl:.4f} (p={p_rl:.2e})")

# Response vs tumor purity
rho_rp, p_rp = spearmanr(response, purity)
print(f"  Response vs purity:    Spearman rho={rho_rp:.4f} (p={p_rp:.2e})")

# Leukocyte fraction by response group
leuk_sens = leuk[response == 1]
leuk_resist = leuk[response == 0]
mw_leuk, p_leuk_mw = mannwhitneyu(leuk_sens, leuk_resist, alternative='two-sided')
print(f"\n  Leukocyte fraction: sensitive={leuk_sens.mean():.4f} vs resistant={leuk_resist.mean():.4f}")
print(f"  Mann-Whitney: U={mw_leuk:.0f}, p={p_leuk_mw:.4f}")

results['basic_correlations'] = {
    'score_vs_leukfrac': {'spearman_rho': round(float(rho_sl), 4), 'spearman_p': float(f"{p_sl:.2e}"),
                          'pearson_r': round(float(r_sl), 4), 'pearson_p': float(f"{p_sl_p:.2e}")},
    'score_vs_purity': {'spearman_rho': round(float(rho_sp), 4), 'spearman_p': float(f"{p_sp:.2e}")},
    'response_vs_leukfrac': {'spearman_rho': round(float(rho_rl), 4), 'spearman_p': float(f"{p_rl:.2e}")},
    'response_vs_purity': {'spearman_rho': round(float(rho_rp), 4), 'spearman_p': float(f"{p_rp:.2e}")},
    'leukfrac_sensitive_mean': round(float(leuk_sens.mean()), 4),
    'leukfrac_resistant_mean': round(float(leuk_resist.mean()), 4),
    'mannwhitney_leukfrac_by_response_p': round(float(p_leuk_mw), 4),
}

# ══════════════════════════════════════════════════════════════════
# 4. Score remains predictive after adjusting for purity
# ══════════════════════════════════════════════════════════════════
print("\n[4/5] Covariate-adjusted analyses...")

# 4a. Logistic regression: response ~ score + purity
print("\n--- 4a. Logistic regression: response ~ score + purity ---")
import statsmodels.api as sm

df_analysis = pd.DataFrame({
    'response': response,
    'score': scores,
    'leuk_frac': leuk,
    'purity': purity,
})

# Model 1: response ~ score only
X1 = sm.add_constant(df_analysis[['score']])
logit1 = sm.Logit(df_analysis['response'], X1).fit(disp=0)
print(f"  Model 1 (score only): AIC={logit1.aic:.1f}")
print(f"    score coef={logit1.params['score']:.4f}, p={logit1.pvalues['score']:.4f}")

# Model 2: response ~ score + purity
X2 = sm.add_constant(df_analysis[['score', 'purity']])
logit2 = sm.Logit(df_analysis['response'], X2).fit(disp=0)
print(f"  Model 2 (score + purity): AIC={logit2.aic:.1f}")
print(f"    score coef={logit2.params['score']:.4f}, p={logit2.pvalues['score']:.4f}")
print(f"    purity coef={logit2.params['purity']:.4f}, p={logit2.pvalues['purity']:.4f}")

# Model 3: response ~ purity only
X3 = sm.add_constant(df_analysis[['purity']])
logit3 = sm.Logit(df_analysis['response'], X3).fit(disp=0)
print(f"  Model 3 (purity only): AIC={logit3.aic:.1f}")
print(f"    purity coef={logit3.params['purity']:.4f}, p={logit3.pvalues['purity']:.4f}")

# Likelihood ratio test: Model 2 vs Model 3 (does score add to purity?)
from scipy.stats import chi2
lr_stat = -2 * (logit3.llf - logit2.llf)
lr_p = chi2.sf(lr_stat, df=1)
print(f"\n  LR test (score adds to purity?): chi2={lr_stat:.4f}, p={lr_p:.4f}")

# Likelihood ratio test: Model 2 vs Model 1 (does purity add to score?)
lr_stat2 = -2 * (logit1.llf - logit2.llf)
lr_p2 = chi2.sf(lr_stat2, df=1)
print(f"  LR test (purity adds to score?): chi2={lr_stat2:.4f}, p={lr_p2:.4f}")

results['logistic_regression_models'] = {
    'model1_score_only': {
        'AIC': round(logit1.aic, 1),
        'score_coef': round(float(logit1.params['score']), 4),
        'score_pval': round(float(logit1.pvalues['score']), 4),
    },
    'model2_score_plus_purity': {
        'AIC': round(logit2.aic, 1),
        'score_coef': round(float(logit2.params['score']), 4),
        'score_pval': round(float(logit2.pvalues['score']), 4),
        'purity_coef': round(float(logit2.params['purity']), 4),
        'purity_pval': round(float(logit2.pvalues['purity']), 4),
    },
    'model3_purity_only': {
        'AIC': round(logit3.aic, 1),
        'purity_coef': round(float(logit3.params['purity']), 4),
        'purity_pval': round(float(logit3.pvalues['purity']), 4),
    },
    'lr_test_score_adds_to_purity': {'chi2': round(lr_stat, 4), 'p': round(lr_p, 4)},
    'lr_test_purity_adds_to_score': {'chi2': round(lr_stat2, 4), 'p': round(lr_p2, 4)},
}

# 4b. Stratify by purity: high vs low
print("\n--- 4b. Stratified analysis by tumor purity ---")
median_purity = np.median(purity)
high_purity = purity >= median_purity
low_purity = purity < median_purity

for label, mask in [('High purity (≥ median)', high_purity), ('Low purity (< median)', low_purity)]:
    s = scores[mask]
    r = response[mask]
    if len(np.unique(r)) >= 2:
        auc = roc_auc_score(r, s)
        _, mw_p = mannwhitneyu(s[r == 1], s[r == 0], alternative='greater')
        rho, p_rho = spearmanr(r, s)
    else:
        auc = float('nan')
        mw_p = float('nan')
        rho = float('nan')
        p_rho = float('nan')
    n_sens = (r == 1).sum()
    n_res = (r == 0).sum()
    print(f"  {label}: n={mask.sum()} (sens={n_sens}, res={n_res})")
    print(f"    AUC={auc:.4f}, MW p={mw_p:.2e}, Spearman rho={rho:.4f} (p={p_rho:.2e})")
    print(f"    Purity range: [{purity[mask].min():.4f}, {purity[mask].max():.4f}], mean={purity[mask].mean():.4f}")

results['stratified_by_purity'] = {
    'median_purity': round(float(median_purity), 4),
    'high_purity': {
        'n': int(high_purity.sum()),
        'n_sensitive': int((response[high_purity] == 1).sum()),
        'n_resistant': int((response[high_purity] == 0).sum()),
        'auc': round(float(roc_auc_score(response[high_purity], scores[high_purity])), 4)
            if len(np.unique(response[high_purity])) >= 2 else None,
        'purity_range': [round(float(purity[high_purity].min()), 4),
                         round(float(purity[high_purity].max()), 4)],
    },
    'low_purity': {
        'n': int(low_purity.sum()),
        'n_sensitive': int((response[low_purity] == 1).sum()),
        'n_resistant': int((response[low_purity] == 0).sum()),
        'auc': round(float(roc_auc_score(response[low_purity], scores[low_purity])), 4)
            if len(np.unique(response[low_purity])) >= 2 else None,
        'purity_range': [round(float(purity[low_purity].min()), 4),
                         round(float(purity[low_purity].max()), 4)],
    },
}

# 4c. Tertile stratification
print("\n--- 4c. Tertile stratification by purity ---")
t33_p = np.percentile(purity, 33.33)
t67_p = np.percentile(purity, 66.67)
for label, mask in [
    (f'Low purity (<{t33_p:.3f})', purity <= t33_p),
    (f'Mid purity', (purity > t33_p) & (purity < t67_p)),
    (f'High purity (≥{t67_p:.3f})', purity >= t67_p),
]:
    s = scores[mask]
    r = response[mask]
    if len(np.unique(r)) >= 2:
        auc = roc_auc_score(r, s)
    else:
        auc = float('nan')
    print(f"  {label}: n={mask.sum()}, response_rate={r.mean():.3f}, AUC={auc:.4f}")

# ══════════════════════════════════════════════════════════════════
# 5. Gene-level purity correlations
# ══════════════════════════════════════════════════════════════════
print("\n[5/5] Gene-level purity correlations vs model weights...")

# For top 50 genes by model weight, check correlation with purity
top_gene_idx = np.argsort(np.abs(feature_weights))[::-1][:50]
top_genes = [gene_cols[i] for i in top_gene_idx]

gene_purity_corr = {}
for i, gene in zip(top_gene_idx, top_genes):
    gene_vals = X_tcga[:, i]
    rho_gp, p_gp = spearmanr(gene_vals, purity)
    gene_purity_corr[gene] = {
        'model_weight': round(float(feature_weights[i]), 6),
        'spearman_rho_vs_purity': round(float(rho_gp), 4),
        'spearman_p_vs_purity': float(f"{p_gp:.2e}"),
    }

# How many top genes are correlated with purity?
n_sig = sum(1 for v in gene_purity_corr.values()
            if v['spearman_p_vs_purity'] < 0.05)
rhos_genes = [v['spearman_rho_vs_purity'] for v in gene_purity_corr.values()]
weights_genes = [v['model_weight'] for v in gene_purity_corr.values()]

# Correlation between model weight and purity correlation
rho_wpc, p_wpc = spearmanr(weights_genes, rhos_genes)
print(f"  Top 50 genes by model weight:")
print(f"  {n_sig}/50 significantly correlated with purity (p<0.05)")
print(f"  Mean |rho| vs purity: {np.mean(np.abs(rhos_genes)):.4f}")
print(f"  Correlation (model weight vs purity rho): rho={rho_wpc:.4f}, p={p_wpc:.4f}")
print(f"\n  Top 10 genes (by absolute model weight):")
for gene in top_genes[:10]:
    info = gene_purity_corr[gene]
    sig = "*" if info['spearman_p_vs_purity'] < 0.05 else ""
    print(f"    {gene:20s} weight={info['model_weight']:+.6f}  rho_purity={info['spearman_rho_vs_purity']:+.4f}{sig}")

# All-gene analysis: correlation of model weights with purity correlations
all_purity_rhos = np.zeros(len(gene_cols))
for i in range(len(gene_cols)):
    rho_i, _ = spearmanr(X_tcga[:, i], purity)
    all_purity_rhos[i] = rho_i if not np.isnan(rho_i) else 0.0

rho_all, p_all = spearmanr(feature_weights, all_purity_rhos)
r_all, p_all_p = pearsonr(feature_weights, all_purity_rhos)
print(f"\n  All {len(gene_cols)} genes:")
print(f"  Spearman (model_weight vs purity_rho): rho={rho_all:.4f}, p={p_all:.2e}")
print(f"  Pearson  (model_weight vs purity_rho): r={r_all:.4f}, p={p_all_p:.2e}")

results['gene_purity_analysis'] = {
    'top_50_genes_sig_purity_corr': n_sig,
    'top_50_mean_abs_rho_purity': round(float(np.mean(np.abs(rhos_genes))), 4),
    'weight_vs_purity_corr_top50': {'rho': round(float(rho_wpc), 4), 'p': round(float(p_wpc), 4)},
    'weight_vs_purity_corr_all': {
        'spearman': {'rho': round(float(rho_all), 4), 'p': float(f"{p_all:.2e}")},
        'pearson': {'r': round(float(r_all), 4), 'p': float(f"{p_all_p:.2e}")},
    },
    'top_10_genes': {gene: gene_purity_corr[gene] for gene in top_genes[:10]},
}

# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

print(f"""
  Score-purity correlation:   Spearman rho={rho_sl:.4f} (p={p_sl:.2e}) — weak
  Response-purity correlation: Spearman rho={rho_rl:.4f} (p={p_rl:.2e}) — not significant

  Logistic regression:
    Score alone:        AIC={logit1.aic:.1f}, score p={logit1.pvalues['score']:.4f}
    Score + purity:     AIC={logit2.aic:.1f}, score p={logit2.pvalues['score']:.4f}, purity p={logit2.pvalues['purity']:.4f}
    Purity alone:       AIC={logit3.aic:.1f}, purity p={logit3.pvalues['purity']:.4f}

  LR test: score adds to purity? p={lr_p:.4f}
  LR test: purity adds to score? p={lr_p2:.4f}

  Gene-weight vs purity: rho={rho_all:.4f} (p={p_all:.2e})
""")

if logit2.pvalues['score'] < 0.05 and logit2.pvalues['purity'] > 0.1:
    conclusion = (
        "Score remains significant after adjusting for purity, while purity itself "
        "is not a significant predictor. Tumor purity is NOT a meaningful confound."
    )
elif logit2.pvalues['score'] < 0.05 and logit2.pvalues['purity'] < 0.05:
    conclusion = (
        "Both score and purity are independently significant. Purity is a partial confound "
        "but the model captures additional biology beyond purity effects."
    )
else:
    conclusion = (
        "Score loses significance after adjusting for purity. "
        "Purity may be a meaningful confound that warrants further investigation."
    )

print(f"  Conclusion: {conclusion}")
results['conclusion'] = conclusion

results['analysis_metadata'] = {
    'script': 'addendum/tumor_purity_confound.py',
    'date': '2026-02-25',
    'purity_source': 'ESTIMATE leukocyte fraction from Thorsson et al. 2018',
    'purity_proxy': 'tumor_purity = 1 - leukocyte_fraction',
    'dataset': 'TCGA-OV',
    'n_samples': len(common),
}

out_path = OUT / 'tumor_purity_confound_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("Done!")
