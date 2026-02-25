#!/usr/bin/env python3
"""
Task 4: Bootstrap CIs + Permutation Tests + Calibration Curves
PhD-auditor statistical validation of Exp8 results.

Part A: Bootstrap 95% CIs on LODO-CV AUCs (2000 iterations)
Part B: Permutation test for I-SPY2 treatment interaction (5000 iterations)
Part C: Calibration curves and Brier scores
"""

import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import statsmodels.formula.api as smf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
OUT = EXP / "validation"
OUT.mkdir(parents=True, exist_ok=True)

# Load data
print("=" * 70, flush=True)
print("Loading data...", flush=True)
pooled = pd.read_parquet(EXP / 'pooled_rank_matrix.parquet')

with open(BASE / "common_genes_all_10_datasets.txt") as f:
    common_genes = [g.strip() for g in f]

meta_cols = ['response_binary', 'response_continuous', 'drug', 'dataset', 'category', 'cancer_type']
gene_cols = [c for c in pooled.columns if c not in meta_cols]
print(f"Pooled: {pooled.shape}, genes: {len(gene_cols)}", flush=True)

# Define Model A samples (same as original Exp8)
ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (pooled['category'] != 'cell_line') &
    ~(
        (pooled['category'] == 'clinical_ispy2') &
        (~pooled['drug'].isin(ispy2_parpi_drugs))
    )
)
model_a = pooled[model_a_mask].copy()

X_all = model_a[gene_cols].values
y_all = model_a['response_binary'].values.astype(int)
datasets_all = model_a['dataset'].values

unique_datasets = sorted(model_a['dataset'].unique())
dataset_to_idx = {d: i for i, d in enumerate(unique_datasets)}
print(f"Model A: {len(model_a)} samples, {len(unique_datasets)} datasets", flush=True)

# =====================================================================
# PART A: Bootstrap 95% CIs on LODO-CV AUCs
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("PART A: Bootstrap 95% CIs on LODO-CV AUCs", flush=True)
print("=" * 70, flush=True)

N_BOOTSTRAP = 2000
np.random.seed(42)

bootstrap_results = {}

for held_out in unique_datasets:
    print(f"\n--- {held_out} ---", flush=True)

    train_mask = datasets_all != held_out
    test_mask = datasets_all == held_out

    X_train = np.nan_to_num(X_all[train_mask], nan=0.5)
    y_train = y_all[train_mask]
    X_test = np.nan_to_num(X_all[test_mask], nan=0.5)
    y_test = y_all[test_mask]

    n_test = len(y_test)
    n_pos = int(y_test.sum())
    n_neg = n_test - n_pos

    if len(np.unique(y_test)) < 2:
        print(f"  SKIPPED: single class in test set", flush=True)
        bootstrap_results[held_out] = {'skipped': True, 'reason': 'single_class'}
        continue

    # Train model (L2 LR, no dataset covariates, matching original)
    clf = LogisticRegression(
        penalty='l2', C=1.0, solver='lbfgs',
        max_iter=5000, class_weight='balanced', random_state=42
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Point estimate
    point_auc = roc_auc_score(y_test, y_prob)
    print(f"  Point AUC: {point_auc:.4f} (n={n_test}, +={n_pos}, -={n_neg})", flush=True)

    # Bootstrap CIs
    boot_aucs = []
    for b in range(N_BOOTSTRAP):
        idx = np.random.randint(0, n_test, n_test)
        y_boot = y_test[idx]
        p_boot = y_prob[idx]
        # Must have both classes
        if len(np.unique(y_boot)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_boot, p_boot))

    boot_aucs = np.array(boot_aucs)
    ci_lo = np.percentile(boot_aucs, 2.5)
    ci_hi = np.percentile(boot_aucs, 97.5)
    includes_random = ci_lo <= 0.5

    print(f"  Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}] ({len(boot_aucs)} valid iterations)", flush=True)
    print(f"  CI includes 0.5 (random): {'YES — CANNOT REJECT RANDOM' if includes_random else 'NO — significantly above random'}", flush=True)

    # DeLong-style standard error (for comparison)
    se_delong = np.std(boot_aucs)

    bootstrap_results[held_out] = {
        'point_auc': round(float(point_auc), 4),
        'ci_lower': round(float(ci_lo), 4),
        'ci_upper': round(float(ci_hi), 4),
        'ci_width': round(float(ci_hi - ci_lo), 4),
        'bootstrap_mean': round(float(boot_aucs.mean()), 4),
        'bootstrap_se': round(float(se_delong), 4),
        'includes_random': bool(includes_random),
        'n_test': int(n_test),
        'n_pos': int(n_pos),
        'n_neg': int(n_neg),
        'n_valid_bootstrap': len(boot_aucs),
    }

# Summary
print(f"\n--- BOOTSTRAP SUMMARY ---", flush=True)
cannot_reject = []
can_reject = []
for ds, res in bootstrap_results.items():
    if isinstance(res, dict) and 'skipped' not in res:
        status = "RANDOM" if res['includes_random'] else "SIGNIFICANT"
        print(f"  {ds:15s}: AUC={res['point_auc']:.3f} [{res['ci_lower']:.3f}, {res['ci_upper']:.3f}] → {status}", flush=True)
        if res['includes_random']:
            cannot_reject.append(ds)
        else:
            can_reject.append(ds)

print(f"\n  Cannot reject random ({len(cannot_reject)}): {cannot_reject}", flush=True)
print(f"  Significantly above random ({len(can_reject)}): {can_reject}", flush=True)

# Compute mean AUC and CI excluding GSE32062
non_gse32062 = {k: v for k, v in bootstrap_results.items()
                if k != 'GSE32062' and isinstance(v, dict) and 'point_auc' in v}
if non_gse32062:
    aucs_excl = [v['point_auc'] for v in non_gse32062.values()]
    print(f"\n  Mean AUC (excluding GSE32062): {np.mean(aucs_excl):.4f}", flush=True)
    print(f"  Median AUC (excluding GSE32062): {np.median(aucs_excl):.4f}", flush=True)

bootstrap_results['_summary'] = {
    'n_bootstrap': N_BOOTSTRAP,
    'cannot_reject_random': cannot_reject,
    'significantly_above_random': can_reject,
    'mean_auc_all': round(float(np.mean([v['point_auc'] for v in bootstrap_results.values()
                                          if isinstance(v, dict) and 'point_auc' in v])), 4),
    'mean_auc_excl_gse32062': round(float(np.mean(aucs_excl)), 4) if non_gse32062 else None,
}

# Save
with open(OUT / 'bootstrap_cis.json', 'w') as f:
    json.dump(bootstrap_results, f, indent=2)
print(f"\nSaved bootstrap_cis.json", flush=True)

# Bootstrap CI forest plot
fig, ax = plt.subplots(figsize=(10, 6))
ds_sorted = sorted([d for d in bootstrap_results if not d.startswith('_') and isinstance(bootstrap_results[d], dict) and 'point_auc' in bootstrap_results[d]],
                    key=lambda x: bootstrap_results[x]['point_auc'])

y_positions = range(len(ds_sorted))
for i, ds in enumerate(ds_sorted):
    r = bootstrap_results[ds]
    color = 'red' if r['includes_random'] else 'steelblue'
    ax.errorbar(r['point_auc'], i,
                xerr=[[r['point_auc'] - r['ci_lower']], [r['ci_upper'] - r['point_auc']]],
                fmt='o', color=color, capsize=5, linewidth=2, markersize=8)
    ax.text(r['ci_upper'] + 0.02, i,
            f"n={r['n_test']} (+={r['n_pos']}/-={r['n_neg']})",
            va='center', fontsize=8)

ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Random (AUC=0.5)')
ax.set_yticks(y_positions)
ax.set_yticklabels(ds_sorted)
ax.set_xlabel('AUC (with 95% bootstrap CI)')
ax.set_title(f'LODO-CV AUCs with Bootstrap 95% CIs (n={N_BOOTSTRAP})')
ax.legend()
ax.set_xlim([0.2, 1.0])
fig.tight_layout()
fig.savefig(OUT / 'bootstrap_forest_plot.png', dpi=150)
plt.close(fig)
print(f"Saved bootstrap_forest_plot.png", flush=True)

# =====================================================================
# PART B: Permutation Tests
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("PART B: Permutation Tests", flush=True)
print("=" * 70, flush=True)

N_PERM_INTERACTION = 5000
N_PERM_LODO = 1000

# --- B1: Permutation test for I-SPY2 interaction ---
print("\n--- B1: I-SPY2 Interaction Permutation Test ---", flush=True)

# Train on non-I-SPY2 clinical data
non_ispy2_mask = (pooled['category'] != 'cell_line') & (pooled['category'] != 'clinical_ispy2')
ispy2_mask = pooled['category'] == 'clinical_ispy2'

X_train_int = np.nan_to_num(pooled.loc[non_ispy2_mask, gene_cols].values, nan=0.5)
y_train_int = pooled.loc[non_ispy2_mask, 'response_binary'].values.astype(int)

X_ispy2 = np.nan_to_num(pooled.loc[ispy2_mask, gene_cols].values, nan=0.5)
y_ispy2 = pooled.loc[ispy2_mask, 'response_binary'].values.astype(int)
drug_ispy2 = pooled.loc[ispy2_mask, 'drug'].values
dataset_ispy2 = pooled.loc[ispy2_mask, 'dataset'].values

lr_int = LogisticRegression(
    penalty='l2', C=1.0, solver='lbfgs',
    max_iter=5000, class_weight='balanced', random_state=42
)
lr_int.fit(X_train_int, y_train_int)
scores_ispy2 = lr_int.predict_proba(X_ispy2)[:, 1]

ispy2_df = pd.DataFrame({
    'score': scores_ispy2,
    'pCR': y_ispy2,
    'is_parpi_arm': [1 if d in ispy2_parpi_drugs else 0 for d in drug_ispy2],
    'drug': drug_ispy2,
    'dataset': dataset_ispy2,
})

# Observed interaction coefficient (ALL I-SPY2)
try:
    obs_model = smf.logit('pCR ~ score * is_parpi_arm', data=ispy2_df).fit(disp=0)
    obs_interaction_all = float(obs_model.params.get('score:is_parpi_arm', np.nan))
    obs_pval_all = float(obs_model.pvalues.get('score:is_parpi_arm', np.nan))
    print(f"Observed interaction (all I-SPY2): coef={obs_interaction_all:.4f}, p={obs_pval_all:.4f}", flush=True)
except Exception as e:
    print(f"Observed model failed: {e}", flush=True)
    obs_interaction_all = np.nan
    obs_pval_all = np.nan

# GSE194040-only
gse194040_df = ispy2_df[ispy2_df['dataset'] == 'GSE194040']
try:
    obs_model_194 = smf.logit('pCR ~ score * is_parpi_arm', data=gse194040_df).fit(disp=0)
    obs_interaction_194 = float(obs_model_194.params.get('score:is_parpi_arm', np.nan))
    obs_pval_194 = float(obs_model_194.pvalues.get('score:is_parpi_arm', np.nan))
    print(f"Observed interaction (GSE194040): coef={obs_interaction_194:.4f}, p={obs_pval_194:.4f}", flush=True)
except Exception as e:
    print(f"GSE194040 model failed: {e}", flush=True)
    obs_interaction_194 = np.nan
    obs_pval_194 = np.nan

# Permutation: shuffle is_parpi_arm labels
print(f"\nRunning {N_PERM_INTERACTION} permutations...", flush=True)
np.random.seed(123)
perm_coefs_all = []
perm_coefs_194 = []

arm_labels_all = ispy2_df['is_parpi_arm'].values.copy()
arm_labels_194 = gse194040_df['is_parpi_arm'].values.copy()

for i in range(N_PERM_INTERACTION):
    if (i + 1) % 1000 == 0:
        print(f"  Permutation {i+1}/{N_PERM_INTERACTION}...", flush=True)

    # Shuffle arm labels for all I-SPY2
    shuffled_all = np.random.permutation(arm_labels_all)
    perm_df_all = ispy2_df.copy()
    perm_df_all['is_parpi_arm'] = shuffled_all

    try:
        m = smf.logit('pCR ~ score * is_parpi_arm', data=perm_df_all).fit(disp=0, maxiter=100)
        coef = float(m.params.get('score:is_parpi_arm', np.nan))
        if not np.isnan(coef) and abs(coef) < 100:  # sanity check
            perm_coefs_all.append(coef)
    except Exception:
        pass

    # GSE194040 only
    shuffled_194 = np.random.permutation(arm_labels_194)
    perm_df_194 = gse194040_df.copy()
    perm_df_194['is_parpi_arm'] = shuffled_194

    try:
        m194 = smf.logit('pCR ~ score * is_parpi_arm', data=perm_df_194).fit(disp=0, maxiter=100)
        coef194 = float(m194.params.get('score:is_parpi_arm', np.nan))
        if not np.isnan(coef194) and abs(coef194) < 100:
            perm_coefs_194.append(coef194)
    except Exception:
        pass

perm_coefs_all = np.array(perm_coefs_all)
perm_coefs_194 = np.array(perm_coefs_194)

# Permutation p-values (one-sided: observed >= permuted)
perm_p_all = np.mean(perm_coefs_all >= obs_interaction_all) if len(perm_coefs_all) > 0 else np.nan
perm_p_194 = np.mean(perm_coefs_194 >= obs_interaction_194) if len(perm_coefs_194) > 0 else np.nan

print(f"\nPermutation results:", flush=True)
print(f"  ALL I-SPY2: observed={obs_interaction_all:.4f}, "
      f"permutation p={perm_p_all:.4f} ({len(perm_coefs_all)} valid permutations)", flush=True)
print(f"  GSE194040:  observed={obs_interaction_194:.4f}, "
      f"permutation p={perm_p_194:.4f} ({len(perm_coefs_194)} valid permutations)", flush=True)

# Also compute two-sided p
perm_p_all_2sided = np.mean(np.abs(perm_coefs_all) >= abs(obs_interaction_all)) if len(perm_coefs_all) > 0 else np.nan
perm_p_194_2sided = np.mean(np.abs(perm_coefs_194) >= abs(obs_interaction_194)) if len(perm_coefs_194) > 0 else np.nan
print(f"  ALL I-SPY2 (two-sided): p={perm_p_all_2sided:.4f}", flush=True)
print(f"  GSE194040 (two-sided):  p={perm_p_194_2sided:.4f}", flush=True)

# Plot permutation distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, coefs, obs, label, pp in [
    (axes[0], perm_coefs_all, obs_interaction_all, 'All I-SPY2', perm_p_all),
    (axes[1], perm_coefs_194, obs_interaction_194, 'GSE194040', perm_p_194),
]:
    if len(coefs) > 0:
        ax.hist(coefs, bins=50, edgecolor='black', alpha=0.7, density=True)
        ax.axvline(obs, color='red', linewidth=2, linestyle='--',
                   label=f'Observed: {obs:.3f}\np={pp:.4f}')
        ax.set_xlabel('Interaction coefficient (score:is_parpi_arm)')
        ax.set_ylabel('Density')
        ax.set_title(f'Permutation test: {label} (n={len(coefs)})')
        ax.legend()

fig.tight_layout()
fig.savefig(OUT / 'permutation_interaction.png', dpi=150)
plt.close(fig)
print(f"Saved permutation_interaction.png", flush=True)

interaction_perm_results = {
    'n_permutations': N_PERM_INTERACTION,
    'all_ispy2': {
        'observed_coef': round(float(obs_interaction_all), 4) if not np.isnan(obs_interaction_all) else None,
        'observed_pval_parametric': round(float(obs_pval_all), 6) if not np.isnan(obs_pval_all) else None,
        'permutation_p_onesided': round(float(perm_p_all), 4) if not np.isnan(perm_p_all) else None,
        'permutation_p_twosided': round(float(perm_p_all_2sided), 4) if not np.isnan(perm_p_all_2sided) else None,
        'null_mean': round(float(perm_coefs_all.mean()), 4) if len(perm_coefs_all) > 0 else None,
        'null_std': round(float(perm_coefs_all.std()), 4) if len(perm_coefs_all) > 0 else None,
        'n_valid_perms': len(perm_coefs_all),
    },
    'gse194040': {
        'observed_coef': round(float(obs_interaction_194), 4) if not np.isnan(obs_interaction_194) else None,
        'observed_pval_parametric': round(float(obs_pval_194), 6) if not np.isnan(obs_pval_194) else None,
        'permutation_p_onesided': round(float(perm_p_194), 4) if not np.isnan(perm_p_194) else None,
        'permutation_p_twosided': round(float(perm_p_194_2sided), 4) if not np.isnan(perm_p_194_2sided) else None,
        'null_mean': round(float(perm_coefs_194.mean()), 4) if len(perm_coefs_194) > 0 else None,
        'null_std': round(float(perm_coefs_194.std()), 4) if len(perm_coefs_194) > 0 else None,
        'n_valid_perms': len(perm_coefs_194),
    },
}

# --- B2: Permutation test for overall LODO-CV ---
print("\n--- B2: Overall LODO-CV Permutation Test ---", flush=True)
print(f"Running {N_PERM_LODO} permutations of response labels...", flush=True)

# Observed mean AUC (no covariates)
obs_aucs = []
for held_out in unique_datasets:
    train_mask = datasets_all != held_out
    test_mask = datasets_all == held_out
    X_tr = np.nan_to_num(X_all[train_mask], nan=0.5)
    y_tr = y_all[train_mask]
    X_te = np.nan_to_num(X_all[test_mask], nan=0.5)
    y_te = y_all[test_mask]
    if len(np.unique(y_te)) < 2:
        continue
    clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                             max_iter=5000, class_weight='balanced', random_state=42)
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    obs_aucs.append(roc_auc_score(y_te, prob))

obs_mean_auc = np.mean(obs_aucs)
print(f"Observed mean LODO-CV AUC: {obs_mean_auc:.4f}", flush=True)

np.random.seed(456)
perm_mean_aucs = []
for i in range(N_PERM_LODO):
    if (i + 1) % 200 == 0:
        print(f"  Permutation {i+1}/{N_PERM_LODO}...", flush=True)

    # Permute response labels WITHIN each dataset (preserves class ratios per dataset)
    y_perm = y_all.copy()
    for ds in unique_datasets:
        ds_mask = datasets_all == ds
        ds_indices = np.where(ds_mask)[0]
        y_perm[ds_indices] = np.random.permutation(y_perm[ds_indices])

    perm_aucs = []
    for held_out in unique_datasets:
        train_mask = datasets_all != held_out
        test_mask = datasets_all == held_out
        X_tr = np.nan_to_num(X_all[train_mask], nan=0.5)
        y_tr = y_perm[train_mask]
        X_te = np.nan_to_num(X_all[test_mask], nan=0.5)
        y_te = y_perm[test_mask]
        if len(np.unique(y_te)) < 2:
            continue
        try:
            clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                                     max_iter=5000, class_weight='balanced', random_state=42)
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            perm_aucs.append(roc_auc_score(y_te, prob))
        except Exception:
            pass

    if perm_aucs:
        perm_mean_aucs.append(np.mean(perm_aucs))

perm_mean_aucs = np.array(perm_mean_aucs)
perm_p_lodo = np.mean(perm_mean_aucs >= obs_mean_auc)
print(f"\nOverall LODO-CV permutation test:", flush=True)
print(f"  Observed mean AUC: {obs_mean_auc:.4f}", flush=True)
print(f"  Null mean AUC: {perm_mean_aucs.mean():.4f} ± {perm_mean_aucs.std():.4f}", flush=True)
print(f"  Permutation p-value: {perm_p_lodo:.4f}", flush=True)

# Also compute excluding GSE32062
obs_aucs_excl = []
for i, held_out in enumerate(unique_datasets):
    if held_out == 'GSE32062':
        continue
    if i < len(obs_aucs):
        # Get from bootstrap_results
        if held_out in bootstrap_results and 'point_auc' in bootstrap_results[held_out]:
            obs_aucs_excl.append(bootstrap_results[held_out]['point_auc'])

obs_mean_auc_excl = np.mean(obs_aucs_excl) if obs_aucs_excl else None
print(f"  Mean AUC (excl GSE32062): {obs_mean_auc_excl:.4f}" if obs_mean_auc_excl else "  (excl GSE32062: N/A)", flush=True)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(perm_mean_aucs, bins=40, edgecolor='black', alpha=0.7, density=True)
ax.axvline(obs_mean_auc, color='red', linewidth=2, linestyle='--',
           label=f'Observed: {obs_mean_auc:.4f}\np={perm_p_lodo:.4f}')
ax.set_xlabel('Mean LODO-CV AUC')
ax.set_ylabel('Density')
ax.set_title(f'Permutation test: overall LODO-CV (n={N_PERM_LODO})')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'permutation_lodo_overall.png', dpi=150)
plt.close(fig)

lodo_perm_results = {
    'n_permutations': N_PERM_LODO,
    'observed_mean_auc': round(float(obs_mean_auc), 4),
    'null_mean_auc': round(float(perm_mean_aucs.mean()), 4),
    'null_std_auc': round(float(perm_mean_aucs.std()), 4),
    'permutation_p': round(float(perm_p_lodo), 4),
    'observed_mean_auc_excl_gse32062': round(float(obs_mean_auc_excl), 4) if obs_mean_auc_excl else None,
}

# Save all permutation results
perm_all_results = {
    'interaction_permutation': interaction_perm_results,
    'lodo_cv_permutation': lodo_perm_results,
}
with open(OUT / 'permutation_interaction.json', 'w') as f:
    json.dump(perm_all_results, f, indent=2)
print(f"\nSaved permutation_interaction.json", flush=True)

# =====================================================================
# PART C: Calibration Curves
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("PART C: Calibration Curves", flush=True)
print("=" * 70, flush=True)

calibration_results = {}

fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes_flat = axes.flatten()

for idx, held_out in enumerate(unique_datasets):
    print(f"\n--- {held_out} ---", flush=True)

    train_mask = datasets_all != held_out
    test_mask = datasets_all == held_out

    X_tr = np.nan_to_num(X_all[train_mask], nan=0.5)
    y_tr = y_all[train_mask]
    X_te = np.nan_to_num(X_all[test_mask], nan=0.5)
    y_te = y_all[test_mask]

    if len(np.unique(y_te)) < 2:
        calibration_results[held_out] = {'skipped': True}
        continue

    clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                             max_iter=5000, class_weight='balanced', random_state=42)
    clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_te)[:, 1]

    # Brier score
    brier = brier_score_loss(y_te, y_prob)

    # Calibration curve (10 bins)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means_pred = []
    bin_means_true = []
    bin_counts = []

    for b in range(n_bins):
        mask = (y_prob >= bin_edges[b]) & (y_prob < bin_edges[b+1])
        if b == n_bins - 1:
            mask = (y_prob >= bin_edges[b]) & (y_prob <= bin_edges[b+1])
        if mask.sum() > 0:
            bin_means_pred.append(float(y_prob[mask].mean()))
            bin_means_true.append(float(y_te[mask].mean()))
            bin_counts.append(int(mask.sum()))
        else:
            bin_means_pred.append(float((bin_edges[b] + bin_edges[b+1]) / 2))
            bin_means_true.append(np.nan)
            bin_counts.append(0)

    # ECE (Expected Calibration Error)
    total = sum(bin_counts)
    ece = 0.0
    for bp, bt, bc in zip(bin_means_pred, bin_means_true, bin_counts):
        if bc > 0 and not np.isnan(bt):
            ece += bc / total * abs(bt - bp)

    # Prediction spread
    pred_range = float(y_prob.max() - y_prob.min())
    pred_std = float(y_prob.std())

    print(f"  Brier: {brier:.4f}, ECE: {ece:.4f}, Pred range: {pred_range:.4f}, Pred std: {pred_std:.4f}", flush=True)

    calibration_results[held_out] = {
        'brier_score': round(float(brier), 4),
        'ece': round(float(ece), 4),
        'pred_range': round(pred_range, 4),
        'pred_std': round(pred_std, 4),
        'pred_mean': round(float(y_prob.mean()), 4),
        'pred_min': round(float(y_prob.min()), 4),
        'pred_max': round(float(y_prob.max()), 4),
        'bin_means_pred': [round(x, 4) for x in bin_means_pred],
        'bin_means_true': [round(x, 4) if not np.isnan(x) else None for x in bin_means_true],
        'bin_counts': bin_counts,
        'n_test': len(y_te),
        'n_pos': int(y_te.sum()),
    }

    # Plot
    if idx < len(axes_flat):
        ax = axes_flat[idx]
        # Calibration curve
        valid_bins = [(bp, bt, bc) for bp, bt, bc in zip(bin_means_pred, bin_means_true, bin_counts) if bc > 0 and not np.isnan(bt)]
        if valid_bins:
            bps, bts, bcs = zip(*valid_bins)
            # Scale marker size by count
            sizes = [max(20, min(200, c * 5)) for c in bcs]
            ax.scatter(bps, bts, s=sizes, c='steelblue', alpha=0.7, edgecolors='black', zorder=3)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Observed proportion')
        ax.set_title(f'{held_out}\nBrier={brier:.3f}, ECE={ece:.3f}, Range={pred_range:.3f}')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8)

# Remove unused axes
for idx in range(len(unique_datasets), len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle('Calibration Curves (LODO-CV, LR no covariates)', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'calibration_curves.png', dpi=150)
plt.close(fig)
print(f"\nSaved calibration_curves.png", flush=True)

# Summary
print(f"\n--- CALIBRATION SUMMARY ---", flush=True)
poor_discrimination = []
for ds, res in calibration_results.items():
    if 'skipped' in res:
        continue
    narrow = res['pred_range'] < 0.1
    status = "POOR DISCRIMINATION" if narrow else "ok"
    if narrow:
        poor_discrimination.append(ds)
    print(f"  {ds:15s}: Brier={res['brier_score']:.3f}, ECE={res['ece']:.3f}, "
          f"range={res['pred_range']:.3f} → {status}", flush=True)

calibration_results['_summary'] = {
    'poor_discrimination_datasets': poor_discrimination,
    'mean_brier': round(float(np.mean([v['brier_score'] for v in calibration_results.values()
                                        if isinstance(v, dict) and 'brier_score' in v])), 4),
    'mean_ece': round(float(np.mean([v['ece'] for v in calibration_results.values()
                                      if isinstance(v, dict) and 'ece' in v])), 4),
}

with open(OUT / 'calibration_results.json', 'w') as f:
    json.dump(calibration_results, f, indent=2)
print(f"Saved calibration_results.json", flush=True)

# =====================================================================
# FINAL SUMMARY
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("FINAL SUMMARY", flush=True)
print("=" * 70, flush=True)

print(f"\nPart A — Bootstrap CIs:", flush=True)
print(f"  Datasets where CI includes 0.5 (random): {cannot_reject}", flush=True)
print(f"  Datasets significantly above random: {can_reject}", flush=True)

print(f"\nPart B — Permutation tests:", flush=True)
print(f"  I-SPY2 interaction (all): observed={obs_interaction_all:.4f}, permutation p={perm_p_all:.4f}", flush=True)
print(f"  I-SPY2 interaction (GSE194040): observed={obs_interaction_194:.4f}, permutation p={perm_p_194:.4f}", flush=True)
print(f"  Overall LODO-CV: observed={obs_mean_auc:.4f}, permutation p={perm_p_lodo:.4f}", flush=True)

print(f"\nPart C — Calibration:", flush=True)
print(f"  Poor discrimination datasets: {poor_discrimination}", flush=True)
print(f"  Mean Brier score: {calibration_results['_summary']['mean_brier']:.4f}", flush=True)

print(f"\nAll results saved to {OUT}", flush=True)
print("DONE.", flush=True)
