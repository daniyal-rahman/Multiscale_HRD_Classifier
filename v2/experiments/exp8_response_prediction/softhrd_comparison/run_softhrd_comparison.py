#!/usr/bin/env python3
"""
softHRD Comparison: Head-to-head evaluation of multiple methods on the same
Exp8 LODO-CV splits.

Methods compared:
  1. Exp8 full model:   L2 LogReg, C=0.01, all ~11,140 genes
  2. softHRD-features:  L2 LogReg, C=0.01, using only the 109 softHRD genes
  3. IRF1 baseline:     Single-gene predictor (IRF1 rank value)
  4. BRCA-label model:  Already computed (loaded from brca_comparison results)

For each method x dataset: AUC + bootstrap 95% CI (2000 iterations).
Paired permutation test for Exp8 vs softHRD-features.
Survival comparison on TCGA-OV (Cox PH for DFS and OS).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import json
import warnings
warnings.filterwarnings('ignore')

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
EXP = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT = EXP / "softhrd_comparison"
OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

print("=" * 70)
print("softHRD Comparison: Multi-Method LODO-CV Head-to-Head")
print("=" * 70)

# ============================================================
# STEP 1: Load data
# ============================================================
print("\n--- Loading data ---")
pooled = pd.read_parquet(EXP / 'pooled_rank_matrix.parquet')
common_genes = [g.strip() for g in open(BASE / "common_genes_all_10_datasets.txt")]
gene_cols = common_genes
print(f"Pooled matrix: {pooled.shape[0]} samples, {len(gene_cols)} genes")

# Load softHRD gene list
sig_df = pd.read_csv(
    Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/signature_analysis/all_signature_gene_lists.csv")
)
softhrd_all = sig_df[sig_df['Signature'] == 'softHRD']['Gene'].tolist()
# Intersect with available genes in pooled matrix
softhrd_genes = [g for g in softhrd_all if g in gene_cols]
print(f"softHRD genes: {len(softhrd_all)} total, {len(softhrd_genes)} available in rank matrix")
softhrd_missing = [g for g in softhrd_all if g not in gene_cols]
if softhrd_missing:
    print(f"  Missing: {softhrd_missing}")

# Load BRCA comparison results
with open(EXP / "brca_comparison" / "brca_vs_response_results.json") as f:
    brca_results = json.load(f)

# ============================================================
# STEP 2: Apply Model A filter
# ============================================================
print("\n--- Applying Model A filter ---")
ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (pooled['category'] != 'cell_line') &
    ~((pooled['category'] == 'clinical_ispy2') & (~pooled['drug'].isin(ispy2_parpi_drugs)))
)
model_a = pooled[model_a_mask].copy()
print(f"Model A samples: {len(model_a)}")

meta_cols = ['response_binary', 'response_continuous', 'drug', 'dataset', 'category', 'cancer_type']
X_all = np.nan_to_num(model_a[gene_cols].values, nan=0.5)
y_all = model_a['response_binary'].values.astype(int)
datasets_all = model_a['dataset'].values
sample_ids_all = model_a.index.values

# softHRD gene indices
softhrd_indices = [gene_cols.index(g) for g in softhrd_genes]
X_softhrd_all = X_all[:, softhrd_indices]

# IRF1 index
irf1_gene = 'IRF1'
if irf1_gene in gene_cols:
    irf1_idx = gene_cols.index(irf1_gene)
    print(f"IRF1 gene found at index {irf1_idx}")
else:
    print(f"WARNING: IRF1 not found in gene columns!")
    irf1_idx = None

unique_datasets = sorted(model_a['dataset'].unique())
print(f"Datasets for LODO-CV: {unique_datasets}")
print(f"  Total: {len(unique_datasets)} folds")

# ============================================================
# STEP 3: Bootstrap AUC function
# ============================================================
def bootstrap_auc(y_true, y_score, n_boot=2000, seed=42):
    """Compute AUC with bootstrap 95% CI."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    auc_point = roc_auc_score(y_true, y_score)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        y_t = y_true[idx]
        y_s = y_score[idx]
        if len(np.unique(y_t)) < 2:
            continue
        aucs.append(roc_auc_score(y_t, y_s))
    if len(aucs) < 100:
        return auc_point, np.nan, np.nan
    aucs = np.array(aucs)
    ci_lo = np.percentile(aucs, 2.5)
    ci_hi = np.percentile(aucs, 97.5)
    return auc_point, ci_lo, ci_hi


# ============================================================
# STEP 4: LODO-CV for all methods
# ============================================================
print("\n" + "=" * 70)
print("LODO-CV: Head-to-Head Comparison")
print("=" * 70)

results_per_dataset = {}
all_predictions = {
    'exp8_full': {},
    'softhrd_features': {},
    'irf1': {},
}

for held_out in unique_datasets:
    print(f"\n--- Held-out: {held_out} ---")
    train_mask = datasets_all != held_out
    test_mask = datasets_all == held_out

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]
    test_ids = sample_ids_all[test_mask]

    if len(np.unique(y_test)) < 2:
        print(f"  SKIPPED (single class)")
        results_per_dataset[held_out] = {'skipped': True, 'reason': 'single_class'}
        continue

    ds_result = {
        'n_test': int(test_mask.sum()),
        'n_pos': int(y_test.sum()),
        'n_neg': int((1 - y_test).sum()),
    }

    # --- 1. Exp8 full model ---
    clf_full = LogisticRegression(
        penalty='l2', C=0.01, solver='lbfgs',
        max_iter=5000, class_weight='balanced', random_state=42
    )
    clf_full.fit(X_train, y_train)
    y_full_prob = clf_full.predict_proba(X_test)[:, 1]
    auc_full, ci_lo_full, ci_hi_full = bootstrap_auc(y_test, y_full_prob)
    ds_result['exp8_full'] = {
        'auc': round(float(auc_full), 4),
        'ci_lo': round(float(ci_lo_full), 4) if not np.isnan(ci_lo_full) else None,
        'ci_hi': round(float(ci_hi_full), 4) if not np.isnan(ci_hi_full) else None,
    }
    for sid, prob in zip(test_ids, y_full_prob):
        all_predictions['exp8_full'][sid] = float(prob)

    print(f"  Exp8 full:       AUC={auc_full:.4f} [{ci_lo_full:.4f}, {ci_hi_full:.4f}]")

    # --- 2. softHRD-features model ---
    X_train_sh = X_all[train_mask][:, softhrd_indices]
    X_test_sh = X_all[test_mask][:, softhrd_indices]

    clf_sh = LogisticRegression(
        penalty='l2', C=0.01, solver='lbfgs',
        max_iter=5000, class_weight='balanced', random_state=42
    )
    clf_sh.fit(X_train_sh, y_train)
    y_sh_prob = clf_sh.predict_proba(X_test_sh)[:, 1]
    auc_sh, ci_lo_sh, ci_hi_sh = bootstrap_auc(y_test, y_sh_prob)
    ds_result['softhrd_features'] = {
        'auc': round(float(auc_sh), 4),
        'ci_lo': round(float(ci_lo_sh), 4) if not np.isnan(ci_lo_sh) else None,
        'ci_hi': round(float(ci_hi_sh), 4) if not np.isnan(ci_hi_sh) else None,
    }
    for sid, prob in zip(test_ids, y_sh_prob):
        all_predictions['softhrd_features'][sid] = float(prob)

    print(f"  softHRD-feat:    AUC={auc_sh:.4f} [{ci_lo_sh:.4f}, {ci_hi_sh:.4f}]")

    # --- 3. IRF1 baseline ---
    if irf1_idx is not None:
        irf1_scores = X_test[:, irf1_idx]
        auc_irf1, ci_lo_irf1, ci_hi_irf1 = bootstrap_auc(y_test, irf1_scores)
        ds_result['irf1'] = {
            'auc': round(float(auc_irf1), 4),
            'ci_lo': round(float(ci_lo_irf1), 4) if not np.isnan(ci_lo_irf1) else None,
            'ci_hi': round(float(ci_hi_irf1), 4) if not np.isnan(ci_hi_irf1) else None,
        }
        for sid, score in zip(test_ids, irf1_scores):
            all_predictions['irf1'][sid] = float(score)
        print(f"  IRF1 baseline:   AUC={auc_irf1:.4f} [{ci_lo_irf1:.4f}, {ci_hi_irf1:.4f}]")
    else:
        ds_result['irf1'] = {'auc': None, 'error': 'IRF1 not found'}

    # --- 4. BRCA-label model (from existing results) ---
    brca_lodo = brca_results.get('lodo_cv_drug_response', {}).get('per_dataset', {})
    if held_out in brca_lodo and not brca_lodo[held_out].get('skipped', False):
        brca_auc = brca_lodo[held_out]['auc_brca_model']
        ds_result['brca_label'] = {
            'auc': round(float(brca_auc), 4),
            'ci_lo': None,  # Not computed in original, could add but skipping
            'ci_hi': None,
        }
        print(f"  BRCA-label:      AUC={brca_auc:.4f}")
    else:
        ds_result['brca_label'] = {'auc': None, 'note': 'not available'}

    results_per_dataset[held_out] = ds_result


# ============================================================
# STEP 5: Summary statistics
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Mean AUC across datasets")
print("=" * 70)

methods = ['exp8_full', 'softhrd_features', 'irf1', 'brca_label']
method_labels = {
    'exp8_full': 'Exp8 Full (~11k genes)',
    'softhrd_features': 'softHRD-features (109 genes)',
    'irf1': 'IRF1 single-gene',
    'brca_label': 'BRCA-label model',
}

summary = {}
for method in methods:
    aucs = []
    for ds, res in results_per_dataset.items():
        if res.get('skipped'):
            continue
        m_res = res.get(method, {})
        if m_res.get('auc') is not None:
            aucs.append(m_res['auc'])
    if aucs:
        summary[method] = {
            'mean_auc': round(float(np.mean(aucs)), 4),
            'median_auc': round(float(np.median(aucs)), 4),
            'std_auc': round(float(np.std(aucs)), 4),
            'n_datasets': len(aucs),
            'per_dataset_aucs': aucs,
        }
        print(f"  {method_labels[method]:35s}: mean={np.mean(aucs):.4f}, "
              f"median={np.median(aucs):.4f}, std={np.std(aucs):.4f} (n={len(aucs)})")
    else:
        summary[method] = {'mean_auc': None, 'n_datasets': 0}


# ============================================================
# STEP 6: Paired permutation test (Exp8 full vs softHRD-features)
# ============================================================
print("\n" + "=" * 70)
print("Paired Permutation Test: Exp8 Full vs softHRD-features")
print("=" * 70)

# Gather paired predictions for all samples
paired_samples = []
for sid in all_predictions['exp8_full']:
    if sid in all_predictions['softhrd_features']:
        # Find the dataset for this sample
        mask = sample_ids_all == sid
        if mask.any():
            ds = datasets_all[mask][0]
            y_true = y_all[mask][0]
            paired_samples.append({
                'sample': sid,
                'dataset': ds,
                'y_true': int(y_true),
                'exp8_score': all_predictions['exp8_full'][sid],
                'softhrd_score': all_predictions['softhrd_features'][sid],
            })

paired_df = pd.DataFrame(paired_samples)
print(f"Paired samples: {len(paired_df)}")

# Compute observed delta AUC per dataset, then mean
def compute_mean_auc_delta(df, score_col_a='exp8_score', score_col_b='softhrd_score'):
    """Compute mean AUC(A) - AUC(B) across datasets."""
    deltas = []
    for ds in df['dataset'].unique():
        sub = df[df['dataset'] == ds]
        y = sub['y_true'].values
        if len(np.unique(y)) < 2:
            continue
        auc_a = roc_auc_score(y, sub[score_col_a].values)
        auc_b = roc_auc_score(y, sub[score_col_b].values)
        deltas.append(auc_a - auc_b)
    return np.mean(deltas) if deltas else 0.0

observed_delta = compute_mean_auc_delta(paired_df)
print(f"Observed mean AUC delta (Exp8 - softHRD): {observed_delta:+.4f}")

# Permutation test: for each permutation, randomly swap predictions per sample
n_perm = 10000
rng = np.random.RandomState(42)
perm_deltas = np.zeros(n_perm)

exp8_scores = paired_df['exp8_score'].values.copy()
softhrd_scores = paired_df['softhrd_score'].values.copy()
datasets_paired = paired_df['dataset'].values
y_true_paired = paired_df['y_true'].values

for i in range(n_perm):
    # Random swap mask
    swap = rng.randint(0, 2, len(paired_df)).astype(bool)
    perm_a = np.where(swap, softhrd_scores, exp8_scores)
    perm_b = np.where(swap, exp8_scores, softhrd_scores)

    # Compute mean delta across datasets
    deltas_perm = []
    for ds in np.unique(datasets_paired):
        ds_mask = datasets_paired == ds
        y_ds = y_true_paired[ds_mask]
        if len(np.unique(y_ds)) < 2:
            continue
        auc_a = roc_auc_score(y_ds, perm_a[ds_mask])
        auc_b = roc_auc_score(y_ds, perm_b[ds_mask])
        deltas_perm.append(auc_a - auc_b)
    perm_deltas[i] = np.mean(deltas_perm) if deltas_perm else 0.0

    if (i + 1) % 2000 == 0:
        print(f"  Permutation {i+1}/{n_perm}...")

# Two-sided p-value
p_value_perm = np.mean(np.abs(perm_deltas) >= np.abs(observed_delta))
print(f"\nPaired permutation test (10,000 permutations):")
print(f"  Observed delta: {observed_delta:+.4f}")
print(f"  Two-sided p-value: {p_value_perm:.4f}")

permutation_result = {
    'observed_delta_mean_auc': round(float(observed_delta), 4),
    'n_permutations': n_perm,
    'p_value_two_sided': round(float(p_value_perm), 4),
    'n_paired_samples': len(paired_df),
    'n_datasets_used': len(paired_df['dataset'].unique()),
}


# ============================================================
# STEP 7: Survival comparison on TCGA-OV
# ============================================================
print("\n" + "=" * 70)
print("Survival Comparison on TCGA-OV")
print("=" * 70)

survival_results = {}

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test

    tcga_clin = pd.read_parquet(BASE / 'tcga_ov_platinum_clinical.parquet')
    tcga_clin.index = tcga_clin.index.astype(str)

    # Build survival dataframe with all method scores
    tcga_mask = model_a['dataset'] == 'TCGA-OV'
    tcga_ids = model_a.index[tcga_mask].astype(str)

    surv_data = []
    for sid in tcga_ids:
        sid_str = str(sid)
        if sid_str not in tcga_clin.index:
            continue

        row = tcga_clin.loc[sid_str]
        os_months = pd.to_numeric(row['OS_MONTHS'], errors='coerce')
        os_event = 1 if '1:' in str(row['OS_STATUS']) or 'DECEASED' in str(row['OS_STATUS']) else 0
        dfs_months = pd.to_numeric(row['DFS_MONTHS'], errors='coerce')
        dfs_event = 1 if '1:' in str(row['DFS_STATUS']) or 'Recurred' in str(row['DFS_STATUS']) else 0

        entry = {
            'patient': sid_str,
            'os_months': os_months,
            'os_event': os_event,
            'dfs_months': dfs_months,
            'dfs_event': dfs_event,
        }

        # Add scores for each method
        if sid in all_predictions['exp8_full']:
            entry['exp8_score'] = all_predictions['exp8_full'][sid]
        elif sid_str in all_predictions['exp8_full']:
            entry['exp8_score'] = all_predictions['exp8_full'][sid_str]
        else:
            entry['exp8_score'] = np.nan

        if sid in all_predictions['softhrd_features']:
            entry['softhrd_score'] = all_predictions['softhrd_features'][sid]
        elif sid_str in all_predictions['softhrd_features']:
            entry['softhrd_score'] = all_predictions['softhrd_features'][sid_str]
        else:
            entry['softhrd_score'] = np.nan

        if sid in all_predictions['irf1']:
            entry['irf1_score'] = all_predictions['irf1'][sid]
        elif sid_str in all_predictions['irf1']:
            entry['irf1_score'] = all_predictions['irf1'][sid_str]
        else:
            entry['irf1_score'] = np.nan

        surv_data.append(entry)

    surv_df = pd.DataFrame(surv_data)
    print(f"Survival cohort: {len(surv_df)} patients")

    for endpoint, time_col, event_col in [
        ('DFS', 'dfs_months', 'dfs_event'),
        ('OS', 'os_months', 'os_event'),
    ]:
        valid = surv_df[surv_df[time_col].notna() & (surv_df[time_col] > 0)].copy()
        if len(valid) < 30:
            print(f"\n  {endpoint}: too few samples ({len(valid)})")
            continue

        print(f"\n--- TCGA-OV {endpoint} (n={len(valid)}, {int(valid[event_col].sum())} events) ---")
        ep_results = {}

        for model_name, score_col in [
            ('exp8_full', 'exp8_score'),
            ('softhrd_features', 'softhrd_score'),
            ('irf1', 'irf1_score'),
        ]:
            score_valid = valid[valid[score_col].notna()].copy()
            if len(score_valid) < 30 or len(np.unique(score_valid[event_col])) < 2:
                print(f"  {model_name}: insufficient data")
                ep_results[model_name] = {'error': 'insufficient_data'}
                continue

            # Cox PH with continuous score
            cox_df = score_valid[[time_col, event_col, score_col]].copy()
            cox_df.columns = ['time', 'event', 'score']
            cox = CoxPHFitter()
            try:
                cox.fit(cox_df, duration_col='time', event_col='event')
                hr = float(np.exp(cox.params_['score']))
                cox_p = float(cox.summary.loc['score', 'p'])
                ci_low = float(np.exp(cox.confidence_intervals_.iloc[0, 0]))
                ci_high = float(np.exp(cox.confidence_intervals_.iloc[0, 1]))
                concordance = float(cox.concordance_index_)
            except Exception as e:
                print(f"  {model_name} Cox failed: {e}")
                hr, cox_p, ci_low, ci_high, concordance = np.nan, np.nan, np.nan, np.nan, np.nan

            # Log-rank test (median split)
            median_score = score_valid[score_col].median()
            high = score_valid[score_valid[score_col] >= median_score]
            low = score_valid[score_valid[score_col] < median_score]

            try:
                lr = logrank_test(high[time_col], low[time_col], high[event_col], low[event_col])
                lr_p = float(lr.p_value)
            except:
                lr_p = np.nan

            print(f"  {model_name:20s}: HR={hr:.4f} (CI: {ci_low:.4f}-{ci_high:.4f}), "
                  f"Cox p={cox_p:.6f}, C-index={concordance:.4f}, log-rank p={lr_p:.6f}")

            ep_results[model_name] = {
                'cox_hr': round(hr, 4) if not np.isnan(hr) else None,
                'cox_p': round(cox_p, 6) if not np.isnan(cox_p) else None,
                'cox_ci_low': round(ci_low, 4) if not np.isnan(ci_low) else None,
                'cox_ci_high': round(ci_high, 4) if not np.isnan(ci_high) else None,
                'concordance_index': round(concordance, 4) if not np.isnan(concordance) else None,
                'logrank_p': round(lr_p, 6) if not np.isnan(lr_p) else None,
            }

        survival_results[endpoint] = ep_results

except ImportError as e:
    print(f"lifelines not available: {e}")
    survival_results = {'error': str(e)}
except Exception as e:
    print(f"Survival analysis error: {e}")
    import traceback
    traceback.print_exc()
    survival_results = {'error': str(e)}


# ============================================================
# STEP 8: Feature weight analysis (softHRD vs full model)
# ============================================================
print("\n" + "=" * 70)
print("Feature Weight Analysis")
print("=" * 70)

# Train full models on all data for weight comparison
clf_full_all = LogisticRegression(
    penalty='l2', C=0.01, solver='lbfgs',
    max_iter=5000, class_weight='balanced', random_state=42
)
clf_full_all.fit(X_all, y_all)
weights_full = clf_full_all.coef_[0]

# Get weights for softHRD genes from the full model
softhrd_weights_in_full = {softhrd_genes[i]: float(weights_full[softhrd_indices[i]])
                           for i in range(len(softhrd_genes))}

# Train softHRD-only model on all data
clf_sh_all = LogisticRegression(
    penalty='l2', C=0.01, solver='lbfgs',
    max_iter=5000, class_weight='balanced', random_state=42
)
clf_sh_all.fit(X_softhrd_all, y_all)
weights_sh = clf_sh_all.coef_[0]
softhrd_weights_restricted = {softhrd_genes[i]: float(weights_sh[i])
                               for i in range(len(softhrd_genes))}

# Correlation between full-model weights and restricted-model weights for softHRD genes
w_full_sh = np.array([softhrd_weights_in_full[g] for g in softhrd_genes])
w_restricted = np.array([softhrd_weights_restricted[g] for g in softhrd_genes])
rho_weights, p_weights = spearmanr(w_full_sh, w_restricted)
print(f"Weight correlation (full vs restricted, softHRD genes): rho={rho_weights:.3f}, p={p_weights:.2e}")

# Top softHRD genes by absolute weight in the restricted model
top_sh_genes = sorted(softhrd_weights_restricted.items(), key=lambda x: abs(x[1]), reverse=True)
print(f"\nTop 20 softHRD genes by |weight| in restricted model:")
for gene, w in top_sh_genes[:20]:
    w_full = softhrd_weights_in_full[gene]
    print(f"  {gene:15s}: restricted={w:+.6f}, full_model={w_full:+.6f}")

weight_analysis = {
    'weight_correlation_rho': round(float(rho_weights), 4),
    'weight_correlation_p': float(p_weights),
    'n_softhrd_genes_available': len(softhrd_genes),
    'top20_softhrd_by_weight': [
        {'gene': g, 'restricted_weight': round(w, 6),
         'full_model_weight': round(softhrd_weights_in_full[g], 6)}
        for g, w in top_sh_genes[:20]
    ],
}


# ============================================================
# STEP 9: Save results
# ============================================================
print("\n" + "=" * 70)
print("Saving results")
print("=" * 70)

all_results = {
    'experiment': 'softHRD Comparison: Multi-Method LODO-CV Head-to-Head',
    'date': '2026-02-24',
    'methods': {
        'exp8_full': 'L2 LogReg, C=0.01, class_weight=balanced, all 11140 genes',
        'softhrd_features': f'L2 LogReg, C=0.01, class_weight=balanced, {len(softhrd_genes)} softHRD genes',
        'irf1': 'Single-gene predictor (IRF1 rank value)',
        'brca_label': 'L2 LogReg, C=0.01, trained on BRCA mutation labels (from brca_comparison)',
    },
    'softhrd_genes': {
        'total_in_signature': len(softhrd_all),
        'available_in_rank_matrix': len(softhrd_genes),
        'missing': softhrd_missing,
        'gene_list': softhrd_genes,
    },
    'lodo_cv': {
        'per_dataset': results_per_dataset,
        'summary': summary,
    },
    'permutation_test': permutation_result,
    'survival_tcga_ov': survival_results,
    'weight_analysis': weight_analysis,
}

with open(OUT / 'softhrd_comparison_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"Results JSON saved to {OUT / 'softhrd_comparison_results.json'}")

# Save comparison table as TSV
print("\nCreating comparison table...")
rows = []
for ds in unique_datasets:
    res = results_per_dataset.get(ds, {})
    if res.get('skipped'):
        continue
    row = {
        'dataset': ds,
        'n_test': res.get('n_test', ''),
        'n_pos': res.get('n_pos', ''),
        'n_neg': res.get('n_neg', ''),
    }
    for method in methods:
        m_res = res.get(method, {})
        auc_val = m_res.get('auc', '')
        ci_lo = m_res.get('ci_lo', '')
        ci_hi = m_res.get('ci_hi', '')
        row[f'{method}_auc'] = auc_val if auc_val is not None else ''
        if ci_lo is not None and ci_hi is not None:
            row[f'{method}_ci'] = f"[{ci_lo}, {ci_hi}]"
        else:
            row[f'{method}_ci'] = ''
    rows.append(row)

# Add summary row
summary_row = {'dataset': 'MEAN', 'n_test': '', 'n_pos': '', 'n_neg': ''}
for method in methods:
    s = summary.get(method, {})
    summary_row[f'{method}_auc'] = s.get('mean_auc', '')
    summary_row[f'{method}_ci'] = ''
rows.append(summary_row)

table_df = pd.DataFrame(rows)
table_df.to_csv(OUT / 'comparison_table.tsv', sep='\t', index=False)
print(f"Comparison table saved to {OUT / 'comparison_table.tsv'}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\nMean AUC across {len(unique_datasets)} datasets:")
for method in methods:
    s = summary.get(method, {})
    if s.get('mean_auc') is not None:
        print(f"  {method_labels[method]:35s}: {s['mean_auc']:.4f} +/- {s.get('std_auc', 0):.4f}")
    else:
        print(f"  {method_labels[method]:35s}: N/A")

print(f"\nPermutation test (Exp8 full vs softHRD-features):")
print(f"  Delta AUC = {permutation_result['observed_delta_mean_auc']:+.4f}, "
      f"p = {permutation_result['p_value_two_sided']:.4f}")

if survival_results and 'error' not in survival_results:
    print(f"\nSurvival on TCGA-OV:")
    for ep in ['DFS', 'OS']:
        if ep in survival_results:
            for model_name in ['exp8_full', 'softhrd_features', 'irf1']:
                r = survival_results[ep].get(model_name, {})
                if r.get('cox_hr') is not None:
                    print(f"  {ep} {model_name:20s}: HR={r['cox_hr']:.4f}, p={r['cox_p']:.6f}, "
                          f"C-index={r['concordance_index']:.4f}")

print("\nDone!")
