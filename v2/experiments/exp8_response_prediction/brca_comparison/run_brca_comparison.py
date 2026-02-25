#!/usr/bin/env python3
"""
BRCA-label vs Response-label model comparison.

This experiment directly challenges the Tempus HRD-RNA approach:
  - Tempus trains on BRCA-biallelic status (a genomic proxy for HRD)
  - We train on actual drug response labels

If the response-trained model outperforms the BRCA-trained model on drug
response endpoints, that's the key argument for our approach.

Data sources for BRCA labels:
  - TCGA-OV: 44 BRCA1/2 mutated, 191 wild-type (from cBioPortal Nature 2011 + PanCancer + Firehose)
  - GSE63885: 21 BRCA1 mutated, 54 wild-type (from GEO metadata)
  - GSE18864: only 1 BRCA1 mutated among 24 cisplatin-treated TNBC (too few, excluded)

Comparison axes:
  1. Drug response AUC on held-out datasets (LODO-CV)
  2. Survival stratification on TCGA-OV (OS + DFS)
  3. BRCA-status prediction (sanity check: BRCA model should predict BRCA well)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, mannwhitneyu
import json
import warnings
warnings.filterwarnings('ignore')

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
EXP = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT = EXP / "brca_comparison"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("BRCA-Label vs Response-Label Model Head-to-Head Comparison")
print("=" * 70)

# ============================================================
# STEP 1: Load pooled rank matrix and common genes
# ============================================================
print("\n--- Loading data ---")
pooled = pd.read_parquet(EXP / 'pooled_rank_matrix.parquet')
common_genes = [g.strip() for g in open(BASE / "common_genes_all_10_datasets.txt")]
gene_cols = common_genes
print(f"Pooled matrix: {pooled.shape[0]} samples, {len(gene_cols)} genes")

# ============================================================
# STEP 2: Build BRCA labels for TCGA-OV
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Building BRCA mutation labels")
print("=" * 70)

# Load combined BRCA mutations from cBioPortal (all three studies merged)
with open('/tmp/ov_tcga_combined_brca.json') as f:
    brca_data = json.load(f)

# Get all TCGA-OV patients in our dataset
tcga_ov = pooled[pooled['dataset'] == 'TCGA-OV'].copy()
tcga_patients = set(tcga_ov.index.astype(str))

# Build BRCA label: 1 = any BRCA1/2 mutation, 0 = wild-type
# All 235 TCGA-OV patients were in the pub study (fully sequenced)
brca_labels_tcga = {}
for pid in tcga_patients:
    if pid in brca_data:
        # Has BRCA1 or BRCA2 mutation
        brca_labels_tcga[pid] = 1
    else:
        # Sequenced but no BRCA1/2 mutation found
        brca_labels_tcga[pid] = 0

n_brca_pos = sum(v == 1 for v in brca_labels_tcga.values())
n_brca_neg = sum(v == 0 for v in brca_labels_tcga.values())
print(f"TCGA-OV BRCA labels: {n_brca_pos} mutated, {n_brca_neg} wild-type, {len(brca_labels_tcga)} total")

# Classify by mutation type for reporting
brca1_germline = set()
brca1_somatic = set()
brca2_germline = set()
brca2_somatic = set()
for pid, data in brca_data.items():
    if pid not in tcga_patients:
        continue
    for mut in data.get('BRCA1', []):
        if mut['mut_status'] == 'Germline':
            brca1_germline.add(pid)
        else:
            brca1_somatic.add(pid)
    for mut in data.get('BRCA2', []):
        if mut['mut_status'] == 'Germline':
            brca2_germline.add(pid)
        else:
            brca2_somatic.add(pid)

print(f"  BRCA1 germline: {len(brca1_germline)}, somatic: {len(brca1_somatic)}")
print(f"  BRCA2 germline: {len(brca2_germline)}, somatic: {len(brca2_somatic)}")

# Build GSE63885 BRCA labels
gse63885_meta = pd.read_parquet(BASE / 'GSE63885_metadata.parquet')
brca_labels_gse63885 = {}
for _, row in gse63885_meta.iterrows():
    sid = row['sample_id']
    brca1_status = str(row['brca1 mutation']).strip().lower()
    if brca1_status == 'no mutation':
        brca_labels_gse63885[sid] = 0
    elif brca1_status not in ['', 'nan', 'none']:
        brca_labels_gse63885[sid] = 1
    # else: unknown, skip

gse63885_pos = sum(v == 1 for v in brca_labels_gse63885.values())
gse63885_neg = sum(v == 0 for v in brca_labels_gse63885.values())
print(f"\nGSE63885 BRCA labels: {gse63885_pos} BRCA1-mutated, {gse63885_neg} wild-type, {len(brca_labels_gse63885)} total")

# Combine all BRCA labels
all_brca_labels = {}
all_brca_labels.update(brca_labels_tcga)
all_brca_labels.update(brca_labels_gse63885)
print(f"\nTotal BRCA-labeled samples: {len(all_brca_labels)} "
      f"({sum(v==1 for v in all_brca_labels.values())} pos, "
      f"{sum(v==0 for v in all_brca_labels.values())} neg)")

# ============================================================
# STEP 3: Train BRCA-label model
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Train BRCA-label model (L2 logistic, C=0.01)")
print("=" * 70)

# Get samples with BRCA labels in the pooled matrix
brca_labeled_mask = pooled.index.astype(str).isin(all_brca_labels.keys())
brca_labeled = pooled[brca_labeled_mask].copy()
brca_labeled['brca_label'] = [all_brca_labels[str(sid)] for sid in brca_labeled.index]

print(f"Samples with BRCA labels in pooled matrix: {len(brca_labeled)}")
print(f"  TCGA-OV: {sum((brca_labeled['dataset'] == 'TCGA-OV'))}")
print(f"  GSE63885: {sum((brca_labeled['dataset'] == 'GSE63885'))}")
print(f"  BRCA+: {sum(brca_labeled['brca_label'] == 1)}, BRCA-: {sum(brca_labeled['brca_label'] == 0)}")

# Train BRCA model on ALL BRCA-labeled samples
X_brca_train = np.nan_to_num(brca_labeled[gene_cols].values, nan=0.5)
y_brca_train = brca_labeled['brca_label'].values.astype(int)

clf_brca = LogisticRegression(
    penalty='l2', C=0.01, solver='lbfgs',
    max_iter=5000, class_weight='balanced', random_state=42
)
clf_brca.fit(X_brca_train, y_brca_train)

# Sanity check: predict on training data
y_brca_pred_train = clf_brca.predict_proba(X_brca_train)[:, 1]
auc_brca_train = roc_auc_score(y_brca_train, y_brca_pred_train)
print(f"\nBRCA model training AUC: {auc_brca_train:.3f}")

# Score ALL pooled samples with BRCA model
X_all_pooled = np.nan_to_num(pooled[gene_cols].values, nan=0.5)
brca_scores_all = clf_brca.predict_proba(X_all_pooled)[:, 1]
pooled['brca_model_score'] = brca_scores_all

# ============================================================
# STEP 4: Train Response-label model (same hyperparams for fairness)
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Train Response-label model (L2 logistic, C=0.01)")
print("=" * 70)

# Use the same clinical sample set as exp8 (exclude GDSC, exclude paclitaxel-only I-SPY2)
ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (pooled['category'] != 'cell_line') &
    ~((pooled['category'] == 'clinical_ispy2') & (~pooled['drug'].isin(ispy2_parpi_drugs)))
)
model_a = pooled[model_a_mask].copy()
print(f"Response model training set: {len(model_a)} samples")

X_resp_all = np.nan_to_num(model_a[gene_cols].values, nan=0.5)
y_resp_all = model_a['response_binary'].values.astype(int)
datasets_all = model_a['dataset'].values
sample_ids_all = model_a.index.values

# ============================================================
# STEP 5: LODO-CV comparison on drug response
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Head-to-Head LODO-CV on Drug Response")
print("=" * 70)

unique_datasets = sorted(model_a['dataset'].unique())

# For each held-out dataset:
#   - Response model: train on all other clinical datasets, predict held-out
#   - BRCA model: trained on all BRCA-labeled data (fixed), predict held-out
# Both use same C=0.01

lodo_results = {}
response_predictions = {}  # sample -> score
brca_model_predictions = {}  # sample -> score

for held_out in unique_datasets:
    train_mask = datasets_all != held_out
    test_mask = datasets_all == held_out

    X_train = X_resp_all[train_mask]
    y_train = y_resp_all[train_mask]
    X_test = X_resp_all[test_mask]
    y_test = y_resp_all[test_mask]
    test_ids = sample_ids_all[test_mask]

    # Skip single-class held-out sets
    if len(np.unique(y_test)) < 2:
        print(f"  {held_out}: SKIPPED (single class)")
        lodo_results[held_out] = {'skipped': True, 'reason': 'single_class'}
        continue

    # --- Response-label model (LODO) ---
    clf_resp = LogisticRegression(
        penalty='l2', C=0.01, solver='lbfgs',
        max_iter=5000, class_weight='balanced', random_state=42
    )
    clf_resp.fit(X_train, y_train)
    y_resp_prob = clf_resp.predict_proba(X_test)[:, 1]
    auc_resp = roc_auc_score(y_test, y_resp_prob)

    for sid, prob in zip(test_ids, y_resp_prob):
        response_predictions[sid] = prob

    # --- BRCA-label model (already trained on all BRCA-labeled data) ---
    # But exclude test set from BRCA training if it overlaps
    brca_test_overlap = set(str(s) for s in test_ids) & set(all_brca_labels.keys())

    if brca_test_overlap:
        # Retrain BRCA model excluding test set samples
        brca_train_mask = ~brca_labeled.index.astype(str).isin(set(str(s) for s in test_ids))
        X_brca_tr = np.nan_to_num(brca_labeled.loc[brca_train_mask, gene_cols].values, nan=0.5)
        y_brca_tr = brca_labeled.loc[brca_train_mask, 'brca_label'].values.astype(int)

        if len(np.unique(y_brca_tr)) < 2:
            # Can't train BRCA model without both classes
            y_brca_prob = np.full(len(X_test), 0.5)
            auc_brca = 0.5
        else:
            clf_brca_loo = LogisticRegression(
                penalty='l2', C=0.01, solver='lbfgs',
                max_iter=5000, class_weight='balanced', random_state=42
            )
            clf_brca_loo.fit(X_brca_tr, y_brca_tr)
            y_brca_prob = clf_brca_loo.predict_proba(X_test)[:, 1]
            auc_brca = roc_auc_score(y_test, y_brca_prob)
    else:
        # No overlap, use full BRCA model
        y_brca_prob = clf_brca.predict_proba(X_test)[:, 1]
        auc_brca = roc_auc_score(y_test, y_brca_prob)

    for sid, prob in zip(test_ids, y_brca_prob):
        brca_model_predictions[sid] = prob

    delta = auc_resp - auc_brca
    winner = "RESPONSE" if delta > 0 else "BRCA" if delta < 0 else "TIE"

    lodo_results[held_out] = {
        'auc_response_model': round(float(auc_resp), 4),
        'auc_brca_model': round(float(auc_brca), 4),
        'delta_auc': round(float(delta), 4),
        'winner': winner,
        'n_test': int(test_mask.sum()),
        'n_pos': int(y_test.sum()),
        'n_neg': int((1 - y_test).sum()),
        'brca_train_excluded_overlap': len(brca_test_overlap),
    }

    print(f"  {held_out:12s}: Response AUC={auc_resp:.3f}  |  BRCA AUC={auc_brca:.3f}  |  "
          f"Delta={delta:+.3f}  |  Winner: {winner}  (n={int(test_mask.sum())})")

# Summary
valid_results = {k: v for k, v in lodo_results.items() if not v.get('skipped', False)}
resp_aucs = [v['auc_response_model'] for v in valid_results.values()]
brca_aucs = [v['auc_brca_model'] for v in valid_results.values()]
deltas = [v['delta_auc'] for v in valid_results.values()]

resp_wins = sum(1 for d in deltas if d > 0)
brca_wins = sum(1 for d in deltas if d < 0)
ties = sum(1 for d in deltas if d == 0)

lodo_summary = {
    'mean_auc_response': round(float(np.mean(resp_aucs)), 4),
    'mean_auc_brca': round(float(np.mean(brca_aucs)), 4),
    'mean_delta': round(float(np.mean(deltas)), 4),
    'median_delta': round(float(np.median(deltas)), 4),
    'response_wins': resp_wins,
    'brca_wins': brca_wins,
    'ties': ties,
    'n_datasets': len(valid_results),
}

print(f"\n{'='*50}")
print(f"  Mean AUC (Response model): {np.mean(resp_aucs):.4f}")
print(f"  Mean AUC (BRCA model):     {np.mean(brca_aucs):.4f}")
print(f"  Mean Delta:                 {np.mean(deltas):+.4f}")
print(f"  Response wins: {resp_wins}/{len(valid_results)}, BRCA wins: {brca_wins}/{len(valid_results)}")


# ============================================================
# STEP 6: Survival comparison on TCGA-OV
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Survival Comparison on TCGA-OV")
print("=" * 70)

try:
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test

    tcga_clin = pd.read_parquet(BASE / 'tcga_ov_platinum_clinical.parquet')
    tcga_clin.index = tcga_clin.index.astype(str)

    # Get LODO-CV predictions for TCGA-OV
    tcga_mask = model_a['dataset'] == 'TCGA-OV'
    tcga_ids = model_a.index[tcga_mask].astype(str)

    # Build survival dataframe
    surv_data = []
    for sid in tcga_ids:
        if sid in tcga_clin.index and sid in response_predictions and sid in brca_model_predictions:
            row = tcga_clin.loc[sid]
            os_months = pd.to_numeric(row['OS_MONTHS'], errors='coerce')
            os_event = 1 if '1:' in str(row['OS_STATUS']) or 'DECEASED' in str(row['OS_STATUS']) else 0
            dfs_months = pd.to_numeric(row['DFS_MONTHS'], errors='coerce')
            dfs_event = 1 if '1:' in str(row['DFS_STATUS']) or 'Recurred' in str(row['DFS_STATUS']) else 0

            surv_data.append({
                'patient': sid,
                'response_score': response_predictions[sid],
                'brca_score': brca_model_predictions[sid],
                'brca_mutated': brca_labels_tcga.get(sid, -1),
                'os_months': os_months,
                'os_event': os_event,
                'dfs_months': dfs_months,
                'dfs_event': dfs_event,
                'platinum_status': row.get('PLATINUM_STATUS', 'Unknown'),
            })

    surv_df = pd.DataFrame(surv_data)
    print(f"Survival cohort: {len(surv_df)} patients")
    print(f"  BRCA mutated: {sum(surv_df['brca_mutated'] == 1)}")
    print(f"  BRCA wild-type: {sum(surv_df['brca_mutated'] == 0)}")

    survival_results = {}

    for endpoint, time_col, event_col in [
        ('OS', 'os_months', 'os_event'),
        ('DFS', 'dfs_months', 'dfs_event'),
    ]:
        valid = surv_df[surv_df[time_col].notna() & (surv_df[time_col] > 0)].copy()
        if len(valid) < 30:
            print(f"\n  {endpoint}: too few samples ({len(valid)})")
            continue

        print(f"\n--- TCGA-OV {endpoint} (n={len(valid)}, {int(valid[event_col].sum())} events) ---")
        ep_results = {}

        for model_name, score_col in [('response_model', 'response_score'), ('brca_model', 'brca_score')]:
            # Cox PH with continuous score
            cox_df = valid[[time_col, event_col, score_col]].copy()
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
            median_score = valid[score_col].median()
            high = valid[valid[score_col] >= median_score]
            low = valid[valid[score_col] < median_score]
            lr = logrank_test(high[time_col], low[time_col], high[event_col], low[event_col])

            print(f"  {model_name:17s}: HR={hr:.3f} (CI: {ci_low:.3f}-{ci_high:.3f}), "
                  f"Cox p={cox_p:.4f}, C-index={concordance:.3f}, log-rank p={lr.p_value:.4f}")

            ep_results[model_name] = {
                'cox_hr': round(hr, 4) if not np.isnan(hr) else None,
                'cox_p': round(cox_p, 6) if not np.isnan(cox_p) else None,
                'cox_ci_low': round(ci_low, 4) if not np.isnan(ci_low) else None,
                'cox_ci_high': round(ci_high, 4) if not np.isnan(ci_high) else None,
                'concordance_index': round(concordance, 4) if not np.isnan(concordance) else None,
                'logrank_p': round(float(lr.p_value), 6),
            }

        # Multivariate: both scores together
        try:
            mv_df = valid[[time_col, event_col, 'response_score', 'brca_score']].copy()
            mv_df.columns = ['time', 'event', 'response_score', 'brca_score']
            cox_mv = CoxPHFitter()
            cox_mv.fit(mv_df, duration_col='time', event_col='event')
            print(f"\n  Multivariate Cox (both scores):")
            for var in ['response_score', 'brca_score']:
                hr_mv = float(np.exp(cox_mv.params_[var]))
                p_mv = float(cox_mv.summary.loc[var, 'p'])
                print(f"    {var:20s}: HR={hr_mv:.3f}, p={p_mv:.4f}")
            ep_results['multivariate'] = {
                'response_hr': round(float(np.exp(cox_mv.params_['response_score'])), 4),
                'response_p': round(float(cox_mv.summary.loc['response_score', 'p']), 6),
                'brca_hr': round(float(np.exp(cox_mv.params_['brca_score'])), 4),
                'brca_p': round(float(cox_mv.summary.loc['brca_score', 'p']), 6),
                'concordance_index': round(float(cox_mv.concordance_index_), 4),
            }
        except Exception as e:
            print(f"  Multivariate failed: {e}")

        # Also try with actual BRCA mutation status as a binary covariate
        try:
            brca_bin_df = valid[[time_col, event_col, 'response_score', 'brca_mutated']].copy()
            brca_bin_df.columns = ['time', 'event', 'response_score', 'brca_mutated']
            brca_bin_df = brca_bin_df[brca_bin_df['brca_mutated'] >= 0]  # exclude unknown
            cox_bin = CoxPHFitter()
            cox_bin.fit(brca_bin_df, duration_col='time', event_col='event')
            print(f"\n  Cox: response_score + brca_mutated (binary):")
            for var in ['response_score', 'brca_mutated']:
                hr_b = float(np.exp(cox_bin.params_[var]))
                p_b = float(cox_bin.summary.loc[var, 'p'])
                print(f"    {var:20s}: HR={hr_b:.3f}, p={p_b:.4f}")
            ep_results['binary_brca_multivariate'] = {
                'response_hr': round(float(np.exp(cox_bin.params_['response_score'])), 4),
                'response_p': round(float(cox_bin.summary.loc['response_score', 'p']), 6),
                'brca_binary_hr': round(float(np.exp(cox_bin.params_['brca_mutated'])), 4),
                'brca_binary_p': round(float(cox_bin.summary.loc['brca_mutated', 'p']), 6),
            }
        except Exception as e:
            print(f"  Binary BRCA multivariate failed: {e}")

        survival_results[endpoint] = ep_results

except ImportError:
    print("lifelines not available, skipping survival analysis")
    survival_results = {'error': 'lifelines_not_available'}


# ============================================================
# STEP 7: BRCA prediction sanity check
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: BRCA Prediction Sanity Check")
print("=" * 70)

brca_check = {}

# On TCGA-OV: can each model predict BRCA mutation status?
tcga_with_brca = surv_df[surv_df['brca_mutated'] >= 0].copy()
print(f"\nTCGA-OV BRCA prediction (n={len(tcga_with_brca)}, "
      f"{int(tcga_with_brca['brca_mutated'].sum())} BRCA+):")

for model_name, score_col in [('response_model', 'response_score'), ('brca_model', 'brca_score')]:
    auc = roc_auc_score(tcga_with_brca['brca_mutated'], tcga_with_brca[score_col])
    # Mean score by BRCA status
    mean_pos = tcga_with_brca[tcga_with_brca['brca_mutated'] == 1][score_col].mean()
    mean_neg = tcga_with_brca[tcga_with_brca['brca_mutated'] == 0][score_col].mean()
    stat, p = mannwhitneyu(
        tcga_with_brca[tcga_with_brca['brca_mutated'] == 1][score_col],
        tcga_with_brca[tcga_with_brca['brca_mutated'] == 0][score_col],
        alternative='two-sided'
    )
    print(f"  {model_name:17s}: AUC={auc:.3f}, mean(BRCA+)={mean_pos:.3f}, "
          f"mean(BRCA-)={mean_neg:.3f}, MW p={p:.4f}")
    brca_check[f'{model_name}_brca_auc'] = round(float(auc), 4)
    brca_check[f'{model_name}_mean_brca_pos'] = round(float(mean_pos), 4)
    brca_check[f'{model_name}_mean_brca_neg'] = round(float(mean_neg), 4)
    brca_check[f'{model_name}_brca_mw_p'] = round(float(p), 6)

# On GSE63885: predict BRCA1 status
gse63885_in_model = model_a[model_a['dataset'] == 'GSE63885'].copy()
gse63885_brca = []
for sid in gse63885_in_model.index.astype(str):
    if sid in brca_labels_gse63885:
        gse63885_brca.append({
            'sample': sid,
            'brca_label': brca_labels_gse63885[sid],
            'response_score': response_predictions.get(sid, np.nan),
            'brca_score': brca_model_predictions.get(sid, np.nan),
        })
gse63885_brca_df = pd.DataFrame(gse63885_brca).dropna()

if len(gse63885_brca_df) > 10 and gse63885_brca_df['brca_label'].nunique() >= 2:
    print(f"\nGSE63885 BRCA1 prediction (n={len(gse63885_brca_df)}, "
          f"{int(gse63885_brca_df['brca_label'].sum())} BRCA1+):")
    for model_name, score_col in [('response_model', 'response_score'), ('brca_model', 'brca_score')]:
        auc = roc_auc_score(gse63885_brca_df['brca_label'], gse63885_brca_df[score_col])
        print(f"  {model_name:17s}: AUC={auc:.3f}")
        brca_check[f'gse63885_{model_name}_brca_auc'] = round(float(auc), 4)


# ============================================================
# STEP 8: Drug response prediction stratified by BRCA status
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Drug Response AUC Stratified by BRCA Status")
print("=" * 70)

# Key question: does the response model work in BRCA-WT patients too?
# (BRCA model should mainly work by identifying BRCA-mutated patients)

stratified_results = {}

# TCGA-OV stratified
tcga_resp_df = model_a[model_a['dataset'] == 'TCGA-OV'].copy()
tcga_resp_df['response_score'] = [response_predictions.get(sid, np.nan) for sid in tcga_resp_df.index]
tcga_resp_df['brca_score'] = [brca_model_predictions.get(sid, np.nan) for sid in tcga_resp_df.index]
tcga_resp_df['brca_mutated'] = [brca_labels_tcga.get(str(sid), -1) for sid in tcga_resp_df.index]

for stratum, label in [(1, 'BRCA_mutated'), (0, 'BRCA_wildtype')]:
    sub = tcga_resp_df[tcga_resp_df['brca_mutated'] == stratum].dropna(subset=['response_score', 'brca_score'])
    y = sub['response_binary'].values.astype(int)
    if len(sub) > 10 and len(np.unique(y)) >= 2:
        auc_r = roc_auc_score(y, sub['response_score'].values)
        auc_b = roc_auc_score(y, sub['brca_score'].values)
        print(f"  TCGA-OV {label} (n={len(sub)}, {int(y.sum())} responders):")
        print(f"    Response model AUC: {auc_r:.3f}")
        print(f"    BRCA model AUC:     {auc_b:.3f}")
        stratified_results[f'TCGA-OV_{label}'] = {
            'n': len(sub), 'n_pos': int(y.sum()),
            'auc_response': round(float(auc_r), 4),
            'auc_brca': round(float(auc_b), 4),
        }
    else:
        print(f"  TCGA-OV {label}: too few samples or single class (n={len(sub)}, unique classes={len(np.unique(y)) if len(sub) > 0 else 0})")
        stratified_results[f'TCGA-OV_{label}'] = {'n': len(sub), 'skipped': True}


# ============================================================
# STEP 9: Feature weight comparison
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Feature Weight Comparison")
print("=" * 70)

# Train full response model on all Model A data (same C=0.01)
clf_resp_full = LogisticRegression(
    penalty='l2', C=0.01, solver='lbfgs',
    max_iter=5000, class_weight='balanced', random_state=42
)
clf_resp_full.fit(X_resp_all, y_resp_all)

weights_resp = clf_resp_full.coef_[0]
weights_brca = clf_brca.coef_[0]

rho, p_corr = spearmanr(weights_resp, weights_brca)
print(f"Spearman correlation of feature weights: rho={rho:.3f}, p={p_corr:.2e}")

# Top genes comparison
abs_resp = np.abs(weights_resp)
abs_brca = np.abs(weights_brca)
top50_resp = set(np.array(gene_cols)[np.argsort(-abs_resp)[:50]])
top50_brca = set(np.array(gene_cols)[np.argsort(-abs_brca)[:50]])
top100_resp = set(np.array(gene_cols)[np.argsort(-abs_resp)[:100]])
top100_brca = set(np.array(gene_cols)[np.argsort(-abs_brca)[:100]])

overlap50 = top50_resp & top50_brca
overlap100 = top100_resp & top100_brca

print(f"Top-50 gene overlap:  {len(overlap50)}/50")
print(f"Top-100 gene overlap: {len(overlap100)}/100")
if overlap50:
    print(f"  Shared top-50: {sorted(overlap50)[:20]}")

# Check specific HR genes
hr_genes = ['BRCA1', 'BRCA2', 'RAD51', 'POLQ', 'PARP1', 'CHEK1', 'ATR', 'ATM', 'FANCD2', 'PALB2']
print(f"\n  Key HR gene weights:")
for gene in hr_genes:
    if gene in gene_cols:
        idx = gene_cols.index(gene)
        print(f"    {gene:10s}: response_weight={weights_resp[idx]:+.6f}, "
              f"brca_weight={weights_brca[idx]:+.6f}")

weight_comparison = {
    'spearman_rho': round(float(rho), 4),
    'spearman_p': float(p_corr),
    'top50_overlap': len(overlap50),
    'top100_overlap': len(overlap100),
    'shared_top50_genes': sorted(list(overlap50)),
}


# ============================================================
# STEP 10: Score correlation
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: Score Correlation Between Models")
print("=" * 70)

# How correlated are the two model scores?
for ds in unique_datasets:
    ds_mask = model_a['dataset'] == ds
    ds_ids = model_a.index[ds_mask].astype(str)
    r_scores = [response_predictions.get(sid, np.nan) for sid in ds_ids]
    b_scores = [brca_model_predictions.get(sid, np.nan) for sid in ds_ids]
    valid_mask = ~(np.isnan(r_scores) | np.isnan(b_scores))
    if sum(valid_mask) > 10:
        rho_ds, p_ds = spearmanr(
            np.array(r_scores)[valid_mask],
            np.array(b_scores)[valid_mask]
        )
        print(f"  {ds:12s}: Spearman rho={rho_ds:.3f}, p={p_ds:.4f} (n={sum(valid_mask)})")


# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("Saving results")
print("=" * 70)

all_results = {
    'experiment': 'BRCA-label vs Response-label Model Comparison',
    'date': '2026-02-24',
    'motivation': (
        'Tempus HRD-RNA is trained on BRCA-biallelic status as a proxy for HRD. '
        'Our model is trained on actual drug response. This comparison tests whether '
        'response-trained models outperform BRCA-trained models on drug response endpoints.'
    ),
    'brca_label_sources': {
        'TCGA-OV': {
            'source': 'cBioPortal (Nature 2011 + PanCancer Atlas + Firehose Legacy)',
            'n_mutated': n_brca_pos,
            'n_wildtype': n_brca_neg,
            'n_total': len(brca_labels_tcga),
            'brca1_germline': len(brca1_germline),
            'brca1_somatic': len(brca1_somatic),
            'brca2_germline': len(brca2_germline),
            'brca2_somatic': len(brca2_somatic),
            'note': 'Using any BRCA1/2 mutation as proxy for biallelic (LOH data not readily available)',
        },
        'GSE63885': {
            'source': 'GEO metadata (BRCA1 status only)',
            'n_mutated': gse63885_pos,
            'n_wildtype': gse63885_neg,
            'n_total': len(brca_labels_gse63885),
        },
        'total_brca_labeled': len(all_brca_labels),
    },
    'model_config': {
        'brca_model': 'L2 logistic regression, C=0.01, class_weight=balanced',
        'response_model': 'L2 logistic regression, C=0.01, class_weight=balanced (same config)',
        'features': f'{len(gene_cols)} rank-transformed genes',
        'note': 'Same hyperparameters for fair comparison',
    },
    'lodo_cv_drug_response': {
        'per_dataset': lodo_results,
        'summary': lodo_summary,
    },
    'survival_comparison': survival_results,
    'brca_prediction_sanity_check': brca_check,
    'stratified_by_brca_status': stratified_results,
    'feature_weight_comparison': weight_comparison,
}

with open(OUT / 'brca_vs_response_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults saved to {OUT / 'brca_vs_response_results.json'}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Response-Label vs BRCA-Label Model")
print("=" * 70)

print(f"\n1. DRUG RESPONSE (LODO-CV):")
print(f"   Response model mean AUC: {lodo_summary['mean_auc_response']:.4f}")
print(f"   BRCA model mean AUC:     {lodo_summary['mean_auc_brca']:.4f}")
print(f"   Response wins {resp_wins}/{len(valid_results)} datasets")

print(f"\n2. SURVIVAL (TCGA-OV):")
for ep in ['OS', 'DFS']:
    if ep in survival_results:
        r_hr = survival_results[ep].get('response_model', {}).get('cox_hr', 'N/A')
        r_p = survival_results[ep].get('response_model', {}).get('cox_p', 'N/A')
        b_hr = survival_results[ep].get('brca_model', {}).get('cox_hr', 'N/A')
        b_p = survival_results[ep].get('brca_model', {}).get('cox_p', 'N/A')
        print(f"   {ep}: Response HR={r_hr}, p={r_p} | BRCA HR={b_hr}, p={b_p}")

print(f"\n3. BRCA PREDICTION (sanity check):")
print(f"   Response model predicts BRCA: AUC={brca_check.get('response_model_brca_auc', 'N/A')}")
print(f"   BRCA model predicts BRCA:     AUC={brca_check.get('brca_model_brca_auc', 'N/A')}")

print(f"\n4. FEATURE OVERLAP:")
print(f"   Weight correlation: rho={weight_comparison['spearman_rho']}")
print(f"   Top-50 gene overlap: {weight_comparison['top50_overlap']}/50")

print("\nDone!")
