#!/usr/bin/env python3
"""
Exp8b: Survival analysis + signature overlap additions to Exp8.

1. Survival analysis on LODO-CV predictions:
   - TCGA-OV: OS + DFS from cBioPortal
   - GSE32062: PFS from GEO metadata
   - GSE30161: OS + PFI from GEO metadata
   For each: KM split by median score, log-rank test, Cox proportional hazards

2. Top gene extraction and signature overlap:
   - Top 100 genes by absolute L2 weight from full clinical model
   - Compare with consensus-93, DD-500, all published HRD signatures

3. I-SPY2 survival interaction: not possible (pCR only, no EFS/DRFS)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import spearmanr
import json
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
EXP = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
SIG = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/signature_analysis")

# Load pooled matrix from Exp8
pooled = pd.read_parquet(EXP / 'pooled_rank_matrix.parquet')
common_genes = [g.strip() for g in open(BASE / "common_genes_all_10_datasets.txt")]

print("=" * 70)
print("Exp8b: Survival Analysis + Signature Overlap")
print("=" * 70)
print(f"Loaded pooled matrix: {pooled.shape[0]} samples, {len(common_genes)} genes")

# ============================================================
# Re-run LODO-CV to get per-sample predictions for survival datasets
# ============================================================
print("\n" + "=" * 70)
print("Generating LODO-CV predictions for survival-annotated datasets")
print("=" * 70)

ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (pooled['category'] != 'cell_line') &
    ~((pooled['category'] == 'clinical_ispy2') & (~pooled['drug'].isin(ispy2_parpi_drugs)))
)
model_a = pooled[model_a_mask].copy()

gene_cols = common_genes
X_all = model_a[gene_cols].values
y_all = model_a['response_binary'].values.astype(int)
datasets_all = model_a['dataset'].values
sample_ids_all = model_a.index.values

# Run LODO-CV, save predictions for each sample
all_predictions = {}
unique_datasets = sorted(model_a['dataset'].unique())

for held_out in unique_datasets:
    train_mask = datasets_all != held_out
    test_mask = datasets_all == held_out

    X_train = np.nan_to_num(X_all[train_mask], nan=0.5)
    X_test = np.nan_to_num(X_all[test_mask], nan=0.5)
    y_train = y_all[train_mask]

    clf = LogisticRegression(
        penalty='l2', C=1.0, solver='lbfgs',
        max_iter=5000, class_weight='balanced', random_state=42
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]

    test_ids = sample_ids_all[test_mask]
    for sid, prob in zip(test_ids, y_prob):
        all_predictions[sid] = prob

print(f"Generated predictions for {len(all_predictions)} samples")

# ============================================================
# PART 1: Survival Analysis
# ============================================================
print("\n" + "=" * 70)
print("PART 1: Survival Analysis on LODO-CV Predictions")
print("=" * 70)

survival_results = {}


def run_survival_analysis(name, sample_ids, scores, times, events, time_unit='months'):
    """Run KM + log-rank + Cox on a dataset."""
    print(f"\n--- {name} (n={len(scores)}, {int(sum(events))} events, {time_unit}) ---")

    df = pd.DataFrame({
        'score': scores,
        'time': times,
        'event': events.astype(int),
    })

    # Remove rows with missing data
    df = df.dropna()
    print(f"  After dropping NaN: n={len(df)}")

    if len(df) < 20:
        print(f"  SKIPPED: too few samples")
        return {'error': 'too_few_samples', 'n': len(df)}

    # Median split
    median_score = df['score'].median()
    df['high_score'] = (df['score'] >= median_score).astype(int)

    high = df[df['high_score'] == 1]
    low = df[df['high_score'] == 0]

    print(f"  High-score group: n={len(high)}, events={int(high['event'].sum())}, "
          f"median time={high['time'].median():.1f}")
    print(f"  Low-score group:  n={len(low)}, events={int(low['event'].sum())}, "
          f"median time={low['time'].median():.1f}")

    # Log-rank test
    lr = logrank_test(high['time'], low['time'], high['event'], low['event'])
    print(f"  Log-rank p={lr.p_value:.4f}")

    # Cox proportional hazards with continuous score
    cox_df = df[['time', 'event', 'score']].copy()
    cox = CoxPHFitter()
    try:
        cox.fit(cox_df, duration_col='time', event_col='event')
        hr = np.exp(cox.params_['score'])
        cox_p = cox.summary.loc['score', 'p']
        ci_low = np.exp(cox.confidence_intervals_.iloc[0, 0])
        ci_high = np.exp(cox.confidence_intervals_.iloc[0, 1])
        print(f"  Cox HR={hr:.3f} (95% CI: {ci_low:.3f}-{ci_high:.3f}), p={cox_p:.4f}")
    except Exception as e:
        print(f"  Cox failed: {e}")
        hr, cox_p, ci_low, ci_high = np.nan, np.nan, np.nan, np.nan

    # Spearman correlation between score and survival time
    rho, p_corr = spearmanr(df['score'], df['time'])
    print(f"  Score-survival Spearman rho={rho:.3f}, p={p_corr:.4f}")

    # KM median survival by group
    kmf = KaplanMeierFitter()
    medians = {}
    for label, group in [('high', high), ('low', low)]:
        kmf.fit(group['time'], group['event'], label=label)
        medians[label] = kmf.median_survival_time_

    result = {
        'n': len(df),
        'n_events': int(df['event'].sum()),
        'logrank_p': round(float(lr.p_value), 6),
        'cox_hr': round(float(hr), 4),
        'cox_p': round(float(cox_p), 6) if not np.isnan(cox_p) else None,
        'cox_ci_low': round(float(ci_low), 4) if not np.isnan(ci_low) else None,
        'cox_ci_high': round(float(ci_high), 4) if not np.isnan(ci_high) else None,
        'spearman_rho': round(float(rho), 4),
        'spearman_p': round(float(p_corr), 6),
        'km_median_high': float(medians['high']) if not np.isinf(medians['high']) else None,
        'km_median_low': float(medians['low']) if not np.isinf(medians['low']) else None,
    }
    return result


# --- TCGA-OV: OS and DFS from cBioPortal ---
print("\n### TCGA-OV ###")
tcga_clin = pd.read_parquet(BASE / 'tcga_ov_platinum_clinical.parquet')
# Index by patientId
tcga_clin.index = tcga_clin.index.astype(str)

# Map LODO-CV predictions to TCGA-OV patients
tcga_ov_mask = model_a['dataset'] == 'TCGA-OV'
tcga_sample_ids = model_a.index[tcga_ov_mask].astype(str)
tcga_scores = np.array([all_predictions[sid] for sid in tcga_sample_ids])

# Match clinical data
matched_ids = [sid for sid in tcga_sample_ids if sid in tcga_clin.index]
print(f"TCGA-OV: {len(tcga_sample_ids)} with predictions, {len(matched_ids)} matched to clinical")

if len(matched_ids) > 50:
    matched_clin = tcga_clin.loc[matched_ids]
    matched_scores = np.array([all_predictions[sid] for sid in matched_ids])

    # OS analysis
    os_months = pd.to_numeric(matched_clin['OS_MONTHS'], errors='coerce')
    os_event = matched_clin['OS_STATUS'].apply(lambda x: 1 if '1:' in str(x) or 'DECEASED' in str(x) else 0)
    valid_os = ~os_months.isna()
    if valid_os.sum() > 50:
        survival_results['TCGA-OV_OS'] = run_survival_analysis(
            'TCGA-OV Overall Survival',
            matched_ids, matched_scores[valid_os.values],
            os_months[valid_os].values, os_event[valid_os].values
        )

    # DFS analysis
    dfs_months = pd.to_numeric(matched_clin['DFS_MONTHS'], errors='coerce')
    dfs_event = matched_clin['DFS_STATUS'].apply(lambda x: 1 if '1:' in str(x) or 'Recurred' in str(x) else 0)
    valid_dfs = ~dfs_months.isna()
    if valid_dfs.sum() > 50:
        survival_results['TCGA-OV_DFS'] = run_survival_analysis(
            'TCGA-OV Disease-Free Survival',
            matched_ids, matched_scores[valid_dfs.values],
            dfs_months[valid_dfs].values, dfs_event[valid_dfs].values
        )


# --- GSE32062: PFS from GEO metadata ---
print("\n### GSE32062 ###")
try:
    import GEOparse
    gse32062 = GEOparse.get_GEO('GSE32062', destdir='/tmp/geo_cache', silent=True)
    gsm_meta = {}
    for gsm_name, gsm in gse32062.gsms.items():
        meta = gsm.metadata
        pfs = None
        rec = None
        os_m = None
        death = None
        for key, vals in meta.items():
            val = vals[0] if vals else ''
            if 'pfs' in key.lower():
                try:
                    pfs = float(val)
                except:
                    pass
            if 'rec' in key.lower() and '(1)' in key.lower():
                try:
                    rec = int(val)
                except:
                    pass
            if 'os' in key.lower() and '(m)' in key.lower():
                try:
                    os_m = float(val)
                except:
                    pass
            if 'death' in key.lower():
                try:
                    death = int(val)
                except:
                    pass
        gsm_meta[gsm_name] = {'pfs_months': pfs, 'recurrence': rec,
                               'os_months': os_m, 'death': death}
    meta_df = pd.DataFrame(gsm_meta).T

    # Match to LODO predictions
    gse32062_mask = model_a['dataset'] == 'GSE32062'
    gse32062_ids = model_a.index[gse32062_mask].astype(str)
    gse32062_scores = np.array([all_predictions[sid] for sid in gse32062_ids])

    matched = [sid for sid in gse32062_ids if sid in meta_df.index]
    print(f"GSE32062: {len(gse32062_ids)} with predictions, {len(matched)} matched to metadata")

    if len(matched) > 50:
        m = meta_df.loc[matched]
        scores_m = np.array([all_predictions[sid] for sid in matched])

        # PFS analysis
        pfs_valid = m['pfs_months'].notna() & m['recurrence'].notna()
        if pfs_valid.sum() > 50:
            survival_results['GSE32062_PFS'] = run_survival_analysis(
                'GSE32062 Progression-Free Survival',
                matched, scores_m[pfs_valid.values],
                m.loc[pfs_valid, 'pfs_months'].values.astype(float),
                m.loc[pfs_valid, 'recurrence'].values.astype(float)
            )

        # OS analysis
        os_valid = m['os_months'].notna() & m['death'].notna()
        if os_valid.sum() > 50:
            survival_results['GSE32062_OS'] = run_survival_analysis(
                'GSE32062 Overall Survival',
                matched, scores_m[os_valid.values],
                m.loc[os_valid, 'os_months'].values.astype(float),
                m.loc[os_valid, 'death'].values.astype(float)
            )
except Exception as e:
    print(f"GSE32062 survival analysis failed: {e}")


# --- GSE30161: OS + PFI from GEO metadata ---
print("\n### GSE30161 ###")
try:
    gse30161 = GEOparse.get_GEO('GSE30161', destdir='/tmp/geo_cache', silent=True)
    gsm_meta2 = {}
    for gsm_name, gsm in gse30161.gsms.items():
        meta = gsm.metadata
        pfi = None
        relapse = None
        os_days = None
        dead = None
        for key, vals in meta.items():
            val = vals[0] if vals else ''
            kl = key.lower()
            if 'pfi' in kl and 'day' in kl:
                try:
                    pfi = float(val)
                except:
                    pass
            if 'relapse' in kl:
                try:
                    relapse = int(val) if val.isdigit() else None
                except:
                    pass
            if 'overall survival' in kl and 'day' in kl:
                try:
                    os_days = float(val)
                except:
                    pass
            if 'censoring' in kl:
                try:
                    dead = int(val) if val.isdigit() else None
                except:
                    pass
        gsm_meta2[gsm_name] = {'pfi_days': pfi, 'relapse': relapse,
                                'os_days': os_days, 'dead': dead}
    meta_df2 = pd.DataFrame(gsm_meta2).T

    gse30161_mask = model_a['dataset'] == 'GSE30161'
    gse30161_ids = model_a.index[gse30161_mask].astype(str)
    gse30161_scores = np.array([all_predictions[sid] for sid in gse30161_ids])

    matched2 = [sid for sid in gse30161_ids if sid in meta_df2.index]
    print(f"GSE30161: {len(gse30161_ids)} with predictions, {len(matched2)} matched")

    if len(matched2) > 30:
        m2 = meta_df2.loc[matched2]
        scores_m2 = np.array([all_predictions[sid] for sid in matched2])

        # PFI analysis (convert days to months)
        pfi_valid = m2['pfi_days'].notna() & m2['relapse'].notna()
        if pfi_valid.sum() > 30:
            survival_results['GSE30161_PFI'] = run_survival_analysis(
                'GSE30161 Progression-Free Interval',
                matched2, scores_m2[pfi_valid.values],
                (m2.loc[pfi_valid, 'pfi_days'].values.astype(float) / 30.44),
                m2.loc[pfi_valid, 'relapse'].values.astype(float),
                time_unit='months (from days/30.44)'
            )

        # OS analysis
        os_valid = m2['os_days'].notna() & m2['dead'].notna()
        if os_valid.sum() > 30:
            survival_results['GSE30161_OS'] = run_survival_analysis(
                'GSE30161 Overall Survival',
                matched2, scores_m2[os_valid.values],
                (m2.loc[os_valid, 'os_days'].values.astype(float) / 30.44),
                m2.loc[os_valid, 'dead'].values.astype(float),
                time_unit='months (from days/30.44)'
            )
except Exception as e:
    print(f"GSE30161 survival analysis failed: {e}")


# --- GSE63885: DFS from GEO metadata ---
print("\n### GSE63885 ###")
try:
    gse63885 = GEOparse.get_GEO('GSE63885', destdir='/tmp/geo_cache', silent=True)
    gsm_meta3 = {}
    for gsm_name, gsm in gse63885.gsms.items():
        meta = gsm.metadata
        dfs = None
        status = None
        for key, vals in meta.items():
            val = vals[0] if vals else ''
            kl = key.lower()
            if 'dfs' in kl and 'survival' in kl:
                try:
                    dfs = float(val)
                except:
                    pass
            if 'clinical status at last' in kl:
                # DOD=dead of disease (event=1), NED/AWD=alive (event=0)
                if 'DOD' in val:
                    status = 1
                elif 'NED' in val or 'AWD' in val:
                    status = 0
        gsm_meta3[gsm_name] = {'dfs_days': dfs, 'event': status}
    meta_df3 = pd.DataFrame(gsm_meta3).T

    gse63885_mask = model_a['dataset'] == 'GSE63885'
    gse63885_ids = model_a.index[gse63885_mask].astype(str)
    gse63885_scores = np.array([all_predictions[sid] for sid in gse63885_ids])

    matched3 = [sid for sid in gse63885_ids if sid in meta_df3.index]
    print(f"GSE63885: {len(gse63885_ids)} with predictions, {len(matched3)} matched")

    if len(matched3) > 30:
        m3 = meta_df3.loc[matched3]
        scores_m3 = np.array([all_predictions[sid] for sid in matched3])

        valid3 = m3['dfs_days'].notna() & m3['event'].notna()
        if valid3.sum() > 30:
            survival_results['GSE63885_DFS'] = run_survival_analysis(
                'GSE63885 Disease-Free Survival',
                matched3, scores_m3[valid3.values],
                (m3.loc[valid3, 'dfs_days'].values.astype(float) / 30.44),
                m3.loc[valid3, 'event'].values.astype(float),
                time_unit='months (from days/30.44)'
            )
except Exception as e:
    print(f"GSE63885 survival analysis failed: {e}")


# ============================================================
# PART 2: Top Gene Extraction + Signature Overlap
# ============================================================
print("\n" + "=" * 70)
print("PART 2: Feature Weight Analysis + Signature Overlap")
print("=" * 70)

# Train full clinical model on all Model A data
X_clinical = np.nan_to_num(X_all, nan=0.5)
clf_full = LogisticRegression(
    penalty='l2', C=1.0, solver='lbfgs',
    max_iter=5000, class_weight='balanced', random_state=42
)
clf_full.fit(X_clinical, y_all)

weights = pd.Series(clf_full.coef_[0], index=common_genes)
abs_weights = weights.abs().sort_values(ascending=False)

print(f"\nTop 30 genes by absolute L2 weight:")
for i, (gene, w) in enumerate(abs_weights.head(30).items()):
    direction = "+" if weights[gene] > 0 else "-"
    print(f"  {i+1:2d}. {gene:15s}  |w|={w:.6f}  ({direction})")

top100 = set(abs_weights.head(100).index)
top200 = set(abs_weights.head(200).index)
top500 = set(abs_weights.head(500).index)

# Load published signatures
print(f"\n--- Signature overlap with Exp8 top genes ---")
all_sigs = pd.read_csv(SIG / 'all_signature_gene_lists.csv')
consensus = pd.read_csv(SIG / 'consensus_hrd_genes.csv')
dd500 = pd.read_csv("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp5_normalization/dd500_genes.csv")

# Build signature gene sets
sig_genes = {}
for sig_name in all_sigs['Signature'].unique():
    genes = set(all_sigs[all_sigs['Signature'] == sig_name]['Gene'].values)
    sig_genes[sig_name] = genes

sig_genes['Consensus-93'] = set(consensus['Gene'].values)
sig_genes['DD-500'] = set(dd500['gene'].values)

overlap_results = {}
print(f"\n{'Signature':<25s} {'N_genes':>7s} {'∩ Top100':>8s} {'∩ Top200':>8s} {'∩ Top500':>8s} {'Jaccard100':>10s}")
print("-" * 75)

for sig_name in sorted(sig_genes.keys()):
    genes = sig_genes[sig_name]
    # Only count genes that are in our common gene set
    genes_avail = genes & set(common_genes)
    o100 = len(top100 & genes_avail)
    o200 = len(top200 & genes_avail)
    o500 = len(top500 & genes_avail)
    jaccard100 = len(top100 & genes_avail) / len(top100 | genes_avail) if len(top100 | genes_avail) > 0 else 0

    print(f"  {sig_name:<23s} {len(genes_avail):>7d} {o100:>8d} {o200:>8d} {o500:>8d} {jaccard100:>10.3f}")

    overlap_results[sig_name] = {
        'n_genes_in_common_set': len(genes_avail),
        'overlap_top100': o100,
        'overlap_top200': o200,
        'overlap_top500': o500,
        'jaccard_top100': round(jaccard100, 4),
    }

# Check specific genes of interest
key_genes = ['BRCA1', 'BRCA2', 'RAD51', 'POLQ', 'PARP1', 'MKI67', 'TOP2A',
             'CDK1', 'FOXM1', 'TTK', 'CHEK1', 'ATR', 'ATM', 'FANCD2']
print(f"\n--- Key gene weights ---")
for gene in key_genes:
    if gene in weights.index:
        rank = int((abs_weights > abs_weights[gene]).sum()) + 1
        print(f"  {gene:10s}  weight={weights[gene]:+.6f}  rank={rank}/{len(weights)}")
    else:
        print(f"  {gene:10s}  NOT IN COMMON GENES")

# Check proliferation metagene overlap
prolif_genes = ['MKI67', 'TOP2A', 'PCNA', 'MCM2', 'MCM4', 'MCM6', 'AURKA',
                'AURKB', 'BUB1', 'BUB1B', 'CCNB1', 'CCNB2', 'CDK1', 'CDC20',
                'PLK1', 'TPX2', 'UBE2C', 'BIRC5', 'MELK', 'CEP55']
prolif_in_top100 = top100 & set(prolif_genes)
prolif_in_top200 = top200 & set(prolif_genes)
print(f"\n--- Proliferation metagene in top genes ---")
print(f"  In top 100: {len(prolif_in_top100)}/20 = {sorted(prolif_in_top100)}")
print(f"  In top 200: {len(prolif_in_top200)}/20 = {sorted(prolif_in_top200)}")


# ============================================================
# PART 3: I-SPY2 survival note
# ============================================================
print("\n" + "=" * 70)
print("PART 3: I-SPY2 Survival Interaction")
print("=" * 70)
print("I-SPY2 datasets (GSE173839, GSE194040) only have pCR as endpoint.")
print("No EFS, DRFS, or survival data available in public GEO deposits.")
print("Cannot test score × treatment arm interaction on survival endpoints.")
print("The pCR-based interaction test from Exp8 Phase 3 remains the best available test.")


# ============================================================
# Save results
# ============================================================
print("\n" + "=" * 70)
print("Saving results")
print("=" * 70)

results = {
    'experiment': 'Exp8b: Survival Analysis + Signature Overlap',
    'date': '2026-02-23',
    'survival_analysis': survival_results,
    'signature_overlap': overlap_results,
    'top_100_genes': sorted(list(top100)),
    'ispy2_survival_note': 'No EFS/DRFS data available in public I-SPY2 GEO deposits. Cannot test survival × treatment arm interaction.',
}

with open(EXP / 'exp8b_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save top genes with weights
top_genes_df = pd.DataFrame({
    'gene': abs_weights.index,
    'abs_weight': abs_weights.values,
    'weight': [weights[g] for g in abs_weights.index],
    'rank': range(1, len(abs_weights) + 1),
})
top_genes_df.to_csv(EXP / 'exp8b_gene_weights_ranked.csv', index=False)

print(f"\nResults saved to {EXP}")
print("Done!")
