#!/usr/bin/env python3
"""
Exp8b fix: Survival analysis for GSE32062, GSE30161, GSE63885
using GEOparse phenotype_data (correct column parsing).
Merges results into exp8b_results.json.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import spearmanr
import GEOparse
import json
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
EXP = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")

# Load pooled matrix and regenerate LODO-CV predictions
pooled = pd.read_parquet(EXP / 'pooled_rank_matrix.parquet')
common_genes = [g.strip() for g in open(BASE / "common_genes_all_10_datasets.txt")]

ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (pooled['category'] != 'cell_line') &
    ~((pooled['category'] == 'clinical_ispy2') & (~pooled['drug'].isin(ispy2_parpi_drugs)))
)
model_a = pooled[model_a_mask].copy()

X_all = model_a[common_genes].values
y_all = model_a['response_binary'].values.astype(int)
datasets_all = model_a['dataset'].values
sample_ids_all = model_a.index.values

# Generate LODO predictions
all_predictions = {}
for held_out in sorted(model_a['dataset'].unique()):
    train_mask = datasets_all != held_out
    test_mask = datasets_all == held_out
    X_train = np.nan_to_num(X_all[train_mask], nan=0.5)
    X_test = np.nan_to_num(X_all[test_mask], nan=0.5)
    clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                             max_iter=5000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_all[train_mask])
    y_prob = clf.predict_proba(X_test)[:, 1]
    for sid, prob in zip(sample_ids_all[test_mask], y_prob):
        all_predictions[sid] = prob


def run_survival(name, sample_ids, scores, times, events, time_unit='months'):
    """KM + log-rank + Cox on continuous score."""
    df = pd.DataFrame({'score': scores, 'time': times, 'event': events.astype(int)}).dropna()
    df = df[df['time'] > 0]  # Remove zero/negative times

    print(f"\n--- {name} (n={len(df)}, {int(df['event'].sum())} events, {time_unit}) ---")
    if len(df) < 20:
        print("  SKIPPED: too few samples")
        return {'error': 'too_few', 'n': len(df)}

    median_score = df['score'].median()
    df['high'] = (df['score'] >= median_score).astype(int)
    high, low = df[df['high'] == 1], df[df['high'] == 0]

    print(f"  High-score: n={len(high)}, events={int(high['event'].sum())}, median_time={high['time'].median():.1f}")
    print(f"  Low-score:  n={len(low)}, events={int(low['event'].sum())}, median_time={low['time'].median():.1f}")

    lr = logrank_test(high['time'], low['time'], high['event'], low['event'])
    print(f"  Log-rank p={lr.p_value:.4f}")

    cox_df = df[['time', 'event', 'score']].copy()
    cox = CoxPHFitter()
    try:
        cox.fit(cox_df, duration_col='time', event_col='event')
        hr = np.exp(cox.params_['score'])
        cox_p = cox.summary.loc['score', 'p']
        ci = np.exp(cox.confidence_intervals_.iloc[0].values)
        print(f"  Cox HR={hr:.3f} (95% CI: {ci[0]:.3f}-{ci[1]:.3f}), p={cox_p:.4f}")
    except Exception as e:
        print(f"  Cox failed: {e}")
        hr, cox_p, ci = np.nan, np.nan, [np.nan, np.nan]

    rho, p_corr = spearmanr(df['score'], df['time'])
    print(f"  Score-time Spearman rho={rho:.3f}, p={p_corr:.4f}")

    return {
        'n': len(df), 'n_events': int(df['event'].sum()),
        'logrank_p': round(float(lr.p_value), 6),
        'cox_hr': round(float(hr), 4),
        'cox_p': round(float(cox_p), 6) if not np.isnan(cox_p) else None,
        'cox_ci': [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        'spearman_rho': round(float(rho), 4),
        'spearman_p': round(float(p_corr), 6),
        'km_median_high': round(float(high['time'].median()), 1),
        'km_median_low': round(float(low['time'].median()), 1),
    }


results = {}

# ============================================================
# GSE32062
# ============================================================
print("=" * 60)
print("GSE32062: PFS + OS")
print("=" * 60)

gse = GEOparse.get_GEO('GSE32062', destdir='/tmp/geo_cache', silent=True)
pheno = gse.phenotype_data

# Find correct column names
pfs_col = [c for c in pheno.columns if 'pfs' in c.lower()][0]
rec_col = [c for c in pheno.columns if 'rec' in c.lower()][0]
os_col = [c for c in pheno.columns if 'os' in c.lower()][0]
death_col = [c for c in pheno.columns if 'death' in c.lower()][0]

print(f"PFS col: {pfs_col}, Rec col: {rec_col}, OS col: {os_col}, Death col: {death_col}")

# Get sample IDs for GSE32062
mask = model_a['dataset'] == 'GSE32062'
ids = model_a.index[mask].astype(str).tolist()
scores = np.array([all_predictions[sid] for sid in ids])
print(f"Samples with predictions: {len(ids)}")

# Match to phenotype data
matched = [sid for sid in ids if sid in pheno.index]
print(f"Matched to phenotype: {len(matched)}")

if len(matched) > 50:
    p = pheno.loc[matched]
    s = np.array([all_predictions[sid] for sid in matched])

    pfs = pd.to_numeric(p[pfs_col], errors='coerce')
    rec = pd.to_numeric(p[rec_col], errors='coerce')
    valid = pfs.notna() & rec.notna()
    print(f"Valid PFS: {valid.sum()}")
    if valid.sum() > 50:
        results['GSE32062_PFS'] = run_survival(
            'GSE32062 PFS', matched, s[valid.values],
            pfs[valid].values, rec[valid].values)

    os_m = pd.to_numeric(p[os_col], errors='coerce')
    death = pd.to_numeric(p[death_col], errors='coerce')
    valid_os = os_m.notna() & death.notna()
    if valid_os.sum() > 50:
        results['GSE32062_OS'] = run_survival(
            'GSE32062 OS', matched, s[valid_os.values],
            os_m[valid_os].values, death[valid_os].values)


# ============================================================
# GSE30161
# ============================================================
print("\n" + "=" * 60)
print("GSE30161: PFI + OS")
print("=" * 60)

gse2 = GEOparse.get_GEO('GSE30161', destdir='/tmp/geo_cache', silent=True)
pheno2 = gse2.phenotype_data

print(f"Columns: {[c for c in pheno2.columns if any(x in c.lower() for x in ['pfi', 'relapse', 'survival', 'censoring', 'os'])]}")

pfi_col = [c for c in pheno2.columns if 'pfi' in c.lower()][0]
relapse_col = [c for c in pheno2.columns if 'relapse' in c.lower()][0]
os_col2 = [c for c in pheno2.columns if 'overall survival' in c.lower()][0]
cens_col = [c for c in pheno2.columns if 'censoring' in c.lower()][0]

mask2 = model_a['dataset'] == 'GSE30161'
ids2 = model_a.index[mask2].astype(str).tolist()
scores2 = np.array([all_predictions[sid] for sid in ids2])

matched2 = [sid for sid in ids2 if sid in pheno2.index]
print(f"Matched: {len(matched2)}")

if len(matched2) > 30:
    p2 = pheno2.loc[matched2]
    s2 = np.array([all_predictions[sid] for sid in matched2])

    # PFI (days → months)
    pfi = pd.to_numeric(p2[pfi_col], errors='coerce')
    rel = pd.to_numeric(p2[relapse_col], errors='coerce')
    valid_pfi = pfi.notna() & rel.notna()
    if valid_pfi.sum() > 30:
        results['GSE30161_PFI'] = run_survival(
            'GSE30161 PFI', matched2, s2[valid_pfi.values],
            pfi[valid_pfi].values / 30.44, rel[valid_pfi].values,
            time_unit='months (days/30.44)')

    # OS (days → months)
    os_d = pd.to_numeric(p2[os_col2], errors='coerce')
    dead = pd.to_numeric(p2[cens_col], errors='coerce')
    valid_os2 = os_d.notna() & dead.notna()
    if valid_os2.sum() > 30:
        results['GSE30161_OS'] = run_survival(
            'GSE30161 OS', matched2, s2[valid_os2.values],
            os_d[valid_os2].values / 30.44, dead[valid_os2].values,
            time_unit='months (days/30.44)')


# ============================================================
# GSE63885
# ============================================================
print("\n" + "=" * 60)
print("GSE63885: DFS + OS")
print("=" * 60)

gse3 = GEOparse.get_GEO('GSE63885', destdir='/tmp/geo_cache', silent=True)
pheno3 = gse3.phenotype_data

surv_cols = [c for c in pheno3.columns if any(x in c.lower() for x in ['dfs', 'os', 'survival', 'status', 'follow'])]
print(f"Survival columns: {surv_cols}")

dfs_col = [c for c in pheno3.columns if 'dfs' in c.lower()][0]
os_col3 = [c for c in pheno3.columns if 'os -' in c.lower() or 'overall survival' in c.lower()][0]
status_col = [c for c in pheno3.columns if 'clinical status at last' in c.lower()][0]

mask3 = model_a['dataset'] == 'GSE63885'
ids3 = model_a.index[mask3].astype(str).tolist()
scores3 = np.array([all_predictions[sid] for sid in ids3])

matched3 = [sid for sid in ids3 if sid in pheno3.index]
print(f"Matched: {len(matched3)}")

if len(matched3) > 30:
    p3 = pheno3.loc[matched3]
    s3 = np.array([all_predictions[sid] for sid in matched3])

    # DFS (days → months), event = not alive/NED
    dfs = pd.to_numeric(p3[dfs_col], errors='coerce')
    # Status: DOD = dead of disease (event), AWD = alive with disease, NED = no evidence of disease
    event3 = p3[status_col].apply(
        lambda x: 1 if 'DOD' in str(x) else (0 if any(s in str(x) for s in ['NED', 'AWD']) else np.nan))
    valid3 = dfs.notna() & event3.notna()
    if valid3.sum() > 30:
        results['GSE63885_DFS'] = run_survival(
            'GSE63885 DFS', matched3, s3[valid3.values],
            dfs[valid3].values / 30.44, event3[valid3].values,
            time_unit='months (days/30.44)')

    # OS
    os_d3 = pd.to_numeric(p3[os_col3], errors='coerce')
    if os_d3.notna().sum() > 30:
        results['GSE63885_OS'] = run_survival(
            'GSE63885 OS', matched3, s3[os_d3.notna().values & event3.notna().values],
            os_d3[os_d3.notna() & event3.notna()].values / 30.44,
            event3[os_d3.notna() & event3.notna()].values,
            time_unit='months (days/30.44)')


# ============================================================
# Merge into existing results
# ============================================================
print("\n" + "=" * 60)
print("Merging results")
print("=" * 60)

existing = json.load(open(EXP / 'exp8b_results.json'))
existing['survival_analysis'].update(results)

with open(EXP / 'exp8b_results.json', 'w') as f:
    json.dump(existing, f, indent=2)

print(f"\nUpdated {EXP / 'exp8b_results.json'} with {len(results)} new survival analyses")
print("\nDone!")
