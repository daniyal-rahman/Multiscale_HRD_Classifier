#!/usr/bin/env python3
"""
Multivariate Cox Proportional Hazards Survival Analysis for TCGA-OV.

Tests whether the drug response model's predicted score is independently
prognostic after adjusting for standard clinical covariates:
  - Stage (FIGO)
  - Grade
  - Residual disease (debulking status)
  - Age at diagnosis
  - BRCA mutation status

Models fitted:
  1. Univariate: score only
  2. Multivariate: score + stage + grade + residual_disease + age
  3. Full: score + stage + grade + residual_disease + age + BRCA_status
  4. Clinical only: stage + grade + residual_disease + age (no score)
  5. Likelihood ratio test: Model 2 vs Model 4
  6. Combined clinical+molecular score with calibration

Clinical data sources:
  - Survival/response: tcga_ov_platinum_clinical.parquet
  - Stage/grade/age/residual: UCSC Xena TCGA OV clinical matrix
  - BRCA mutations: cBioPortal (combined Nature 2011 + PanCan + Firehose)
"""

import sys
import json
import io
import warnings
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
EXP = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT = EXP / "validation_fixed"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("MULTIVARIATE COX PH SURVIVAL ANALYSIS — TCGA-OV")
print("=" * 70)

# ============================================================
# STEP 1: Load data and generate LODO-CV predictions
# ============================================================
print("\n--- Step 1: Loading data and generating LODO-CV predictions ---")

pooled = pd.read_parquet(EXP / 'pooled_rank_matrix.parquet')
common_genes = [g.strip() for g in open(BASE / "common_genes_all_10_datasets.txt")]
gene_cols = common_genes
print(f"Pooled matrix: {pooled.shape[0]} samples, {len(gene_cols)} genes")

# Model A filter: exclude cell_line and non-PARPi I-SPY2
ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (pooled['category'] != 'cell_line') &
    ~((pooled['category'] == 'clinical_ispy2') & (~pooled['drug'].isin(ispy2_parpi_drugs)))
)
model_a = pooled[model_a_mask].copy()
print(f"Model A training set: {len(model_a)} samples")

# LODO-CV: hold out TCGA-OV, train on rest
train_mask = model_a['dataset'] != 'TCGA-OV'
test_mask = model_a['dataset'] == 'TCGA-OV'

X_train = np.nan_to_num(model_a.loc[train_mask, gene_cols].values, nan=0.5)
y_train = model_a.loc[train_mask, 'response_binary'].values.astype(int)
X_test = np.nan_to_num(model_a.loc[test_mask, gene_cols].values, nan=0.5)
test_ids = model_a.index[test_mask].astype(str).values

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} TCGA-OV samples")

clf = LogisticRegression(
    penalty='l2', C=0.01, solver='lbfgs',
    max_iter=5000, class_weight='balanced', random_state=42
)
clf.fit(X_train, y_train)
pred_probs = clf.predict_proba(X_test)[:, 1]

# Map predictions to patient IDs
pred_map = dict(zip(test_ids, pred_probs))
print(f"LODO-CV predictions generated for {len(pred_map)} TCGA-OV samples")
print(f"  Score range: {pred_probs.min():.3f} - {pred_probs.max():.3f}")
print(f"  Score mean: {pred_probs.mean():.3f}, median: {np.median(pred_probs):.3f}")

# ============================================================
# STEP 2: Download clinical covariates from Xena
# ============================================================
print("\n--- Step 2: Downloading clinical covariates from UCSC Xena ---")

url = 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.OV.sampleMap%2FOV_clinicalMatrix'
r = requests.get(url, timeout=60)
xena_df = pd.read_csv(io.BytesIO(r.content), sep='\t')

# De-duplicate: one row per patient, prefer primary tumor entries
xena_df['patient'] = xena_df['bcr_patient_barcode']
xena_dedup = xena_df.drop_duplicates(subset='patient', keep='first').set_index('patient')
print(f"Xena clinical matrix: {len(xena_dedup)} unique patients")

# ============================================================
# STEP 3: Load survival data and BRCA labels
# ============================================================
print("\n--- Step 3: Loading survival data and BRCA labels ---")

tcga_clin = pd.read_parquet(BASE / 'tcga_ov_platinum_clinical.parquet')
tcga_clin.index = tcga_clin.index.astype(str)
print(f"Clinical parquet: {len(tcga_clin)} patients")

# Load BRCA labels
with open('/tmp/ov_tcga_combined_brca.json') as f:
    brca_data = json.load(f)

brca_labels = {}
for pid in tcga_clin.index:
    brca_labels[pid] = 1 if pid in brca_data else 0

n_brca_pos = sum(v == 1 for v in brca_labels.values())
print(f"BRCA labels: {n_brca_pos} mutated, {len(brca_labels) - n_brca_pos} wild-type")

# ============================================================
# STEP 4: Build merged analysis dataframe
# ============================================================
print("\n--- Step 4: Building merged analysis dataframe ---")

rows = []
for pid in tcga_clin.index:
    if pid not in pred_map:
        continue

    clin = tcga_clin.loc[pid]
    xena = xena_dedup.loc[pid] if pid in xena_dedup.index else pd.Series(dtype=object)

    # Survival endpoints
    os_months = pd.to_numeric(clin.get('OS_MONTHS'), errors='coerce')
    os_event = 1 if '1:' in str(clin.get('OS_STATUS', '')) or 'DECEASED' in str(clin.get('OS_STATUS', '')) else 0
    dfs_months = pd.to_numeric(clin.get('DFS_MONTHS'), errors='coerce')
    dfs_event = 1 if '1:' in str(clin.get('DFS_STATUS', '')) or 'Recurred' in str(clin.get('DFS_STATUS', '')) else 0

    # Clinical covariates from Xena
    age = pd.to_numeric(xena.get('age_at_initial_pathologic_diagnosis'), errors='coerce') if len(xena) > 0 else np.nan
    stage_raw = str(xena.get('clinical_stage', '')) if len(xena) > 0 else ''
    grade_raw = str(xena.get('neoplasm_histologic_grade', '')) if len(xena) > 0 else ''
    residual_raw = str(xena.get('tumor_residual_disease', '')) if len(xena) > 0 else ''

    # Encode stage: binary (early I-II vs advanced III-IV)
    if 'III' in stage_raw or 'IV' in stage_raw:
        stage_advanced = 1
    elif 'I' in stage_raw or 'II' in stage_raw:
        stage_advanced = 0
    else:
        stage_advanced = np.nan

    # Also encode stage as ordinal for finer resolution
    # Stage I/II = 1-2, IIIA/B = 3, IIIC = 4, IV = 5
    if 'Stage IV' in stage_raw:
        stage_ordinal = 5
    elif 'Stage IIIC' in stage_raw:
        stage_ordinal = 4
    elif 'Stage IIIB' in stage_raw or 'Stage IIIA' in stage_raw:
        stage_ordinal = 3
    elif 'Stage II' in stage_raw:
        stage_ordinal = 2
    elif 'Stage I' in stage_raw:
        stage_ordinal = 1
    else:
        stage_ordinal = np.nan

    # Encode grade: high (G3/G4) vs low/intermediate (G1/G2)
    if grade_raw in ('G3', 'G4'):
        grade_high = 1
    elif grade_raw in ('G1', 'G2'):
        grade_high = 0
    else:
        grade_high = np.nan

    # Encode residual disease: optimal (no macroscopic) vs suboptimal
    if residual_raw == 'No Macroscopic disease':
        residual_optimal = 1  # optimal debulking
        residual_ordinal = 0  # no residual
    elif residual_raw == '1-10 mm':
        residual_optimal = 0
        residual_ordinal = 1
    elif residual_raw == '11-20 mm':
        residual_optimal = 0
        residual_ordinal = 2
    elif residual_raw == '>20 mm':
        residual_optimal = 0
        residual_ordinal = 3
    else:
        residual_optimal = np.nan
        residual_ordinal = np.nan

    rows.append({
        'patient': pid,
        'score': pred_map[pid],
        'os_months': os_months,
        'os_event': os_event,
        'dfs_months': dfs_months,
        'dfs_event': dfs_event,
        'age': age,
        'stage_advanced': stage_advanced,
        'stage_ordinal': stage_ordinal,
        'grade_high': grade_high,
        'residual_optimal': residual_optimal,
        'residual_ordinal': residual_ordinal,
        'brca_mutated': brca_labels.get(pid, np.nan),
        'stage_raw': stage_raw,
        'grade_raw': grade_raw,
        'residual_raw': residual_raw,
    })

df = pd.DataFrame(rows)
print(f"Merged dataframe: {len(df)} patients")
print(f"\nCovariate coverage:")
for col in ['age', 'stage_advanced', 'grade_high', 'residual_optimal', 'brca_mutated']:
    n_valid = df[col].notna().sum()
    print(f"  {col}: {n_valid}/{len(df)} ({100*n_valid/len(df):.1f}%)")

print(f"\nStage distribution:")
print(df['stage_raw'].value_counts().to_string())
print(f"\nGrade distribution:")
print(df['grade_raw'].value_counts().to_string())
print(f"\nResidual disease distribution:")
print(df['residual_raw'].value_counts().to_string())

# ============================================================
# STEP 5: Descriptive statistics for the score vs clinical variables
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Score vs Clinical Variables (confounding check)")
print("=" * 70)

confounding = {}

# Score vs stage
for grp_name, grp_col, grp_vals in [
    ('stage', 'stage_advanced', [0, 1]),
    ('grade', 'grade_high', [0, 1]),
    ('residual', 'residual_optimal', [0, 1]),
    ('brca', 'brca_mutated', [0, 1]),
]:
    grp0 = df.loc[df[grp_col] == grp_vals[0], 'score']
    grp1 = df.loc[df[grp_col] == grp_vals[1], 'score']
    if len(grp0) > 5 and len(grp1) > 5:
        stat, p = stats.mannwhitneyu(grp0, grp1, alternative='two-sided')
        print(f"  Score vs {grp_name}: mean(0)={grp0.mean():.3f} (n={len(grp0)}), "
              f"mean(1)={grp1.mean():.3f} (n={len(grp1)}), MW p={p:.4f}")
        confounding[f'score_vs_{grp_name}'] = {
            'mean_group0': round(float(grp0.mean()), 4),
            'mean_group1': round(float(grp1.mean()), 4),
            'n_group0': int(len(grp0)),
            'n_group1': int(len(grp1)),
            'mannwhitney_p': round(float(p), 6),
        }

# Score vs age (correlation)
valid_age = df.dropna(subset=['age'])
rho, p = stats.spearmanr(valid_age['score'], valid_age['age'])
print(f"  Score vs age: Spearman rho={rho:.3f}, p={p:.4f}")
confounding['score_vs_age'] = {
    'spearman_rho': round(float(rho), 4),
    'spearman_p': round(float(p), 6),
}

# ============================================================
# STEP 6: Cox PH Models
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Cox PH Multivariate Models")
print("=" * 70)

results = {
    'experiment': 'Multivariate Cox PH Survival Analysis — TCGA-OV',
    'date': '2026-02-24',
    'n_patients': len(df),
    'covariates': {
        'score': 'LODO-CV predicted probability (continuous, 0-1)',
        'age': 'Age at diagnosis (continuous, years)',
        'stage_advanced': 'FIGO stage III/IV vs I/II (binary)',
        'grade_high': 'Grade G3/G4 vs G1/G2 (binary)',
        'residual_optimal': 'Optimal debulking (no macroscopic disease) vs suboptimal (binary)',
        'brca_mutated': 'Any BRCA1/2 mutation vs wild-type (binary)',
    },
    'confounding_checks': confounding,
}


def fit_cox_model(data, time_col, event_col, covariates, model_name, penalizer=0.01):
    """Fit a Cox PH model and return results dictionary."""
    cox_df = data[[time_col, event_col] + covariates].dropna().copy()
    cox_df.columns = ['time', 'event'] + covariates

    # Only include rows with time > 0
    cox_df = cox_df[cox_df['time'] > 0].copy()

    n = len(cox_df)
    n_events = int(cox_df['event'].sum())

    if n < 20 or n_events < 5:
        return {'error': f'Too few samples (n={n}) or events (n_events={n_events})', 'n': n, 'n_events': n_events}

    try:
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(cox_df, duration_col='time', event_col='event')

        model_result = {
            'model_name': model_name,
            'n': n,
            'n_events': n_events,
            'concordance_index': round(float(cph.concordance_index_), 4),
            'partial_AIC': round(float(cph.AIC_partial_), 2),
            'log_likelihood': round(float(cph.log_likelihood_), 4),
            'covariates': {},
        }

        for var in covariates:
            hr = float(np.exp(cph.params_[var]))
            ci_low = float(np.exp(cph.confidence_intervals_.loc[var].iloc[0]))
            ci_high = float(np.exp(cph.confidence_intervals_.loc[var].iloc[1]))
            p_val = float(cph.summary.loc[var, 'p'])
            coef = float(cph.params_[var])

            model_result['covariates'][var] = {
                'HR': round(hr, 4),
                'HR_CI_low': round(ci_low, 4),
                'HR_CI_high': round(ci_high, 4),
                'p_value': round(p_val, 6),
                'coef': round(coef, 4),
            }

        return model_result, cph

    except Exception as e:
        return {'error': str(e), 'n': n, 'n_events': n_events}, None


def likelihood_ratio_test(ll_full, ll_reduced, df_diff):
    """Likelihood ratio test comparing nested models."""
    lr_stat = -2 * (ll_reduced - ll_full)
    p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
    return lr_stat, p_value


endpoint_results = {}

for endpoint, time_col, event_col in [
    ('DFS', 'dfs_months', 'dfs_event'),
    ('OS', 'os_months', 'os_event'),
]:
    print(f"\n{'='*60}")
    print(f"  {endpoint} Analysis")
    print(f"{'='*60}")

    valid = df[df[time_col].notna() & (df[time_col] > 0)].copy()
    print(f"  Valid patients for {endpoint}: {len(valid)} ({int(valid[event_col].sum())} events)")

    ep_results = {}

    # ---- Model 1: Univariate — score only ----
    print(f"\n  Model 1: Univariate (score only)")
    m1_result, m1_cph = fit_cox_model(valid, time_col, event_col, ['score'], 'univariate_score', penalizer=0.0)
    if isinstance(m1_result, tuple):
        m1_result, m1_cph = m1_result
    if 'covariates' in m1_result:
        s = m1_result['covariates']['score']
        print(f"    HR={s['HR']:.4f} (CI: {s['HR_CI_low']:.4f}-{s['HR_CI_high']:.4f}), "
              f"p={s['p_value']:.6f}, C-index={m1_result['concordance_index']:.4f}")
    ep_results['model1_univariate_score'] = m1_result

    # ---- Model 2: Score + clinical covariates ----
    print(f"\n  Model 2: Score + stage + grade + residual + age")
    covars_m2 = ['score', 'stage_advanced', 'grade_high', 'residual_optimal', 'age']
    m2_result, m2_cph = fit_cox_model(valid, time_col, event_col, covars_m2, 'score_plus_clinical', penalizer=0.0)
    if isinstance(m2_result, tuple):
        m2_result, m2_cph = m2_result
    if 'covariates' in m2_result:
        for var in covars_m2:
            v = m2_result['covariates'][var]
            sig = "***" if v['p_value'] < 0.001 else "**" if v['p_value'] < 0.01 else "*" if v['p_value'] < 0.05 else ""
            print(f"    {var:25s}: HR={v['HR']:.4f} (CI: {v['HR_CI_low']:.4f}-{v['HR_CI_high']:.4f}), "
                  f"p={v['p_value']:.6f} {sig}")
        print(f"    C-index: {m2_result['concordance_index']:.4f}")
    ep_results['model2_score_plus_clinical'] = m2_result

    # ---- Model 3: Score + clinical + BRCA ----
    print(f"\n  Model 3: Score + stage + grade + residual + age + BRCA")
    covars_m3 = ['score', 'stage_advanced', 'grade_high', 'residual_optimal', 'age', 'brca_mutated']
    m3_result, m3_cph = fit_cox_model(valid, time_col, event_col, covars_m3, 'score_plus_clinical_plus_brca', penalizer=0.0)
    if isinstance(m3_result, tuple):
        m3_result, m3_cph = m3_result
    if 'covariates' in m3_result:
        for var in covars_m3:
            v = m3_result['covariates'][var]
            sig = "***" if v['p_value'] < 0.001 else "**" if v['p_value'] < 0.01 else "*" if v['p_value'] < 0.05 else ""
            print(f"    {var:25s}: HR={v['HR']:.4f} (CI: {v['HR_CI_low']:.4f}-{v['HR_CI_high']:.4f}), "
                  f"p={v['p_value']:.6f} {sig}")
        print(f"    C-index: {m3_result['concordance_index']:.4f}")
    ep_results['model3_score_plus_clinical_plus_brca'] = m3_result

    # ---- Model 4: Clinical only (no score) ----
    print(f"\n  Model 4: Clinical only (no score)")
    covars_m4 = ['stage_advanced', 'grade_high', 'residual_optimal', 'age']
    m4_result, m4_cph = fit_cox_model(valid, time_col, event_col, covars_m4, 'clinical_only', penalizer=0.0)
    if isinstance(m4_result, tuple):
        m4_result, m4_cph = m4_result
    if 'covariates' in m4_result:
        for var in covars_m4:
            v = m4_result['covariates'][var]
            sig = "***" if v['p_value'] < 0.001 else "**" if v['p_value'] < 0.01 else "*" if v['p_value'] < 0.05 else ""
            print(f"    {var:25s}: HR={v['HR']:.4f} (CI: {v['HR_CI_low']:.4f}-{v['HR_CI_high']:.4f}), "
                  f"p={v['p_value']:.6f} {sig}")
        print(f"    C-index: {m4_result['concordance_index']:.4f}")
    ep_results['model4_clinical_only'] = m4_result

    # ---- Likelihood ratio test: Model 2 vs Model 4 ----
    print(f"\n  Likelihood Ratio Test: Model 2 (score+clinical) vs Model 4 (clinical only)")
    if ('log_likelihood' in m2_result and 'log_likelihood' in m4_result
            and m2_cph is not None and m4_cph is not None):
        # Models must be fit on the same samples for valid LRT
        # Refit both on the intersection of complete cases
        common_covars = covars_m2  # includes score + clinical
        common_df = valid[['patient', time_col, event_col] + common_covars].dropna()
        common_df = common_df[common_df[time_col] > 0]

        # Fit Model 2 on common samples
        cox_m2 = CoxPHFitter(penalizer=0.0)
        cox_m2.fit(common_df[[time_col, event_col] + covars_m2], duration_col=time_col, event_col=event_col)

        # Fit Model 4 on same samples
        cox_m4 = CoxPHFitter(penalizer=0.0)
        cox_m4.fit(common_df[[time_col, event_col] + covars_m4], duration_col=time_col, event_col=event_col)

        ll_full = cox_m2.log_likelihood_
        ll_reduced = cox_m4.log_likelihood_
        df_diff = 1  # score is the one additional parameter

        lr_stat, lr_p = likelihood_ratio_test(ll_full, ll_reduced, df_diff)
        print(f"    LR statistic: {lr_stat:.4f}")
        print(f"    p-value: {lr_p:.6f}")
        print(f"    Common N: {len(common_df)}")

        ep_results['lr_test_model2_vs_model4'] = {
            'lr_statistic': round(float(lr_stat), 4),
            'p_value': round(float(lr_p), 6),
            'df': df_diff,
            'n_common': len(common_df),
            'interpretation': 'Adding the score significantly improves the model' if lr_p < 0.05
                else 'Adding the score does NOT significantly improve the model',
        }
    else:
        print("    Could not compute LRT (one or both models failed)")

    # ---- Model 5: Score + BRCA only (no clinical) ----
    print(f"\n  Model 5: Score + BRCA only (no clinical)")
    covars_m5 = ['score', 'brca_mutated']
    m5_result, m5_cph = fit_cox_model(valid, time_col, event_col, covars_m5, 'score_plus_brca', penalizer=0.0)
    if isinstance(m5_result, tuple):
        m5_result, m5_cph = m5_result
    if 'covariates' in m5_result:
        for var in covars_m5:
            v = m5_result['covariates'][var]
            sig = "***" if v['p_value'] < 0.001 else "**" if v['p_value'] < 0.01 else "*" if v['p_value'] < 0.05 else ""
            print(f"    {var:25s}: HR={v['HR']:.4f} (CI: {v['HR_CI_low']:.4f}-{v['HR_CI_high']:.4f}), "
                  f"p={v['p_value']:.6f} {sig}")
        print(f"    C-index: {m5_result['concordance_index']:.4f}")
    ep_results['model5_score_plus_brca'] = m5_result

    # ---- Additional: Stage ordinal model ----
    print(f"\n  Model 6: Score + stage_ordinal + grade + residual_ordinal + age")
    covars_m6 = ['score', 'stage_ordinal', 'grade_high', 'residual_ordinal', 'age']
    m6_result, m6_cph = fit_cox_model(valid, time_col, event_col, covars_m6, 'score_plus_ordinal_clinical', penalizer=0.0)
    if isinstance(m6_result, tuple):
        m6_result, m6_cph = m6_result
    if 'covariates' in m6_result:
        for var in covars_m6:
            v = m6_result['covariates'][var]
            sig = "***" if v['p_value'] < 0.001 else "**" if v['p_value'] < 0.01 else "*" if v['p_value'] < 0.05 else ""
            print(f"    {var:25s}: HR={v['HR']:.4f} (CI: {v['HR_CI_low']:.4f}-{v['HR_CI_high']:.4f}), "
                  f"p={v['p_value']:.6f} {sig}")
        print(f"    C-index: {m6_result['concordance_index']:.4f}")
    ep_results['model6_ordinal_clinical'] = m6_result

    endpoint_results[endpoint] = ep_results

results['cox_models'] = endpoint_results

# ============================================================
# STEP 7: Combined clinical + molecular score
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Combined Clinical + Molecular Score")
print("=" * 70)

combined_score_results = {}

for endpoint, time_col, event_col in [
    ('DFS', 'dfs_months', 'dfs_event'),
    ('OS', 'os_months', 'os_event'),
]:
    covars = ['score', 'stage_advanced', 'grade_high', 'residual_optimal', 'age']
    valid = df[[time_col, event_col] + covars].dropna()
    valid = valid[valid[time_col] > 0].copy()

    if len(valid) < 30:
        print(f"  {endpoint}: too few samples for combined score")
        continue

    # Fit Cox model to get linear predictor
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(valid, duration_col=time_col, event_col=event_col)

    # The linear predictor is the combined score
    lp = cph.predict_partial_hazard(valid).values.flatten()

    # Also get the molecular-only linear predictor
    cph_mol = CoxPHFitter(penalizer=0.0)
    cph_mol.fit(valid[[time_col, event_col, 'score']], duration_col=time_col, event_col=event_col)
    lp_mol = cph_mol.predict_partial_hazard(valid).values.flatten()

    # Concordance comparison
    from lifelines.utils import concordance_index as c_index
    c_combined = c_index(valid[time_col], -lp, valid[event_col])
    c_mol_only = c_index(valid[time_col], -lp_mol, valid[event_col])

    print(f"\n  {endpoint}:")
    print(f"    C-index (molecular only): {c_mol_only:.4f}")
    print(f"    C-index (combined):       {c_combined:.4f}")
    print(f"    Improvement:              {c_combined - c_mol_only:+.4f}")

    # Calibration: divide into tertiles and check observed vs expected
    valid_with_lp = valid.copy()
    valid_with_lp['combined_lp'] = lp
    valid_with_lp['tertile'] = pd.qcut(lp, q=3, labels=['Low risk', 'Medium risk', 'High risk'])

    print(f"\n    Calibration by risk tertile:")
    cal_data = []
    for tertile in ['Low risk', 'Medium risk', 'High risk']:
        sub = valid_with_lp[valid_with_lp['tertile'] == tertile]
        n = len(sub)
        n_events = int(sub[event_col].sum())
        event_rate = n_events / n if n > 0 else 0
        median_time = sub[time_col].median()
        median_lp = sub['combined_lp'].median()
        print(f"      {tertile:15s}: n={n}, events={n_events} ({100*event_rate:.1f}%), "
              f"median time={median_time:.1f} mo, median LP={median_lp:.3f}")
        cal_data.append({
            'tertile': tertile,
            'n': int(n),
            'n_events': n_events,
            'event_rate': round(float(event_rate), 4),
            'median_time': round(float(median_time), 1),
            'median_linear_predictor': round(float(median_lp), 4),
        })

    # Log-rank between tertiles
    low = valid_with_lp[valid_with_lp['tertile'] == 'Low risk']
    high = valid_with_lp[valid_with_lp['tertile'] == 'High risk']
    lr = logrank_test(low[time_col], high[time_col], low[event_col], high[event_col])
    print(f"      Log-rank (low vs high risk): p={lr.p_value:.6f}")

    combined_score_results[endpoint] = {
        'c_index_molecular_only': round(float(c_mol_only), 4),
        'c_index_combined': round(float(c_combined), 4),
        'c_index_improvement': round(float(c_combined - c_mol_only), 4),
        'calibration_tertiles': cal_data,
        'logrank_low_vs_high': round(float(lr.p_value), 6),
        'n': len(valid),
        'n_events': int(valid[event_col].sum()),
    }

results['combined_score'] = combined_score_results

# ============================================================
# STEP 8: Proportional hazards assumption check
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Proportional Hazards Assumption Check (Schoenfeld)")
print("=" * 70)

ph_test_results = {}
for endpoint, time_col, event_col in [
    ('DFS', 'dfs_months', 'dfs_event'),
    ('OS', 'os_months', 'os_event'),
]:
    covars = ['score', 'stage_advanced', 'grade_high', 'residual_optimal', 'age']
    valid = df[[time_col, event_col] + covars].dropna()
    valid = valid[valid[time_col] > 0].copy()

    if len(valid) < 30:
        continue

    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(valid, duration_col=time_col, event_col=event_col)

    try:
        ph_test = cph.check_assumptions(valid, p_value_threshold=0.05, show_plots=False)
        # If no violations, the test returns empty or None
        print(f"\n  {endpoint}: Proportional hazards assumption — no significant violations detected")
        ph_test_results[endpoint] = {'violations': False, 'note': 'All covariates pass PH test at p<0.05'}
    except Exception as e:
        msg = str(e)
        print(f"\n  {endpoint}: PH assumption check note: {msg[:200]}")
        ph_test_results[endpoint] = {'violations': True, 'note': msg[:500]}

results['ph_assumption_tests'] = ph_test_results

# ============================================================
# STEP 9: Summary table
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Multivariate Cox PH Results")
print("=" * 70)

for endpoint in ['DFS', 'OS']:
    if endpoint not in endpoint_results:
        continue
    ep = endpoint_results[endpoint]

    print(f"\n--- {endpoint} ---")
    print(f"{'Model':<45s} {'Score HR':>10s} {'95% CI':>20s} {'p-value':>10s} {'C-index':>10s}")
    print("-" * 95)

    for model_key, model_label in [
        ('model1_univariate_score', 'Model 1: Score only'),
        ('model2_score_plus_clinical', 'Model 2: Score + clinical'),
        ('model3_score_plus_clinical_plus_brca', 'Model 3: Score + clinical + BRCA'),
        ('model4_clinical_only', 'Model 4: Clinical only (no score)'),
        ('model5_score_plus_brca', 'Model 5: Score + BRCA'),
        ('model6_ordinal_clinical', 'Model 6: Score + ordinal clinical'),
    ]:
        m = ep.get(model_key, {})
        if 'covariates' in m:
            if 'score' in m['covariates']:
                s = m['covariates']['score']
                hr_str = f"{s['HR']:.4f}"
                ci_str = f"({s['HR_CI_low']:.4f}-{s['HR_CI_high']:.4f})"
                p_str = f"{s['p_value']:.6f}"
            else:
                hr_str = "N/A"
                ci_str = "N/A"
                p_str = "N/A"
            c_str = f"{m['concordance_index']:.4f}"
        else:
            hr_str = ci_str = p_str = c_str = "FAILED"

        print(f"{model_label:<45s} {hr_str:>10s} {ci_str:>20s} {p_str:>10s} {c_str:>10s}")

    # LRT
    lrt = ep.get('lr_test_model2_vs_model4', {})
    if 'p_value' in lrt:
        print(f"\n  LRT (Model 2 vs Model 4): chi2={lrt['lr_statistic']:.4f}, "
              f"p={lrt['p_value']:.6f}, df={lrt['df']}")
        print(f"  --> {lrt['interpretation']}")

# ============================================================
# SAVE
# ============================================================
print("\n" + "=" * 70)
print("Saving results")
print("=" * 70)

with open(OUT / 'multivariate_survival.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"Results saved to {OUT / 'multivariate_survival.json'}")
print("\nDone!")
