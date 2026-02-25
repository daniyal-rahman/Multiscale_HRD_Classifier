#!/usr/bin/env python3
"""
Copy Number / HRD Score Covariate Analysis for TCGA-OV

Addresses reviewer concern: "look at copynumber and try adding it as
covariate in survival model and immune corr analysis."

Data sources:
1. Knijnenburg et al. 2018 (PanCan DDR paper) via GerkeLab/TCGAhrd:
   HRD_LOH, HRD_TAI, HRD_LST, HRD_Score (sum), CNA_frac_altered,
   aneuploidy_score, purity, ploidy, mutSig3 (BRCA-like mut signature)
2. cBioPortal: FRACTION_GENOME_ALTERED, MUTATION_COUNT

Analyses:
A. Correlation: HRD scores vs RNA model score
B. Predictive value: HRD scores vs response (AUC, logistic regression)
C. Survival covariate: Cox PH with HRD scores added
D. Independence: Does RNA score add beyond HRD? Does HRD add beyond RNA?
E. Combined model: RNA score + HRD score
"""

import csv
import json
import sys
import io
import warnings
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
EXP = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT = EXP / "addendum"

# ============================================================
# STEP 1: Generate LODO-CV predictions for TCGA-OV
# ============================================================
print("=" * 70)
print("COPY NUMBER / HRD COVARIATE ANALYSIS — TCGA-OV")
print("=" * 70)

print("\n--- Step 1: Loading data and generating LODO-CV predictions ---")

pooled = pd.read_parquet(EXP / 'validation_fixed' / 'pooled_rank_matrix_fixed.parquet')
common_genes = [g.strip() for g in open(BASE / "gse32062_fix" / "common_genes_all_10_datasets_fixed.txt")]
gene_cols = common_genes
print(f"Pooled matrix: {pooled.shape[0]} samples, {len(gene_cols)} genes")

# Model A filter
ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (pooled['category'] != 'cell_line') &
    ~((pooled['category'] == 'clinical_ispy2') & (~pooled['drug'].isin(ispy2_parpi_drugs)))
)
model_a = pooled[model_a_mask].copy()

train_mask = model_a['dataset'] != 'TCGA-OV'
test_mask = model_a['dataset'] == 'TCGA-OV'

X_train = np.nan_to_num(model_a.loc[train_mask, gene_cols].values, nan=0.5)
y_train = model_a.loc[train_mask, 'response_binary'].values.astype(int)
X_test = np.nan_to_num(model_a.loc[test_mask, gene_cols].values, nan=0.5)
test_ids = model_a.index[test_mask].astype(str).values

clf = LogisticRegression(
    penalty='l2', C=0.01, solver='lbfgs',
    max_iter=5000, class_weight='balanced', random_state=42
)
clf.fit(X_train, y_train)
pred_probs = clf.predict_proba(X_test)[:, 1]
pred_map = dict(zip(test_ids, pred_probs))
print(f"LODO-CV predictions for {len(pred_map)} TCGA-OV samples")

# Response labels
response_map = dict(zip(
    model_a.loc[test_mask].index.astype(str),
    model_a.loc[test_mask, 'response_binary'].values.astype(int)
))

# ============================================================
# STEP 2: Load HRD scores (Knijnenburg / GerkeLab)
# ============================================================
print("\n--- Step 2: Loading HRD scores from Knijnenburg et al. 2018 ---")

hrd_df = pd.read_csv('/tmp/DDRscores.txt', sep='\t', quotechar='"')
hrd_ov = hrd_df[hrd_df['acronym'] == 'OV'].copy()
# Create 12-char patient ID
hrd_ov['patient'] = hrd_ov['patient_id'].str[:12]
hrd_ov = hrd_ov.set_index('patient')
print(f"OV samples in DDRscores: {len(hrd_ov)}")

hrd_cols = ['HRD_TAI', 'HRD_LST', 'HRD_LOH', 'HRD_Score',
            'CNA_frac_altered', 'aneuploidy_score', 'purity', 'ploidy',
            'mutLoad_nonsilent', 'mutSig3']
for col in hrd_cols:
    hrd_ov[col] = pd.to_numeric(hrd_ov[col], errors='coerce')

# ============================================================
# STEP 3: Load survival/clinical data
# ============================================================
print("\n--- Step 3: Loading survival and clinical data ---")

tcga_clin = pd.read_parquet(BASE / 'tcga_ov_platinum_clinical.parquet')
tcga_clin.index = tcga_clin.index.astype(str)

# Xena clinical data for stage/grade/residual
url = 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.OV.sampleMap%2FOV_clinicalMatrix'
r = requests.get(url, timeout=60)
xena_df = pd.read_csv(io.BytesIO(r.content), sep='\t')
xena_df['patient'] = xena_df['bcr_patient_barcode']
xena_dedup = xena_df.drop_duplicates(subset='patient', keep='first').set_index('patient')
print(f"Xena clinical: {len(xena_dedup)} patients")

# BRCA labels
brca_path = Path('/tmp/ov_tcga_combined_brca.json')
if brca_path.exists():
    with open(brca_path) as f:
        brca_data = json.load(f)
else:
    brca_data = {}
    print("  WARNING: BRCA data not found, skipping BRCA covariate")

# ============================================================
# STEP 4: Build merged dataframe
# ============================================================
print("\n--- Step 4: Building merged analysis dataframe ---")

rows = []
for pid in tcga_clin.index:
    if pid not in pred_map:
        continue

    clin = tcga_clin.loc[pid]

    # Survival
    os_months = pd.to_numeric(clin.get('OS_MONTHS'), errors='coerce')
    os_event = 1 if '1:' in str(clin.get('OS_STATUS', '')) or 'DECEASED' in str(clin.get('OS_STATUS', '')) else 0
    dfs_months = pd.to_numeric(clin.get('DFS_MONTHS'), errors='coerce')
    dfs_event = 1 if '1:' in str(clin.get('DFS_STATUS', '')) or 'Recurred' in str(clin.get('DFS_STATUS', '')) else 0

    # Clinical from Xena
    xena = xena_dedup.loc[pid] if pid in xena_dedup.index else pd.Series(dtype=object)
    age = pd.to_numeric(xena.get('age_at_initial_pathologic_diagnosis'), errors='coerce') if len(xena) > 0 else np.nan
    stage_raw = str(xena.get('clinical_stage', '')) if len(xena) > 0 else ''
    grade_raw = str(xena.get('neoplasm_histologic_grade', '')) if len(xena) > 0 else ''
    residual_raw = str(xena.get('tumor_residual_disease', '')) if len(xena) > 0 else ''

    if 'III' in stage_raw or 'IV' in stage_raw:
        stage_advanced = 1
    elif 'I' in stage_raw or 'II' in stage_raw:
        stage_advanced = 0
    else:
        stage_advanced = np.nan

    if grade_raw in ('G3', 'G4'):
        grade_high = 1
    elif grade_raw in ('G1', 'G2'):
        grade_high = 0
    else:
        grade_high = np.nan

    if residual_raw == 'No Macroscopic disease':
        residual_optimal = 1
    elif residual_raw in ('1-10 mm', '11-20 mm', '>20 mm'):
        residual_optimal = 0
    else:
        residual_optimal = np.nan

    # HRD scores
    hrd_row = hrd_ov.loc[pid] if pid in hrd_ov.index else pd.Series(dtype=float)
    hrd_score = hrd_row.get('HRD_Score', np.nan) if len(hrd_row) > 0 else np.nan
    hrd_loh = hrd_row.get('HRD_LOH', np.nan) if len(hrd_row) > 0 else np.nan
    hrd_tai = hrd_row.get('HRD_TAI', np.nan) if len(hrd_row) > 0 else np.nan
    hrd_lst = hrd_row.get('HRD_LST', np.nan) if len(hrd_row) > 0 else np.nan
    fga = hrd_row.get('CNA_frac_altered', np.nan) if len(hrd_row) > 0 else np.nan
    aneuploidy = hrd_row.get('aneuploidy_score', np.nan) if len(hrd_row) > 0 else np.nan
    tumor_purity = hrd_row.get('purity', np.nan) if len(hrd_row) > 0 else np.nan
    mut_load = hrd_row.get('mutLoad_nonsilent', np.nan) if len(hrd_row) > 0 else np.nan
    mut_sig3 = hrd_row.get('mutSig3', np.nan) if len(hrd_row) > 0 else np.nan

    rows.append({
        'patient': pid,
        'score': pred_map[pid],
        'response': response_map[pid],
        'os_months': os_months,
        'os_event': os_event,
        'dfs_months': dfs_months,
        'dfs_event': dfs_event,
        'age': age,
        'stage_advanced': stage_advanced,
        'grade_high': grade_high,
        'residual_optimal': residual_optimal,
        'brca_mutated': 1 if pid in brca_data else 0,
        'hrd_score': hrd_score,
        'hrd_loh': hrd_loh,
        'hrd_tai': hrd_tai,
        'hrd_lst': hrd_lst,
        'fga': fga,
        'aneuploidy': aneuploidy,
        'tumor_purity': tumor_purity,
        'mut_load': mut_load,
        'mut_sig3': mut_sig3,
    })

df = pd.DataFrame(rows)
print(f"Merged dataframe: {len(df)} patients")

# Check HRD data availability
n_hrd = df['hrd_score'].notna().sum()
n_fga = df['fga'].notna().sum()
print(f"\nCopy number data coverage:")
print(f"  HRD_Score (LOH+TAI+LST): {n_hrd}/{len(df)} ({100*n_hrd/len(df):.1f}%)")
print(f"  CNA_frac_altered: {n_fga}/{len(df)} ({100*n_fga/len(df):.1f}%)")
for col in ['hrd_loh', 'hrd_tai', 'hrd_lst', 'aneuploidy', 'mut_sig3']:
    n = df[col].notna().sum()
    print(f"  {col}: {n}/{len(df)} ({100*n/len(df):.1f}%)")

results = {
    'experiment': 'Copy Number / HRD Score Covariate Analysis — TCGA-OV',
    'data_source': 'Knijnenburg et al. 2018 (PanCan DDR) via GerkeLab/TCGAhrd',
    'n_patients': len(df),
    'n_with_hrd': int(n_hrd),
    'n_with_fga': int(n_fga),
}

# ============================================================
# STEP 5: Correlation analysis
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS A: Correlation — RNA Score vs Copy Number Features")
print("=" * 70)

correlations = {}
for cn_col, cn_label in [
    ('hrd_score', 'HRD_Score (LOH+TAI+LST)'),
    ('hrd_loh', 'HRD_LOH'),
    ('hrd_tai', 'HRD_TAI'),
    ('hrd_lst', 'HRD_LST'),
    ('fga', 'CNA fraction altered'),
    ('aneuploidy', 'Aneuploidy score'),
    ('mut_sig3', 'Mut Signature 3 (BRCA-like)'),
    ('mut_load', 'Mutation load (nonsilent)'),
]:
    valid = df.dropna(subset=['score', cn_col])
    if len(valid) < 10:
        continue
    rho, p = stats.spearmanr(valid['score'], valid[cn_col])
    print(f"  Score vs {cn_label:35s}: rho={rho:+.3f}, p={p:.4f} (n={len(valid)})")
    correlations[cn_col] = {
        'label': cn_label,
        'spearman_rho': round(float(rho), 4),
        'spearman_p': round(float(p), 6),
        'n': int(len(valid)),
    }

results['correlations_score_vs_cn'] = correlations

# ============================================================
# STEP 6: Predictive value for response
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS B: CN Features as Predictors of Drug Response")
print("=" * 70)

response_prediction = {}
y_resp = df['response'].values

# RNA score AUC
valid_resp = df.dropna(subset=['response'])
rna_auc = roc_auc_score(valid_resp['response'], valid_resp['score'])
print(f"\n  RNA model score AUC: {rna_auc:.3f} (n={len(valid_resp)})")
response_prediction['rna_score'] = {
    'AUC': round(float(rna_auc), 4),
    'n': int(len(valid_resp)),
}

for cn_col, cn_label in [
    ('hrd_score', 'HRD_Score'),
    ('hrd_loh', 'HRD_LOH'),
    ('hrd_tai', 'HRD_TAI'),
    ('hrd_lst', 'HRD_LST'),
    ('fga', 'CNA fraction altered'),
    ('aneuploidy', 'Aneuploidy'),
    ('mut_sig3', 'Mut Sig 3'),
]:
    valid = df.dropna(subset=['response', cn_col])
    if len(valid) < 20:
        continue
    y = valid['response'].values
    x = valid[cn_col].values
    try:
        auc = roc_auc_score(y, x)
        # flip if AUC < 0.5 (higher CN -> resistance)
        direction = "higher=sensitive" if auc >= 0.5 else "higher=resistant"
        auc_reported = max(auc, 1 - auc)
    except Exception:
        continue

    # MW test
    sens = valid.loc[valid['response'] == 1, cn_col]
    res = valid.loc[valid['response'] == 0, cn_col]
    mw_stat, mw_p = stats.mannwhitneyu(sens, res, alternative='two-sided')

    print(f"  {cn_label:25s}: AUC={auc_reported:.3f} ({direction}), MW p={mw_p:.4f} (n={len(valid)}, sens_mean={sens.mean():.2f}, res_mean={res.mean():.2f})")
    response_prediction[cn_col] = {
        'label': cn_label,
        'AUC': round(float(auc_reported), 4),
        'direction': direction,
        'mannwhitney_p': round(float(mw_p), 6),
        'n': int(len(valid)),
        'sensitive_mean': round(float(sens.mean()), 4),
        'resistant_mean': round(float(res.mean()), 4),
    }

results['response_prediction'] = response_prediction

# ============================================================
# STEP 7: Logistic regression — independence tests
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS C: Independence Tests (Logistic Regression)")
print("=" * 70)

independence = {}

# Focus on HRD_Score as the primary CN covariate
valid_lr = df.dropna(subset=['response', 'hrd_score']).copy()
print(f"\n  Samples with both response and HRD_Score: {len(valid_lr)}")

if len(valid_lr) >= 30:
    from sklearn.preprocessing import StandardScaler

    y = valid_lr['response'].values.astype(int)
    score_std = StandardScaler().fit_transform(valid_lr[['score']].values)
    hrd_std = StandardScaler().fit_transform(valid_lr[['hrd_score']].values)

    # Model A: RNA score only
    lr_rna = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    lr_rna.fit(score_std, y)
    auc_rna = roc_auc_score(y, lr_rna.predict_proba(score_std)[:, 1])

    # Model B: HRD score only
    lr_hrd = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    lr_hrd.fit(hrd_std, y)
    auc_hrd = roc_auc_score(y, lr_hrd.predict_proba(hrd_std)[:, 1])

    # Model C: RNA + HRD combined
    X_both = np.hstack([score_std, hrd_std])
    lr_both = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    lr_both.fit(X_both, y)
    auc_both = roc_auc_score(y, lr_both.predict_proba(X_both)[:, 1])

    print(f"\n  Logistic regression AUCs (on same subset, n={len(valid_lr)}):")
    print(f"    RNA score only:  AUC = {auc_rna:.3f}")
    print(f"    HRD score only:  AUC = {auc_hrd:.3f}")
    print(f"    RNA + HRD:       AUC = {auc_both:.3f}")
    print(f"    Improvement:     +{auc_both - auc_rna:.3f} over RNA, +{auc_both - auc_hrd:.3f} over HRD")

    independence['logistic_regression'] = {
        'n': int(len(valid_lr)),
        'auc_rna_only': round(float(auc_rna), 4),
        'auc_hrd_only': round(float(auc_hrd), 4),
        'auc_rna_plus_hrd': round(float(auc_both), 4),
        'improvement_over_rna': round(float(auc_both - auc_rna), 4),
        'improvement_over_hrd': round(float(auc_both - auc_hrd), 4),
    }

    # Statsmodels logistic for formal LR test
    import statsmodels.api as sm

    # Model: response ~ score
    X_sm_rna = sm.add_constant(valid_lr[['score']].values)
    glm_rna = sm.GLM(y, X_sm_rna, family=sm.families.Binomial()).fit()

    # Model: response ~ hrd_score
    X_sm_hrd = sm.add_constant(valid_lr[['hrd_score']].values)
    glm_hrd = sm.GLM(y, X_sm_hrd, family=sm.families.Binomial()).fit()

    # Model: response ~ score + hrd_score
    X_sm_both = sm.add_constant(valid_lr[['score', 'hrd_score']].values)
    glm_both = sm.GLM(y, X_sm_both, family=sm.families.Binomial()).fit()

    # LR tests
    lr_stat_hrd_adds = -2 * (glm_rna.llf - glm_both.llf)
    p_hrd_adds = 1 - stats.chi2.cdf(lr_stat_hrd_adds, 1)

    lr_stat_rna_adds = -2 * (glm_hrd.llf - glm_both.llf)
    p_rna_adds = 1 - stats.chi2.cdf(lr_stat_rna_adds, 1)

    print(f"\n  Likelihood ratio tests:")
    print(f"    HRD adds to RNA model: LR stat={lr_stat_hrd_adds:.3f}, p={p_hrd_adds:.4f}")
    print(f"    RNA adds to HRD model: LR stat={lr_stat_rna_adds:.3f}, p={p_rna_adds:.4f}")
    print(f"\n  AIC comparison:")
    print(f"    RNA only:  AIC={glm_rna.aic:.1f}")
    print(f"    HRD only:  AIC={glm_hrd.aic:.1f}")
    print(f"    RNA + HRD: AIC={glm_both.aic:.1f}")

    independence['likelihood_ratio_tests'] = {
        'hrd_adds_to_rna': {
            'lr_statistic': round(float(lr_stat_hrd_adds), 4),
            'p_value': round(float(p_hrd_adds), 6),
        },
        'rna_adds_to_hrd': {
            'lr_statistic': round(float(lr_stat_rna_adds), 4),
            'p_value': round(float(p_rna_adds), 6),
        },
        'aic_rna_only': round(float(glm_rna.aic), 2),
        'aic_hrd_only': round(float(glm_hrd.aic), 2),
        'aic_rna_plus_hrd': round(float(glm_both.aic), 2),
    }

    # Also test with FGA
    valid_fga = df.dropna(subset=['response', 'fga']).copy()
    if len(valid_fga) >= 30:
        y_fga = valid_fga['response'].values.astype(int)
        X_fga_sm = sm.add_constant(valid_fga[['score', 'fga']].values)
        glm_fga = sm.GLM(y_fga, X_fga_sm, family=sm.families.Binomial()).fit()
        X_rna_only_fga = sm.add_constant(valid_fga[['score']].values)
        glm_rna_fga = sm.GLM(y_fga, X_rna_only_fga, family=sm.families.Binomial()).fit()

        lr_fga = -2 * (glm_rna_fga.llf - glm_fga.llf)
        p_fga = 1 - stats.chi2.cdf(lr_fga, 1)
        print(f"\n  FGA adds to RNA model: LR stat={lr_fga:.3f}, p={p_fga:.4f} (n={len(valid_fga)})")

        independence['fga_adds_to_rna'] = {
            'lr_statistic': round(float(lr_fga), 4),
            'p_value': round(float(p_fga), 6),
            'n': int(len(valid_fga)),
        }

results['independence_tests'] = independence

# ============================================================
# STEP 8: Cox PH survival models with CN covariates
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS D: Cox PH Survival Models with Copy Number Covariates")
print("=" * 70)


def fit_cox_model(data, time_col, event_col, covariates, model_name, penalizer=0.01):
    cox_df = data[[time_col, event_col] + covariates].dropna().copy()
    cox_df.columns = ['time', 'event'] + covariates
    cox_df = cox_df[cox_df['time'] > 0].copy()
    n = len(cox_df)
    n_events = int(cox_df['event'].sum())
    if n < 20 or n_events < 5:
        return {'error': f'Too few (n={n}, events={n_events})', 'n': n, 'n_events': n_events}, None
    try:
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(cox_df, duration_col='time', event_col='event')
        result = {
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
            result['covariates'][var] = {
                'HR': round(hr, 4),
                'HR_CI_low': round(ci_low, 4),
                'HR_CI_high': round(ci_high, 4),
                'p_value': round(p_val, 6),
            }
        return result, cph
    except Exception as e:
        return {'error': str(e), 'n': n, 'n_events': n_events}, None


survival_results = {}

for endpoint, time_col, event_col in [
    ('DFS', 'dfs_months', 'dfs_event'),
    ('OS', 'os_months', 'os_event'),
]:
    print(f"\n  --- {endpoint} ---")
    valid = df[df[time_col].notna() & (df[time_col] > 0)].copy()
    n_events = int(valid[event_col].sum())
    print(f"  Valid patients: {len(valid)} ({n_events} events)")

    ep = {}

    # IMPORTANT: For LR tests, all models must use the same patient subset.
    # Restrict to patients with HRD data for all pairwise comparisons.
    valid_hrd = valid.dropna(subset=['hrd_score']).copy()
    n_hrd_valid = len(valid_hrd)
    n_hrd_events = int(valid_hrd[event_col].sum())
    print(f"  Patients with HRD data: {n_hrd_valid} ({n_hrd_events} events)")

    # --- All-patients models (for reporting) ---
    # Model 1a: Score only (all patients)
    m1a, _ = fit_cox_model(valid, time_col, event_col, ['score'], 'score_only_all', penalizer=0.0)
    if 'covariates' in m1a:
        s = m1a['covariates']['score']
        print(f"  Score only (all, n={m1a['n']}): HR={s['HR']:.4f}, p={s['p_value']:.6f}, C={m1a['concordance_index']:.4f}")
    ep['score_only_all_patients'] = m1a

    # --- HRD-subset models (for valid LR tests) ---
    # Model 1: Score only (HRD subset)
    m1, c1 = fit_cox_model(valid_hrd, time_col, event_col, ['score'], 'score_only', penalizer=0.0)
    if 'covariates' in m1:
        s = m1['covariates']['score']
        print(f"  Score only (HRD subset, n={m1['n']}): HR={s['HR']:.4f}, p={s['p_value']:.6f}, C={m1['concordance_index']:.4f}")
    ep['score_only'] = m1

    # Model 2: HRD_Score only
    m2, c2 = fit_cox_model(valid_hrd, time_col, event_col, ['hrd_score'], 'hrd_only', penalizer=0.0)
    if 'covariates' in m2:
        s = m2['covariates']['hrd_score']
        print(f"  HRD only (n={m2['n']}):        HR={s['HR']:.4f}, p={s['p_value']:.6f}, C={m2['concordance_index']:.4f}")
    ep['hrd_only'] = m2

    # Model 3: Score + HRD_Score
    m3, c3 = fit_cox_model(valid_hrd, time_col, event_col, ['score', 'hrd_score'], 'score_plus_hrd', penalizer=0.0)
    if 'covariates' in m3:
        for var in ['score', 'hrd_score']:
            v = m3['covariates'][var]
            sig = "***" if v['p_value'] < 0.001 else "**" if v['p_value'] < 0.01 else "*" if v['p_value'] < 0.05 else ""
            print(f"  Score+HRD {var:12s}: HR={v['HR']:.4f}, p={v['p_value']:.6f} {sig}, C={m3['concordance_index']:.4f}")
    ep['score_plus_hrd'] = m3

    # Model 4: Score + clinical + HRD_Score
    covars_full = ['score', 'stage_advanced', 'grade_high', 'residual_optimal', 'age', 'hrd_score']
    m4, c4 = fit_cox_model(valid_hrd, time_col, event_col, covars_full, 'full_with_hrd', penalizer=0.0)
    if 'covariates' in m4:
        print(f"  Full model (score + clinical + HRD, n={m4['n']}):")
        for var in covars_full:
            if var in m4['covariates']:
                v = m4['covariates'][var]
                sig = "***" if v['p_value'] < 0.001 else "**" if v['p_value'] < 0.01 else "*" if v['p_value'] < 0.05 else ""
                print(f"    {var:25s}: HR={v['HR']:.4f} (CI: {v['HR_CI_low']:.4f}-{v['HR_CI_high']:.4f}), p={v['p_value']:.6f} {sig}")
        print(f"    C-index: {m4['concordance_index']:.4f}")
    ep['full_with_hrd'] = m4

    # Model 5: Clinical + HRD (no RNA score)
    covars_no_rna = ['stage_advanced', 'grade_high', 'residual_optimal', 'age', 'hrd_score']
    m5, c5 = fit_cox_model(valid_hrd, time_col, event_col, covars_no_rna, 'clinical_plus_hrd_no_rna', penalizer=0.0)
    if 'covariates' in m5:
        print(f"  Clinical+HRD (no RNA, n={m5['n']}): C={m5['concordance_index']:.4f}")
    ep['clinical_plus_hrd_no_rna'] = m5

    # LR tests (all on same HRD subset — valid comparisons)
    if all('log_likelihood' in ep[k] for k in ['score_plus_hrd', 'score_only', 'hrd_only']):
        lr_hrd_adds = -2 * (ep['score_only']['log_likelihood'] - ep['score_plus_hrd']['log_likelihood'])
        p_hrd_adds = 1 - stats.chi2.cdf(lr_hrd_adds, 1)
        lr_rna_adds = -2 * (ep['hrd_only']['log_likelihood'] - ep['score_plus_hrd']['log_likelihood'])
        p_rna_adds = 1 - stats.chi2.cdf(lr_rna_adds, 1)
        print(f"\n  LR test (same n={ep['score_plus_hrd']['n']}):")
        print(f"    HRD adds to RNA score: chi2={lr_hrd_adds:.3f}, p={p_hrd_adds:.4f}")
        print(f"    RNA adds to HRD score: chi2={lr_rna_adds:.3f}, p={p_rna_adds:.4f}")
        ep['lr_test_hrd_adds_to_rna'] = {'chi2': round(float(lr_hrd_adds), 4), 'p': round(float(p_hrd_adds), 6)}
        ep['lr_test_rna_adds_to_hrd'] = {'chi2': round(float(lr_rna_adds), 4), 'p': round(float(p_rna_adds), 6)}

    if all('log_likelihood' in ep[k] for k in ['full_with_hrd', 'clinical_plus_hrd_no_rna']):
        ll_full = ep['full_with_hrd']['log_likelihood']
        ll_no_rna = ep['clinical_plus_hrd_no_rna']['log_likelihood']
        lr_rna_adds_full = -2 * (ll_no_rna - ll_full)
        p_rna_adds_full = 1 - stats.chi2.cdf(lr_rna_adds_full, 1)
        print(f"    RNA adds to clinical+HRD: chi2={lr_rna_adds_full:.3f}, p={p_rna_adds_full:.4f}")
        ep['lr_test_rna_adds_to_clinical_plus_hrd'] = {'chi2': round(float(lr_rna_adds_full), 4), 'p': round(float(p_rna_adds_full), 6)}

    survival_results[endpoint] = ep

results['survival_models'] = survival_results

# ============================================================
# STEP 9: HRD score vs response stratified by BRCA
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS E: HRD Score vs Response Stratified by BRCA Status")
print("=" * 70)

stratified = {}
for brca_val, brca_label in [(0, 'BRCA-wildtype'), (1, 'BRCA-mutated')]:
    sub = df[(df['brca_mutated'] == brca_val) & df['hrd_score'].notna() & df['response'].notna()]
    if len(sub) < 10:
        print(f"  {brca_label}: too few samples (n={len(sub)})")
        continue

    sens = sub[sub['response'] == 1]
    res = sub[sub['response'] == 0]
    print(f"\n  {brca_label} (n={len(sub)}, sens={len(sens)}, res={len(res)}):")

    for cn_col, cn_label in [('hrd_score', 'HRD_Score'), ('fga', 'FGA')]:
        valid_sub = sub.dropna(subset=[cn_col])
        if len(valid_sub) < 10:
            continue
        s = valid_sub[valid_sub['response'] == 1][cn_col]
        r = valid_sub[valid_sub['response'] == 0][cn_col]
        if len(s) > 3 and len(r) > 3:
            mw, mp = stats.mannwhitneyu(s, r, alternative='two-sided')
            try:
                auc = roc_auc_score(valid_sub['response'], valid_sub[cn_col])
            except Exception:
                auc = np.nan
            print(f"    {cn_label}: sens_mean={s.mean():.2f}, res_mean={r.mean():.2f}, MW p={mp:.4f}, AUC={auc:.3f}")
            stratified[f'{brca_label}_{cn_col}'] = {
                'n': int(len(valid_sub)),
                'sensitive_mean': round(float(s.mean()), 4),
                'resistant_mean': round(float(r.mean()), 4),
                'mannwhitney_p': round(float(mp), 6),
                'AUC': round(float(auc), 4),
            }

    # RNA score in this BRCA stratum
    rna_sub = sub.dropna(subset=['score'])
    if len(rna_sub) >= 10:
        try:
            auc_rna = roc_auc_score(rna_sub['response'], rna_sub['score'])
        except Exception:
            auc_rna = np.nan
        print(f"    RNA score: AUC={auc_rna:.3f}")
        stratified[f'{brca_label}_rna_score'] = {
            'n': int(len(rna_sub)),
            'AUC': round(float(auc_rna), 4),
        }

results['stratified_by_brca'] = stratified

# ============================================================
# STEP 10: Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n  TCGA-OV patients: {len(df)}")
print(f"  Patients with HRD scores: {n_hrd}")
print(f"  RNA model AUC: {rna_auc:.3f}")
if 'logistic_regression' in independence:
    lr = independence['logistic_regression']
    print(f"  HRD-only AUC: {lr['auc_hrd_only']:.3f}")
    print(f"  RNA+HRD AUC: {lr['auc_rna_plus_hrd']:.3f}")
if 'likelihood_ratio_tests' in independence:
    lrt = independence['likelihood_ratio_tests']
    print(f"  HRD adds to RNA: p={lrt['hrd_adds_to_rna']['p_value']:.4f}")
    print(f"  RNA adds to HRD: p={lrt['rna_adds_to_hrd']['p_value']:.4f}")

# Write results
with open(OUT / 'copynumber_covariate_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults written to {OUT / 'copynumber_covariate_results.json'}")
