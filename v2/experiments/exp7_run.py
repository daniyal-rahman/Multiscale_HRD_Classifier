#!/usr/bin/env python3
"""
Experiment 7: GDSC Olaparib IC50 Validation for HRD Transcriptomic Classifier

Goal: Apply our BRCA-trained HRD model to GDSC cell lines and test whether
predicted HRD score correlates with olaparib sensitivity (IC50/AUC) and
BRCA mutation status.

Steps:
  1. Train ElasticNet on BRCA VST + tiered labels (also rank-based model)
  2. Prepare GDSC expression (rank-transform for cross-platform compatibility)
  3. Match cell lines (olaparib IC50 + expression)
  4. Apply model; evaluate correlations, BRCA prediction, drug sensitivity
  5. Tissue-stratified analysis
  6. Compare across PARPi drugs

Usage:
    srun --mem=8G --time=00:30:00 /home/dani/miniconda3/envs/ML/bin/python exp7_run.py
"""

import json
import os
import sys
import warnings
from functools import partial
from pathlib import Path

# Force unbuffered output for slurm
print = partial(print, flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
EXP5_DIR = REPO_ROOT / "v2" / "experiments" / "exp5_normalization"
GDSC_DIR = REPO_ROOT / "v2" / "data_acquisition" / "processed" / "gdsc"
RESULTS_DIR = Path(__file__).resolve().parent / "exp7_gdsc_olaparib"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "v2"))
from label_engineering.tiered_labels import TieredHRDLabeler


# =============================================================================
# 1. Load TCGA-BRCA data and create tiered labels
# =============================================================================

def load_tcga_data():
    """Load BRCA expression (VST + ranks), HRD scores, BRCA status."""
    print("=" * 70)
    print("Step 1: Loading TCGA-BRCA data and creating tiered labels")
    print("=" * 70)

    # Expression
    vst = pd.read_parquet(EXP5_DIR / "tcga_brca_vst.parquet")
    ranks = pd.read_parquet(EXP5_DIR / "tcga_brca_ranks.parquet")
    print(f"VST expression: {vst.shape}")
    print(f"Rank expression: {ranks.shape}")

    # Common genes (cross-platform)
    with open(EXP5_DIR / "common_genes.txt") as f:
        common_genes = [g.strip() for g in f.readlines()]
    print(f"Cross-platform common genes: {len(common_genes)}")

    # DD500 genes
    dd500 = pd.read_csv(EXP5_DIR / "dd500_genes.csv")
    dd500_genes = dd500["gene"].tolist()
    print(f"DD500 DE genes: {len(dd500_genes)}")

    # HRD scores
    hrd = pd.read_excel(DATA_DIR / "tcga.hrdscore.xlsx")
    print(f"HRD scores: {hrd.shape[0]} samples")

    # BRCA status
    brca = pd.read_csv(DATA_DIR / "toga.breast.brca.status.txt", sep="\t", index_col=0)
    brca.index = brca.index.str.replace(".", "-", regex=False)
    print(f"BRCA status: {brca.shape[0]} samples")

    return vst, ranks, common_genes, dd500_genes, hrd, brca


def create_tiered_labels(hrd, brca):
    """Create tiered labels using TieredHRDLabeler."""
    labeler = TieredHRDLabeler(
        gis_positive_threshold=42,
        gis_negative_threshold=20,
    )
    tiered = labeler.label_tcga_cohort(brca, hrd)
    print(f"\nTiered labels:")
    print(f"  Tier 1 (gold HRD+): {(tiered['tier'] == 1).sum()}")
    print(f"  Tier 2 (gold HRP):  {(tiered['tier'] == 2).sum()}")
    print(f"  Tier 3 (ambiguous): {(tiered['tier'] == 3).sum()}")
    return tiered


def prepare_training_data(expr_df, tiered, gene_list, label="vst"):
    """Prepare X, y for training from tiered labels."""
    # Use only common genes present in expression
    available_genes = [g for g in gene_list if g in expr_df.columns]
    print(f"\n  [{label}] Available genes from list: {len(available_genes)}/{len(gene_list)}")

    X = expr_df[available_genes]

    # Align with tiered labels
    common_samples = X.index.intersection(tiered.index)
    X = X.loc[common_samples]
    tier_aligned = tiered.loc[common_samples]

    # Tier 1 + Tier 2 only
    tier12_mask = tier_aligned["tier"].isin([1, 2])
    tier12_samples = tier_aligned[tier12_mask].index
    X_train = X.loc[tier12_samples]
    y_train = (tier_aligned.loc[tier12_samples, "label"] == "HRD").astype(int)

    print(f"  [{label}] Training: {len(X_train)} samples "
          f"(HRD={y_train.sum()}, HRP={len(y_train)-y_train.sum()})")
    return X_train, y_train, available_genes


# =============================================================================
# 2. Train models
# =============================================================================

def train_model(X, y, model_name="model"):
    """Train ElasticNet logistic regression with CV for evaluation, then final model."""
    print(f"\n--- Training {model_name} ---")

    # Cross-validation first
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    fold_aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr = scaler.fit_transform(X.iloc[train_idx])
        X_te = scaler.transform(X.iloc[test_idx])
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]

        model = LogisticRegression(
            penalty="elasticnet", solver="saga",
            l1_ratio=0.5, max_iter=5000, tol=1e-3,
            C=1.0, random_state=42,
        )
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        fold_auc = roc_auc_score(y_te, y_prob)
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold+1}: AUC={fold_auc:.4f}")

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    print(f"  Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

    # Train final model on all data
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    final_model = LogisticRegression(
        penalty="elasticnet", solver="saga",
        l1_ratio=0.5, max_iter=5000, tol=1e-3,
        C=1.0, random_state=42,
    )
    final_model.fit(X_scaled, y)

    n_nonzero = np.sum(final_model.coef_[0] != 0)
    print(f"  Final model: {n_nonzero} non-zero features out of {X.shape[1]}")

    return final_model, final_scaler, {
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "fold_aucs": fold_aucs,
        "n_nonzero_features": int(n_nonzero),
    }


# =============================================================================
# 3. Prepare GDSC data
# =============================================================================

def load_gdsc_data():
    """Load GDSC expression, olaparib IC50, metadata."""
    print("\n" + "=" * 70)
    print("Step 2-3: Loading and preparing GDSC data")
    print("=" * 70)

    # Expression
    expr = pd.read_parquet(GDSC_DIR / "expression.parquet")
    print(f"GDSC expression: {expr.shape}")

    # Olaparib IC50 (enriched with ModelID + BRCA status)
    ola = pd.read_parquet(GDSC_DIR / "olaparib_ic50_enriched.parquet")
    print(f"Olaparib IC50 (enriched): {ola.shape}")

    # PARPi IC50 (all 3 drugs)
    parpi = pd.read_parquet(GDSC_DIR / "parpi_ic50.parquet")
    print(f"PARPi IC50 (all drugs): {parpi.shape}")

    # Cell line metadata
    meta = pd.read_parquet(GDSC_DIR / "cell_line_metadata.parquet")
    print(f"Cell line metadata: {meta.shape}")

    # BRCA status
    brca = pd.read_parquet(GDSC_DIR / "brca_status.parquet")
    print(f"BRCA status: {brca.shape}")

    return expr, ola, parpi, meta, brca


def prepare_gdsc_expression(gdsc_expr, gene_list, method="rank"):
    """Prepare GDSC expression for cross-platform prediction.

    method="rank": rank-transform within each sample (recommended for cross-platform)
    method="standardize": standardize within each sample (zero mean, unit variance)
    """
    # Restrict to genes in gene_list
    available = [g for g in gene_list if g in gdsc_expr.columns]
    missing = [g for g in gene_list if g not in gdsc_expr.columns]
    print(f"\n  [{method}] GDSC genes available: {len(available)}/{len(gene_list)}")
    print(f"  [{method}] Missing genes: {len(missing)}")

    X = gdsc_expr[available].copy()

    # Fill missing genes with 0 (neutral)
    for g in missing:
        X[g] = 0.0

    # Reorder to match gene_list
    X = X[gene_list]

    if method == "rank":
        # Rank-transform within each sample (row), same as TCGA ranks
        X = X.rank(axis=1, method='average')
        # Normalize ranks to [0, 1]
        n_genes = X.shape[1]
        X = X / n_genes
    elif method == "standardize":
        # Standardize within each sample
        row_mean = X.mean(axis=1)
        row_std = X.std(axis=1)
        row_std = row_std.replace(0, 1)  # avoid division by zero
        X = X.sub(row_mean, axis=0).div(row_std, axis=0)

    print(f"  [{method}] Prepared GDSC expression: {X.shape}")
    return X


def match_cell_lines(gdsc_expr, ola_df, meta_df, brca_df):
    """Match cell lines across expression, olaparib IC50, and metadata."""
    print("\n--- Matching cell lines ---")

    # Expression indices (DepMap ModelIDs)
    expr_models = set(gdsc_expr.index)
    print(f"  Expression models: {len(expr_models)}")

    # Olaparib has ModelID column (some may be NaN)
    ola_with_model = ola_df[ola_df['ModelID'].notna()].copy()
    ola_models = set(ola_with_model['ModelID'])
    print(f"  Olaparib models (with ModelID): {len(ola_models)}")

    # Intersection
    common_models = expr_models & ola_models
    print(f"  Intersection (expression + olaparib): {len(common_models)}")

    # Build matched dataframe
    # Deduplicate olaparib by ModelID (some may have duplicates)
    ola_matched = ola_with_model.drop_duplicates(subset='ModelID', keep='first')
    ola_matched = ola_matched[ola_matched['ModelID'].isin(common_models)]
    ola_matched = ola_matched.set_index('ModelID')

    # Add metadata (tissue type)
    if 'ModelID' in meta_df.columns:
        meta_indexed = meta_df.set_index('ModelID')
    else:
        meta_indexed = meta_df

    meta_cols = ['OncotreeLineage', 'OncotreePrimaryDisease', 'OncotreeSubtype',
                 'CellLineName', 'StrippedCellLineName']
    available_meta_cols = [c for c in meta_cols if c in meta_indexed.columns]
    ola_matched = ola_matched.join(meta_indexed[available_meta_cols], how='left')

    # Report BRCA mutation status
    n_brca_any = ola_matched['BRCA_any_mutated'].sum() if 'BRCA_any_mutated' in ola_matched.columns else 0
    n_brca1 = ola_matched['BRCA1_mutated'].sum() if 'BRCA1_mutated' in ola_matched.columns else 0
    n_brca2 = ola_matched['BRCA2_mutated'].sum() if 'BRCA2_mutated' in ola_matched.columns else 0
    print(f"\n  Matched cell lines: {len(ola_matched)}")
    print(f"  BRCA1 mutated: {n_brca1}")
    print(f"  BRCA2 mutated: {n_brca2}")
    print(f"  Any BRCA mutated: {n_brca_any}")

    if 'OncotreeLineage' in ola_matched.columns:
        print(f"\n  Top tissue types:")
        for tissue, count in ola_matched['OncotreeLineage'].value_counts().head(10).items():
            print(f"    {tissue}: {count}")

    return ola_matched


# =============================================================================
# 4. Apply model and evaluate
# =============================================================================

def apply_model_to_gdsc(model, scaler, gene_list, gdsc_expr_prepared, ola_matched, model_name="Rank"):
    """Apply trained model to GDSC cell lines."""
    print(f"\n--- Applying {model_name} model to GDSC ---")

    # Get expression for matched cell lines
    common_models = ola_matched.index.intersection(gdsc_expr_prepared.index)
    X_gdsc = gdsc_expr_prepared.loc[common_models]
    ola_data = ola_matched.loc[common_models]

    print(f"  Predicting on {len(X_gdsc)} cell lines")

    # Scale using the TCGA-trained scaler
    X_scaled = scaler.transform(X_gdsc)

    # Predict
    pred_prob = model.predict_proba(X_scaled)[:, 1]
    pred_class = model.predict(X_scaled)

    ola_data = ola_data.copy()
    ola_data['pred_hrd_prob'] = pred_prob
    ola_data['pred_hrd_class'] = pred_class

    print(f"  Predicted HRD+: {pred_class.sum()}/{len(pred_class)}")
    print(f"  Predicted HRP:  {(1 - pred_class).sum()}/{len(pred_class)}")
    print(f"  HRD prob range: [{pred_prob.min():.3f}, {pred_prob.max():.3f}], "
          f"median={np.median(pred_prob):.3f}")

    return ola_data


def evaluate_correlations(ola_data, model_name="Rank"):
    """Evaluate correlations between predicted HRD score and drug sensitivity."""
    print(f"\n{'=' * 70}")
    print(f"Step 4a: Correlation analysis ({model_name})")
    print(f"{'=' * 70}")

    results = {}

    # Overall correlation
    for metric in ['LN_IC50', 'AUC']:
        valid = ola_data[['pred_hrd_prob', metric]].dropna()
        rho, p = spearmanr(valid['pred_hrd_prob'], valid[metric])
        r_pear, p_pear = pearsonr(valid['pred_hrd_prob'], valid[metric])
        print(f"\n  {metric} (n={len(valid)}):")
        print(f"    Spearman rho={rho:.4f}, p={p:.2e}")
        print(f"    Pearson  r={r_pear:.4f}, p={p_pear:.2e}")
        results[f'all_{metric}_spearman_rho'] = float(rho)
        results[f'all_{metric}_spearman_p'] = float(p)
        results[f'all_{metric}_pearson_r'] = float(r_pear)
        results[f'all_{metric}_pearson_p'] = float(p_pear)
        results[f'all_{metric}_n'] = int(len(valid))

    # Tissue-specific correlations
    if 'OncotreeLineage' in ola_data.columns:
        for tissue in ['Breast', 'Ovary/Fallopian Tube']:
            tissue_data = ola_data[ola_data['OncotreeLineage'] == tissue]
            if len(tissue_data) >= 10:
                for metric in ['LN_IC50', 'AUC']:
                    valid = tissue_data[['pred_hrd_prob', metric]].dropna()
                    if len(valid) >= 5:
                        rho, p = spearmanr(valid['pred_hrd_prob'], valid[metric])
                        tissue_key = tissue.replace('/', '_').replace(' ', '_')
                        print(f"\n  {tissue} {metric} (n={len(valid)}):")
                        print(f"    Spearman rho={rho:.4f}, p={p:.2e}")
                        results[f'{tissue_key}_{metric}_spearman_rho'] = float(rho)
                        results[f'{tissue_key}_{metric}_spearman_p'] = float(p)
                        results[f'{tissue_key}_{metric}_n'] = int(len(valid))

    return results


def evaluate_brca_prediction(ola_data, model_name="Rank"):
    """Evaluate model's ability to predict BRCA mutation status."""
    print(f"\n{'=' * 70}")
    print(f"Step 4b: BRCA mutation prediction ({model_name})")
    print(f"{'=' * 70}")

    results = {}

    if 'BRCA_any_mutated' not in ola_data.columns:
        print("  No BRCA status available")
        return results

    valid = ola_data[['pred_hrd_prob', 'BRCA_any_mutated']].dropna()
    y_true = valid['BRCA_any_mutated'].astype(int)
    y_score = valid['pred_hrd_prob']

    if y_true.sum() < 2 or (1 - y_true).sum() < 2:
        print("  Too few BRCA-mutated or wild-type samples")
        return results

    # AUC
    brca_auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Optimal threshold (Youden's J)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_sens = tpr[optimal_idx]
    optimal_spec = 1 - fpr[optimal_idx]

    print(f"\n  BRCA any mutation prediction:")
    print(f"    AUC = {brca_auc:.4f}")
    print(f"    Optimal threshold = {optimal_threshold:.3f}")
    print(f"    Sensitivity = {optimal_sens:.3f}")
    print(f"    Specificity = {optimal_spec:.3f}")
    print(f"    n_mutated = {y_true.sum()}, n_WT = {(1 - y_true).sum()}")

    results['brca_any_auc'] = float(brca_auc)
    results['brca_any_optimal_threshold'] = float(optimal_threshold)
    results['brca_any_sensitivity'] = float(optimal_sens)
    results['brca_any_specificity'] = float(optimal_spec)
    results['brca_any_n_mutated'] = int(y_true.sum())
    results['brca_any_n_wt'] = int((1 - y_true).sum())
    results['brca_fpr'] = fpr.tolist()
    results['brca_tpr'] = tpr.tolist()

    # Also try BRCA1 and BRCA2 separately
    for gene in ['BRCA1_mutated', 'BRCA2_mutated']:
        if gene in ola_data.columns:
            valid_g = ola_data[['pred_hrd_prob', gene]].dropna()
            y_true_g = valid_g[gene].astype(int)
            if y_true_g.sum() >= 2 and (1 - y_true_g).sum() >= 2:
                gene_auc = roc_auc_score(y_true_g, valid_g['pred_hrd_prob'])
                gene_key = gene.replace('_mutated', '')
                print(f"    {gene_key}: AUC = {gene_auc:.4f} (n_mut={y_true_g.sum()})")
                results[f'{gene_key}_auc'] = float(gene_auc)

    return results


def evaluate_drug_sensitivity_groups(ola_data, model_name="Rank"):
    """Compare drug sensitivity between predicted HRD+ and HRP groups."""
    print(f"\n{'=' * 70}")
    print(f"Step 4c: Drug sensitivity by predicted HRD status ({model_name})")
    print(f"{'=' * 70}")

    results = {}

    # Split at median
    median_prob = ola_data['pred_hrd_prob'].median()
    ola_data = ola_data.copy()
    ola_data['pred_hrd_group'] = np.where(
        ola_data['pred_hrd_prob'] >= median_prob, 'pred_HRD', 'pred_HRP'
    )

    hrd_group = ola_data[ola_data['pred_hrd_group'] == 'pred_HRD']
    hrp_group = ola_data[ola_data['pred_hrd_group'] == 'pred_HRP']

    print(f"\n  Split at median ({median_prob:.3f}):")
    print(f"    Predicted HRD+: n={len(hrd_group)}")
    print(f"    Predicted HRP:  n={len(hrp_group)}")

    for metric in ['LN_IC50', 'AUC']:
        hrd_vals = hrd_group[metric].dropna()
        hrp_vals = hrp_group[metric].dropna()
        if len(hrd_vals) > 1 and len(hrp_vals) > 1:
            u_stat, p_val = mannwhitneyu(hrd_vals, hrp_vals, alternative='two-sided')
            # Also test one-sided (HRD should have lower IC50 = more sensitive)
            if metric == 'LN_IC50':
                _, p_one = mannwhitneyu(hrd_vals, hrp_vals, alternative='less')
            else:
                _, p_one = mannwhitneyu(hrd_vals, hrp_vals, alternative='less')

            print(f"\n    {metric}:")
            print(f"      pred_HRD: mean={hrd_vals.mean():.3f}, median={hrd_vals.median():.3f}")
            print(f"      pred_HRP: mean={hrp_vals.mean():.3f}, median={hrp_vals.median():.3f}")
            print(f"      Mann-Whitney U: U={u_stat:.1f}, p(two-sided)={p_val:.2e}, p(one-sided)={p_one:.2e}")

            results[f'median_split_{metric}_hrd_mean'] = float(hrd_vals.mean())
            results[f'median_split_{metric}_hrp_mean'] = float(hrp_vals.mean())
            results[f'median_split_{metric}_hrd_median'] = float(hrd_vals.median())
            results[f'median_split_{metric}_hrp_median'] = float(hrp_vals.median())
            results[f'median_split_{metric}_mwu_p_twosided'] = float(p_val)
            results[f'median_split_{metric}_mwu_p_onesided'] = float(p_one)

    # Also test by BRCA mutation status
    if 'BRCA_any_mutated' in ola_data.columns:
        brca_mut = ola_data[ola_data['BRCA_any_mutated'] == True]
        brca_wt = ola_data[ola_data['BRCA_any_mutated'] == False]
        for metric in ['LN_IC50', 'AUC']:
            mut_vals = brca_mut[metric].dropna()
            wt_vals = brca_wt[metric].dropna()
            if len(mut_vals) > 1 and len(wt_vals) > 1:
                u_stat, p_val = mannwhitneyu(mut_vals, wt_vals, alternative='two-sided')
                print(f"\n    BRCA mutated vs WT {metric}:")
                print(f"      Mutated: mean={mut_vals.mean():.3f}, n={len(mut_vals)}")
                print(f"      WT:      mean={wt_vals.mean():.3f}, n={len(wt_vals)}")
                print(f"      Mann-Whitney p={p_val:.2e}")
                results[f'brca_mut_vs_wt_{metric}_p'] = float(p_val)
                results[f'brca_mut_{metric}_mean'] = float(mut_vals.mean())
                results[f'brca_wt_{metric}_mean'] = float(wt_vals.mean())

    results['median_threshold'] = float(median_prob)
    return results


def evaluate_parpi_drugs(gdsc_expr_prepared, model, scaler, parpi_df, meta_df, brca_df, model_name="Rank"):
    """Compare correlations across PARPi drugs."""
    print(f"\n{'=' * 70}")
    print(f"Step 4d: PARPi drug comparison ({model_name})")
    print(f"{'=' * 70}")

    results = {}

    # Map parpi cell lines to DepMap ModelIDs
    # Need COSMIC_ID -> ModelID mapping from metadata
    if 'COSMICID' in meta_df.columns and 'ModelID' in meta_df.columns:
        # Drop rows with NaN COSMICID to avoid duplicate index issues
        valid_meta = meta_df[meta_df['COSMICID'].notna()].drop_duplicates(subset='COSMICID', keep='first')
        cosmic_to_model = dict(zip(valid_meta['COSMICID'].astype(int), valid_meta['ModelID']))
    else:
        print("  Cannot map COSMIC to ModelID, skipping PARPi comparison")
        return results

    for drug_name in ['Olaparib', 'Talazoparib', 'Rucaparib']:
        drug_data = parpi_df[parpi_df['DRUG_NAME'] == drug_name].copy()
        print(f"\n  {drug_name}: {len(drug_data)} cell lines in GDSC")

        # Map to ModelID
        drug_data['ModelID'] = drug_data['COSMIC_ID'].map(cosmic_to_model)
        drug_data = drug_data[drug_data['ModelID'].notna()]
        drug_data = drug_data.drop_duplicates(subset='ModelID', keep='first')
        drug_data = drug_data.set_index('ModelID')

        # Intersect with expression
        common = gdsc_expr_prepared.index.intersection(drug_data.index)
        if len(common) < 10:
            print(f"    Only {len(common)} matched lines, skipping")
            continue

        X = gdsc_expr_prepared.loc[common]
        X_scaled = scaler.transform(X)
        pred_prob = model.predict_proba(X_scaled)[:, 1]

        drug_matched = drug_data.loc[common].copy()
        drug_matched['pred_hrd_prob'] = pred_prob

        for metric in ['LN_IC50', 'AUC']:
            valid = drug_matched[['pred_hrd_prob', metric]].dropna()
            if len(valid) >= 10:
                rho, p = spearmanr(valid['pred_hrd_prob'], valid[metric])
                drug_key = drug_name.lower()
                print(f"    {metric}: Spearman rho={rho:.4f}, p={p:.2e} (n={len(valid)})")
                results[f'{drug_key}_{metric}_spearman_rho'] = float(rho)
                results[f'{drug_key}_{metric}_spearman_p'] = float(p)
                results[f'{drug_key}_{metric}_n'] = int(len(valid))

    return results


# =============================================================================
# 5. Tissue-stratified analysis
# =============================================================================

def tissue_stratified_analysis(ola_data, model_name="Rank"):
    """Analyze results by tissue type."""
    print(f"\n{'=' * 70}")
    print(f"Step 5: Tissue-stratified analysis ({model_name})")
    print(f"{'=' * 70}")

    results = {}

    if 'OncotreeLineage' not in ola_data.columns:
        print("  No tissue information available")
        return results

    tissue_counts = ola_data['OncotreeLineage'].value_counts()
    # Analyze tissues with >= 15 cell lines
    for tissue in tissue_counts[tissue_counts >= 15].index:
        tissue_data = ola_data[ola_data['OncotreeLineage'] == tissue]
        tissue_key = tissue.replace('/', '_').replace(' ', '_')

        entry = {'n': int(len(tissue_data))}

        # Correlation with LN_IC50
        valid = tissue_data[['pred_hrd_prob', 'LN_IC50']].dropna()
        if len(valid) >= 10:
            rho, p = spearmanr(valid['pred_hrd_prob'], valid['LN_IC50'])
            entry['LN_IC50_spearman_rho'] = float(rho)
            entry['LN_IC50_spearman_p'] = float(p)
            print(f"\n  {tissue} (n={len(tissue_data)}):")
            print(f"    LN_IC50: rho={rho:.4f}, p={p:.2e}")

        # Correlation with AUC
        valid_auc = tissue_data[['pred_hrd_prob', 'AUC']].dropna()
        if len(valid_auc) >= 10:
            rho_auc, p_auc = spearmanr(valid_auc['pred_hrd_prob'], valid_auc['AUC'])
            entry['AUC_spearman_rho'] = float(rho_auc)
            entry['AUC_spearman_p'] = float(p_auc)
            print(f"    AUC:     rho={rho_auc:.4f}, p={p_auc:.2e}")

        # BRCA status if available
        if 'BRCA_any_mutated' in tissue_data.columns:
            n_brca = tissue_data['BRCA_any_mutated'].sum()
            entry['n_brca_mutated'] = int(n_brca)

        # Mean predicted HRD prob
        entry['mean_pred_hrd_prob'] = float(tissue_data['pred_hrd_prob'].mean())
        entry['median_pred_hrd_prob'] = float(tissue_data['pred_hrd_prob'].median())

        results[tissue_key] = entry

    return results


# =============================================================================
# 6. Plots
# =============================================================================

def plot_scatter_hrd_vs_ic50_brca(ola_data, model_name="Rank"):
    """Scatter: predicted HRD score vs olaparib LN_IC50, colored by BRCA status."""
    fig, ax = plt.subplots(figsize=(10, 7))

    brca_mut = ola_data[ola_data.get('BRCA_any_mutated', False) == True]
    brca_wt = ola_data[ola_data.get('BRCA_any_mutated', False) == False]

    ax.scatter(brca_wt['pred_hrd_prob'], brca_wt['LN_IC50'],
               alpha=0.3, s=20, c='gray', label=f'BRCA WT (n={len(brca_wt)})', edgecolors='none')
    ax.scatter(brca_mut['pred_hrd_prob'], brca_mut['LN_IC50'],
               alpha=0.7, s=40, c='red', label=f'BRCA mutated (n={len(brca_mut)})',
               edgecolors='darkred', linewidth=0.5)

    # Add trend line
    valid = ola_data[['pred_hrd_prob', 'LN_IC50']].dropna()
    rho, p = spearmanr(valid['pred_hrd_prob'], valid['LN_IC50'])
    z = np.polyfit(valid['pred_hrd_prob'], valid['LN_IC50'], 1)
    poly = np.poly1d(z)
    x_range = np.linspace(valid['pred_hrd_prob'].min(), valid['pred_hrd_prob'].max(), 100)
    ax.plot(x_range, poly(x_range), 'b--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Predicted HRD Probability', fontsize=13)
    ax.set_ylabel('Olaparib LN_IC50 (lower = more sensitive)', fontsize=13)
    ax.set_title(f'{model_name} Model: HRD Score vs Olaparib Sensitivity\n'
                 f'Spearman rho={rho:.3f}, p={p:.2e}', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f'scatter_hrd_vs_ic50_brca_{model_name.lower()}.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: scatter_hrd_vs_ic50_brca_{model_name.lower()}.png")


def plot_scatter_hrd_vs_ic50_tissue(ola_data, model_name="Rank"):
    """Scatter: predicted HRD score vs olaparib LN_IC50, colored by tissue type."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get top tissues
    top_tissues = ola_data['OncotreeLineage'].value_counts().head(8).index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_tissues)))

    other = ola_data[~ola_data['OncotreeLineage'].isin(top_tissues)]
    ax.scatter(other['pred_hrd_prob'], other['LN_IC50'],
               alpha=0.2, s=15, c='lightgray', label='Other', edgecolors='none')

    for tissue, color in zip(top_tissues, colors):
        t_data = ola_data[ola_data['OncotreeLineage'] == tissue]
        ax.scatter(t_data['pred_hrd_prob'], t_data['LN_IC50'],
                   alpha=0.6, s=30, c=[color], label=f'{tissue} (n={len(t_data)})',
                   edgecolors='none')

    valid = ola_data[['pred_hrd_prob', 'LN_IC50']].dropna()
    rho, p = spearmanr(valid['pred_hrd_prob'], valid['LN_IC50'])

    ax.set_xlabel('Predicted HRD Probability', fontsize=13)
    ax.set_ylabel('Olaparib LN_IC50 (lower = more sensitive)', fontsize=13)
    ax.set_title(f'{model_name} Model: HRD Score vs Olaparib Sensitivity by Tissue\n'
                 f'Spearman rho={rho:.3f}, p={p:.2e}', fontsize=14)
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f'scatter_hrd_vs_ic50_tissue_{model_name.lower()}.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: scatter_hrd_vs_ic50_tissue_{model_name.lower()}.png")


def plot_boxplots_hrd_groups(ola_data, model_name="Rank"):
    """Box plots: olaparib IC50 in predicted-HRD+ vs HRP and BRCA-mutated vs WT."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    median_prob = ola_data['pred_hrd_prob'].median()

    # Panel 1: Predicted HRD+ vs HRP (LN_IC50)
    ax = axes[0]
    hrd_group = ola_data[ola_data['pred_hrd_prob'] >= median_prob]['LN_IC50'].dropna()
    hrp_group = ola_data[ola_data['pred_hrd_prob'] < median_prob]['LN_IC50'].dropna()
    _, p_val = mannwhitneyu(hrd_group, hrp_group, alternative='two-sided')

    bp = ax.boxplot([hrp_group.values, hrd_group.values],
                    labels=[f'Pred HRP\n(n={len(hrp_group)})', f'Pred HRD+\n(n={len(hrd_group)})'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#c7d4e8')
    bp['boxes'][1].set_facecolor('#f4a582')
    for i, data in enumerate([hrp_group.values, hrd_group.values], 1):
        jitter = np.random.RandomState(42).uniform(-0.08, 0.08, size=len(data))
        ax.scatter(np.full_like(data, i) + jitter, data, alpha=0.3, s=8, c='black', zorder=3)
    ax.set_ylabel('Olaparib LN_IC50', fontsize=12)
    ax.set_title(f'By Predicted HRD Status\nMann-Whitney p={p_val:.2e}', fontsize=11)

    # Panel 2: BRCA mutated vs WT (LN_IC50)
    ax = axes[1]
    if 'BRCA_any_mutated' in ola_data.columns:
        brca_mut = ola_data[ola_data['BRCA_any_mutated'] == True]['LN_IC50'].dropna()
        brca_wt = ola_data[ola_data['BRCA_any_mutated'] == False]['LN_IC50'].dropna()
        _, p_brca = mannwhitneyu(brca_mut, brca_wt, alternative='two-sided')
        bp2 = ax.boxplot([brca_wt.values, brca_mut.values],
                         labels=[f'BRCA WT\n(n={len(brca_wt)})', f'BRCA mut\n(n={len(brca_mut)})'],
                         patch_artist=True, widths=0.5)
        bp2['boxes'][0].set_facecolor('#c7d4e8')
        bp2['boxes'][1].set_facecolor('#f4a582')
        for i, data in enumerate([brca_wt.values, brca_mut.values], 1):
            jitter = np.random.RandomState(42).uniform(-0.08, 0.08, size=len(data))
            ax.scatter(np.full_like(data, i) + jitter, data, alpha=0.3, s=8, c='black', zorder=3)
        ax.set_ylabel('Olaparib LN_IC50', fontsize=12)
        ax.set_title(f'By BRCA Mutation Status\nMann-Whitney p={p_brca:.2e}', fontsize=11)
    else:
        ax.text(0.5, 0.5, 'No BRCA data', ha='center', va='center', transform=ax.transAxes)

    # Panel 3: Predicted HRD+ vs HRP (AUC)
    ax = axes[2]
    hrd_auc = ola_data[ola_data['pred_hrd_prob'] >= median_prob]['AUC'].dropna()
    hrp_auc = ola_data[ola_data['pred_hrd_prob'] < median_prob]['AUC'].dropna()
    _, p_auc = mannwhitneyu(hrd_auc, hrp_auc, alternative='two-sided')
    bp3 = ax.boxplot([hrp_auc.values, hrd_auc.values],
                     labels=[f'Pred HRP\n(n={len(hrp_auc)})', f'Pred HRD+\n(n={len(hrd_auc)})'],
                     patch_artist=True, widths=0.5)
    bp3['boxes'][0].set_facecolor('#c7d4e8')
    bp3['boxes'][1].set_facecolor('#f4a582')
    for i, data in enumerate([hrp_auc.values, hrd_auc.values], 1):
        jitter = np.random.RandomState(42).uniform(-0.08, 0.08, size=len(data))
        ax.scatter(np.full_like(data, i) + jitter, data, alpha=0.3, s=8, c='black', zorder=3)
    ax.set_ylabel('Olaparib AUC', fontsize=12)
    ax.set_title(f'By Predicted HRD Status\nMann-Whitney p={p_auc:.2e}', fontsize=11)

    fig.suptitle(f'{model_name} Model: Olaparib Sensitivity by Group', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f'boxplots_hrd_groups_{model_name.lower()}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: boxplots_hrd_groups_{model_name.lower()}.png")


def plot_roc_brca(ola_data, model_name="Rank"):
    """ROC curve: predicted HRD score vs BRCA mutation status."""
    if 'BRCA_any_mutated' not in ola_data.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    valid = ola_data[['pred_hrd_prob', 'BRCA_any_mutated', 'BRCA1_mutated', 'BRCA2_mutated']].dropna()

    for col, label, color in [
        ('BRCA_any_mutated', 'Any BRCA', 'tab:blue'),
        ('BRCA1_mutated', 'BRCA1', 'tab:red'),
        ('BRCA2_mutated', 'BRCA2', 'tab:green'),
    ]:
        y_true = valid[col].astype(int)
        if y_true.sum() >= 2 and (1 - y_true).sum() >= 2:
            fpr, tpr, _ = roc_curve(y_true, valid['pred_hrd_prob'])
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f'{label}: AUC={auc_val:.3f} (n_mut={y_true.sum()})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(f'{model_name} Model: ROC for BRCA Mutation Prediction', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f'roc_brca_{model_name.lower()}.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: roc_brca_{model_name.lower()}.png")


def plot_parpi_comparison(parpi_results, model_name="Rank"):
    """Bar chart: Spearman correlations across PARPi drugs."""
    if not parpi_results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    drugs = []
    rhos_ic50 = []
    rhos_auc = []
    for drug in ['olaparib', 'talazoparib', 'rucaparib']:
        key_ic50 = f'{drug}_LN_IC50_spearman_rho'
        key_auc = f'{drug}_AUC_spearman_rho'
        if key_ic50 in parpi_results:
            drugs.append(drug.capitalize())
            rhos_ic50.append(parpi_results[key_ic50])
            rhos_auc.append(parpi_results.get(key_auc, 0))

    if not drugs:
        plt.close(fig)
        return

    x = np.arange(len(drugs))
    width = 0.35

    bars1 = ax.bar(x - width/2, rhos_ic50, width, label='LN_IC50', color='#4575b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, rhos_auc, width, label='AUC', color='#d73027', alpha=0.8)

    ax.set_ylabel('Spearman Correlation (rho)', fontsize=13)
    ax.set_xlabel('PARPi Drug', fontsize=13)
    ax.set_title(f'{model_name} Model: HRD Score Correlation with PARPi Sensitivity', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(drugs, fontsize=12)
    ax.legend(fontsize=11)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f'parpi_comparison_{model_name.lower()}.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: parpi_comparison_{model_name.lower()}.png")


def plot_tissue_correlations(tissue_results, model_name="Rank"):
    """Bar chart: Spearman correlations by tissue type."""
    if not tissue_results:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    tissues = []
    rhos = []
    ns = []
    pvals = []

    for tissue_key, entry in sorted(tissue_results.items(), key=lambda x: x[1].get('LN_IC50_spearman_rho', 0)):
        if 'LN_IC50_spearman_rho' in entry:
            tissues.append(tissue_key.replace('_', ' '))
            rhos.append(entry['LN_IC50_spearman_rho'])
            ns.append(entry['n'])
            pvals.append(entry.get('LN_IC50_spearman_p', 1.0))

    if not tissues:
        plt.close(fig)
        return

    colors = ['#d73027' if p < 0.05 else '#4575b4' if p < 0.1 else 'gray' for p in pvals]

    y_pos = np.arange(len(tissues))
    bars = ax.barh(y_pos, rhos, color=colors, alpha=0.8, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{t} (n={n})' for t, n in zip(tissues, ns)], fontsize=10)
    ax.set_xlabel('Spearman Correlation (HRD score vs LN_IC50)', fontsize=13)
    ax.set_title(f'{model_name} Model: Tissue-Stratified HRD-Olaparib Correlation\n'
                 f'(Red=p<0.05, Blue=p<0.10, Gray=n.s.)', fontsize=13)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f'tissue_correlations_{model_name.lower()}.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: tissue_correlations_{model_name.lower()}.png")


def plot_summary_figure(ola_data_rank, ola_data_vst, model_name="combined"):
    """Create a combined summary figure."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Scatter HRD vs IC50 (rank model), colored by BRCA
    ax = fig.add_subplot(gs[0, 0])
    if 'BRCA_any_mutated' in ola_data_rank.columns:
        brca_mut = ola_data_rank[ola_data_rank['BRCA_any_mutated'] == True]
        brca_wt = ola_data_rank[ola_data_rank['BRCA_any_mutated'] == False]
        ax.scatter(brca_wt['pred_hrd_prob'], brca_wt['LN_IC50'],
                   alpha=0.3, s=12, c='gray', label=f'WT (n={len(brca_wt)})', edgecolors='none')
        ax.scatter(brca_mut['pred_hrd_prob'], brca_mut['LN_IC50'],
                   alpha=0.7, s=25, c='red', label=f'BRCA mut (n={len(brca_mut)})',
                   edgecolors='darkred', linewidth=0.3)
    valid = ola_data_rank[['pred_hrd_prob', 'LN_IC50']].dropna()
    rho, p = spearmanr(valid['pred_hrd_prob'], valid['LN_IC50'])
    ax.set_xlabel('Predicted HRD Prob', fontsize=10)
    ax.set_ylabel('Olaparib LN_IC50', fontsize=10)
    ax.set_title(f'A) Rank Model: rho={rho:.3f}, p={p:.1e}', fontsize=11)
    ax.legend(fontsize=8)

    # Panel B: Scatter HRD vs IC50 (vst model)
    ax = fig.add_subplot(gs[0, 1])
    if 'BRCA_any_mutated' in ola_data_vst.columns:
        brca_mut = ola_data_vst[ola_data_vst['BRCA_any_mutated'] == True]
        brca_wt = ola_data_vst[ola_data_vst['BRCA_any_mutated'] == False]
        ax.scatter(brca_wt['pred_hrd_prob'], brca_wt['LN_IC50'],
                   alpha=0.3, s=12, c='gray', label=f'WT (n={len(brca_wt)})', edgecolors='none')
        ax.scatter(brca_mut['pred_hrd_prob'], brca_mut['LN_IC50'],
                   alpha=0.7, s=25, c='red', label=f'BRCA mut (n={len(brca_mut)})',
                   edgecolors='darkred', linewidth=0.3)
    valid_vst = ola_data_vst[['pred_hrd_prob', 'LN_IC50']].dropna()
    rho_vst, p_vst = spearmanr(valid_vst['pred_hrd_prob'], valid_vst['LN_IC50'])
    ax.set_xlabel('Predicted HRD Prob', fontsize=10)
    ax.set_ylabel('Olaparib LN_IC50', fontsize=10)
    ax.set_title(f'B) VST Model: rho={rho_vst:.3f}, p={p_vst:.1e}', fontsize=11)
    ax.legend(fontsize=8)

    # Panel C: ROC for BRCA prediction (rank model)
    ax = fig.add_subplot(gs[0, 2])
    if 'BRCA_any_mutated' in ola_data_rank.columns:
        v = ola_data_rank[['pred_hrd_prob', 'BRCA_any_mutated']].dropna()
        y_true = v['BRCA_any_mutated'].astype(int)
        if y_true.sum() >= 2:
            fpr, tpr, _ = roc_curve(y_true, v['pred_hrd_prob'])
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, 'b-', lw=2, label=f'AUC={auc_val:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            ax.legend(fontsize=10)
    ax.set_xlabel('FPR', fontsize=10)
    ax.set_ylabel('TPR', fontsize=10)
    ax.set_title('C) ROC: BRCA Mutation (Rank)', fontsize=11)

    # Panel D: Boxplot HRD vs HRP (LN_IC50)
    ax = fig.add_subplot(gs[1, 0])
    median_prob = ola_data_rank['pred_hrd_prob'].median()
    hrd_ic50 = ola_data_rank[ola_data_rank['pred_hrd_prob'] >= median_prob]['LN_IC50'].dropna()
    hrp_ic50 = ola_data_rank[ola_data_rank['pred_hrd_prob'] < median_prob]['LN_IC50'].dropna()
    _, p_mw = mannwhitneyu(hrd_ic50, hrp_ic50, alternative='two-sided')
    bp = ax.boxplot([hrp_ic50.values, hrd_ic50.values],
                    labels=[f'Pred HRP\n(n={len(hrp_ic50)})', f'Pred HRD+\n(n={len(hrd_ic50)})'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#c7d4e8')
    bp['boxes'][1].set_facecolor('#f4a582')
    ax.set_ylabel('Olaparib LN_IC50', fontsize=10)
    ax.set_title(f'D) LN_IC50 by Pred HRD\np={p_mw:.2e}', fontsize=11)

    # Panel E: Boxplot BRCA mut vs WT
    ax = fig.add_subplot(gs[1, 1])
    if 'BRCA_any_mutated' in ola_data_rank.columns:
        brca_mut_ic50 = ola_data_rank[ola_data_rank['BRCA_any_mutated'] == True]['LN_IC50'].dropna()
        brca_wt_ic50 = ola_data_rank[ola_data_rank['BRCA_any_mutated'] == False]['LN_IC50'].dropna()
        if len(brca_mut_ic50) > 1 and len(brca_wt_ic50) > 1:
            _, p_brca = mannwhitneyu(brca_mut_ic50, brca_wt_ic50, alternative='two-sided')
            bp2 = ax.boxplot([brca_wt_ic50.values, brca_mut_ic50.values],
                             labels=[f'BRCA WT\n(n={len(brca_wt_ic50)})',
                                     f'BRCA mut\n(n={len(brca_mut_ic50)})'],
                             patch_artist=True, widths=0.5)
            bp2['boxes'][0].set_facecolor('#c7d4e8')
            bp2['boxes'][1].set_facecolor('#f4a582')
            ax.set_ylabel('Olaparib LN_IC50', fontsize=10)
            ax.set_title(f'E) LN_IC50 by BRCA Status\np={p_brca:.2e}', fontsize=11)

    # Panel F: HRD distribution by tissue (top tissues)
    ax = fig.add_subplot(gs[1, 2])
    if 'OncotreeLineage' in ola_data_rank.columns:
        top_tissues = ola_data_rank['OncotreeLineage'].value_counts().head(6).index
        tissue_data = []
        tissue_labels = []
        for t in top_tissues:
            vals = ola_data_rank[ola_data_rank['OncotreeLineage'] == t]['pred_hrd_prob']
            tissue_data.append(vals.values)
            tissue_labels.append(f'{t[:15]}\n(n={len(vals)})')
        bp3 = ax.boxplot(tissue_data, labels=tissue_labels, patch_artist=True,
                         widths=0.6)
        colors = plt.cm.Set2(np.linspace(0, 1, len(tissue_data)))
        for patch, color in zip(bp3['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_ylabel('Predicted HRD Prob', fontsize=10)
        ax.set_title('F) HRD Score by Tissue', fontsize=11)
        ax.tick_params(axis='x', labelsize=7)

    fig.suptitle('Experiment 7: GDSC Olaparib IC50 Validation', fontsize=15, fontweight='bold', y=1.01)
    fig.savefig(RESULTS_DIR / 'exp7_summary_figure.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: exp7_summary_figure.png")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("EXPERIMENT 7: GDSC Olaparib IC50 Validation")
    print("=" * 70)

    # ---- Step 1: Load TCGA data, create labels, train models ----
    vst, ranks, common_genes, dd500_genes, hrd, brca = load_tcga_data()
    tiered = create_tiered_labels(hrd, brca)

    # Prepare training data - RANK model (use common genes for cross-platform)
    X_rank, y_rank, rank_genes = prepare_training_data(
        ranks, tiered, common_genes, label="rank"
    )
    rank_model, rank_scaler, rank_cv = train_model(X_rank, y_rank, "Rank")

    # Prepare training data - VST model (same common genes)
    X_vst, y_vst, vst_genes = prepare_training_data(
        vst, tiered, common_genes, label="vst"
    )
    vst_model, vst_scaler, vst_cv = train_model(X_vst, y_vst, "VST")

    # ---- Step 2-3: Load GDSC data, prepare expression, match cell lines ----
    gdsc_expr, ola_enriched, parpi, meta, gdsc_brca = load_gdsc_data()

    # Prepare GDSC expression for both models
    gdsc_rank = prepare_gdsc_expression(gdsc_expr, rank_genes, method="rank")
    gdsc_vst_std = prepare_gdsc_expression(gdsc_expr, vst_genes, method="standardize")

    # Match cell lines
    ola_matched = match_cell_lines(gdsc_expr, ola_enriched, meta, gdsc_brca)

    # ---- Step 4: Apply models and evaluate ----

    # --- Rank model ---
    print("\n" + "=" * 70)
    print("RANK MODEL EVALUATION")
    print("=" * 70)

    ola_rank = apply_model_to_gdsc(rank_model, rank_scaler, rank_genes,
                                    gdsc_rank, ola_matched, "Rank")
    corr_rank = evaluate_correlations(ola_rank, "Rank")
    brca_pred_rank = evaluate_brca_prediction(ola_rank, "Rank")
    groups_rank = evaluate_drug_sensitivity_groups(ola_rank, "Rank")
    parpi_rank = evaluate_parpi_drugs(gdsc_rank, rank_model, rank_scaler,
                                       parpi, meta, gdsc_brca, "Rank")
    tissue_rank = tissue_stratified_analysis(ola_rank, "Rank")

    # --- VST model ---
    print("\n" + "=" * 70)
    print("VST MODEL EVALUATION")
    print("=" * 70)

    ola_vst = apply_model_to_gdsc(vst_model, vst_scaler, vst_genes,
                                   gdsc_vst_std, ola_matched, "VST")
    corr_vst = evaluate_correlations(ola_vst, "VST")
    brca_pred_vst = evaluate_brca_prediction(ola_vst, "VST")
    groups_vst = evaluate_drug_sensitivity_groups(ola_vst, "VST")
    parpi_vst = evaluate_parpi_drugs(gdsc_vst_std, vst_model, vst_scaler,
                                      parpi, meta, gdsc_brca, "VST")
    tissue_vst = tissue_stratified_analysis(ola_vst, "VST")

    # ---- Step 6: Generate plots ----
    print("\n" + "=" * 70)
    print("Step 6: Generating plots")
    print("=" * 70)

    plot_scatter_hrd_vs_ic50_brca(ola_rank, "Rank")
    plot_scatter_hrd_vs_ic50_brca(ola_vst, "VST")
    plot_scatter_hrd_vs_ic50_tissue(ola_rank, "Rank")
    plot_scatter_hrd_vs_ic50_tissue(ola_vst, "VST")
    plot_boxplots_hrd_groups(ola_rank, "Rank")
    plot_boxplots_hrd_groups(ola_vst, "VST")
    plot_roc_brca(ola_rank, "Rank")
    plot_roc_brca(ola_vst, "VST")
    plot_parpi_comparison(parpi_rank, "Rank")
    plot_parpi_comparison(parpi_vst, "VST")
    plot_tissue_correlations(tissue_rank, "Rank")
    plot_tissue_correlations(tissue_vst, "VST")
    plot_summary_figure(ola_rank, ola_vst)

    # ---- Save results JSON ----
    print("\n" + "=" * 70)
    print("Saving results")
    print("=" * 70)

    results = {
        "experiment": "exp7_gdsc_olaparib_validation",
        "description": "Apply BRCA-trained HRD classifier to GDSC cell lines; "
                       "test correlation with olaparib sensitivity and BRCA mutation status",
        "n_matched_cell_lines": int(len(ola_matched)),
        "rank_model": {
            "training_cv": rank_cv,
            "n_genes": len(rank_genes),
            "correlations": corr_rank,
            "brca_prediction": {k: v for k, v in brca_pred_rank.items()
                                if not isinstance(v, list)},
            "group_comparisons": groups_rank,
            "parpi_comparison": parpi_rank,
            "tissue_analysis": tissue_rank,
        },
        "vst_model": {
            "training_cv": vst_cv,
            "n_genes": len(vst_genes),
            "correlations": corr_vst,
            "brca_prediction": {k: v for k, v in brca_pred_vst.items()
                                if not isinstance(v, list)},
            "group_comparisons": groups_vst,
            "parpi_comparison": parpi_vst,
            "tissue_analysis": tissue_vst,
        },
        "caveats": [
            "Cell lines lack tumor microenvironment -- tests cell-intrinsic HRD signal",
            "GDSC expression (DepMap log2(TPM+1)) differs from TCGA VST normalization",
            "Rank-based model is expected to be more robust to platform differences",
            "BRCA mutation status from DepMap may include heterozygous (non-biallelic) mutations",
            "LN_IC50 is natural log of IC50 in micromolar; lower = more sensitive",
        ],
    }

    json_path = RESULTS_DIR / "exp7_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {json_path}")

    # ---- Save predictions for notebook ----
    ola_rank.to_parquet(RESULTS_DIR / "predictions_rank.parquet")
    ola_vst.to_parquet(RESULTS_DIR / "predictions_vst.parquet")
    print(f"Saved prediction parquet files")

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 7 SUMMARY")
    print("=" * 70)

    print(f"\nMatched cell lines: {len(ola_matched)}")

    print(f"\nRank model (5-fold CV on BRCA): AUC = {rank_cv['mean_auc']:.4f} +/- {rank_cv['std_auc']:.4f}")
    print(f"VST model  (5-fold CV on BRCA): AUC = {vst_cv['mean_auc']:.4f} +/- {vst_cv['std_auc']:.4f}")

    print(f"\nRank model GDSC results:")
    print(f"  All: LN_IC50 rho={corr_rank.get('all_LN_IC50_spearman_rho', 'N/A'):.4f}, "
          f"p={corr_rank.get('all_LN_IC50_spearman_p', 'N/A'):.2e}")
    print(f"  BRCA prediction AUC={brca_pred_rank.get('brca_any_auc', 'N/A')}")
    print(f"  Group diff (LN_IC50): p={groups_rank.get('median_split_LN_IC50_mwu_p_twosided', 'N/A'):.2e}")

    print(f"\nVST model GDSC results:")
    print(f"  All: LN_IC50 rho={corr_vst.get('all_LN_IC50_spearman_rho', 'N/A'):.4f}, "
          f"p={corr_vst.get('all_LN_IC50_spearman_p', 'N/A'):.2e}")
    print(f"  BRCA prediction AUC={brca_pred_vst.get('brca_any_auc', 'N/A')}")
    print(f"  Group diff (LN_IC50): p={groups_vst.get('median_split_LN_IC50_mwu_p_twosided', 'N/A'):.2e}")

    # Interpretation
    rank_rho = corr_rank.get('all_LN_IC50_spearman_rho', 0)
    vst_rho = corr_vst.get('all_LN_IC50_spearman_rho', 0)

    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")

    if rank_rho < -0.1 and corr_rank.get('all_LN_IC50_spearman_p', 1) < 0.05:
        print("POSITIVE: Rank model shows significant negative correlation between")
        print("predicted HRD score and olaparib IC50 (higher HRD -> lower IC50 -> more sensitive)")
    elif rank_rho < 0 and corr_rank.get('all_LN_IC50_spearman_p', 1) < 0.05:
        print("WEAK POSITIVE: Rank model shows weak but significant negative correlation")
    elif rank_rho > 0:
        print("UNEXPECTED: Positive correlation (higher HRD score -> HIGHER IC50 -> LESS sensitive)")
        print("This could indicate tissue-of-origin confounding or model overfit to BRCA tumor biology")
    else:
        print("NO SIGNIFICANT CORRELATION detected between HRD score and olaparib sensitivity")

    better_model = "Rank" if abs(rank_rho) > abs(vst_rho) else "VST"
    print(f"\nBetter cross-platform model: {better_model} "
          f"(rank rho={rank_rho:.4f}, VST rho={vst_rho:.4f})")

    print("\nKey caveats:")
    print("1. Cell lines lack TME -- only cell-intrinsic HRD signal tested")
    print("2. Cross-platform normalization may attenuate true signal")
    print("3. BRCA mutations in DepMap may not all cause functional HRD")
    print("4. Pan-cancer analysis confounds tissue-specific drug sensitivity")

    print(f"\nDone! Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
