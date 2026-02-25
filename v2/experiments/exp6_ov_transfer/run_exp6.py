#!/usr/bin/env python3
"""
Experiment 6: OV Zero-Shot Transfer with Confound Decomposition
================================================================
Tests BRCA-trained HRD classifier on TCGA-OV with and without
correcting for tumor purity, proliferation, and immune infiltration.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# ============================================================
# Paths
# ============================================================
REPO = Path('/home/dani/repos2/Multiscale_HRD_Classifier')
EXP5 = REPO / 'v2' / 'experiments' / 'exp5_normalization'
OUT  = REPO / 'v2' / 'experiments' / 'exp6_ov_transfer'
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / 'v2'))
from label_engineering.tiered_labels import TieredHRDLabeler

# ============================================================
# Load Data
# ============================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

# Expression data
brca_vst = pd.read_parquet(EXP5 / 'tcga_brca_vst.parquet')
ov_vst   = pd.read_parquet(EXP5 / 'tcga_ov_vst.parquet')
brca_ranks = pd.read_parquet(EXP5 / 'tcga_brca_ranks.parquet')
ov_ranks   = pd.read_parquet(EXP5 / 'tcga_ov_ranks.parquet')

# Gene lists
with open(EXP5 / 'common_genes.txt') as f:
    common_genes = [l.strip() for l in f if l.strip()]
dd500 = pd.read_csv(EXP5 / 'dd500_genes.csv')
dd500_genes = dd500['gene'].tolist()

# HRD scores
brca_hrd_df = pd.read_excel(REPO / 'data' / 'tcga.hrdscore.xlsx')
brca_status_df = pd.read_csv(REPO / 'data' / 'toga.breast.brca.status.txt', sep='\t')
ov_hrd_df = pd.read_parquet(EXP5 / 'tcga_ov_hrd_scores.parquet')

# DDR footprints (purity, mutational signatures, etc.)
ddr_fp = pd.read_excel(EXP5 / 'TCGA_DDR_Data_Resources.xlsx', sheet_name='DDR footprints')
# Parse DDR footprints: skip header rows (0,1,2), data starts at row 3
ddr_data = ddr_fp.iloc[3:].copy()
ddr_data.columns = ['patient_barcode', 'sample_barcode', 'disease'] + ddr_fp.columns[3:].tolist()
ddr_data = ddr_data.reset_index(drop=True)
# Convert numeric columns
for col in ddr_data.columns[3:]:
    ddr_data[col] = pd.to_numeric(ddr_data[col], errors='coerce')

# Split by cancer type
ddr_brca = ddr_data[ddr_data['disease'] == 'BRCA'].copy()
ddr_brca = ddr_brca.set_index('patient_barcode')
ddr_ov = ddr_data[ddr_data['disease'] == 'OV'].copy()
ddr_ov = ddr_ov.set_index('patient_barcode')

print(f"BRCA VST: {brca_vst.shape}")
print(f"OV VST: {ov_vst.shape}")
print(f"Common genes: {len(common_genes)}")
print(f"DD500 genes: {len(dd500_genes)}")
print(f"DDR footprints BRCA: {ddr_brca.shape}, OV: {ddr_ov.shape}")

# DDR gene-level alteration data for OV
ddr_alt = pd.read_excel(EXP5 / 'TCGA_DDR_Data_Resources.xlsx', sheet_name='DDR gene alterations')
ddr_mut = pd.read_excel(EXP5 / 'TCGA_DDR_Data_Resources.xlsx', sheet_name='DDR gene mutations')
ddr_del = pd.read_excel(EXP5 / 'TCGA_DDR_Data_Resources.xlsx', sheet_name='DDR deep deletions')
ddr_sil = pd.read_excel(EXP5 / 'TCGA_DDR_Data_Resources.xlsx', sheet_name='DDR epigenetic silencing')

def get_ddr_ov_gene_status(sheet_df, gene_name):
    """Get OV samples with alteration for a given gene from DDR sheet."""
    type_row = sheet_df.iloc[0, 2:]
    ov_cols = [col for col, val in type_row.items() if str(val) == 'OV']
    gene_symbols = sheet_df.iloc[:, 1].tolist()
    if gene_name in gene_symbols:
        idx = gene_symbols.index(gene_name)
        row = sheet_df.iloc[idx]
        ov_vals = row[ov_cols]
        altered = ov_vals[(ov_vals.astype(str) != '0') & (ov_vals.astype(str) != 'nan')]
        # Map column names (sample barcodes) to patient barcodes
        return set(s[:12] for s in altered.index.tolist())
    return set()

# OV BRCA1/2 status from DDR resource
ov_brca1_mut = get_ddr_ov_gene_status(ddr_mut, 'BRCA1')
ov_brca1_del = get_ddr_ov_gene_status(ddr_del, 'BRCA1')
ov_brca1_sil = get_ddr_ov_gene_status(ddr_sil, 'BRCA1')
ov_brca2_mut = get_ddr_ov_gene_status(ddr_mut, 'BRCA2')
ov_brca2_del = get_ddr_ov_gene_status(ddr_del, 'BRCA2')
ov_brca2_sil = get_ddr_ov_gene_status(ddr_sil, 'BRCA2')

ov_brca1_all = ov_brca1_mut | ov_brca1_del | ov_brca1_sil
ov_brca2_all = ov_brca2_mut | ov_brca2_del | ov_brca2_sil

print(f"\nOV BRCA1 alterations: {len(ov_brca1_all)} (mut={len(ov_brca1_mut)}, del={len(ov_brca1_del)}, sil={len(ov_brca1_sil)})")
print(f"OV BRCA2 alterations: {len(ov_brca2_all)} (mut={len(ov_brca2_mut)}, del={len(ov_brca2_del)}, sil={len(ov_brca2_sil)})")

# ============================================================
# STEP 1: Compute tiered HRD labels for TCGA-OV
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: TIERED OV LABELS")
print("=" * 70)

# Prepare OV HRD scores with patient barcodes as index
ov_hrd = ov_hrd_df.copy()
ov_hrd['patient'] = ov_hrd['patient_barcode'].str[:12]
ov_hrd = ov_hrd.set_index('patient')
ov_hrd = ov_hrd[~ov_hrd['HRD_Score'].isna()]

print(f"OV samples with valid HRD scores: {len(ov_hrd)}")
print(f"HRD_Score distribution: mean={ov_hrd['HRD_Score'].mean():.1f}, median={ov_hrd['HRD_Score'].median():.1f}")
print(f"HRD_Score >= 42: {(ov_hrd['HRD_Score'] >= 42).sum()} ({(ov_hrd['HRD_Score'] >= 42).mean()*100:.1f}%)")
print(f"HRD_Score Q75: {ov_hrd['HRD_Score'].quantile(0.75):.0f}")

# Tiered OV labels:
# Tier 1 (HRD+): HRD_Score >= 42 AND (BRCA1/2 mutation OR HRD_Score >= 63 [Q75])
# Tier 2 (HRP):  HRD_Score < 20
# Tier 3 (ambiguous): everything else
ov_q75 = ov_hrd['HRD_Score'].quantile(0.75)

ov_labels = pd.DataFrame(index=ov_hrd.index)
ov_labels['HRD_Score'] = ov_hrd['HRD_Score']
ov_labels['has_brca_alt'] = ov_labels.index.isin(ov_brca1_all | ov_brca2_all)

tier1_mask = (ov_labels['HRD_Score'] >= 42) & (
    ov_labels['has_brca_alt'] | (ov_labels['HRD_Score'] >= ov_q75)
)
tier2_mask = ov_labels['HRD_Score'] < 20
tier3_mask = ~tier1_mask & ~tier2_mask

ov_labels['tier'] = 3
ov_labels.loc[tier1_mask, 'tier'] = 1
ov_labels.loc[tier2_mask, 'tier'] = 2
ov_labels['label'] = 'ambiguous'
ov_labels.loc[tier1_mask, 'label'] = 'HRD'
ov_labels.loc[tier2_mask, 'label'] = 'HRP'

print(f"\nOV Tiered Labels:")
print(f"  Tier 1 (HRD+): {(ov_labels['tier']==1).sum()}")
print(f"  Tier 2 (HRP):  {(ov_labels['tier']==2).sum()}")
print(f"  Tier 3 (ambiguous): {(ov_labels['tier']==3).sum()}")

# Binary labels for evaluation (HRD_Score >= 42)
ov_binary = (ov_hrd['HRD_Score'] >= 42).astype(int)
ov_binary.name = 'hrd_binary'
print(f"\nOV binary labels (threshold 42): HRD={ov_binary.sum()}, HRP={(1-ov_binary).sum()}")

# ============================================================
# STEP 1b: Compute BRCA tiered labels
# ============================================================
print("\n" + "-" * 40)
print("BRCA Tiered Labels")
print("-" * 40)

labeler = TieredHRDLabeler()
brca_status_indexed = brca_status_df.set_index('sample')
brca_tiered = labeler.label_tcga_cohort(brca_status_indexed, brca_hrd_df)

print(f"BRCA Tiered Labels:")
print(f"  Tier 1 (HRD): {(brca_tiered['tier']==1).sum()}")
print(f"  Tier 2 (HRP): {(brca_tiered['tier']==2).sum()}")
print(f"  Tier 3 (ambiguous): {(brca_tiered['tier']==3).sum()}")

# ============================================================
# STEP 3: Compute confound scores
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: CONFOUND SCORES")
print("=" * 70)

# 3a. Tumor purity from DDR footprints (ABSOLUTE estimates)
brca_purity = ddr_brca['purity'].dropna()
ov_purity = ddr_ov['purity'].dropna()
print(f"\nPurity data: BRCA={len(brca_purity)}, OV={len(ov_purity)}")
print(f"  BRCA purity: mean={brca_purity.mean():.3f}, median={brca_purity.median():.3f}")
print(f"  OV purity: mean={ov_purity.mean():.3f}, median={ov_purity.median():.3f}")

# 3b. Proliferation metagene
PROLIF_GENES = ['MKI67', 'TOP2A', 'PCNA', 'MCM2', 'MCM4', 'MCM6', 'AURKA', 'AURKB',
                'BUB1', 'BUB1B', 'CCNB1', 'CCNB2', 'CDK1', 'CDC20', 'PLK1', 'TPX2',
                'UBE2C', 'BIRC5', 'MELK', 'CEP55']

prolif_avail_brca = [g for g in PROLIF_GENES if g in brca_vst.columns]
prolif_avail_ov = [g for g in PROLIF_GENES if g in ov_vst.columns]
print(f"\nProliferation genes available: BRCA={len(prolif_avail_brca)}/20, OV={len(prolif_avail_ov)}/20")
missing = [g for g in PROLIF_GENES if g not in brca_vst.columns]
if missing:
    print(f"  Missing: {missing}")

brca_prolif = brca_vst[prolif_avail_brca].mean(axis=1)
brca_prolif.name = 'proliferation'
ov_prolif = ov_vst[prolif_avail_ov].mean(axis=1)
ov_prolif.name = 'proliferation'
print(f"  BRCA proliferation: mean={brca_prolif.mean():.2f}, std={brca_prolif.std():.2f}")
print(f"  OV proliferation: mean={ov_prolif.mean():.2f}, std={ov_prolif.std():.2f}")

# 3c. Immune metagene (simplified immune score)
IMMUNE_GENES = ['PTPRC', 'CD3D', 'CD3E', 'CD4', 'CD8A', 'CD8B', 'CD19', 'CD79A',
                'FOXP3', 'CTLA4', 'PDCD1', 'GZMA', 'GZMB', 'PRF1', 'IFNG']

imm_avail_brca = [g for g in IMMUNE_GENES if g in brca_vst.columns]
imm_avail_ov = [g for g in IMMUNE_GENES if g in ov_vst.columns]
print(f"\nImmune genes available: BRCA={len(imm_avail_brca)}/15, OV={len(imm_avail_ov)}/15")
missing_imm = [g for g in IMMUNE_GENES if g not in brca_vst.columns]
if missing_imm:
    print(f"  Missing: {missing_imm}")

brca_immune = brca_vst[imm_avail_brca].mean(axis=1)
brca_immune.name = 'immune'
ov_immune = ov_vst[imm_avail_ov].mean(axis=1)
ov_immune.name = 'immune'
print(f"  BRCA immune: mean={brca_immune.mean():.2f}, std={brca_immune.std():.2f}")
print(f"  OV immune: mean={ov_immune.mean():.2f}, std={ov_immune.std():.2f}")

# CD8 T cell subset
CD8_GENES = ['CD8A', 'CD8B', 'GZMA', 'GZMB', 'PRF1', 'IFNG']
cd8_avail_brca = [g for g in CD8_GENES if g in brca_vst.columns]
cd8_avail_ov = [g for g in CD8_GENES if g in ov_vst.columns]
brca_cd8 = brca_vst[cd8_avail_brca].mean(axis=1)
brca_cd8.name = 'cd8_score'
ov_cd8 = ov_vst[cd8_avail_ov].mean(axis=1)
ov_cd8.name = 'cd8_score'

# ============================================================
# Helper: regress out confounds from expression
# ============================================================
def regress_out(expression, confounds_df):
    """Regress out confound variables from expression matrix.

    Parameters
    ----------
    expression : DataFrame (samples x genes)
    confounds_df : DataFrame (samples x confounds), aligned to expression index

    Returns
    -------
    DataFrame of residuals, same shape as expression
    """
    # Align
    common_idx = expression.index.intersection(confounds_df.index)
    expr = expression.loc[common_idx].copy()
    conf = confounds_df.loc[common_idx].copy()

    # Drop samples with missing confounds
    valid = conf.dropna().index
    expr = expr.loc[valid]
    conf = conf.loc[valid]

    X = conf.values
    residuals = expr.copy()

    for i, gene in enumerate(expr.columns):
        lr = LinearRegression().fit(X, expr[gene].values)
        residuals[gene] = expr[gene].values - lr.predict(X)

    return residuals

# ============================================================
# STEP 2 & 4: Train on BRCA, test on OV
# ============================================================
print("\n" + "=" * 70)
print("STEP 2 & 4: TRAIN ON BRCA, TEST ON OV")
print("=" * 70)

# Prepare BRCA training data with tiered labels
brca_tier12 = brca_tiered[brca_tiered['tier'].isin([1, 2])].copy()
brca_y = (brca_tier12['label'] == 'HRD').astype(int)
print(f"\nBRCA training: {len(brca_y)} samples (HRD={brca_y.sum()}, HRP={(1-brca_y).sum()})")

# Ensure gene alignment
genes_for_model = [g for g in dd500_genes if g in brca_vst.columns and g in ov_vst.columns]
print(f"DD500 genes available in both: {len(genes_for_model)}")

# Also prepare common_genes version
common_genes_avail = [g for g in common_genes if g in brca_vst.columns and g in ov_vst.columns]

# Prepare confound DataFrames
def make_confound_df(purity_series, prolif_series, samples):
    """Build confound DataFrame for given samples."""
    df = pd.DataFrame(index=samples)
    df['purity'] = purity_series.reindex(samples)
    df['proliferation'] = prolif_series.reindex(samples)
    return df

brca_confounds = make_confound_df(brca_purity, brca_prolif, brca_vst.index)
ov_confounds = make_confound_df(ov_purity, ov_prolif, ov_vst.index)

# For conditions requiring confound correction, we need samples with valid confounds
brca_has_purity = brca_confounds['purity'].dropna().index
brca_has_prolif = brca_confounds['proliferation'].dropna().index
brca_has_both = brca_has_purity.intersection(brca_has_prolif)

ov_has_purity = ov_confounds['purity'].dropna().index
ov_has_prolif = ov_confounds['proliferation'].dropna().index
ov_has_both = ov_has_purity.intersection(ov_has_prolif)

print(f"\nBRCA with purity: {len(brca_has_purity)}, with prolif: {len(brca_has_prolif)}, with both: {len(brca_has_both)}")
print(f"OV with purity: {len(ov_has_purity)}, with prolif: {len(ov_has_prolif)}, with both: {len(ov_has_both)}")

# Prepare OV evaluation data
# Use binary labels (HRD_Score >= 42) for all OV samples with HRD scores
ov_eval_samples = ov_vst.index.intersection(ov_binary.index)
print(f"OV samples with expression AND HRD labels: {len(ov_eval_samples)}")

def train_and_evaluate(brca_X, brca_y_sub, ov_X, ov_y_sub, condition_name, gene_set_name="DD500"):
    """Train ElasticNet on BRCA, evaluate on OV."""
    # Align BRCA
    common_brca = brca_X.index.intersection(brca_y_sub.index)
    X_train = brca_X.loc[common_brca]
    y_train = brca_y_sub.loc[common_brca]

    # Align OV
    common_ov = ov_X.index.intersection(ov_y_sub.index)
    X_test = ov_X.loc[common_ov]
    y_test = ov_y_sub.loc[common_ov]

    if len(X_train) < 20 or len(X_test) < 20:
        print(f"  [{condition_name}] Insufficient samples: train={len(X_train)}, test={len(X_test)}")
        return None

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train ElasticNet logistic regression
    model = LogisticRegression(
        penalty='elasticnet', solver='saga', l1_ratio=0.5,
        C=1.0, max_iter=2000, random_state=42, n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    # Predict
    y_prob = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    # Also compute AUC on tiered OV labels
    ov_tiered_eval = ov_labels[ov_labels['tier'].isin([1, 2])].copy()
    tiered_common = X_test.index.intersection(ov_tiered_eval.index)
    tiered_auc = None
    if len(tiered_common) >= 20:
        y_tiered = (ov_tiered_eval.loc[tiered_common, 'label'] == 'HRD').astype(int)
        tiered_probs = model.predict_proba(scaler.transform(ov_X.loc[tiered_common])[:, :X_train.shape[1]])[:, 1]
        tiered_auc = roc_auc_score(y_tiered, tiered_probs)

    n_nonzero = np.sum(model.coef_[0] != 0)

    print(f"  [{condition_name}] ({gene_set_name}) AUC={auc:.4f} | "
          f"train={len(X_train)}, test={len(X_test)}, features={X_train.shape[1]}, "
          f"non-zero coefs={n_nonzero}"
          f"{f', tiered_AUC={tiered_auc:.4f}' if tiered_auc else ''}")

    return {
        'condition': condition_name,
        'gene_set': gene_set_name,
        'auc_binary': float(auc),
        'auc_tiered': float(tiered_auc) if tiered_auc else None,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': int(X_train.shape[1]),
        'n_nonzero_coefs': int(n_nonzero),
        'y_prob': y_prob,
        'y_test': y_test.values,
        'test_samples': X_test.index.tolist(),
        'model': model,
        'scaler': scaler,
    }


# Run all 4 conditions
results = {}

# (a) Raw zero-shot
print("\n--- Condition (a): Raw zero-shot ---")
brca_train_idx = brca_y.index.intersection(brca_vst.index)
for gene_set_name, genes in [("DD500", genes_for_model), ("Common17K", common_genes_avail)]:
    r = train_and_evaluate(
        brca_vst.loc[:, genes], brca_y,
        ov_vst.loc[:, genes], ov_binary,
        "raw_zeroshot", gene_set_name
    )
    if r:
        results[f"a_raw_{gene_set_name}"] = r

# (b) Purity-corrected
print("\n--- Condition (b): Purity-corrected ---")
brca_purity_conf = brca_confounds[['purity']].dropna()
ov_purity_conf = ov_confounds[['purity']].dropna()

brca_vst_purity_corr = regress_out(brca_vst[genes_for_model], brca_purity_conf)
ov_vst_purity_corr = regress_out(ov_vst[genes_for_model], ov_purity_conf)
print(f"  Purity-corrected: BRCA={brca_vst_purity_corr.shape[0]}, OV={ov_vst_purity_corr.shape[0]}")

r = train_and_evaluate(
    brca_vst_purity_corr, brca_y,
    ov_vst_purity_corr, ov_binary,
    "purity_corrected", "DD500"
)
if r:
    results["b_purity_DD500"] = r

# (c) Proliferation-corrected
print("\n--- Condition (c): Proliferation-corrected ---")
brca_prolif_conf = brca_confounds[['proliferation']].dropna()
ov_prolif_conf = ov_confounds[['proliferation']].dropna()

brca_vst_prolif_corr = regress_out(brca_vst[genes_for_model], brca_prolif_conf)
ov_vst_prolif_corr = regress_out(ov_vst[genes_for_model], ov_prolif_conf)
print(f"  Proliferation-corrected: BRCA={brca_vst_prolif_corr.shape[0]}, OV={ov_vst_prolif_corr.shape[0]}")

r = train_and_evaluate(
    brca_vst_prolif_corr, brca_y,
    ov_vst_prolif_corr, ov_binary,
    "prolif_corrected", "DD500"
)
if r:
    results["c_prolif_DD500"] = r

# (d) Purity + proliferation corrected
print("\n--- Condition (d): Purity + Proliferation corrected ---")
brca_both_conf = brca_confounds[['purity', 'proliferation']].dropna()
ov_both_conf = ov_confounds[['purity', 'proliferation']].dropna()

brca_vst_both_corr = regress_out(brca_vst[genes_for_model], brca_both_conf)
ov_vst_both_corr = regress_out(ov_vst[genes_for_model], ov_both_conf)
print(f"  Both-corrected: BRCA={brca_vst_both_corr.shape[0]}, OV={ov_vst_both_corr.shape[0]}")

r = train_and_evaluate(
    brca_vst_both_corr, brca_y,
    ov_vst_both_corr, ov_binary,
    "purity_prolif_corrected", "DD500"
)
if r:
    results["d_both_DD500"] = r

# Also test with rank-transformed data
print("\n--- Rank-transformed variants ---")
for gene_set_name, genes in [("DD500", genes_for_model)]:
    r = train_and_evaluate(
        brca_ranks.loc[:, genes], brca_y,
        ov_ranks.loc[:, genes], ov_binary,
        "raw_zeroshot_ranks", gene_set_name
    )
    if r:
        results[f"a_raw_ranks_{gene_set_name}"] = r

# ============================================================
# STEP 5: Confound analysis of DD-500 genes
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: CONFOUND ANALYSIS OF DD-500 GENES")
print("=" * 70)

# Compute Spearman correlations of DD-500 genes with confounds
# Use BRCA data since it was the training cohort

# Align samples
brca_with_confounds = brca_vst.index.intersection(brca_purity.index).intersection(brca_prolif.index)
print(f"BRCA samples with all confound data: {len(brca_with_confounds)}")

confound_corrs = []
for gene in genes_for_model:
    expr = brca_vst.loc[brca_with_confounds, gene]

    # Purity correlation
    pur = brca_purity.loc[brca_with_confounds]
    rho_pur, p_pur = stats.spearmanr(expr, pur, nan_policy='omit')

    # Proliferation correlation
    pro = brca_prolif.loc[brca_with_confounds]
    rho_pro, p_pro = stats.spearmanr(expr, pro, nan_policy='omit')

    # CD8 T cell correlation
    cd8 = brca_cd8.loc[brca_with_confounds]
    rho_cd8, p_cd8 = stats.spearmanr(expr, cd8, nan_policy='omit')

    # Immune overall
    imm = brca_immune.loc[brca_with_confounds]
    rho_imm, p_imm = stats.spearmanr(expr, imm, nan_policy='omit')

    confound_corrs.append({
        'gene': gene,
        'rho_purity': rho_pur, 'p_purity': p_pur,
        'rho_prolif': rho_pro, 'p_prolif': p_pro,
        'rho_cd8': rho_cd8, 'p_cd8': p_cd8,
        'rho_immune': rho_imm, 'p_immune': p_imm,
    })

confound_df = pd.DataFrame(confound_corrs)

# BH correction
for confound in ['purity', 'prolif', 'cd8', 'immune']:
    _, pvals_adj, _, _ = multipletests(confound_df[f'p_{confound}'], method='fdr_bh')
    confound_df[f'p_{confound}_bh'] = pvals_adj
    n_sig = (pvals_adj < 0.05).sum()
    frac = n_sig / len(confound_df)
    median_rho = confound_df.loc[pvals_adj < 0.05, f'rho_{confound}'].abs().median() if n_sig > 0 else 0
    print(f"\n{confound}: {n_sig}/{len(confound_df)} ({frac*100:.1f}%) genes significantly correlated (BH p<0.05)")
    print(f"  Median |rho| of significant genes: {median_rho:.3f}")

confound_df.to_csv(OUT / 'dd500_confound_correlations.csv', index=False)

# ============================================================
# STEP 6: Pan-cancer retrain with leave-one-cancer-out CV
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: PAN-CANCER LEAVE-ONE-CANCER-OUT CV")
print("=" * 70)

# Use purity+proliferation-corrected expression
# Pool BRCA + OV

# First, get samples that have: expression + HRD labels + confounds
# BRCA tiered
brca_t12 = brca_tiered[brca_tiered['tier'].isin([1, 2])]
brca_pool_idx = brca_t12.index.intersection(brca_vst_both_corr.index)
brca_pool_X = brca_vst_both_corr.loc[brca_pool_idx, genes_for_model]
brca_pool_y = (brca_t12.loc[brca_pool_idx, 'label'] == 'HRD').astype(int)

# OV binary
ov_t12 = ov_labels[ov_labels['tier'].isin([1, 2])]
ov_pool_idx = ov_t12.index.intersection(ov_vst_both_corr.index)
ov_pool_X = ov_vst_both_corr.loc[ov_pool_idx, genes_for_model]
ov_pool_y = (ov_t12.loc[ov_pool_idx, 'label'] == 'HRD').astype(int)

print(f"BRCA pool: {len(brca_pool_X)} (HRD={brca_pool_y.sum()}, HRP={(1-brca_pool_y).sum()})")
print(f"OV pool: {len(ov_pool_X)} (HRD={ov_pool_y.sum()}, HRP={(1-ov_pool_y).sum()})")

# LOCO: Train on BRCA, test on OV
if len(ov_pool_X) >= 10 and len(brca_pool_X) >= 10:
    scaler_loco = StandardScaler()
    X_train_loco = scaler_loco.fit_transform(brca_pool_X)
    X_test_loco = scaler_loco.transform(ov_pool_X)

    model_loco = LogisticRegression(
        penalty='elasticnet', solver='saga', l1_ratio=0.5,
        C=1.0, max_iter=2000, random_state=42, n_jobs=-1
    )
    model_loco.fit(X_train_loco, brca_pool_y)
    y_prob_loco = model_loco.predict_proba(X_test_loco)[:, 1]
    auc_brca2ov = roc_auc_score(ov_pool_y, y_prob_loco)
    print(f"\nLOCO: Train BRCA -> Test OV: AUC = {auc_brca2ov:.4f}")

    # Reverse: Train on OV, test on BRCA
    scaler_loco2 = StandardScaler()
    X_train_loco2 = scaler_loco2.fit_transform(ov_pool_X)
    X_test_loco2 = scaler_loco2.transform(brca_pool_X)

    model_loco2 = LogisticRegression(
        penalty='elasticnet', solver='saga', l1_ratio=0.5,
        C=1.0, max_iter=2000, random_state=42, n_jobs=-1
    )
    model_loco2.fit(X_train_loco2, ov_pool_y)
    y_prob_loco2 = model_loco2.predict_proba(X_test_loco2)[:, 1]
    auc_ov2brca = roc_auc_score(brca_pool_y, y_prob_loco2)
    print(f"LOCO: Train OV -> Test BRCA: AUC = {auc_ov2brca:.4f}")

    # Pooled with cancer_type covariate
    pooled_X = pd.concat([brca_pool_X, ov_pool_X])
    pooled_y = pd.concat([brca_pool_y, ov_pool_y])
    cancer_type = pd.Series(0, index=pooled_X.index)
    cancer_type.loc[ov_pool_X.index] = 1
    pooled_X_with_ct = pooled_X.copy()
    pooled_X_with_ct['cancer_type'] = cancer_type

    # Train on BRCA, test on OV with cancer_type
    brca_X_ct = pooled_X_with_ct.loc[brca_pool_X.index]
    ov_X_ct = pooled_X_with_ct.loc[ov_pool_X.index]

    scaler_ct = StandardScaler()
    X_train_ct = scaler_ct.fit_transform(brca_X_ct)
    X_test_ct = scaler_ct.transform(ov_X_ct)

    model_ct = LogisticRegression(
        penalty='elasticnet', solver='saga', l1_ratio=0.5,
        C=1.0, max_iter=2000, random_state=42, n_jobs=-1
    )
    model_ct.fit(X_train_ct, brca_pool_y)
    y_prob_ct = model_ct.predict_proba(X_test_ct)[:, 1]
    auc_ct = roc_auc_score(ov_pool_y, y_prob_ct)
    print(f"LOCO with cancer_type covariate: Train BRCA -> Test OV: AUC = {auc_ct:.4f}")
else:
    auc_brca2ov = auc_ov2brca = auc_ct = None
    print("Insufficient samples for LOCO CV")

# ============================================================
# STEP 7: Generate plots
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: GENERATING PLOTS")
print("=" * 70)

# --- Plot 1: ROC curves for all conditions ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left panel: DD500 conditions
ax = axes[0]
condition_map = {
    'a_raw_DD500': ('Raw zero-shot', 'C0'),
    'b_purity_DD500': ('Purity-corrected', 'C1'),
    'c_prolif_DD500': ('Proliferation-corrected', 'C2'),
    'd_both_DD500': ('Purity + Prolif corrected', 'C3'),
    'a_raw_ranks_DD500': ('Rank-transformed', 'C4'),
}

for key, (label, color) in condition_map.items():
    if key in results:
        r = results[key]
        fpr, tpr, _ = roc_curve(r['y_test'], r['y_prob'])
        ax.plot(fpr, tpr, label=f"{label} (AUC={r['auc_binary']:.3f})", color=color, lw=2)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC: BRCA-trained ElasticNet on TCGA-OV\n(DD500 genes, binary HRD labels)', fontsize=13)
ax.legend(fontsize=10, loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Right panel: Common17K raw
ax = axes[1]
if 'a_raw_Common17K' in results:
    r = results['a_raw_Common17K']
    fpr, tpr, _ = roc_curve(r['y_test'], r['y_prob'])
    ax.plot(fpr, tpr, label=f"Common 17K (AUC={r['auc_binary']:.3f})", color='C5', lw=2)
if 'a_raw_DD500' in results:
    r = results['a_raw_DD500']
    fpr, tpr, _ = roc_curve(r['y_test'], r['y_prob'])
    ax.plot(fpr, tpr, label=f"DD500 (AUC={r['auc_binary']:.3f})", color='C0', lw=2)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC: Gene set comparison\n(Raw zero-shot)', fontsize=13)
ax.legend(fontsize=10, loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(OUT / 'roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved roc_curves.png")

# --- Plot 2: Confound correlation bar chart ---
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
confound_names = ['purity', 'prolif', 'cd8', 'immune']
confound_titles = ['Tumor Purity', 'Proliferation', 'CD8 T cells', 'Immune Overall']

for ax, cname, ctitle in zip(axes, confound_names, confound_titles):
    rhos = confound_df[f'rho_{cname}'].values
    sig = confound_df[f'p_{cname}_bh'] < 0.05

    # Sort by absolute correlation
    order = np.argsort(np.abs(rhos))[::-1]
    rhos_sorted = rhos[order]
    sig_sorted = sig.values[order]

    colors = ['firebrick' if s else 'lightgray' for s in sig_sorted]
    ax.bar(range(len(rhos_sorted)), rhos_sorted, color=colors, width=1.0, edgecolor='none')
    ax.set_xlabel('DD500 genes (sorted by |rho|)', fontsize=10)
    ax.set_ylabel(f'Spearman rho', fontsize=10)
    ax.set_title(f'{ctitle}\n(red = BH p<0.05: {sig.sum()}/{len(sig)})', fontsize=11)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlim(-5, len(rhos_sorted) + 5)

plt.tight_layout()
plt.savefig(OUT / 'dd500_confound_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved dd500_confound_correlations.png")

# --- Plot 3: Scatter plots - model predictions vs HRD score in OV ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
scatter_conditions = [
    ('a_raw_DD500', 'Raw zero-shot'),
    ('b_purity_DD500', 'Purity-corrected'),
    ('c_prolif_DD500', 'Proliferation-corrected'),
    ('d_both_DD500', 'Purity + Prolif corrected'),
]

for ax, (key, title) in zip(axes.flat, scatter_conditions):
    if key in results:
        r = results[key]
        samples = r['test_samples']
        probs = r['y_prob']

        # Get HRD scores for these samples
        hrd_scores = ov_hrd.loc[ov_hrd.index.intersection(samples), 'HRD_Score']
        common_s = [s for s in samples if s in hrd_scores.index]
        prob_dict = dict(zip(samples, probs))

        x = [hrd_scores.loc[s] for s in common_s]
        y = [prob_dict[s] for s in common_s]

        ax.scatter(x, y, alpha=0.5, s=20, c='steelblue')
        ax.axvline(42, color='red', ls='--', alpha=0.5, label='HRD threshold (42)')
        ax.axhline(0.5, color='gray', ls=':', alpha=0.5, label='Prediction threshold (0.5)')

        rho, pval = stats.spearmanr(x, y)
        ax.set_xlabel('HRD Score (LOH+TAI+LST)', fontsize=11)
        ax.set_ylabel('Predicted HRD probability', fontsize=11)
        ax.set_title(f'{title}\nrho={rho:.3f}, p={pval:.2e}', fontsize=12)
        ax.legend(fontsize=9)
    else:
        ax.set_title(f'{title}\n(no data)')

plt.tight_layout()
plt.savefig(OUT / 'scatter_predictions_vs_hrd.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved scatter_predictions_vs_hrd.png")

# --- Plot 4: PCA of BRCA vs OV ---
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# Use DD500 genes, raw VST
combined = pd.concat([brca_vst[genes_for_model], ov_vst[genes_for_model]])
cancer_labels = ['BRCA'] * len(brca_vst) + ['OV'] * len(ov_vst)

pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(StandardScaler().fit_transform(combined))

# Panel 1: Color by cancer type
ax = axes[0]
brca_mask = np.array(cancer_labels) == 'BRCA'
ax.scatter(pcs[brca_mask, 0], pcs[brca_mask, 1], alpha=0.3, s=10, c='C0', label='BRCA')
ax.scatter(pcs[~brca_mask, 0], pcs[~brca_mask, 1], alpha=0.3, s=10, c='C1', label='OV')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax.set_title('PCA: BRCA vs OV\n(DD500 genes, VST)', fontsize=12)
ax.legend(fontsize=10)

# Panel 2: Color by HRD status
ax = axes[1]
# BRCA HRD status
brca_hrd_indexed = brca_hrd_df.set_index('sample')
brca_hrd_binary = (brca_hrd_indexed['HRD-sum'] >= 42).astype(int)

brca_pcs = pcs[:len(brca_vst)]
ov_pcs = pcs[len(brca_vst):]

for i, sample in enumerate(brca_vst.index):
    if sample in brca_hrd_binary.index:
        c = 'red' if brca_hrd_binary.loc[sample] else 'blue'
        ax.scatter(brca_pcs[i, 0], brca_pcs[i, 1], alpha=0.2, s=8, c=c, marker='o')

for i, sample in enumerate(ov_vst.index):
    if sample in ov_binary.index:
        c = 'red' if ov_binary.loc[sample] else 'blue'
        ax.scatter(ov_pcs[i, 0], ov_pcs[i, 1], alpha=0.3, s=15, c=c, marker='^')

# Legend
ax.scatter([], [], c='red', marker='o', label='BRCA HRD+', s=30)
ax.scatter([], [], c='blue', marker='o', label='BRCA HRP', s=30)
ax.scatter([], [], c='red', marker='^', label='OV HRD+', s=30)
ax.scatter([], [], c='blue', marker='^', label='OV HRP', s=30)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax.set_title('PCA: Colored by HRD status\n(threshold 42)', fontsize=12)
ax.legend(fontsize=9, markerscale=1.5)

# Panel 3: After confound correction
if len(brca_vst_both_corr) > 0 and len(ov_vst_both_corr) > 0:
    combined_corr = pd.concat([brca_vst_both_corr[genes_for_model], ov_vst_both_corr[genes_for_model]])
    pca2 = PCA(n_components=2, random_state=42)
    pcs2 = pca2.fit_transform(StandardScaler().fit_transform(combined_corr))

    ax = axes[2]
    n_brca = len(brca_vst_both_corr)
    ax.scatter(pcs2[:n_brca, 0], pcs2[:n_brca, 1], alpha=0.3, s=10, c='C0', label='BRCA')
    ax.scatter(pcs2[n_brca:, 0], pcs2[n_brca:, 1], alpha=0.3, s=10, c='C1', label='OV')
    ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.set_title('PCA: After purity+prolif correction\n(DD500 genes)', fontsize=12)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT / 'pca_brca_ov.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved pca_brca_ov.png")

# --- Plot 5: AUC summary bar chart ---
fig, ax = plt.subplots(figsize=(10, 6))
bar_data = []
for key, label in [
    ('a_raw_DD500', 'Raw\nzero-shot'),
    ('b_purity_DD500', 'Purity\ncorrected'),
    ('c_prolif_DD500', 'Prolif\ncorrected'),
    ('d_both_DD500', 'Purity+Prolif\ncorrected'),
    ('a_raw_ranks_DD500', 'Rank\ntransformed'),
    ('a_raw_Common17K', 'Raw\n(17K genes)'),
]:
    if key in results:
        bar_data.append((label, results[key]['auc_binary']))

if bar_data:
    labels, aucs = zip(*bar_data)
    colors = plt.cm.Set2(np.linspace(0, 1, len(aucs)))
    bars = ax.bar(range(len(aucs)), aucs, color=colors, edgecolor='black', lw=0.5)
    ax.set_xticks(range(len(aucs)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Cross-cancer Transfer AUC: BRCA -> OV', fontsize=13)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.5)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT / 'auc_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved auc_summary.png")

# ============================================================
# Save results JSON
# ============================================================
results_summary = {
    'experiment': 'Exp6: OV Zero-Shot Transfer with Confound Decomposition',
    'date': '2026-02-23',
    'data': {
        'brca_vst_samples': int(brca_vst.shape[0]),
        'ov_vst_samples': int(ov_vst.shape[0]),
        'brca_tiered': {
            'tier1_HRD': int((brca_tiered['tier']==1).sum()),
            'tier2_HRP': int((brca_tiered['tier']==2).sum()),
            'tier3_ambiguous': int((brca_tiered['tier']==3).sum()),
        },
        'ov_tiered': {
            'tier1_HRD': int((ov_labels['tier']==1).sum()),
            'tier2_HRP': int((ov_labels['tier']==2).sum()),
            'tier3_ambiguous': int((ov_labels['tier']==3).sum()),
        },
        'ov_binary_hrd_prevalence': float((ov_binary == 1).mean()),
        'ov_brca_alterations': {
            'BRCA1_total': len(ov_brca1_all),
            'BRCA1_mutation': len(ov_brca1_mut),
            'BRCA1_deletion': len(ov_brca1_del),
            'BRCA1_silencing': len(ov_brca1_sil),
            'BRCA2_total': len(ov_brca2_all),
            'BRCA2_mutation': len(ov_brca2_mut),
            'BRCA2_deletion': len(ov_brca2_del),
        },
    },
    'confound_stats': {
        'brca_purity': {'mean': float(brca_purity.mean()), 'median': float(brca_purity.median()), 'n': int(len(brca_purity))},
        'ov_purity': {'mean': float(ov_purity.mean()), 'median': float(ov_purity.median()), 'n': int(len(ov_purity))},
        'brca_proliferation': {'mean': float(brca_prolif.mean()), 'std': float(brca_prolif.std())},
        'ov_proliferation': {'mean': float(ov_prolif.mean()), 'std': float(ov_prolif.std())},
    },
    'transfer_results': {},
    'confound_analysis': {
        'purity': {
            'n_significant': int((confound_df['p_purity_bh'] < 0.05).sum()),
            'fraction_significant': float((confound_df['p_purity_bh'] < 0.05).mean()),
            'median_abs_rho_significant': float(confound_df.loc[confound_df['p_purity_bh'] < 0.05, 'rho_purity'].abs().median()) if (confound_df['p_purity_bh'] < 0.05).sum() > 0 else 0,
        },
        'proliferation': {
            'n_significant': int((confound_df['p_prolif_bh'] < 0.05).sum()),
            'fraction_significant': float((confound_df['p_prolif_bh'] < 0.05).mean()),
            'median_abs_rho_significant': float(confound_df.loc[confound_df['p_prolif_bh'] < 0.05, 'rho_prolif'].abs().median()) if (confound_df['p_prolif_bh'] < 0.05).sum() > 0 else 0,
        },
        'cd8': {
            'n_significant': int((confound_df['p_cd8_bh'] < 0.05).sum()),
            'fraction_significant': float((confound_df['p_cd8_bh'] < 0.05).mean()),
            'median_abs_rho_significant': float(confound_df.loc[confound_df['p_cd8_bh'] < 0.05, 'rho_cd8'].abs().median()) if (confound_df['p_cd8_bh'] < 0.05).sum() > 0 else 0,
        },
        'immune': {
            'n_significant': int((confound_df['p_immune_bh'] < 0.05).sum()),
            'fraction_significant': float((confound_df['p_immune_bh'] < 0.05).mean()),
            'median_abs_rho_significant': float(confound_df.loc[confound_df['p_immune_bh'] < 0.05, 'rho_immune'].abs().median()) if (confound_df['p_immune_bh'] < 0.05).sum() > 0 else 0,
        },
    },
}

# Add transfer results
for key, r in results.items():
    results_summary['transfer_results'][key] = {
        'auc_binary': r['auc_binary'],
        'auc_tiered': r['auc_tiered'],
        'n_train': r['n_train'],
        'n_test': r['n_test'],
        'n_features': r['n_features'],
        'n_nonzero_coefs': r['n_nonzero_coefs'],
    }

# Add LOCO results
if auc_brca2ov is not None:
    results_summary['loco_cv'] = {
        'train_brca_test_ov': float(auc_brca2ov),
        'train_ov_test_brca': float(auc_ov2brca),
        'with_cancer_type_covariate': float(auc_ct),
        'note': 'Using purity+proliferation-corrected DD500, tiered labels',
    }

with open(OUT / 'exp6_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"\nSaved exp6_results.json")

# ============================================================
# Print summary
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 6 SUMMARY")
print("=" * 70)

print("\n--- Transfer AUCs (BRCA -> OV) ---")
for key in ['a_raw_DD500', 'b_purity_DD500', 'c_prolif_DD500', 'd_both_DD500', 'a_raw_ranks_DD500', 'a_raw_Common17K']:
    if key in results:
        r = results[key]
        print(f"  {key:30s}  AUC = {r['auc_binary']:.4f}  (train={r['n_train']}, test={r['n_test']})")

if auc_brca2ov is not None:
    print(f"\n--- LOCO CV (purity+prolif corrected, tiered labels) ---")
    print(f"  Train BRCA -> Test OV:  AUC = {auc_brca2ov:.4f}")
    print(f"  Train OV -> Test BRCA:  AUC = {auc_ov2brca:.4f}")
    print(f"  With cancer_type covariate:  AUC = {auc_ct:.4f}")

print(f"\n--- KEY QUESTION ---")
raw_auc = results.get('a_raw_DD500', {}).get('auc_binary', 0)
both_auc = results.get('d_both_DD500', {}).get('auc_binary', 0)
print(f"  Raw zero-shot AUC:          {raw_auc:.4f}")
print(f"  Purity+Prolif corrected AUC: {both_auc:.4f}")
if both_auc > raw_auc:
    print(f"  -> Confound correction IMPROVES transfer by {both_auc - raw_auc:.4f}")
elif both_auc < raw_auc:
    print(f"  -> Confound correction HURTS transfer by {raw_auc - both_auc:.4f}")
else:
    print(f"  -> No difference")

print(f"\n--- DD-500 Confound Contamination ---")
for cname in ['purity', 'prolif', 'cd8', 'immune']:
    n_sig = (confound_df[f'p_{cname}_bh'] < 0.05).sum()
    frac = n_sig / len(confound_df)
    print(f"  {cname:15s}: {n_sig:3d}/{len(confound_df)} genes ({frac*100:.1f}%) significantly correlated")

print("\nDone!")
