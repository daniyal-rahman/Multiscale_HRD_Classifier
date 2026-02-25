"""
Investigate Consensus-93 AUC=0.500 on I-SPY2.
Peer reviewer concern: the diagonal ROC looks fake / predictions may have collapsed.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
np.random.seed(42)

REPO = Path('/home/dani/repos2/Multiscale_HRD_Classifier')
DATA_DIR = REPO / 'v2' / 'data_acquisition'
SIG_DIR = REPO / 'v2' / 'signature_analysis'
OUT_DIR = REPO / 'v2' / 'experiments'

# ── 1. Load TCGA-BRCA expression ──────────────────────────────────────────
print("=" * 70, flush=True)
print("1. Loading TCGA-BRCA expression", flush=True)
expr_raw = pd.read_parquet(DATA_DIR / 'processed' / 'tcga' / 'TCGA-BRCA' / 'rnaseq_fpkm.parquet')
non_gene = ['N_unmapped', 'N_multimapping', 'N_noFeature', 'N_ambiguous']
expr_raw = expr_raw.drop(index=[i for i in non_gene if i in expr_raw.index])

# Gene symbol mapping from a raw STAR-Counts file
raw_rnaseq_dir = DATA_DIR / 'raw' / 'tcga' / 'TCGA-BRCA' / 'rnaseq'
raw_files = list(raw_rnaseq_dir.glob('*.tsv'))
raw_df = pd.read_csv(raw_files[0], sep='\t', comment='#', index_col=0, usecols=[0, 1])
gene_map = raw_df['gene_name'].dropna().to_dict()

expr_raw.index = expr_raw.index.map(lambda x: gene_map.get(x, x))
expr_raw = expr_raw[~expr_raw.index.str.startswith('ENSG')]
expr_raw = expr_raw.groupby(expr_raw.index).mean()

# Map UUIDs -> barcodes
barcode_df = pd.read_parquet(DATA_DIR / 'processed' / 'tcga' / 'TCGA-BRCA' / 'uuid_to_barcode.parquet')
uuid_to_barcode = dict(zip(barcode_df['filename_uuid'], barcode_df['case_id']))
expr = expr_raw.rename(columns=uuid_to_barcode)
expr = expr.T.groupby(level=0).first().T
expr = expr[[c for c in expr.columns if c.startswith('TCGA-')]]
print(f"  Expression: {expr.shape[0]} genes x {expr.shape[1]} patients", flush=True)

# ── 2. Build HRD labels ──────────────────────────────────────────────────
print("\n2. Building HRD labels", flush=True)
mut_df = pd.read_csv(OUT_DIR / 'tcga_brca_hrr_mutations.csv')
pathogenic_types = ['Frame_Shift_Del', 'Frame_Shift_Ins', 'Nonsense_Mutation', 'Splice_Site']
path_muts = mut_df[mut_df['mutation_type'].isin(pathogenic_types)]
patients_with_pathogenic_hrr = set(path_muts['patient'].unique())

subtype_df = pd.read_csv(OUT_DIR / 'tcga_brca_cbio_subtypes.csv', index_col=0)
basal_patients = set(subtype_df[subtype_df['subtype'] == 'BRCA_Basal'].index)

genomic_df = pd.read_csv(OUT_DIR / 'tcga_brca_cbio_genomic.csv', index_col=0)
genomic_df.index = genomic_df.index.str[:12]
genomic_df = genomic_df.groupby(level=0).first()

expr_patients = set(expr.columns)
all_labeled_patients = expr_patients & set(subtype_df.index)

hrd_pos, hrd_neg = set(), set()
for p in all_labeled_patients:
    has_mut = p in patients_with_pathogenic_hrr
    is_basal = p in basal_patients
    subtype = subtype_df.loc[p, 'subtype'] if p in subtype_df.index else None
    fga = genomic_df.loc[p, 'fga'] if p in genomic_df.index else None
    if has_mut:
        hrd_pos.add(p)
    elif is_basal and fga is not None and fga > 0.3:
        hrd_pos.add(p)
    elif subtype in ('BRCA_LumA', 'BRCA_Normal') and not has_mut:
        if fga is None or fga < 0.3:
            hrd_neg.add(p)
    elif subtype == 'BRCA_LumB' and not has_mut and fga is not None and fga < 0.2:
        hrd_neg.add(p)

labels = pd.Series(dtype=int)
for p in hrd_pos: labels[p] = 1
for p in hrd_neg: labels[p] = 0
labeled_patients_list = sorted(labels.index)
X_all = expr[labeled_patients_list].T
y = labels[labeled_patients_list]
print(f"  Training set: {X_all.shape[0]} samples, {sum(y==1)} HRD+ / {sum(y==0)} HRD-", flush=True)

# ── 3. Load gene lists ───────────────────────────────────────────────────
print("\n3. Loading gene lists", flush=True)
consensus_df = pd.read_csv(SIG_DIR / 'consensus_hrd_genes.csv')
all_sig_df = pd.read_csv(SIG_DIR / 'all_signature_gene_lists.csv')

consensus_genes = consensus_df['Gene'].tolist()
all_sig_genes = all_sig_df['Gene'].unique().tolist()

available_genes = set(X_all.columns)
consensus_available = [g for g in consensus_genes if g in available_genes]
all_sig_available = [g for g in all_sig_genes if g in available_genes]

# Data-driven gene selection (t-test)
gene_vars = X_all.var()
expressed_genes = gene_vars[gene_vars > 0.01].index.tolist()
hrd_samples = y[y == 1].index
hrp_samples = y[y == 0].index
ttest_results = []
for gene in expressed_genes:
    hrd_vals = X_all.loc[hrd_samples, gene].dropna()
    hrp_vals = X_all.loc[hrp_samples, gene].dropna()
    if len(hrd_vals) > 2 and len(hrp_vals) > 2:
        t_stat, p_val = stats.ttest_ind(hrd_vals, hrp_vals, equal_var=False)
        if np.isfinite(t_stat):
            ttest_results.append({'gene': gene, 'abs_t': abs(t_stat)})
ttest_df = pd.DataFrame(ttest_results).sort_values('abs_t', ascending=False)
data_driven_500 = ttest_df.head(500)['gene'].tolist()

gene_sets = {
    'Consensus-93': consensus_available,
    'All-signature-410': all_sig_available,
    'Data-driven-500': data_driven_500,
}
for name, genes in gene_sets.items():
    print(f"  {name}: {len(genes)} genes", flush=True)

# ── 4. Train 5-fold CV models ────────────────────────────────────────────
print("\n4. Training ElasticNet models (5-fold CV)", flush=True)

def train_cv(X, y, gene_list, n_splits=5, random_state=42):
    genes_avail = [g for g in gene_list if g in X.columns]
    X_sub = X[genes_avail].replace([np.inf, -np.inf], np.nan).fillna(0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipelines = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_sub, y)):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                penalty='elasticnet', solver='saga', l1_ratio=0.5,
                C=1.0, max_iter=5000, random_state=random_state))
        ])
        pipe.fit(X_sub.iloc[train_idx], y.iloc[train_idx])
        pipelines.append(pipe)
    return {'pipelines': pipelines, 'genes_used': genes_avail}

results = {}
for name, genes in gene_sets.items():
    print(f"  Training {name}...", flush=True)
    results[name] = train_cv(X_all, y, genes)

# ── 5. Load I-SPY2 data ──────────────────────────────────────────────────
print("\n5. Loading I-SPY2 data", flush=True)
ispy2_expr = pd.read_parquet(OUT_DIR / 'ispy2_expression_cache.parquet')
ispy2_meta = pd.read_parquet(DATA_DIR / 'processed' / 'ispy2' / 'geo_metadata.parquet')
ispy2_meta = ispy2_meta[ispy2_meta['pcr'].astype(str).isin(['0', '1'])].copy()
ispy2_meta['pcr_int'] = ispy2_meta['pcr'].astype(int)
common_samples = list(set(ispy2_expr.columns) & set(ispy2_meta['sample_id']))
ispy2_val_meta = ispy2_meta.set_index('sample_id').loc[common_samples]
ispy2_val_expr = ispy2_expr[common_samples].T
ispy2_val_y = ispy2_val_meta['pcr_int']
print(f"  I-SPY2: {len(ispy2_val_y)} samples, pCR rate={ispy2_val_y.mean():.1%}", flush=True)
print(f"  I-SPY2 expression: {ispy2_val_expr.shape[1]} genes", flush=True)

# ── 6. Generate ensemble predictions on I-SPY2 ───────────────────────────
print("\n6. Generating I-SPY2 predictions", flush=True)

def predict_ispy2(r, ispy2_val_expr):
    genes = r['genes_used']
    genes_common = [g for g in genes if g in ispy2_val_expr.columns]
    genes_missing = [g for g in genes if g not in ispy2_val_expr.columns]
    preds_list = []
    for pipe in r['pipelines']:
        X_al = pd.DataFrame(0.0, index=ispy2_val_expr.index, columns=genes)
        for g in genes_common:
            X_al[g] = ispy2_val_expr[g].values
        X_al = X_al.fillna(0.0)
        preds_list.append(pipe.predict_proba(X_al)[:, 1])
    ens_pred = np.mean(preds_list, axis=0)
    return ens_pred, genes_common, genes_missing, preds_list

report = []
report.append("# Consensus-93 AUC=0.500 Investigation\n")
report.append("## Peer Review Concern\n")
report.append("The Consensus-93 blue line on the I-SPY2 ROC plot has AUC=0.500,")
report.append(" which lies exactly on the diagonal. The reviewer asks whether this")
report.append(" is a genuine result or an artifact of collapsed/constant predictions.\n")
report.append("\n## Investigation Results\n")

all_preds = {}
all_fold_preds = {}
for name in ['Consensus-93', 'All-signature-410', 'Data-driven-500']:
    ens_pred, genes_common, genes_missing, fold_preds = predict_ispy2(results[name], ispy2_val_expr)
    all_preds[name] = ens_pred
    all_fold_preds[name] = fold_preds
    auc = roc_auc_score(ispy2_val_y, ens_pred)
    ap = average_precision_score(ispy2_val_y, ens_pred)
    q25, q75 = np.percentile(ens_pred, [25, 75])
    iqr = q75 - q25
    print(f"\n  --- {name} ---", flush=True)
    print(f"  Genes: {len(genes_common)}/{len(results[name]['genes_used'])} available", flush=True)
    if genes_missing:
        print(f"  Missing genes: {genes_missing}", flush=True)
    print(f"  AUC={auc:.4f}, AP={ap:.4f}", flush=True)
    print(f"  Predictions: min={ens_pred.min():.6f}, max={ens_pred.max():.6f}", flush=True)
    print(f"               mean={ens_pred.mean():.6f}, std={ens_pred.std():.6f}", flush=True)
    print(f"               Q25={q25:.6f}, Q75={q75:.6f}, IQR={iqr:.6f}", flush=True)
    print(f"  N unique values: {len(np.unique(np.round(ens_pred, 6)))}", flush=True)

    # Per-fold variance
    fold_stds = [np.std(fp) for fp in fold_preds]
    fold_ranges = [np.ptp(fp) for fp in fold_preds]
    print(f"  Per-fold std: {[f'{s:.6f}' for s in fold_stds]}", flush=True)
    print(f"  Per-fold range: {[f'{r:.6f}' for r in fold_ranges]}", flush=True)

# ── 7. Detailed Consensus-93 analysis ────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("7. DETAILED CONSENSUS-93 ANALYSIS", flush=True)
print("=" * 70, flush=True)

c93_pred = all_preds['Consensus-93']
c93_y = ispy2_val_y.values

# Check if predictions are near-constant
print(f"\n  Prediction range: {c93_pred.max() - c93_pred.min():.8f}", flush=True)
print(f"  Prediction std:   {c93_pred.std():.8f}", flush=True)

# Check if all predictions cluster near a single value
print(f"\n  Histogram of predictions (10 bins):", flush=True)
counts, edges = np.histogram(c93_pred, bins=10)
for i in range(len(counts)):
    bar = '#' * counts[i]
    print(f"    [{edges[i]:.4f}, {edges[i+1]:.4f}): {counts[i]:3d} {bar}", flush=True)

# Check prediction by label
pcr_pos = c93_pred[c93_y == 1]
pcr_neg = c93_pred[c93_y == 0]
print(f"\n  pCR=1 predictions: mean={pcr_pos.mean():.6f}, std={pcr_pos.std():.6f}, n={len(pcr_pos)}", flush=True)
print(f"  pCR=0 predictions: mean={pcr_neg.mean():.6f}, std={pcr_neg.std():.6f}, n={len(pcr_neg)}", flush=True)
t_stat, t_pval = stats.ttest_ind(pcr_pos, pcr_neg)
print(f"  t-test: t={t_stat:.4f}, p={t_pval:.4f}", flush=True)
u_stat, u_pval = stats.mannwhitneyu(pcr_pos, pcr_neg, alternative='two-sided')
print(f"  Mann-Whitney U: U={u_stat:.0f}, p={u_pval:.4f}", flush=True)

# Check ROC curve in detail
fpr, tpr, thresholds = roc_curve(c93_y, c93_pred)
print(f"\n  ROC curve points: {len(fpr)}", flush=True)
print(f"  FPR values: {fpr[:20]}...", flush=True)
print(f"  TPR values: {tpr[:20]}...", flush=True)

# Measure how close the ROC is to the diagonal
# If the curve oscillates around the diagonal, predictions have variance but no signal
deviations = tpr - fpr  # Distance from diagonal at each ROC point
print(f"\n  ROC deviation from diagonal (TPR - FPR):", flush=True)
print(f"    max positive: {deviations.max():.4f}", flush=True)
print(f"    max negative: {deviations.min():.4f}", flush=True)
print(f"    mean:         {deviations.mean():.4f}", flush=True)
print(f"    std:          {deviations.std():.4f}", flush=True)

# ── 8. Model coefficient analysis ────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("8. CONSENSUS-93 MODEL COEFFICIENT ANALYSIS", flush=True)
print("=" * 70, flush=True)

c93_r = results['Consensus-93']
all_coefs = []
for fold_i, pipe in enumerate(c93_r['pipelines']):
    coefs = pipe.named_steps['model'].coef_[0]
    intercept = pipe.named_steps['model'].intercept_[0]
    n_nonzero = np.sum(np.abs(coefs) > 1e-8)
    all_coefs.append(coefs)
    print(f"  Fold {fold_i}: {n_nonzero}/{len(coefs)} non-zero coefs, intercept={intercept:.4f}", flush=True)

mean_coefs = np.mean(all_coefs, axis=0)
coef_df = pd.DataFrame({
    'gene': c93_r['genes_used'],
    'mean_coef': mean_coefs,
    'abs_coef': np.abs(mean_coefs)
}).sort_values('abs_coef', ascending=False)
print(f"\n  Top 20 Consensus-93 genes by |coefficient|:", flush=True)
for _, row in coef_df.head(20).iterrows():
    in_ispy2 = row['gene'] in ispy2_val_expr.columns
    marker = "  " if in_ispy2 else " *MISSING*"
    print(f"    {row['gene']:>12s}: {row['mean_coef']:+.6f}{marker}", flush=True)

n_zero = (coef_df['abs_coef'] < 1e-8).sum()
print(f"\n  Genes with zero coefficient (all folds): {n_zero}/{len(coef_df)}", flush=True)

# ── 9. Check TCGA-BRCA scaling vs I-SPY2 scaling ─────────────────────────
print("\n" + "=" * 70, flush=True)
print("9. DISTRIBUTION SHIFT ANALYSIS: TCGA vs I-SPY2", flush=True)
print("=" * 70, flush=True)

c93_genes = c93_r['genes_used']
c93_genes_ispy2 = [g for g in c93_genes if g in ispy2_val_expr.columns]

# Compare raw expression distributions for top genes
tcga_expr_subset = X_all[c93_genes].replace([np.inf, -np.inf], np.nan).fillna(0)
ispy2_expr_subset = pd.DataFrame(0.0, index=ispy2_val_expr.index, columns=c93_genes)
for g in c93_genes_ispy2:
    ispy2_expr_subset[g] = ispy2_val_expr[g].values
ispy2_expr_subset = ispy2_expr_subset.fillna(0.0)

print(f"\n  Comparing top 10 genes (raw expression distributions):", flush=True)
top10 = coef_df.head(10)['gene'].tolist()
for gene in top10:
    tcga_vals = tcga_expr_subset[gene]
    ispy2_vals = ispy2_expr_subset[gene]
    in_ispy2 = gene in c93_genes_ispy2
    print(f"\n    {gene} (in I-SPY2: {in_ispy2}):", flush=True)
    print(f"      TCGA:  mean={tcga_vals.mean():.2f}, std={tcga_vals.std():.2f}, "
          f"min={tcga_vals.min():.2f}, max={tcga_vals.max():.2f}", flush=True)
    print(f"      I-SPY2: mean={ispy2_vals.mean():.2f}, std={ispy2_vals.std():.2f}, "
          f"min={ispy2_vals.min():.2f}, max={ispy2_vals.max():.2f}", flush=True)

# Check what the scaler does
print(f"\n  Effect of TCGA-trained scaler on I-SPY2 data:", flush=True)
pipe0 = c93_r['pipelines'][0]
scaler = pipe0.named_steps['scaler']
tcga_means = scaler.mean_
tcga_scales = scaler.scale_

# After scaling, the I-SPY2 data will be: (ispy2 - tcga_mean) / tcga_scale
scaled_ispy2 = (ispy2_expr_subset.values - tcga_means) / tcga_scales
print(f"  Scaled I-SPY2 stats:", flush=True)
print(f"    mean across features: {np.mean(scaled_ispy2):.4f}", flush=True)
print(f"    std across features:  {np.std(scaled_ispy2):.4f}", flush=True)
print(f"    min: {np.min(scaled_ispy2):.4f}, max: {np.max(scaled_ispy2):.4f}", flush=True)

# Check if TCGA (FPKM) vs I-SPY2 (microarray) are on very different scales
print(f"\n  TCGA expression range (FPKM): {tcga_expr_subset.values.min():.2f} to {tcga_expr_subset.values.max():.2f}", flush=True)
print(f"  I-SPY2 expression range (microarray): {ispy2_expr_subset.values.min():.2f} to {ispy2_expr_subset.values.max():.2f}", flush=True)

# ── 10. Specific check: are predictions truly constant? ──────────────────
print("\n" + "=" * 70, flush=True)
print("10. IS CONSENSUS-93 TRULY AUC=0.500 OR A ROUNDING ARTIFACT?", flush=True)
print("=" * 70, flush=True)

print(f"\n  Exact AUC: {roc_auc_score(c93_y, c93_pred):.10f}", flush=True)

# Check per-fold predictions (before ensembling)
for fold_i, fp in enumerate(all_fold_preds['Consensus-93']):
    fold_auc = roc_auc_score(c93_y, fp)
    print(f"  Fold {fold_i} AUC: {fold_auc:.6f}, pred range: [{fp.min():.6f}, {fp.max():.6f}], std={fp.std():.6f}", flush=True)

# Check if a random permutation would give similar AUC
n_perms = 1000
perm_aucs = []
rng = np.random.RandomState(42)
for _ in range(n_perms):
    perm = rng.permutation(c93_pred)
    perm_aucs.append(roc_auc_score(c93_y, perm))
perm_aucs = np.array(perm_aucs)
print(f"\n  Permutation test (shuffled predictions):", flush=True)
print(f"    Mean AUC: {perm_aucs.mean():.4f} +/- {perm_aucs.std():.4f}", flush=True)
print(f"    Actual AUC: {roc_auc_score(c93_y, c93_pred):.4f}", flush=True)
p_perm = np.mean(perm_aucs >= roc_auc_score(c93_y, c93_pred))
print(f"    p-value (actual >= permuted): {p_perm:.4f}", flush=True)

# ── 11. Rank correlation between predictions and labels ───────────────────
print("\n" + "=" * 70, flush=True)
print("11. RANK CORRELATION ANALYSIS", flush=True)
print("=" * 70, flush=True)

for name in ['Consensus-93', 'All-signature-410', 'Data-driven-500']:
    pred = all_preds[name]
    spearman_r, spearman_p = stats.spearmanr(pred, c93_y)
    point_biserial_r, pb_p = stats.pointbiserialr(c93_y, pred)
    print(f"\n  {name}:", flush=True)
    print(f"    Spearman r={spearman_r:.4f}, p={spearman_p:.4f}", flush=True)
    print(f"    Point-biserial r={point_biserial_r:.4f}, p={pb_p:.4f}", flush=True)

# ── 12. Cross-model prediction correlations ───────────────────────────────
print("\n" + "=" * 70, flush=True)
print("12. CROSS-MODEL PREDICTION CORRELATIONS", flush=True)
print("=" * 70, flush=True)
for n1 in all_preds:
    for n2 in all_preds:
        if n1 >= n2: continue
        r, p = stats.spearmanr(all_preds[n1], all_preds[n2])
        print(f"  {n1} vs {n2}: Spearman r={r:.4f}, p={p:.4f}", flush=True)

# ── SUMMARY ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("SUMMARY", flush=True)
print("=" * 70, flush=True)

c93_std = c93_pred.std()
c93_range = c93_pred.max() - c93_pred.min()
c93_iqr = np.percentile(c93_pred, 75) - np.percentile(c93_pred, 25)

a410_std = all_preds['All-signature-410'].std()
dd500_std = all_preds['Data-driven-500'].std()

print(f"\n  Prediction variance comparison:", flush=True)
print(f"    Consensus-93:       std={c93_std:.6f}, range={c93_range:.6f}, IQR={c93_iqr:.6f}", flush=True)
print(f"    All-signature-410:  std={a410_std:.6f}, range={all_preds['All-signature-410'].max()-all_preds['All-signature-410'].min():.6f}", flush=True)
print(f"    Data-driven-500:    std={dd500_std:.6f}, range={all_preds['Data-driven-500'].max()-all_preds['Data-driven-500'].min():.6f}", flush=True)

print(f"\n  Consensus-93 variance ratio vs All-sig-410: {c93_std/a410_std:.4f}", flush=True)
print(f"  Consensus-93 variance ratio vs DD-500:      {c93_std/dd500_std:.4f}", flush=True)

if c93_std < 0.01:
    print("\n  CONCLUSION: Predictions are near-constant (std < 0.01).", flush=True)
    print("  The AUC=0.500 is likely because the model produces nearly identical", flush=True)
    print("  predictions for all I-SPY2 samples, making ranking impossible.", flush=True)
    print("  This is NOT 'fake' but rather a consequence of distribution shift", flush=True)
    print("  between TCGA (RNA-seq FPKM) and I-SPY2 (microarray) platforms.", flush=True)
elif c93_std < 0.05:
    print("\n  CONCLUSION: Predictions have low but non-trivial variance.", flush=True)
    print("  The model produces some variation but lacks discriminative power.", flush=True)
else:
    print("\n  CONCLUSION: Predictions have meaningful variance.", flush=True)
    print("  The AUC=0.500 reflects genuine lack of predictive signal,", flush=True)
    print("  not collapsed predictions.", flush=True)

# Check if the issue is platform mismatch specifically for consensus genes
n_missing = len(c93_genes) - len(c93_genes_ispy2)
print(f"\n  Missing consensus genes in I-SPY2: {n_missing}/{len(c93_genes)}", flush=True)
if n_missing > 0:
    missing = [g for g in c93_genes if g not in ispy2_val_expr.columns]
    print(f"  Missing: {missing}", flush=True)

print("\nDone.", flush=True)
