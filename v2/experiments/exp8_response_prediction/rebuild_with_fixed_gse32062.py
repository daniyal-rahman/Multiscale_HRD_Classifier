#!/usr/bin/env python3
"""
Rebuild pooled rank matrix with fixed GSE32062 gene mapping.

Changes from original:
1. Uses fixed GSE32062 expression (Agilent-only, no platform collision)
2. Uses updated common gene list (11,089 genes, down from 11,140)
3. Re-runs LODO-CV with C=0.01 (tuned model)
4. Produces new bootstrap CIs and permutation test

Usage: srun --mem=4G --time=00:45:00 python rebuild_with_fixed_gse32062.py
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
FIX = BASE.parent / "response_data" / "gse32062_fix"
EXP = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT = EXP / "validation_fixed"
OUT.mkdir(parents=True, exist_ok=True)

C_OPT = 0.01
EXCLUDE = ['GSE28739']  # Only exclude GSE28739 now; GSE32062 is FIXED
N_BOOTSTRAP = 2000
N_PERMUTATION = 5000

print("=" * 70)
print("Rebuilding Pooled Matrix with Fixed GSE32062")
print("=" * 70)

# ─── Step 1: Load new common gene list ────────────────────────────
common_genes_file = FIX / "common_genes_all_10_datasets_fixed.txt"
common_genes = [g.strip() for g in open(common_genes_file)]
print(f"\nNew common genes: {len(common_genes)} (was 11,140)")

# ─── Step 2: Rebuild pooled matrix ───────────────────────────────
print("\nRebuilding pooled rank matrix...")

datasets_config = {
    'TCGA-OV':   {'expr': BASE / 'TCGA-OV_expression_std.parquet',   'resp': BASE / 'TCGA-OV_response.parquet',   'category': 'clinical_platinum'},
    'GSE32062':  {'expr': FIX / 'GSE32062_expression_genes_fixed.parquet', 'resp': BASE / 'GSE32062_response.parquet', 'category': 'clinical_platinum', 'transpose': True},
    'GSE156699': {'expr': BASE / 'GSE156699_expression_std.parquet', 'resp': BASE / 'GSE156699_response.parquet', 'category': 'clinical_platinum'},
    'GSE63885':  {'expr': BASE / 'GSE63885_expression_std.parquet',  'resp': BASE / 'GSE63885_response.parquet',  'category': 'clinical_platinum'},
    'GSE30161':  {'expr': BASE / 'GSE30161_expression_std.parquet',  'resp': BASE / 'GSE30161_response.parquet',  'category': 'clinical_platinum'},
    'GSE28739':  {'expr': BASE / 'GSE28739_expression_std.parquet',  'resp': BASE / 'GSE28739_response.parquet',  'category': 'clinical_platinum'},
    'GSE18864':  {'expr': BASE / 'GSE18864_expression_std.parquet',  'resp': BASE / 'GSE18864_response.parquet',  'category': 'clinical_cisplatin'},
    'GSE173839': {'expr': BASE / 'GSE173839_expression_std.parquet', 'resp': BASE / 'GSE173839_response.parquet', 'category': 'clinical_ispy2'},
    'GSE194040': {'expr': BASE / 'GSE194040_expression_std.parquet', 'resp': BASE / 'GSE194040_response.parquet', 'category': 'clinical_ispy2'},
    'GDSC':      {'expr': BASE / 'GDSC_expression_std.parquet',      'resp': BASE / 'GDSC_response.parquet',      'category': 'cell_line'},
}

pooled_dfs = []
for name, cfg in datasets_config.items():
    print(f"  {name}...", end=" ", flush=True)

    expr = pd.read_parquet(cfg['expr'])

    # Fixed GSE32062 file is genes×samples (transposed)
    if cfg.get('transpose', False):
        expr = expr.T

    # Subset to common genes
    available = [g for g in common_genes if g in expr.columns]
    expr = expr[available]

    # Fill any missing common genes with NaN
    for g in common_genes:
        if g not in expr.columns:
            expr[g] = np.nan
    expr = expr[common_genes]

    # Rank transform within each sample → [0, 1]
    ranks = expr.rank(axis=1, pct=True)

    # Load response labels
    resp = pd.read_parquet(cfg['resp'])
    if name == 'GDSC':
        resp = resp[resp['drug'] == 'olaparib']

    resp = resp.set_index('sample_id')
    ranks.index = ranks.index.astype(str)
    resp.index = resp.index.astype(str)
    common_samples = ranks.index.intersection(resp.index)

    ranks_matched = ranks.loc[common_samples].copy()
    resp_matched = resp.loc[common_samples]

    ranks_matched['response_binary'] = resp_matched['response_binary'].values
    if 'response_continuous' in resp_matched.columns:
        ranks_matched['response_continuous'] = resp_matched['response_continuous'].values
    else:
        ranks_matched['response_continuous'] = np.nan
    ranks_matched['drug'] = resp_matched['drug'].values if 'drug' in resp_matched.columns else 'unknown'
    ranks_matched['dataset'] = name
    ranks_matched['category'] = cfg['category']
    if 'cancer_type' in resp_matched.columns:
        ranks_matched['cancer_type'] = resp_matched['cancer_type'].values
    else:
        ranks_matched['cancer_type'] = 'unknown'

    pooled_dfs.append(ranks_matched)
    n_nan = ranks_matched[common_genes].isna().any(axis=1).sum()
    print(f"n={len(common_samples)}, available={len(available)}/{len(common_genes)}, nan_rows={n_nan}", flush=True)

pooled = pd.concat(pooled_dfs)
print(f"\nPooled matrix: {pooled.shape[0]} samples x {len(common_genes)} genes + metadata")

# Check GSE32062 specifically
gse32062 = pooled[pooled['dataset'] == 'GSE32062']
nan_frac = gse32062[common_genes].isna().mean().mean()
print(f"\nGSE32062 check: {len(gse32062)} samples, NaN fraction: {nan_frac:.4f}")

pooled.to_parquet(OUT / 'pooled_rank_matrix_fixed.parquet')
print("Saved pooled_rank_matrix_fixed.parquet")

# ─── Step 3: LODO-CV with C=0.01 ─────────────────────────────────
print(f"\n{'=' * 70}")
print(f"LODO-CV with C={C_OPT} on fixed data")
print("=" * 70)

ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (pooled['category'] != 'cell_line') &
    ~((pooled['category'] == 'clinical_ispy2') & (~pooled['drug'].isin(ispy2_parpi_drugs)))
)
model_a = pooled[model_a_mask].copy()

X_all = np.nan_to_num(model_a[common_genes].values, nan=0.5).astype(np.float32)
y_all = model_a['response_binary'].values.astype(int)
datasets_all = model_a['dataset'].values
sample_ids = model_a.index.tolist()

print(f"Model A: {len(model_a)} samples across {model_a['dataset'].nunique()} datasets")
print(f"Response rate: {y_all.mean():.3f}")

all_predictions = {}
lodo_aucs = {}
for held_out in sorted(model_a['dataset'].unique()):
    train_mask = datasets_all != held_out
    test_mask = datasets_all == held_out
    X_train = X_all[train_mask]
    X_test = X_all[test_mask]
    y_train = y_all[train_mask]

    clf = LogisticRegression(penalty='l2', C=C_OPT, solver='lbfgs',
                              max_iter=3000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]

    test_ids = [sample_ids[i] for i in range(len(sample_ids)) if test_mask[i]]
    for sid, prob in zip(test_ids, y_prob):
        all_predictions[sid] = prob

    if len(np.unique(y_all[test_mask])) >= 2:
        auc = roc_auc_score(y_all[test_mask], y_prob)
        lodo_aucs[held_out] = auc
        print(f"  {held_out}: AUC={auc:.3f} (n={test_mask.sum()}, pos={y_all[test_mask].sum()}, "
              f"pred_range={y_prob.max()-y_prob.min():.3f})", flush=True)
    else:
        print(f"  {held_out}: SKIPPED (single class, n={test_mask.sum()})", flush=True)

mean_all = np.mean(list(lodo_aucs.values()))
clean_aucs = {k: v for k, v in lodo_aucs.items() if k not in EXCLUDE}
mean_clean = np.mean(list(clean_aucs.values()))
print(f"\nMean AUC (all {len(lodo_aucs)}): {mean_all:.4f}")
print(f"Mean AUC (clean {len(clean_aucs)}, excl {EXCLUDE}): {mean_clean:.4f}")

# Compare to old GSE32062
old_gse32062_auc = 0.4894
if 'GSE32062' in lodo_aucs:
    print(f"\nGSE32062 AUC: {lodo_aucs['GSE32062']:.4f} (was {old_gse32062_auc:.4f}, delta={lodo_aucs['GSE32062']-old_gse32062_auc:+.4f})")

# ─── Step 4: Bootstrap CIs ───────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"Bootstrap CIs ({N_BOOTSTRAP} iterations)")
print("=" * 70)

bootstrap_results = {}
for ds in sorted(lodo_aucs.keys()):
    mask = datasets_all == ds
    ids = [sample_ids[i] for i in range(len(sample_ids)) if mask[i]]
    y_true = y_all[mask]
    y_pred = np.array([all_predictions[sid] for sid in ids])

    point_auc = roc_auc_score(y_true, y_pred)
    boot_aucs = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))

    boot_aucs = np.array(boot_aucs)
    ci_lower = np.percentile(boot_aucs, 2.5)
    ci_upper = np.percentile(boot_aucs, 97.5)

    bootstrap_results[ds] = {
        'point_auc': round(float(point_auc), 4),
        'ci_lower': round(float(ci_lower), 4),
        'ci_upper': round(float(ci_upper), 4),
        'ci_width': round(float(ci_upper - ci_lower), 4),
        'includes_random': bool(ci_lower <= 0.5),
        'n_test': int(mask.sum()),
        'n_pos': int(y_true.sum()),
        'n_neg': int((1 - y_true).sum()),
    }
    sig = "*random*" if ci_lower <= 0.5 else "SIGNIFICANT"
    print(f"  {ds}: AUC={point_auc:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] {sig}", flush=True)

cannot_reject = [ds for ds, r in bootstrap_results.items() if r['includes_random']]
significant = [ds for ds, r in bootstrap_results.items() if not r['includes_random']]
all_aucs_list = [r['point_auc'] for r in bootstrap_results.values()]
clean_aucs_list = [r['point_auc'] for ds, r in bootstrap_results.items() if ds not in EXCLUDE]

bootstrap_results['_summary'] = {
    'model': f'L2_C={C_OPT}',
    'n_bootstrap': N_BOOTSTRAP,
    'gene_set': f'{len(common_genes)}_fixed',
    'cannot_reject_random': cannot_reject,
    'significantly_above_random': significant,
    'mean_auc_all': round(float(np.mean(all_aucs_list)), 4),
    'mean_auc_clean': round(float(np.mean(clean_aucs_list)), 4),
    'exclude_list': EXCLUDE,
}

with open(OUT / "bootstrap_cis_fixed.json", 'w') as f:
    json.dump(bootstrap_results, f, indent=2)

# ─── Step 5: Permutation test ────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"Permutation test ({N_PERMUTATION} permutations)")
print("=" * 70)

obs_per_ds = {}
for ds in sorted(lodo_aucs.keys()):
    mask = datasets_all == ds
    ids = [sample_ids[i] for i in range(len(sample_ids)) if mask[i]]
    y_true = y_all[mask]
    y_pred = np.array([all_predictions[sid] for sid in ids])
    if len(np.unique(y_true)) >= 2:
        obs_per_ds[ds] = roc_auc_score(y_true, y_pred)

obs_mean_all = np.mean(list(obs_per_ds.values()))
obs_mean_clean = np.mean([v for k, v in obs_per_ds.items() if k not in EXCLUDE])

null_means_all = []
null_means_clean = []
for perm in range(N_PERMUTATION):
    if perm % 1000 == 0:
        print(f"  perm {perm}/{N_PERMUTATION}...", flush=True)
    perm_aucs = {}
    for ds in obs_per_ds:
        mask = datasets_all == ds
        ids = [sample_ids[i] for i in range(len(sample_ids)) if mask[i]]
        y_true = y_all[mask].copy()
        np.random.shuffle(y_true)
        y_pred = np.array([all_predictions[sid] for sid in ids])
        if len(np.unique(y_true)) >= 2:
            perm_aucs[ds] = roc_auc_score(y_true, y_pred)
    if perm_aucs:
        null_means_all.append(np.mean(list(perm_aucs.values())))
        null_means_clean.append(np.mean([v for k, v in perm_aucs.items() if k not in EXCLUDE]))

null_means_all = np.array(null_means_all)
null_means_clean = np.array(null_means_clean)

perm_p_all = np.mean(null_means_all >= obs_mean_all)
perm_p_clean = np.mean(null_means_clean >= obs_mean_clean)

print(f"  All {len(obs_per_ds)}: observed={obs_mean_all:.4f}, null={null_means_all.mean():.4f}, p={perm_p_all}", flush=True)
print(f"  Clean {len([k for k in obs_per_ds if k not in EXCLUDE])}: observed={obs_mean_clean:.4f}, null={null_means_clean.mean():.4f}, p={perm_p_clean}", flush=True)

permutation_results = {
    'model': f'L2_C={C_OPT}',
    'gene_set': f'{len(common_genes)}_fixed',
    'n_permutations': N_PERMUTATION,
    'overall_all': {
        'n_datasets': len(obs_per_ds),
        'observed_mean_auc': round(float(obs_mean_all), 4),
        'null_mean': round(float(null_means_all.mean()), 4),
        'null_std': round(float(null_means_all.std()), 4),
        'permutation_p': round(float(perm_p_all), 4),
    },
    'overall_clean': {
        'n_datasets': len([k for k in obs_per_ds if k not in EXCLUDE]),
        'observed_mean_auc': round(float(obs_mean_clean), 4),
        'null_mean': round(float(null_means_clean.mean()), 4),
        'null_std': round(float(null_means_clean.std()), 4),
        'permutation_p': round(float(perm_p_clean), 4),
    },
    'exclude_list': EXCLUDE,
}

with open(OUT / "permutation_fixed.json", 'w') as f:
    json.dump(permutation_results, f, indent=2)

# ─── Step 6: Calibration ─────────────────────────────────────────
print(f"\n{'=' * 70}")
print("Calibration analysis")
print("=" * 70)

calibration_results = {}
for ds in sorted(lodo_aucs.keys()):
    mask = datasets_all == ds
    ids = [sample_ids[i] for i in range(len(sample_ids)) if mask[i]]
    y_true = y_all[mask]
    y_pred = np.array([all_predictions[sid] for sid in ids])

    brier = brier_score_loss(y_true, y_pred)

    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            avg_pred = y_pred[in_bin].mean()
            avg_true = y_true[in_bin].mean()
            ece += (in_bin.sum() / len(y_true)) * abs(avg_pred - avg_true)

    calibration_results[ds] = {
        'brier_score': round(float(brier), 4),
        'ece': round(float(ece), 4),
        'pred_mean': round(float(y_pred.mean()), 4),
        'pred_std': round(float(y_pred.std()), 4),
        'pred_range': round(float(y_pred.max() - y_pred.min()), 4),
        'n_test': int(mask.sum()),
    }
    print(f"  {ds}: Brier={brier:.3f}, ECE={ece:.3f}, range={y_pred.max()-y_pred.min():.3f}", flush=True)

all_ece = [r['ece'] for r in calibration_results.values()]
clean_ece = [r['ece'] for ds, r in calibration_results.items() if ds not in EXCLUDE]
calibration_results['_summary'] = {
    'model': f'L2_C={C_OPT}',
    'gene_set': f'{len(common_genes)}_fixed',
    'mean_ece_all': round(float(np.mean(all_ece)), 4),
    'mean_ece_clean': round(float(np.mean(clean_ece)), 4),
    'mean_brier_all': round(float(np.mean([r['brier_score'] for r in calibration_results.values()])), 4),
}

with open(OUT / "calibration_fixed.json", 'w') as f:
    json.dump(calibration_results, f, indent=2)

# ─── Summary ──────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("SUMMARY: Fixed GSE32062 Results")
print("=" * 70)

lodo_summary = {
    'model': f'L2_C={C_OPT}',
    'gene_set': f'{len(common_genes)}_fixed',
    'n_total_samples': len(model_a),
    'per_dataset_auc': {ds: round(float(auc), 4) for ds, auc in sorted(lodo_aucs.items())},
    'mean_auc_all': round(float(mean_all), 4),
    'mean_auc_clean': round(float(mean_clean), 4),
    'exclude_list': EXCLUDE,
    'gse32062_auc_old': old_gse32062_auc,
    'gse32062_auc_fixed': round(float(lodo_aucs.get('GSE32062', 0)), 4),
    'gse32062_improvement': round(float(lodo_aucs.get('GSE32062', 0) - old_gse32062_auc), 4),
}

with open(OUT / "lodo_summary_fixed.json", 'w') as f:
    json.dump(lodo_summary, f, indent=2)

print(f"\nPer-dataset AUC:")
for ds, auc in sorted(lodo_aucs.items()):
    old_marker = ""
    if ds == 'GSE32062':
        old_marker = f" (was {old_gse32062_auc:.3f}, delta={auc-old_gse32062_auc:+.3f})"
    print(f"  {ds}: {auc:.4f}{old_marker}")

print(f"\nMean AUC all {len(lodo_aucs)}: {mean_all:.4f} (was 0.6929)")
print(f"Mean AUC clean {len(clean_aucs)}: {mean_clean:.4f} (was 0.7142)")
print(f"\nBootstrap: {len(significant)}/{len(bootstrap_results)-1} significant")
print(f"Permutation p (all): {perm_p_all}")
print(f"Permutation p (clean): {perm_p_clean}")
print(f"Mean ECE (clean): {np.mean(clean_ece):.4f} (was 0.2155)")

print(f"\nAll results saved to {OUT}/")
print("Done!")
