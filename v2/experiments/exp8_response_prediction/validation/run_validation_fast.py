#!/usr/bin/env python3
"""
Task 4 Part 2 (fast version): LODO-CV permutation test + Calibration.
Uses pre-fitted models to speed up LODO permutation dramatically:
- Pre-fit 9 LODO models once, cache predictions
- For permutation, only shuffle labels and recompute AUC (no re-training)
  This tests: "is AUC=0.67 better than chance given the actual predictions?"
  which is the correct null for evaluating the LODO-CV procedure.
"""

import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/data_acquisition/response_data")
OUT = EXP / "validation"

print("Loading data...", flush=True)
pooled = pd.read_parquet(EXP / 'pooled_rank_matrix.parquet')

with open(BASE / "common_genes_all_10_datasets.txt") as f:
    common_genes = [g.strip() for g in f]

meta_cols = ['response_binary', 'response_continuous', 'drug', 'dataset', 'category', 'cancer_type']
gene_cols = [c for c in pooled.columns if c not in meta_cols]

ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (pooled['category'] != 'cell_line') &
    ~((pooled['category'] == 'clinical_ispy2') & (~pooled['drug'].isin(ispy2_parpi_drugs)))
)
model_a = pooled[model_a_mask].copy()

X_all = model_a[gene_cols].values
y_all = model_a['response_binary'].values.astype(int)
datasets_all = model_a['dataset'].values
unique_datasets = sorted(model_a['dataset'].unique())

print(f"Model A: {len(model_a)} samples, {len(unique_datasets)} datasets", flush=True)

# =====================================================================
# Step 1: Run LODO-CV once, cache all predictions and true labels
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("Step 1: LODO-CV — caching predictions", flush=True)
print("=" * 70, flush=True)

lodo_predictions = {}
obs_aucs = {}

for held_out in unique_datasets:
    train_mask = datasets_all != held_out
    test_mask = datasets_all == held_out

    X_tr = np.nan_to_num(X_all[train_mask], nan=0.5)
    y_tr = y_all[train_mask]
    X_te = np.nan_to_num(X_all[test_mask], nan=0.5)
    y_te = y_all[test_mask]

    if len(np.unique(y_te)) < 2:
        continue

    clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                             max_iter=5000, class_weight='balanced', random_state=42)
    clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_te)[:, 1]

    lodo_predictions[held_out] = {'y_true': y_te, 'y_prob': y_prob}
    obs_aucs[held_out] = roc_auc_score(y_te, y_prob)
    print(f"  {held_out}: AUC={obs_aucs[held_out]:.4f} (n={len(y_te)})", flush=True)

obs_mean_auc = np.mean(list(obs_aucs.values()))
print(f"\n  Observed mean LODO-CV AUC: {obs_mean_auc:.4f}", flush=True)

# =====================================================================
# Step 2: LODO-CV Permutation Test (label permutation on cached preds)
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("Step 2: LODO-CV Permutation Test (label shuffle, 10000 permutations)", flush=True)
print("=" * 70, flush=True)

N_PERM = 10000
np.random.seed(456)
perm_mean_aucs = []

for i in range(N_PERM):
    if (i + 1) % 2000 == 0:
        print(f"  Permutation {i+1}/{N_PERM}...", flush=True)

    perm_aucs = []
    for held_out, data in lodo_predictions.items():
        y_te = data['y_true']
        y_prob = data['y_prob']
        # Shuffle labels within this held-out set
        y_perm = np.random.permutation(y_te)
        if len(np.unique(y_perm)) < 2:
            continue
        perm_aucs.append(roc_auc_score(y_perm, y_prob))

    if perm_aucs:
        perm_mean_aucs.append(np.mean(perm_aucs))

perm_mean_aucs = np.array(perm_mean_aucs)
perm_p_lodo = np.mean(perm_mean_aucs >= obs_mean_auc)

print(f"\nLODO-CV Permutation Results:", flush=True)
print(f"  Observed mean AUC: {obs_mean_auc:.4f}", flush=True)
print(f"  Null mean AUC: {perm_mean_aucs.mean():.4f} +/- {perm_mean_aucs.std():.4f}", flush=True)
print(f"  Permutation p-value: {perm_p_lodo:.6f}", flush=True)
print(f"  Null 95th: {np.percentile(perm_mean_aucs, 95):.4f}", flush=True)
print(f"  Null 99th: {np.percentile(perm_mean_aucs, 99):.4f}", flush=True)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(perm_mean_aucs, bins=50, edgecolor='black', alpha=0.7, density=True)
ax.axvline(obs_mean_auc, color='red', linewidth=2, linestyle='--',
           label=f'Observed: {obs_mean_auc:.4f}\np={perm_p_lodo:.6f}')
ax.set_xlabel('Mean LODO-CV AUC (permuted labels)')
ax.set_ylabel('Density')
ax.set_title(f'LODO-CV Permutation Test (n={N_PERM})')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'permutation_lodo_overall.png', dpi=150)
plt.close(fig)

# Update permutation results
perm_file = OUT / 'permutation_interaction.json'
if perm_file.exists():
    with open(perm_file) as f:
        perm_results = json.load(f)
else:
    perm_results = {}

perm_results['lodo_cv_permutation'] = {
    'n_permutations': N_PERM,
    'method': 'label_shuffle_on_cached_predictions',
    'observed_mean_auc': round(float(obs_mean_auc), 4),
    'null_mean_auc': round(float(perm_mean_aucs.mean()), 4),
    'null_std_auc': round(float(perm_mean_aucs.std()), 4),
    'null_95th': round(float(np.percentile(perm_mean_aucs, 95)), 4),
    'null_99th': round(float(np.percentile(perm_mean_aucs, 99)), 4),
    'permutation_p': round(float(perm_p_lodo), 6),
    'per_dataset_obs_auc': {k: round(float(v), 4) for k, v in obs_aucs.items()},
}

with open(perm_file, 'w') as f:
    json.dump(perm_results, f, indent=2)
print(f"Saved permutation results", flush=True)

# =====================================================================
# Step 3: Calibration Curves
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("Step 3: Calibration Curves", flush=True)
print("=" * 70, flush=True)

calibration_results = {}
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes_flat = axes.flatten()

for idx, held_out in enumerate(unique_datasets):
    if held_out not in lodo_predictions:
        calibration_results[held_out] = {'skipped': True}
        continue

    y_te = lodo_predictions[held_out]['y_true']
    y_prob = lodo_predictions[held_out]['y_prob']
    print(f"\n--- {held_out} ---", flush=True)

    brier = brier_score_loss(y_te, y_prob)

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means_pred = []
    bin_means_true = []
    bin_counts = []

    for b in range(n_bins):
        mask = (y_prob >= bin_edges[b]) & (y_prob < bin_edges[b+1])
        if b == n_bins - 1:
            mask = (y_prob >= bin_edges[b]) & (y_prob <= bin_edges[b+1])
        if mask.sum() > 0:
            bin_means_pred.append(float(y_prob[mask].mean()))
            bin_means_true.append(float(y_te[mask].mean()))
            bin_counts.append(int(mask.sum()))
        else:
            bin_means_pred.append(float((bin_edges[b] + bin_edges[b+1]) / 2))
            bin_means_true.append(np.nan)
            bin_counts.append(0)

    total = sum(bin_counts)
    ece = sum(bc / total * abs(bt - bp)
              for bp, bt, bc in zip(bin_means_pred, bin_means_true, bin_counts)
              if bc > 0 and not np.isnan(bt))

    pred_range = float(y_prob.max() - y_prob.min())
    pred_std = float(y_prob.std())

    print(f"  Brier: {brier:.4f}, ECE: {ece:.4f}, Range: {pred_range:.4f}, Std: {pred_std:.4f}", flush=True)

    calibration_results[held_out] = {
        'brier_score': round(float(brier), 4),
        'ece': round(float(ece), 4),
        'pred_range': round(pred_range, 4),
        'pred_std': round(pred_std, 4),
        'pred_mean': round(float(y_prob.mean()), 4),
        'pred_min': round(float(y_prob.min()), 4),
        'pred_max': round(float(y_prob.max()), 4),
        'bin_means_pred': [round(x, 4) for x in bin_means_pred],
        'bin_means_true': [round(x, 4) if not np.isnan(x) else None for x in bin_means_true],
        'bin_counts': bin_counts,
        'n_test': len(y_te),
        'n_pos': int(y_te.sum()),
    }

    if idx < len(axes_flat):
        ax = axes_flat[idx]
        valid_bins = [(bp, bt, bc) for bp, bt, bc in
                      zip(bin_means_pred, bin_means_true, bin_counts)
                      if bc > 0 and not np.isnan(bt)]
        if valid_bins:
            bps, bts, bcs = zip(*valid_bins)
            sizes = [max(20, min(200, c * 5)) for c in bcs]
            ax.scatter(bps, bts, s=sizes, c='steelblue', alpha=0.7, edgecolors='black', zorder=3)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        ax.set_xlabel('Mean predicted prob')
        ax.set_ylabel('Observed proportion')
        ax.set_title(f'{held_out}\nBrier={brier:.3f}, ECE={ece:.3f}, Range={pred_range:.3f}')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8)

for idx in range(len(unique_datasets), len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle('Calibration Curves (LODO-CV, LR no covariates)', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'calibration_curves.png', dpi=150)
plt.close(fig)

# Summary
print(f"\n--- CALIBRATION SUMMARY ---", flush=True)
poor_discrimination = []
for ds, res in calibration_results.items():
    if 'skipped' in res:
        continue
    narrow = res['pred_range'] < 0.1
    if narrow:
        poor_discrimination.append(ds)
    status = "POOR DISCRIMINATION" if narrow else "ok"
    print(f"  {ds:15s}: Brier={res['brier_score']:.3f}, ECE={res['ece']:.3f}, "
          f"range={res['pred_range']:.3f} -> {status}", flush=True)

calibration_results['_summary'] = {
    'poor_discrimination_datasets': poor_discrimination,
    'mean_brier': round(float(np.mean([v['brier_score'] for v in calibration_results.values()
                                        if isinstance(v, dict) and 'brier_score' in v])), 4),
    'mean_ece': round(float(np.mean([v['ece'] for v in calibration_results.values()
                                      if isinstance(v, dict) and 'ece' in v])), 4),
}

with open(OUT / 'calibration_results.json', 'w') as f:
    json.dump(calibration_results, f, indent=2)

print(f"\n" + "=" * 70, flush=True)
print("FINAL SUMMARY", flush=True)
print("=" * 70, flush=True)

with open(OUT / 'bootstrap_cis.json') as f:
    bootstrap = json.load(f)

cannot_reject = bootstrap.get('_summary', {}).get('cannot_reject_random', [])
can_reject = bootstrap.get('_summary', {}).get('significantly_above_random', [])
print(f"\nPart A - Bootstrap CIs:", flush=True)
print(f"  Cannot reject random: {cannot_reject}", flush=True)
print(f"  Significantly above random: {can_reject}", flush=True)

print(f"\nPart B - Permutation tests:", flush=True)
int_res = perm_results.get('interaction_permutation', {})
print(f"  I-SPY2 interaction (all): perm p={int_res.get('all_ispy2', {}).get('permutation_p_onesided', 'N/A')}", flush=True)
print(f"  I-SPY2 interaction (GSE194040): perm p={int_res.get('gse194040', {}).get('permutation_p_onesided', 'N/A')}", flush=True)
print(f"  LODO-CV overall: perm p={perm_p_lodo:.6f}", flush=True)

print(f"\nPart C - Calibration:", flush=True)
print(f"  Poor discrimination: {poor_discrimination}", flush=True)
print(f"  Mean Brier: {calibration_results['_summary']['mean_brier']:.4f}", flush=True)
print(f"  Mean ECE: {calibration_results['_summary']['mean_ece']:.4f}", flush=True)

print(f"\nAll results saved to {OUT}", flush=True)
print("DONE.", flush=True)
