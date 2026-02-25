#!/usr/bin/env python3
"""
Corrected validation: Bootstrap CIs + permutation tests + calibration
using C=0.01 (tuned model) instead of C=1.0 (default).
Also excludes GSE28739 and GSE32062 from the clean analysis.
"""
import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = Path("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction")
OUT = BASE / "validation"
OUT.mkdir(parents=True, exist_ok=True)

C_OPT = 0.01
EXCLUDE = ['GSE28739', 'GSE32062']
N_BOOTSTRAP = 2000
N_PERMUTATION = 5000

# Load data
print("Loading data...", flush=True)
df = pd.read_parquet(BASE / "pooled_rank_matrix.parquet")
meta_cols = ['response_binary', 'response_continuous', 'drug', 'dataset', 'category', 'cancer_type']
gene_cols = [c for c in df.columns if c not in meta_cols]

ispy2_parpi_drugs = ['durvalumab+olaparib', 'veliparib+carboplatin']
model_a_mask = (
    (df['category'] != 'cell_line') &
    ~((df['category'] == 'clinical_ispy2') & (~df['drug'].isin(ispy2_parpi_drugs)))
)
model_a = df[model_a_mask].copy()

X_all = np.nan_to_num(model_a[gene_cols].values, nan=0.5).astype(np.float32)
y_all = model_a['response_binary'].values.astype(int)
datasets_all = model_a['dataset'].values
sample_ids = model_a.index.tolist()

# Generate LODO predictions with C=0.01
print("Generating LODO predictions with C=0.01...", flush=True)
all_predictions = {}
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

    auc = roc_auc_score(y_all[test_mask], y_prob)
    print(f"  {held_out}: AUC={auc:.3f}, n={test_mask.sum()}", flush=True)

# ─── Bootstrap CIs (all 9 datasets) ───────────────────────────────
print(f"\nBootstrap CIs ({N_BOOTSTRAP} iterations)...", flush=True)
bootstrap_results = {}

for ds in sorted(model_a['dataset'].unique()):
    mask = datasets_all == ds
    ids = [sample_ids[i] for i in range(len(sample_ids)) if mask[i]]
    y_true = y_all[mask]
    y_pred = np.array([all_predictions[sid] for sid in ids])

    if len(np.unique(y_true)) < 2:
        continue

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
        'model': f'L2_C={C_OPT}',
    }
    print(f"  {ds}: AUC={point_auc:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
          f" {'*random*' if ci_lower <= 0.5 else 'SIGNIFICANT'}", flush=True)

# Summary
cannot_reject = [ds for ds, r in bootstrap_results.items() if r['includes_random']]
significant = [ds for ds, r in bootstrap_results.items() if not r['includes_random']]
all_aucs = [r['point_auc'] for r in bootstrap_results.values()]
clean_aucs = [r['point_auc'] for ds, r in bootstrap_results.items() if ds not in EXCLUDE]

bootstrap_results['_summary'] = {
    'model': f'L2_C={C_OPT}',
    'n_bootstrap': N_BOOTSTRAP,
    'cannot_reject_random': cannot_reject,
    'significantly_above_random': significant,
    'mean_auc_all_9': round(float(np.mean(all_aucs)), 4),
    'mean_auc_clean_7': round(float(np.mean(clean_aucs)), 4),
}

with open(OUT / "bootstrap_cis_tuned.json", 'w') as f:
    json.dump(bootstrap_results, f, indent=2)
print(f"\nSaved bootstrap CIs to {OUT / 'bootstrap_cis_tuned.json'}", flush=True)

# ─── Permutation test: overall LODO mean AUC ──────────────────────
print(f"\nPermutation test for overall LODO AUC ({N_PERMUTATION} perms)...", flush=True)

# Observed: per-dataset AUCs
obs_per_ds = {}
for ds in sorted(model_a['dataset'].unique()):
    mask = datasets_all == ds
    ids = [sample_ids[i] for i in range(len(sample_ids)) if mask[i]]
    y_true = y_all[mask]
    y_pred = np.array([all_predictions[sid] for sid in ids])
    if len(np.unique(y_true)) >= 2:
        obs_per_ds[ds] = roc_auc_score(y_true, y_pred)

obs_mean_all = np.mean(list(obs_per_ds.values()))
obs_mean_clean = np.mean([v for k, v in obs_per_ds.items() if k not in EXCLUDE])

# Permutation: shuffle labels within each dataset
null_means_all = []
null_means_clean = []
for perm in range(N_PERMUTATION):
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

permutation_results = {
    'model': f'L2_C={C_OPT}',
    'n_permutations': N_PERMUTATION,
    'overall_all_9': {
        'observed_mean_auc': round(float(obs_mean_all), 4),
        'null_mean': round(float(null_means_all.mean()), 4),
        'null_std': round(float(null_means_all.std()), 4),
        'permutation_p': round(float(perm_p_all), 4),
    },
    'overall_clean_7': {
        'observed_mean_auc': round(float(obs_mean_clean), 4),
        'null_mean': round(float(null_means_clean.mean()), 4),
        'null_std': round(float(null_means_clean.std()), 4),
        'permutation_p': round(float(perm_p_clean), 4),
    },
}

print(f"  All 9 datasets: observed={obs_mean_all:.3f}, null={null_means_all.mean():.3f}, p={perm_p_all}", flush=True)
print(f"  Clean 7 datasets: observed={obs_mean_clean:.3f}, null={null_means_clean.mean():.3f}, p={perm_p_clean}", flush=True)

# ─── Permutation: I-SPY2 interaction (if applicable) ──────────────
print("\nPermutation test for treatment interaction...", flush=True)
import statsmodels.api as sm

# Load I-SPY2 predictions
ispy2_mask = model_a['category'] == 'clinical_ispy2'
ispy2_data = model_a[ispy2_mask].copy()
ispy2_ids = ispy2_data.index.tolist()
ispy2_scores = np.array([all_predictions[sid] for sid in ispy2_ids])
ispy2_data['score'] = ispy2_scores
ispy2_data['is_parpi'] = ispy2_data['drug'].isin(ispy2_parpi_drugs).astype(int)

# Interaction test per I-SPY2 dataset (GSE173839, GSE194040)
interaction_results = {}
for ds_name in ['GSE173839', 'GSE194040']:
    ds_mask = ispy2_data['dataset'] == ds_name
    ds_data = ispy2_data[ds_mask].copy()
    if len(ds_data) < 20:
        continue

    y = ds_data['response_binary'].values.astype(float)
    score = ds_data['score'].values
    is_parpi = ds_data['is_parpi'].values.astype(float)
    interaction = score * is_parpi

    X = np.column_stack([score, is_parpi, interaction])
    X = sm.add_constant(X)

    try:
        model = sm.Logit(y, X).fit(disp=0)
        obs_coef = model.params[3]  # interaction coefficient
        obs_pval = model.pvalues[3]
    except Exception:
        continue

    # Permute treatment arm labels
    perm_coefs = []
    for _ in range(N_PERMUTATION):
        perm_parpi = is_parpi.copy()
        np.random.shuffle(perm_parpi)
        perm_inter = score * perm_parpi
        X_perm = np.column_stack([score, perm_parpi, perm_inter])
        X_perm = sm.add_constant(X_perm)
        try:
            m = sm.Logit(y, X_perm).fit(disp=0, maxiter=100)
            perm_coefs.append(m.params[3])
        except Exception:
            pass

    perm_coefs = np.array(perm_coefs)
    perm_p = np.mean(np.abs(perm_coefs) >= np.abs(obs_coef))

    interaction_results[ds_name] = {
        'observed_coef': round(float(obs_coef), 4),
        'observed_pval_parametric': round(float(obs_pval), 6),
        'permutation_p_twosided': round(float(perm_p), 4),
        'n_valid_perms': len(perm_coefs),
        'n_samples': len(ds_data),
    }
    print(f"  {ds_name}: coef={obs_coef:.3f}, parametric_p={obs_pval:.4f}, perm_p={perm_p:.4f}", flush=True)

permutation_results['interaction'] = interaction_results

with open(OUT / "permutation_tuned.json", 'w') as f:
    json.dump(permutation_results, f, indent=2)

# ─── Calibration ──────────────────────────────────────────────────
print("\nCalibration analysis...", flush=True)
calibration_results = {}

for ds in sorted(model_a['dataset'].unique()):
    mask = datasets_all == ds
    ids = [sample_ids[i] for i in range(len(sample_ids)) if mask[i]]
    y_true = y_all[mask]
    y_pred = np.array([all_predictions[sid] for sid in ids])

    if len(np.unique(y_true)) < 2:
        continue

    brier = brier_score_loss(y_true, y_pred)

    # ECE with 10 bins
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

# Summary
all_ece = [r['ece'] for r in calibration_results.values()]
clean_ece = [r['ece'] for ds, r in calibration_results.items() if ds not in EXCLUDE]
calibration_results['_summary'] = {
    'model': f'L2_C={C_OPT}',
    'mean_ece_all': round(float(np.mean(all_ece)), 4),
    'mean_ece_clean': round(float(np.mean(clean_ece)), 4),
    'mean_brier_all': round(float(np.mean([r['brier_score'] for r in calibration_results.values() if 'brier_score' in r])), 4),
}

with open(OUT / "calibration_tuned.json", 'w') as f:
    json.dump(calibration_results, f, indent=2)

print(f"\nMean ECE (all 9): {np.mean(all_ece):.3f}", flush=True)
print(f"Mean ECE (clean 7): {np.mean(clean_ece):.3f}", flush=True)

print("\nAll validation results saved with C=0.01 tuned model.", flush=True)
print("Done!", flush=True)
