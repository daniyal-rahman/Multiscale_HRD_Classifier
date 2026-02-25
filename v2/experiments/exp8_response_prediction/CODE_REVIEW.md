# Code Review: Exp8 Drug Response Prediction Scripts

**Reviewer:** Claude (automated code review agent)
**Date:** 2026-02-24
**Scope:** 11 scripts in `exp8_response_prediction/`
**Focus:** Data leakage, statistical correctness, consistency, reproducibility

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 1     |
| MAJOR    | 5     |
| MINOR    | 8     |
| NITPICK  | 5     |

Overall the codebase is well-structured and the core LODO-CV logic is correct across all scripts. The most serious issue is a data leakage bug in the batch residualization code. Several reproducibility issues (hardcoded `/tmp/` paths, runtime downloads) should be addressed before publication.

---

## CRITICAL

### C1. Residualization data leakage in nested CV v2
**File:** `tuning/run_nested_cv_v2.py`, lines 68-78
**What:** The `residualize_by_dataset()` function computes per-dataset feature means on ALL samples (including future test-set samples) before the LODO-CV loop. For the held-out dataset, the centering uses the test set's own statistics, which constitutes data leakage.

```python
# Line 78: applied to ALL data before LODO loop
X_resid = residualize_by_dataset(X_raw, datasets_all)
```

In LODO-CV, the entire held-out dataset is the test set. The residualization centers each dataset using its own mean, meaning the test set is preprocessed using test-set statistics. In a truly prospective evaluation, these statistics would be unavailable.

**Practical impact:** Likely small (only affects centering, not rank ordering), but methodologically unsound. Any results comparing "raw" vs "resid" features are tainted.

**Fix:** Move residualization inside the LODO loop. For training datasets, compute means from training data only. For the held-out dataset, either (a) don't residualize, or (b) use the global training mean as a proxy:

```python
def residualize_lodo(X, datasets, held_out):
    X_res = X.copy()
    for ds in np.unique(datasets):
        mask = datasets == ds
        if ds == held_out:
            # Use global training mean or skip
            continue
        ds_mean = X[mask].mean(axis=0)
        X_res[mask] -= ds_mean
        X_res[mask] += 0.5
    return X_res
```

---

## MAJOR

### M1. Non-reproducible BRCA mutation data loaded from `/tmp/`
**Files:** `brca_comparison/run_brca_comparison.py` line 63; `validation_fixed/run_multivariate_survival.py` line 119
**What:** Both scripts load BRCA mutation data from `/tmp/ov_tcga_combined_brca.json`, a non-persistent location. This file will be lost on reboot, making the analysis non-reproducible.

```python
with open('/tmp/ov_tcga_combined_brca.json') as f:
    brca_data = json.load(f)
```

**Fix:** Move the file to the repository's `data_acquisition/response_data/` directory and update the path.

### M2. Runtime internet download in multivariate survival script
**File:** `validation_fixed/run_multivariate_survival.py`, lines 100-102
**What:** The Xena clinical matrix is downloaded at runtime from an AWS S3 URL. If the server is unavailable or the URL changes, the script fails.

```python
url = 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.OV.sampleMap%2FOV_clinicalMatrix'
r = requests.get(url, timeout=60)
```

**Fix:** Download once, save to the data directory, and load locally. Add a comment with the source URL and download date.

### M3. Inconsistent use of original vs fixed pooled matrix across scripts
**Files:** Multiple
**What:** Some scripts use the original `pooled_rank_matrix.parquet` (11,140 genes, pre-GSE32062 fix) while others use the fixed version `pooled_rank_matrix_fixed.parquet` (11,089 genes). Scripts in the same analysis pipeline use different versions:

| Script | Matrix Used |
|--------|-------------|
| `run_exp8.py` | Creates original |
| `run_brca_comparison.py` | Original |
| `run_immune_baselines.py` | Original |
| `run_immune_deconv.py` (gene signature) | Original |
| `run_softhrd_comparison.py` | Original |
| `run_validation_tuned.py` | Original |
| `rebuild_with_fixed_gse32062.py` | Creates fixed |
| `run_cell_type_deconv.py` (CIBERSORT) | **Fixed** |

This means the CIBERSORT deconvolution analysis uses different gene features than the BRCA comparison, immune baselines, and softHRD comparison.

**Fix:** Decide on a canonical matrix (likely the fixed one) and update all downstream scripts to use it. Or explicitly document which analyses use which version and why.

### M4. BRCA comparison uses unfair training data volumes
**File:** `brca_comparison/run_brca_comparison.py`
**What:** The comparison between response-trained and BRCA-trained models has a systematic advantage for the response model:

- **Response model:** ~800+ training samples from 8-9 datasets (LODO-CV per fold)
- **BRCA model:** ~280 training samples from 2 datasets (TCGA-OV + GSE63885)

The response model has 3x more training data AND trains directly on the outcome being evaluated (drug response). The BRCA model is trained on a proxy label (mutation status) and has far fewer labeled examples.

This is acknowledged in the docstring but not quantified in the results. The comparison doesn't tell you whether response labels are inherently better than BRCA labels -- it conflates label quality with training data volume.

**Fix:** Add a matched-volume comparison: train the response model on only TCGA-OV + GSE63885 (same datasets as the BRCA model) and compare. This would isolate the effect of label type from training data volume.

### M5. run_exp8.py uses C=1.0 (default) while all downstream scripts use C=0.01 (tuned)
**File:** `run_exp8.py`, lines 228, 311, 458, 468, 474
**What:** The main experiment script uses `C=1.0` for all logistic regression models. All subsequent analysis scripts (validation, comparison, etc.) correctly use `C=0.01`. The `exp8_results.json` output contains C=1.0 results, which could be confused with the tuned results.

This is by design (run_exp8.py was the initial exploration, tuning came later), but the exp8_results.json is misleading if cited without context.

**Fix:** Either (a) add a clear warning at the top of exp8_results.json, or (b) re-run run_exp8.py with C=0.01, or (c) create a separate "final" results file that consolidates tuned results.

---

## MINOR

### m1. ECE calculation misses predictions exactly equal to 1.0
**Files:** `validation/run_validation_tuned.py` lines 270-271; `rebuild_with_fixed_gse32062.py` lines 325-326
**What:** The last bin uses strict `<` for the upper bound:
```python
in_bin = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
```
If any prediction equals exactly 1.0, it falls outside all bins. Logistic regression outputs are strictly in (0,1), so this likely never triggers, but it's technically incorrect.

**Fix:** Use `<=` for the last bin's upper bound.

### m2. Unclosed file handles in multiple scripts
**Files:** `run_exp8.py` line 61; `rebuild_with_fixed_gse32062.py` line 44; `brca_comparison/run_brca_comparison.py` line 51; `immune_deconv/run_immune_deconv.py` line 50; `validation_fixed/run_multivariate_survival.py` line 58; `softhrd_comparison/run_softhrd_comparison.py` line 47
**What:** Common gene files are opened without a `with` statement:
```python
common_genes = [g.strip() for g in open(BASE / "common_genes_all_10_datasets.txt")]
```

**Fix:** Use `with open(...) as f:` or `Path(...).read_text().splitlines()`.

### m3. run_exp8.py dataset one-hot encoding includes held-out dataset column
**File:** `run_exp8.py`, lines 202-211
**What:** The one-hot encoding creates columns for ALL datasets, including the held-out one. During training, the held-out column is always zero; during testing, it's always one. The model learns a near-zero coefficient for this column (driven by L2 regularization), which then gets applied to test samples. This wastes a feature dimension and adds noise.

**Fix:** Use N-1 dummy coding, or exclude the held-out dataset's column. Or simply note that this only affects the initial C=1.0 results (the covariate model isn't used in downstream scripts).

### m4. Single-gene baselines may show AUC < 0.5 without interpretation
**File:** `immune_baselines/run_immune_baselines.py`
**What:** The `run_lodo_score` function computes raw AUC for single-gene scores. If a gene is inversely associated with response (higher expression = less response), AUC < 0.5. The script doesn't flag or interpret this -- an AUC of 0.3 is actually a STRONG predictor (just in the reverse direction), not a poor one.

**Fix:** Either flip scores when AUC < 0.5 (and note the direction), or add interpretation text explaining that AUC < 0.5 indicates inverse association.

### m5. Fragile stage encoding using substring matching
**File:** `validation_fixed/run_multivariate_survival.py`, lines 155-160
**What:** Stage classification uses `'III' in stage_raw` and `'I' in stage_raw`:
```python
if 'III' in stage_raw or 'IV' in stage_raw:
    stage_advanced = 1
elif 'I' in stage_raw or 'II' in stage_raw:
    stage_advanced = 0
```
This works because 'III'/'IV' are checked first, but the `elif` branch matches ANY string containing 'I' -- including unexpected values like "Invalid" or "Missing".

**Fix:** Use explicit stage value matching:
```python
if any(s in stage_raw for s in ['Stage III', 'Stage IV']):
    stage_advanced = 1
elif any(s in stage_raw for s in ['Stage I', 'Stage II']):
    stage_advanced = 0
```

### m6. run_validation_tuned.py gene_cols derived by exclusion
**File:** `validation/run_validation_tuned.py`, line 32
**What:** Gene columns are derived by excluding known metadata columns:
```python
gene_cols = [c for c in df.columns if c not in meta_cols]
```
If the pooled matrix ever gains additional non-gene columns, they would silently be included as features. Other scripts use the explicit gene list from `common_genes_all_10_datasets.txt`.

**Fix:** Load the explicit gene list instead of deriving by exclusion. This pattern appears in several scripts (`run_immune_baselines.py` line 48, `run_nested_cv.py` line 44, `run_nested_cv_v2.py` line 41).

### m7. run_nested_cv_v2.py confusing variable re-assignment
**File:** `tuning/run_nested_cv_v2.py`, lines 185-186
**What:**
```python
train_mask = datasets == held_out  # This is actually the TEST mask!
train_mask = ~train_mask           # Now it's the train mask
```
The variable is first assigned the test mask, then immediately negated. This is confusing and error-prone.

**Fix:**
```python
test_mask = datasets == held_out
train_mask = ~test_mask
```

### m8. run_brca_comparison.py: BRCA prediction sanity check uses training data predictions for TCGA-OV
**File:** `brca_comparison/run_brca_comparison.py`, lines 462-481
**What:** The BRCA prediction sanity check (Step 7) evaluates both models' ability to predict BRCA mutation status on `surv_df`, which contains TCGA-OV patients. The BRCA model was trained on these very patients (BRCA labels from TCGA-OV). So the BRCA model's high AUC on this sanity check is partially due to training-set evaluation, not true generalization.

The response model's scores come from LODO-CV (held out from TCGA-OV training), so its evaluation is fair. But the BRCA model's scores come from the LODO loop where TCGA-OV overlap is excluded... actually, let me recheck.

Looking at lines 239-256: When TCGA-OV is held out, and there's overlap with BRCA training data, the BRCA model IS retrained excluding those samples. So the BRCA model predictions for TCGA-OV also come from a held-out model. However, the BRCA model still trained on GSE63885 BRCA labels, which isn't held out. For the specific case of TCGA-OV BRCA prediction, the BRCA model's predictions ARE out-of-sample (retrained excluding TCGA-OV). So this is actually OK on closer inspection.

**Severity revised: Not a bug, but worth adding a comment explaining this in the code.**

---

## NITPICK

### n1. Dead code: tuple isinstance check in multivariate survival
**File:** `validation_fixed/run_multivariate_survival.py`, lines 371-372 (and similar for m2-m6)
**What:** `fit_cox_model()` always returns a 2-tuple `(dict, cph_or_None)`. After unpacking, the code checks `isinstance(m1_result, tuple)`, which is always False since `m1_result` is already a dict.
```python
m1_result, m1_cph = fit_cox_model(...)
if isinstance(m1_result, tuple):  # This is never True
    m1_result, m1_cph = m1_result
```

**Fix:** Remove the dead isinstance checks.

### n2. Redundant path construction
**File:** `rebuild_with_fixed_gse32062.py`, line 28
**What:** `FIX = BASE.parent / "response_data" / "gse32062_fix"` where `BASE` is already `.../response_data`. This simplifies to `BASE / "gse32062_fix"`.

**Fix:** `FIX = BASE / "gse32062_fix"`

### n3. No global random seed in several scripts
**Files:** `run_exp8.py`, `run_brca_comparison.py`, `run_immune_baselines.py`, `run_immune_deconv.py`, `run_multivariate_survival.py`, `run_cell_type_deconv.py`, `run_nested_cv.py`, `run_nested_cv_v2.py`
**What:** These scripts don't call `np.random.seed(42)` at the top. All sklearn models use `random_state=42`, which covers model reproducibility. But any numpy random operations (bootstrap, permutation) in scripts without a global seed would be non-reproducible across runs.

In practice, the scripts that DO use numpy random (bootstrap/permutation tests) all set the global seed. Scripts that only use sklearn don't need it. So this is not a functional issue.

### n4. Hardcoded date strings
**Files:** Multiple scripts contain `'date': '2026-02-23'` or `'date': '2026-02-24'`
**What:** Dates are hardcoded rather than generated dynamically.

**Fix:** Use `datetime.date.today().isoformat()` if accurate dates matter.

### n5. warnings.filterwarnings('ignore') in all scripts
**Files:** All 11 scripts
**What:** All warnings are suppressed globally. This can hide convergence warnings from sklearn or deprecation warnings that might indicate bugs.

**Fix:** Use more specific warning filters, e.g., `warnings.filterwarnings('ignore', category=ConvergenceWarning)`.

---

## Checks with No Issues Found

The following areas were reviewed and found to be correct:

1. **LODO-CV split logic:** All scripts correctly implement leave-one-dataset-out. Train/test splits are clean with no sample overlap.

2. **Data leakage (main pipeline):** No test-set information leaks into training in the core LODO-CV loops. NaN imputation uses a constant (0.5), not data-derived statistics.

3. **Rank transformation:** Correctly applied within-sample (axis=1) with `pct=True`. Ties handled by pandas default averaging.

4. **class_weight='balanced':** Applied consistently in all LogisticRegression models. LightGBM equivalently uses `scale_pos_weight`.

5. **Permutation tests:** Labels shuffled correctly (within-dataset for LODO permutation, treatment-arm for interaction test). P-values computed correctly (one-sided for AUC > chance, two-sided for interaction and paired comparisons).

6. **Bootstrap CIs:** Standard percentile method, correctly skipping iterations with single-class resamples. 2000 iterations is adequate.

7. **Survival event coding:** OS and DFS events parsed reasonably from clinical data strings. Censored observations correctly coded as 0.

8. **Model A mask logic:** Consistently applied across all 11 scripts. Correctly excludes cell lines and non-PARPi I-SPY2 controls.

9. **Nested CV nesting:** Inner 5-fold CV properly contained within each outer LODO fold. No information leakage between inner and outer loops.

10. **GSE32062 transpose fix:** The `expr.T` correctly converts the genes-by-samples format to samples-by-genes before rank transformation and ID matching.

---

## Recommendations for Publication Readiness

1. **Move BRCA data from /tmp/ to the repo** (M1) -- immediate fix needed
2. **Cache Xena download locally** (M2) -- for reproducibility
3. **Standardize on the fixed pooled matrix** (M3) -- or clearly document the difference
4. **Add matched-volume BRCA comparison** (M4) -- strengthens the argument
5. **Fix residualization leakage or drop the residualized results** (C1) -- if citing them
6. **Re-generate exp8_results.json with C=0.01** or create a final consolidated results file (M5)
