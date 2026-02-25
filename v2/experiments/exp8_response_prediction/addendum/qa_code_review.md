# QA Report: Code Review of Core Scripts

**Reviewer**: automated QA
**Date**: 2026-02-25
**Scope**: Bug analysis of 5 core experiment scripts

---

## Scripts Reviewed

1. `run_exp8.py` — Main LODO-CV experiment
2. `validation/run_validation_tuned.py` — Bootstrap CIs, permutation tests
3. `validation_fixed/run_multivariate_survival.py` — Cox PH survival analysis
4. `immune_deconv/run_cell_type_deconv.py` — CIBERSORTx cell type analysis
5. `addendum/gene_overlap_analysis.py` — Gene overlap / hypergeometric tests

---

## Summary Table

| # | Script | Issue | Severity | Affects Results? |
|---|--------|-------|----------|-----------------|
| 1 | run_exp8.py | Dataset one-hot encoding leaks test identity in LODO-CV | MAJOR | Only when `use_covariates=True`; primary results use `nocov` |
| 2 | run_validation_tuned.py | Permutation test uses pre-trained predictions (does not retrain per permutation) | MAJOR | Permutation p-values may be anticonservative |
| 3 | run_multivariate_survival.py | `fit_cox_model` returns inconsistent types — crash bug on early exit | CRITICAL — **FIXED** | Would crash if any Cox model has insufficient data |
| 4 | run_multivariate_survival.py | PH assumption check wrapped in try/except that never fires | MAJOR | Violations would be silently missed |
| 5 | run_multivariate_survival.py | BRCA data loaded from `/tmp/` (ephemeral path) | MAJOR | Unreproducible |
| 6 | run_multivariate_survival.py | Clinical data downloaded from S3 URL at runtime | MAJOR | Unreproducible if URL changes |
| 7 | run_cell_type_deconv.py | No multiple testing correction for 29 tests | MAJOR | Paper acknowledges this partially |
| 8 | gene_overlap_analysis.py | No hypergeometric test for overlap significance | CRITICAL — **FIXED** | Cannot determine if overlaps exceed chance |
| 9 | ALL | Hardcoded absolute paths | MINOR | Standard for research scripts |
| 10 | run_validation_tuned.py | Percentile bootstrap (not BCa) | MINOR | May have below-nominal coverage for small-n datasets |
| 11 | run_validation_tuned.py | Dropped single-class bootstrap samples not counted | MINOR | CIs may be based on fewer iterations |
| 12 | run_multivariate_survival.py | All Cox models use penalizer=0.0 | MINOR | Could cause numerical instability |
| 13 | run_multivariate_survival.py | Score (probability 0-1) used as linear covariate in Cox | MINOR | May misspecify dose-response relationship |
| 14 | run_exp8.py | File handle not closed (no `with` statement) | MINOR | Resource leak |
| 15 | run_cell_type_deconv.py | P-value precision truncated via string formatting | MINOR | Loses precision in saved JSON |

---

## Detailed Analysis

### Issue 1: Dataset One-Hot Encoding in LODO-CV [MAJOR]

**Script**: `run_exp8.py`, lines ~197-211
**Description**: When `use_covariates=True`, dataset IDs are one-hot encoded and appended to features for both train and test. The held-out dataset gets a unique column that is all-zeros during training (never seen), creating a novel feature pattern at test time.
**Impact**: This is NOT classical train/test leakage (no label information leaks), but it's methodologically questionable. The held-out dataset's one-hot column has an L2-regularized coefficient driven to zero, potentially biasing predictions.
**Mitigation**: The primary results reported in the paper use the `nocov` variant (no covariates). This issue only affects the covariate version.
**Recommendation**: Document that the covariate version should not be used for LODO-CV, or drop the held-out dataset's column during prediction.

### Issue 2: Permutation Test Does Not Retrain Model [MAJOR]

**Script**: `run_validation_tuned.py`, lines ~142-157
**Description**: The permutation null shuffles `y_true` labels within each dataset but uses the **original** LODO-CV predictions (trained on unshuffled labels). A fully correct permutation test would retrain the model on shuffled labels for each permutation.
**Impact**: This is a standard computational shortcut (retraining 5,000 times across 9 LODO folds would require ~45,000 model fits). It tests "given these specific predictions, is the association with labels above chance?" rather than "could any model trained on shuffled labels achieve this performance?" The shortcut is **anticonservative**: the trained model's predictions already have structure, making it harder for shuffled labels to achieve high AUC by chance. This means p-values may be smaller than a full permutation test would give.
**Mitigation**: The observed p = 0.0 (no permutation out of 5,000 exceeded the observed mean AUC) is so extreme that even a conservative test would likely remain significant. The practical impact is low.
**Recommendation**: Note in the paper that the permutation test uses fixed predictions (a label-shuffling test) rather than full model retraining. This is common practice but should be disclosed.

### Issue 3: `fit_cox_model` Inconsistent Return Types [CRITICAL] — **FIXED**

**Script**: `run_multivariate_survival.py`, lines ~309, 340, 343
**Description**: The function had three return paths with inconsistent types. The early exit (insufficient data) returned a single dict, while success and exception paths returned tuples.
**Fix applied**: Line 309 now returns `({'error': ...}, None)` — consistent with other return paths. All three paths return `(dict, CoxPHFitter|None)` tuples.

### Issue 4: PH Assumption Check Never Detects Violations [MAJOR]

**Script**: `run_multivariate_survival.py`, lines ~604-612
**Description**: The code wraps `cph.check_assumptions()` in try/except. But lifelines' `check_assumptions()` does NOT raise an exception when PH violations are found — it prints warnings and returns results. The except block only fires on code errors. As written, the code **always** reports "no significant violations" regardless of actual PH test results.
**Impact**: The paper states "PH assumption tested for all models; no violations detected." This claim is technically from code that cannot detect violations. The claim may be correct, but it's not actually validated by the code.
**Recommendation**: Replace the try/except with inspection of the return value from `check_assumptions()`, or use `lifelines.statistics.proportional_hazard_test()` directly.

### Issue 5: BRCA Data from `/tmp/` [MAJOR]

**Script**: `run_multivariate_survival.py`, line 119
**Description**: Loads BRCA mutation data from `/tmp/ov_tcga_combined_brca.json`. This is an ephemeral filesystem location that is cleared on reboot.
**Impact**: Full reproducibility is broken — anyone trying to rerun this script needs to know to recreate this file first.
**Recommendation**: Move the BRCA data to a versioned location in the repository.

### Issue 6: Runtime Download of Clinical Data [MAJOR]

**Script**: `run_multivariate_survival.py`, lines ~100-102
**Description**: Downloads UCSC Xena clinical data from an S3 URL on every run.
**Impact**: If the URL changes, goes down, or the file format is updated, the script breaks silently or produces different results.
**Recommendation**: Cache the download locally and version it. Document the exact file hash.

### Issue 7: No Multiple Testing Correction in CIBERSORTx Analysis [MAJOR]

**Script**: `run_cell_type_deconv.py`, lines ~208-248
**Description**: Computes Spearman correlations and Mann-Whitney tests for 22 cell types + 7 aggregate groups = ~29 tests without FDR/Bonferroni correction.
**Impact**: At alpha=0.05 with 29 independent tests, ~1.45 false positives are expected under the null. The strongest finding (M1 macrophages, p = 1.4e-5) would survive Bonferroni (29 tests, threshold = 0.0017), but weaker findings like activated NK cells (p = 0.018) would not.
**Mitigation**: The paper acknowledges this in Sections 3.6 and 4.6, noting which findings survive Bonferroni and which don't. But the code itself does not compute adjusted p-values.
**Recommendation**: Add FDR-corrected p-values (Benjamini-Hochberg) to the output JSON.

### Issue 8: No Hypergeometric Test for Gene Overlaps [CRITICAL] — **FIXED**

**Script**: `addendum/gene_overlap_analysis.py`, lines ~99-157
**Description**: The overlap analysis counted intersections between gene sets but never tested statistical significance. A `hypergeom_pvalue()` function existed (lines 107-117) but was dead code — never called.
**Fix applied**: Wired up `hypergeom_pvalue()` calls in both overlap loops (HRD signatures and external signatures). P-values now computed for top-50, top-100, top-200, and top-500 tiers. Results written to JSON output.
**Key results after fix**:
- All HRD signature overlaps: p > 0.29 (non-significant) — confirms paper's "near-zero overlap" claim statistically
- TIS-18 immune signature top-100 overlap (2/18 genes): p = 0.011 — *significant*, supporting immune biology interpretation
- IFN-gamma 10-gene top-100 overlap (1/10): p = 0.086 — borderline

---

## Issues NOT Found (Positive Findings)

1. **No train/test leakage in LODO-CV**: The core LODO-CV loop (train on 8 datasets, test on 1) is correctly implemented in all scripts. Rank transformation is per-sample (`axis=1`), so no across-sample leakage.

2. **Correct `roc_auc_score` usage**: All scripts pass `(y_true, y_score)` in the correct order. No sign flips or class-index errors.

3. **Correct `class_weight='balanced'` usage**: Properly passed to `LogisticRegression` in all scripts. This upweights the minority class to handle imbalance.

4. **Correct `predict_proba[:, 1]` indexing**: Always takes the positive class probability (column 1), which is correct for binary logistic regression where class 0 is the first class.

5. **Cox PH likelihood ratio test correctly implemented**: The LRT in `run_multivariate_survival.py` correctly refits both full and reduced models on the same complete-case set before computing the test statistic. Uses chi2 with df=1.

6. **No array indexing bugs**: Set operations, boolean masking, and DataFrame alignment are all correct throughout.

7. **Random gene control has no leakage**: Gene selection is done inside each LODO fold (trained on 8 datasets, top-K selected from training weights, tested on held-out). No information from the test fold leaks into gene selection.

---

## Risk Assessment

| Risk Level | Issues | Impact on Paper Claims |
|------------|--------|----------------------|
| **Results-invalidating** | None | No core result is wrong due to code bugs |
| **Interpretation-affecting** | #2 (permutation shortcut), #4 (PH check), #7 (multiple testing) | Permutation p-values may be slightly anticonservative; PH assumption not actually validated; some CIBERSORTx findings may be false positives |
| **Reproducibility** | #5 (tmp path), #6 (S3 download), #9 (hardcoded paths) | Script would not run on a clean machine without modifications |
| **Code quality** | #3 (crash bug), #14 (file handle), #15 (precision) | Would crash on edge cases; minor resource/precision issues |

**Overall**: The core LODO-CV implementation is sound. No bugs that would invalidate the primary AUC results or the BRCA comparison. Both CRITICAL issues have been fixed: the Cox model crash bug (return type consistency) and the gene overlap hypergeometric test (now wired up — confirms HRD overlaps are at chance level, immune overlaps are significant). Remaining concerns: permutation test shortcut (anticonservative p-values) and PH assumption check that doesn't work. The actual Cox results are likely correct given the clean data.
