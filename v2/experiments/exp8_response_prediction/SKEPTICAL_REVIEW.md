# Skeptical Scientific Review: "Training on Drug Response, Not Genotype"

**Reviewer:** Adversarial Reviewer (AI-generated review for internal quality control)
**Date:** 2026-02-24
**Overall recommendation:** Major revision required

---

## Summary

This manuscript describes an L2-regularized logistic regression model trained on clinical drug response labels (rather than BRCA/HRD genotype) across 9 platinum-treated cancer datasets (mostly ovarian, n=904). The model achieves a mean AUC of 0.693 via leave-one-dataset-out cross-validation (LODO-CV), outperforms a BRCA-label-trained model (0.693 vs 0.555), and is interpreted as capturing immune microenvironment biology rather than DNA repair. DFS survival prediction in TCGA-OV is strong (multivariate HR=0.098, p=0.0002).

The study has a genuinely interesting premise and the LODO-CV framework is laudable. However, I identify several critical and major issues with comparison fairness, statistical practice, biological interpretation, and internal consistency that must be addressed before publication.

---

## 1. Statistical Rigor

### 1.1 Multiple testing correction is absent [MAJOR]

**Concern:** The manuscript reports p-values across at least the following distinct analyses: 9 per-dataset bootstrap CIs, permutation test, 22 CIBERSORT cell type correlations, 7+ aggregate cell group correlations, 14 pathway correlations, 6+ Cox models (DFS and OS), multiple comparison methods (softHRD, BRCA, immune baselines). Conservatively, there are 60+ nominal p-values reported. No formal multiple testing correction is applied anywhere.

The justification offered --- "the LODO-CV procedure inherently controls for overfitting" and "the permutation test provides an omnibus test" --- is partially valid for the primary LODO-CV comparison but does NOT extend to the secondary analyses (immune deconvolution, pathway correlations, survival models). The CIBERSORT analysis alone tests 22 cell types; claiming M1 macrophages are "significant" at p=1.39e-5 without noting the multiple comparison context is misleading (though this one would survive Bonferroni correction at 22 tests, many of the other "significant" findings like activated NK cells at p=0.018 would not).

**Severity:** MAJOR
**Action:** Apply Benjamini-Hochberg FDR correction to all secondary analyses. Report both nominal and adjusted p-values. Acknowledge explicitly which findings survive correction.

### 1.2 Permutation test design: within-dataset shuffling may be too conservative or misspecified [MINOR]

**Concern:** The permutation test shuffles labels within each dataset, preserving dataset structure and class balance. This is the correct null for "does the model predict response above chance?" But it does NOT test the more interesting null: "does training on response labels outperform training on random labels, given the cross-dataset structure?" Because all 9 datasets contribute to both training and testing across LODO folds, the permutation test implicitly tests whether response labels are informative across ALL datasets jointly. This is fine, but the paper should be explicit that the p < 0.001 applies to the aggregate, not to individual datasets.

**Severity:** MINOR
**Action:** Clarify in methods that the permutation p-value is an omnibus test and that individual dataset results should be interpreted descriptively.

### 1.3 Bootstrap CIs: percentile method may undercover [MINOR]

**Concern:** The 2,000-iteration percentile bootstrap is adequate for point estimates but can have poor coverage properties for AUC, especially with small n and extreme class imbalance (GSE32062: 225/35). The bias-corrected and accelerated (BCa) method would be more appropriate. For GSE32062 with pred_std = 0.0008, the bootstrap is essentially resampling near-identical predictions, making the CI meaningless.

**Severity:** MINOR
**Action:** Report BCa intervals, or at minimum note the limitation. For GSE32062 specifically, the bootstrap CI [0.391, 0.588] is computed on predictions with essentially zero variance (SD=0.0008), which undermines its validity.

### 1.4 The paper selectively highlights significant p-values [MAJOR]

**Concern:** The survival analysis is presented asymmetrically:
- DFS multivariate: HR=0.098, p=0.0002 --- headline result, prominent figure
- OS multivariate: HR=0.385, p=0.167 --- buried in text with "weaker than DFS, which is expected"

The OS likelihood ratio test explicitly says "Adding the score does NOT significantly improve the model" (p=0.166 from `multivariate_survival.json`). Yet the paper's Discussion section claims the model predicts "clinical outcomes" without qualification. The abstract mentions DFS HR but doesn't prominently note the OS failure.

Additionally, the paper reports univariate OS p=0.037 and then separately reports multivariate OS p=0.167, but the reader's attention is steered toward the univariate result by presenting it first. The clinically relevant result is the multivariate one.

**Severity:** MAJOR
**Action:** The abstract and discussion must clearly state that OS prediction fails to reach significance after multivariate adjustment. "Independent of clinical covariates" should only be claimed for DFS.

---

## 2. Methodological Concerns

### 2.1 LODO-CV is not truly "external" validation [MAJOR]

**Concern:** The paper calls LODO-CV "the strictest form of external validation short of a prospective trial." This overstates the case. All 9 datasets share:
- Overlapping cancer type (7/9 ovarian, mostly HGSOC)
- Overlapping treatment (all platinum-based)
- Overlapping biology (all selected for having platinum response annotations)
- Overlapping era (mostly 2005-2020 samples)

True external validation would involve a completely independent cohort, ideally prospective, with blinded predictions. LODO-CV is leave-one-*study*-out, but the studies are not biologically independent. They all sample from the same underlying population of platinum-treated ovarian cancer patients. Dataset-level effects (institution, sample handling, RNA extraction protocol, array lot) are removed, but shared biological signals (e.g., "ovarian cancer immune microenvironment predicts platinum response") are not.

The strongest claim LODO-CV supports is: "the model generalizes across platforms and institutions within ovarian cancer." It does NOT support: "this is externally validated." The I-SPY2 breast cancer dataset (GSE173839) is the closest to a true external validation, and it performs well (AUC=0.818), but this is a single dataset.

**Severity:** MAJOR
**Action:** Soften "strictest form of external validation" language. Acknowledge that LODO-CV tests platform/institution generalization, not biological generalization to new cancer types or treatment contexts.

### 2.2 C=0.01 sensitivity: the paper claims stability but the evidence is ambiguous [MAJOR]

**Concern:** The paper states C=0.01 was "consistently optimal across all outer folds." Verification from `nested_cv_results.json` confirms this is TRUE for L2 LogReg --- C=0.01 wins in all 9 inner CV folds. However, examining the inner CV performance surface reveals:

| C value | Mean inner AUC (across folds) |
|---------|-------------------------------|
| 0.001   | ~0.72                         |
| **0.01**| **~0.76**                     |
| 0.1     | ~0.74                         |
| 1.0     | ~0.71                         |

The difference between C=0.001 and C=0.01 is ~4 AUC points in inner CV. The difference between C=0.01 and C=0.1 is ~2 AUC points. This is a relatively flat optimum. The question is: would C=0.005 or C=0.02 give materially different LODO results? The nested CV grid {0.001, 0.01, 0.1} has order-of-magnitude spacing, so we cannot assess sensitivity between grid points.

More importantly, the L2 LogReg with default C=1.0 achieves mean AUC = 0.670 vs tuned C=0.01 at 0.693. The tuning gains only 2.3 AUC points. This is reassuring for robustness but also means the headline AUC is at the optimized end of a narrow range.

**Severity:** MAJOR (for the gap in the grid; the results are likely robust)
**Action:** Run additional C values (0.005, 0.02, 0.05) to demonstrate the performance curve is smooth. Report the sensitivity of the final AUC to C choice. The ElasticNet results (0.694, 0.691) already suggest the results are robust, which should be highlighted.

### 2.3 The "clean 7" subset is post-hoc cherry-picking [CRITICAL]

**Concern:** This is perhaps the most problematic analytical choice in the paper. The "clean 7" subset (excluding GSE32062 and GSE18864) is defined AFTER seeing the results. The criteria sound reasonable:
- GSE32062: data bug (compressed predictions)
- GSE18864: too small (n=24)

But the selection is outcome-dependent. GSE63885 (AUC 0.605, CI includes 0.5) is NOT excluded from the "clean 7" despite being non-significant. Why? Because removing it would reduce the count of "clean" datasets to 6 and hurt the narrative.

Worse, the exclusion criteria shift across analyses:
- Main results: "clean 7" = all minus GSE32062 and GSE18864
- Fixed LODO: "clean" = all minus GSE28739 (from `lodo_summary_fixed.json`, `exclude_list: ["GSE28739"]`)
- Ablation: "excluding GSE32062" = 8 datasets

This inconsistency --- different "clean" definitions for different analyses --- is a red flag for selective reporting. The reader cannot tell which definition of "clean" is being used at any given point.

**Most damning:** After the GSE32062 fix, GSE30161 dropped from AUC 0.731 to 0.633 --- a loss of 0.098 AUC points. This dataset went from being one of the best performers to borderline, yet this regression receives minimal discussion. The fix improved the mean (0.693 -> 0.705) only because the GSE32062 improvement (+0.145) exceeds the GSE30161 degradation (-0.098), but the instability of individual dataset results to the gene set used is alarming.

**Severity:** CRITICAL
**Action:**
1. Report ALL results on ALL 9 datasets as the primary analysis. The "clean" subset should be a clearly-labeled sensitivity analysis, not a co-equal result.
2. Standardize the exclusion criteria and apply them consistently.
3. Prominently discuss the GSE30161 degradation after the GSE32062 fix.
4. Remove or strongly caveat the "clean 7" mean AUC from the abstract.

### 2.4 class_weight='balanced' with extreme imbalance [MINOR]

**Concern:** GSE32062 has 225 positive / 35 negative (87% positive rate). With `class_weight='balanced'`, the negative class gets weight ~6.4x. When this dataset is in the training set (8 of 9 LODO folds), these 35 heavily-weighted negative samples can dominate the loss function for that dataset's contribution. When it's the test set, the 87% positive rate means even a constant predictor gets AUC=0.5 trivially, and any small discriminative signal produces AUC > 0.5.

The paper acknowledges this but doesn't analyze how much the `balanced` weighting affects results for datasets with extreme imbalance.

**Severity:** MINOR
**Action:** Run a sensitivity analysis with `class_weight=None` and report the effect. If results are similar, this concern is mitigated.

### 2.5 Rank transformation discards potentially informative information [MINOR]

**Concern:** The rank transformation converts expression to within-sample percentile ranks, destroying all absolute level information. A gene with uniformly high expression across all ovarian cancer samples (but variable in other cancer types) would get near-uniform ranks, losing signal. For a single-cancer-type analysis (7/9 datasets are ovarian), absolute expression levels within a platform might carry more signal than ranks.

The paper's justification is sound (cross-platform compatibility), but the trade-off isn't evaluated. What if you used rank transformation only for cross-platform testing but used raw values (with standard normalization) when training and testing on the same platform?

**Severity:** MINOR
**Action:** Acknowledge as a design trade-off. Optionally run a within-platform analysis using raw expression to quantify the rank transformation cost.

---

## 3. Comparison Fairness

### 3.1 BRCA comparison has a 3x training data advantage [CRITICAL]

**Concern:** The code review (M4) correctly identifies this, but the paper does not adequately address it. The comparison:

| Model | Training samples | Training datasets | Label type |
|-------|-----------------|-------------------|------------|
| Response model | ~800-880 | 8 datasets | Drug response |
| BRCA model | ~235-310 | 2 datasets | BRCA mutation |

The response model has **3x more training data** AND trains directly on the evaluation metric (drug response). This isn't a fair comparison of "response labels vs genotype labels" --- it's a comparison of "lots of diverse response data vs limited genotype data."

The BRCA model's cross-dataset AUC for predicting BRCA mutations (its own training objective) is 0.495 --- essentially random. This proves the BRCA model failed to learn anything generalizable, but is this because BRCA labels are inherently less informative, or because 310 samples from 2 datasets is insufficient? We cannot distinguish these explanations.

A fair comparison would require either:
1. **Matched volume:** Train the response model on only TCGA-OV + GSE63885 (same datasets as BRCA model), using response labels
2. **Augmented BRCA data:** Use BRCA labels from additional public datasets (TCGA-BRCA has ~1000 BRCA annotations)
3. **Simulation:** Subsample the response model's training data to match the BRCA model's size

Without such controls, the claim "training on response outperforms training on genotype" is confounded with "more training data outperforms less training data."

**Severity:** CRITICAL
**Action:** Run the matched-volume comparison (option 1). If the response model still wins with equal data, the claim is much stronger. If it doesn't, the current framing is misleading.

### 3.2 BRCA labels use "any mutation" instead of biallelic inactivation [MAJOR]

**Concern:** The paper compares against "BRCA-label models" and implies this represents what Tempus HRD-RNA does. But the actual BRCA labels used are "any BRCA1/2 mutation" (`brca_vs_response_results.json`: "note: Using any BRCA1/2 mutation as proxy for biallelic (LOH data not readily available)"). Tempus HRD-RNA trains on biallelic inactivation (both alleles lost), which is a much cleaner signal.

Many monoallelic BRCA mutations are functionally wild-type. Using "any mutation" as the label introduces noise into the BRCA model's training targets, making it appear weaker than a properly-trained biallelic model would be. The comparison is stacked against the BRCA approach.

Additionally, GSE63885 only has BRCA1 status (not BRCA2), further degrading the BRCA model's training quality.

**Severity:** MAJOR
**Action:** Explicitly acknowledge that the BRCA model uses a weaker label than Tempus and that the comparison is therefore conservative from the BRCA model's perspective (i.e., a proper biallelic model might perform better). Remove or strongly qualify the claim of "outperforming Tempus HRD-RNA."

### 3.3 softHRD comparison is fundamentally unfair [CRITICAL]

**Concern:** The softHRD comparison uses only 90 of 109 genes (82.6%). The 19 missing genes include:

- **BRCA1** --- the single most important HRD gene
- **BRCA2** --- the second most important HRD gene
- **FANCI** --- Fanconi anemia pathway, core HRD
- **CDK1** --- master cell cycle regulator
- **CCNA2** --- cyclin A2, cell cycle
- **EXO1** --- DNA mismatch repair
- **AURKB** --- Aurora kinase B, mitosis
- **UBE2C** --- ubiquitin ligase, proliferation
- **H2AX** --- DNA damage response
- **KIF14, KIF20A, KIFC1** --- kinesin family, proliferation
- **CDC45** --- DNA replication

These are not random missing genes. They include the most biologically critical components of the HRD/proliferation signature. BRCA1 and BRCA2 alone are arguably the backbone of any HRD score. Testing softHRD without these genes is like testing a car engine with the pistons removed and concluding engines are slow.

Moreover, looking at the softHRD results more carefully: GSE32062 gives softHRD AUC = 0.500 with CI [0.5, 0.5] --- the predictions have zero variance, just like the main model before the fix. This suggests the same gene mapping bug affects the softHRD comparison on GSE32062. The softHRD's mean AUC of 0.552 is therefore partially an artifact of including a dataset where it literally cannot make predictions.

**Severity:** CRITICAL
**Action:**
1. Either obtain the full 109-gene softHRD signature or remove the softHRD comparison entirely.
2. If keeping the comparison, prominently state "softHRD was tested with 17% of its signature missing, including BRCA1/2, and should not be considered a fair evaluation."
3. Investigate whether the GSE32062 gene mapping bug also affects softHRD predictions.

### 3.4 Immune baselines are straw men [MAJOR]

**Concern:** The immune baseline comparison pits an 11,140-gene trained model against single-gene predictors (CXCL9, CD8A, IRF1) and 5-15 gene averages/trained models. This is not a serious comparison. Obviously a model with 1000x more features and trained coefficients will outperform a single gene.

A fairer comparison would be:
1. The Ayers et al. Tumor Inflammation Signature (TIS-18) --- the paper uses TIS-15 (15/18 genes available) but doesn't train the full signature properly
2. The Cristescu et al. (2018) GEP score used in KEYNOTE trials
3. An established pan-cancer immune signature like the Immunoscore
4. A published ovarian-cancer-specific immune signature

The "trained" TIS-15 and Immune-5 models perform WORSE than raw averages (AUC 0.499 and 0.491), suggesting overfitting with so few features. But this is expected with L2 regularization on 5-15 features across 9 heterogeneous datasets --- it doesn't prove the concept is wrong, just that the implementation is too constrained.

**Severity:** MAJOR
**Action:** Include at least one established multi-gene immune/inflammation signature as a serious comparison. The Ayers TIS-18 trained on the proper data would be the minimum.

---

## 4. Biological Claims

### 4.1 "The model learned immune contexture, not DNA repair" is inferred, not proven [MAJOR]

**Concern:** The paper presents this as a key finding, but the evidence is entirely correlative:
1. The top feature weights include some immune genes
2. The model score correlates with immune pathway scores
3. CIBERSORT fractions show M1 macrophage enrichment in high-scoring tumors

But the model uses 11,140 genes. The top 100 genes by weight may represent immune biology, but the remaining 11,040 genes contribute to the prediction too (the feature selection analysis shows performance improves monotonically up to all 11,140 genes). We have no evidence that the model isn't learning some other signal (e.g., tumor purity, proliferation rate, metabolic state) that happens to correlate with both immune markers and drug response.

The claim would be much stronger with:
- A mediation analysis (does the immune signal mediate the drug response prediction?)
- Training a model on ONLY immune genes and showing comparable performance (but the paper shows this fails --- TIS-15 trained: AUC 0.499)
- A negative control: training on 11,140 random genes and showing it doesn't learn immune biology

The fact that TIS-15 trained gives AUC 0.499 (random!) is actually evidence AGAINST the "immune contexture" interpretation. If the model's predictive power comes from immune biology, why can't 15 immune genes replicate it?

**Severity:** MAJOR
**Action:** Reframe the immune interpretation as a hypothesis, not a conclusion. Add language like "the model score is associated with immune microenvironment features, though we cannot exclude that other biological signals contribute to the prediction." Consider a formal mediation analysis.

### 4.2 CIBERSORT has known limitations in ovarian cancer [MAJOR]

**Concern:** The M1 macrophage finding (2x enrichment, p=1.39e-5) is the centerpiece of the biological interpretation. However:

1. **CIBERSORT LM22 uses blood-derived reference profiles.** Tumor-associated macrophages (TAMs) have distinct transcriptomic profiles from blood monocyte-derived macrophages. M1/M2 polarization in solid tumors is more of a continuum than the discrete categories LM22 assumes.

2. **Ovarian cancer has exceptionally high macrophage infiltration.** CIBERSORT fractions for macrophages M2 (mean ~29%) dominate the mixture. Small perturbations in the deconvolution can produce large changes in M1 vs M0 classification.

3. **The fractions are compositional.** CIBERSORT relative fractions sum to 1.0. If macrophage M0 fractions decrease (fold change 0.54), M1 mechanically increases in the relative space, even without absolute M1 changes. The paper reports fold changes in RELATIVE fractions, not absolute abundance estimates.

4. **The CD8+ T cell null result (rho=0.001) could be an artifact.** CIBERSORT's CD8 signature may be overwhelmed by the high macrophage content. TCGA-OV is known to have relatively low CD8+ infiltration compared to other solid tumors.

**Severity:** MAJOR
**Action:**
1. Use CIBERSORTx absolute mode (or EPIC/MCP-counter) as a second deconvolution method to verify the M1 finding.
2. Acknowledge the compositional nature of CIBERSORT fractions explicitly.
3. Validate the M1 finding with direct M1 marker gene expression (e.g., NOS2, TNF, IL1B) rather than relying solely on deconvolution.

### 4.3 Stroma negative correlation may be a composition artifact [MINOR]

**Concern:** The negative correlation with stroma (FAP, COL1A1, FN1) could simply reflect that tumors with higher immune content mechanically have lower stroma content (they're competing for the same tissue space in gene expression mixtures). The paper interprets this biologically ("fibrotic stroma physically excludes immune cells and reduces drug penetration") but the simpler explanation is a mathematical coupling.

**Severity:** MINOR
**Action:** Acknowledge the compositional interpretation as an alternative. Test whether the stroma correlation remains after regressing out immune content.

### 4.4 Age confound (rho = -0.333) is insufficiently addressed [MAJOR]

**Concern:** The model score is strongly negatively correlated with age (rho = -0.333, p < 0.001). This is a substantial confound because:
- Younger ovarian cancer patients tend to have more BRCA germline mutations
- Younger patients may receive more aggressive first-line therapy
- Tumor biology changes with age (immune infiltration, mutation burden)
- The multivariate Cox model "controls for age" but age is entered as a linear covariate, which may not capture the full confounding structure

The paper mentions this once and moves on. But rho = -0.333 means age explains ~11% of the variance in model scores. The multivariate model shows age is non-significant for DFS (HR=0.998, p=0.81), which helps, but the score-age correlation could still inflate the univariate estimates.

**Severity:** MAJOR
**Action:** Investigate whether the model is partly learning an "age signature." Run the LODO-CV with age as an additional covariate in the logistic regression to see if performance drops. Stratify results by age tertiles.

---

## 5. Clinical Claims

### 5.1 "Works in BRCA-WT patients" overstates modest performance [MINOR]

**Concern:** The BRCA-WT subgroup analysis shows AUC = 0.649 (from `brca_vs_response_results.json`: TCGA-OV_BRCA_wildtype AUC = 0.649). This is presented as evidence the model "works" in BRCA-WT patients. But AUC 0.649 in a single dataset's subgroup is barely above the TCGA-OV overall AUC of 0.645, suggesting BRCA status contributes essentially nothing to the model's performance in this cohort.

More importantly, AUC 0.649 is not clinically actionable. A test that correctly ranks patients only 65% of the time would not change clinical decision-making.

**Severity:** MINOR
**Action:** Present the BRCA-WT result honestly as "modest" and discuss what AUC threshold would be needed for clinical utility.

### 5.2 "Outperforms Tempus HRD-RNA" is unsupported [CRITICAL]

**Concern:** The paper never tested against Tempus HRD-RNA. Tempus's model is proprietary. The comparison is against a BRCA-label model trained with inferior data (any mutation vs biallelic, 310 samples vs Tempus's 100K+). The paper states "We cannot directly benchmark against Tempus" but then proceeds to imply superiority throughout the Discussion.

The BRCA-label model in this paper is not a proxy for Tempus. It's a weak straw man trained with limited data and noisy labels. A proper comparison would require either:
1. Obtaining Tempus predictions on these datasets
2. Replicating the Tempus approach (biallelic labels, deep learning) with adequate training data
3. Comparing against published Tempus validation results

**Severity:** CRITICAL (for the implied claim; the explicit comparison is acknowledged as limited)
**Action:** Remove all language implying the paper "outperforms" Tempus. The appropriate framing is: "Our approach suggests that response labels may outperform genotype labels, though direct comparison with commercial tests is needed."

### 5.3 "Independent of clinical covariates" applies only to DFS [MAJOR]

**Concern:** As noted in 1.4, the OS multivariate result is non-significant (p=0.167). The likelihood ratio test for OS explicitly says adding the score does NOT improve the model. Yet the paper's framing of "independent predictor" doesn't clearly restrict this to DFS.

From the data:
- DFS LR test: p = 0.00021 (score adds value)
- OS LR test: p = 0.16559 (score does NOT add value)

The DFS result is strong and appears genuine. The OS result is null. The paper should not blur this distinction.

**Severity:** MAJOR (repeated from 1.4 for emphasis)
**Action:** Section headers and abstract should specify "independent predictor of DFS" not "clinical outcomes" generically.

---

## 6. Cherry-Picking and Selective Reporting

### 6.1 Two parallel result sets create a "pick the better number" problem [CRITICAL]

**Concern:** The paper presents results from two gene sets:
1. **"Tuned"**: 11,140 genes, original gene intersection (pre-GSE32062 fix)
2. **"Fixed"**: 11,089 genes, corrected gene intersection (post-GSE32062 fix)

Different claims in the paper draw from different result sets:

| Claim | Source | Gene set |
|-------|--------|----------|
| Per-dataset AUCs, forest plot | bootstrap_cis_tuned.json | Tuned (11,140) |
| BRCA comparison | brca_vs_response_results.json | Tuned (11,140) |
| softHRD comparison | softhrd_comparison_results.json | Tuned (11,140) |
| Immune baselines | (tuned) | Tuned (11,140) |
| CIBERSORT deconvolution | cell_type_deconv_results.json | **Fixed (11,089)** |
| Multivariate survival | multivariate_survival.json | **Fixed (11,089)** |
| "Fixed" LODO AUCs | lodo_summary_fixed.json | Fixed (11,089) |

The primary results (per-dataset AUCs, comparisons) use the tuned model. The survival analysis uses the fixed model. The paper presents the "fixed" mean AUC (0.705) as an improvement over the "tuned" mean (0.693), but the survival analysis was run on the fixed model's predictions. We don't know if the tuned model's survival results would be identical.

This is not fraud, but it's sloppy. A reader cannot determine which exact model produced any given result without reading the code.

**Severity:** CRITICAL
**Action:** Declare a single canonical model and re-run ALL analyses on it. Every claim in the paper should come from the same model. Present the other as a sensitivity analysis.

### 6.2 GSE30161 degradation after GSE32062 fix is underreported [MAJOR]

**Concern:** From comparing `bootstrap_cis_tuned.json` (tuned) and `lodo_summary_fixed.json` (fixed):

| Dataset | Tuned AUC | Fixed AUC | Change |
|---------|-----------|-----------|--------|
| GSE32062 | 0.489 | 0.634 | **+0.145** |
| GSE30161 | 0.731 | 0.633 | **-0.098** |
| GSE28739 | 0.747 | 0.787 | +0.040 |
| GSE194040 | 0.892 | 0.876 | -0.016 |
| GSE173839 | 0.818 | 0.814 | -0.004 |

GSE30161 goes from one of the best performers (AUC 0.731, significant) to borderline (AUC 0.633, likely non-significant). This is a 10-point AUC drop from removing 51 genes. If individual dataset performance is this sensitive to the gene set, the model is fragile.

The paper mentions the GSE32062 improvement prominently (Section 3.6) but does not discuss the GSE30161 degradation at all.

**Severity:** MAJOR
**Action:** Report all per-dataset changes between tuned and fixed models. Discuss the instability and what it implies about model robustness.

### 6.3 The "clean" definition is inconsistent [MAJOR]

**Concern:** As noted in 2.3:
- Main results "clean 7": exclude GSE32062 + GSE18864
- Fixed results "clean": exclude GSE28739 (from `lodo_summary_fixed.json`)
- Ablation "excluding GSE32062": exclude only GSE32062

Three different "clean" definitions appear across the paper. The main results report mean AUC = 0.714 for "clean 7." The fixed results report mean AUC = 0.695 for their "clean." These numbers are not comparable but might be confused by a reader.

**Severity:** MAJOR
**Action:** Standardize terminology. If "clean" is used, define it once and apply consistently. Better yet, report all 9 datasets as the primary analysis and drop the "clean" framing.

---

## 7. Reproducibility Concerns

### 7.1 BRCA data in /tmp/ is non-reproducible [MAJOR]

**Concern:** The code review (M1) identifies that BRCA mutation data is loaded from `/tmp/ov_tcga_combined_brca.json`. This file is lost on reboot. The survival analysis also downloads clinical data from an AWS URL at runtime. Neither of these is acceptable for reproducibility.

**Severity:** MAJOR
**Action:** Commit all input data files (or scripts to generate them) to the repository. Remove all `/tmp/` paths and runtime downloads.

### 7.2 Inconsistent matrix usage across scripts [MAJOR]

**Concern:** From the code review (M3), some scripts use the original pooled matrix (11,140 genes) and others use the fixed version (11,089 genes). The CIBERSORTx analysis uses the fixed matrix while everything else uses the original. This means the immune interpretation is based on a different model than the primary AUC results.

**Severity:** MAJOR
**Action:** Standardize on a single matrix. Re-run all analyses.

### 7.3 run_exp8.py uses C=1.0, not C=0.01 [MAJOR]

**Concern:** The code review (M5) identifies that the main experiment script uses `C=1.0` while all downstream analyses use `C=0.01`. The `exp8_results.json` contains C=1.0 results. A reader running `run_exp8.py` would get different results than reported in the paper.

**Severity:** MAJOR
**Action:** Update `run_exp8.py` to use C=0.01 or clearly mark it as exploratory. Create a single "final pipeline" script that reproduces all paper results.

### 7.4 Data leakage in nested CV residualization [MINOR for main results]

**Concern:** The code review (C1) identifies that the residualization in `run_nested_cv_v2.py` uses test-set statistics. This only affects the nested CV comparison (Supplementary Table 1), not the primary results, since the main pipeline doesn't use residualization. But if the nested CV results are cited in the paper, they are tainted.

**Severity:** MINOR (for the paper; CRITICAL for the code)
**Action:** Fix the residualization leakage. Re-run nested CV and confirm results are unchanged.

---

## 8. Missing Controls and Analyses

### 8.1 No random gene set control [CRITICAL]

**Concern:** The model uses 11,140 genes. At this dimensionality, any large gene set will capture substantial transcriptomic variation, much of which correlates with immune infiltration, tumor purity, and other microenvironment features. The question is: **would 11,140 random genes also predict platinum response via immune infiltration?**

This is the most fundamental missing control. If a random gene set achieves similar performance, then the model isn't learning drug-response-specific biology --- it's learning that bulk gene expression captures immune infiltration, which correlates with response. That would still be a valid predictor, but the interpretation and claims about gene selection would change dramatically.

**Severity:** CRITICAL
**Action:** Train and evaluate models on 10 random draws of 11,140 genes from a larger gene universe. If they perform similarly, the specific gene set doesn't matter and the paper's framing should change. If they perform much worse, this is strong evidence for signal in the specific genes.

### 8.2 No cross-cancer-type analysis [MAJOR]

**Concern:** With 7 ovarian and 2 breast cancer datasets, an obvious analysis is: what happens when you train on ONLY ovarian and test on breast (or vice versa)? This would test whether the immune signal is cancer-type-specific or generalizable. The ablation section touches on this ("Ovarian only: 0.668") but doesn't isolate the cross-cancer transfer question.

**Severity:** MAJOR
**Action:** Report ovarian-only model tested on breast cancer datasets, and vice versa.

### 8.3 No interaction test for I-SPY2 (PARPi arm) [MAJOR]

**Concern:** GSE173839 (I-SPY2) is described as a breast cancer TNBC dataset treated with "platinum + PARPi." If there is a control arm (platinum without PARPi), this provides a unique opportunity to test whether the model predicts PARPi-specific benefit vs general platinum sensitivity. The paper mentions this dataset but never reports an interaction test.

The code review mentions "treatment-arm for interaction test" in the permutation test checks, suggesting this analysis was done but not reported.

**Severity:** MAJOR
**Action:** If a control arm exists in I-SPY2, report the interaction test. If the model predicts response equally in both arms, it's a general chemo-sensitivity predictor, not PARPi-specific. This is clinically important.

### 8.4 No comparison to published HRD classifiers beyond softHRD [MAJOR]

**Concern:** The field has many published HRD/platinum response signatures:
- Marquard et al. (2015) HRD gene expression signature
- Wang et al. (2022) PanHRD
- Peng et al. (2014) ProHRDness
- Konecny et al. (2014) ovarian cancer chemo-response signature
- The CIN70 proliferation signature

The paper only compares against softHRD (with 17% of genes missing) and a BRCA-label model (with 3x less training data). These are weak comparisons. Testing against established ovarian cancer response signatures would provide much more informative benchmarks.

**Severity:** MAJOR
**Action:** Include at least 2-3 published platinum response or HRD signatures as comparators.

### 8.5 No calibration analysis on fixed data [MINOR]

**Concern:** Calibration is reported only for the tuned model (ECE = 0.25). After the GSE32062 fix, calibration properties may change. The fixed model's calibration is not reported.

**Severity:** MINOR
**Action:** Report calibration for the fixed model.

---

## 9. Presentation and Framing Issues

### 9.1 The paper oversells the novelty [MINOR]

**Concern:** Training on clinical outcomes rather than genotype is not a new idea. Many ovarian cancer gene expression studies have trained directly on platinum response (e.g., Helleman et al. 2006, Dressman et al. 2007, Konstantinopoulos et al. 2010). The novelty here is the cross-dataset evaluation and the immune interpretation, not the concept of response-trained models.

**Severity:** MINOR
**Action:** Cite prior work on response-trained models and clarify that the novelty is in the systematic cross-dataset evaluation and biological interpretation.

### 9.2 The writing style is inappropriately casual for a scientific paper [MINOR]

**Concern:** Phrases like "What happens if you skip the genotype proxy," "This is not a subtle difference in emphasis," and "This is a useful lesson" are blog-post style, not journal style. The TL;DR section is unusual for a scientific manuscript.

**Severity:** MINOR
**Action:** Formal revision for journal style. Remove TL;DR or rename to "Graphical Abstract" or "Highlights."

### 9.3 Concordance indices are modest [MINOR]

**Concern:** The DFS concordance index is 0.597 (univariate) and 0.638 (multivariate). These are low for a clinical predictor. The paper highlights HRs and p-values but doesn't discuss the C-index, which is a more interpretable measure of discrimination. A C-index of 0.60 means the model correctly ranks 60% of patient pairs, which is marginal.

**Severity:** MINOR
**Action:** Report and discuss C-indices prominently. Contextualize against other ovarian cancer prognostic models.

---

## 10. Summary of Issues by Severity

### CRITICAL (5 issues --- must be addressed)
| # | Issue | Section |
|---|-------|---------|
| 1 | "Clean 7" subset is post-hoc cherry-picking with shifting definitions | 2.3 |
| 2 | BRCA comparison has 3x training data advantage | 3.1 |
| 3 | softHRD comparison missing 17% of signature including BRCA1/2 | 3.3 |
| 4 | "Outperforms Tempus" is unsupported | 5.2 |
| 5 | No random gene set control | 8.1 |
| 6 | Two parallel result sets with no canonical model | 6.1 |

### MAJOR (15 issues)
| # | Issue | Section |
|---|-------|---------|
| 1 | No multiple testing correction on secondary analyses | 1.1 |
| 2 | Selective p-value highlighting (OS non-significant buried) | 1.4 |
| 3 | LODO-CV overstated as "external validation" | 2.1 |
| 4 | C parameter sensitivity not evaluated between grid points | 2.2 |
| 5 | BRCA labels use "any mutation" vs biallelic | 3.2 |
| 6 | Immune baselines are straw men | 3.4 |
| 7 | "Immune contexture" is inferred, not proven | 4.1 |
| 8 | CIBERSORT limitations in ovarian cancer | 4.2 |
| 9 | Age confound (rho=-0.333) insufficiently addressed | 4.4 |
| 10 | "Independent of covariates" applies only to DFS | 5.3 |
| 11 | GSE30161 degradation after fix is underreported | 6.2 |
| 12 | Inconsistent "clean" definitions | 6.3 |
| 13 | BRCA data in /tmp/, runtime downloads | 7.1-7.3 |
| 14 | No cross-cancer-type analysis | 8.2 |
| 15 | No comparison to published HRD signatures beyond softHRD | 8.4 |

### MINOR (8 issues)
| # | Issue | Section |
|---|-------|---------|
| 1 | Permutation test scope clarification | 1.2 |
| 2 | Bootstrap method (percentile vs BCa) | 1.3 |
| 3 | class_weight sensitivity with extreme imbalance | 2.4 |
| 4 | Rank transformation trade-offs | 2.5 |
| 5 | Stroma composition artifact | 4.3 |
| 6 | BRCA-WT AUC is modest | 5.1 |
| 7 | Missing fixed-model calibration | 8.5 |
| 8 | Writing style and concordance reporting | 9.1-9.3 |

---

## 11. What's Good About This Paper

Despite the extensive criticism above, several aspects deserve recognition:

1. **The premise is genuinely valuable.** Training on clinical outcomes rather than genotype proxies is a sound idea that deserves rigorous investigation.
2. **LODO-CV is the right evaluation framework** for cross-dataset generalization, even if it's not true external validation.
3. **The paper is unusually transparent** about failures (GSE32062, GSE18864, OS non-significance, calibration issues). Most papers would quietly drop failing datasets.
4. **The nested CV model comparison** is properly implemented (inner/outer loops, no leakage in the main pipeline).
5. **The biological interpretation**, while over-claimed, is interesting and hypothesis-generating.
6. **The simplicity of the approach** (logistic regression, rank transformation) makes it reproducible and interpretable.
7. **The feature selection analysis** showing monotonic improvement up to all genes is informative and honest.

---

## 12. Recommended Revisions (Priority Order)

1. **[Required]** Establish a single canonical model (tuned or fixed) and re-run ALL analyses on it
2. **[Required]** Run matched-volume BRCA comparison (same datasets, same n)
3. **[Required]** Run random gene set control (10 random draws of 11K genes)
4. **[Required]** Fix softHRD comparison or remove it; acknowledge missing key genes
5. **[Required]** Remove "outperforms Tempus" language; reframe as hypothesis
6. **[Required]** Report all 9 datasets as primary; relegate "clean" to sensitivity analysis
7. **[Required]** Apply FDR correction to all secondary analyses
8. **[High priority]** Report OS non-significance prominently; restrict "independent predictor" to DFS
9. **[High priority]** Investigate age confound more thoroughly
10. **[High priority]** Add cross-cancer analysis (ovarian-trained, breast-tested)
11. **[High priority]** Compare against published platinum response signatures
12. **[High priority]** Fix reproducibility issues (commit data, remove /tmp/ paths)
13. **[Medium]** Run C parameter sensitivity between grid points
14. **[Medium]** Validate M1 macrophage finding with second deconvolution method
15. **[Medium]** Add I-SPY2 interaction test if control arm available

---

*Review generated 2026-02-24. This review is intentionally adversarial to identify weaknesses; the overall quality of the work is above average for the field, and most issues are addressable with additional analyses and revised framing.*
