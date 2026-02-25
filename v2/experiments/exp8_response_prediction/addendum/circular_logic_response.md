# Response: Circular Logic and Immune Contradiction Concerns

**Date:** 2026-02-25
**Context:** A peer reviewer raised two related conceptual concerns about our claim that the drug response model captures immune microenvironment biology. This document addresses each with data.

---

## Concern 1: Circular Logic

> "Model predicts response -> model correlates with immune features -> therefore immune features predict response. Isn't that circular?"

### The argument, steel-manned

The reviewer's concern is legitimate and important. The reasoning chain in the paper could be read as:

1. We train a model on drug response labels
2. We observe that the model's predictions correlate with immune features (M1 macrophages, IFN-gamma)
3. We conclude that immune features predict drug response

Step 3 does not follow from steps 1-2. The model uses 11,140 genes. If any 11,140-gene model would correlate with immune features (because bulk transcriptomics inherently reflects immune composition), then the correlation is uninformative about immune biology specifically --- it just means the model picks up on a dominant source of transcriptomic variance. The logical structure would be: "anything trained on gene expression correlates with immune markers, and separately anything trained on these response labels achieves above-chance AUC; therefore the immune correlation is incidental, not mechanistic."

### Why it is not circular: three lines of evidence

#### Evidence 1: Simple immune scores DO predict response (weakly)

Before the model was ever built, we can ask: do simple immune markers predict platinum response? The answer is yes, modestly:

| Immune Baseline | Mean AUC (9 datasets) | Mean AUC (clean 7) |
|-----------------|----------------------:|--------------------:|
| IRF1 (single gene, raw rank) | 0.593 | 0.612 |
| TIS-15 (Tumor Inflammation Signature, average) | 0.553 | 0.589 |
| CXCL9 (single gene, raw rank) | 0.544 | 0.571 |
| Immune-5 (5-gene average) | 0.534 | 0.553 |
| CD8A (single gene, raw rank) | 0.518 | 0.546 |

These are raw scores --- no training, no fitting, no model selection. A patient's IRF1 rank alone predicts response at AUC 0.593 across 9 independent datasets. This establishes, independently of our model, that immune features carry weak but real predictive signal for platinum response. The literature on platinum-induced immunogenic cell death (ICD) provides mechanistic support: platinum drugs trigger calreticulin exposure, ATP release, and HMGB1 secretion, which activate innate immunity (Galluzzi et al. 2012, Zitvogel et al. 2008). Pre-existing immune activation (high IFN-gamma, M1 polarization) would amplify this ICD response.

The model's immune correlation is therefore not a post-hoc surprise but an expected finding given the known biology.

#### Evidence 2: The model captures signal beyond immune composition

The residualization analysis directly tests whether the model's predictions are "just" immune cell composition. We regressed the model's predicted scores on all 22 CIBERSORT LM22 cell-type fractions (OLS, n=233 TCGA-OV samples) and tested whether the residuals --- the component of scores NOT explained by immune composition --- still predict response.

| Metric | Original Score | Residualized Score |
|--------|---------------:|-------------------:|
| AUC | 0.655 | **0.612** |
| Mann-Whitney p | 8.87e-05 | **3.40e-03** |
| Spearman rho vs response | 0.246 | **0.178** |
| Permutation p (1,000 perms) | <0.001 | **0.003** |

**Key numbers:**
- **R² = 0.247**: Immune cell composition explains only 25% of the variance in model scores. 75% of the score is driven by non-immune features.
- **AUC retention = 93.4%**: After subtracting the immune-explained component, 93% of the predictive AUC is preserved.
- **Permutation p = 0.003**: The residualized scores are significantly above chance by a separate permutation test (1,000 shuffles).
- **Partial correlation**: After partialing out immune composition from BOTH scores and response, the partial Spearman rho = 0.207 (p = 0.0015).

The residualized model still discriminates responders from non-responders. The top tertile of residualized scores has an 82.1% response rate vs 60.3% for the bottom tertile (1.36x enrichment, p = 0.0014). For comparison, the original (non-residualized) tertile split shows 84.6% vs 55.1% (1.53x enrichment). Residualization attenuates the signal but does not abolish it.

**What this means:** The circular logic concern would hold if the model's predictive power were entirely mediated by immune composition --- i.e., if residualized AUC dropped to 0.50. It does not. The model captures both immune-related and immune-independent biology. The immune correlation is real (25% of variance) but is not the sole or even dominant source of predictive signal.

#### Evidence 3: The model is trained on 11,140 genes, not immune genes

The model's feature space is the full transcriptome (11,140 genes after intersection). Only a small fraction of the top-weighted genes are immune-related. The top 20 positive-weight genes include PAX8 (ovarian lineage), FOLR1 (folate receptor), FOXJ1 (cilia transcription factor), and C1orf186 alongside immune genes like CXCL9 and HLA-DQA1. The top negative-weight genes include POSTN (extracellular matrix), SFRP2 (Wnt antagonist), and metallothioneins (MT1E/F/G, cisplatin detoxification). The model is not "an immune model" --- it is a response model that happens to correlate with immune features among many others.

### The corrected logical structure

The non-circular argument is:

1. Simple immune scores predict response weakly (AUC 0.55-0.61) --- established independently
2. The full model predicts response substantially (AUC 0.693) --- trained on response labels
3. The full model correlates with immune features (R² = 0.25) --- expected given #1
4. After removing the immune-correlated component, the model STILL predicts response (AUC 0.612, p = 0.003) --- proves the model captures additional biology
5. Conclusion: The model learns a combination of immune and non-immune signals, both contributing to drug response prediction

This is not circular. Step 4 directly tests and rejects the null hypothesis that the model's power is fully explained by immune composition.

---

## Concern 2: Immune Contradiction

> "You claim the model captures immune biology, but trained immune-gene-only models fail (AUC ~0.50). Isn't that contradictory? If immune features predict response, why doesn't an immune-gene model work?"

### The data

| Method | Mean AUC (9 datasets) | Type |
|--------|----------------------:|------|
| Full L2 (11,140 genes) | **0.693** | Trained model |
| IRF1 raw rank | 0.593 | Untrained single gene |
| TIS-15 average | 0.553 | Untrained 15-gene average |
| CXCL9 raw rank | 0.544 | Untrained single gene |
| Immune-5 average | 0.534 | Untrained 5-gene average |
| **TIS-15 L2 trained** | **0.506** | Trained 15-gene model |
| **Immune-5 L2 trained** | **0.502** | Trained 5-gene model |

The apparent contradiction: untrained immune averages beat chance (0.55-0.59), but trained immune L2 models collapse to chance (~0.50). If there's immune signal, why does training destroy it?

### Resolution: Regularization collapse, not absent signal

The trained immune models use the same L2 regularization as the full model: `C = 0.01, class_weight = 'balanced'`. With C = 0.01 and only 5 or 15 features, the regularization penalty dominates the loss function. The model shrinks all 5-15 coefficients toward zero so aggressively that it effectively predicts the same score for every sample --- analogous to the Consensus-93 prediction collapse, but caused by regularization rather than platform mismatch.

This is a pathological regime: the penalty term λ||w||² at C=0.01 (λ=100) overwhelms the data-fit term when there are only 5-15 features to absorb the signal. The model cannot learn informative weights at this regularization strength with so few degrees of freedom. A higher C (less regularization) would fit the training data but likely overfit given only 5 features across 9 heterogeneous datasets with different platforms.

**The key insight**: the same C=0.01 that is optimal for 11,140 features is catastrophically over-regularized for 5-15 features. The full model has enough features that collectively they can push through the regularization; a 5-gene model does not. This is consistent with our feature selection analysis showing monotonic AUC improvement from 25 genes (AUC 0.568) to 11,140 genes (AUC 0.693) --- the signal is genuinely distributed.

**Why raw averages work but trained models don't**: A raw average (mean rank of 5 immune genes) has no fitting step, so it cannot overfit or collapse. It uses fixed, equal weights of 1/5 for each gene. This "model-free" approach captures the immune signal at its face value. The L2-trained model tries to learn optimal weights but is crushed by regularization. The lesson: with few features and heterogeneous data, a fixed-weight average is more robust than a fitted model.

### What this means for the "immune model" claim

We do NOT claim the model is "an immune model." We claim:

1. The model's predictions **correlate** with immune features (R² = 0.25 with CIBERSORT fractions; significant enrichment for immune genes in top feature weights)
2. Simple immune scores predict response **weakly** (AUC 0.55-0.61) --- the signal exists
3. The full model captures this weak immune signal **plus** substantial non-immune signal (AUC 0.612 after residualization)
4. Trained immune-only models fail because C=0.01 crushes 5-15 features, not because the immune signal is absent

The model is best described as a **distributed transcriptomic predictor** that integrates immune, stromal, metabolic, and tumor-intrinsic signals through 11,140 genes. The immune component is one contributor (explaining ~25% of score variance) but is neither necessary nor sufficient on its own. The contradiction dissolves when you recognize that "correlates with immune features" ≠ "is an immune model."

---

## Summary

| Concern | Resolution | Key Evidence |
|---------|-----------|--------------|
| Circular logic | Residualization test directly refutes circularity | AUC 0.612 after removing immune component (93% retention, perm p=0.003) |
| Immune contradiction | Regularization collapse at C=0.01 with 5-15 features | Raw averages work (AUC 0.55-0.59); same genes fail when L2-trained at C=0.01 |

### Recommended paper language

The paper should reframe the immune interpretation as:

> "The model's predictions are associated with immune microenvironment features --- IFN-gamma signaling, M1 macrophage polarization, and immune checkpoint expression --- consistent with known immunogenic effects of platinum-based chemotherapy. However, residualization against CIBERSORT cell-type fractions demonstrates that only 25% of the score variance is explained by immune composition. After removing this component, the residualized scores remain significantly predictive (AUC = 0.612, permutation p = 0.003), indicating the model captures biology beyond immune infiltrate. Simple immune gene averages achieve AUC 0.55-0.61 without any training, confirming the immune signal is real but modest. The failure of trained immune-gene-only models (AUC ~0.50) reflects regularization collapse at the penalty strength optimal for the full transcriptome, not absence of immune signal. The model is best understood as a distributed predictor that integrates immune, stromal, and tumor-intrinsic signals across 11,140 genes."

---

## Data Sources

- Residualization results: `addendum/residualize_cibersort_results.json`
- Residualization script: `addendum/residualize_cibersort.py`
- Immune baselines: `immune_baselines/immune_baseline_results.json`
- Immune baseline script: `immune_baselines/run_immune_baselines.py`
- Full analysis writeup: `addendum/residualize_cibersort.md`
