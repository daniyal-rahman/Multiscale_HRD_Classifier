# Tumor Purity Confound Analysis

## Reviewer Concern
"Is the model score confounded by tumor purity? Higher/lower purity tumors may have different gene expression patterns regardless of drug response."

## Method
- Purity proxy: ESTIMATE leukocyte fraction from Thorsson et al. 2018 (tumor purity = 1 - leukocyte fraction)
- TCGA-OV: n=235 matched samples
- Tests: correlation analysis, logistic regression with purity covariate, stratified AUC, gene-level analysis

## Key Results

### Purity Does NOT Correlate with Response
| Comparison | Spearman rho | p-value |
|---|---:|---:|
| Score vs leukocyte fraction | 0.148 | 0.024 |
| Score vs tumor purity | -0.148 | 0.024 |
| **Response vs leukocyte fraction** | **0.078** | **0.232** |
| **Response vs tumor purity** | **-0.078** | **0.232** |

Leukocyte fraction does NOT differ between sensitive and resistant tumors (MW p=0.23).

### Score Remains Significant After Adjusting for Purity

| Model | AIC | Score p | Purity p |
|---|---:|---:|---:|
| Response ~ score | 275.9 | 0.0003 | — |
| Response ~ score + purity | 277.9 | 0.0003 | **0.864** |
| Response ~ purity | 289.9 | — | 0.532 |

- Likelihood ratio test: score adds to purity model (p=0.0002)
- Likelihood ratio test: purity does NOT add to score model (p=0.864)
- Adding purity as a covariate does not change the score coefficient (4.07 → 4.05)

### Stratified Analysis
Score predicts response in BOTH high and low purity subgroups:
| Stratum | n | AUC | MW p |
|---|---:|---:|---:|
| High purity (≥ median) | 118 | 0.618 | 0.019 |
| Low purity (< median) | 117 | 0.686 | 0.001 |

### Gene-Level Analysis
- Correlation between model weights and purity correlations across all 11,089 genes: rho = -0.035
- Top model genes (e.g., CP, WNT10A, CLDN6, FOLR1) are not strongly purity-dependent
- Exception: CXCL9 (immune chemokine, rho=-0.61 with purity) — but immune genes are expected and the residualization analysis (Task #1) shows the signal survives after removing immune composition

## Conclusion
Tumor purity is **NOT a meaningful confound** for the drug response model:
1. Purity does not correlate with response (p=0.23)
2. Score remains equally significant after purity adjustment (p=0.0003 unchanged)
3. Purity has zero additional predictive value beyond the score (LR test p=0.86)
4. The model works in both high and low purity subgroups

## Files
- Script: `addendum/tumor_purity_confound.py`
- Results: `addendum/tumor_purity_confound_results.json`
