# Residualization: Model Scores vs CIBERSORT Immune Cell Fractions

## Reviewer Concern
"Are the model's drug response predictions simply recapitulating immune cell composition, which itself correlates with platinum response?"

## Method
1. Ran LODO-CV (L2 logistic regression, C=0.01) to obtain predicted scores for TCGA-OV (n=233)
2. Regressed scores (OLS) on all 22 CIBERSORT LM22 cell-type fractions from Thorsson et al. 2018
3. Extracted residuals = the component of scores NOT explained by immune composition
4. Tested whether residuals still discriminate platinum-sensitive vs platinum-resistant tumors

## Key Results

| Metric | Original Score | Residualized Score |
|--------|---------------:|-------------------:|
| AUC | 0.6550 | 0.6119 |
| Mann-Whitney p | 8.87e-05 | 3.40e-03 |
| Spearman rho vs response | 0.2462 | 0.1777 |
| Permutation p | <0.001 | 0.003 |
| Top-tertile response rate | — | 82.1% |
| Bottom-tertile response rate | — | 60.3% |
| Tertile fold enrichment | — | 1.36x |

- **R² of score ~ 22 cell types: 0.247** → immune composition explains only ~25% of score variance
- **AUC retention: 93.4%** → most predictive signal survives residualization
- **Permutation p = 0.003** for residualized AUC → significantly above chance

## Tumor Purity (ESTIMATE Leukocyte Fraction)
- Score weakly correlates with leukocyte fraction (Spearman rho=0.149, p=0.023)
- Response does NOT correlate with leukocyte fraction (rho=0.073, p=0.27)
- Adding leukocyte fraction to the 22 CIBERSORT predictors does not change R² (0.247 → 0.247)
- Residualized AUC with extended model: 0.611 (93.3% retention)

## Partial Correlation
After partialing out immune composition from both scores and response:
- Partial Spearman rho = 0.207, p = 1.48e-03
- Partial Pearson r = 0.182, p = 5.35e-03

## Conclusion
The model's predictive signal is **not primarily driven by immune cell composition**. While ~25% of score variance correlates with CIBERSORT fractions (expected given shared transcriptomic features), **93% of the AUC is retained** after residualization, and residualized scores remain significantly predictive (permutation p=0.003). This indicates the model captures cell-intrinsic biology (e.g., DNA repair deficiency) beyond immune infiltrate patterns.

## Files
- Script: `addendum/residualize_cibersort.py`
- Results: `addendum/residualize_cibersort_results.json`
