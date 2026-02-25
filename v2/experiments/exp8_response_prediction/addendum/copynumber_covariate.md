# Copy Number / HRD Score Covariate Analysis

## Reviewer Request
"Look at copynumber and try adding it as covariate in survival model and immune corr analysis."

## Data Source
HRD scores (LOH, TAI, LST) and copy number features from **Knijnenburg et al. 2018** (PanCan DDR paper), accessed via GerkeLab/TCGAhrd. Available for 82 of 235 TCGA-OV patients (34.9%) in our model.

---

## Key Results

### 1. RNA Score Correlates with HRD (but not generic CNA burden)

| Copy Number Feature | Spearman rho | p-value | n |
|---|---:|---:|---:|
| HRD_Score (LOH+TAI+LST) | +0.291 | 0.008 | 82 |
| HRD_TAI | +0.333 | 0.002 | 82 |
| HRD_LST | +0.263 | 0.017 | 82 |
| HRD_LOH | +0.247 | 0.025 | 82 |
| Aneuploidy score | -0.267 | 0.015 | 82 |
| CNA fraction altered | +0.042 | 0.699 | 86 |
| Mutation load | +0.006 | 0.954 | 86 |
| Mut Signature 3 (BRCA-like) | +0.036 | 0.761 | 75 |

The RNA model score positively correlates with HRD component scores (especially TAI), consistent with HRD tumors being more platinum-sensitive. It does NOT correlate with generic CNA burden (FGA) or mutation load.

### 2. Both RNA and HRD Predict Response, but RNA is Stronger

On the n=82 subset with both measures:

| Predictor | AUC | MW p-value |
|---|---:|---:|
| RNA model score | 0.745 | — |
| HRD_Score | 0.652 | 0.034 |
| HRD_LST | 0.647 | 0.040 |
| HRD_TAI | 0.642 | 0.047 |
| HRD_LOH | 0.629 | 0.072 |
| CNA fraction | 0.587 | 0.213 |
| Aneuploidy | 0.520 | 0.788 |

### 3. RNA Score is Independent of HRD; HRD Does Not Add to RNA

**Logistic regression independence tests** (n=82):

| Model | AUC | AIC |
|---|---:|---:|
| RNA score only | 0.745 | 87.1 |
| HRD score only | 0.652 | 95.7 |
| RNA + HRD | 0.772 | 87.4 |

| Likelihood Ratio Test | LR stat | p-value |
|---|---:|---:|
| HRD adds to RNA | 1.677 | 0.195 |
| **RNA adds to HRD** | **10.299** | **0.001** |

RNA score significantly adds predictive value beyond HRD (p=0.001), but HRD does NOT significantly add beyond RNA (p=0.195).

### 4. Survival Analysis (Cox PH)

**DFS (Disease-Free Survival)** — n=82 patients with HRD data:

| Model | C-index | Score p | HRD p |
|---|---:|---:|---:|
| Score only | 0.651 | 0.0009 | — |
| HRD only | 0.599 | — | 0.107 |
| Score + HRD | 0.664 | 0.001 | 0.132 |
| Full (score + clinical + HRD) | 0.682 | 0.004 | 0.411 |
| Clinical + HRD (no RNA) | 0.618 | — | — |

LR tests (all on same n=82 subset):
- HRD adds to RNA: p=0.136 (not significant)
- **RNA adds to HRD: p=0.0008**
- **RNA adds to clinical+HRD: p=0.003**

**OS (Overall Survival)** — n=82 patients:
- Score alone: HR=0.41, p=0.38 (underpowered with n=82, 49 events)
- Neither score nor HRD significant in full model (small sample)

### 5. BRCA-Stratified Analysis

| Stratum | n | RNA AUC | HRD AUC | HRD MW p |
|---|---:|---:|---:|---:|
| BRCA-wildtype | 63 | 0.785 | 0.630 | 0.100 |
| BRCA-mutated | 19 | 0.562 | — | — |

In BRCA-wildtype tumors, the RNA score strongly predicts response (AUC=0.785) while HRD_Score is borderline (AUC=0.630, p=0.10).

---

## Interpretation

1. **RNA and HRD capture overlapping biology**: The moderate correlation (rho=0.29) between RNA score and HRD_Score confirms both reflect platinum-sensitivity pathways, but they are far from redundant.

2. **RNA subsumes HRD**: The RNA model's predictive signal encompasses what HRD captures (adding HRD does not improve RNA, p=0.20), while RNA adds substantial independent information beyond HRD (p=0.001).

3. **HRD scores alone are modestly predictive**: HRD_Score achieves AUC=0.652 for response prediction (consistent with published literature on HRD as a platinum biomarker in HGSOC).

4. **RNA adds to clinical+HRD in survival**: Even after adjusting for stage, grade, residual disease, age, AND HRD score, the RNA score remains independently prognostic for DFS (p=0.003).

5. **Limitation**: Only 82/235 (35%) TCGA-OV patients have HRD scores from Knijnenburg et al. The subset may not be representative. OS analyses are underpowered.

## Files
- Script: `addendum/copynumber_covariate.py`
- Results: `addendum/copynumber_covariate_results.json`
- Data: Knijnenburg et al. 2018, accessed via https://github.com/GerkeLab/TCGAhrd
