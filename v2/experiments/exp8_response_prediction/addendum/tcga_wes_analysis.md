# TCGA-OV WES Analysis: Can Genomic Features Predict Platinum Response?

**Date:** 2026-02-25
**Purpose:** Download and analyze TCGA-OV whole-exome sequencing features to test whether WES-derived genomic features predict platinum response, and whether they add complementary signal to our RNA-based model.

---

## Data Sources

- **Mutations**: cBioPortal API, PanCancer Atlas study (`ov_tcga_pan_can_atlas_2018`)
- **Copy number**: GISTIC 2.0 discrete values from same study
- **TMB**: Clinical attributes (mutation count) from cBioPortal
- **RNA model scores**: LODO-CV predictions from our L2 LogReg (C=0.01, 11,089-gene fixed matrix)
- **Response labels**: Platinum sensitivity binary (from our pooled response data)

**Cohort**: 235 TCGA-OV patients with both WES and RNA-seq data, 165 sensitive / 70 resistant.

---

## WES Feature Summary

### Mutation Features

| Feature | N Positive | % of Cohort | Notes |
|---------|-----------|-------------|-------|
| TP53 mutated | 145 | 61.7% | Expected ~96% in HGSOC; lower here reflects PanCancer Atlas calling |
| BRCA any (existing data) | 44 | 18.7% | From our curated cBioPortal merge (3 studies) |
| BRCA any (PanCancer Atlas) | 14 | 6.0% | PanCancer Atlas alone captures fewer |
| HRR pathway mutated | 37 | 15.7% | Any of 19 HRR genes (BRCA1/2, ATM, PALB2, etc.) |
| BRCA damaging (truncating) | 10 | 4.3% | Nonsense, frameshift, splice site |
| NF1 mutated | 13 | 5.5% | Potential resistance gene |
| CDK12 mutated | 9 | 3.8% | Tandem duplicator phenotype |
| RB1 mutated | 7 | 3.0% | Potential resistance gene |

**Note on BRCA ascertainment**: The PanCancer Atlas study found only 14 BRCA-mutated patients in our 235-patient subset, while our curated merge of three cBioPortal studies (Nature 2011 + PanCancer + Firehose Legacy) found 44. The difference reflects germline variants included in the earlier studies but not in the PanCancer Atlas somatic-only calls. We use the existing 44-patient label (`brca_existing`) for more accurate analyses.

### Copy Number Features (GISTIC)

| Feature | N Positive | % of Cohort | GISTIC Threshold |
|---------|-----------|-------------|------------------|
| CCNE1 amplified | 45 | 19.1% | GISTIC = 2 (high-level amp) |
| MYC amplified | 80 | 34.0% | GISTIC = 2 |
| BRCA1 deleted | 180 | 76.6% | GISTIC ≤ -1 (shallow or deep del) |
| PTEN deleted | 103 | 43.8% | GISTIC ≤ -1 |
| RB1 deleted | 158 | 67.2% | GISTIC ≤ -1 |

### TMB

- Available for 220/235 patients
- Median TMB: 1.8 mutations/Mb (low, consistent with HGSOC)

---

## Univariate Associations with Platinum Response

| Feature | Response Rate (mutated) | Response Rate (wildtype) | Odds Ratio | Fisher p | AUC |
|---------|----------------------:|------------------------:|----------:|--------:|----:|
| **BRCA any (existing)** | **84.1%** | **67.0%** | **2.60** | **0.028** | **0.562** |
| **BRCA any (PCA)** | **100%** | **68.3%** | **inf** | **0.012** | **0.542** |
| BRCA damaging | 100% | 68.9% | inf | 0.036 | 0.530 |
| PTEN deleted | 76.7% | 65.2% | 1.76 | 0.062 | 0.568 |
| CCNE1 amplified | 60.0% | 72.6% | 0.57 | 0.105 | 0.453 |
| HRR pathway | 81.1% | 68.2% | 2.00 | 0.169 | 0.541 |
| TP53 mutated | 73.1% | 65.6% | 1.43 | 0.242 | 0.543 |
| TMB (continuous) | — | — | — | 0.041 | 0.587 |
| NF1 mutated | 69.2% | 70.3% | 0.95 | 1.000 | 0.499 |

**Key findings:**

1. **BRCA mutations predict response** (84% vs 67% response rate, OR=2.60, p=0.028), but the effect is modest (AUC 0.562). All 14 PanCancer Atlas BRCA patients responded (100%), but this extreme rate likely reflects ascertainment bias.

2. **CCNE1 amplification trends toward resistance** (60% vs 73% response rate, OR=0.57, p=0.105) --- consistent with Patch et al. 2015, though not significant at n=235.

3. **TMB weakly predicts response** (Spearman rho=0.138, AUC=0.587) --- higher mutation burden correlates with sensitivity, possibly via neoantigen load.

4. **No individual WES feature achieves AUC > 0.59.** All are near-chance predictors individually.

---

## Multivariate Model Comparison

| Model | AUC | 95% CI | Method |
|-------|----:|--------|--------|
| **RNA model (LODO-CV)** | **0.652** | [0.575, 0.728] | Trained on 8 other datasets, tested on TCGA-OV |
| RNA model (5-fold CV) | 0.647 | [0.564, 0.722] | 5-fold CV within TCGA-OV |
| WES features (5-fold CV) | 0.548 | [0.465, 0.633] | 17 WES features, L2 LogReg C=0.1 |
| WES + RNA combined | 0.557 | [0.474, 0.640] | 17 WES features + RNA score |
| BRCA-any (raw) | 0.545 | [0.524, 0.568] | Single binary feature |

**The RNA model substantially outperforms all WES-based approaches.** The 17-feature WES model (AUC 0.548) barely exceeds the single BRCA-any feature (AUC 0.545), suggesting that the additional WES features (TMB, copy number, HRR pathway) add minimal predictive value beyond BRCA status in this cohort.

**The combined WES+RNA model does NOT outperform RNA alone** (0.557 vs 0.652). Adding noisy WES features to the RNA score actually degrades performance. This is likely because:
1. The WES features have very low individual AUCs (all < 0.59)
2. With only 220 samples and 18 features (17 WES + RNA), the model overfits in 5-fold CV
3. The WES features may need more patients to learn stable associations

---

## RNA-WES Complementarity

| Comparison | Spearman rho | p-value |
|-----------|-------------|---------|
| RNA score vs WES CV score | 0.152 | 0.025 |
| RNA score vs BRCA status | 0.105 | 0.120 |

The RNA and WES scores are weakly correlated (rho=0.15), confirming they capture partially different biology. However, the WES signal is too weak to contribute meaningfully when combined.

RNA scores do not significantly differ between BRCA-mutant and BRCA-wildtype patients (mean 0.565 vs 0.508, MW p=0.12), consistent with the paper's finding that the RNA model is orthogonal to BRCA status (main analysis rho=0.108).

---

## Interpretation

### Why WES features underperform RNA for drug response prediction

1. **HGSOC has limited mutational diversity.** TP53 is mutated in virtually all patients (near-universal, therefore non-discriminating). Beyond BRCA1/2, no single gene mutation occurs in >6% of patients. Most WES features have very low prevalence, producing unstable associations.

2. **WES captures the blueprint, not the state.** A BRCA1 mutation tells you the HR pathway is disrupted, but not whether the tumor has compensated (e.g., via a reversion mutation, alternative pathway activation, or epigenetic changes). RNA-seq captures the real-time functional state, including immune microenvironment, which WES cannot see.

3. **These are hand-engineered features, not foundation model embeddings.** We tested binary mutation calls and GISTIC calls — the simplest possible WES features. A foundation model approach (DNABERT-2, Nucleotide Transformer) that encodes sequence context around variants might extract richer representations. The comparison here is WES-features-as-usually-used, not the frontier of what WGS/WES data could offer.

4. **Sample size limits multivariate learning.** With n=235 and 17 features, 5-fold CV produces training sets of ~188 samples — marginal for learning stable associations among sparse binary features where most have <20 positive cases.

### Implications for the foundation model proposal

These results set a **lower bound** for what WES data can achieve with standard feature engineering. The foundation model approach could exceed this by:
- Encoding sequence context (distinguishing truncating vs missense variants)
- Learning variant embeddings from pre-trained DNA language models
- Capturing mutational signature information implicitly from variant patterns

However, the fundamental challenge remains: with ~100-140 somatic variants per HGSOC patient from WES (low mutation burden), even a foundation model may struggle to aggregate enough signal for patient-level prediction.

### Implications for multi-modal integration

The near-zero complementarity between WES features and RNA score suggests that simple feature concatenation is not the right integration strategy. More sophisticated approaches (e.g., late fusion, stacked generalization with external validation) or richer WES representations (foundation model embeddings, mutational signatures from full WGS) may be needed.

---

## Files

- Script: `addendum/tcga_wes_analysis.py`
- Results: `addendum/tcga_wes_results.json`
- Features: `addendum/tcga_wes_features.csv`
