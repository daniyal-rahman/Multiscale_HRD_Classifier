# Addendum: Peer Review Response — Additional Analyses

**Date:** 2026-02-24
**Context:** A peer reviewer raised three questions about our drug response classifier (Experiment 8). This addendum addresses each with new analysis.

---

## Table of Contents

1. [Is the Consensus-93 AUC=0.500 Real?](#1-consensus-93-auc0500-investigation)
2. [Do Our Response Model's Genes Overlap with Published Signatures?](#2-gene-overlap-analysis)
3. [Could WGS Features Do Better Than RNA-seq?](#3-wgs-dataset-survey-and-feasibility)

---

## 1. Consensus-93 AUC=0.500 Investigation

> **Peer question:** "There is no way this blue line is real" — the Consensus-93 ROC curve on I-SPY2 is a perfect diagonal (AUC=0.500), which looks fabricated.

### Verdict: Real, but it's prediction collapse, not chance performance

The AUC=0.500 is **mathematically exact** (0.5000 to 10 decimal places) because the Consensus-93 model predicts **exactly 1.0 for every single I-SPY2 sample** — all 100 samples, all 5 folds, identically. With zero variance in predictions, `roc_auc_score` returns 0.5 by convention and the ROC curve is a 2-point line from (0,0) to (1,1).

This is not a bug. It is a real failure mode caused by platform distribution shift.

### Root cause: Platform mismatch saturates the logistic regression

The TCGA training data uses RNA-seq FPKM values (range 0-458), while I-SPY2 uses Agilent microarray values (range 0-14, log2-scale). When the TCGA-trained StandardScaler is applied to I-SPY2, it produces extreme z-scores — I-SPY2 feature means land **+3.07 SD** from the TCGA center, with some values reaching **+220 SD**. The logistic regression saturates to predict_proba = 1.0 for all samples.

### Why Consensus-93 fails worse than larger models

| Model | Prediction std | Prediction range | AUC | N unique predictions |
|-------|---------------:|-----------------:|----:|---------------------:|
| **Consensus-93** | **0.000** | **0.000** | **0.500** | **1** |
| All-signature-410 | 0.002 | 0.012 | 0.751 | 100 |
| Data-driven-500 | 0.010 | 0.041 | 0.776 | 100 |

With only 93 genes, all biologically related (cell cycle, DNA repair), every feature shifts in the same direction due to the platform difference, causing uniform saturation. Larger gene sets include more diverse features, so at least some folds produce enough variance for partial ranking.

### Per-fold analysis (Consensus-93)

All 5 folds produce identical predictions of [1.0, 1.0] with AUC=0.500. The model itself is well-trained (56-63/93 non-zero coefficients, CV AUC=0.964 on TCGA), and 91/93 genes are found in I-SPY2. The failure is exclusively from cross-platform transfer.

### Implications

1. The AUC=0.500 is honest and should not be hidden — it accurately shows the model cannot discriminate on external data.
2. The mechanism should be disclosed: this is **prediction collapse** from platform mismatch, not "chance-level" performance where predictions vary but are uninformative. The biology may or may not be wrong — the methodology prevents us from testing it.
3. **This is precisely why our Exp8 response model uses within-sample rank transformation** instead of StandardScaler. Rank transformation is platform-agnostic, which is how our model achieves real cross-platform transfer across 6 different platforms.

> Full investigation: [consensus93_investigation.md](consensus93_investigation.md)

---

## 2. Gene Overlap Analysis

> **Peer question:** "You should check if the genes with the pCR models are similar" — do our response model's top features overlap with published HRD/pCR gene signatures?

### Summary

| Comparison | Top-100 overlap | Expected | Direction |
|-----------|----------------|----------|-----------|
| 12 HRD signatures (410 genes) | 2/100 | 3.7 | **Depleted** (0.54x) |
| TIS-18 immune signature (18 genes) | 2/100 | 0.16 | **Enriched** (12.4x, p=0.011) |
| Expanded Immune 28 genes | 2/100 | 0.25 | **Enriched** (8.0x, p=0.026) |
| Published platinum-response signatures | 0-1 | — | No enrichment |

### Key findings

**Our model is orthogonal to HRD signatures.** Across 12 published HRD gene lists (410 unique genes), overlap with our top-200 response genes is exactly at chance level (7 observed, 7.4 expected, hypergeometric p=0.61). The 7 overlapping genes are ECM genes (POSTN, LUM), HER2-proximal (GRB7), and broad-function genes (TP53) — none are core DNA repair.

**Our model is significantly enriched for immune genes.** The top-100 features overlap with the Tumor Inflammation Signature at 12.4-fold enrichment (p=0.011), driven by:
- **CXCL9** (rank 76, weight=+0.284) — T-cell chemoattractant, core IFN-gamma gene
- **HLA-DQA1** (rank 73, weight=+0.290) — MHC class II antigen presentation

### Two opposing immune programs in the model

**Positive weight (predict response):**
| Gene | Rank | Category |
|------|------|----------|
| HLA-DQA1 | 73 | Antigen presentation |
| CXCL9 | 76 | T-cell chemokine |
| GBP5 | 96 | IFN-gamma effector |
| HLA-DQB1 | 132 | Antigen presentation |
| CD52 | 194 | T-cell marker |

These define an active adaptive immune response: antigen presentation, T-cell recruitment, and IFN-gamma signaling.

**Negative weight (predict resistance):**
| Gene | Rank | Category |
|------|------|----------|
| MT1E | 11 | Cisplatin detoxification |
| GBP3 | 24 | Tumor-promoting inflammation |
| MT1F | 36 | Cisplatin detoxification |
| MT1G | 49 | Cisplatin detoxification |
| BCL2 | 119 | Anti-apoptotic |

Metallothioneins (MT1E/F/G) are well-established cisplatin-binding proteins that detoxify platinum compounds. BCL2 confers apoptosis resistance. These are biologically plausible resistance markers that are orthogonal to HRD.

### Bottom line

The response model does NOT recapitulate published HRD or pCR gene signatures. Instead, it identifies an immune/inflammatory axis of platinum sensitivity — consistent with our observation that it is orthogonal to BRCA status (rho=0.108) while independently predicting drug response (AUC 0.693 vs 0.555 for BRCA-trained model).

> Full analysis with per-signature tables: [gene_overlap_analysis.md](gene_overlap_analysis.md)
> Script: [gene_overlap_analysis.py](gene_overlap_analysis.py) | Data: [gene_overlap_results.json](gene_overlap_results.json)

---

## 3. WGS Dataset Survey and Feasibility

> **Peer question:** "This would be so much more interesting if you train on WGS instead of RNA-seq and using same labels — because then you can identify genomic features responsible for drug response instead of just genes."

### Available datasets

| Dataset | N (WGS/WES) | Sequencing | Response Labels | Access |
|---------|-------------|------------|-----------------|--------|
| **TCGA-OV** | ~316 | WES | Platinum sens/resist/refractory, PFI, OS | Controlled (dbGaP) |
| **BriTROC-1** | ~276 | sWGS + deep WGS | Plat-sensitive vs resistant at relapse | Controlled (EGA) |
| **DECIDER** | ~165-316 | WGS (multiregion) | Chemoresponse, platinum status, PFS, OS | Controlled (EGA) |
| **Patch/AOCS** | 92 | Deep WGS (60-100X) | Refractory/resistant/sensitive/acquired | Controlled (EGA) |
| **PCAWG-OV** | ~112 | Deep WGS | Limited clinical annotations | Controlled (ICGC) |
| **Genomics England** | ~454 | Clinical WGS | NHS treatment records | Very restricted |

**Total: ~900-1,100 unique patients with WGS/WES + platinum response data.**

TCGA-OV is particularly valuable because the same 316 patients have both WES and RNA-seq, enabling direct multi-modal comparison with our existing model.

### WGS features relevant to drug response

| Feature class | Examples | Key evidence |
|---------------|----------|-------------|
| **Mutational signatures** | SBS3 (HRD), microhomology deletions | HRDetect AUC=0.837 on ovarian WES (Sztupinszki 2021) |
| **Copy number signatures** | 7 CN signatures (Macintyre 2018) | Predict platinum-resistant relapse in BriTROC-1 |
| **Structural variants** | SV burden, tandem duplications, gene breakage | BRCA1-type vs BRCA2-type HRD patterns |
| **Focal events** | CCNE1 amplification, BRCA1/2 reversion | CCNE1 amp strongly predicts primary platinum resistance |
| **Composite classifiers** | HRDetect, CHORD, CN signatures | HRDetect-high: median OS 6.2 vs 4.1 years |

### Why RNA and WGS are complementary, not competitive

RNA-seq and WGS capture fundamentally different biology:

- **WGS** captures the static genomic blueprint: mutations, rearrangements, copy number changes that *caused* the tumor. But it cannot tell you whether a BRCA1 mutation has been functionally compensated, or whether the immune system has been recruited.
- **RNA-seq** captures the tumor microenvironment: immune infiltration, stromal interactions, signaling pathway activity, and the real-time transcriptional state. Our model works precisely because it captures downstream consequences of genomic instability, including immune activation.

A BRCA1-mutant tumor with high immune infiltration (visible on RNA-seq) behaves very differently from one with immune evasion, even though their genomes look identical.

### Recommended next steps

1. **Start with TCGA-OV WES** — same patients as our RNA data, enabling direct comparison
2. Extract HRDetect scores, HRD indices, gene mutations, CN profiles as features
3. Train a WGS-feature logistic regression with same LODO-CV framework
4. Compare, then integrate with RNA-based predictions for a multi-omic response model
5. Validate on Patch/AOCS (92 patients, gold-standard response labels) and BriTROC-1

### Challenges

- Most datasets require controlled access applications (weeks to months)
- WGS vs WES: full mutational signatures and SVs require WGS; WES is more limited
- Cross-cohort harmonization is non-trivial (different pipelines, reference genomes, depths)
- Platinum response definitions vary across studies (6-month cutoff vs continuous PFI)
- Sample sizes per dataset are smaller than RNA-seq cohorts

> Full survey with per-dataset details: [wgs_dataset_survey.md](wgs_dataset_survey.md)

---

## References

- Ayers et al. (2017) "IFN-gamma-related mRNA profile predicts clinical response to PD-1 blockade." *J Clin Invest* 127(8):2930-2940.
- Danaher et al. (2018) "Pan-cancer adaptive immune resistance as defined by the Tumor Inflammation Signature." *J Immunother Cancer* 6:63.
- Lin et al. (2025) "A Novel Platinum-Resistance-related Gene Signature in Ovarian Cancer." *PubMed* 39901543.
- Macintyre et al. (2018) "Copy number signatures and mutational processes in ovarian carcinoma." *Nat Genet* 50:1262-1270.
- Matondo et al. (2017) "The Prognostic 97 Chemoresponse Gene Signature in Ovarian Cancer." *Sci Rep* 7:9689.
- Patch et al. (2015) "Whole-genome characterization of chemoresistant ovarian cancer." *Nature* 521:489-494.
- Smith & Bradley et al. (2023) "The copy number and mutational landscape of recurrent ovarian high-grade serous carcinoma." *Nat Commun* 14:4387.
- Sztupinszki et al. (2021) "Migrating the SNP array-based homologous recombination deficiency measures to next generation sequencing data." *npj Breast Cancer* 7:78.
- TCGA (2011) "Integrated genomic analyses of ovarian carcinoma." *Nature* 474:609-615.
- Vázquez-García et al. (2023) "Ovarian cancer mutational processes drive site-specific immune evasion." *Cancer Discovery* 15(11):2262.

---

*Addendum generated 2026-02-24. Supporting scripts and data in this directory.*
