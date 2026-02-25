# Consensus-93 AUC=0.500 Investigation

## Peer Review Concern

> The Consensus-93 blue line (AUC=0.500) on the I-SPY2 ROC plot looks fake — it's too perfectly diagonal.

## Verdict: Not fake, but a complete prediction collapse

The AUC=0.500 is **mathematically exact** (0.5000000000 to 10 decimal places) because the Consensus-93 model predicts **exactly 1.0 for every single I-SPY2 sample**. All 100 samples, all 5 folds, identically. With zero variance in predictions, sklearn's `roc_auc_score` returns 0.5 by convention, and the ROC curve has only 2 points: (0,0) and (1,1) — a perfect diagonal.

This is not a bug or fabrication. It is a real failure mode caused by platform distribution shift.

## Root Cause: Platform mismatch saturates the logistic regression

The TCGA training data uses **RNA-seq FPKM** values (range 0–458), while I-SPY2 uses **Agilent microarray** values (range 0–14, likely log2-scale). When the TCGA-trained StandardScaler is applied to I-SPY2 expression, it produces extreme z-scores:

| Gene | TCGA mean (FPKM) | I-SPY2 mean (microarray) | Scaled I-SPY2 mean |
|------|------------------:|-------------------------:|-------------------:|
| PRC1 | 0.06 | 10.55 | ~175x shift |
| HMGA1 | 69.69 | 9.66 | extreme negative |
| ZWINT | 17.46 | 8.99 | mild negative |
| TTK | 3.90 | 7.28 | moderate positive |

After scaling: I-SPY2 feature means = **+3.07 SD** from TCGA center, with values reaching **+220 SD**. The logistic regression's linear combination produces extreme positive scores, pushing `predict_proba` to exactly 1.0 for all samples.

## Why Consensus-93 fails worse than other models

| Model | Prediction std | Prediction range | AUC | N unique predictions |
|-------|---------------:|-----------------:|----:|---------------------:|
| **Consensus-93** | **0.000000** | **0.000000** | **0.500** | **1** |
| All-signature-410 | 0.002464 | 0.011675 | 0.751 | 100 |
| Data-driven-500 | 0.010218 | 0.041067 | 0.776 | 100 |

With only 93 genes, all biologically related (cell cycle, DNA repair), the Consensus-93 model has no "escape route" — every feature shifts in the same direction due to the platform difference, and the model uniformly saturates.

Larger gene sets (410, 500 genes) include more diverse features spanning different expression ranges, so at least some fold-level models produce enough variation for partial ranking. Even so, All-signature-410 achieves its AUC=0.751 with vanishingly small prediction variance (std=0.002), and only 2 of 5 folds contribute any variance at all.

## Per-fold analysis

### Consensus-93 (all folds identical)
| Fold | Pred range | Pred std | AUC |
|------|-----------|----------|-----|
| 0 | [1.0, 1.0] | 0.000000 | 0.500 |
| 1 | [1.0, 1.0] | 0.000000 | 0.500 |
| 2 | [1.0, 1.0] | 0.000000 | 0.500 |
| 3 | [1.0, 1.0] | 0.000000 | 0.500 |
| 4 | [1.0, 1.0] | 0.000000 | 0.500 |

### All-signature-410 (sparse fold-level variation)
| Fold | Pred std | Pred range |
|------|----------|-----------|
| 0 | 0.000000 | 0.000000 |
| 1 | 0.010217 | 0.042279 |
| 2 | 0.002882 | 0.017239 |
| 3 | 0.000000 | 0.000000 |
| 4 | 0.000000 | 0.000000 |

This means the All-signature-410 AUC=0.751 rests almost entirely on fold 1's predictions. This is fragile.

## Model coefficients (Consensus-93)

The model itself is well-trained — 56–63 out of 93 features have non-zero coefficients across folds, with biologically sensible genes dominating:

| Gene | Mean coeff | Role |
|------|-----------|------|
| TTK | +1.00 | Mitotic checkpoint kinase |
| RFC4 | +0.98 | DNA replication |
| PRC1 | +0.83 | Cytokinesis |
| PGR | -0.48 | Progesterone receptor |
| RAD51 | -0.40 | HR repair |

The issue is not that the model is degenerate — it achieves CV AUC=0.964 on TCGA. The issue is exclusively the cross-platform transfer.

## Gene availability

91/93 consensus genes found in I-SPY2 (missing: H2AX, MRE11 — filled with 0). The 2 missing genes are not the cause; the platform-scale mismatch affects all 91 present genes.

## Implications for the manuscript

1. **The AUC=0.500 is honest** — it accurately reflects that the model cannot discriminate on this external dataset. It should not be hidden.

2. **The mechanism should be disclosed** — the diagonal ROC is not "chance-level performance" in the usual sense (where predictions vary but are uninformative). It is prediction collapse from platform mismatch. This distinction matters because:
   - "Chance-level" implies the biology is wrong
   - "Prediction collapse" implies the transfer methodology needs work

3. **All models suffer the same issue to varying degrees** — even All-signature-410 (AUC=0.751) achieves this with prediction std=0.002 across a [0.80, 0.81] range. The fact that it works at all is somewhat fortunate.

4. **Recommended fix**: Apply quantile normalization or ComBat batch correction between TCGA and I-SPY2 expression before prediction. Alternatively, train on log2(FPKM+1) and compare against microarray log2-intensity directly, or use rank-based normalization that is platform-agnostic.
