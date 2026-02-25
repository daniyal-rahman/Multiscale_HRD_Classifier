# Experiment 8: Drug Response Prediction via Transcriptomic Classifier

**Training on Drug Response, Not Genotype: A Transcriptomic Classifier That Predicts Platinum Sensitivity Across 9 Independent Cohorts**

## Key Results

| Metric | Value |
|--------|-------|
| Mean AUC (all 9 datasets, LODO-CV) | **0.693** (permutation p < 0.001) |
| Mean AUC (post-GSE32062 fix) | **0.705** |
| vs BRCA-trained model | 0.693 vs 0.555 (response wins 8/9 datasets) |
| vs BRCA model (matched volume) | 0.686 vs 0.576 (response wins 5/7 datasets) |
| DFS multivariate Cox HR | 0.098 (p = 0.0002), independent of stage/grade/age/BRCA |
| OS multivariate Cox HR | 0.385 (p = 0.167, **not significant**) |
| Top biological association | M1 macrophages 2x enriched (p = 1.4e-5), CD8+ T cells null |

## Approach

- **904 patients** across **9 datasets**, **6 platforms** (Affymetrix, Agilent, RNA-seq)
- Within-sample **rank transformation** for cross-platform compatibility
- **L2 logistic regression** (C=0.01, class_weight='balanced')
- **Leave-one-dataset-out cross-validation** (LODO-CV): train on 8, test on 9th

## Full Paper

See **[PAPER_DRAFT.md](PAPER_DRAFT.md)** for the complete writeup with figures, methods, and limitations.

## Quality Reviews

- **[CODE_REVIEW.md](CODE_REVIEW.md)** — Technical code review (1 CRITICAL, 5 MAJOR, 8 MINOR issues identified)
- **[SKEPTICAL_REVIEW.md](SKEPTICAL_REVIEW.md)** — Adversarial scientific review (6 CRITICAL, 15 MAJOR issues identified and addressed in paper revision)

## Directory Structure

```
exp8_response_prediction/
├── PAPER_DRAFT.md              # Full paper draft
├── CODE_REVIEW.md              # Technical code review
├── SKEPTICAL_REVIEW.md         # Adversarial scientific review
├── README.md                   # This file
│
├── run_exp8.py                 # Main experiment (initial C=1.0 exploration)
├── rebuild_with_fixed_gse32062.py  # Rebuild with corrected GSE32062
│
├── figures/                    # All publication figures (PNG + PDF)
│   ├── fig1a-c                 # LODO-CV performance
│   ├── fig2a-b                 # Head-to-head comparisons
│   ├── fig3a-d                 # Survival analysis
│   ├── fig4a-d                 # Biological mechanism
│   ├── fig5a-c                 # Diagnostics (calibration, scores, features)
│   ├── fig6a-b                 # Study overview
│   └── fig_supp_*              # Supplementary figures
│
├── validation/                 # Bootstrap CIs, permutation tests, calibration (C=0.01)
├── validation_fixed/           # Same but with GSE32062 fix (11,089 genes)
│   └── run_multivariate_survival.py  # Cox PH survival analysis
├── brca_comparison/            # Response vs BRCA-label model comparison
│   └── matched_volume_results.json   # Matched-volume (310 vs 310) comparison
├── immune_baselines/           # Single-gene and multi-gene immune benchmarks
├── immune_deconv/              # CIBERSORTx + pathway correlation analysis
├── softhrd_comparison/         # softHRD head-to-head (90/109 genes)
├── random_gene_control/        # Random gene set control experiment
├── tuning/                     # Nested CV model selection
├── ablation/                   # Training composition experiments
├── qc/                         # Quality control checks
└── gse32062_investigation/     # GSE32062 platform bug investigation
```

## Data

All input datasets are publicly available:
- **GEO**: GSE32062, GSE156699, GSE63885, GSE194040, GSE173839, GSE30161, GSE28739, GSE18864
- **TCGA**: TCGA-OV via GDC/Xena
- **CIBERSORTx fractions**: Thorsson et al. 2018 (GDC Pan-Immune portal)

Large data files (parquet matrices) are excluded from the repository. Run `run_exp8.py` to regenerate from source data.

## Dependencies

```
Python 3.10+
scikit-learn
pandas
numpy
scipy
lifelines
matplotlib
seaborn
lightgbm (for nested CV comparison only)
```

## Reproducing

```bash
# 1. Generate pooled rank matrix and initial LODO-CV results
python run_exp8.py

# 2. Run tuned validation (C=0.01)
python validation/run_validation_tuned.py

# 3. Run all comparison analyses
python brca_comparison/run_brca_comparison.py
python immune_baselines/run_immune_baselines.py
python immune_deconv/run_immune_deconv.py
python softhrd_comparison/run_softhrd_comparison.py

# 4. Rebuild with GSE32062 fix
python rebuild_with_fixed_gse32062.py

# 5. Run survival analysis
python validation_fixed/run_multivariate_survival.py
```

All experiments run on a single CPU (i5-8600K). No GPU required. Total compute: ~2 hours.

---

*February 2026. This is a retrospective analysis. No prospective clinical validation has been performed.*
