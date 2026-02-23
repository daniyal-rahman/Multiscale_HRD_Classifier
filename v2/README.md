# softHRD v2: Next-Generation Transcriptomic HRD Signature

## Research Goals

The existing softHRD v1 (and other transcriptomic HRD signatures) suffer from low
inter-signature agreement because each was trained against a different proxy label for
HRD. v2 aims to build a signature that captures **functional/dynamic HRD state** rather
than historical genomic scars, with the following key innovations:

1. **Tiered gold-standard labeling** — Train only on samples where multiple orthogonal
   measures agree (genotypic + scar + mutational signatures), exclude the ambiguous middle
2. **Platform-agnostic features** — Rank-based gene pair comparisons and pathway-level
   scores that survive batch effects across datasets and platforms
3. **Multi-task learning** — Jointly predict HRD status, BRCA1 vs BRCA2-type, and drug
   response for more robust shared representations
4. **Pan-cancer generalization** — Not just breast/ovarian, but prostate, pancreatic, GI
5. **Dynamic state capture** — Expression captures what the tumor *is*, not what it *was*

## Directory Structure

```
v2/
├── data_acquisition/    # Scripts to download/process public datasets
├── signature_analysis/  # Literature review, gene overlap, signature comparison
├── label_engineering/   # Tiered labeling system implementation
├── feature_engineering/ # Rank-based, pathway-score, and other features
├── models/              # Model training (XGBoost multi-task, ElasticNet comparison)
├── validation/          # External validation across cohorts
└── utils/               # Shared utilities (normalization, plotting, etc.)
```

## Key Datasets

| Dataset | N | Data Types | Key Endpoint |
|---------|---|------------|-------------|
| TCGA-BRCA | ~1100 | RNA-seq, WES, methylation | HRD labels (multi-modal) |
| TCGA-OV | ~560 | RNA-seq, WES, SNP array | Platinum response (PFI) |
| PCAWG | ~2800 | WGS + RNA-seq | HRDetect, CHORD, SBS3 |
| I-SPY2 | ~105 (olaparib arm) | Expression arrays | pCR |
| GDSC/CCLE | ~1000 cell lines | Expression + drug IC50 | PARPi/platinum sensitivity |
| GEO ovarian | ~800 combined | Expression arrays | Platinum response |
| METABRIC | ~2000 | Expression + CNV | Clinical |

## Compute

All compute runs through Slurm (see ~/.claude/CLAUDE.md):
- `srun --mem=XG [--gres=gpu:1] [--qos=short] --time=HH:MM:SS <cmd>`
- `sbatch --mem=XG [--gres=gpu:1] [--qos=training] --time=D-HH:MM:SS <script.sh>`
