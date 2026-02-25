# Feasibility Assessment: Genome Foundation Models for WGS-Based Drug Response Prediction

**Date:** 2026-02-25
**Context:** A peer reviewer (Leo) suggested training a transformer on whole-genome sequencing data to compute log-likelihood scores for gene sequences, essentially creating custom HRD scores from WGS to predict drug response. This document assesses the feasibility, available models, practical workflow, hardware requirements, and expected timeline for such an approach.

---

## 1. The Proposal

The reviewer's suggestion distills to: instead of training on RNA-seq gene expression, use a DNA sequence foundation model to encode genomic variants directly, then fine-tune a classifier head on drug response labels. The model would take raw DNA sequences (e.g., a gene region with its somatic variants) as input and output a per-patient score analogous to HRDetect or our RNA-based response score --- but learned end-to-end from sequence rather than hand-crafted from summary statistics (SBS3, LOH, etc.).

This is a meaningfully different approach from existing WGS classifiers like HRDetect and CHORD, which engineer features (mutational signature exposures, indel contexts, SV types) from called variants and then train a random forest or logistic regression. A foundation model approach would instead learn variant representations directly from sequence context, potentially capturing patterns that engineered features miss.

---

## 2. Available Genome Foundation Models

The genomics foundation model landscape has matured rapidly since 2023. Four model families are most relevant:

### 2.1 DNABERT-2 (117M parameters)

| Field | Details |
|-------|---------|
| **Architecture** | BERT-style transformer with Byte Pair Encoding (BPE) tokenization |
| **Training data** | Multi-species genomes (human, mouse, yeast, etc.) |
| **Context window** | Up to ~4,000 bp with standard positional embeddings |
| **Key advance** | Replaces fixed k-mer tokenization (DNABERT-1) with learned BPE tokens, improving cross-species transfer and capturing variable-length sequence motifs |
| **Publication** | ICLR 2024 (Zhou et al.) |
| **Availability** | Open-source, HuggingFace (`zhihan1996/DNABERT-2-117M`) |
| **Fine-tuning cost** | ~4-6 GB VRAM with LoRA; fits on RTX 5080 16GB comfortably |
| **Relevance** | Best suited for variant-level classification (e.g., pathogenicity of individual SNVs/indels within a sequence window). Natural fit for encoding local sequence context around somatic mutations |

### 2.2 Nucleotide Transformer (50M-500M parameters)

| Field | Details |
|-------|---------|
| **Architecture** | Transformer encoder, multiple model sizes (50M, 100M, 250M, 500M) |
| **Training data** | 3,202 diverse human genomes + 850 species genomes (for multi-species variants) |
| **Context window** | 6,000 bp (6-mer tokenization) |
| **Key advance** | Trained on population-level genomic diversity; embeddings capture variant effects in population context |
| **Publication** | Nature Methods 2023 (Dalla-Torre et al., InstaDeep/NVIDIA) |
| **Availability** | Open-source, HuggingFace (`InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`) |
| **Fine-tuning cost** | 500M model: ~8-10 GB VRAM with LoRA; fits on RTX 5080. 50M model: <4 GB |
| **Relevance** | Pre-trained on human population genomes, making it particularly relevant for encoding somatic variants relative to population-level variation. The 500M multi-species model has shown strong performance on variant effect prediction benchmarks |

### 2.3 HyenaDNA (~180M parameters)

| Field | Details |
|-------|---------|
| **Architecture** | Hyena operator (sub-quadratic long-range convolution), not a transformer |
| **Training data** | Human reference genome |
| **Context window** | Up to **1M bp** (and experimental 32K-450K variants available) |
| **Key advance** | Handles extremely long sequences efficiently via implicit convolutions; single-nucleotide resolution without tokenization |
| **Publication** | NeurIPS 2023 (Nguyen et al., Stanford) |
| **Availability** | Open-source, HuggingFace (`LongSafari/hyenadna-large-1m-seqlen`) |
| **Fine-tuning cost** | ~6-8 GB at 32K context; 1M context requires gradient checkpointing (~12-14 GB) |
| **Relevance** | The long context window is uniquely suited for encoding structural variants and copy number changes that span kilobases to megabases --- exactly the features (tandem duplications, large deletions, gene breakage events) that distinguish BRCA1-type from BRCA2-type HRD |

### 2.4 Evo / Evo 2 (7B-9.4B parameters)

| Field | Details |
|-------|---------|
| **Architecture** | StripedHyena (hybrid attention + Hyena layers) |
| **Training data** | Evo 2 trained on OpenGenome2, 9.3 trillion tokens across all domains of life |
| **Context window** | Up to **1M bp** (Evo 2) |
| **Key advance** | Largest genome model to date; generates functional DNA sequences (promoters, regulatory elements, even short gene-length sequences); single-nucleotide resolution; log-likelihood scores correspond to fitness effects |
| **Publication** | Evo: Science 2024 (Nguyen et al., Arc Institute); Evo 2: 2025 preprint |
| **Availability** | Open-source, HuggingFace (`togethercomputer/evo-1-131k-base`) |
| **Fine-tuning cost** | 7B model: **>40 GB VRAM** (does not fit RTX 5080). Requires A100 80GB or multi-GPU. Inference-only with 8-bit quantization: ~14 GB, marginal fit |
| **Relevance** | The log-likelihood scoring is directly what Leo suggested. Evo's per-nucleotide log-likelihoods have been shown to correlate with mutational fitness effects. A variant that reduces the sequence log-likelihood is more likely to be functionally damaging. This could directly produce the "custom HRD scores" Leo envisions --- but the model is too large for local fine-tuning |

### Model Selection Summary

| Model | Params | Context | Fine-tune on RTX 5080? | Best for |
|-------|--------|---------|------------------------|----------|
| DNABERT-2 | 117M | ~4K bp | Yes (LoRA, ~5 GB) | SNV/indel context windows |
| NT-500M | 500M | 6K bp | Yes (LoRA, ~10 GB) | Population-aware variant encoding |
| HyenaDNA | 180M | 1M bp | Yes (gradient ckpt, ~12 GB) | Structural variants, CNAs |
| Evo 2 | 9.4B | 1M bp | **No** (needs A100 80GB) | Zero-shot log-likelihoods |

**Recommendation:** Start with **DNABERT-2** or **NT-500M** for proof-of-concept (both fit on our hardware with LoRA). Use **HyenaDNA** if encoding structural variants is critical. **Evo** is the aspirational target for zero-shot log-likelihood scoring but requires cloud compute or institutional GPU cluster access for fine-tuning; inference-only with quantization may be feasible locally for scoring a limited number of sequences.

---

## 3. Practical Workflow

### 3.1 Data Preparation: VCF to Model Input

The key engineering challenge is converting variant call format (VCF) files into model-ready DNA sequences.

```
Step 1: Obtain VCF files
  ├── TCGA-OV: somatic mutation calls (MAF files from GDC, convertible to VCF)
  ├── Controlled access via dbGaP (phs000178)
  └── ~316 patients with WES + clinical response labels

Step 2: Extract context windows around variants
  ├── For each somatic variant (SNV, indel):
  │   ├── Extract reference sequence ±2,000 bp from hg38
  │   ├── Introduce the variant into the sequence (alt allele)
  │   └── Produce (reference, variant) sequence pairs
  ├── For structural variants / CNAs (if WGS available):
  │   ├── Extract breakpoint regions ±50K bp
  │   └── Requires HyenaDNA-class context windows
  └── Output: per-patient collection of (ref_seq, alt_seq) pairs

Step 3: Foundation model embedding
  ├── Forward pass through pre-trained model
  ├── Extract [CLS] token embedding or mean-pool hidden states
  ├── Compute delta embedding: embed(alt) - embed(ref)
  │   → Captures the model's "surprise" at the variant
  └── Output: per-variant embedding vectors (768-dim for DNABERT-2)

Step 4: Patient-level aggregation
  ├── Option A: Mean-pool all variant embeddings per patient
  ├── Option B: Attention-weighted aggregation (learnable)
  ├── Option C: Concatenate summary statistics
  │   (mean, max, variance of embeddings across variants)
  └── Output: one fixed-length vector per patient

Step 5: Fine-tune classifier head
  ├── Linear layer or small MLP on patient-level vectors
  ├── Binary target: platinum sensitive vs resistant
  ├── Same LODO-CV protocol as RNA model for fair comparison
  └── Output: per-patient response probability (our "genomic HRD score")
```

### 3.2 Alternative: Zero-Shot Log-Likelihood Scoring (Evo-style)

This is closer to what Leo described --- no fine-tuning, just scoring sequences:

```
For each patient:
  For each variant:
    score = log P(alt_sequence | model) - log P(ref_sequence | model)
    → negative score = variant reduces sequence likelihood = likely damaging

  Patient score = aggregate(variant scores)
    → e.g., sum of damaging variant scores in HR pathway genes
    → or: weighted sum across all genes, weights from HRDetect features
```

This requires only inference (no training), so even Evo could work with 8-bit quantization on the RTX 5080. However, the scoring is generic (not trained on drug response), so it would need to be combined with a downstream classifier.

### 3.3 Multi-Modal Integration (RNA + WGS)

Since TCGA-OV has both WES and RNA-seq for the same 316 patients, the ultimate goal is:

```
RNA model score  ──┐
                    ├──→ Combined predictor ──→ Response probability
WGS model score  ──┘
```

The simplest approach: logistic regression on (RNA_score, WGS_score) as two features. If the models capture complementary biology (RNA: immune microenvironment; WGS: genomic blueprint), the combination should outperform either alone.

---

## 4. Training Data Availability

From our [WGS dataset survey](wgs_dataset_survey.md), the following datasets are available:

| Dataset | N | Sequencing | Response Labels | Foundation Model Compatible? |
|---------|---|------------|-----------------|------------------------------|
| **TCGA-OV** | 316 | WES | Platinum sens/resist | Yes (MAF → context windows) |
| **Patch/AOCS** | 92 | Deep WGS | Plat refractory/resistant/sensitive | Yes (full WGS, best quality) |
| **BriTROC-1** | 276 | sWGS + WGS | Plat-sensitive vs resistant | Partial (sWGS too shallow for SNV calling; deep WGS subset n=56 usable) |
| **DECIDER** | 165-316 | WGS | Chemoresponse, PFS, OS | Yes (full WGS) |
| **PCAWG-OV** | ~112 | Deep WGS | Limited annotations | Yes for embeddings; limited for response training |

**Realistic starting point:** TCGA-OV (316 patients, WES). This is the only dataset where the same patients have both WES and RNA-seq, enabling direct comparison with our existing model. WES is sufficient for SNV/indel context windows (the primary input for DNABERT-2 and NT). Structural variant and CNA-based approaches require WGS (Patch/AOCS, DECIDER).

**Data access timeline:** TCGA-OV somatic mutation calls (MAF files) are available as open-access masked somatic mutations through GDC. Controlled access (full VCF with germline) requires dbGaP approval (~2-4 weeks). For proof-of-concept, the open-access masked MAFs are sufficient.

---

## 5. What Existing WGS Classifiers Already Achieve

Before building a foundation model approach, it's worth benchmarking against what hand-engineered WGS features already achieve:

| Classifier | Input | Performance | Notes |
|------------|-------|-------------|-------|
| **HRDetect** | 6 WGS features (SBS3, SBS8, SV types, microhomology dels, HRD index) | AUC=0.837 for HRD on ovarian WES (Sztupinszki 2021); HRDetect-high: median OS 6.2 vs 4.1 years | Gold standard. Retrained on WES achieves near-WGS performance |
| **CHORD** | SNV, indel, SV contexts (random forest) | Pan-cancer HRD; distinguishes BRCA1-type vs BRCA2-type | Requires WGS |
| **CN signatures** | 7 copy number signatures from sWGS | Predict platinum-resistant relapse in BriTROC-1 | Works from shallow WGS (~$50/sample) |
| **Myriad myChoice** | LOH + TAI + LST (SNP array or WES) | FDA-approved companion diagnostic for PARPi | Clinical standard; GIS ≥42 defines HRD-positive |

The foundation model approach must demonstrate value beyond these established classifiers. The potential advantages:

1. **End-to-end learning**: Foundation models can discover variant effect patterns that hand-crafted features miss. HRDetect uses 6 features; the model embedding space is 768-dimensional.
2. **Context-aware variant scoring**: Unlike simple mutation counting, foundation models encode the sequence context around each variant, potentially distinguishing functionally important mutations from passengers.
3. **Transfer learning**: Pre-training on billions of nucleotides provides a prior on sequence grammar that small training sets (n=316) cannot learn from scratch.
4. **Unified framework**: One model handles SNVs, indels, and (with long-context models) structural variants, rather than separate pipelines for each feature type.

The potential disadvantages:

1. **Sample size**: 316 patients is small for fine-tuning even with LoRA. Overfitting risk is high.
2. **Black box**: Interpreting what a transformer learns from DNA sequence is harder than interpreting HRDetect's 6 features.
3. **Unproven for clinical genomics**: These models excel on benchmark tasks (promoter prediction, splice site detection) but have not been validated for patient-level drug response prediction from somatic variants.
4. **Feature aggregation**: Going from per-variant embeddings to per-patient scores requires an aggregation step that introduces its own modeling choices and potential information loss.

---

## 6. Hardware Assessment

**Available hardware:** RTX 5080 16GB VRAM, Intel i5-8600K, 16GB RAM, Slurm job scheduler.

| Task | Model | VRAM Required | Fits? | Time Estimate |
|------|-------|---------------|-------|---------------|
| Fine-tune DNABERT-2 (LoRA, rank=8) | 117M | ~5 GB | Yes | ~2-4 hours per LODO fold |
| Fine-tune NT-500M (LoRA, rank=8) | 500M | ~10 GB | Yes | ~4-8 hours per LODO fold |
| Fine-tune HyenaDNA (LoRA, 32K context) | 180M | ~8 GB | Yes | ~3-6 hours per LODO fold |
| Fine-tune HyenaDNA (LoRA, 1M context) | 180M | ~14 GB | Tight (gradient ckpt) | ~8-16 hours per fold |
| Fine-tune Evo-7B (LoRA) | 7B | ~40 GB | **No** | Requires A100 80GB |
| Inference Evo-7B (8-bit quant) | 7B | ~14 GB | Marginal | ~10 min/patient |
| Embedding extraction (DNABERT-2) | 117M | ~4 GB | Yes | ~5 min/patient |
| Downstream classifier (sklearn) | - | CPU only | Yes | ~1 min total |

**Conclusion:** DNABERT-2 and NT-500M fine-tuning is comfortable on our hardware. HyenaDNA at moderate context is feasible. Evo requires either cloud compute or inference-only with quantization.

**Slurm job structure:**
```bash
# Embedding extraction (GPU, ~1 hour)
srun --mem=12G --gres=gpu:1 --qos=short --time=01:00:00 \
  python extract_embeddings.py --model dnabert2 --vcf tcga_ov_maf.tsv

# Fine-tune classifier (GPU, ~4 hours per fold)
sbatch --mem=12G --gres=gpu:1 --qos=training --time=0-06:00:00 \
  finetune_lodo.sh

# Downstream analysis (CPU only)
srun --mem=8G --time=00:30:00 \
  python train_response_classifier.py --embeddings embeddings.parquet
```

---

## 7. Expected Outcomes and Risks

### Optimistic scenario
- Foundation model embeddings capture variant effects that HRDetect's 6 features miss
- WGS-based response score achieves AUC 0.70-0.80 on TCGA-OV (comparable to RNA model)
- Multi-modal combination (RNA + WGS) pushes AUC above 0.75 by capturing complementary biology
- Timeline: 4-8 weeks for proof-of-concept

### Realistic scenario
- Foundation model embeddings perform comparably to HRDetect features (AUC ~0.65-0.75 for HRD, translating to ~0.55-0.65 for drug response)
- The small sample size (n=316) limits fine-tuning benefit; pre-trained embeddings + linear head performs similarly to LoRA fine-tuning
- Multi-modal combination shows modest improvement (+2-5 AUC points) over RNA alone
- The main value is demonstrating the workflow, not beating existing methods
- Timeline: 6-10 weeks

### Pessimistic scenario
- WES lacks the variant density needed for meaningful embeddings (WES covers ~2% of genome; most foundation model training is on non-coding sequence)
- 316 patients is too few for stable fine-tuning even with LoRA; overfitting dominates
- Aggregation from variant-level to patient-level embeddings loses critical information
- Foundation model adds no value over HRDetect + simple mutation features
- Timeline: 8-12 weeks of effort with negative result

### Key risks
1. **WES vs WGS gap**: Foundation models are trained on complete genomes. WES captures only exonic regions (~35 Mb of 3.2 Gb). The model's learned representations of non-coding sequence context may not transfer well to exon-only data.
2. **Variant density**: HGSOC has moderate mutation burden (~3-4 mutations/Mb in WES regions). With ~35 Mb captured, that's ~100-140 somatic variants per patient. This may be too few data points for robust patient-level aggregation.
3. **Label noise**: The same response label heterogeneity that affects our RNA model applies here. Training on noisy labels with only 316 samples and high-dimensional embeddings is a recipe for overfitting.
4. **Evaluation circularity**: If we use the same TCGA-OV patients for both RNA and WGS models, comparison is straightforward but we have no independent WGS-only validation cohort without controlled access applications.

---

## 8. Proposed Proof-of-Concept Plan

### Phase 1: Data preparation (Week 1-2)
- Download TCGA-OV open-access masked somatic mutation MAF files from GDC
- Map mutations to hg38 coordinates
- Extract reference ± 2,000 bp context windows for each variant
- Create (ref, alt) sequence pairs per patient
- Validate against known BRCA1/2 mutation calls

### Phase 2: Embedding extraction (Week 2-3)
- Extract DNABERT-2 embeddings for all (ref, alt) pairs
- Compute delta embeddings (alt - ref) per variant
- Aggregate to patient-level vectors (mean-pool, max-pool, attention)
- Also extract NT-500M embeddings for comparison
- Store as parquet for downstream analysis

### Phase 3: Response classifier (Week 3-4)
- Train L2 logistic regression on patient-level embeddings → response labels
- Use same LODO-CV framework as RNA model (TCGA-OV as one fold, cross-validate across datasets that have WES)
- **Practical limitation**: Only TCGA-OV has both WES and response labels in our current access. LODO-CV across WGS datasets requires controlled access to BriTROC-1, DECIDER, Patch/AOCS
- **Fallback**: 5-fold CV within TCGA-OV (n=316), with bootstrap CIs

### Phase 4: Comparison and integration (Week 4-6)
- Compare WGS model AUC vs RNA model AUC on same TCGA-OV patients
- Compare vs HRDetect scores (extract from TCGA-OV WES using sigtools)
- Train multi-modal model (RNA score + WGS score → response)
- Analyze whether models capture complementary biology:
  - Correlate WGS embeddings with HRD features (SBS3, LOH, BRCA status)
  - Correlate RNA scores with immune features (CIBERSORTx fractions)
  - If correlation between RNA and WGS scores is low, complementarity is supported

### Phase 5: Zero-shot Evo scoring (Week 5-8, optional)
- If RTX 5080 can run Evo-7B at 8-bit quantization:
  - Compute per-variant log-likelihood scores (no training needed)
  - Aggregate into patient-level "genomic surprise" scores
  - Compare with fine-tuned DNABERT-2 approach
- This is the closest implementation of Leo's original suggestion

---

## 9. How This Connects to Our RNA Work

The genome foundation model approach is **complementary to, not a replacement for**, our RNA-based response classifier. The key insight from our existing work is:

- **RNA captures the tumor microenvironment**: immune infiltration (M1 macrophages, IFN-gamma signaling), stromal composition, and the real-time transcriptional state. This is the biology our current model leverages.
- **WGS captures the genomic blueprint**: the mutations, structural rearrangements, and copy number changes that caused the tumor's phenotype. This is what HRDetect, CHORD, and foundation model approaches would capture.

These are different layers of the same biology. A BRCA1-mutant tumor (visible on WGS) with high M1 macrophage polarization (visible on RNA) behaves differently from one with immune evasion, even though their genomes are identical. Conversely, two tumors with similar immune profiles but different genomic instability patterns may respond differently to DNA-damaging agents.

The TCGA-OV cohort (316 patients with both WES and RNA-seq) is the natural proving ground for this multi-modal hypothesis. If the RNA and WGS models capture non-overlapping signal (low score correlation, distinct feature profiles), their combination should outperform either alone.

---

## 10. Conclusion

Training a genome foundation model on WGS data for drug response prediction is **feasible as a proof-of-concept** with our current hardware and publicly available data. The most practical path is:

1. **Start with DNABERT-2 or NT-500M** on TCGA-OV WES data (open access, same patients as our RNA model)
2. **Extract variant embeddings**, aggregate to patient-level, fine-tune on response labels
3. **Compare against HRDetect** (established baseline) and **our RNA model** (complementarity test)
4. **If successful, pursue multi-modal integration** (RNA + WGS scores)

The approach is novel in the ovarian cancer space --- to our knowledge, no published work has applied genome foundation models to somatic variant embeddings for drug response prediction. However, the small sample size (n=316), WES limitations (exome-only coverage), and label noise present genuine risks of a null result.

**Estimated timeline:** 4-8 weeks for Phase 1-4 (proof-of-concept with comparison). 8-12 weeks if including Evo zero-shot scoring and controlled-access dataset applications.

**Hardware requirements:** All phases except Evo fine-tuning fit on our RTX 5080 16GB. Evo inference may be feasible with 8-bit quantization.

---

## References

- Dalla-Torre, H., et al. (2023). "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics." *Nature Methods*.
- Nguyen, E., et al. (2023). "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution." *NeurIPS 2023*.
- Nguyen, E., et al. (2024). "Sequence modeling and design from molecular to genome scale with Evo." *Science* 386(6723).
- Nguyen, E., et al. (2025). "Evo 2: Genome modeling at all scales." *bioRxiv preprint*.
- Sztupinszki, Z., et al. (2021). "Migrating the SNP array-based homologous recombination deficiency measures to next generation sequencing data of human breast cancer." *npj Breast Cancer* 7:78.
- Zhou, Z., et al. (2024). "DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome." *ICLR 2024*.

---

*Assessment generated 2026-02-25. See [wgs_dataset_survey.md](wgs_dataset_survey.md) for detailed dataset descriptions.*
