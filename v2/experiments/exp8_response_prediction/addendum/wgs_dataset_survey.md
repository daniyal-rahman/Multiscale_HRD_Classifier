# Survey: WGS/WES Datasets with Drug Response Annotations for Ovarian Cancer

**Purpose:** Catalog publicly available datasets that have both whole genome/exome sequencing (WGS/WES) data and clinical drug response annotations, for potential future work training genomic-feature-based drug response models.

**Context:** A peer reviewer suggested training on WGS features (mutational signatures, CNV profiles, structural variants) instead of RNA-seq, arguing these could identify upstream genomic causes of drug response. This survey assesses data availability for such work.

---

## 1. Datasets with Both WGS/WES + Drug Response Data

### 1.1 TCGA-OV (The Cancer Genome Atlas — Ovarian Cancer)

| Field | Details |
|-------|---------|
| **Accession** | phs000178 (dbGaP); TCGA-OV project at GDC |
| **URL** | https://portal.gdc.cancer.gov/projects/TCGA-OV |
| **Cancer type** | High-grade serous ovarian adenocarcinoma (HGS-OvCa) |
| **Total patients** | 587 (489 in original analysis; 450 high-risk subset) |
| **WES samples** | 316 patients with whole-exome sequencing |
| **Response labels** | Platinum sensitivity status (sensitive vs resistant vs refractory); platinum-free interval; overall survival; progression-free survival. ~31% experienced disease progression within 6 months of completing platinum-based therapy |
| **Extractable genomic features** | Somatic mutations (TP53 in 96%, NF1, BRCA1, BRCA2, RB1, CDK12), focal CNAs (113 significant), gene-level mutations, TMB, HRD score (LOH/TAI/LST from SNP arrays), HRDetect from WES |
| **Access** | Open access: clinical, masked somatic mutations. Controlled access (dbGaP): raw WES BAMs, germline variants |
| **Overlap with our RNA datasets** | **YES — direct overlap.** TCGA-OV is one of our 9 RNA-seq training datasets. Same patients have both RNA-seq and WES, enabling multi-modal analysis |
| **Key references** | [TCGA 2011 Nature](https://www.nature.com/articles/nature10166); [Reannotation JCO CCI](https://ascopubs.org/doi/10.1200/CCI.17.00096) |
| **Notes** | No WGS — only WES. Sztupinszki et al. 2021 showed HRDetect retrained on TCGA-OV WES achieves AUC=0.837 for HRD classification and AUC=0.787-0.823 for predicting long-term platinum survival |

### 1.2 Patch et al. 2015 — ICGC Australian Ovarian Cancer Study (AOCS)

| Field | Details |
|-------|---------|
| **Accession** | EGAS00001000397 (EGA) |
| **URL** | https://ega-archive.org/studies/EGAS00001000397 |
| **Cancer type** | High-grade serous ovarian cancer |
| **N patients** | 92 patients with deep WGS (~60-100X) |
| **Response labels** | Primary refractory, primary resistant, primary sensitive, and matched acquired resistant disease — explicitly stratified by platinum response |
| **Extractable genomic features** | Full WGS: mutational signatures (SBS, DBS, ID), structural variants, gene breakage events (RB1, NF1, RAD51B, PTEN), CCNE1 amplification, BRCA1/2 reversion mutations, CNV profiles, HRD scores, HRDetect, CHORD |
| **Access** | Controlled access via ICGC/EGA Data Access Committee |
| **Overlap with our RNA datasets** | Unlikely direct overlap; AOCS samples are Australian |
| **Key reference** | [Patch et al. Nature 2015](https://www.nature.com/articles/nature14410) |
| **Notes** | **Gold standard dataset for WGS + platinum response.** First large-scale WGS study explicitly designed to characterize chemoresistance mechanisms. Found CCNE1 amplification enriched in primary resistant/refractory disease and identified gene breakage as resistance mechanism |

### 1.3 BriTROC-1 (British Translational Research Ovarian Cancer Collaborative)

| Field | Details |
|-------|---------|
| **Accession** | EGAS00001002557 (sWGS: EGAD00001004174; deep WGS: EGAD00001004189) |
| **URL** | https://ega-archive.org/studies/EGAS00001002557 |
| **Cancer type** | Recurrent high-grade serous ovarian cancer |
| **N patients** | 276 patients (209 platinum-sensitive, 67 platinum-resistant) |
| **Sequencing** | Shallow WGS (0.1X) for 319 samples from 300 tumors; deep WGS for 56 tumor-normal pairs |
| **Response labels** | Platinum-sensitive vs platinum-resistant at relapse; overall survival |
| **Extractable genomic features** | Copy number signatures (7 CN signatures), CCNE1/KRAS amplification, CN signature exposures, limited mutation calling from deep WGS subset |
| **Access** | Controlled access via BriTROC Data Access Committee (EGAC00001000518) |
| **Overlap with our RNA datasets** | No known overlap |
| **Key references** | [Macintyre et al. Nat Genet 2018](https://www.nature.com/articles/s41588-018-0179-8); [Smith & Bradley et al. Nat Commun 2023](https://www.nature.com/articles/s41467-023-39867-7) |
| **Notes** | **Critical finding:** Copy number signature exposures at diagnosis predict probability of platinum-resistant relapse. CN signature 1 enriched in primary platinum resistance. Code available at [github.com/BRITROC](https://github.com/BRITROC/britroc-1-HGSOC-landscape) |

### 1.4 DECIDER Trial (Finland)

| Field | Details |
|-------|---------|
| **Accession** | EGAS00001006775 (EGA); NCT04846933 (ClinicalTrials.gov) |
| **URL** | https://ega-archive.org/studies/EGAS00001006775 |
| **Cancer type** | High-grade serous ovarian carcinoma |
| **N patients** | 165-316 patients (discovery n=243, validation n=73); >640 tumor samples from multiregion sampling |
| **Sequencing** | Whole genome sequencing of 456 fresh tumor samples from multiple intra-abdominal regions |
| **Response labels** | Chemoresponse assessment, platinum sensitivity status, PFS, OS |
| **Extractable genomic features** | Full WGS: mutational signatures (SBS3/Sig3 validated as predictor), CNV profiles, structural variants, HRD scores, intratumor heterogeneity, clonal evolution |
| **Access** | Controlled access via EGA |
| **Overlap with our RNA datasets** | No known overlap (Finnish cohort) |
| **Key references** | [Vázquez-García et al. Cancer Discovery 2023](https://aacrjournals.org/cancerdiscovery/article/15/11/2262/766788); [DECIDER EU Project](https://cordis.europa.eu/project/id/965193/results) |
| **Notes** | Multiregion WGS enables study of intratumor heterogeneity and clonal evolution during treatment. SBS3/Sig3 predicts PFS and OS. EU-funded €15M project specifically targeting platinum resistance |

### 1.5 PCAWG Ovarian Cancer Subset

| Field | Details |
|-------|---------|
| **Accession** | EGAS00001001692 (EGA); part of ICGC PCAWG |
| **URL** | https://ega-archive.org/studies/EGAS00001001692 |
| **Cancer type** | Multiple cancer types including ovarian |
| **N ovarian patients** | ~112 ovarian samples (used as validation in CN signature studies) |
| **Sequencing** | Deep WGS (30-60X typical) |
| **Response labels** | Limited — PCAWG was primarily a genomic characterization project; clinical annotations vary by contributing study |
| **Extractable genomic features** | Full WGS: all mutational signatures, structural variants, CNVs, driver mutations, non-coding mutations |
| **Access** | Controlled access via ICGC DACO (EGAC00001000010) |
| **Overlap with our RNA datasets** | Some TCGA-OV samples are included in PCAWG |
| **Key reference** | [PCAWG Consortium Nature 2020](https://link.springer.com/article/10.1007/s10555-021-09969-z) |
| **Notes** | Large pan-cancer resource but clinical response data for ovarian subset is limited. Best used for extracting genomic features rather than training response models directly. 2,658 whole-cancer genomes across 38 tumor types total |

### 1.6 EGA Drug Screening + WGS Study (Ovarian Cell Lines)

| Field | Details |
|-------|---------|
| **Accession** | EGAS00001002239 |
| **URL** | https://ega-archive.org/studies/EGAS00001002239 |
| **Cancer type** | Ovarian cancer (patient-derived cell lines) |
| **N samples** | 9 patient-derived cell lines screened with 22 drugs/combinations |
| **Sequencing** | Whole genome sequencing |
| **Response labels** | In vitro drug sensitivity to carboplatin, olaparib, DNA demethylation drugs (22 drugs total) |
| **Extractable genomic features** | Full WGS: HRD scores, mutations, CNVs, structural variants |
| **Access** | Controlled access via EGA |
| **Overlap** | No overlap (cell lines, not patient tumors) |
| **Notes** | Small but useful for feature validation. Genome-derived HRD scores correlated with carboplatin and olaparib sensitivity |

### 1.7 Genomics England 100,000 Genomes Project

| Field | Details |
|-------|---------|
| **URL** | https://www.genomicsengland.co.uk/initiatives/100000-genomes-project/cancer |
| **Cancer type** | Pan-cancer including ovarian |
| **N ovarian patients** | ~454 ovarian cancer samples in the cancer programme |
| **Sequencing** | WGS (clinical-grade) |
| **Response labels** | Clinical outcomes linked to NHS records; treatment data available |
| **Extractable genomic features** | Full WGS: HRD status (40% of HGSOC found HRD), BRCA1/2 germline variants (13% of HGSOC), mutational signatures, CNVs, SVs |
| **Access** | Highly restricted — requires Genomics England Research Environment access, UK-based institution affiliation |
| **Overlap with our RNA datasets** | No overlap (UK NHS cohort) |
| **Key references** | [Sosinsky et al. Nat Med 2024](https://www.nature.com/articles/s41591-023-02682-0); [JCO 2024](https://ascopubs.org/doi/10.1200/JCO.23.02761) |
| **Notes** | Large-scale but access is very restricted. Clinical-grade WGS with NHS treatment records. Supports pharmacogenomics analysis |

### 1.8 Clonal Somatic CNA Driver Events Study

| Field | Details |
|-------|---------|
| **Accession** | EGAD00001008716 (EGA) |
| **URL** | https://ega-archive.org/datasets/EGAD00001008716 |
| **Cancer type** | High-grade serous ovarian cancer |
| **N patients** | 26 patients (21 HGSOC, 4 LGSOC, 1 clear cell) |
| **Sequencing** | Shallow WGS + amplicon sequencing |
| **Response labels** | Drug sensitivity annotations |
| **Access** | Controlled access via EGA |
| **Notes** | Small but focused on drug sensitivity. Study: "Clonal somatic copy number altered driver events inform drug sensitivity in HGSOC" |

---

## 2. Datasets with Partial Overlap or Limited Response Data

### 2.1 AACR Project GENIE

| Field | Details |
|-------|---------|
| **URL** | https://genie.synapse.org/ ; https://www.aacr.org/professionals/research/aacr-project-genie/ |
| **Cancer type** | Pan-cancer (227,696 patients total in v19.0) |
| **Sequencing** | Targeted panel sequencing (NOT WGS/WES) — varies by institution |
| **Response labels** | GENIE BPC has detailed treatment and response data for select cancer types |
| **Ovarian cancer status** | **No dedicated ovarian cancer BPC cohort yet.** Currently released BPC cohorts: NSCLC, CRC, breast, bladder, pancreatic, prostate. Phase 2 may expand to additional cancer types |
| **Access** | Open access (registry data); BPC requires Synapse account |
| **Notes** | Panel sequencing only — not suitable for full WGS feature extraction. May be useful for targeted mutation analysis but cannot derive mutational signatures, structural variants, or comprehensive CNV profiles |

### 2.2 cBioPortal Ovarian Cancer Studies

| Field | Details |
|-------|---------|
| **URL** | https://www.cbioportal.org/ |
| **Content** | Aggregates multiple ovarian cancer studies including TCGA-OV |
| **Sequencing** | Mostly WES and targeted panels |
| **Response labels** | Clinical data varies by study; TCGA-OV has platinum response |
| **Notes** | Useful as a portal for accessing and visualizing existing datasets rather than a unique data source. Most WES data comes from TCGA-OV already cataloged above |

### 2.3 ICON7 Trial

| Field | Details |
|-------|---------|
| **Accession** | EGAD00001004988 (EGA) |
| **Cancer type** | High-grade serous ovarian cancer |
| **N patients** | 370 with RNA-seq; GWAS on 437 patients |
| **Sequencing** | RNA-seq and GWAS genotyping — **no WGS/WES** |
| **Response labels** | Bevacizumab response, PFS, OS |
| **Notes** | RNA-seq only, no WGS/WES. Relevant to our RNA-based work but not for WGS feature extraction |

### 2.4 SCOTROC Trials

| Field | Details |
|-------|---------|
| **Cancer type** | Advanced ovarian cancer |
| **N patients** | ~1,067 (SCOTROC I) |
| **Sequencing** | **No WGS/WES data found** |
| **Response labels** | PFS, OS for docetaxel/carboplatin vs paclitaxel/carboplatin |
| **Notes** | Clinical trial data without genomic sequencing. Not suitable for WGS-based analysis |

---

## 3. WGS-Derived Features for Drug Response Prediction

### 3.1 Mutational Signatures

| Feature | Description | Drug Response Relevance |
|---------|-------------|------------------------|
| **SBS3** | Associated with defective homologous recombination repair (BRCA1/2 deficiency) | Predicts platinum sensitivity and PARPi response. AUC=0.837 (HRDetect trained on ovarian WES). COSMIC validated |
| **SBS8** | Also associated with HRD | Co-occurs with SBS3; used by HRDetect |
| **Other SBS** | Age-related (SBS1, SBS5), APOBEC (SBS2, SBS13) | May contribute to TMB and neoantigen load |
| **ID signatures** | Small insertions/deletions with microhomology | Microhomology-mediated deletions are HRD hallmarks |
| **DBS signatures** | Doublet base substitutions | Some associated with chemotherapy exposure |

**Tools:** SigProfiler, deconstructSigs, MutationalPatterns, signal
**Key limitation:** WES can detect SBS signatures but with reduced accuracy vs WGS (fewer mutations to decompose)

### 3.2 Copy Number Profiles

| Feature | Description | Drug Response Relevance |
|---------|-------------|------------------------|
| **HRD-LOH** | Number of LOH regions >15 Mb | Component of HRD score; predicts PARPi response |
| **TAI** | Telomeric allelic imbalance (>11 Mb extending to subtelomere) | Component of HRD score |
| **LST** | Large-scale state transitions (breaks between regions >10 Mb) | Component of HRD score |
| **HRD score** | Sum of LOH + TAI + LST | Myriad myChoice CDx uses HRD ≥42 as cutoff for PARPi eligibility |
| **Copy number signatures** | 7 CN signatures from Macintyre et al. | **Predict platinum-resistant relapse and OS.** CN-Sig1 enriched in primary platinum resistance |
| **Focal events** | CCNE1 amplification, KRAS amplification, MYC amplification | CCNE1 amp. strongly associated with primary platinum resistance |
| **Arm-level events** | Chromosome arm gains/losses | Contribute to genomic instability phenotype |

**Tools:** ASCAT, Sequenza, FACETS, CNVkit
**Key finding:** Macintyre CN signatures from sWGS are currently the strongest WGS-derived predictor of platinum response

### 3.3 Structural Variants

| Feature | Description | Drug Response Relevance |
|---------|-------------|------------------------|
| **SV burden** | Total count of structural variants | Higher in HRD tumors |
| **Tandem duplications** | Small tandem duplications (<10 kb) | Enriched in BRCA1-mutant tumors specifically |
| **Deletions** | Large genomic deletions | Enriched in BRCA2-mutant tumors |
| **Gene breakage** | SVs disrupting tumor suppressors (RB1, NF1, RAD51B, PTEN) | Contributes to acquired chemotherapy resistance (Patch 2015) |
| **Complex rearrangements** | Chromothripsis, chromoplexy | Associated with genome instability and poor prognosis |
| **BRCA1/2 reversions** | Reversion mutations restoring BRCA reading frame | Direct mechanism of acquired PARPi/platinum resistance |

**Tools:** DELLY, Manta, GRIDSS, SvABA
**Key limitation:** SV calling requires WGS (≥30X); not possible from WES or panels

### 3.4 Gene-Level Mutations (Beyond BRCA1/2)

| Gene | Alteration | Response Implication |
|------|-----------|---------------------|
| **TP53** | Mutated in 96% of HGSOC | Near-universal; limited discriminative value |
| **BRCA1/2** | Germline/somatic mutations | Best-established predictor of platinum/PARPi sensitivity |
| **CDK12** | Biallelic inactivation | Associated with tandem duplicator phenotype; possible PARPi sensitivity |
| **NF1** | Inactivation (mutation/breakage) | Possible resistance mechanism |
| **RB1** | Inactivation | Associated with acquired resistance |
| **CCNE1** | Amplification | Strong predictor of primary platinum resistance |
| **RAD51B/C/D, PALB2** | HR pathway mutations | PARPi sensitivity |
| **ABCB1** | Promoter fusion/upregulation | Drug efflux pump; acquired platinum/PARPi resistance |

### 3.5 Composite/Integrative Classifiers

| Classifier | Input | Performance |
|------------|-------|-------------|
| **HRDetect** | WGS-derived (6 features: SBS3, SBS8, SV types, mmdels, HRD index, microhomology) | AUC=0.837 retrained on ovarian WES (Sztupinszki 2021). Patients with HRDetect-high: median OS 6.2 vs 4.1 years |
| **CHORD** | WGS-derived (SNV, indel, SV contexts) | Random forest classifier. Pan-cancer HRD detection. Distinguishes BRCA1-type vs BRCA2-type HRD |
| **CN signatures** | sWGS (0.1X sufficient) | 7 signatures predict platinum-resistant relapse. Cost-effective (sWGS ~$50/sample) |
| **DirectHRD** | WGS-derived | Newer scar-based HRD classifier with improved sensitivity |

---

## 4. Summary Table: Datasets Ranked by Utility

| Dataset | N (WGS/WES + Response) | Sequencing | Response Labels | Access | Overlap w/ Our RNA Data |
|---------|------------------------|------------|-----------------|--------|------------------------|
| **TCGA-OV** | ~316 (WES) | WES | Platinum sensitivity/resistance/refractory, PFI, OS | Controlled (dbGaP) | **YES** |
| **BriTROC-1** | ~276 (sWGS/WGS) | sWGS + deep WGS subset | Platinum-sensitive vs resistant at relapse | Controlled (EGA) | No |
| **DECIDER** | ~165-316 (WGS) | WGS (multiregion) | Chemoresponse, platinum status, PFS, OS | Controlled (EGA) | No |
| **Patch/AOCS** | 92 (WGS) | Deep WGS | Refractory/resistant/sensitive/acquired resistant | Controlled (EGA) | No |
| **PCAWG-OV** | ~112 (WGS) | Deep WGS | Limited clinical annotations | Controlled (EGA/ICGC) | Partial (TCGA subset) |
| **Genomics England** | ~454 (WGS) | Clinical WGS | NHS treatment records | Very restricted | No |
| **EGA Drug Screen** | 9 cell lines (WGS) | WGS | In vitro drug sensitivity | Controlled (EGA) | No |
| **EGA CNA Drivers** | 26 (sWGS) | sWGS | Drug sensitivity | Controlled (EGA) | No |

**Total unique patients with WGS/WES + platinum response data: ~900-1,100** (accounting for TCGA overlap between datasets)

---

## 5. Feasibility Assessment

### What's realistic for a WGS-based response predictor:

**Strengths:**
- ~900+ patients with WGS/WES + platinum response is a reasonable starting point
- TCGA-OV has both WES and RNA-seq for the same patients, enabling direct comparison
- CN signatures from sWGS are already proven predictors (Macintyre 2018, BriTROC-1)
- Multiple validated classifiers exist (HRDetect, CHORD, CN signatures) that could serve as features

**Challenges:**
- Most datasets require controlled access applications (weeks-months to obtain)
- WGS vs WES: full mutational signatures and SVs require WGS; WES is more limited
- Cross-cohort harmonization is non-trivial (different WGS pipelines, reference genomes, depths)
- Platinum response definitions vary across studies (6-month cutoff vs continuous PFI)
- Sample sizes per dataset are smaller than RNA-seq cohorts (largest is TCGA-OV at 316 WES)

**Recommended approach:**
1. Start with TCGA-OV WES (controlled access) — direct comparison with our RNA models on same patients
2. Extract HRD scores, HRDetect, gene mutations, CN profiles as features
3. Validate on Patch/AOCS WGS cohort (92 patients, gold-standard response labels)
4. Incorporate BriTROC-1 CN signatures as external validation

---

## 6. Future Direction: Genomic Features (Discussion Paragraph)

The reviewer raises a genuinely interesting point about using whole-genome sequencing features to predict platinum response. WGS captures upstream genomic causes — mutational signatures like SBS3 that directly reflect defective DNA repair, structural variant patterns that reveal mechanism-specific genome instability, and copy number profiles that have been shown to predict platinum-resistant relapse (Macintyre et al. 2018). Several validated classifiers already exist: HRDetect integrates six WGS-derived features to identify HRD tumors with median OS of 6.2 vs 4.1 years (HRDetect-high vs low), and CHORD provides pan-cancer HRD classification from SNV/indel/SV contexts. Copy number signatures derived from even shallow (0.1X) whole genome sequencing predict probability of platinum-resistant relapse in the BriTROC-1 cohort.

However, RNA-seq and WGS capture fundamentally different biology, and both have blind spots the other can see through. RNA-seq captures the tumor microenvironment — immune cell infiltration, stromal interactions, signaling pathway activity, and the real-time transcriptional state of the tumor at the moment of biopsy. Our HRD classifier works precisely because it captures these downstream consequences of genomic instability, including immune activation signatures that reflect how the host responds to DNA damage. WGS, by contrast, captures the static genomic blueprint: the mutations, rearrangements, and copy number changes that caused the tumor to behave as it does, but it cannot tell you whether a BRCA1 mutation has been functionally compensated by an alternative pathway, or whether the immune system has been recruited to the fight. The two modalities are complementary rather than competitive — a BRCA1-mutant tumor with high immune infiltration (visible on RNA-seq) behaves very differently from one that has evaded immune surveillance, even though their genomes look identical.

The practical landscape supports this complementarity: approximately 900-1,100 ovarian cancer patients have both WGS/WES and platinum response data across publicly available datasets (TCGA-OV, BriTROC-1, DECIDER, Patch/AOCS, PCAWG), with TCGA-OV offering direct comparison since the same 316 patients have both WES and RNA-seq. A natural next step would be training genomic-feature models on these cohorts and comparing — or ideally integrating — their predictions with our RNA-based classifiers to build a truly multi-omic response predictor. The WGS features most likely to add complementary signal are copy number signatures, HRDetect scores, and structural variant burden, as these capture mechanistic information that RNA cannot directly reflect.
