# Additional Public Datasets for HRD Research

**Compiled: 2026-02-23**

This catalog identifies publicly available datasets relevant to Homologous Recombination
Deficiency (HRD) research that are NOT already in our pipeline. Organized by priority.

**Datasets we already have (excluded from this catalog):**
- TCGA-BRCA (1111 samples)
- TCGA-OV (~560 samples)
- GEO ovarian: GSE9891, GSE26712, GSE51088, GSE63885, GSE30161, GSE32062
- I-SPY2 (105 samples, durvalumab/olaparib arm)
- GDSC/CCLE cell lines
- METABRIC (2509 breast, clinical only so far)
- PCAWG pan-cancer (controlled access)

---

## PRIORITY 1: PARPi / Platinum Treatment Response + Expression Data

These datasets contain pre-treatment gene expression paired with drug response outcomes.
This is the most valuable data category for training/validating our classifier.

### 1.1 GSE113863 -- Neoadjuvant Talazoparib in gBRCA+ Breast Cancer

| Field | Value |
|---|---|
| **Accession** | [GSE113863](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113863) |
| **Samples** | 13 treatment-naive gBRCA+ breast tumors |
| **Cancer type** | Breast cancer (gBRCA1/2+) |
| **Data types** | RNA-seq + whole-exome sequencing (pre-treatment core needle biopsies) |
| **Platform** | Illumina RNA-seq |
| **Key endpoint** | Residual cancer burden (RCB) after 6 months talazoparib monotherapy |
| **Access** | Public (GEO); also available via EGA under EGAD00001008270 (controlled) |
| **Download** | `wget` from GEO supplementary files or GEOquery in R |
| **Key findings** | All resistant tumors exhibited loss of SHLD2, hypoxia signature, or stem cell signature |
| **Citation** | Chopra N et al., npj Breast Cancer, 2022 |

**Why critical:** Single-agent PARPi with pre-treatment expression + response. Gold standard for
validating whether our HRD signature predicts PARPi response.

---

### 1.2 I-SPY2-990 Data Resource -- Veliparib/Carboplatin Arm

| Field | Value |
|---|---|
| **Accession** | [GSE194040](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194040) (mRNA SubSeries of GSE196096 SuperSeries) |
| **Samples** | ~987 patients across 10 arms; veliparib/carboplatin arm has ~72 patients (HER2-) |
| **Cancer type** | Breast cancer (neoadjuvant) |
| **Data types** | Gene expression (Agilent microarray, 19,134 genes after batch correction), RPPA |
| **Key endpoint** | pCR (pathologic complete response) |
| **Access** | Public (GEO + I-SPY2 Google Cloud: www.ispytrials.org/results/data) |
| **Download** | GEO supplementary files or I-SPY2 Google Cloud repository |
| **Key findings** | PARPi-7 gene signature, BRCA1ness 77-gene signature, MammaPrint predict response to veliparib/carboplatin |
| **Citation** | Wolf DM et al., Cell Reports Medicine, 2024; Rugo HS et al., NEJM, 2016 |

**Why critical:** We already have 105 samples from the durvalumab/olaparib arm. The full 990-patient
dataset includes the veliparib/carboplatin arm (PARP inhibitor + DNA-damaging agent) with
pre-treatment expression and pCR. This is an independent PARPi validation cohort.

**Note:** Expression data from all 10 arms can be used for biomarker discovery. The veliparib/carboplatin
arm is highest priority, but paclitaxel-only control arm is also valuable as a negative control.

---

### 1.3 GSE51373 -- Platinum-Sensitive vs Resistant HGSOC

| Field | Value |
|---|---|
| **Accession** | [GSE51373](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE51373) |
| **Samples** | 28 (12 platinum-resistant, 16 platinum-sensitive) |
| **Cancer type** | High-grade serous ovarian carcinoma |
| **Data types** | Gene expression microarray (Affymetrix HG-U133 Plus 2.0) |
| **Key endpoint** | Platinum sensitivity (PFS < 8 months = resistant; PFS > 18 months = sensitive) |
| **Access** | Public |
| **Download** | GEOquery or direct download from GEO |
| **Citation** | Helleman J et al. |

**Why critical:** Direct platinum response labels with expression profiling. Small but clean dataset.

---

### 1.4 GSE23554 -- Ovarian Cancer Chemotherapy Response

| Field | Value |
|---|---|
| **Accession** | [GSE23554](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE23554) |
| **Samples** | 28 (18 responders, 10 non-responders) |
| **Cancer type** | Serous ovarian cancer |
| **Data types** | Gene expression microarray |
| **Key endpoint** | Chemotherapy response (platinum-based) |
| **Access** | Public |
| **Download** | GEOquery |
| **Citation** | Integrated Analysis of Gene Expression and Tumor Nuclear Image Profiles (2012) |

---

### 1.5 GSE3149 -- Ovarian Cancer with Platinum/Taxane Response

| Field | Value |
|---|---|
| **Accession** | [GSE3149](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE3149) |
| **Samples** | ~153 (ovarian cancer with chemotherapy response data) |
| **Cancer type** | Ovarian cancer |
| **Data types** | Gene expression microarray |
| **Key endpoint** | Chemotherapy response and survival (platinum + taxane regimens) |
| **Access** | Public |
| **Download** | GEOquery |
| **Citation** | Dressman HK et al., JCO, 2007 |

**Why critical:** Larger ovarian cohort with clinical drug response labels.

---

### 1.6 GSE28739 -- Platinum-Resistant vs Sensitive Ovarian Cancer

| Field | Value |
|---|---|
| **Accession** | [GSE28739](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE28739) |
| **Samples** | Small cohort (platinum-resistant vs platinum-sensitive) |
| **Cancer type** | Ovarian cancer |
| **Data types** | mRNA expression microarray |
| **Key endpoint** | Platinum resistance vs sensitivity |
| **Access** | Public |
| **Download** | GEOquery |
| **Citation** | Used in Deng et al. (PeerJ, 2021) and others |

---

### 1.7 Patch et al. 2015 -- Chemoresistant Ovarian Cancer (WGS + Expression)

| Field | Value |
|---|---|
| **Accession** | EGA: [EGAD00001000877](https://ega-archive.org/datasets/EGAD00001000877) (WGS/RNA-seq); GEO: [GSE65821](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65821) (methylation/miRNA) |
| **Samples** | 92 patients (primary refractory, resistant, sensitive, matched acquired-resistant) |
| **Cancer type** | High-grade serous ovarian cancer |
| **Data types** | Whole-genome sequencing, RNA-seq, methylation, miRNA, SNP arrays |
| **Key endpoint** | Platinum resistance status (refractory/resistant/sensitive/acquired-resistant) |
| **Access** | Controlled (EGA -- ICGC DAC); GEO component is public |
| **Download** | EGA data access request via ICGC DAC (EGAC00001000010) |
| **Key findings** | BRCA1/2 reversion mutations, loss of BRCA1 methylation, MDR1 promoter fusion |
| **Citation** | Patch AM et al., Nature, 2015 |

**Why critical:** Contains BRCA reversion mutations + platinum resistance + expression/WGS.
Overlaps with Priority 1 AND Priority 3 (reversion mutations). Extremely valuable.

---

### 1.8 GSE98230 -- Platinum Sensitive vs Resistant Ovarian Cancer Cell Lines

| Field | Value |
|---|---|
| **Accession** | [GSE98230](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98230) |
| **Samples** | A2780 (sensitive) and A2780cis (resistant) cell lines, multiple conditions |
| **Cancer type** | Ovarian cancer cell lines |
| **Data types** | RNA-seq |
| **Key endpoint** | Platinum sensitivity/resistance |
| **Access** | Public |
| **Download** | GEOquery or SRA |
| **Citation** | van Jaarsveld et al. |

---

## PRIORITY 2: Functional HRD (RAD51 Foci / RECAP Test) + Expression

### 2.1 Meijer et al. 2018 -- RECAP Test in Breast Cancer

| Field | Value |
|---|---|
| **Accession** | No GEO accession identified for expression data |
| **Samples** | 170 primary breast cancers tested with RECAP (RAD51 foci) assay |
| **Cancer type** | Breast cancer |
| **Data types** | RECAP test results (RAD51 foci), BRCA status, mutational signatures, TIL, MSI |
| **Key endpoint** | Functional HRD status (RAD51 foci formation) |
| **Access** | Data not publicly deposited as microarray/RNA-seq; contact authors |
| **Citation** | Meijer TG et al., Clinical Cancer Research, 2018 (PMID: 30139880) |

**Status:** The RECAP assay is a functional (ex vivo irradiation) test, not a gene expression
study. The Erasmus MC group (Meijer, Naipal, van Dijk) has not deposited paired
gene expression data with RAD51 foci results in public repositories. Their data would
need to be obtained via collaboration or data sharing agreement.

---

### 2.2 RAD51 Foci as PARPi Resistance Biomarker

| Field | Value |
|---|---|
| **Accession** | No public GEO dataset identified |
| **Samples** | PDX models and patient tumors |
| **Cancer type** | gBRCA-mutated breast cancer |
| **Data types** | RAD51 foci immunofluorescence |
| **Key endpoint** | PARPi resistance |
| **Citation** | Castroviejo-Bermejo et al., Annals of Oncology, 2018 |

**Status:** Functional assay data; no paired transcriptomics deposited publicly.

---

### 2.3 GSE54266 -- BRCA1/RAD51 Knockdown Expression Profiles

| Field | Value |
|---|---|
| **Accession** | [GSE54266](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54266) |
| **Samples** | Cell line knockdown experiments (BRCA1 and RAD51 RNAi) |
| **Cancer type** | Breast cancer cell lines |
| **Data types** | Gene expression microarray post-knockdown |
| **Key endpoint** | Expression changes upon loss of BRCA1/RAD51 function |
| **Access** | Public |
| **Download** | GEOquery |
| **Citation** | Peng G et al. (used in Genome Medicine, 2016) |

**Why useful:** Provides reference knockdown profiles for BRCA1/RAD51 deficiency. Can be used
to derive "BRCAness" similarity scores for tumors.

---

## PRIORITY 3: BRCA Reversion Mutations + Expression

### 3.1 Patch et al. 2015 (See Priority 1.7 above)

The most comprehensive dataset with BRCA reversions + expression. Contains multiple
independent BRCA1/2 reversion events in individual patients, paired with WGS and RNA-seq.
EGA accession: EGAD00001000877.

---

### 3.2 Lin et al. 2019 -- BRCA Reversion Mutations in ctDNA

| Field | Value |
|---|---|
| **Accession** | No public expression data; ctDNA sequencing from clinical trials |
| **Studies involved** | LIGHT (NCT02983799), SOLO3 (NCT02282020), OlympiAD (NCT02000622) |
| **Samples** | >500 plasma samples from ovarian/breast cancer patients with BRCA1/2 mutations |
| **Data types** | Targeted ctDNA sequencing |
| **Key endpoint** | BRCA reversion mutations at progression on olaparib/chemotherapy |
| **Access** | Controlled; contact AstraZeneca/sponsors |
| **Citation** | Lin KK et al., Cancer Discovery, 2019 |

**Status:** ctDNA data only (no tumor expression profiles). Limited utility for our expression-based
classifier, but useful for understanding reversion biology.

---

### 3.3 Quigley et al. 2018 -- mCRPC WGS/RNA-seq (reversion mutations observed)

| Field | Value |
|---|---|
| **Accession** | dbGaP: [phs001648.v2.p1](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001648.v2.p1); EGA: [EGAS00001005954](https://ega-archive.org/studies/EGAS00001005954) |
| **Samples** | 101 mCRPC tumor biopsies (WGS); expanded to ~224 in later releases (RNA-seq: EGAD00001008487, EGAD00001008991, EGAD00001009065) |
| **Cancer type** | Metastatic castration-resistant prostate cancer |
| **Data types** | WGS + RNA-seq |
| **Key endpoint** | Resistance mechanisms (including BRCA2 reversions); AR alterations |
| **Access** | Controlled (dbGaP + EGA) |
| **Download** | Apply via dbGaP or EGA DAC; also available on AWS Open Data (registry.opendata.aws/mcrpc) |
| **Citation** | Quigley DA et al., Cell, 2018; de Sarkar et al., Nat Commun, 2021 |

**Why critical:** Large mCRPC cohort with both WGS and RNA-seq. Contains samples with
BRCA2 mutations (including reversions). Valuable for both reversion analysis and
pan-cancer HRD generalization to prostate cancer. The AWS Open Data portal may
allow faster access than standard dbGaP.

---

## PRIORITY 4: Prostate Cancer HRD Datasets

### 4.1 SU2C/PCF mCRPC 2015 (Robinson et al.)

| Field | Value |
|---|---|
| **Accession** | cBioPortal: [prad_su2c_2015](https://www.cbioportal.org/study/summary?id=prad_su2c_2015) |
| **Samples** | 150 mCRPC (WES + RNA-seq) |
| **Cancer type** | Metastatic castration-resistant prostate cancer |
| **Data types** | Whole-exome sequencing, RNA-seq, clinical |
| **Key endpoint** | HRR gene alterations (BRCA2/BRCA1/ATM at 19.3% frequency), treatment outcomes |
| **Access** | Public (cBioPortal) and dbGaP (phs000915) |
| **Download** | cBioPortal bulk download or GitHub: github.com/cBioPortal/datahub/tree/master/public/prad_su2c_2015 |
| **Citation** | Robinson DR et al., Cell, 2015 |

**Why critical:** First large mCRPC WES+RNA-seq cohort. Public expression data via cBioPortal
with known HRR mutation status. Direct download, no DAC needed.

---

### 4.2 SU2C/PCF mCRPC 2019 (Abida et al.)

| Field | Value |
|---|---|
| **Accession** | cBioPortal: [prad_su2c_2019](https://www.cbioportal.org/study?id=prad_su2c_2019) |
| **Samples** | 429 mCRPC patients (208 with RNA-seq) |
| **Cancer type** | Metastatic castration-resistant prostate cancer |
| **Data types** | WES, RNA-seq, clinical outcomes |
| **Key endpoint** | Clinical outcomes correlated with genomic subtypes; HRR alterations |
| **Access** | Public (cBioPortal + GitHub) |
| **Download** | cBioPortal bulk download or GitHub: github.com/cBioPortal/datahub/tree/master/public/prad_su2c_2019 |
| **Citation** | Abida W et al., PNAS, 2019 |

**Why critical:** Larger updated cohort of SU2C. ~208 samples with RNA-seq + mutation status
for HRR genes. Can identify BRCA2/ATM/CDK12-mutant tumors and test our HRD
signature on them.

---

### 4.3 TCGA-PRAD

| Field | Value |
|---|---|
| **Accession** | GDC: [TCGA-PRAD](https://portal.gdc.cancer.gov/projects/TCGA-PRAD); cBioPortal: [prad_tcga](https://www.cbioportal.org/study?id=prad_tcga) |
| **Samples** | ~500 primary prostate adenocarcinoma |
| **Cancer type** | Primary prostate cancer |
| **Data types** | RNA-seq, WES, SNP array, methylation, clinical |
| **Key endpoint** | HRD score (LOH + TAI + LST from SNP arrays); BRCA2/ATM mutation status |
| **Access** | Public (processed data via cBioPortal/GDC); controlled (raw BAMs via dbGaP) |
| **Download** | GDC Data Portal or cBioPortal |
| **Key findings** | Median HRD score = 11 in wild-type; ~27 in germline BRCA2-mutant tumors |
| **Citation** | TCGA Research Network, Cell, 2015 |

**Why useful:** Primary prostate cancer (low HRD prevalence) but provides baseline for
pan-cancer extension. HRD scores can be computed from SNP arrays.

---

### 4.4 GSE35988 -- Grasso et al. mCRPC

| Field | Value |
|---|---|
| **Accession** | [GSE35988](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE35988) |
| **Samples** | 122 (28 benign, 59 localized PCa, 35 metastatic CRPC) |
| **Cancer type** | Prostate cancer (benign to mCRPC) |
| **Data types** | Gene expression microarray (Agilent 4x44K) |
| **Key endpoint** | Progression from localized to metastatic CRPC |
| **Access** | Public |
| **Download** | GEOquery |
| **Citation** | Grasso CS et al., Nature, 2012 |

---

### 4.5 TOPARP-A/B -- Olaparib in mCRPC (Limited Public Data)

| Field | Value |
|---|---|
| **Accession** | No public GEO/EGA accession identified for RNA-seq data |
| **Samples** | TOPARP-A: 50 men; TOPARP-B: ~98 men |
| **Cancer type** | mCRPC with DDR gene alterations |
| **Data types** | WES, RNA-seq (transcriptome), targeted sequencing |
| **Key endpoint** | Response to olaparib (radiographic response, PSA decline, CTC conversion) |
| **Access** | Controlled; data access through Institute of Cancer Research |
| **Citation** | Mateo J et al., NEJM, 2015 (TOPARP-A); Mateo J et al., Lancet Oncol, 2020 (TOPARP-B) |

**Status:** The TOPARP trials included transcriptome sequencing but the data does not appear
to be publicly deposited in EGA/dbGaP. Contact authors at ICR for potential collaboration.
TOPARP-B showed BRCA2 reversion mutations in 79% of progressing tumors.

---

### 4.6 16-GPS HRDness Signature Validation Datasets

| Field | Value |
|---|---|
| **Accession** | [GSE143791](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE143791) and additional datasets listed in the paper's Key Resources Table |
| **Samples** | Multiple prostate cancer cohorts used for signature validation |
| **Cancer type** | Prostate cancer |
| **Data types** | Gene expression |
| **Key endpoint** | HRDness classification (16 gene-pair signature) |
| **Access** | Public |
| **Citation** | Deng Y et al., iScience, 2021 |

**Why useful:** Pre-validated HRD gene expression signature for prostate cancer. Can compare
with our classifier.

---

## PRIORITY 5: Pancreatic Cancer HRD Datasets

### 5.1 TCGA-PAAD

| Field | Value |
|---|---|
| **Accession** | GDC: [TCGA-PAAD](https://portal.gdc.cancer.gov/projects/TCGA-PAAD); cBioPortal: [paad_tcga](https://www.cbioportal.org/study?id=paad_tcga) |
| **Samples** | ~185 PDAC |
| **Cancer type** | Pancreatic ductal adenocarcinoma |
| **Data types** | RNA-seq, WES, SNP array, methylation, clinical |
| **Key endpoint** | BRCA1/2/PALB2 mutation status; HRD score from SNP arrays |
| **Access** | Public (processed) / Controlled (raw) |
| **Download** | GDC Data Portal or cBioPortal |
| **Citation** | TCGA Research Network, Cancer Cell, 2017 |

**Why critical:** BRCA/PALB2 mutations in ~7-10% of PDAC. HRD score computable from
SNP arrays. Expression + mutation status available.

---

### 5.2 COMPASS Trial -- Real-Time Genomic Characterization of Advanced PDAC

| Field | Value |
|---|---|
| **Accession** | EGA (check ICGC PACA-CA project); cBioPortal may have processed data |
| **Samples** | ~195 metastatic PDAC |
| **Cancer type** | Advanced/metastatic PDAC |
| **Data types** | WGS, RNA-seq |
| **Key endpoint** | Real-time genomic characterization; BRCA/HRR status; treatment assignment |
| **Access** | Controlled (EGA/ICGC) |
| **Citation** | Aung KL et al., Cancer Discovery, 2018 |

**Why critical:** One of the largest WGS+RNA-seq PDAC cohorts. Contains patients with
BRCA/PALB2 mutations who went on to receive platinum or PARPi therapy.

---

### 5.3 pdacR Datasets (13 PDAC Expression Cohorts)

| Field | Value |
|---|---|
| **Accession** | Multiple GEO: GSE15471, GSE19650, GSE32676, GSE71989, GSE78229, GSE62452, GSE79670 and others |
| **Samples** | Combined >1,200 PDAC samples across 13 datasets |
| **Cancer type** | Pancreatic ductal adenocarcinoma |
| **Data types** | Gene expression microarray/RNA-seq |
| **Key endpoint** | Molecular subtypes (classical/basal-like); can be queried for HRR gene expression |
| **Access** | Public |
| **Download** | R package pdacR (github.com/rmoffitt/pdacR) or individual GEO downloads |
| **Citation** | Moffitt RA et al., Communications Biology, 2023 |

**Why useful:** Large curated collection of PDAC expression data. Not HRD-focused, but can
extract HRR gene expression patterns and overlay with known BRCA mutations from
TCGA-PAAD.

---

### 5.4 GSE300887 / GSE300888 -- BRCA2-Mutated Pancreatic Cancer

| Field | Value |
|---|---|
| **Accession** | [GSE300887](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE300887), [GSE300888](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE300888) |
| **Samples** | BRCA2-mutated pancreatic cancer models |
| **Cancer type** | Pancreatic cancer (BRCA2 mutant) |
| **Data types** | RNA-seq |
| **Key endpoint** | BRCA2-mutation effects; platinum/PARPi/immunotherapy sensitivity |
| **Access** | Public |
| **Download** | GEOquery |
| **Citation** | 2026 Scientific Reports study on BRCA2-mutated pancreatic cancer |

---

## PRIORITY 6: Additional Breast Cancer Cohorts with BRCA/HRD Status

### 6.1 GSE40115 -- Familial Breast Cancer with BRCA1/2 Mutations

| Field | Value |
|---|---|
| **Accession** | [GSE40115](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40115) |
| **Samples** | 183 (55 familial + 128 sporadic; 33 BRCA1-mutant, 22 BRCA2-mutant) |
| **Cancer type** | Breast cancer (familial and sporadic) |
| **Data types** | Gene expression microarray |
| **Key endpoint** | BRCA1/2 mutation status + BRCAness score |
| **Access** | Public |
| **Download** | GEOquery |
| **Citation** | Larsen MJ et al. |

**Why critical:** Known BRCA1/2 mutation status + expression. Can validate whether our HRD
signature correctly identifies BRCA-mutant tumors vs sporadic.

---

### 6.2 GSE27830 -- Familial Breast Cancer Panel

| Field | Value |
|---|---|
| **Accession** | [GSE27830](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE27830) |
| **Samples** | 155 familial breast cancers (47 BRCA1, 6 BRCA2, 26 CHEK2, 76 non-mutant) |
| **Cancer type** | Breast cancer (familial) |
| **Data types** | Gene expression microarray |
| **Key endpoint** | BRCA1/BRCA2/CHEK2 germline mutation status |
| **Access** | Public |
| **Download** | GEOquery |
| **Citation** | Waddell N et al. |

---

### 6.3 GSE19177 -- BRCA1/2 Familial Breast Cancer

| Field | Value |
|---|---|
| **Accession** | [GSE19177](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19177) |
| **Samples** | 74 (19 BRCA1, 30 BRCA2, 25 non-BRCA1/2 familial) |
| **Cancer type** | Breast cancer (familial) |
| **Data types** | Gene expression microarray |
| **Key endpoint** | BRCA1/BRCA2 mutation status |
| **Access** | Public |
| **Download** | GEOquery |

---

### 6.4 GSE50567 -- Hereditary Breast Cancer

| Field | Value |
|---|---|
| **Accession** | [GSE50567](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50567) |
| **Samples** | 41 (12 BRCA1-mutant, 1 BRCA2-mutant, 8 BRCAx, 14 sporadic, 6 normal) |
| **Cancer type** | Breast cancer (hereditary) |
| **Data types** | Gene expression microarray |
| **Key endpoint** | BRCA1/2 mutation status |
| **Access** | Public |
| **Download** | GEOquery |

---

### 6.5 METABRIC Expression Data (cBioPortal)

| Field | Value |
|---|---|
| **Accession** | cBioPortal: [brca_metabric](https://www.cbioportal.org/study?id=brca_metabric); EGA: EGAS00000000083 (raw data, controlled) |
| **Samples** | ~2,509 breast cancers |
| **Cancer type** | Breast cancer |
| **Data types** | Gene expression (Illumina HT12 microarray), copy number, mutations (Pereira 2016), clinical |
| **Key endpoint** | IntClust molecular subtypes, survival, BRCA1/2 mutation status |
| **Access** | Public (processed via cBioPortal/Kaggle); Controlled (raw via EGA DAC) |
| **Download** | cBioPortal bulk download: mRNA z-scores, mutations, clinical data. Or Kaggle: kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric |
| **Citation** | Curtis C et al., Nature, 2012; Pereira B et al., Nat Commun, 2016 |

**Action item:** We currently have METABRIC clinical data only. The expression data (mRNA z-scores)
is freely available via cBioPortal. Download and integrate immediately.

---

### 6.6 Lin et al. 2022 -- Pan-Gynaecological HRD Transcriptional Signature

| Field | Value |
|---|---|
| **Accession** | Trained on TCGA PanCanAtlas (BRCA/OV/UCEC); validated on CARPEM, CPTAC, SCAN cohorts |
| **Samples** | Training: 1,684; Validation: 4,038 |
| **Cancer type** | Breast, ovarian, endometrial |
| **Data types** | RNA-seq (656 genes commonly dysregulated with HRD) |
| **Key endpoint** | HRD genomic score prediction from expression (ridge regression model) |
| **Access** | Public (TCGA components); CPTAC data public via Proteomics Data Commons |
| **Citation** | Lin E et al., British Journal of Cancer, 2022 |

**Why useful:** Independent HRD-from-expression signature to benchmark against our classifier.
The 656-gene list and model coefficients may be in supplementary data.

---

## PRIORITY 7: Cell Line Datasets with PARPi Sensitivity

### 7.1 CTRP (Cancer Therapeutics Response Portal)

| Field | Value |
|---|---|
| **Accession** | [CTRP](https://portals.broadinstitute.org/ctrp/) |
| **Samples** | ~860 cell lines, 481 compounds |
| **Cancer type** | Pan-cancer cell lines |
| **Data types** | Drug sensitivity (AUC), paired with CCLE gene expression |
| **PARPi compounds** | Olaparib, talazoparib, veliparib, niraparib, rucaparib (total 14 PARPi assays across versions) |
| **Access** | Public |
| **Download** | portals.broadinstitute.org/ctrp or PharmacoDB (pharmacodb.ca) |
| **Citation** | Basu A et al., Cell, 2013; Seashore-Ludlow B et al., Cancer Discovery, 2015 |

**Why critical:** Complementary to GDSC. Different assay conditions may capture different
aspects of PARPi sensitivity. Link to CCLE expression for training.

---

### 7.2 PRISM Repurposing Dataset

| Field | Value |
|---|---|
| **Accession** | [DepMap/PRISM](https://depmap.org/repurposing/) |
| **Samples** | 578 cell lines, 4,518 compounds |
| **Cancer type** | Pan-cancer cell lines |
| **Data types** | Pooled cell viability (PRISM barcoding), paired with DepMap expression/mutation |
| **PARPi compounds** | Olaparib, talazoparib, niraparib, rucaparib, veliparib |
| **Access** | Public |
| **Download** | depmap.org/portal or DepMap API |
| **Citation** | Corsello SM et al., Nature Cancer, 2020 |

**Why critical:** Largest-scale drug sensitivity dataset. Barcoding approach is different from
GDSC/CTRP (viability assay vs dose-response). Provides third independent source
for PARPi sensitivity profiling.

---

### 7.3 gCSI (Genentech Cell Line Screening Initiative)

| Field | Value |
|---|---|
| **Accession** | Available via PharmacoDB (pharmacodb.ca) |
| **Samples** | ~357 cell lines |
| **Cancer type** | Pan-cancer cell lines |
| **Data types** | Drug sensitivity + gene expression |
| **PARPi compounds** | Includes olaparib and other DNA-damaging agents |
| **Access** | Public |
| **Download** | PharmacoDB or direct from Genentech publication supplements |
| **Citation** | Haverty PM et al., Genome Biology, 2016 |

---

### 7.4 Pan-Cancer HRD in Cell Lines (2024)

| Field | Value |
|---|---|
| **Accession** | Analysis used CCLE/DepMap/GDSC/CTRP/PRISM data |
| **Samples** | Pan-cancer cell lines |
| **Data types** | HRD scores computed for cell lines + drug sensitivity |
| **Key finding** | HRD status in cell lines was NOT strongly correlated with PARPi/platinum sensitivity |
| **Access** | Public (underlying data from DepMap) |
| **Citation** | Cancer Research Communications, 2024 |

**Important caveat:** This study found that HRD (genomic scar) scores in cell lines do not
predict PARPi response well. This is consistent with the idea that functional HRD
(expression-based) may be more predictive than genomic scars in cell lines.

---

### 7.5 GSE55830 -- Veliparib in SCLC Cell Lines

| Field | Value |
|---|---|
| **Accession** | [GSE55830](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE55830) |
| **Samples** | 9 SCLC cell lines |
| **Cancer type** | Small cell lung cancer |
| **Data types** | Gene expression, veliparib cytotoxicity data |
| **Key endpoint** | PARPi sensitivity (linked to SLFN11 expression) |
| **Access** | Public |
| **Download** | GEOquery |

---

## ADDITIONAL RESOURCES AND INTEGRATIVE PLATFORMS

### PharmacoDB

**URL:** https://pharmacodb.pmgenomics.ca/
Integrates GDSC, CCLE, CTRP, gCSI, and PRISM into a single queryable platform.
Can extract PARPi sensitivity data across all platforms with harmonized gene expression.

### DepMap Portal

**URL:** https://depmap.org/portal/
Provides CRISPR/RNAi essentiality screens, drug sensitivity (PRISM), copy number,
expression, and mutations for >1,800 cell lines. Can identify cell lines with HRR gene
knockouts or mutations and test gene essentiality patterns.

### cBioPortal

**URL:** https://www.cbioportal.org/
Many of the datasets above (SU2C, TCGA, METABRIC) are queryable here. Can filter
for specific HRR gene alterations across all datasets simultaneously using the
"Comparison" feature.

### ICGC/PCAWG

**URL:** https://dcc.icgc.org/
Contains the PCAWG dataset we already have, plus individual project data from ICGC
(PACA-AU for pancreatic, OV-AU for ovarian, BRCA-UK/EU for breast) with WGS and
often RNA-seq.

---

## DOWNLOAD PRIORITY AND ACTION ITEMS

### Immediate (Public, direct download):

1. **METABRIC expression** from cBioPortal (we have clinical only -- add expression now)
2. **GSE194040** (I-SPY2-990, all arms including veliparib/carboplatin)
3. **GSE113863** (talazoparib neoadjuvant, 13 samples)
4. **GSE40115** (183 breast, BRCA1/2 status labeled)
5. **GSE51373** (28 OvCa, platinum response)
6. **GSE23554** (28 OvCa, chemo response)
7. **GSE3149** (~153 OvCa, platinum/taxane response)
8. **SU2C prad_su2c_2015** and **prad_su2c_2019** from cBioPortal (prostate mCRPC)
9. **TCGA-PRAD** and **TCGA-PAAD** from GDC (prostate + pancreatic)
10. **CTRP** and **PRISM** PARPi sensitivity data

### Short-term (Controlled access, apply for):

1. **Patch et al. EGAD00001000877** (92 OvCa, WGS+RNA-seq, platinum resistance, BRCA reversions)
2. **Quigley mCRPC phs001648** (101+ mCRPC, WGS+RNA-seq via dbGaP or AWS Open Data)
3. **COMPASS PDAC** (195 PDAC, WGS+RNA-seq via ICGC/EGA)

### Contact authors for:

1. **TOPARP-A/B** RNA-seq data (ICR, contact Mateo/de Bono)
2. **Meijer/Naipal RECAP** data with any paired expression (Erasmus MC)
3. **ARIEL2** rucaparib RNA-seq data (Clovis/Foundation Medicine)

---

## SUMMARY TABLE

| Dataset | N | Cancer | Expression | Response Data | Access | Priority |
|---|---|---|---|---|---|---|
| GSE113863 (talazoparib) | 13 | Breast | RNA-seq | RCB to PARPi | Public | 1 |
| I-SPY2-990 (GSE194040) | ~987 | Breast | Microarray | pCR (incl PARPi arm) | Public | 1 |
| GSE51373 | 28 | Ovarian | Microarray | Platinum sens/res | Public | 1 |
| GSE23554 | 28 | Ovarian | Microarray | Chemo response | Public | 1 |
| GSE3149 | ~153 | Ovarian | Microarray | Platinum/taxane | Public | 1 |
| Patch 2015 (EGAD877) | 92 | Ovarian | WGS+RNA | Platinum + reversions | Controlled | 1/3 |
| GSE54266 (BRCA1/RAD51 KD) | CL | Breast | Microarray | Knockdown profiles | Public | 2 |
| Quigley (phs001648) | 101+ | Prostate | WGS+RNA | Resistance + reversions | Controlled | 3/4 |
| SU2C 2015 (cBioPortal) | 150 | Prostate | WES+RNA | HRR mutations | Public | 4 |
| SU2C 2019 (cBioPortal) | 429 | Prostate | WES+RNA | HRR mutations | Public | 4 |
| TCGA-PRAD | ~500 | Prostate | RNA-seq | HRD score | Public | 4 |
| GSE35988 | 122 | Prostate | Microarray | PCa progression | Public | 4 |
| TCGA-PAAD | ~185 | Pancreas | RNA-seq | BRCA/PALB2 status | Public | 5 |
| COMPASS | ~195 | Pancreas | WGS+RNA | Treatment outcomes | Controlled | 5 |
| GSE300887/300888 | var | Pancreas | RNA-seq | BRCA2 effects | Public | 5 |
| GSE40115 | 183 | Breast | Microarray | BRCA1/2 status | Public | 6 |
| GSE27830 | 155 | Breast | Microarray | BRCA1/2/CHEK2 | Public | 6 |
| GSE19177 | 74 | Breast | Microarray | BRCA1/2 status | Public | 6 |
| METABRIC (cBioPortal) | 2509 | Breast | Microarray | Subtypes + survival | Public | 6 |
| CTRP | ~860 CL | Pan-cancer | CCLE expr | PARPi sensitivity | Public | 7 |
| PRISM | ~578 CL | Pan-cancer | DepMap expr | PARPi sensitivity | Public | 7 |
| gCSI | ~357 CL | Pan-cancer | Expr | Drug sensitivity | Public | 7 |
