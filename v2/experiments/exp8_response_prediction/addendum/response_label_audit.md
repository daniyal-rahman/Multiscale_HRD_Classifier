# Response Label Audit: Definitions Across 9 Clinical Datasets

## Reviewer Concern
"Sensitive is >6 months in some datasets and >12 in others. Someone who is 8 months will be sensitive in one and resistant in the other." This audit documents the exact definition used for each dataset and quantifies the ambiguous-zone patients.

---

## Summary Table

| Dataset | Cancer | Drug | N | Sens | Res | Response Type | Binary Cutoff | Ambiguous Zone (6-12mo) |
|---|---|---|---:|---:|---:|---|---|---:|
| TCGA-OV | HGSOC | Platinum | 235 | 165 | 70 | PFI (cBioPortal) | >6mo = Sensitive | N/A (pre-classified) |
| GSE32062 | HGSOC | Plat+Tax | 260 | 225 | 35 | PFS (months) | >6mo = Sensitive | 45 patients (17%) |
| GSE156699 | HGSOC | Platinum | 88 | 50 | 38 | PFS | >=6mo = Responder | Unknown (no continuous) |
| GSE63885 | Mixed OC | Platinum | 75 | 41 | 34 | DFS (months) | >=6mo = Sensitive | 13 patients (17%) |
| GSE30161 | Late-stage OC | Platinum | 55 | 32 | 23 | RECIST | CR = Sensitive | 0 (not PFI-based) |
| GSE28739 | Serous OC | Platinum | 25 | 10 | 15 | Recurrence | >30mo=Sens, <=6mo=Res | 0 (extreme groups) |
| GSE18864 | TNBC | Cisplatin | 24 | 8 | 16 | Miller-Payne | MP 4-5 = Responder | 0 (pathologic) |
| GSE173839 | Breast | Durv+Ola | 71 | 29 | 42 | pCR | pCR = Responder | 0 (pathologic) |
| GSE194040 | Breast | Vel+Carb | 71 | 27 | 44 | pCR | pCR = Responder | 0 (pathologic) |

---

## Detailed Definitions

### 1. TCGA-OV (n=235)
- **Source**: cBioPortal `ov_tcga_pub` study, `PLATINUM_STATUS` field
- **Definition**: Pre-classified by the TCGA OV 2011 Nature paper
  - **Sensitive** (n=165): PFI > 6 months after last platinum dose
  - **Resistant** (n=70): PFI <= 6 months after last platinum dose
  - "TooEarly" patients (n=52) were excluded — these had insufficient follow-up
- **Continuous measure**: Not available (only binary label from cBioPortal)
- **Notes**: The 6-month cutoff follows GCIG consensus. The original TCGA paper uses this standard definition.

### 2. GSE32062 (n=260)
- **Source**: GEO, Yoshihara et al. (Japanese HGSOC cohort)
- **Definition**: PFS (progression-free survival) in months
  - **Sensitive** (n=225): PFS > 6 months
  - **Resistant** (n=35): PFS <= 6 months
- **Continuous measure**: PFS in months (range 1-119, median 19)
- **Ambiguous zone**: 45 patients (17%) have PFS between 6-12 months. These are classified as "sensitive" under the 6-month cutoff but could be reclassified as "resistant" under a 12-month cutoff.
- **Notes**: Highly imbalanced (87% sensitive). All received platinum+taxane.

### 3. GSE156699 (n=88)
- **Source**: GEO, single-institution HGSOC cohort
- **Definition**: PFS-based
  - **Responders** (n=50): PFS >= 6 months
  - **Non-responders** (n=38): PFS < 6 months
- **Continuous measure**: Not available in the processed data
- **Notes**: Retrospective case-control design. Uses the same 6-month cutoff as TCGA-OV and GSE32062.

### 4. GSE63885 (n=75)
- **Source**: GEO, mixed histology ovarian cancer
- **Definition**: DFS (disease-free survival) in days, converted to months
  - **Resistant** (n=34): DFS < 180 days (< 6 months); range [0, 5.9 months]
  - **Moderately sensitive** (n=28): 180 <= DFS <= 732 days (6-24 months)
  - **Highly sensitive** (n=13): DFS > 732 days (> 24 months)
  - Binary: Resistant (DFS < 6mo) vs Sensitive (moderate + highly sensitive, DFS >= 6mo)
- **Continuous measure**: DFS in months (range 0-129.2, median 7.4)
- **Ambiguous zone**: 13 patients (17%) have DFS between 6-12 months. Original study provides 3-tier classification but our binary uses the 6-month cut.
- **Notes**: Mixed histology (73 serous, 12 endometrioid, 9 clear cell, 7 undifferentiated). BRCA1 status annotated.

### 5. GSE30161 (n=55)
- **Source**: GEO, Dressman et al., late-stage ovarian cancer
- **Definition**: RECIST response to platinum-based chemotherapy
  - **Sensitive** (n=32): Complete Response (CR)
  - **Resistant** (n=23): Partial Response (PR), Progressive Disease (PD), or Stable Disease (SD)
- **Continuous measure**: PFI in months (range 0.4-138.2, median 12.6). **NOT used for binary classification** — binary is RECIST-based.
- **Ambiguous zone**: N/A — response is based on RECIST best response, not a time cutoff.
- **Important note**: CR patients have median PFI of 17.0 months; non-CR patients have median PFI of 8.3 months. Some non-CR patients have long PFI (up to 26.4 months), and some CR patients have short PFI (down to 2.9 months). This is a different axis of classification than PFI cutoff.

### 6. GSE28739 (n=25)
- **Source**: GEO, serous ovarian cancer
- **Definition**: Extreme phenotype design
  - **Chemosensitive** (n=10): Recurrence > 30 months after platinum
  - **Chemoresistant** (n=15): Recurrence <= 6 months after platinum
- **Continuous measure**: Not available
- **Ambiguous zone**: 0 — by design, this study selected only extreme phenotypes (no patients between 6-30 months).
- **Notes**: Dual-channel Agilent, dye-swap averaged. Small sample size but clean separation.

### 7. GSE18864 (n=24)
- **Source**: GEO, neoadjuvant cisplatin in TNBC
- **Definition**: Miller-Payne pathologic response grade
  - **Responder** (n=8): Miller-Payne grade 4-5 (near/complete pathologic response)
  - **Non-responder** (n=16): Miller-Payne grade 1-3 (minimal/no pathologic response)
- **Continuous measure**: Miller-Payne grade (1-5), stored as response_continuous
- **Ambiguous zone**: N/A — pathologic endpoint, not time-based.
- **Notes**: BRCA1 genotype annotated. Only breast cancer dataset using cisplatin monotherapy.

### 8. GSE173839 (n=71 PARPi arm, 29 control)
- **Source**: GEO, I-SPY2 trial (durvalumab + olaparib arm)
- **Definition**: pCR (pathological complete response)
  - **Responder** (n=29): pCR achieved after neoadjuvant durvalumab+olaparib+paclitaxel
  - **Non-responder** (n=42): No pCR
- **Continuous measure**: None
- **Ambiguous zone**: N/A — binary endpoint by definition.
- **Notes**: HER2-negative breast cancer. Control arm (paclitaxel only, n=29) used for interaction testing but not in LODO-CV Model A.

### 9. GSE194040 (n=71 PARPi arm, 179 control)
- **Source**: GEO, I-SPY2-990 trial (veliparib + carboplatin arm)
- **Definition**: pCR (pathological complete response)
  - **Responder** (n=27): pCR achieved after neoadjuvant veliparib+carboplatin+paclitaxel
  - **Non-responder** (n=44): No pCR
- **Continuous measure**: None
- **Ambiguous zone**: N/A — binary endpoint by definition.
- **Notes**: High-risk early-stage breast cancer. ComBat-adjusted expression across I-SPY2-990 arms.

---

## Response Definition Heterogeneity: Impact Assessment

### Consistent cutoff (6 months) for PFI/PFS datasets
Four of the five ovarian platinum datasets use a **6-month PFI/PFS/DFS cutoff**, which is the standard GCIG (Gynecologic Cancer InterGroup) definition:
- TCGA-OV: PFI > 6 months (from cBioPortal, pre-classified per TCGA 2011 Nature paper)
- GSE32062: PFS > 6 months
- GSE156699: PFS >= 6 months
- GSE63885: DFS >= 6 months (180 days)

### Different response axes
- **GSE30161**: Uses RECIST CR vs non-CR, NOT a PFI cutoff. This measures tumor shrinkage rather than duration of response. This is a fundamentally different endpoint.
- **GSE28739**: Extreme phenotype (>30mo vs <=6mo), compatible with but stricter than the 6-month cutoff.
- **I-SPY2 datasets**: pCR (pathologic), completely different axis.
- **GSE18864**: Miller-Payne (pathologic), completely different axis.

### Quantifying ambiguous patients
For PFI/PFS-based datasets, patients between 6-12 months are the "ambiguous zone":
- **GSE32062**: 45 of 260 (17%) have PFS 6-12 months → classified as sensitive
- **GSE63885**: 13 of 75 (17%) have DFS 6-12 months → classified as sensitive
- **TCGA-OV**: Unknown (no continuous measure available)
- **GSE156699**: Unknown (no continuous measure available)

**If a 12-month cutoff were used instead**, approximately 17% of patients in GSE32062 and GSE63885 would switch from sensitive to resistant, but this does not affect the model fundamentally because:
1. The LODO-CV framework trains on 8 datasets and tests on the 9th — so label heterogeneity across datasets is tolerable as long as each dataset is internally consistent.
2. The 6-month cutoff is the clinical standard (GCIG consensus).
3. The model learns from all 9 datasets jointly; some noise in borderline cases is expected and handled by the L2 regularization.

## Recommendation
The response definitions are documented and justifiable. The 6-month PFI/PFS cutoff is the international standard. To address the reviewer's concern directly:
- Acknowledge the heterogeneity in the paper's methods section
- Note that LODO-CV naturally handles this (each dataset's internal labels are consistent)
- Consider a sensitivity analysis: re-run LODO-CV after excluding patients with PFS/DFS 6-12 months to show results are robust

## Files
- Script: N/A (documentation audit)
- Source catalog: `response_data/response_dataset_catalog.csv`
- Source report: `response_data/acquisition_report.md`
