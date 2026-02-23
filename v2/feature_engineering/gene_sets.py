"""
Curated gene sets for HRD-relevant pathway scoring.

Gene sets sourced from MSigDB (Hallmark, C2/C5), KEGG, Reactome, and
HRD-specific literature. These are used by PathwayScorer and PROGENyScorer.
"""

# -- Homologous recombination repair pathway ----------------------------------
HR_REPAIR = [
    "BRCA1", "BRCA2", "PALB2", "RAD51", "RAD51B", "RAD51C", "RAD51D",
    "XRCC2", "XRCC3", "DMC1", "MRE11", "RAD50", "NBN",  # MRN complex
    "ATM", "ATR", "CHEK1", "CHEK2", "BARD1", "BRIP1",
    "RBBP8",  # CtIP
    "BLM", "EXO1", "DNA2", "RPA1", "RPA2", "RPA3",
    "FANCA", "FANCB", "FANCC", "FANCD2", "FANCE", "FANCF", "FANCG",
    "FANCI", "FANCL", "FANCM",
    "CDK12",
]

# -- Non-homologous end joining -----------------------------------------------
NHEJ = [
    "XRCC4", "XRCC5", "XRCC6",  # Ku70/Ku80
    "PRKDC",  # DNA-PKcs
    "LIG4", "NHEJ1",  # XLF
    "DCLRE1C",  # Artemis
    "POLL", "POLM",  # Polymerases
]

# -- Alternative end joining / microhomology-mediated -------------------------
ALT_EJ = [
    "POLQ",  # Pol theta — key alt-EJ gene, often upregulated in HRD
    "PARP1", "LIG1", "LIG3", "XRCC1", "FEN1",
]

# -- Fanconi anemia pathway (overlaps with HR) --------------------------------
FANCONI_ANEMIA = [
    "FANCA", "FANCB", "FANCC", "FANCD2", "FANCE", "FANCF", "FANCG",
    "FANCI", "FANCL", "FANCM", "BRCA1", "BRCA2", "PALB2", "RAD51C",
    "BRIP1", "SLX4", "ERCC4", "MAD2L2", "RFWD3", "UBE2T",
]

# -- DNA damage response (broad) ---------------------------------------------
DDR_BROAD = [
    "ATM", "ATR", "CHEK1", "CHEK2", "TP53", "MDM2",
    "CDKN1A",  # p21
    "CDKN2A",  # p16
    "RB1", "E2F1",
    "H2AX",  # gamma-H2AX proxy
    "RNF8", "RNF168", "53BP1", "TP53BP1",
    "PARP1", "PARP2",
    "RAD18", "REV1", "REV3L",
]

# -- Cell cycle / proliferation (Hallmark-derived) ----------------------------
CELL_CYCLE = [
    "CDK1", "CDK2", "CDK4", "CDK6",
    "CCNA2", "CCNB1", "CCNB2", "CCND1", "CCNE1", "CCNE2",
    "MKI67", "PCNA", "TOP2A",
    "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
    "CDC6", "CDC20", "CDC25A", "CDC25C", "CDC45",
    "AURKB", "PLK1", "BUB1", "BUB1B", "MAD2L1",
    "E2F1", "E2F2", "RB1", "CDKN1A", "CDKN2A",
]

# -- Immune / tumour microenvironment ----------------------------------------
IMMUNE_INFILTRATION = [
    # T-cell markers
    "CD3D", "CD3E", "CD4", "CD8A", "CD8B",
    # Cytotoxic
    "GZMA", "GZMB", "GZMK", "PRF1", "GNLY", "NKG7", "IFNG",
    # Immune checkpoints
    "PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT",
    # Antigen presentation
    "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2",
    # Inflammatory
    "CXCL9", "CXCL10", "CXCL11", "CCL5", "STAT1",
    # cGAS-STING (relevant to HRD → immune activation)
    "CGAS", "TMEM173", "TBK1", "IRF3",
    # Macrophage
    "CD68", "CD163",
]

# -- Replication stress -------------------------------------------------------
REPLICATION_STRESS = [
    "RPA1", "RPA2", "ATRIP", "ATR", "CHEK1",
    "CLSPN",  # Claspin
    "TIMELESS", "TIPIN",
    "WRN", "BLM", "RECQL4",
    "FANCD2", "FANCI",
    "PCNA", "RFC1",
    "RAD17", "RAD9A", "HUS1",
]

# -- Chromatin remodeling (relevant to BRCA1 function) ------------------------
CHROMATIN_REMODELING = [
    "SMARCA4", "SMARCB1", "ARID1A", "ARID1B",
    "KDM5A", "KDM5B",  # BRCA1/BARD1-associated H3K4 demethylases
    "SETD2", "KMT2C", "KMT2D",
    "TET2", "DNMT3A", "DNMT3B",
    "EZH2",  # PRC2 — synthetic lethal with BRCA1
]

# -- Estrogen response (relevant for ER status context) -----------------------
ESTROGEN_RESPONSE = [
    "ESR1", "PGR", "GATA3", "FOXA1", "TFF1", "TFF3",
    "XBP1", "AGR2", "CA12", "SLC39A6",
]

# -- Master dictionary for all gene sets --------------------------------------
GENE_SETS = {
    "HR_repair": HR_REPAIR,
    "NHEJ": NHEJ,
    "alt_EJ": ALT_EJ,
    "fanconi_anemia": FANCONI_ANEMIA,
    "DDR_broad": DDR_BROAD,
    "cell_cycle": CELL_CYCLE,
    "immune_infiltration": IMMUNE_INFILTRATION,
    "replication_stress": REPLICATION_STRESS,
    "chromatin_remodeling": CHROMATIN_REMODELING,
    "estrogen_response": ESTROGEN_RESPONSE,
}
