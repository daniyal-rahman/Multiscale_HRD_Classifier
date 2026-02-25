#!/usr/bin/env python3
"""
Gene Overlap Analysis: Response Model vs Published Signatures

Compares our Exp8 drug response classifier's top genes against:
1. Our 11 published HRD signature gene lists
2. Ayers/NanoString Tumor Inflammation Signature (TIS-18)
3. Ayers IFN-gamma 10-gene signature
4. Konstantinopoulos BRCAness 60-gene signature (already in our collection)
5. Published platinum-response gene signatures
"""

import csv
import json
import sys
from collections import defaultdict
from scipy.stats import hypergeom

# --- Load our response model genes ---
response_genes = []
response_weights = {}
with open("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction/exp8b_gene_weights_ranked.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        gene = row["gene"]
        response_genes.append(gene)
        response_weights[gene] = {
            "abs_weight": float(row["abs_weight"]),
            "weight": float(row["weight"]),
            "rank": int(row["rank"])
        }

print(f"Total response model genes: {len(response_genes)}")
top50 = set(response_genes[:50])
top100 = set(response_genes[:100])
top200 = set(response_genes[:200])
top500 = set(response_genes[:500])
all_genes = set(response_genes)

# --- Load HRD signature gene lists ---
hrd_signatures = defaultdict(set)
with open("/home/dani/repos2/Multiscale_HRD_Classifier/v2/signature_analysis/all_signature_gene_lists.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        hrd_signatures[row["Signature"]].add(row["Gene"])

print(f"\nHRD signatures loaded: {list(hrd_signatures.keys())}")
for sig, genes in hrd_signatures.items():
    print(f"  {sig}: {len(genes)} genes")

# --- Define published immune/response signatures ---

# TIS-18: Ayers et al. / Danaher et al. 2018 (JITC)
tis18 = {
    "CCL5", "CD27", "CD274", "CD276", "CD8A", "CMKLR1", "CXCL9", "CXCR6",
    "HLA-DQA1", "HLA-DRB1", "HLA-E", "IDO1", "LAG3", "NKG7", "PDCD1LG2",
    "PSMB10", "STAT1", "TIGIT"
}

# IFN-gamma 10-gene signature: Ayers et al. 2017 (JCI)
ifng_10 = {
    "IFNG", "STAT1", "CCR5", "CXCL9", "CXCL10", "CXCL11", "IDO1",
    "PRF1", "GZMA", "HLA-DRA"
}

# Expanded immune signature (28 genes refined to 18 = TIS, but original 28):
# From Ayers 2017 JCI supplemental - preliminary expanded immune
expanded_immune_28 = tis18 | {
    "CXCL10", "CXCL11", "GZMA", "GZMB", "PRF1", "IFNG", "HLA-DRA",
    "CCR5", "CD3D", "CD3E"
}

# Matondo 2017 - 97 chemoresponse signature (partial list from paper + HIF1a)
# Known genes from the paper
matondo97_partial = {
    "ACTA2", "EPOR", "CTGF", "FOS", "SCG2", "IGF1", "CYR61", "NT5E", "NR4A1",
    "COL3A1", "COL10A1", "LAMB1", "INHBA", "F3", "PDGFRA", "PDGFRB",
    "PLAU", "ROR2", "ZFHX4", "COL11A1", "COL5A2", "THBS2", "TIMP3", "EGR2"
}

# Lin 2025 - 9-gene platinum-resistance signature
lin9_plat = {
    "SLC22A2", "TAP1", "PC", "MCM3", "GTF2H2", "FXYD5", "SUPT6H", "IGKC", "MATN2"
}

# Compile all external signatures
external_signatures = {
    "TIS-18 (Ayers/Danaher 2018)": tis18,
    "IFN-gamma 10 (Ayers 2017)": ifng_10,
    "Expanded Immune 28 (Ayers 2017)": expanded_immune_28,
    "Plat-Resist 9 (Lin 2025)": lin9_plat,
    "Chemoresponse 97 (Matondo 2017, partial)": matondo97_partial,
}

# --- Load BRCA comparison data ---
with open("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction/brca_comparison/brca_vs_response_results.json") as f:
    brca_data = json.load(f)

# --- Compute overlaps ---
# Background universe: all genes in the model
N_UNIVERSE = len(all_genes)  # 11,089

def compute_overlap(gene_set, signature_set):
    overlap = gene_set & signature_set
    return overlap, len(overlap), len(signature_set)

def hypergeom_pvalue(k, K, n, N):
    """
    Hypergeometric test for overlap enrichment.
    k = overlap count, K = signature size (successes in population),
    n = query set size (draws), N = universe size (population).
    Returns p-value = P(X >= k).
    """
    if k == 0:
        return 1.0
    # sf(k-1) = P(X >= k)
    return float(hypergeom.sf(k - 1, N, K, n))

print("\n" + "="*80)
print("OVERLAP ANALYSIS: Response Model Top Genes vs Published Signatures")
print("="*80)

# Table format results
results = []

# 1. HRD signatures
print(f"\n--- vs HRD Signatures (from our collection) --- [universe N={N_UNIVERSE}]")
for sig_name, sig_genes in sorted(hrd_signatures.items()):
    sig_in_univ = len(sig_genes & all_genes)  # K: signature genes present in universe
    ol50, n50, nsig = compute_overlap(top50, sig_genes)
    ol100, n100, _ = compute_overlap(top100, sig_genes)
    ol200, n200, _ = compute_overlap(top200, sig_genes)
    ol500, n500, _ = compute_overlap(top500, sig_genes)
    ol_all, n_all, _ = compute_overlap(all_genes, sig_genes)

    # Hypergeometric p-values (K = sig genes in universe, not total sig size)
    p50 = hypergeom_pvalue(n50, sig_in_univ, 50, N_UNIVERSE)
    p100 = hypergeom_pvalue(n100, sig_in_univ, 100, N_UNIVERSE)
    p200 = hypergeom_pvalue(n200, sig_in_univ, 200, N_UNIVERSE)
    p500 = hypergeom_pvalue(n500, sig_in_univ, 500, N_UNIVERSE)

    results.append({
        "signature": sig_name,
        "sig_size": nsig,
        "sig_in_universe": sig_in_univ,
        "top50": n50,
        "top100": n100,
        "top200": n200,
        "top500": n500,
        "all": n_all,
        "top50_genes": sorted(ol50),
        "top100_genes": sorted(ol100),
        "top200_genes": sorted(ol200),
        "p_top50": p50,
        "p_top100": p100,
        "p_top200": p200,
        "p_top500": p500,
        "category": "HRD"
    })
    sig_marker = " ***" if p100 < 0.001 else " **" if p100 < 0.01 else " *" if p100 < 0.05 else ""
    print(f"  {sig_name} ({nsig} genes, {sig_in_univ} in univ): top100={n100} (p={p100:.2e}){sig_marker}, top500={n500} (p={p500:.2e})")
    if ol100:
        print(f"    Top-100 overlapping: {sorted(ol100)}")

# 2. External signatures
print(f"\n--- vs Published Immune/Response Signatures --- [universe N={N_UNIVERSE}]")
for sig_name, sig_genes in external_signatures.items():
    sig_in_univ = len(sig_genes & all_genes)
    ol50, n50, nsig = compute_overlap(top50, sig_genes)
    ol100, n100, _ = compute_overlap(top100, sig_genes)
    ol200, n200, _ = compute_overlap(top200, sig_genes)
    ol500, n500, _ = compute_overlap(top500, sig_genes)
    ol_all, n_all, _ = compute_overlap(all_genes, sig_genes)

    p50 = hypergeom_pvalue(n50, sig_in_univ, 50, N_UNIVERSE)
    p100 = hypergeom_pvalue(n100, sig_in_univ, 100, N_UNIVERSE)
    p200 = hypergeom_pvalue(n200, sig_in_univ, 200, N_UNIVERSE)
    p500 = hypergeom_pvalue(n500, sig_in_univ, 500, N_UNIVERSE)

    results.append({
        "signature": sig_name,
        "sig_size": nsig,
        "sig_in_universe": sig_in_univ,
        "top50": n50,
        "top100": n100,
        "top200": n200,
        "top500": n500,
        "all": n_all,
        "top50_genes": sorted(ol50),
        "top100_genes": sorted(ol100),
        "top200_genes": sorted(ol200),
        "p_top50": p50,
        "p_top100": p100,
        "p_top200": p200,
        "p_top500": p500,
        "category": "Immune/Response"
    })
    sig_marker = " ***" if p100 < 0.001 else " **" if p100 < 0.01 else " *" if p100 < 0.05 else ""
    print(f"  {sig_name} ({nsig} genes, {sig_in_univ} in univ): top100={n100} (p={p100:.2e}){sig_marker}, top500={n500} (p={p500:.2e})")
    if ol200:
        print(f"    Top-200 overlapping: {sorted(ol200)}")

# --- Functional annotation of top genes ---
print("\n" + "="*80)
print("IMMUNE/INFLAMMATORY GENES IN RESPONSE MODEL")
print("="*80)

# Curated list of known immune/inflammatory gene categories
immune_genes = {
    "HLA/Antigen presentation": {"HLA-DQA1", "HLA-DQB1", "HLA-DRB1", "HLA-DRA", "HLA-A", "HLA-B", "HLA-E", "HLA-C", "HLA-F", "HLA-G", "PSMB10", "TAP1", "TAP2", "B2M"},
    "Chemokines/Cytokines": {"CXCL9", "CXCL10", "CXCL11", "CXCL13", "CCL5", "CCL4", "CCL2", "CCL19", "CCL21", "IFNG", "TNF", "IL2", "IL12A", "IL12B", "IL15", "IL18", "IL21"},
    "T-cell markers": {"CD8A", "CD8B", "CD3D", "CD3E", "CD3G", "CD4", "CD27", "CD28", "CD52", "ICOS"},
    "Immune checkpoints": {"CD274", "PDCD1", "PDCD1LG2", "CTLA4", "LAG3", "TIGIT", "HAVCR2", "IDO1"},
    "IFN-gamma signaling": {"STAT1", "IRF1", "IRF7", "IRF8", "IRF9", "GBP1", "GBP2", "GBP3", "GBP4", "GBP5"},
    "Cytotoxic effectors": {"GZMA", "GZMB", "GZMH", "GZMK", "PRF1", "NKG7", "GNLY", "FASLG"},
    "NK cell markers": {"KLRK1", "KLRD1", "KLRB1", "NKG7", "NCR1", "NCR3"},
    "Complement/Innate": {"C1QA", "C1QB", "C1QC", "C3", "C7"},
    "Metallothioneins (stress)": {"MT1E", "MT1F", "MT1G", "MT1H", "MT1M", "MT1X", "MT2A"},
    "Immune/inflammatory other": {"BST1", "BOK", "BCL2", "TP53", "PRKCA"}
}

print("\nImmune/inflammatory genes found in response model top tiers:")
for category, gene_set in immune_genes.items():
    in_top50 = sorted(top50 & gene_set)
    in_top100 = sorted(top100 & gene_set)
    in_top200 = sorted(top200 & gene_set)
    in_top500 = sorted(top500 & gene_set)

    if in_top500:
        print(f"\n  {category}:")
        for g in in_top500:
            rank = response_weights[g]["rank"]
            weight = response_weights[g]["weight"]
            direction = "+" if weight > 0 else "-"
            tier = "top50" if g in top50 else "top100" if g in top100 else "top200" if g in top200 else "top500"
            print(f"    {g:15s} rank={rank:4d}  weight={weight:+.3f}  [{tier}]")

# --- Count immune genes at each tier ---
all_immune = set()
for gene_set in immune_genes.values():
    all_immune |= gene_set

n_immune_50 = len(top50 & all_immune)
n_immune_100 = len(top100 & all_immune)
n_immune_200 = len(top200 & all_immune)
n_immune_500 = len(top500 & all_immune)

print(f"\n  Immune gene counts: top50={n_immune_50}/50, top100={n_immune_100}/100, "
      f"top200={n_immune_200}/200, top500={n_immune_500}/500")

# --- Analyze weight direction patterns ---
print("\n" + "="*80)
print("WEIGHT DIRECTION ANALYSIS")
print("="*80)

for tier_name, tier_set in [("Top 50", top50), ("Top 100", top100), ("Top 200", top200)]:
    pos = sum(1 for g in tier_set if response_weights[g]["weight"] > 0)
    neg = len(tier_set) - pos
    print(f"  {tier_name}: {pos} positive (predict response), {neg} negative (predict resistance)")

# Immune genes direction
print("\n  Direction of immune genes in top 200:")
immune_in_200 = top200 & all_immune
for g in sorted(immune_in_200, key=lambda x: response_weights[x]["rank"]):
    w = response_weights[g]["weight"]
    direction = "RESPONSE" if w > 0 else "RESISTANCE"
    print(f"    {g:15s} weight={w:+.4f} -> predicts {direction}")

# --- Write JSON summary ---
summary = {
    "response_model_total_genes": len(response_genes),
    "overlap_with_hrd_signatures": {},
    "overlap_with_immune_signatures": {},
    "immune_gene_enrichment": {
        "top50": n_immune_50,
        "top100": n_immune_100,
        "top200": n_immune_200,
        "top500": n_immune_500,
    },
    "brca_comparison": brca_data["feature_weight_comparison"],
}

for r in results:
    target = summary["overlap_with_hrd_signatures"] if r["category"] == "HRD" else summary["overlap_with_immune_signatures"]
    target[r["signature"]] = {
        "signature_size": r["sig_size"],
        "sig_genes_in_universe": r.get("sig_in_universe", r["sig_size"]),
        "overlap_top50": r["top50"],
        "overlap_top100": r["top100"],
        "overlap_top200": r["top200"],
        "overlap_top500": r["top500"],
        "overlap_all": r["all"],
        "top100_overlapping_genes": r["top100_genes"],
        "hypergeom_p_top50": r["p_top50"],
        "hypergeom_p_top100": r["p_top100"],
        "hypergeom_p_top200": r["p_top200"],
        "hypergeom_p_top500": r["p_top500"],
    }

with open("/home/dani/repos2/Multiscale_HRD_Classifier/v2/experiments/exp8_response_prediction/addendum/gene_overlap_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n\nResults written to gene_overlap_results.json")
