"""
Reversion analysis — the killer validation for transcriptomic HRD signatures.

Key hypothesis: Scar-based methods (GIS, MyChoice) remain elevated after
BRCA reversion mutations restore HR function, but a *transcriptomic*
signature should drop because gene expression reflects the CURRENT
functional state, not accumulated genomic damage.

This module provides:
  1. Framework for analyzing known reversion cases
  2. Simulated reversion analysis using BRCA-wildtype as proxy
  3. Comparison of scar-based vs expression-based predictions
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .metrics import ValidationMetrics

logger = logging.getLogger(__name__)


class ReversionAnalysis:
    """Analyze whether the transcriptomic signature captures functional HRD.

    Three sample groups (ideally from matched data):
      1. BRCA-mutant, no reversion → HRD by ALL methods
      2. BRCA-mutant, WITH reversion → HRD by scars, but HRP by expression
      3. BRCA-wildtype → HRP by ALL methods

    If our signature correctly identifies group 2 as HRP-like while
    scar-based scores remain high, this demonstrates that expression-
    based HRD captures the dynamic repair state, not just history.
    """

    def __init__(self, scorer=None, scar_score_col="HRD_sum"):
        """
        Parameters
        ----------
        scorer : SignatureScorer or None
            Expression-based HRD scorer. Can be set later via set_scorer().
        scar_score_col : str
            Column name in labels for the scar-based score (GIS/HRD-sum).
        """
        self.scorer = scorer
        self.scar_score_col = scar_score_col

    def set_scorer(self, scorer):
        """Set or update the expression-based scorer."""
        self.scorer = scorer

    def simulate_reversion_effect(self, expression_df, sample_annotations,
                                  brca_col="BRCA_status",
                                  reversion_col="has_reversion"):
        """Compare scores across BRCA groups to test reversion hypothesis.

        When real reversion data is unavailable, this uses the three-group
        design as a proxy:
          - BRCA-mut (no reversion): high scar + high expression HRD
          - BRCA-mut (with reversion): high scar + LOW expression HRD
          - BRCA-WT: low scar + low expression HRD

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix (samples × genes).
        sample_annotations : pd.DataFrame
            Must contain brca_col and optionally reversion_col,
            plus scar_score_col.
        brca_col : str
            Column with BRCA mutation status.
        reversion_col : str
            Column indicating reversion (True/False).

        Returns
        -------
        dict
            group_scores, statistical_tests, interpretation
        """
        if self.scorer is None:
            raise RuntimeError("Set a scorer first via set_scorer()")

        # Align
        common = expression_df.index.intersection(sample_annotations.index)
        expr = expression_df.loc[common]
        ann = sample_annotations.loc[common]

        # Score with expression-based method
        expr_scores = self.scorer.score(expr)

        # Define groups
        has_reversion = reversion_col in ann.columns
        groups = {}

        if has_reversion:
            brca_mut = ann[brca_col].isin(["BRCA1", "BRCA2", "HRD", "mutant", 1, True])
            reversion = ann[reversion_col].astype(bool)

            groups["BRCA_mut_no_reversion"] = common[brca_mut & ~reversion]
            groups["BRCA_mut_with_reversion"] = common[brca_mut & reversion]
            groups["BRCA_wildtype"] = common[~brca_mut]
        else:
            # Without explicit reversion data, just compare BRCA-mut vs WT
            brca_mut = ann[brca_col].isin(["BRCA1", "BRCA2", "HRD", "mutant", 1, True])
            groups["BRCA_mutant"] = common[brca_mut]
            groups["BRCA_wildtype"] = common[~brca_mut]

        # Collect scores per group
        group_scores = {}
        for gname, gidx in groups.items():
            if len(gidx) == 0:
                continue
            group_scores[gname] = {
                "n": len(gidx),
                "expr_score_mean": float(expr_scores.loc[gidx].mean()),
                "expr_score_std": float(expr_scores.loc[gidx].std()),
                "expr_scores": expr_scores.loc[gidx],
            }
            if self.scar_score_col in ann.columns:
                scar_vals = pd.to_numeric(ann.loc[gidx, self.scar_score_col],
                                          errors="coerce")
                group_scores[gname]["scar_score_mean"] = float(scar_vals.mean())
                group_scores[gname]["scar_score_std"] = float(scar_vals.std())
                group_scores[gname]["scar_scores"] = scar_vals

        # Statistical tests
        stat_tests = self._run_group_tests(group_scores)

        # Interpretation
        interpretation = self._interpret_results(group_scores, has_reversion)

        return {
            "groups": group_scores,
            "tests": stat_tests,
            "interpretation": interpretation,
            "has_reversion_data": has_reversion,
        }

    def analyze_tempus_reversions(self, tempus_data=None):
        """Analyze reversion cases from Tempus cohort if available.

        Parameters
        ----------
        tempus_data : dict or None
            Expected keys: 'expression', 'annotations', 'reversion_info'.
            If None, returns a documentation-only result describing what
            the analysis would look like.

        Returns
        -------
        dict
            Analysis results or documentation framework.
        """
        if tempus_data is None:
            return self._document_tempus_framework()

        if self.scorer is None:
            raise RuntimeError("Set a scorer first via set_scorer()")

        expr = tempus_data["expression"]
        ann = tempus_data["annotations"]
        rev = tempus_data["reversion_info"]

        # Merge reversion info
        ann = ann.join(rev[["has_reversion", "reversion_gene", "reversion_type"]],
                       how="left")
        ann["has_reversion"] = ann["has_reversion"].fillna(False)

        return self.simulate_reversion_effect(
            expr, ann,
            brca_col="BRCA_status",
            reversion_col="has_reversion",
        )

    def plot_reversion_comparison(self, result, figsize=(12, 5)):
        """Visualize expression vs scar scores across BRCA groups.

        Parameters
        ----------
        result : dict
            Output from simulate_reversion_effect().
        figsize : tuple

        Returns
        -------
        matplotlib.figure.Figure
        """
        groups = result["groups"]
        has_scar = any("scar_scores" in v for v in groups.values())

        ncols = 2 if has_scar else 1
        fig, axes = plt.subplots(1, ncols, figsize=figsize)
        if ncols == 1:
            axes = [axes]

        # Expression-based scores
        plot_data = []
        for gname, gdata in groups.items():
            for val in gdata["expr_scores"]:
                plot_data.append({"Group": gname, "Score": val,
                                  "Type": "Expression-based"})
        df_expr = pd.DataFrame(plot_data)

        palette = {
            "BRCA_mut_no_reversion": "#d62728",
            "BRCA_mut_with_reversion": "#ff7f0e",
            "BRCA_wildtype": "#2ca02c",
            "BRCA_mutant": "#d62728",
        }

        sns.boxplot(data=df_expr, x="Group", y="Score", hue="Group",
                    ax=axes[0], palette=palette, width=0.5, legend=False)
        sns.stripplot(data=df_expr, x="Group", y="Score", ax=axes[0],
                      color=".3", alpha=0.4, size=3)
        axes[0].set_title("Expression-based HRD Score")
        axes[0].set_ylabel("HRD Score")
        axes[0].tick_params(axis="x", rotation=30)

        # Scar-based scores
        if has_scar:
            plot_data_scar = []
            for gname, gdata in groups.items():
                if "scar_scores" not in gdata:
                    continue
                for val in gdata["scar_scores"].dropna():
                    plot_data_scar.append({"Group": gname, "Score": val,
                                           "Type": "Scar-based (GIS)"})
            df_scar = pd.DataFrame(plot_data_scar)

            if not df_scar.empty:
                sns.boxplot(data=df_scar, x="Group", y="Score", hue="Group",
                            ax=axes[1], palette=palette, width=0.5, legend=False)
                sns.stripplot(data=df_scar, x="Group", y="Score", ax=axes[1],
                              color=".3", alpha=0.4, size=3)
                axes[1].set_title("Scar-based Score (GIS/HRD-sum)")
                axes[1].set_ylabel("HRD-sum")
                axes[1].tick_params(axis="x", rotation=30)

        fig.suptitle("Reversion Analysis: Expression vs Scar-based Scores",
                     fontsize=13, y=1.02)
        fig.tight_layout()
        return fig

    def plot_scatter_expr_vs_scar(self, result, ax=None, figsize=(7, 6)):
        """Scatter: expression score vs scar score, colored by group.

        Key insight: reversion samples should fall in the upper-left
        quadrant (high scar, low expression-HRD).

        Returns
        -------
        matplotlib.figure.Figure
        """
        groups = result["groups"]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        palette = {
            "BRCA_mut_no_reversion": "#d62728",
            "BRCA_mut_with_reversion": "#ff7f0e",
            "BRCA_wildtype": "#2ca02c",
            "BRCA_mutant": "#d62728",
        }

        for gname, gdata in groups.items():
            if "scar_scores" not in gdata:
                continue
            common_idx = gdata["expr_scores"].index.intersection(
                gdata["scar_scores"].dropna().index
            )
            if len(common_idx) == 0:
                continue
            ax.scatter(
                gdata["scar_scores"].loc[common_idx],
                gdata["expr_scores"].loc[common_idx],
                label=f"{gname} (n={len(common_idx)})",
                alpha=0.6, s=30,
                color=palette.get(gname, "gray"),
            )

        ax.set_xlabel("Scar-based Score (GIS/HRD-sum)")
        ax.set_ylabel("Expression-based HRD Score")
        ax.set_title("Expression vs Scar: Identifying Reversions")
        ax.legend(fontsize=8)

        # Add quadrant annotations
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmid = (xlim[0] + xlim[1]) / 2
        ymid = (ylim[0] + ylim[1]) / 2
        ax.axhline(ymid, color="gray", ls="--", alpha=0.3)
        ax.axvline(xmid, color="gray", ls="--", alpha=0.3)
        ax.text(xlim[1] * 0.95, ylim[0] * 1.05, "Reversion\ncandidate",
                ha="right", va="bottom", fontsize=8, style="italic",
                color="orange", alpha=0.7)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _run_group_tests(group_scores):
        """Run statistical comparisons between groups."""
        tests = {}
        group_names = list(group_scores.keys())

        for i, ga in enumerate(group_names):
            for j, gb in enumerate(group_names):
                if j <= i:
                    continue
                key = f"{ga}_vs_{gb}"
                sa = group_scores[ga]["expr_scores"].values
                sb = group_scores[gb]["expr_scores"].values

                if len(sa) > 1 and len(sb) > 1:
                    t, p_t = stats.ttest_ind(sa, sb, equal_var=False)
                    u, p_u = stats.mannwhitneyu(sa, sb, alternative="two-sided")
                    tests[key] = {
                        "ttest_stat": float(t),
                        "ttest_p": float(p_t),
                        "mannwhitney_u": float(u),
                        "mannwhitney_p": float(p_u),
                    }

        return tests

    @staticmethod
    def _interpret_results(group_scores, has_reversion):
        """Generate human-readable interpretation."""
        lines = []

        if has_reversion and "BRCA_mut_with_reversion" in group_scores:
            rev = group_scores["BRCA_mut_with_reversion"]
            mut = group_scores.get("BRCA_mut_no_reversion", {})
            wt = group_scores.get("BRCA_wildtype", {})

            lines.append(f"Reversion samples (n={rev['n']}): "
                         f"mean expr score = {rev['expr_score_mean']:.3f}")

            if mut:
                lines.append(f"BRCA-mut no reversion (n={mut['n']}): "
                             f"mean expr score = {mut['expr_score_mean']:.3f}")
            if wt:
                lines.append(f"BRCA-WT (n={wt['n']}): "
                             f"mean expr score = {wt['expr_score_mean']:.3f}")

            # Key test: do reversion samples look more like WT than mut?
            if mut and wt:
                rev_mean = rev["expr_score_mean"]
                mut_mean = mut["expr_score_mean"]
                wt_mean = wt["expr_score_mean"]

                closer_to_wt = abs(rev_mean - wt_mean) < abs(rev_mean - mut_mean)
                if closer_to_wt:
                    lines.append(
                        "RESULT: Reversion samples are CLOSER to BRCA-WT on "
                        "expression score → signature captures functional state."
                    )
                else:
                    lines.append(
                        "RESULT: Reversion samples still closer to BRCA-mut → "
                        "signature may still reflect residual HRD biology or "
                        "insufficient reversion sample size."
                    )

            # Check scar scores
            if "scar_score_mean" in rev:
                lines.append(
                    f"Scar-based: reversion mean = {rev['scar_score_mean']:.1f}, "
                    f"mut mean = {mut.get('scar_score_mean', float('nan')):.1f}, "
                    f"WT mean = {wt.get('scar_score_mean', float('nan')):.1f}"
                )
                if rev.get("scar_score_mean", 0) > 30:
                    lines.append(
                        "As expected, scar score REMAINS HIGH in reversion "
                        "samples (genomic damage is permanent)."
                    )
        else:
            lines.append(
                "No explicit reversion data — using BRCA-mut vs WT comparison "
                "as proxy for functional HRD detection."
            )
            for gname, gdata in group_scores.items():
                lines.append(
                    f"{gname} (n={gdata['n']}): "
                    f"mean expr = {gdata['expr_score_mean']:.3f}"
                )

        return "\n".join(lines)

    @staticmethod
    def _document_tempus_framework():
        """Document the expected Tempus reversion analysis."""
        return {
            "status": "framework_only",
            "description": (
                "Tempus reversion analysis framework. When Tempus data is "
                "available with identified BRCA reversion mutations, this "
                "analysis will:\n"
                "1. Identify samples with pathogenic BRCA1/2 mutations AND "
                "   secondary reversion mutations (frameshift corrections, "
                "   intragenic deletions restoring reading frame)\n"
                "2. Score all samples with both expression-based and scar-based "
                "   methods\n"
                "3. Test whether expression score drops in reversion cases "
                "   while scar score remains high\n"
                "4. Quantify the 'reversion effect' as the gap between "
                "   expression-predicted and scar-predicted HRD status\n"
                "5. Correlate with clinical outcomes (PARPi response post-"
                "   reversion should be poor despite high scar scores)"
            ),
            "expected_data_format": {
                "expression": "pd.DataFrame, samples × genes",
                "annotations": "pd.DataFrame with columns: BRCA_status, HRD_sum",
                "reversion_info": (
                    "pd.DataFrame with columns: sample_id, has_reversion, "
                    "reversion_gene, reversion_type"
                ),
            },
        }
