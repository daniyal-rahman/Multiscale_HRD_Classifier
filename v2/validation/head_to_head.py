"""
Head-to-head multi-signature comparison on shared datasets.

Runs all configured HRD signatures on the same expression data,
computes metrics against shared labels, and generates comparison plots.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import minmax_scale

from .metrics import ValidationMetrics

logger = logging.getLogger(__name__)


class HeadToHead:
    """Run multiple HRD signatures on the same dataset and compare.

    Parameters
    ----------
    signatures : dict
        {name: SignatureScorer} — each scorer will be applied to the
        expression data and evaluated against the provided labels.
    """

    def __init__(self, signatures: dict):
        self.signatures = signatures
        self.scores_ = None
        self.metrics_ = None

    def run_all(self, expression_df, labels_df,
                label_col="label", time_col=None, event_col=None,
                response_col=None) -> pd.DataFrame:
        """Score all signatures, compute metrics, return comparison table.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix (samples x genes).
        labels_df : pd.DataFrame
            Must contain `label_col` (binary HRD 1/0).
            Optionally `time_col`/`event_col` for survival,
            or `response_col` for drug response.
        label_col : str
            Column with binary ground truth.
        time_col, event_col : str or None
            For survival metrics.
        response_col : str or None
            For drug response metrics.

        Returns
        -------
        pd.DataFrame
            Comparison table with one row per signature and metric columns.
        """
        # Align samples
        common = expression_df.index.intersection(labels_df.index)
        if len(common) == 0:
            raise ValueError("No overlapping sample IDs between expression and labels")
        logger.info("HeadToHead: %d common samples", len(common))

        expr = expression_df.loc[common]
        labels = labels_df.loc[common]
        y_true = labels[label_col].values.astype(int)

        all_scores = {}
        all_metrics = []

        for name, scorer in self.signatures.items():
            logger.info("Scoring: %s", name)
            try:
                scores = scorer.score(expr)
                scores = scores.loc[common]
            except Exception as e:
                logger.warning("Scorer '%s' failed: %s", name, e)
                continue

            all_scores[name] = scores

            # Binary metrics
            row = {"signature": name}
            bm = ValidationMetrics.binary_metrics(y_true, scores.values)
            row.update({f"binary_{k}": v for k, v in bm.items()
                        if not isinstance(v, (list, dict))})

            # Survival metrics
            if time_col and event_col and time_col in labels.columns:
                sm = ValidationMetrics.survival_metrics(
                    labels[time_col].values,
                    labels[event_col].values,
                    scores.values,
                )
                row.update({f"surv_{k}": v for k, v in sm.items()
                            if not isinstance(v, (list, dict))})

            # Drug response metrics
            if response_col and response_col in labels.columns:
                dm = ValidationMetrics.drug_response_metrics(
                    labels[response_col].values, scores.values,
                )
                row.update({f"resp_{k}": v for k, v in dm.items()
                            if not isinstance(v, (list, dict))})

            all_metrics.append(row)

        self.scores_ = pd.DataFrame(all_scores, index=common)
        self.metrics_ = pd.DataFrame(all_metrics)
        self.y_true_ = y_true
        self.labels_ = labels

        return self.metrics_

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_roc_comparison(self, ax=None, title="ROC Comparison",
                           figsize=(7, 6)):
        """Overlay ROC curves for all signatures.

        Parameters
        ----------
        ax : matplotlib Axes or None
        title : str
        figsize : tuple

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.scores_ is None:
            raise RuntimeError("Call run_all() first")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        palette = sns.color_palette("husl", len(self.scores_.columns))

        for i, name in enumerate(self.scores_.columns):
            y_score = self.scores_[name].values
            try:
                auc = roc_auc_score(self.y_true_, y_score)
                fpr, tpr, _ = roc_curve(self.y_true_, y_score)
                ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                        color=palette[i], lw=2)
            except Exception:
                continue

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        return fig

    def plot_agreement_heatmap(self, threshold=0.5, ax=None,
                               title="Inter-signature Agreement",
                               figsize=(8, 6)):
        """Heatmap of pairwise Spearman correlation between signatures.

        Parameters
        ----------
        threshold : float
            For kappa calculation.
        ax : matplotlib Axes or None
        title : str
        figsize : tuple

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.scores_ is None:
            raise RuntimeError("Call run_all() first")

        corr = self.scores_.corr(method="spearman")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, ax=ax, linewidths=0.5,
        )
        ax.set_title(title)
        fig.tight_layout()
        return fig

    def plot_score_distributions(self, figsize=(12, 5)):
        """Box + strip plots of each signature score, split by label.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.scores_ is None:
            raise RuntimeError("Call run_all() first")

        n_sigs = len(self.scores_.columns)
        fig, axes = plt.subplots(1, n_sigs, figsize=(4 * n_sigs, 5),
                                 squeeze=False)
        axes = axes.ravel()

        for i, name in enumerate(self.scores_.columns):
            df_plot = pd.DataFrame({
                "score": self.scores_[name].values,
                "label": ["HRD" if y else "HRP" for y in self.y_true_],
            })
            sns.boxplot(data=df_plot, x="label", y="score", hue="label",
                        ax=axes[i], palette="Set2", width=0.5, legend=False)
            sns.stripplot(data=df_plot, x="label", y="score", ax=axes[i],
                          color=".3", alpha=0.4, size=3, legend=False)
            axes[i].set_title(name, fontsize=10)
            axes[i].set_xlabel("")
            axes[i].set_ylabel("Score" if i == 0 else "")

        fig.suptitle("Score Distributions by HRD Status", fontsize=12)
        fig.tight_layout()
        return fig

    def identify_discordant_samples(self, threshold=0.5,
                                    min_disagreements=2) -> pd.DataFrame:
        """Find samples where signatures disagree.

        These are the interesting cases — potentially functional HRD
        without scars, or scars without functional HRD.

        Parameters
        ----------
        threshold : float
            Binarization threshold (applied per-signature after min-max scaling).
        min_disagreements : int
            Minimum number of signature pairs that must disagree.

        Returns
        -------
        pd.DataFrame
            Discordant samples with their scores and predicted classes.
        """
        if self.scores_ is None:
            raise RuntimeError("Call run_all() first")

        # Min-max scale each signature to [0, 1] before thresholding
        scaled = self.scores_.apply(
            lambda s: pd.Series(minmax_scale(s), index=s.index), axis=0
        )
        binary = (scaled >= threshold).astype(int)

        # Count per-sample: how many signatures call HRD
        hrd_count = binary.sum(axis=1)
        n_sigs = len(binary.columns)

        # Discordant = not unanimous
        discord_mask = (hrd_count > 0) & (hrd_count < n_sigs)

        result = self.scores_.loc[discord_mask].copy()
        result["n_hrd_calls"] = hrd_count.loc[discord_mask]
        result["n_signatures"] = n_sigs
        if self.y_true_ is not None:
            true_labels = pd.Series(self.y_true_, index=self.scores_.index)
            result["true_label"] = true_labels.loc[discord_mask]

        # Filter by minimum disagreements
        # A sample has k*(n-k) disagreeing pairs where k = n_hrd_calls
        pairs = result["n_hrd_calls"] * (n_sigs - result["n_hrd_calls"])
        result = result[pairs >= min_disagreements]

        return result.sort_values("n_hrd_calls", ascending=False)

    def summary_table(self) -> pd.DataFrame:
        """Return a clean summary table sorted by AUC.

        Returns
        -------
        pd.DataFrame
            Columns: signature, AUC, Sensitivity, Specificity, MCC, F1.
        """
        if self.metrics_ is None:
            raise RuntimeError("Call run_all() first")

        cols = {
            "signature": "Signature",
            "binary_auc": "AUC",
            "binary_sensitivity": "Sensitivity",
            "binary_specificity": "Specificity",
            "binary_mcc": "MCC",
            "binary_f1": "F1",
            "binary_brier": "Brier",
        }
        available = {k: v for k, v in cols.items() if k in self.metrics_.columns}
        summary = self.metrics_[list(available.keys())].rename(columns=available)
        if "AUC" in summary.columns:
            summary = summary.sort_values("AUC", ascending=False)
        return summary.reset_index(drop=True)
