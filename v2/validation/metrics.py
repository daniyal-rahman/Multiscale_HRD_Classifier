"""
Comprehensive evaluation metrics for HRD signature validation.

Covers binary classification, survival analysis, drug response,
calibration, and inter-signature agreement.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn import metrics as sk_metrics

logger = logging.getLogger(__name__)


class ValidationMetrics:
    """Compute all metrics relevant to HRD signature evaluation."""

    # ------------------------------------------------------------------
    # Binary classification
    # ------------------------------------------------------------------
    @staticmethod
    def binary_metrics(y_true, y_score, y_pred=None, pos_label=1) -> dict:
        """Full binary classification metrics.

        Parameters
        ----------
        y_true : array-like
            Ground-truth binary labels (0/1).
        y_score : array-like
            Continuous scores (e.g. P(HRD)).
        y_pred : array-like or None
            Binary predictions. If None, uses y_score > 0.5.
        pos_label : int
            Positive class label.

        Returns
        -------
        dict
            Keys: auc, sensitivity, specificity, ppv, npv, f1, mcc,
            brier, accuracy, balanced_accuracy, n_pos, n_neg,
            youden_threshold (optimal threshold from ROC).
        """
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)

        if y_pred is None:
            y_pred = (y_score >= 0.5).astype(int)
        else:
            y_pred = np.asarray(y_pred, dtype=int)

        result = {}

        # AUC (handle edge cases)
        n_pos = int((y_true == pos_label).sum())
        n_neg = int((y_true != pos_label).sum())
        result["n_pos"] = n_pos
        result["n_neg"] = n_neg

        if n_pos == 0 or n_neg == 0:
            result["auc"] = float("nan")
            result["youden_threshold"] = float("nan")
        else:
            result["auc"] = sk_metrics.roc_auc_score(y_true, y_score)
            fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_score)
            youden_idx = np.argmax(tpr - fpr)
            result["youden_threshold"] = float(thresholds[youden_idx])

        # Confusion-matrix derived
        tn, fp, fn, tp = sk_metrics.confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        result["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        result["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        result["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        result["npv"] = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
        result["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
        result["balanced_accuracy"] = sk_metrics.balanced_accuracy_score(y_true, y_pred)
        result["f1"] = sk_metrics.f1_score(y_true, y_pred, zero_division=0)
        result["mcc"] = sk_metrics.matthews_corrcoef(y_true, y_pred)

        # Brier score requires probabilities in [0, 1]
        if np.all((y_score >= 0) & (y_score <= 1)):
            result["brier"] = sk_metrics.brier_score_loss(y_true, y_score)
        else:
            # Rescale to [0, 1] for Brier computation
            s_min, s_max = y_score.min(), y_score.max()
            if s_max > s_min:
                y_prob = (y_score - s_min) / (s_max - s_min)
                result["brier"] = sk_metrics.brier_score_loss(y_true, y_prob)
            else:
                result["brier"] = float("nan")

        return result

    # ------------------------------------------------------------------
    # Survival analysis
    # ------------------------------------------------------------------
    @staticmethod
    def survival_metrics(y_time, y_event, y_score) -> dict:
        """Survival-based evaluation using lifelines.

        Parameters
        ----------
        y_time : array-like
            Time-to-event (e.g. PFI in months).
        y_event : array-like
            Event indicator (1 = event, 0 = censored).
        y_score : array-like
            Continuous HRD score.

        Returns
        -------
        dict
            Keys: c_index, logrank_p, hazard_ratio, hr_ci_lower,
            hr_ci_upper, median_score_cutoff.
        """
        try:
            from lifelines import CoxPHFitter
            from lifelines.statistics import logrank_test
            from lifelines.utils import concordance_index
        except ImportError:
            logger.warning("lifelines not installed — survival metrics unavailable")
            return {"error": "lifelines not installed"}

        y_time = np.asarray(y_time, dtype=float)
        y_event = np.asarray(y_event, dtype=int)
        y_score = np.asarray(y_score, dtype=float)

        # Remove NaN rows
        valid = ~(np.isnan(y_time) | np.isnan(y_event) | np.isnan(y_score))
        y_time, y_event, y_score = y_time[valid], y_event[valid], y_score[valid]

        result = {}

        # C-index
        try:
            result["c_index"] = concordance_index(y_time, -y_score, y_event)
        except Exception as e:
            logger.warning("C-index failed: %s", e)
            result["c_index"] = float("nan")

        # Log-rank test by median split
        median_score = float(np.median(y_score))
        result["median_score_cutoff"] = median_score
        high = y_score >= median_score
        low = ~high

        if high.sum() > 0 and low.sum() > 0:
            lr = logrank_test(
                y_time[high], y_time[low],
                event_observed_A=y_event[high],
                event_observed_B=y_event[low],
            )
            result["logrank_p"] = lr.p_value
        else:
            result["logrank_p"] = float("nan")

        # Cox regression for hazard ratio
        try:
            df = pd.DataFrame({
                "time": y_time, "event": y_event, "score": y_score,
            })
            cph = CoxPHFitter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(df, duration_col="time", event_col="event")
            result["hazard_ratio"] = float(np.exp(cph.params_["score"]))
            ci = cph.confidence_intervals_
            result["hr_ci_lower"] = float(np.exp(ci.iloc[0, 0]))
            result["hr_ci_upper"] = float(np.exp(ci.iloc[0, 1]))
        except Exception as e:
            logger.warning("Cox regression failed: %s", e)
            result["hazard_ratio"] = float("nan")
            result["hr_ci_lower"] = float("nan")
            result["hr_ci_upper"] = float("nan")

        return result

    # ------------------------------------------------------------------
    # Drug response
    # ------------------------------------------------------------------
    @staticmethod
    def drug_response_metrics(response, y_score) -> dict:
        """Metrics for binary clinical response (pCR, platinum sensitivity).

        Parameters
        ----------
        response : array-like
            Binary response (1 = responder, 0 = non-responder).
        y_score : array-like
            Continuous HRD score.

        Returns
        -------
        dict
            Keys: auc, ttest_p, mannwhitney_p, mean_responder,
            mean_nonresponder, effect_size_cohend, odds_ratio, fisher_p.
        """
        response = np.asarray(response, dtype=int)
        y_score = np.asarray(y_score, dtype=float)

        valid = ~(np.isnan(response) | np.isnan(y_score))
        response, y_score = response[valid], y_score[valid]

        responders = y_score[response == 1]
        nonresponders = y_score[response == 0]

        result = {
            "n_responders": int(len(responders)),
            "n_nonresponders": int(len(nonresponders)),
            "mean_responder": float(np.mean(responders)) if len(responders) > 0 else float("nan"),
            "mean_nonresponder": float(np.mean(nonresponders)) if len(nonresponders) > 0 else float("nan"),
        }

        # AUC
        if len(responders) > 0 and len(nonresponders) > 0:
            result["auc"] = sk_metrics.roc_auc_score(response, y_score)
        else:
            result["auc"] = float("nan")

        # t-test and Mann-Whitney
        if len(responders) > 1 and len(nonresponders) > 1:
            t, p_t = sp_stats.ttest_ind(responders, nonresponders, equal_var=False)
            result["ttest_stat"] = float(t)
            result["ttest_p"] = float(p_t)

            u, p_u = sp_stats.mannwhitneyu(
                responders, nonresponders, alternative="two-sided"
            )
            result["mannwhitney_u"] = float(u)
            result["mannwhitney_p"] = float(p_u)

            # Cohen's d effect size
            pooled_std = np.sqrt(
                (np.var(responders, ddof=1) + np.var(nonresponders, ddof=1)) / 2
            )
            if pooled_std > 0:
                result["effect_size_cohend"] = float(
                    (np.mean(responders) - np.mean(nonresponders)) / pooled_std
                )
            else:
                result["effect_size_cohend"] = float("nan")
        else:
            result["ttest_p"] = float("nan")
            result["mannwhitney_p"] = float("nan")
            result["effect_size_cohend"] = float("nan")

        # Odds ratio via median split
        if len(y_score) > 4:
            median = np.median(y_score)
            high_pred = y_score >= median
            # 2x2 table: high_pred x response
            a = int(((high_pred) & (response == 1)).sum())
            b = int(((high_pred) & (response == 0)).sum())
            c = int(((~high_pred) & (response == 1)).sum())
            d = int(((~high_pred) & (response == 0)).sum())
            result["contingency_table"] = [[a, b], [c, d]]

            if b > 0 and c > 0:
                result["odds_ratio"] = (a * d) / (b * c)
            else:
                result["odds_ratio"] = float("inf") if a * d > 0 else float("nan")

            _, fisher_p = sp_stats.fisher_exact([[a, b], [c, d]])
            result["fisher_p"] = float(fisher_p)
        else:
            result["odds_ratio"] = float("nan")
            result["fisher_p"] = float("nan")

        return result

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    @staticmethod
    def calibration_metrics(y_true, y_prob, n_bins=10) -> dict:
        """Probability calibration assessment.

        Parameters
        ----------
        y_true : array-like
            Binary ground truth.
        y_prob : array-like
            Predicted probabilities.
        n_bins : int
            Number of bins for calibration curve.

        Returns
        -------
        dict
            Keys: brier, calibration_curve (dict of bin_means, true_fracs),
            hosmer_lemeshow_stat, hosmer_lemeshow_p.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)

        result = {}
        result["brier"] = float(sk_metrics.brier_score_loss(y_true, y_prob))

        # Calibration curve data
        from sklearn.calibration import calibration_curve
        true_frac, pred_mean = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        result["calibration_curve"] = {
            "predicted_mean": pred_mean.tolist(),
            "true_fraction": true_frac.tolist(),
        }

        # Hosmer-Lemeshow test
        try:
            hl_stat, hl_p = _hosmer_lemeshow(y_true, y_prob, n_bins)
            result["hosmer_lemeshow_stat"] = float(hl_stat)
            result["hosmer_lemeshow_p"] = float(hl_p)
        except Exception as e:
            logger.warning("Hosmer-Lemeshow failed: %s", e)
            result["hosmer_lemeshow_stat"] = float("nan")
            result["hosmer_lemeshow_p"] = float("nan")

        return result

    # ------------------------------------------------------------------
    # Inter-signature agreement
    # ------------------------------------------------------------------
    @staticmethod
    def signature_agreement(scores_dict: dict, threshold=0.5) -> pd.DataFrame:
        """Pairwise agreement between multiple signatures.

        Parameters
        ----------
        scores_dict : dict
            {signature_name: pd.Series of continuous scores}
            All series must share the same index (aligned samples).
        threshold : float
            Threshold for binarizing scores before computing kappa.

        Returns
        -------
        pd.DataFrame
            Pairwise comparison table with columns: sig_a, sig_b,
            spearman_r, spearman_p, cohen_kappa, concordance_rate.
        """
        names = list(scores_dict.keys())
        # Align to common index
        common_idx = scores_dict[names[0]].dropna().index
        for name in names[1:]:
            common_idx = common_idx.intersection(scores_dict[name].dropna().index)

        records = []
        for i, a_name in enumerate(names):
            for j, b_name in enumerate(names):
                if j <= i:
                    continue
                a = scores_dict[a_name].loc[common_idx].values
                b = scores_dict[b_name].loc[common_idx].values

                r, p = sp_stats.spearmanr(a, b)

                a_bin = (a >= threshold).astype(int)
                b_bin = (b >= threshold).astype(int)
                kappa = sk_metrics.cohen_kappa_score(a_bin, b_bin)
                concordance = float((a_bin == b_bin).mean())

                records.append({
                    "sig_a": a_name,
                    "sig_b": b_name,
                    "n_samples": len(common_idx),
                    "spearman_r": float(r),
                    "spearman_p": float(p),
                    "cohen_kappa": float(kappa),
                    "concordance_rate": concordance,
                })

        return pd.DataFrame(records)


# ======================================================================
# Helpers
# ======================================================================

def _hosmer_lemeshow(y_true, y_prob, n_groups=10):
    """Hosmer-Lemeshow goodness-of-fit test."""
    order = np.argsort(y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]

    groups = np.array_split(np.arange(len(y_true)), n_groups)

    hl_stat = 0.0
    for g in groups:
        obs = y_true[g].sum()
        n_g = len(g)
        exp = y_prob[g].sum()

        if exp > 0 and (n_g - exp) > 0:
            hl_stat += (obs - exp) ** 2 / exp
            hl_stat += ((n_g - obs) - (n_g - exp)) ** 2 / (n_g - exp)

    p_value = 1 - sp_stats.chi2.cdf(hl_stat, df=max(n_groups - 2, 1))
    return hl_stat, p_value
