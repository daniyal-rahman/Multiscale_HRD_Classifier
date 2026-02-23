"""
Label smoothing utilities for HRD classification.

Converts hard binary labels into soft probabilistic targets using:
  1. Continuous GIS / HRDetect scores as informative soft targets
  2. Uniform label smoothing for regularization

Soft labels help gradient boosting models learn calibrated probabilities
rather than overfitting to noisy binary thresholds (e.g. HRD-sum >= 42).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def smooth_binary_labels(
    y_binary: np.ndarray | pd.Series,
    continuous_score: np.ndarray | pd.Series,
    temperature: float = 1.0,
    score_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Convert hard 0/1 labels to soft targets using a continuous score.

    The continuous score (e.g. HRD-sum, HRDetect probability) is rescaled
    to [0, 1] and blended with the binary label using a sigmoid-temperature
    mapping. Samples near the decision boundary get labels closer to 0.5,
    while clear-cut samples stay near 0/1.

    Parameters
    ----------
    y_binary : array-like of {0, 1}
        Hard binary labels.
    continuous_score : array-like of float
        Continuous score correlated with HRD status (higher = more HRD).
    temperature : float
        Controls how much the soft label deviates from the binary label.
        temperature=0 → hard labels unchanged.
        temperature=1 → fully determined by rescaled score.
        Values in (0, 1) blend between hard and soft.
    score_range : tuple of (min, max) or None
        Range for min-max normalization. If None, uses data min/max.

    Returns
    -------
    np.ndarray
        Soft labels in [0, 1].
    """
    y = np.asarray(y_binary, dtype=float)
    scores = np.asarray(continuous_score, dtype=float)

    if score_range is not None:
        lo, hi = score_range
    else:
        lo, hi = np.nanmin(scores), np.nanmax(scores)

    # Rescale scores to [0, 1]
    denom = hi - lo
    if denom == 0:
        soft = np.full_like(scores, 0.5)
    else:
        soft = np.clip((scores - lo) / denom, 0.0, 1.0)

    # Blend: (1 - temperature) * hard + temperature * soft
    temperature = np.clip(temperature, 0.0, 1.0)
    blended = (1.0 - temperature) * y + temperature * soft

    return np.clip(blended, 0.0, 1.0)


def apply_label_smoothing(
    y: np.ndarray | pd.Series,
    alpha: float = 0.1,
    n_classes: int = 2,
) -> np.ndarray:
    """Apply uniform label smoothing.

    Shifts hard labels toward uniform distribution:
        y_smooth = (1 - alpha) * y + alpha / K

    Parameters
    ----------
    y : array-like
        Labels in [0, 1] (can be hard or already soft).
    alpha : float
        Smoothing strength. 0 = no smoothing, 1 = uniform distribution.
    n_classes : int
        Number of classes (K). Default 2 for binary.

    Returns
    -------
    np.ndarray
        Smoothed labels.
    """
    y = np.asarray(y, dtype=float)
    alpha = np.clip(alpha, 0.0, 1.0)
    return (1.0 - alpha) * y + alpha / n_classes


def soft_label_from_score(
    hrd_sum: np.ndarray | pd.Series,
    hrd_threshold: float = 42.0,
    hrp_threshold: float = 10.0,
    method: str = "sigmoid",
) -> np.ndarray:
    """Create soft labels from HRD-sum score using various mappings.

    Parameters
    ----------
    hrd_sum : array-like
        HRD-sum (LOH + TAI + LST) scores.
    hrd_threshold : float
        Score above which label approaches 1.0.
    hrp_threshold : float
        Score below which label approaches 0.0.
    method : str
        Mapping method: 'sigmoid', 'linear', or 'quadratic'.

    Returns
    -------
    np.ndarray
        Soft labels in [0, 1].
    """
    scores = np.asarray(hrd_sum, dtype=float)

    if method == "linear":
        soft = np.clip(
            (scores - hrp_threshold) / (hrd_threshold - hrp_threshold),
            0.0, 1.0,
        )
    elif method == "sigmoid":
        midpoint = (hrd_threshold + hrp_threshold) / 2
        scale = (hrd_threshold - hrp_threshold) / 4  # ~4 SDs in sigmoid
        if scale == 0:
            soft = np.where(scores >= midpoint, 1.0, 0.0)
        else:
            soft = 1.0 / (1.0 + np.exp(-(scores - midpoint) / scale))
    elif method == "quadratic":
        # Reproduce v1 softLabel approach: quadratic mapping in ambiguous zone
        median = (hrd_threshold + hrp_threshold) / 2
        soft = np.zeros_like(scores)
        for i, x in enumerate(scores):
            if x < hrp_threshold:
                soft[i] = 0.0
            elif x >= hrd_threshold:
                soft[i] = 1.0
            else:
                adj = 2 * ((((hrd_threshold - x) / (hrd_threshold - hrp_threshold)) - 0.5) ** 2) + 0.5
                if x >= median:
                    soft[i] = min(adj, 1.0)
                else:
                    soft[i] = max(1.0 - adj, 0.0)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sigmoid', 'linear', or 'quadratic'.")

    return soft
