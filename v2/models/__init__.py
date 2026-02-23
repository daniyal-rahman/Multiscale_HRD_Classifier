"""
HRD classification models — v2.

Modules:
  confident_learning  — Label noise detection (Cleanlab-inspired)
  label_smoothing     — Soft label utilities
  multi_task_model    — Multi-task XGBoost/LightGBM
  baseline_models     — ElasticNet, centroid, single-gene baselines
  training_pipeline   — End-to-end orchestrator
"""

from .confident_learning import ConfidentLearner
from .label_smoothing import (
    apply_label_smoothing,
    smooth_binary_labels,
    soft_label_from_score,
)
from .multi_task_model import MultiTaskHRDModel
from .baseline_models import (
    CentroidBaseline,
    ElasticNetBaseline,
    SingleGeneBaseline,
)
from .training_pipeline import (
    FeatureConfig,
    HRDTrainingPipeline,
    LabelConfig,
    ModelConfig,
)
