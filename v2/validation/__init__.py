"""
v2.validation — Comprehensive validation framework for HRD signatures.

Modules:
    scoring       — Unified interface for scoring samples with any HRD method
    metrics       — Binary, survival, drug-response, and calibration metrics
    head_to_head  — Multi-signature comparison on shared datasets
    cohort_validators — Dataset-specific validation classes (TCGA-OV, I-SPY2, GEO, etc.)
    reversion_analysis — Functional HRD vs scar-based analysis
    validation_report  — Automated HTML report generation
"""

from .scoring import SignatureScorer, CentroidScorer
from .metrics import ValidationMetrics
from .head_to_head import HeadToHead
