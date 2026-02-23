"""
End-to-end HRD training pipeline orchestrator.

Chains together:
  1. Tiered labeling (from v2/label_engineering/)
  2. Confident learning for noise detection
  3. Feature engineering (from v2/feature_engineering/)
  4. Multi-task gradient boosting model
  5. Nested cross-validation
  6. Baseline comparison

Designed to consume raw expression + metadata DataFrames and produce
a complete evaluation report.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..label_engineering.tiered_labels import TieredHRDLabeler
from .confident_learning import ConfidentLearner
from .label_smoothing import smooth_binary_labels, soft_label_from_score
from .multi_task_model import MultiTaskHRDModel
from .baseline_models import ElasticNetBaseline, CentroidBaseline, SingleGeneBaseline

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering step."""
    use_rank_pairs: bool = True
    use_pathway_scores: bool = True
    use_ratios: bool = True
    use_raw_expression: bool = False
    n_pairs: int = 50
    pathway_method: str = "ssgsea"


@dataclass
class ModelConfig:
    """Configuration for model training."""
    base_model: str = "xgboost"
    tasks: list = field(default_factory=lambda: ["hrd_status", "brca_type"])
    use_label_smoothing: bool = True
    cross_task_features: bool = False
    model_params: dict = field(default_factory=dict)
    cv_n_splits: int = 5
    cv_strategy: str = "stratified"


@dataclass
class LabelConfig:
    """Configuration for labeling and noise detection."""
    gis_positive_threshold: int = 42
    gis_negative_threshold: int = 20
    soft_label_method: str = "sigmoid"
    soft_label_temperature: float = 0.8
    run_confident_learning: bool = True
    noise_strategy: str = "weight"  # 'prune', 'relabel', or 'weight'
    confident_learning_folds: int = 5


class HRDTrainingPipeline:
    """Full HRD training pipeline from raw data to evaluated models.

    Parameters
    ----------
    feature_config : FeatureConfig or None
    model_config : ModelConfig or None
    label_config : LabelConfig or None
    """

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        model_config: ModelConfig | None = None,
        label_config: LabelConfig | None = None,
    ):
        self.feature_config = feature_config or FeatureConfig()
        self.model_config = model_config or ModelConfig()
        self.label_config = label_config or LabelConfig()

        # Populated during run()
        self.tiered_labels_: Optional[pd.DataFrame] = None
        self.noise_results_: Optional[dict] = None
        self.feature_matrix_: Optional[pd.DataFrame] = None
        self.model_: Optional[MultiTaskHRDModel] = None
        self.cv_results_: Optional[dict] = None
        self.baseline_results_: Optional[dict] = None
        self.timing_: dict = {}

    def run(
        self,
        expression_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        cnv_df: Optional[pd.DataFrame] = None,
        brca_status_df: Optional[pd.DataFrame] = None,
        hrd_scores_df: Optional[pd.DataFrame] = None,
        dataset_col: Optional[pd.Series] = None,
        run_baselines: bool = True,
    ) -> dict:
        """Execute the full pipeline.

        Parameters
        ----------
        expression_df : DataFrame
            Gene expression matrix (samples x genes).
        metadata_df : DataFrame
            Must contain 'HRD-sum' and optionally BRCA event columns.
            Index = sample IDs.
        cnv_df : DataFrame or None
            Copy number data (optional).
        brca_status_df : DataFrame or None
            Detailed BRCA status for tiered labeling.
        hrd_scores_df : DataFrame or None
            HRD scores for tiered labeling.
        dataset_col : Series or None
            Dataset/batch identifiers for leave-one-dataset-out CV.
        run_baselines : bool
            Whether to run baseline comparisons.

        Returns
        -------
        dict with keys:
            labels, noise_results, features, cv_results, baseline_results, timing
        """
        t0 = time.time()

        # ---- Step 1: Labeling ----
        logger.info("Step 1: Generating labels...")
        t1 = time.time()
        y_dict, sample_ids = self._generate_labels(metadata_df, brca_status_df, hrd_scores_df)
        self.timing_["labeling"] = time.time() - t1

        # Align expression to labeled samples
        common = expression_df.index.intersection(pd.Index(sample_ids))
        X = expression_df.loc[common]
        y_dict = {k: v.loc[common] if isinstance(v, pd.Series) else v for k, v in y_dict.items()}

        # ---- Step 2: Feature engineering ----
        logger.info("Step 2: Engineering features...")
        t2 = time.time()
        X_feat = self._engineer_features(X, y_dict, cnv_df)
        self.feature_matrix_ = X_feat
        self.timing_["feature_engineering"] = time.time() - t2

        # Re-align after feature engineering (some samples may drop)
        common = X_feat.index
        y_dict = {k: v.loc[common] if isinstance(v, pd.Series) else v for k, v in y_dict.items()}
        dataset_aligned = dataset_col.loc[common] if dataset_col is not None else None

        # ---- Step 3: Confident learning ----
        sample_weights = None
        if self.label_config.run_confident_learning and "hrd_status" in y_dict:
            logger.info("Step 3: Running confident learning...")
            t3 = time.time()
            sample_weights = self._run_confident_learning(X_feat, y_dict)
            self.timing_["confident_learning"] = time.time() - t3
        else:
            logger.info("Step 3: Skipping confident learning")

        # ---- Step 4: Cross-validate multi-task model ----
        logger.info("Step 4: Cross-validating multi-task model...")
        t4 = time.time()
        self.cv_results_ = self._cross_validate(X_feat, y_dict, sample_weights, dataset_aligned)
        self.timing_["cross_validation"] = time.time() - t4

        # ---- Step 5: Fit final model on all data ----
        logger.info("Step 5: Fitting final model...")
        t5 = time.time()
        self.model_ = MultiTaskHRDModel(
            tasks=self.model_config.tasks,
            base_model=self.model_config.base_model,
            use_label_smoothing=self.model_config.use_label_smoothing,
            model_params=self.model_config.model_params,
            cross_task_features=self.model_config.cross_task_features,
        )

        # Convert y_dict to arrays
        y_arrays = {}
        for k, v in y_dict.items():
            y_arrays[k] = np.asarray(v)
        self.model_.fit(X_feat, y_arrays, sample_weights=sample_weights)
        self.timing_["final_fit"] = time.time() - t5

        # ---- Step 6: Baseline comparison ----
        if run_baselines:
            logger.info("Step 6: Running baseline comparisons...")
            t6 = time.time()
            self.baseline_results_ = self._run_baselines(X_feat, y_dict, dataset_aligned)
            self.timing_["baselines"] = time.time() - t6

        self.timing_["total"] = time.time() - t0
        logger.info("Pipeline complete in %.1fs", self.timing_["total"])

        return {
            "labels": self.tiered_labels_,
            "noise_results": self.noise_results_,
            "features": self.feature_matrix_,
            "cv_results": self.cv_results_,
            "baseline_results": self.baseline_results_,
            "model": self.model_,
            "timing": self.timing_,
        }

    # ------------------------------------------------------------------- #
    # Step implementations                                                  #
    # ------------------------------------------------------------------- #

    def _generate_labels(
        self,
        metadata_df: pd.DataFrame,
        brca_status_df: Optional[pd.DataFrame],
        hrd_scores_df: Optional[pd.DataFrame],
    ) -> tuple[dict, pd.Index]:
        """Generate multi-task labels from metadata."""
        y_dict = {}
        sample_ids = metadata_df.index

        # -- HRD status labels --
        if "HRD-sum" in metadata_df.columns:
            hrd_sum = metadata_df["HRD-sum"].astype(float)

            # Binary labels from threshold
            hrd_binary = (hrd_sum >= self.label_config.gis_positive_threshold).astype(int)

            if self.model_config.use_label_smoothing:
                soft = soft_label_from_score(
                    hrd_sum,
                    hrd_threshold=self.label_config.gis_positive_threshold,
                    hrp_threshold=self.label_config.gis_negative_threshold,
                    method=self.label_config.soft_label_method,
                )
                y_dict["hrd_status"] = pd.Series(soft, index=sample_ids, name="hrd_status")
            else:
                y_dict["hrd_status"] = pd.Series(hrd_binary, index=sample_ids, name="hrd_status")

            # Keep binary for confident learning
            y_dict["_hrd_binary"] = pd.Series(hrd_binary, index=sample_ids)

        # -- BRCA type labels --
        brca_type = self._infer_brca_type(metadata_df)
        if brca_type is not None:
            y_dict["brca_type"] = brca_type

        # -- Tiered labeling (if BRCA status data available) --
        if brca_status_df is not None and hrd_scores_df is not None:
            labeler = TieredHRDLabeler(
                gis_positive_threshold=self.label_config.gis_positive_threshold,
                gis_negative_threshold=self.label_config.gis_negative_threshold,
            )
            self.tiered_labels_ = labeler.label_tcga_cohort(brca_status_df, hrd_scores_df)

        return y_dict, sample_ids

    def _engineer_features(
        self,
        X: pd.DataFrame,
        y_dict: dict,
        cnv_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Run feature engineering pipeline."""
        sample_ids = X.index

        try:
            from ..feature_engineering.feature_transforms import HRDFeaturePipeline

            # Get binary labels for supervised feature selection
            y_binary = y_dict.get("_hrd_binary")
            if y_binary is None and "hrd_status" in y_dict:
                y_binary = (np.asarray(y_dict["hrd_status"]) >= 0.5).astype(int)
                y_binary = pd.Series(y_binary, index=X.index)

            pipeline = HRDFeaturePipeline(
                use_rank_pairs=self.feature_config.use_rank_pairs,
                use_pathway_scores=self.feature_config.use_pathway_scores,
                use_ratios=self.feature_config.use_ratios,
                use_raw_expression=self.feature_config.use_raw_expression,
                rank_kwargs={"n_pairs": self.feature_config.n_pairs},
                pathway_kwargs={"method": self.feature_config.pathway_method},
            )
            X_feat = pipeline.fit_transform(X, y_binary)

            # Validate output: index should be sample IDs, not gene names
            if len(X_feat) == 0 or not X_feat.index.intersection(sample_ids).any():
                raise ValueError("Feature pipeline produced empty or misaligned output")

            logger.info("Feature engineering produced %d features", X_feat.shape[1])
            return X_feat

        except (ImportError, Exception) as e:
            logger.warning("Feature engineering failed (%s), using raw expression", e)
            # Fallback: z-score raw expression
            X_z = (X - X.mean()) / (X.std() + 1e-8)
            return X_z

    def _run_confident_learning(
        self,
        X: pd.DataFrame,
        y_dict: dict,
    ) -> Optional[np.ndarray]:
        """Run confident learning and return sample weights."""
        # Use binary labels for confident learning
        y_binary = y_dict.get("_hrd_binary")
        if y_binary is None:
            y_binary = (np.asarray(y_dict["hrd_status"]) >= 0.5).astype(int)

        cl = ConfidentLearner(
            n_folds=self.label_config.confident_learning_folds,
        )
        results = cl.find_noisy_labels(X, y_binary)
        self.noise_results_ = results

        n_noisy = results["noise_mask"].sum()
        logger.info("Confident learning: %d potentially noisy samples", n_noisy)

        if self.label_config.noise_strategy == "weight":
            _, _, weights = cl.clean_dataset(X, y_binary, strategy="weight")
            return weights
        elif self.label_config.noise_strategy == "prune":
            # For pruning, we'd need to filter samples — handled by caller
            return None
        else:
            return None

    def _cross_validate(
        self,
        X: pd.DataFrame,
        y_dict: dict,
        sample_weights: Optional[np.ndarray],
        dataset_col: Optional[pd.Series],
    ) -> dict:
        """Run nested cross-validation."""
        # Remove internal labels
        y_cv = {k: v for k, v in y_dict.items() if not k.startswith("_")}

        model = MultiTaskHRDModel(
            tasks=self.model_config.tasks,
            base_model=self.model_config.base_model,
            use_label_smoothing=self.model_config.use_label_smoothing,
            model_params=self.model_config.model_params,
            cross_task_features=self.model_config.cross_task_features,
        )

        cv_strategy = self.model_config.cv_strategy
        if dataset_col is not None:
            cv_strategy = "group"

        return model.cross_validate(
            X, y_cv,
            cv_strategy=cv_strategy,
            n_splits=self.model_config.cv_n_splits,
            dataset_col=dataset_col,
            sample_weights=sample_weights,
        )

    def _run_baselines(
        self,
        X: pd.DataFrame,
        y_dict: dict,
        dataset_col: Optional[pd.Series],
    ) -> dict:
        """Run baseline models for comparison."""
        results = {}

        # Get binary labels for baselines
        if "_hrd_binary" in y_dict:
            y_binary = np.asarray(y_dict["_hrd_binary"])
        elif "hrd_status" in y_dict:
            y_binary = (np.asarray(y_dict["hrd_status"]) >= 0.5).astype(int)
        else:
            return results

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None

        # --- ElasticNet baseline ---
        try:
            enet = ElasticNetBaseline(mode="classification")
            enet_cv = self._cv_baseline(enet, X_arr, y_binary)
            results["elastic_net"] = enet_cv
            logger.info("ElasticNet baseline AUC: %.3f +/- %.3f",
                       enet_cv.get("auc_mean", 0), enet_cv.get("auc_std", 0))
        except Exception as e:
            logger.warning("ElasticNet baseline failed: %s", e)

        # --- Centroid baseline ---
        try:
            centroid = CentroidBaseline(n_genes=500)
            cent_cv = self._cv_baseline(centroid, X_arr, y_binary)
            results["centroid"] = cent_cv
            logger.info("Centroid baseline AUC: %.3f +/- %.3f",
                       cent_cv.get("auc_mean", 0), cent_cv.get("auc_std", 0))
        except Exception as e:
            logger.warning("Centroid baseline failed: %s", e)

        # --- Single gene baseline ---
        try:
            single = SingleGeneBaseline()
            single.fit(X, y_binary, feature_names=feature_names)
            results["single_gene"] = {
                "per_gene_aucs": single.aucs_,
                "best_gene": single.best_gene_,
                "summary": single.summary(),
            }
        except Exception as e:
            logger.warning("Single gene baseline failed: %s", e)

        return results

    @staticmethod
    def _cv_baseline(model, X, y, n_splits=5):
        """Simple stratified CV for a baseline model."""
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.base import clone

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs = []
        accs = []

        for train_idx, test_idx in skf.split(X, y):
            fold_model = clone(model)
            fold_model.fit(X[train_idx], y[train_idx])

            y_pred = fold_model.predict(X[test_idx])
            y_proba = fold_model.predict_proba(X[test_idx])

            acc = (y_pred == y[test_idx]).mean() if y_pred.dtype == y.dtype else None
            accs.append(acc)

            try:
                proba_1 = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                auc = roc_auc_score(y[test_idx], proba_1)
                aucs.append(auc)
            except ValueError:
                pass

        return {
            "auc_per_fold": aucs,
            "auc_mean": np.mean(aucs) if aucs else None,
            "auc_std": np.std(aucs) if aucs else None,
            "accuracy_per_fold": accs,
            "accuracy_mean": np.mean([a for a in accs if a is not None]) if accs else None,
        }

    @staticmethod
    def _infer_brca_type(metadata_df: pd.DataFrame) -> Optional[pd.Series]:
        """Infer BRCA subtype from metadata columns."""
        # Check for event columns
        has_brca1 = "event.BRCA1" in metadata_df.columns
        has_brca2 = "event.BRCA2" in metadata_df.columns

        if not (has_brca1 or has_brca2):
            return None

        brca_type = []
        for _, row in metadata_df.iterrows():
            b1 = str(row.get("event.BRCA1", "0"))
            b2 = str(row.get("event.BRCA2", "0"))

            if b1 == "Bi-allelic-inactivation":
                brca_type.append("BRCA1")
            elif b2 == "Bi-allelic-inactivation":
                brca_type.append("BRCA2")
            elif b1 not in ("0", "nan", "") or b2 not in ("0", "nan", ""):
                brca_type.append("HRD_BRCApos")
            else:
                brca_type.append("HRP")

        return pd.Series(brca_type, index=metadata_df.index, name="brca_type")
