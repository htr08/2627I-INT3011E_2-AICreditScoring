
"""Huấn luyện baseline: Logistic Regression và Decision Tree với 5-fold CV trên Train.

Task: Sep 28 + Sep 29 (CoReach) — Baseline CV + Log toàn bộ vào MLflow.

Chạy chính thức:
    python scripts/run_pipeline.py
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src.config import load_config
from src.data_split import load_split_data
from src.evaluate import evaluate_predictions
from src.features import build_features
from src.pipelines import make_pipeline, build_scorecard_pipeline
from src.tracking import setup_mlflow

logger = logging.getLogger(__name__)

REPORT_METRICS: List[str] = [
    "roc_auc",
    "gini",
    "ks",
    "pr_auc",
    "brier_score",
    "precision",
    "recall",
    "f1",
]

N_SPLITS = 5


def get_baseline_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Factory tạo dictionary estimator mới."""
    return {
        "logreg_baseline": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "dt_baseline": DecisionTreeClassifier(
            max_depth=5,
            random_state=random_state,
        ),
    }


def get_advanced_default_models(
    random_state: int = 42,
) -> Dict[str, BaseEstimator]:
    """Factory tạo mô hình nâng cao với tham số mặc định."""
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    return {
        "rf_default": RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost_default": XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def get_boosting_default_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Factory tạo dictionary mô hình LightGBM và CatBoost với tham số mặc định (Tuần 2 - T3: Member B)."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier

    return {
        "lightgbm_default": LGBMClassifier(
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
        "catboost_default": CatBoostClassifier(
            random_state=random_state,
            verbose=0,
            thread_count=-1,
            allow_writing_files=False,
        ),
    }


def get_imbalance_weighted_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Factory tạo dictionary mô hình LightGBM và CatBoost với class_weight / auto_class_weights (Tuần 2 - T3: Member B)."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier

    return {
        "lightgbm_balanced": LGBMClassifier(
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1,
        ),
        "catboost_balanced": CatBoostClassifier(
            random_state=random_state,
            auto_class_weights="Balanced",
            verbose=0,
            thread_count=-1,
            allow_writing_files=False,
        ),
    }


def get_data_version_tags() -> Dict[str, str]:
    """Lấy checksum dataset hoặc hash splits.json để log vào MLflow."""
    tags: Dict[str, str] = {}
    base_dir = Path(__file__).resolve().parents[1]

    sha256_path = (
        base_dir
        / "data"
        / "raw"
        / "default-of-credit-card-clients.sha256"
    )
    if sha256_path.exists():
        tags["data_sha256"] = sha256_path.read_text(
            encoding="utf-8"
        ).strip()

    splits_path = base_dir / "data" / "splits" / "splits.json"
    if splits_path.exists():
        tags["splits_hash"] = hashlib.sha256(
            splits_path.read_bytes()
        ).hexdigest()[:16]
        tags["splits_file"] = "data/splits/splits.json"

    return tags


def get_cv_splitter(
    random_state: Optional[int] = None,
    n_splits: int = N_SPLITS,
) -> StratifiedKFold:
    """Tạo bộ chia fold StratifiedKFold."""
    if random_state is None:
        cfg = load_config()
        random_state = cfg.get("random_state", 42)

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def run_cv(
    pipeline,
    X,
    y,
    n_splits: int = N_SPLITS,
    random_state: Optional[int] = None,
) -> List[Dict]:
    """Chạy Stratified K-Fold CV và xuất báo cáo feature selection nếu bật."""

    skf = get_cv_splitter(
        random_state=random_state,
        n_splits=n_splits,
    )
    fold_metrics = []
    selection_report = []

    has_feature_selection = (
        "feature_selection" in pipeline.named_steps
    )

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X, y),
        start=1,
    ):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)

        selector = pipeline.named_steps.get("feature_selection")

        if selector is not None:
            logger.info(
                "  Fold %d: selected %d features: %s",
                fold,
                len(selector.selected_features_),
                selector.selected_features_,
            )
            logger.info(
                "  Fold %d: passthrough features: %s",
                fold,
                selector.passthrough_features_,
            )

            # Lấy trực tiếp quyết định loại biến từ transformer.
            for item in getattr(
                selector,
                "dropped_features_",
                [],
            ):
                selection_report.append({
                    "fold": fold,
                    **item,
                })

        y_proba = pipeline.predict_proba(X_val)[:, 1]

        all_metrics = evaluate_predictions(
            y_val.to_numpy(),
            y_proba,
        )
        metrics = {
            key: all_metrics[key]
            for key in REPORT_METRICS
            if key in all_metrics
        }
        metrics["fold"] = fold
        fold_metrics.append(metrics)

        logger.info(
            "  Fold %d: AUC=%.4f  KS=%.4f  Gini=%.4f",
            fold,
            metrics["roc_auc"],
            metrics["ks"],
            metrics["gini"],
        )

    # Chỉ tạo báo cáo khi bật feature selection.
    if has_feature_selection:
        report_dir = Path("reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        report_columns = [
            "fold",
            "feature",
            "iv",
            "reason",
            "correlated_with",
            "correlation",
        ]

        report_path = report_dir / "correlation_feature_selection.csv"

        pd.DataFrame(
            selection_report,
            columns=report_columns,
        ).to_csv(
            report_path,
            index=False,
            encoding="utf-8-sig",
        )

        logger.info(
            "Feature selection report saved to %s",
            report_path,
        )

    return fold_metrics


def summarize_folds(
    fold_metrics: List[Dict],
) -> Dict[str, tuple]:
    """Tổng hợp mean/std (ddof=1) từ metrics của các fold."""
    if not fold_metrics:
        return {}

    keys = [
        key
        for key in fold_metrics[0]
        if key != "fold"
    ]

    return {
        key: (
            round(
                float(np.mean([metrics[key] for metrics in fold_metrics])),
                6,
            ),
            round(
                float(np.std(
                    [metrics[key] for metrics in fold_metrics],
                    ddof=1,
                )),
                6,
            ),
        )
        for key in keys
    }


def _train_and_log(
    models: Dict[str, BaseEstimator],
    alias: str,
    task: str,
    include_sex: bool = False,
    sampler: Optional[BaseEstimator] = None,
    register_model: bool = True,
) -> None:
    """Chạy 5-fold CV, log MLflow và đăng ký model."""

    cfg = load_config()
    random_state = cfg.get("random_state", 42)
    target_col = cfg.get("data", {}).get(
        "target_col",
        "default.payment.next.month",
    )

    setup_mlflow()
    train_df, _, _ = load_split_data()
    X, y = build_features(
        train_df,
        target_col=target_col,
        include_sex=include_sex,
    )

    logger.info(
        "Train size: %d samples, %d input columns",
        len(X),
        X.shape[1],
    )

    data_tags = get_data_version_tags()

    for model_name, estimator in models.items():
        run_name = (
            f"{model_name}_with_sex"
            if include_sex
            else model_name
        )
        logger.info("=== %s ===", run_name)

        pipe = make_pipeline(
            clone(estimator),
            include_sex=include_sex,
        )
        fold_metrics = run_cv(
            pipe,
            X,
            y,
            n_splits=N_SPLITS,
            random_state=random_state,
        )
        summary = summarize_folds(fold_metrics)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model", model_name)
            mlflow.log_param("n_splits", N_SPLITS)
            mlflow.log_param("n_input_columns", X.shape[1])
            mlflow.log_params(clone(estimator).get_params())
            if sampler is not None:
                mlflow.log_param("sampler", sampler.__class__.__name__)
                mlflow.log_params({f"sampler_{k}": v for k, v in clone(sampler).get_params().items()})

            for metric, (mean, std) in summary.items():
                mlflow.log_metric(f"{metric}_mean", mean)
                mlflow.log_metric(f"{metric}_std", std)

            for metrics in fold_metrics:
                fold = metrics["fold"]
                for key, value in metrics.items():
                    if key != "fold":
                        mlflow.log_metric(
                            f"{key}_fold",
                            value,
                            step=fold,
                        )

            tags = {
                "train_size": len(X),
                "default_rate": round(float(y.mean()), 4),
                "feature_strategy": "feature_freeze_v1",
                "imbalance_strategy": (
                    "smote"
                    if sampler is not None
                    else ("class_weight" if "balanced" in model_name else "none")
                ),
                "include_sex": include_sex,
                "random_state": random_state,
                "target_col": target_col,
                "task": task,
                "threshold_note": (
                    "fixed at 0.5, imbalanced data (22% default)"
                ),
            }
            tags.update(data_tags)
            mlflow.set_tags(tags)

            mlflow.log_dict(
                {"feature_names": X.columns.tolist()},
                artifact_file="feature_names.json",
            )

            pipe.fit(X, y)
            signature = infer_signature(
                X,
                pipe.predict_proba(X),
            )
            model_info = mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                signature=signature,
                registered_model_name=(
                    None if include_sex else model_name
                ),
            )

            if not include_sex:
                MlflowClient().set_registered_model_alias(
                    name=model_name,
                    alias=alias,
                    version=model_info.registered_model_version,
                )

        logger.info(
            "  → AUC: %.4f ± %.4f",
            summary["roc_auc"][0],
            summary["roc_auc"][1],
        )
        logger.info(
            "  → KS:  %.4f ± %.4f",
            summary["ks"][0],
            summary["ks"][1],
        )
        logger.info(
            "  → Gini: %.4f ± %.4f\n",
            summary["gini"][0],
            summary["gini"][1],
        )


def train_baseline(include_sex: bool = False) -> None:
    """Huấn luyện LR và DT với 5-fold CV."""
    random_state = load_config().get("random_state", 42)

    _train_and_log(
        get_baseline_models(random_state=random_state),
        alias="baseline",
        task="baseline_cv",
        include_sex=include_sex,
    )


def train_rf_xgboost_default(
    include_sex: bool = False,
) -> None:
    """Huấn luyện Random Forest và XGBoost mặc định."""
    random_state = load_config().get("random_state", 42)

    _train_and_log(
        get_advanced_default_models(random_state=random_state),
        alias="default",
        task="rf_xgboost_default_cv",
        include_sex=include_sex,
    )


def train_scorecard(
    include_sex: bool = False,
    enable_feature_selection: bool = False,
) -> None:
    """Huấn luyện Logistic Scorecard (WoE) với 5-fold CV."""
    cfg = load_config()
    random_state = cfg.get("random_state", 42)
    target_col = cfg.get("data", {}).get(
        "target_col",
        "default.payment.next.month",
    )

    train_df, _, _ = load_split_data()
    X, y = build_features(
        train_df,
        target_col=target_col,
        include_sex=include_sex,
    )

    logger.info("=== logistic_scorecard ===")
    logger.info(
        "Train size: %d samples, %d input columns",
        len(X),
        X.shape[1],
    )

    pipe = build_scorecard_pipeline(
        drop_sensitive=not include_sex,
        enable_feature_selection=enable_feature_selection,
    )

    logger.info(
        "Fairness experiment: include_sex=%s, feature_selection=%s",
        include_sex,
        enable_feature_selection,
    )

    fold_metrics = run_cv(
        pipe,
        X,
        y,
        n_splits=N_SPLITS,
        random_state=random_state,
    )
    summary = summarize_folds(fold_metrics)

    logger.info(
        "  → Scorecard AUC: %.4f ± %.4f",
        summary["roc_auc"][0],
        summary["roc_auc"][1],
    )
    logger.info(
        "  → Scorecard KS:  %.4f ± %.4f",
        summary["ks"][0],
        summary["ks"][1],
    )
    logger.info(
        "  → Scorecard Gini: %.4f ± %.4f",
        summary["gini"][0],
        summary["gini"][1],
    )
