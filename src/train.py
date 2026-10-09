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
from src.pipelines import make_pipeline
from src.tracking import setup_mlflow

logger = logging.getLogger(__name__)

# Whitelist các chỉ số được phép log vào MLflow — theo khuyến nghị review PR #16.
# Không bao gồm:
#   - threshold (hằng số 0.5)
#   - tn, fp, fn, tp (phụ thuộc kích thước fold, không có nghĩa thống kê khi tính mean/std)
#   - optimal_threshold, min_expected_cost (bị tối ưu trên chính val fold → lạc quan quá mức)
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
    """Factory tạo dictionary estimator mới, tránh chia sẻ instance bị mutate."""
    return {
        "logreg_baseline": LogisticRegression(max_iter=1000, random_state=random_state),
        "dt_baseline": DecisionTreeClassifier(max_depth=5, random_state=random_state),
    }


def get_advanced_default_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Factory tạo dictionary mô hình nâng cao với tham số mặc định (Tuần 2 - T2: Member B)."""
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    return {
        "rf_default": RandomForestClassifier(random_state=random_state, n_jobs=-1),
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
    """Lấy checksum dataset hoặc hash splits.json để log vào tag MLflow."""
    tags: Dict[str, str] = {}
    base_dir = Path(__file__).resolve().parents[1]

    # 1. Dataset checksum (từ file .sha256 nếu có)
    sha256_path = base_dir / "data" / "raw" / "default-of-credit-card-clients.sha256"
    if sha256_path.exists():
        tags["data_sha256"] = sha256_path.read_text(encoding="utf-8").strip()

    # 2. Hash của splits.json
    splits_path = base_dir / "data" / "splits" / "splits.json"
    if splits_path.exists():
        tags["splits_hash"] = hashlib.sha256(splits_path.read_bytes()).hexdigest()[:16]
        tags["splits_file"] = "data/splits/splits.json"

    return tags


def get_cv_splitter(
    random_state: Optional[int] = None,
    n_splits: int = N_SPLITS,
) -> StratifiedKFold:
    """Tạo bộ chia fold StratifiedKFold cho cross-validation."""
    if random_state is None:
        cfg = load_config()
        random_state = cfg.get("random_state", 42)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def run_cv(
    pipeline,
    X,
    y,
    n_splits: int = N_SPLITS,
    random_state: Optional[int] = None,
) -> List[Dict]:
    """Chạy Stratified K-Fold CV, trả về list metrics mỗi fold.

    Args:
        pipeline: sklearn Pipeline (src.pipelines.make_pipeline).
        X: DataFrame cột gốc (từ build_features).
        y: Series nhãn nhị phân.
        n_splits: Số fold (mặc định 5).
        random_state: Seed cho StratifiedKFold (nếu None sẽ đọc từ config).

    Returns:
        List[Dict]: Mỗi phần tử là dict metrics của 1 fold (có thêm key "fold").
                    Chỉ chứa các key trong REPORT_METRICS (+ "fold").
    """
    skf = get_cv_splitter(random_state=random_state, n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        all_metrics = evaluate_predictions(y_val.to_numpy(), y_proba)
        # Chỉ giữ lại các metric trong whitelist — loại optimal_threshold & min_expected_cost
        metrics = {k: all_metrics[k] for k in REPORT_METRICS if k in all_metrics}
        metrics["fold"] = fold
        fold_metrics.append(metrics)
        logger.info(
            "  Fold %d: AUC=%.4f  KS=%.4f  Gini=%.4f",
            fold,
            metrics["roc_auc"],
            metrics["ks"],
            metrics["gini"],
        )

    return fold_metrics


def summarize_folds(fold_metrics: List[Dict]) -> Dict[str, tuple]:
    """Tổng hợp mean/std (ddof=1) từ danh sách metrics các fold.

    Args:
        fold_metrics: List[Dict] trả về từ run_cv.

    Returns:
        Dict mapping metric_name -> (mean, std) đã round 6 chữ số.
    """
    keys = [k for k in fold_metrics[0] if k != "fold"]
    return {
        k: (
            round(float(np.mean([m[k] for m in fold_metrics])), 6),
            round(float(np.std([m[k] for m in fold_metrics], ddof=1)), 6),
        )
        for k in keys
    }


def _train_and_log(
    models: Dict[str, BaseEstimator],
    alias: str,
    task: str,
    include_sex: bool = False,
    sampler: Optional[BaseEstimator] = None,
    register_model: bool = True,
) -> None:
    """Chạy 5-fold CV cho từng model, log MLflow và đăng ký model với alias.

    include_sex=True: bản đối chiếu fairness (Charter mục 1.3) — run có hậu tố "_with_sex",
    chỉ log metrics & artifact, không đăng ký vào Model Registry.
    sampler: Tuỳ chọn sampler imblearn (như SMOTE) được áp dụng trong từng fold.
    register_model: Nếu False, chỉ log runs và metrics mà không đăng ký vào Model Registry.
    """
    cfg = load_config()
    random_state = cfg.get("random_state", 42)
    target_col = cfg.get("data", {}).get("target_col", "default.payment.next.month")

    setup_mlflow()
    train_df, _, _ = load_split_data()
    X, y = build_features(train_df, target_col=target_col, include_sex=include_sex)

    logger.info("Train size: %d samples, %d input columns", len(X), X.shape[1])

    data_tags = get_data_version_tags()

    for model_name, estimator in models.items():
        run_name = f"{model_name}_with_sex" if include_sex else model_name
        logger.info("=== %s ===", run_name)
        # clone() đảm bảo estimator luôn ở trạng thái mới — tránh mutate instance
        pipe = make_pipeline(
            clone(estimator),
            include_sex=include_sex,
            sampler=clone(sampler) if sampler is not None else None,
        )
        fold_metrics = run_cv(pipe, X, y, n_splits=N_SPLITS, random_state=random_state)

        summary = summarize_folds(fold_metrics)

        with mlflow.start_run(run_name=run_name):
            # 1. Params
            mlflow.log_param("model", model_name)
            mlflow.log_param("n_splits", N_SPLITS)
            mlflow.log_param("n_input_columns", X.shape[1])
            mlflow.log_params(clone(estimator).get_params())
            if sampler is not None:
                mlflow.log_param("sampler", sampler.__class__.__name__)
                mlflow.log_params({f"sampler_{k}": v for k, v in clone(sampler).get_params().items()})

            # 2. Metrics — mean & std (chỉ whitelist REPORT_METRICS)
            for metric, (mean, std) in summary.items():
                mlflow.log_metric(f"{metric}_mean", mean)
                mlflow.log_metric(f"{metric}_std", std)

            # 3. Per-fold metrics (dùng step=fold_number để xem xu hướng trong MLflow UI)
            for m in fold_metrics:
                fold = m["fold"]
                for k, v in m.items():
                    if k != "fold":
                        mlflow.log_metric(f"{k}_fold", v, step=fold)

            # 4. Dataset tags — đọc từ config, checksum, và giải thích ngữ cảnh
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
                "threshold_note": "fixed at 0.5, imbalanced data (22% default)",
            }
            tags.update(data_tags)
            mlflow.set_tags(tags)

            # feature_names log riêng bằng log_dict để tránh giới hạn độ dài của tag
            mlflow.log_dict(
                {"feature_names": X.columns.tolist()},
                artifact_file="feature_names.json",
            )

            # 5. Model artifact — fit lại toàn bộ train set, thêm signature để MLflow biết input schema.
            pipe.fit(X, y)
            signature = infer_signature(X, pipe.predict_proba(X))
            should_register = register_model and not include_sex
            model_info = mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                signature=signature,
                registered_model_name=model_name if should_register else None,
            )
            if should_register and alias:
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
            "  → Gini:%.4f ± %.4f\n",
            summary["gini"][0],
            summary["gini"][1],
        )


def train_baseline(include_sex: bool = False) -> None:
    """Huấn luyện LR và DT với 5-fold CV, đăng ký alias "baseline"."""
    random_state = load_config().get("random_state", 42)
    _train_and_log(
        get_baseline_models(random_state=random_state),
        alias="baseline",
        task="baseline_cv",
        include_sex=include_sex,
    )


def train_rf_xgboost_default(include_sex: bool = False) -> None:
    """Huấn luyện Random Forest và XGBoost với tham số mặc định (Tuần 2 - T2: Member B)."""
    random_state = load_config().get("random_state", 42)
    _train_and_log(
        get_advanced_default_models(random_state=random_state),
        alias="default",
        task="rf_xgboost_default_cv",
        include_sex=include_sex,
    )


def train_boosting_default(include_sex: bool = False) -> None:
    """Huấn luyện LightGBM và CatBoost với tham số mặc định (Tuần 2 - T3: Member B)."""
    random_state = load_config().get("random_state", 42)
    _train_and_log(
        get_boosting_default_models(random_state=random_state),
        alias="default",
        task="boosting_default_cv",
        include_sex=include_sex,
        register_model=True,
    )


def train_imbalance_experiments(include_sex: bool = False) -> None:
    """Thực hiện các thử nghiệm mất cân bằng: class_weight và SMOTE in-fold (Tuần 2 - T3: Member B).

    1. Thử class_weight / auto_class_weights cho LightGBM và CatBoost.
    2. Thí nghiệm phụ: SMOTE đặt bên trong từng fold bằng imblearn.pipeline.Pipeline,
       không resample trước CV.
    """
    from imblearn.over_sampling import SMOTE

    random_state = load_config().get("random_state", 42)

    # 1. Thử nghiệm class_weight / auto_class_weights
    logger.info("=== Bắt đầu thử nghiệm class_weight / auto_class_weights ===")
    _train_and_log(
        get_imbalance_weighted_models(random_state=random_state),
        alias="balanced",
        task="imbalance_class_weight_cv",
        include_sex=include_sex,
        register_model=False,
    )

    # 2. Thí nghiệm phụ: SMOTE đặt bên trong từng fold bằng imblearn.pipeline.Pipeline
    logger.info("=== Bắt đầu thí nghiệm phụ: SMOTE trong từng fold ===")
    smote_models = {
        "lightgbm_smote": get_boosting_default_models(random_state=random_state)["lightgbm_default"],
        "catboost_smote": get_boosting_default_models(random_state=random_state)["catboost_default"],
    }
    _train_and_log(
        smote_models,
        alias="smote",
        task="imbalance_smote_cv",
        include_sex=include_sex,
        sampler=SMOTE(random_state=random_state),
        register_model=False,
    )

