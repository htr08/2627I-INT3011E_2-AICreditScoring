"""Huấn luyện baseline: Logistic Regression và Decision Tree với 5-fold CV trên Train.

Task: Sep 28 + Sep 29 (CoReach) — Baseline CV + Log MLflow.

Chạy:
    python -m scripts.run_pipeline
"""

from typing import Dict, List

import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src.data_split import load_split_data
from src.evaluate import evaluate_predictions
from src.features import build_features, make_pipeline
from src.tracking import setup_mlflow

MODELS: Dict = {
    "logreg_baseline": LogisticRegression(max_iter=1000, random_state=42),
    "dt_baseline": DecisionTreeClassifier(max_depth=5, random_state=42),
}

N_SPLITS = 5


def run_cv(pipeline, X, y, n_splits: int = N_SPLITS) -> List[Dict]:
    """Chạy Stratified K-Fold CV, trả về list metrics mỗi fold.

    Args:
        pipeline: sklearn Pipeline (StandardScaler + estimator).
        X: Feature DataFrame (từ build_features).
        y: Series nhãn nhị phân.
        n_splits: Số fold (mặc định 5).

    Returns:
        List[Dict]: Mỗi phần tử là dict metrics của 1 fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        metrics = evaluate_predictions(y_val.to_numpy(), y_proba)
        metrics["fold"] = fold
        fold_metrics.append(metrics)
        print(f"  Fold {fold}: AUC={metrics['roc_auc']:.4f}  KS={metrics['ks']:.4f}")

    return fold_metrics


def train_baseline() -> None:
    """Huấn luyện LR và DT với 5-fold CV, log kết quả vào MLflow."""
    setup_mlflow()
    train_df, _, _ = load_split_data()
    X, y = build_features(train_df)

    print(f"Train size: {len(X)} samples, {X.shape[1]} features\n")

    for run_name, estimator in MODELS.items():
        print(f"=== {run_name} ===")
        pipe = make_pipeline(estimator)
        fold_metrics = run_cv(pipe, X, y)

        # Tính mean/std trên các fold
        import numpy as np
        keys = [k for k in fold_metrics[0] if k != "fold"]
        summary = {k: (
            round(float(np.mean([m[k] for m in fold_metrics])), 6),
            round(float(np.std([m[k] for m in fold_metrics])), 6),
        ) for k in keys}

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model", run_name)
            mlflow.log_param("n_splits", N_SPLITS)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_params(estimator.get_params())

            for metric, (mean, std) in summary.items():
                mlflow.log_metric(f"{metric}_mean", mean)
                mlflow.log_metric(f"{metric}_std", std)

        print(f"  → AUC: {summary['roc_auc'][0]:.4f} ± {summary['roc_auc'][1]:.4f}")
        print(f"  → KS:  {summary['ks'][0]:.4f} ± {summary['ks'][1]:.4f}")
        print(f"  → Gini:{summary['gini'][0]:.4f} ± {summary['gini'][1]:.4f}\n")


if __name__ == "__main__":
    train_baseline()
