"""Tuning Optuna cho 2 mô hình tốt nhất (CatBoost, LightGBM) với 5-fold Stratified CV.

Task: Tuần 2 – T4 (thành viên B).

- Tiền xử lý không có siêu tham số cần tuning nên được fit một lần trên phần train của mỗi fold
  rồi tái sử dụng cho mọi trial (không leakage: fold val chỉ được transform).
- Monotonic constraints được dựng theo tên cột đầu ra của từng fold, vì số cột one-hot
  có thể khác nhau giữa các fold.
- Điều kiện dừng theo Project Charter mục 1.3 (PlateauStopper + timeout).
- MLflow: một run cha cho mỗi study, mỗi trial là một nested run.

Chạy chính thức:
    python scripts/run_pipeline.py --mode tune
    python scripts/run_pipeline.py --mode tuned --include-sex   # Tuần 2 – T5: phiên bản có SEX
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT, load_config
from src.data_split import load_split_data
from src.evaluate import evaluate_predictions
from src.features import build_features
from src.preprocessing import build_preprocessing_pipeline
from src.train import REPORT_METRICS, get_cv_splitter, get_data_version_tags, summarize_folds
from src.tracking import setup_mlflow

logger = logging.getLogger(__name__)

TUNABLE_MODELS = ("lightgbm", "catboost")
TUNED_PARAMS_PATH = PROJECT_ROOT / "configs" / "tuned_params.yaml"
# Hai ứng viên mô hình cuối (reports/experiments_optuna_tuning.md mục 7).
FINAL_CANDIDATES = ("lightgbm_monotone", "catboost")


# ---------------------------------------------------------------------------
# Không gian tìm kiếm
# ---------------------------------------------------------------------------

def suggest_lightgbm(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def suggest_catboost(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 200, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.01, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


# Trial đầu tiên của mỗi study là tham số mặc định (xấp xỉ với CatBoost vì learning_rate
# mặc định được tự chọn) để bảo đảm kết quả tuning không thấp hơn mốc mặc định của T3.
DEFAULT_TRIAL_PARAMS: Dict[str, Dict[str, Any]] = {
    "lightgbm": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 1e-8,
        "reg_lambda": 1e-8,
    },
    "catboost": {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "subsample": 0.8,
    },
}

SEARCH_SPACES: Dict[str, Callable[[optuna.Trial], Dict[str, Any]]] = {
    "lightgbm": suggest_lightgbm,
    "catboost": suggest_catboost,
}


def build_estimator(
    model_name: str,
    params: Dict[str, Any],
    random_state: int = 42,
    monotone_constraints: Optional[List[int]] = None,
) -> BaseEstimator:
    """Tạo estimator từ bộ tham số Optuna; monotone_constraints là vector theo vị trí cột."""
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        extra = {"monotone_constraints": monotone_constraints} if monotone_constraints else {}
        # subsample_freq=1 để bagging (subsample < 1) thực sự có hiệu lực
        return LGBMClassifier(
            **params, subsample_freq=1, random_state=random_state, n_jobs=-1, verbose=-1, **extra
        )
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        extra = {"monotone_constraints": monotone_constraints} if monotone_constraints else {}
        return CatBoostClassifier(
            **params,
            random_seed=random_state,
            thread_count=-1,
            verbose=0,
            allow_writing_files=False,
            **extra,
        )
    raise ValueError(f"Mô hình không hỗ trợ tuning: {model_name!r}. Chọn một trong {TUNABLE_MODELS}.")


def monotone_vector(feature_names: Sequence[str], increasing: Sequence[str]) -> List[int]:
    """Vector ràng buộc theo vị trí cột đầu ra ColumnTransformer ('numeric__PAY_MAX' → +1).

    Chỉ khớp khối numeric: biến one-hot ('categorical__PAY_1_2') không ràng buộc được.
    """
    targets = {f"numeric__{name}" for name in increasing}
    return [1 if name in targets else 0 for name in feature_names]


# ---------------------------------------------------------------------------
# Cross-validation trên fold đã tiền xử lý sẵn
# ---------------------------------------------------------------------------

def prepare_folds(
    X, y, n_splits: int = 5, random_state: int = 42, include_sex: bool = False
) -> List[Dict[str, Any]]:
    """Fit tiền xử lý trên phần train của từng fold, transform phần val; trả về cache dùng chung.

    Cùng StratifiedKFold (random_state) với run_cv nên kết quả so sánh được với T2/T3. Chia fold chỉ
    phụ thuộc y, nên bản có và không có SEX dùng đúng cùng 5 fold (so sánh từng cặp được).
    include_sex: True giữ SEX (one-hot) cho bản đối chiếu fairness (Charter mục 1.3).
    """
    folds = []
    for fold, (tr_idx, val_idx) in enumerate(
        get_cv_splitter(random_state=random_state, n_splits=n_splits).split(X, y), start=1
    ):
        prep = build_preprocessing_pipeline(drop_sensitive=not include_sex).fit(
            X.iloc[tr_idx], y.iloc[tr_idx]
        )
        folds.append({
            "fold": fold,
            "X_train": prep.transform(X.iloc[tr_idx]),
            "y_train": y.iloc[tr_idx].to_numpy(),
            "X_val": prep.transform(X.iloc[val_idx]),
            "y_val": y.iloc[val_idx].to_numpy(),
            "feature_names": list(prep.named_steps["preprocessor"].get_feature_names_out()),
        })
    return folds


def cv_score(
    model_name: str,
    params: Dict[str, Any],
    folds: List[Dict[str, Any]],
    monotone_features: Optional[Sequence[str]] = None,
    random_state: int = 42,
    trial: Optional[optuna.Trial] = None,
) -> List[Dict[str, float]]:
    """Đánh giá một bộ tham số trên các fold đã cache; báo AUC trung bình tích lũy cho pruner."""
    fold_metrics = []
    for i, f in enumerate(folds):
        constraints = (
            monotone_vector(f["feature_names"], monotone_features) if monotone_features else None
        )
        clf = build_estimator(model_name, params, random_state, constraints)
        clf.fit(f["X_train"], f["y_train"])
        y_proba = clf.predict_proba(f["X_val"])[:, 1]

        all_metrics = evaluate_predictions(f["y_val"], y_proba)
        metrics = {k: all_metrics[k] for k in REPORT_METRICS if k in all_metrics}
        metrics["fold"] = f["fold"]
        fold_metrics.append(metrics)

        if trial is not None:
            trial.report(float(np.mean([m["roc_auc"] for m in fold_metrics])), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return fold_metrics


# ---------------------------------------------------------------------------
# Điều kiện dừng (Project Charter mục 1.3)
# ---------------------------------------------------------------------------

class PlateauStopper:
    """Dừng study khi `patience` trial liên tiếp không vượt mốc tốt nhất quá `min_delta`.

    Mốc chỉ được dời khi có cải thiện > min_delta, nên các cải thiện nhỏ cộng dồn
    vượt min_delta vẫn được ghi nhận. Trial bị prune được tính là không cải thiện.
    """

    def __init__(self, patience: int = 30, min_delta: float = 0.002):
        self.patience = patience
        self.min_delta = min_delta
        self.anchor = -np.inf
        self.stale = 0
        self.triggered = False

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value > self.anchor + self.min_delta:
            self.anchor = trial.value
            self.stale = 0
        elif trial.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED):
            self.stale += 1
        if self.stale >= self.patience:
            self.triggered = True
            study.stop()


# ---------------------------------------------------------------------------
# Study + MLflow
# ---------------------------------------------------------------------------

def _log_cv_summary(fold_metrics: List[Dict[str, float]], prefix: str = "") -> Dict[str, tuple]:
    summary = summarize_folds(fold_metrics)
    for metric, (mean, std) in summary.items():
        mlflow.log_metric(f"{prefix}{metric}_mean", mean)
        mlflow.log_metric(f"{prefix}{metric}_std", std)
    return summary


def _fit_final_pipeline(
    model_name: str,
    params: Dict[str, Any],
    X,
    y,
    monotone_features: Optional[Sequence[str]],
    random_state: int,
    include_sex: bool = False,
) -> Pipeline:
    """Fit tiền xử lý + mô hình trên toàn bộ Train; clf là estimator gốc (dùng được TreeExplainer)."""
    prep = build_preprocessing_pipeline(drop_sensitive=not include_sex).fit(X, y)
    names = prep.named_steps["preprocessor"].get_feature_names_out()
    constraints = monotone_vector(names, monotone_features) if monotone_features else None
    clf = build_estimator(model_name, params, random_state, constraints)
    clf.fit(prep.transform(X), y)
    return Pipeline([("preprocess", prep), ("clf", clf)])


def tune_model(
    model_name: str,
    monotone: bool = False,
    max_trials: Optional[int] = None,
    timeout: Optional[float] = None,
    check_monotone: bool = True,
    register_model: bool = True,
) -> Dict[str, Any]:
    """Chạy một study Optuna cho `model_name`, log MLflow dạng nested runs và đăng ký mô hình tốt nhất.

    Args:
        model_name: "lightgbm" hoặc "catboost".
        monotone: True → mọi trial dùng monotonic constraints (config tuning.monotone_increasing).
        max_trials / timeout: ghi đè config (dùng cho chạy thử nhanh).
        check_monotone: với study không ràng buộc, chạy thêm 1 lượt CV bộ tham số tốt nhất
            có ràng buộc để đối chiếu.
        register_model: đăng ký vào Model Registry với alias "tuned".

    Returns:
        Dict tóm tắt: best_params, cv_summary, n_trials, stop_reason, ...
    """
    cfg = load_config()
    tcfg = cfg["tuning"]
    random_state = cfg.get("random_state", 42)
    target_col = cfg.get("data", {}).get("target_col", "default.payment.next.month")
    max_trials = max_trials or tcfg["max_trials"]
    timeout = timeout or tcfg["timeout_seconds"]
    increasing = tcfg["monotone_increasing"]
    monotone_features = increasing if monotone else None
    suggest = SEARCH_SPACES[model_name]

    setup_mlflow()
    train_df, _, _ = load_split_data()
    X, y = build_features(train_df, target_col=target_col)
    folds = prepare_folds(X, y, n_splits=tcfg["n_splits"], random_state=random_state)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}{'_monotone' if monotone else ''}",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=tcfg["pruner_startup_trials"],
            n_warmup_steps=tcfg["pruner_warmup_folds"],
        ),
    )
    study.enqueue_trial(DEFAULT_TRIAL_PARAMS[model_name])
    stopper = PlateauStopper(patience=tcfg["patience"], min_delta=tcfg["min_delta"])
    fold_cache: Dict[int, List[Dict[str, float]]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        start = time.time()
        pruned = False
        with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
            mlflow.log_params(params)
            try:
                fold_metrics = cv_score(
                    model_name, params, folds, monotone_features, random_state, trial
                )
            except optuna.TrialPruned:
                pruned = True
            mlflow.set_tag("trial_state", "pruned" if pruned else "complete")
            mlflow.log_metric("fit_seconds", round(time.time() - start, 2))
            if not pruned:
                summary = _log_cv_summary(fold_metrics)
        if pruned:
            raise optuna.TrialPruned()
        fold_cache[trial.number] = fold_metrics
        logger.info(
            "  trial %03d: AUC=%.4f ± %.4f (%.0fs)",
            trial.number, summary["roc_auc"][0], summary["roc_auc"][1], time.time() - start,
        )
        return summary["roc_auc"][0]

    run_name = f"optuna_{study.study_name}"
    logger.info("=== %s (patience=%d, min_delta=%.3f, timeout=%ds) ===",
                run_name, stopper.patience, stopper.min_delta, timeout)
    t0 = time.time()
    with mlflow.start_run(run_name=run_name):
        study.optimize(objective, n_trials=max_trials, timeout=timeout, callbacks=[stopper])
        elapsed = time.time() - t0

        states = [t.state for t in study.trials]
        n_complete = states.count(optuna.trial.TrialState.COMPLETE)
        n_pruned = states.count(optuna.trial.TrialState.PRUNED)
        if stopper.triggered:
            stop_reason = "plateau"
        elif elapsed >= timeout:
            stop_reason = "timeout"
        else:
            stop_reason = "max_trials"

        best = study.best_trial
        best_folds = fold_cache[best.number]
        default_folds = fold_cache.get(0)

        mlflow.log_params({
            "model": model_name,
            "monotone": monotone,
            "n_splits": tcfg["n_splits"],
            "patience": stopper.patience,
            "min_delta": stopper.min_delta,
            "timeout_seconds": timeout,
            "sampler": "TPESampler",
            "pruner": "MedianPruner",
        })
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        cv_summary = _log_cv_summary(best_folds)
        for m in best_folds:
            mlflow.log_metric("roc_auc_fold", m["roc_auc"], step=m["fold"])
        if default_folds is not None:
            mlflow.log_metric("default_trial_roc_auc_mean", summarize_folds(default_folds)["roc_auc"][0])
        mlflow.log_metrics({
            "n_trials": len(study.trials),
            "n_complete": n_complete,
            "n_pruned": n_pruned,
            "best_trial_number": best.number,
            "elapsed_seconds": round(elapsed, 1),
        })

        tags = {
            "task": "optuna_tuning_cv",
            "feature_strategy": "feature_freeze_v1",
            "imbalance_strategy": "none",
            "include_sex": False,
            "random_state": random_state,
            "stop_reason": stop_reason,
            "train_size": len(X),
        }
        if monotone:
            tags["monotone_features"] = ",".join(increasing)
        tags.update(get_data_version_tags())
        mlflow.set_tags(tags)
        mlflow.log_text(study.trials_dataframe().to_csv(index=False), "optuna_trials.csv")

        monotone_check = None
        if check_monotone and not monotone:
            logger.info("  Đối chiếu: bộ tham số tốt nhất + monotonic constraints")
            with mlflow.start_run(run_name="best_params_monotone_check", nested=True):
                mlflow.log_params(best.params)
                mlflow.set_tag("monotone_features", ",".join(increasing))
                check_start = time.time()
                check_folds = cv_score(model_name, best.params, folds, increasing, random_state)
                monotone_check = _log_cv_summary(check_folds)
                mlflow.log_metric("fit_seconds", round(time.time() - check_start, 2))
            mlflow.log_metric("monotone_check_roc_auc_mean", monotone_check["roc_auc"][0])
            mlflow.log_metric("monotone_check_roc_auc_std", monotone_check["roc_auc"][1])

        # Mô hình cuối của study: fit lại trên toàn bộ Train với bộ tham số tốt nhất.
        pipe = _fit_final_pipeline(model_name, best.params, X, y, monotone_features, random_state)
        registered_name = f"{study.study_name}_tuned"
        model_info = mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=infer_signature(X, pipe.predict_proba(X)),
            registered_model_name=registered_name if register_model else None,
        )
        if register_model:
            MlflowClient().set_registered_model_alias(
                name=registered_name, alias="tuned", version=model_info.registered_model_version
            )

    logger.info(
        "  → best trial %d: AUC %.4f ± %.4f | %d trial (%d pruned), %.0fs, dừng: %s",
        best.number, cv_summary["roc_auc"][0], cv_summary["roc_auc"][1],
        len(study.trials), n_pruned, elapsed, stop_reason,
    )
    return {
        "study_name": study.study_name,
        "best_params": best.params,
        "best_trial": best.number,
        "cv_summary": cv_summary,
        "default_summary": summarize_folds(default_folds) if default_folds else None,
        "monotone_check": monotone_check,
        "n_trials": len(study.trials),
        "n_pruned": n_pruned,
        "elapsed_seconds": elapsed,
        "stop_reason": stop_reason,
    }


def tune_best_models(max_trials: Optional[int] = None, timeout: Optional[float] = None) -> List[Dict]:
    """Kế hoạch tuning T4: LightGBM (không ràng buộc + có ràng buộc), CatBoost (không ràng buộc
    + đối chiếu ràng buộc trên bộ tham số tốt nhất, do monotonic constraints làm CatBoost chậm ~5 lần).
    """
    return [
        tune_model("lightgbm", monotone=False, max_trials=max_trials, timeout=timeout),
        tune_model("lightgbm", monotone=True, max_trials=max_trials, timeout=timeout),
        tune_model("catboost", monotone=False, max_trials=max_trials, timeout=timeout),
    ]


# ---------------------------------------------------------------------------
# Huấn luyện lại mô hình đã tuning (không chạy Optuna) — Tuần 2 – T5
# ---------------------------------------------------------------------------

def load_tuned_params(study_name: str, path=TUNED_PARAMS_PATH) -> Dict[str, Any]:
    """Đọc model, monotone, params của một study từ configs/tuned_params.yaml."""
    with open(path, encoding="utf-8") as f:
        tuned = yaml.safe_load(f)
    if study_name not in tuned:
        raise KeyError(f"Không có study {study_name!r} trong {path}. Có: {sorted(tuned)}")
    return tuned[study_name]


def sex_importance_share(pipe: Pipeline) -> float:
    """Tỷ trọng tầm quan trọng (gain / PredictionValuesChange) của các cột SEX trong mô hình đã fit."""
    names = list(pipe.named_steps["preprocess"].named_steps["preprocessor"].get_feature_names_out())
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "booster_"):
        importance = clf.booster_.feature_importance(importance_type="gain")
    else:
        importance = clf.get_feature_importance()
    importance = np.asarray(importance, dtype=float)
    is_sex = np.array([n.startswith("categorical__SEX_") for n in names])
    return float(importance[is_sex].sum() / importance.sum()) if importance.sum() > 0 else 0.0


def train_tuned(
    studies: Sequence[str] = FINAL_CANDIDATES,
    include_sex: bool = True,
) -> List[Dict[str, Any]]:
    """Huấn luyện lại các mô hình đã tuning với bộ tham số trong configs/tuned_params.yaml.

    include_sex=True (Tuần 2 – T5): phiên bản có SEX để đối chiếu fairness. CV 5-fold trên cùng
    fold với bản không có SEX, log chênh lệch theo từng fold (sex_effect_*) và tỷ trọng tầm quan
    trọng của SEX. Mô hình fit trên toàn bộ Train được log artifact, không đăng ký Model Registry
    (cùng quy ước với các run _with_sex khác).
    """
    cfg = load_config()
    random_state = cfg.get("random_state", 42)
    target_col = cfg.get("data", {}).get("target_col", "default.payment.next.month")
    increasing = cfg["tuning"]["monotone_increasing"]
    n_splits = cfg["tuning"]["n_splits"]

    setup_mlflow()
    train_df, _, _ = load_split_data()
    X_base, y = build_features(train_df, target_col=target_col, include_sex=False)
    folds_base = prepare_folds(X_base, y, n_splits=n_splits, random_state=random_state)
    if include_sex:
        X, _ = build_features(train_df, target_col=target_col, include_sex=True)
        folds = prepare_folds(X, y, n_splits=n_splits, random_state=random_state, include_sex=True)
    else:
        X, folds = X_base, folds_base

    results = []
    for study_name in studies:
        tuned = load_tuned_params(study_name)
        model_name, params = tuned["model"], tuned["params"]
        monotone_features = increasing if tuned["monotone"] else None
        run_name = f"{study_name}_tuned" + ("_with_sex" if include_sex else "")
        logger.info("=== %s ===", run_name)

        start = time.time()
        fold_metrics = cv_score(model_name, params, folds, monotone_features, random_state)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model": model_name,
                "study": study_name,
                "monotone": tuned["monotone"],
                "n_splits": n_splits,
                **params,
            })
            summary = _log_cv_summary(fold_metrics)
            for m in fold_metrics:
                mlflow.log_metric("roc_auc_fold", m["roc_auc"], step=m["fold"])

            result = {"study": study_name, "include_sex": include_sex, "cv_summary": summary}
            if include_sex:
                base_metrics = cv_score(model_name, params, folds_base, monotone_features, random_state)
                base_summary = _log_cv_summary(base_metrics, prefix="without_sex_")
                diff = np.array([a["roc_auc"] - b["roc_auc"] for a, b in zip(fold_metrics, base_metrics)])
                mlflow.log_metrics({
                    "sex_effect_roc_auc_mean": float(diff.mean()),
                    "sex_effect_roc_auc_min": float(diff.min()),
                    "sex_effect_roc_auc_max": float(diff.max()),
                })
                result.update(without_sex_summary=base_summary, sex_effect_folds=diff.round(6).tolist())

            pipe = _fit_final_pipeline(
                model_name, params, X, y, monotone_features, random_state, include_sex=include_sex
            )
            if include_sex:
                share = sex_importance_share(pipe)
                mlflow.log_metric("sex_importance_share", share)
                result["sex_importance_share"] = share
            mlflow.log_metric("fit_seconds", round(time.time() - start, 1))

            tags = {
                "task": "tuned_with_sex_cv" if include_sex else "tuned_cv",
                "feature_strategy": "feature_freeze_v1",
                "imbalance_strategy": "none",
                "include_sex": include_sex,
                "random_state": random_state,
                "train_size": len(X),
                "params_source": "configs/tuned_params.yaml",
            }
            tags.update(get_data_version_tags())
            mlflow.set_tags(tags)
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                signature=infer_signature(X, pipe.predict_proba(X)),
            )

        logger.info("  → AUC: %.4f ± %.4f", *summary["roc_auc"])
        if include_sex:
            logger.info(
                "  → chênh lệch AUC có/không SEX theo fold: %s (mean %+.4f), tỷ trọng SEX %.2f%%",
                np.round(diff, 4), diff.mean(), 100 * share,
            )
        results.append(result)
    return results
