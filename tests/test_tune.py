"""Unit tests cho src/tune.py (Optuna tuning, Tuần 2 – T4)."""

import sys
from unittest.mock import MagicMock

import numpy as np
import optuna
import pytest

from conftest import make_raw_credit_df
from src.features import build_features
from src.tune import (
    DEFAULT_TRIAL_PARAMS,
    SEARCH_SPACES,
    PlateauStopper,
    build_estimator,
    cv_score,
    monotone_vector,
    prepare_folds,
    tune_model,
)


@pytest.fixture(scope="module")
def dummy_folds():
    X, y = build_features(make_raw_credit_df(n=600, seed=42))
    return prepare_folds(X, y, n_splits=3, random_state=42)


# 1. Không gian tìm kiếm
@pytest.mark.parametrize("model_name", ["lightgbm", "catboost"])
def test_default_trial_inside_search_space(model_name):
    """Trial mặc định được enqueue phải nằm trong không gian tìm kiếm (Optuna không báo lỗi)."""
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(DEFAULT_TRIAL_PARAMS[model_name])
    trial = study.ask()
    params = SEARCH_SPACES[model_name](trial)
    assert params == pytest.approx(DEFAULT_TRIAL_PARAMS[model_name])


def test_build_estimator_rejects_unknown_model():
    with pytest.raises(ValueError, match="không hỗ trợ"):
        build_estimator("xgboost", {})


# 2. Monotonic constraints
def test_monotone_vector_only_matches_numeric_block():
    names = ["categorical__PAY_1_2", "numeric__PAY_MAX", "numeric__UTIL_MEAN", "numeric__LIMIT_BAL"]
    assert monotone_vector(names, ["PAY_MAX", "UTIL_MEAN", "PAY_1"]) == [0, 1, 1, 0]


def test_prepare_folds_no_val_leakage(dummy_folds):
    """Mỗi fold có số cột khớp tên đặc trưng và tổng số dòng val bằng toàn bộ dữ liệu."""
    assert sum(len(f["y_val"]) for f in dummy_folds) == 600
    for f in dummy_folds:
        assert f["X_train"].shape[1] == f["X_val"].shape[1] == len(f["feature_names"])


@pytest.mark.parametrize("model_name", ["lightgbm", "catboost"])
def test_monotone_constraint_respected(model_name, dummy_folds):
    """Tăng PAY_MAX (giữ nguyên các cột khác) không được làm giảm PD dự đoán."""
    f = dummy_folds[0]
    small = {"lightgbm": {"n_estimators": 50}, "catboost": {"iterations": 50}}[model_name]
    cons = monotone_vector(f["feature_names"], ["PAY_MAX"])
    clf = build_estimator(model_name, small, monotone_constraints=cons).fit(f["X_train"], f["y_train"])

    col = f["feature_names"].index("numeric__PAY_MAX")
    grid = np.linspace(f["X_train"][:, col].min(), f["X_train"][:, col].max(), 20)
    rows = np.repeat(f["X_val"][:50], len(grid), axis=0)
    rows[:, col] = np.tile(grid, 50)
    proba = clf.predict_proba(rows)[:, 1].reshape(50, len(grid))
    assert np.all(np.diff(proba, axis=1) >= -1e-9)


def test_cv_score_returns_report_metrics(dummy_folds):
    metrics = cv_score("lightgbm", {"n_estimators": 30}, dummy_folds)
    assert len(metrics) == 3
    assert all(0.5 < m["roc_auc"] <= 1.0 for m in metrics)
    assert "optimal_threshold" not in metrics[0]


# 3. Điều kiện dừng
def _run_stopper(values, patience=3, min_delta=0.002):
    stopper = PlateauStopper(patience=patience, min_delta=min_delta)
    study = optuna.create_study(direction="maximize")
    it = iter(values)
    study.optimize(lambda t: next(it), n_trials=len(values), callbacks=[stopper])
    return stopper, len(study.trials)


def test_plateau_stopper_stops_after_patience():
    stopper, n = _run_stopper([0.70, 0.701, 0.7015, 0.701, 0.80, 0.80])
    assert stopper.triggered and n == 4


def test_plateau_stopper_accumulated_small_gains_reset():
    """Các cải thiện nhỏ cộng dồn vượt min_delta so với mốc phải reset bộ đếm."""
    stopper, n = _run_stopper([0.70, 0.701, 0.7025, 0.703, 0.7035, 0.705])
    assert not stopper.triggered and n == 6


# 4. Tích hợp MLflow (nested runs + Model Registry)
def test_tune_model_mlflow_nested_runs(tmp_path, monkeypatch):
    from mlflow.tracking import MlflowClient

    tracking_uri = tmp_path.as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    df = make_raw_credit_df(n=500, seed=7)
    monkeypatch.setattr("src.tune.load_split_data", lambda: (df, None, None))

    result = tune_model("lightgbm", max_trials=3, timeout=600)

    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name("credit_scoring")
    runs = client.search_runs([exp.experiment_id])
    parent = [r for r in runs if r.info.run_name == "optuna_lightgbm"]
    assert len(parent) == 1
    children = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == parent[0].info.run_id]
    # 3 trial + 1 lượt đối chiếu monotonic
    assert len(children) == 4
    assert {"roc_auc_mean", "roc_auc_std", "n_trials"} <= set(parent[0].data.metrics)
    assert parent[0].data.tags["stop_reason"] == "max_trials"
    assert result["n_trials"] == 3 and result["monotone_check"] is not None
    assert client.get_model_version_by_alias("lightgbm_tuned", "tuned") is not None


def test_run_pipeline_cli_dispatch_tune(monkeypatch):
    from scripts.run_pipeline import main

    mock_tune = MagicMock()
    monkeypatch.setattr("scripts.run_pipeline.tune_model", mock_tune)
    monkeypatch.setattr(
        sys, "argv",
        ["run_pipeline.py", "--mode", "tune", "--tune-model", "catboost", "--monotone", "--max-trials", "5"],
    )
    main()
    mock_tune.assert_called_once_with("catboost", monotone=True, max_trials=5, timeout=None)


# 5. Tuần 2 – T5: mô hình đã tuning, phiên bản có SEX
def test_tuned_params_config_matches_search_space():
    """configs/tuned_params.yaml có đủ 2 ứng viên cuối, tham số khớp không gian tìm kiếm."""
    from src.tune import FINAL_CANDIDATES, load_tuned_params

    for study in FINAL_CANDIDATES:
        tuned = load_tuned_params(study)
        assert set(tuned["params"]) == set(DEFAULT_TRIAL_PARAMS[tuned["model"]])


def test_prepare_folds_with_sex_same_splits():
    """Bản có SEX dùng đúng cùng fold với bản không có SEX và thêm cột SEX one-hot."""
    df = make_raw_credit_df(n=600, seed=42)
    X0, y = build_features(df)
    X1, _ = build_features(df, include_sex=True)
    base = prepare_folds(X0, y, n_splits=3)
    with_sex = prepare_folds(X1, y, n_splits=3, include_sex=True)
    for a, b in zip(base, with_sex):
        np.testing.assert_array_equal(a["y_val"], b["y_val"])
        assert not any(n.startswith("categorical__SEX_") for n in a["feature_names"])
        assert any(n.startswith("categorical__SEX_") for n in b["feature_names"])


def test_train_tuned_with_sex_mlflow(tmp_path, monkeypatch):
    """train_tuned(include_sex=True) log run _with_sex, chênh lệch theo fold, không đăng ký model."""
    from mlflow.tracking import MlflowClient

    from src.tune import train_tuned

    tracking_uri = tmp_path.as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    df = make_raw_credit_df(n=500, seed=7)
    monkeypatch.setattr("src.tune.load_split_data", lambda: (df, None, None))
    monkeypatch.setattr(
        "src.tune.load_tuned_params",
        lambda study: {"model": "lightgbm", "monotone": True, "params": {"n_estimators": 30}},
    )

    results = train_tuned(studies=["lightgbm_monotone"], include_sex=True)

    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name("credit_scoring")
    run = client.search_runs([exp.experiment_id])[0]
    assert run.info.run_name == "lightgbm_monotone_tuned_with_sex"
    assert {"roc_auc_mean", "without_sex_roc_auc_mean", "sex_effect_roc_auc_mean", "sex_importance_share"} <= set(
        run.data.metrics
    )
    assert run.data.tags["include_sex"] == "True"
    assert client.search_registered_models() == []
    assert len(results[0]["sex_effect_folds"]) == 5
    assert 0.0 <= results[0]["sex_importance_share"] <= 1.0


def test_run_pipeline_cli_dispatch_tuned_with_sex(monkeypatch):
    from scripts.run_pipeline import main

    mock_train = MagicMock()
    monkeypatch.setattr("scripts.run_pipeline.train_tuned", mock_train)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--mode", "tuned", "--include-sex"])
    main()
    mock_train.assert_called_once_with(include_sex=True)


def test_fit_tuned_pipeline_matches_config(monkeypatch):
    """fit_tuned_pipeline dựng đúng estimator và ràng buộc đơn điệu từ tuned_params."""
    from src.tune import fit_tuned_pipeline

    monkeypatch.setattr(
        "src.tune.load_tuned_params",
        lambda study: {"model": "lightgbm", "monotone": True, "params": {"n_estimators": 20}},
    )
    X, y = build_features(make_raw_credit_df(n=400, seed=1))
    pipe = fit_tuned_pipeline("lightgbm_monotone", X, y)

    clf = pipe.named_steps["clf"]
    names = list(pipe.named_steps["preprocess"].named_steps["preprocessor"].get_feature_names_out())
    assert clf.n_estimators == 20
    assert clf.monotone_constraints[names.index("numeric__PAY_MAX")] == 1
    assert pipe.predict_proba(X).shape == (len(X), 2)
