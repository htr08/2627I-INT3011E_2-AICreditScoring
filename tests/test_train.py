"""Unit tests cho src/train.py và src/features.py.

Theo phản hồi review PR #16:
- Fixture dùng np.random.default_rng(42) cục bộ thay vì np.random.seed()
- Nhãn sinh có tương quan với PAY_0 và LIMIT_BAL (fixture đúng schema CSV gốc) để assert AUC > 0.6
- Parametrize kiểm thử đồng thời cho LogisticRegression và DecisionTreeClassifier
- Kiểm tra whitelist REPORT_METRICS: không lọt optimal_threshold hay min_expected_cost
- Kiểm tra std mẫu (ddof=1)
- Test build_features loại ID, target và báo KeyError khi thiếu target
- Test tính phân tầng (Stratify): tỷ lệ default ở mỗi fold xấp xỉ tỷ lệ tổng
- Test tính tất định: chạy run_cv hai lần phải ra cùng kết quả
- Test train_baseline với MLflow: dùng tmp_path làm tracking URI, kiểm tra đủ 2 run và metrics
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from conftest import make_raw_credit_df
from src.features import build_features
from src.pipelines import make_pipeline
from src.train import REPORT_METRICS, get_cv_splitter, run_cv, summarize_folds, train_baseline


@pytest.fixture
def dummy_train_df():
    """500 mẫu đúng schema CSV gốc (có PAY_0, SEX) với nhãn có tín hiệu từ PAY_0 và LIMIT_BAL."""
    return make_raw_credit_df(n=500, seed=42)


# 1. Test build_features
def test_build_features_drops_id_and_target_and_raises_keyerror(dummy_train_df):
    """build_features phải loại bỏ cột ID và target, và báo KeyError khi thiếu target."""
    X, y = build_features(dummy_train_df)
    assert "ID" not in X.columns
    assert "default.payment.next.month" not in X.columns
    assert "LIMIT_BAL" in X.columns
    assert len(y) == len(dummy_train_df)
    assert issubclass(y.dtype.type, (np.integer, int))

    # Báo KeyError khi thiếu cột target
    df_missing_target = dummy_train_df.drop(columns=["default.payment.next.month"])
    with pytest.raises(KeyError, match="Không tìm thấy cột nhãn"):
        build_features(df_missing_target)


def test_make_pipeline_uses_preprocessing_on_raw_columns(dummy_train_df):
    """make_pipeline nhận cột gốc (PAY_0), dùng tiền xử lý của preprocessing.py và encode thêm SEX khi bật."""
    X, y = build_features(dummy_train_df)

    pipe = make_pipeline(LogisticRegression(max_iter=300)).fit(X, y)
    pipe_sex = make_pipeline(LogisticRegression(max_iter=300), include_sex=True).fit(X, y)

    assert "preprocess" in pipe.named_steps
    assert pipe_sex.named_steps["clf"].coef_.shape[1] > pipe.named_steps["clf"].coef_.shape[1]


# 2. Fixture và CV tests cho cả 2 model
@pytest.fixture(
    params=[
        LogisticRegression(max_iter=300, random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
    ],
    ids=["logreg", "decision_tree"],
)
def estimator(request):
    """Fixture cung cấp instance mới (clone) cho từng model và từng test."""
    return clone(request.param)


def test_run_cv_returns_5_folds(dummy_train_df, estimator):
    """Số fold trả về phải đúng với n_splits."""
    X, y = build_features(dummy_train_df)
    pipe = make_pipeline(estimator)
    metrics = run_cv(pipe, X, y, n_splits=5)
    assert len(metrics) == 5


def test_run_cv_only_report_metrics_whitelist(dummy_train_df, estimator):
    """Mỗi fold chỉ chứa whitelist REPORT_METRICS (+ 'fold').

    Loại bỏ optimal_threshold, min_expected_cost và threshold hằng số.
    """
    X, y = build_features(dummy_train_df)
    pipe = make_pipeline(estimator)
    metrics = run_cv(pipe, X, y, n_splits=5)
    allowed_keys = set(REPORT_METRICS) | {"fold"}

    for m in metrics:
        unexpected = set(m.keys()) - allowed_keys
        assert not unexpected, f"Fold chứa key ngoài whitelist: {unexpected}"
        assert "optimal_threshold" not in m
        assert "min_expected_cost" not in m
        assert "threshold" not in m


def test_run_cv_auc_signal(dummy_train_df, estimator):
    """Nhãn có tín hiệu (phụ thuộc PAY_1) nên AUC phải đạt > 0.6 ở mọi fold."""
    X, y = build_features(dummy_train_df)
    pipe = make_pipeline(estimator)
    metrics = run_cv(pipe, X, y, n_splits=5)
    for m in metrics:
        assert m["roc_auc"] > 0.6, f"Fold {m.get('fold')} có AUC quá thấp: {m['roc_auc']}"


def test_run_cv_required_metrics_present(dummy_train_df, estimator):
    """Các metrics chính trong credit scoring phải có mặt đầy đủ."""
    X, y = build_features(dummy_train_df)
    pipe = make_pipeline(estimator)
    metrics = run_cv(pipe, X, y, n_splits=5)
    for m in metrics:
        for required in ("roc_auc", "ks", "gini", "pr_auc", "brier_score"):
            assert required in m


# 3. Test tính phân tầng (Stratified)
def test_stratified_folds_default_rate(dummy_train_df):
    """Tỷ lệ default ở mỗi fold phải xấp xỉ tỷ lệ default của toàn bộ dữ liệu."""
    X, y = build_features(dummy_train_df)
    total_default_rate = float(y.mean())
    skf = get_cv_splitter(random_state=42)
    assert isinstance(skf, StratifiedKFold)

    for _, val_idx in skf.split(X, y):
        fold_default_rate = float(y.iloc[val_idx].mean())
        # Chênh lệch tỷ lệ ở mỗi fold không vượt quá 2% so với tổng thể
        assert abs(fold_default_rate - total_default_rate) < 0.02


# 4. Test tính tất định (Determinism)
def test_run_cv_determinism(dummy_train_df):
    """Chạy run_cv hai lần với cùng random_state phải cho kết quả giống hệt nhau."""
    X, y = build_features(dummy_train_df)
    clf1 = LogisticRegression(max_iter=300, random_state=42)
    clf2 = LogisticRegression(max_iter=300, random_state=42)

    metrics_run1 = run_cv(make_pipeline(clf1), X, y, n_splits=5, random_state=42)
    metrics_run2 = run_cv(make_pipeline(clf2), X, y, n_splits=5, random_state=42)

    assert len(metrics_run1) == len(metrics_run2)
    for m1, m2 in zip(metrics_run1, metrics_run2):
        assert m1["fold"] == m2["fold"]
        for key in REPORT_METRICS:
            if key in m1:
                assert m1[key] == pytest.approx(m2[key], abs=1e-7)


# 5. Test std mẫu (ddof=1) — kiểm tra summarize_folds dùng ddof=1 thật sự
def test_summarize_folds_uses_ddof1():
    """summarize_folds phải tính std với ddof=1 (std mẫu), kiểm tra bằng dữ liệu cố định."""
    # 3 fold giả với giá trị đã biết trước
    fake_folds = [
        {"roc_auc": 0.7, "fold": 1},
        {"roc_auc": 0.8, "fold": 2},
        {"roc_auc": 0.9, "fold": 3},
    ]
    summary = summarize_folds(fake_folds)
    expected_mean = round(float(np.mean([0.7, 0.8, 0.9])), 6)
    expected_std = round(float(np.std([0.7, 0.8, 0.9], ddof=1)), 6)
    assert summary["roc_auc"][0] == pytest.approx(expected_mean, abs=1e-6)
    assert summary["roc_auc"][1] == pytest.approx(expected_std, abs=1e-6)
    # ddof=1 phải cho std lớn hơn ddof=0
    std_ddof0 = round(float(np.std([0.7, 0.8, 0.9], ddof=0)), 6)
    assert summary["roc_auc"][1] > std_ddof0


# 6. Test train_baseline tích hợp MLflow với tmp_path
def test_train_baseline_mlflow(tmp_path, dummy_train_df, monkeypatch):
    """train_baseline chạy đủ 2 run (LR, DT), log đủ metrics whitelist, tags và artifacts."""
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = tmp_path.as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setattr("src.train.load_split_data", lambda: (dummy_train_df, None, None))

    train_baseline()

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("credit_scoring")
    assert experiment is not None

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 2

    run_names = {r.data.tags.get("mlflow.runName") for r in runs}
    assert run_names == {"logreg_baseline", "dt_baseline"}

    for r in runs:
        # Kiểm tra metrics whitelist
        for metric in ("roc_auc_mean", "ks_mean", "gini_mean", "pr_auc_mean"):
            assert metric in r.data.metrics
            assert 0.0 <= r.data.metrics[metric] <= 1.0

        # Kiểm tra tags quan trọng
        assert "threshold_note" in r.data.tags
        assert "train_size" in r.data.tags
        assert "random_state" in r.data.tags
        assert "target_col" in r.data.tags

        # Kiểm tra artifacts: feature_names.json và model
        artifacts = [a.path for a in client.list_artifacts(r.info.run_id)]
        assert "feature_names.json" in artifacts
        assert "model" in artifacts

    # Kiểm tra alias "baseline" cho cả 2 model trong Model Registry
    for name in ("logreg_baseline", "dt_baseline"):
        model_version = client.get_model_version_by_alias(name, "baseline")
        assert model_version is not None
        assert "baseline" in model_version.aliases


def test_get_advanced_default_models():
    """Kiểm tra get_advanced_default_models trả về đúng estimator RF và XGBoost."""
    from src.train import get_advanced_default_models
    models = get_advanced_default_models(random_state=42)
    assert "rf_default" in models
    assert "xgboost_default" in models


def test_train_rf_xgboost_default_mlflow(tmp_path, dummy_train_df, monkeypatch):
    """train_rf_xgboost_default chạy đủ 2 run (RF, XGBoost), log metrics whitelist và alias 'default'."""
    import mlflow
    from mlflow.tracking import MlflowClient
    from src.train import train_rf_xgboost_default

    tracking_uri = tmp_path.as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setattr("src.train.load_split_data", lambda: (dummy_train_df, None, None))

    train_rf_xgboost_default()

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("credit_scoring")
    assert experiment is not None

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 2

    run_names = {r.data.tags.get("mlflow.runName") for r in runs}
    assert run_names == {"rf_default", "xgboost_default"}

    for r in runs:
        for metric in ("roc_auc_mean", "ks_mean", "gini_mean", "pr_auc_mean"):
            assert metric in r.data.metrics
            assert 0.0 <= r.data.metrics[metric] <= 1.0

    for name in ("rf_default", "xgboost_default"):
        model_version = client.get_model_version_by_alias(name, "default")
        assert model_version is not None
        assert "default" in model_version.aliases
