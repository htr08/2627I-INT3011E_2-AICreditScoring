"""Unit tests cho src/train.py và src/features.py.

Theo phản hồi review PR #16:
- Fixture dùng np.random.default_rng(42) cục bộ thay vì np.random.seed()
- Nhãn sinh có tương quan với PAY_1 và LIMIT_BAL để assert AUC > 0.6
- Parametrize kiểm thử đồng thời cho LogisticRegression và DecisionTreeClassifier
- Kiểm tra whitelist REPORT_METRICS: không lọt optimal_threshold hay min_expected_cost
- Kiểm tra std mẫu (ddof=1)
- Test build_features loại ID, target và báo KeyError khi thiếu target
- Test tính phân tầng (Stratify): tỷ lệ default ở mỗi fold xấp xỉ tỷ lệ tổng
- Test tính tất định: chạy run_cv hai lần phải ra cùng kết quả
- Test train_baseline với MLflow: dùng tmp_path làm tracking URI, kiểm tra đủ 2 run và metrics
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src.features import build_features, make_pipeline
from src.train import REPORT_METRICS, run_cv, train_baseline


@pytest.fixture
def dummy_train_df():
    """500 mẫu với nhãn có tín hiệu mạnh từ PAY_1 và LIMIT_BAL để AUC > 0.6."""
    rng = np.random.default_rng(42)
    n = 500
    limit_bal = rng.integers(10_000, 500_000, n)
    pay_1 = rng.integers(-1, 9, n)
    pay_2 = rng.integers(-1, 9, n)

    # Tương quan dương mạnh với PAY_1, âm với LIMIT_BAL → AUC đạt > 0.6 ổn định
    log_odds = -1.5 + 0.8 * pay_1 + 0.3 * pay_2 - 2.0 * (limit_bal / 500_000)
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    labels = (rng.random(n) < prob).astype(int)

    # Đảm bảo có cả hai class
    if labels.sum() == 0:
        labels[:50] = 1
    elif labels.sum() == n:
        labels[:50] = 0

    data = {
        "ID": np.arange(1, n + 1),
        "LIMIT_BAL": limit_bal,
        "PAY_1": pay_1,
        "PAY_2": pay_2,
        "default.payment.next.month": labels,
    }
    return pd.DataFrame(data)


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


# 2. Parametrized CV tests cho cả 2 model
@pytest.mark.parametrize(
    "estimator",
    [
        LogisticRegression(max_iter=300, random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
    ],
    ids=["logreg", "decision_tree"],
)
def test_run_cv_returns_5_folds(dummy_train_df, estimator):
    """Số fold trả về phải đúng với n_splits."""
    X, y = build_features(dummy_train_df)
    pipe = make_pipeline(estimator)
    metrics = run_cv(pipe, X, y, n_splits=5)
    assert len(metrics) == 5


@pytest.mark.parametrize(
    "estimator",
    [
        LogisticRegression(max_iter=300, random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
    ],
    ids=["logreg", "decision_tree"],
)
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


@pytest.mark.parametrize(
    "estimator",
    [
        LogisticRegression(max_iter=300, random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
    ],
    ids=["logreg", "decision_tree"],
)
def test_run_cv_auc_signal(dummy_train_df, estimator):
    """Nhãn có tín hiệu (phụ thuộc PAY_1) nên AUC phải đạt > 0.6 ở mọi fold."""
    X, y = build_features(dummy_train_df)
    pipe = make_pipeline(estimator)
    metrics = run_cv(pipe, X, y, n_splits=5)
    for m in metrics:
        assert m["roc_auc"] > 0.6, f"Fold {m.get('fold')} có AUC quá thấp: {m['roc_auc']}"


@pytest.mark.parametrize(
    "estimator",
    [
        LogisticRegression(max_iter=300, random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
    ],
    ids=["logreg", "decision_tree"],
)
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
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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


# 5. Test std mẫu (ddof=1)
def test_run_cv_std_ddof1(dummy_train_df):
    """Kiểm tra std tính với ddof=1 (std mẫu) không âm và đúng logic ddof=1 >= ddof=0."""
    X, y = build_features(dummy_train_df)
    pipe = make_pipeline(LogisticRegression(max_iter=300, random_state=42))
    fold_metrics = run_cv(pipe, X, y, n_splits=5)

    auc_values = [m["roc_auc"] for m in fold_metrics]
    std_ddof0 = float(np.std(auc_values, ddof=0))
    std_ddof1 = float(np.std(auc_values, ddof=1))
    assert std_ddof1 >= std_ddof0 >= 0.0


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
