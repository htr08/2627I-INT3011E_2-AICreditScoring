import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.features import build_features, make_pipeline
from src.train import run_cv


@pytest.fixture
def dummy_train_df():
    np.random.seed(42)
    n = 500
    data = {
        "ID": np.arange(1, n + 1),
        "LIMIT_BAL": np.random.randint(10000, 500000, n),
        "PAY_1": np.random.randint(-1, 9, n),
        "PAY_2": np.random.randint(-1, 9, n),
        "default.payment.next.month": np.random.choice([0, 1], n, p=[0.78, 0.22]),
    }
    return pd.DataFrame(data)


def test_run_cv_returns_5_folds(dummy_train_df):
    X, y = build_features(dummy_train_df)
    pipe = make_pipeline(LogisticRegression(max_iter=300))
    metrics = run_cv(pipe, X, y, n_splits=5)
    assert len(metrics) == 5


def test_run_cv_metrics_keys(dummy_train_df):
    X, y = build_features(dummy_train_df)
    pipe = make_pipeline(LogisticRegression(max_iter=300))
    metrics = run_cv(pipe, X, y, n_splits=5)
    for m in metrics:
        assert "roc_auc" in m
        assert "ks" in m
        assert "gini" in m


def test_run_cv_auc_in_range(dummy_train_df):
    X, y = build_features(dummy_train_df)
    pipe = make_pipeline(LogisticRegression(max_iter=300))
    metrics = run_cv(pipe, X, y, n_splits=5)
    for m in metrics:
        assert 0.0 <= m["roc_auc"] <= 1.0


def test_run_cv_decision_tree(dummy_train_df):
    X, y = build_features(dummy_train_df)
    pipe = make_pipeline(DecisionTreeClassifier(max_depth=5, random_state=42))
    metrics = run_cv(pipe, X, y, n_splits=5)
    assert len(metrics) == 5
