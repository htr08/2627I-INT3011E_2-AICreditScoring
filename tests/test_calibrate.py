"""Unit tests cho src/calibrate.py (Probability Calibration, Cost Threshold, and PDO Scoring).

Task: Tuần 3 – T2 (Thành viên B).
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from conftest import make_raw_credit_df
from src.calibrate import (
    CalibratedPipeline,
    IsotonicCalibrator,
    PlattCalibrator,
    compute_ece,
    evaluate_calibration,
    predict_margin,
)
from src.features import build_features
from src.pipelines import make_pipeline
from src.scoring import pd_to_score


@pytest.fixture(scope="module")
def sample_credit_data():
    df = make_raw_credit_df(n=500, seed=42)
    X, y = build_features(df)
    return X, y


# ---------------------------------------------------------------------------
# 1. Trích xuất Margin
# ---------------------------------------------------------------------------

def test_predict_margin_logistic_regression(sample_credit_data):
    X, y = sample_credit_data
    pipe = make_pipeline(LogisticRegression(max_iter=300, random_state=42)).fit(X, y)
    margin = predict_margin(pipe, X)

    assert isinstance(margin, np.ndarray)
    assert margin.shape == (len(X),)
    # Kiểm tra margin khớp decision_function của classifier cuối
    X_trans = pipe.named_steps["preprocess"].transform(X)
    expected = pipe.named_steps["clf"].decision_function(X_trans)
    np.testing.assert_allclose(margin, expected, rtol=1e-5)


def test_predict_margin_tree_models():
    """Kiểm tra trích xuất margin từ LightGBM và CatBoost."""
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)

    lgb_clf = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1).fit(X, y)
    lgb_margin = predict_margin(lgb_clf, X)
    assert lgb_margin.shape == (50,)
    np.testing.assert_allclose(lgb_margin, lgb_clf.predict(X, raw_score=True))

    cb_clf = CatBoostClassifier(iterations=10, random_seed=42, verbose=0).fit(X, y)
    cb_margin = predict_margin(cb_clf, X)
    assert cb_margin.shape == (50,)
    np.testing.assert_allclose(cb_margin, cb_clf.predict(X, prediction_type="RawFormulaVal"))


def test_predict_margin_fallback():
    """Kiểm tra fallback qua logit(predict_proba) khi estimator chỉ có predict_proba."""
    from sklearn.naive_bayes import GaussianNB

    X = np.random.randn(40, 3)
    y = np.random.randint(0, 2, 40)
    nb = GaussianNB().fit(X, y)
    margin = predict_margin(nb, X)
    assert margin.shape == (40,)
    assert np.all(np.isfinite(margin))


# ---------------------------------------------------------------------------
# 2. PlattCalibrator
# ---------------------------------------------------------------------------

def test_platt_calibrator_fit_and_predict():
    rng = np.random.default_rng(42)
    margin = rng.normal(0, 2, 300)
    true_prob = 1.0 / (1.0 + np.exp(-(1.2 * margin + 0.1)))
    y = (rng.random(300) < true_prob).astype(int)

    platt = PlattCalibrator().fit(margin, y)
    assert platt.a_ is not None
    assert platt.b_ is not None
    assert platt.a_ > 0.5  # Hệ số tương quan dương theo rủi ro

    proba = platt.predict_proba(margin)
    assert proba.shape == (300, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(300), rtol=1e-6)

    # Đơn điệu tăng: margin lớn hơn -> PD lớn hơn
    sorted_idx = np.argsort(margin)
    pd_sorted = proba[sorted_idx, 1]
    assert np.all(np.diff(pd_sorted) >= 0)

    # Logit đã hiệu chuẩn và scale SHAP
    cal_logit = platt.calibrated_logit(margin)
    np.testing.assert_allclose(cal_logit, platt.a_ * margin + platt.b_, rtol=1e-6)
    shap_val = np.array([0.5, -0.3, 0.2])
    scaled = platt.scale_shap(shap_val)
    np.testing.assert_allclose(scaled, platt.a_ * shap_val, rtol=1e-6)


# ---------------------------------------------------------------------------
# 3. IsotonicCalibrator
# ---------------------------------------------------------------------------

def test_isotonic_calibrator_monotonic():
    rng = np.random.default_rng(42)
    p_raw = rng.uniform(0.05, 0.95, 200)
    y = (rng.random(200) < p_raw).astype(int)

    iso = IsotonicCalibrator().fit(p_raw, y)
    p_cal = iso.transform(p_raw)
    assert len(p_cal) == 200
    assert 0.0 <= p_cal.min() and p_cal.max() <= 1.0

    # Kiểm tra tính đơn điệu
    sorted_idx = np.argsort(p_raw)
    assert np.all(np.diff(p_cal[sorted_idx]) >= -1e-9)


# ---------------------------------------------------------------------------
# 4. Metrics & Evaluation
# ---------------------------------------------------------------------------

def test_compute_ece_perfect_calibration():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    ece = compute_ece(y_true, y_prob, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_evaluate_calibration_metrics(sample_credit_data):
    X, y = sample_credit_data
    pipe = make_pipeline(LogisticRegression(max_iter=300, random_state=42)).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]

    metrics = evaluate_calibration(y.to_numpy(), proba, n_bins=10)
    assert "brier_score" in metrics
    assert "log_loss" in metrics
    assert "roc_auc" in metrics
    assert "ece" in metrics
    assert 0.0 <= metrics["brier_score"] <= 0.25
    assert metrics["roc_auc"] > 0.6
    assert len(metrics["prob_true"]) == len(metrics["prob_pred"])


# ---------------------------------------------------------------------------
# 5. CalibratedPipeline & PDO Scoring
# ---------------------------------------------------------------------------

def test_calibrated_pipeline_platt_and_scoring(sample_credit_data):
    X, y = sample_credit_data
    split = len(X) // 2
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_val, y_val = X.iloc[split:], y.iloc[split:]

    base_pipe = make_pipeline(LogisticRegression(max_iter=300, random_state=42)).fit(X_tr, y_tr)

    cal_pipe = CalibratedPipeline(base_pipe, method="platt")
    cal_pipe.fit_calibrator(X_val, y_val.to_numpy(), optimize_thresh=True)

    assert cal_pipe.platt_a_ is not None
    assert cal_pipe.optimal_threshold_ is not None
    assert 0.01 <= cal_pipe.optimal_threshold_ <= 0.99

    proba = cal_pipe.predict_proba(X_val)
    assert proba.shape == (len(X_val), 2)

    # Dự đoán điểm tín dụng PDO
    scores = cal_pipe.predict_score(X_val)
    assert scores.shape == (len(X_val),)
    assert scores.min() >= 300
    assert scores.max() <= 850

    # Điểm nghịch biến với xác suất vỡ nợ (PD cao -> điểm thấp)
    p1 = proba[:, 1]
    sorted_idx = np.argsort(p1)
    # Bỏ qua các giá trị bị clip ở biên
    unclipped_mask = (scores[sorted_idx] > 300) & (scores[sorted_idx] < 850)
    assert np.all(np.diff(scores[sorted_idx][unclipped_mask]) <= 0)


def test_calibrated_pipeline_isotonic(sample_credit_data):
    X, y = sample_credit_data
    split = len(X) // 2
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_val, y_val = X.iloc[split:], y.iloc[split:]

    base_pipe = make_pipeline(LogisticRegression(max_iter=300, random_state=42)).fit(X_tr, y_tr)

    cal_pipe = CalibratedPipeline(base_pipe, method="isotonic")
    cal_pipe.fit_calibrator(X_val, y_val.to_numpy(), optimize_thresh=True)

    proba = cal_pipe.predict_proba(X_val)
    assert proba.shape == (len(X_val), 2)
    assert 0.0 <= proba.min() and proba.max() <= 1.0
