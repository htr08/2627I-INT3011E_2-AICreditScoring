import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc dự án luôn có trong sys.path khi chạy test trực tiếp
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.evaluate import (
    bootstrap_auc_diff_ci,
    bootstrap_metric_ci,
    compute_brier_score,
    compute_confusion_matrix_metrics,
    compute_expected_cost,
    compute_gini,
    compute_ks,
    compute_pr_auc,
    compute_roc_auc,
    delong_auc_ci,
    delong_test,
    evaluate_predictions,
    find_optimal_threshold,
    sensitivity_analysis_cost_ratios,
)


@pytest.fixture
def sample_classification_data():
    """Tạo bộ dữ liệu phân loại mẫu 200 dòng để test nhanh."""
    np.random.seed(42)
    n = 200
    y_true = np.random.binomial(1, 0.25, size=n)
    # Mô hình A (tương đối tốt)
    y_prob_a = np.clip(y_true * 0.45 + np.random.uniform(0.1, 0.45, size=n), 0.01, 0.99)
    # Mô hình B (yếu hơn)
    y_prob_b = np.clip(y_true * 0.15 + np.random.uniform(0.2, 0.6, size=n), 0.01, 0.99)
    ead = np.random.uniform(1000, 50000, size=n)
    return y_true, y_prob_a, y_prob_b, ead


def test_discrimination_metrics():
    # Mô hình hoàn hảo
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    auc = compute_roc_auc(y_true, y_prob)
    assert auc == 1.0

    gini = compute_gini(y_true, y_prob)
    assert gini == 1.0

    ks = compute_ks(y_true, y_prob)
    assert ks == 1.0

    pr_auc = compute_pr_auc(y_true, y_prob)
    assert pr_auc == 1.0


def test_gini_relationship(sample_classification_data):
    y_true, y_prob_a, _, _ = sample_classification_data
    auc = compute_roc_auc(y_true, y_prob_a)
    gini = compute_gini(y_true, y_prob_a)
    assert np.isclose(gini, 2.0 * auc - 1.0)


def test_brier_score():
    y_true = np.array([1, 0])
    y_prob_perfect = np.array([1.0, 0.0])
    assert compute_brier_score(y_true, y_prob_perfect) == 0.0

    y_prob_uncertain = np.array([0.5, 0.5])
    assert compute_brier_score(y_true, y_prob_uncertain) == 0.25


def test_confusion_matrix_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.4, 0.9])
    # threshold 0.5 -> y_pred = [0, 1, 0, 1]
    # TN=1, FP=1, FN=1, TP=1
    res = compute_confusion_matrix_metrics(y_true, y_prob, threshold=0.5)

    assert res["tn"] == 1
    assert res["fp"] == 1
    assert res["fn"] == 1
    assert res["tp"] == 1
    assert res["precision"] == 0.5
    assert res["recall"] == 0.5
    assert res["f1"] == 0.5
    assert res["flag_rate"] == 0.5


def test_expected_cost_and_optimal_threshold(sample_classification_data):
    y_true, y_prob_a, _, ead = sample_classification_data

    # Test không có EAD
    cost_no_ead = compute_expected_cost(
        y_true, y_prob_a, threshold=0.5, cost_fn=0.45, cost_fp=0.05, normalize=True
    )
    assert cost_no_ead >= 0.0

    # Test có EAD
    cost_with_ead = compute_expected_cost(
        y_true, y_prob_a, threshold=0.5, cost_fn=0.45, cost_fp=0.05, ead=ead, normalize=True
    )
    assert cost_with_ead >= 0.0

    # Tìm ngưỡng tối ưu
    opt_thresh, min_cost = find_optimal_threshold(
        y_true, y_prob_a, cost_fn=0.45, cost_fp=0.05
    )
    assert 0.01 <= opt_thresh <= 0.99
    assert min_cost <= cost_no_ead + 1e-6


def test_sensitivity_analysis_costs(sample_classification_data):
    y_true, y_prob_a, _, _ = sample_classification_data
    ratios = (3.0, 5.0, 9.0)
    results = sensitivity_analysis_cost_ratios(y_true, y_prob_a, cost_ratios=ratios)

    assert len(results) == len(ratios)
    for row, ratio in zip(results, ratios):
        assert row["fn_fp_ratio"] == ratio
        assert "optimal_threshold" in row
        assert "expected_cost" in row


def test_bootstrap_ci(sample_classification_data):
    y_true, y_prob_a, _, _ = sample_classification_data
    point, lower, upper = bootstrap_metric_ci(
        y_true, y_prob_a, metric_fn=compute_roc_auc, n_bootstraps=200, random_state=42
    )

    assert 0.0 <= lower <= point <= upper <= 1.0


def test_bootstrap_auc_diff_ci(sample_classification_data):
    y_true, y_prob_a, y_prob_b, _ = sample_classification_data
    res = bootstrap_auc_diff_ci(
        y_true, y_prob_a, y_prob_b, n_bootstraps=200, random_state=42
    )

    assert res["auc_a"] > res["auc_b"]
    assert res["diff"] > 0
    assert res["ci_lower"] <= res["diff"] <= res["ci_upper"]


def test_delong_auc_ci(sample_classification_data):
    y_true, y_prob_a, _, _ = sample_classification_data
    auc, ci_lower, ci_upper = delong_auc_ci(y_true, y_prob_a, ci=0.95)

    sk_auc = compute_roc_auc(y_true, y_prob_a)
    assert np.isclose(auc, sk_auc)
    assert ci_lower <= auc <= ci_upper


def test_delong_test_identical_and_different_models(sample_classification_data):
    y_true, y_prob_a, y_prob_b, _ = sample_classification_data

    # So sánh mô hình với chính nó -> diff = 0, p_value = 1.0
    res_same = delong_test(y_true, y_prob_a, y_prob_a)
    assert res_same["diff"] == 0.0
    assert res_same["p_value"] == 1.0
    assert not res_same["is_significant"]

    # So sánh mô hình tốt hơn với mô hình kém hơn
    res_diff = delong_test(y_true, y_prob_a, y_prob_b)
    assert res_diff["diff"] > 0
    assert 0.0 <= res_diff["p_value"] <= 1.0
    assert res_diff["ci_lower"] <= res_diff["diff"] <= res_diff["ci_upper"]


def test_evaluate_predictions_comprehensive(sample_classification_data):
    y_true, y_prob_a, _, _ = sample_classification_data
    all_metrics = evaluate_predictions(y_true, y_prob_a)

    expected_keys = {
        "roc_auc",
        "gini",
        "ks",
        "pr_auc",
        "brier_score",
        "expected_cost",
        "optimal_threshold",
        "min_expected_cost",
        "threshold",
        "tn",
        "fp",
        "fn",
        "tp",
        "precision",
        "recall",
        "f1",
        "flag_rate",
    }
    assert expected_keys.issubset(all_metrics.keys())
    assert 0.0 <= all_metrics["roc_auc"] <= 1.0
    assert 0.0 <= all_metrics["ks"] <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
