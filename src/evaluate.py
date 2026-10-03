"""Module đánh giá mô hình phân loại rủi ro tín dụng.

Bao gồm:
- Các chỉ số phân tách: ROC-AUC, Gini, Kolmogorov-Smirnov (KS), PR-AUC.
- Chỉ số hiệu chuẩn: Brier score.
- Ma trận nhầm lẫn và các chỉ số theo ngưỡng: Precision, Recall, F1, Flag rate.
- Khoảng tin cậy Bootstrap (Bootstrap Confidence Interval).
- Kiểm định DeLong test so sánh cặp mô hình (DeLong p-value & CI).
- Hàm chi phí kỳ vọng và tối ưu ngưỡng theo ma trận chi phí FN/FP.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Tính chỉ số ROC-AUC (Area Under the Receiver Operating Characteristic Curve)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return float(roc_auc_score(y_true, y_prob))


def compute_gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Tính hệ số Gini chuẩn hóa: Gini = 2 * AUC - 1."""
    auc = compute_roc_auc(y_true, y_prob)
    return float(2.0 * auc - 1.0)


def compute_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Tính chỉ số Kolmogorov-Smirnov (KS) đo độ phân tách giữa phân phối Good và Bad.
    
    KS = max |TPR - FPR|.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks_stat = np.max(np.abs(tpr - fpr))
    return float(ks_stat)


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Tính PR-AUC (Average Precision Score) cho dữ liệu mất cân bằng."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return float(average_precision_score(y_true, y_prob))


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Tính Brier score (Mean Squared Error giữa xác suất dự đoán và nhãn thực tế)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return float(brier_score_loss(y_true, y_prob))


def compute_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Union[int, float]]:
    """Tính ma trận nhầm lẫn và các chỉ số phụ thuộc ngưỡng.
    
    Trả về:
    - tn, fp, fn, tp
    - precision, recall, f1
    - flag_rate: tỷ lệ khách hàng bị gắn cờ can thiệp (TP + FP) / Total.
    - threshold
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Độ dài y_true ({len(y_true)}) và y_prob ({len(y_prob)}) phải bằng nhau."
        )

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    flag_rate = (tp + fp) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "flag_rate": float(flag_rate),
    }


def compute_expected_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    cost_fn: float = 0.45,
    cost_fp: float = 0.05,
    ead: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> float:
    """Tính chi phí kỳ vọng kinh doanh theo ma trận chi phí trong Project Charter.
    
    Chi phí:
    - FN (bỏ sót khách vỡ nợ): LGD * EAD (mặc định LGD = 0.45)
    - FP (cảnh báo oan khách tốt): Lãi/phí mất = 5% * EAD (mặc định 0.05)
    
    Nếu cung cấp mảng `ead`:
      Tổng chi phí = sum(cost_fn * ead[FN]) + sum(cost_fp * ead[FP])
    Nếu không cung cấp `ead`:
      Tổng chi phí = cost_fn * FN_count + cost_fp * FP_count
      
    Tham số `normalize`:
      Nếu True: trả về chi phí bình quân trên mỗi khách hàng (cost per customer).
      Nếu False: trả về tổng chi phí toàn tập dữ liệu.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Độ dài y_true ({len(y_true)}) và y_prob ({len(y_prob)}) phải bằng nhau."
        )

    y_pred = (y_prob >= threshold).astype(int)

    is_fn = (y_true == 1) & (y_pred == 0)
    is_fp = (y_true == 0) & (y_pred == 1)

    if ead is not None:
        ead_arr = np.asarray(ead)
        if len(ead_arr) != len(y_true):
            raise ValueError(
                f"Độ dài ead ({len(ead_arr)}) và y_true ({len(y_true)}) phải bằng nhau."
            )
        total_cost = np.sum(cost_fn * ead_arr[is_fn]) + np.sum(cost_fp * ead_arr[is_fp])
    else:
        total_cost = cost_fn * np.sum(is_fn) + cost_fp * np.sum(is_fp)

    if normalize and len(y_true) > 0:
        return float(total_cost / len(y_true))
    return float(total_cost)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 0.45,
    cost_fp: float = 0.05,
    ead: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Tìm ngưỡng cắt xác suất tối ưu hóa chi phí kỳ vọng (nhỏ nhất).
    
    Trả về:
    - optimal_threshold: Ngưỡng đạt chi phí nhỏ nhất.
    - min_expected_cost: Chi phí kỳ vọng bình quân tại ngưỡng tối ưu.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    best_thresh = float(thresholds[0])
    min_cost = float("inf")

    for th in thresholds:
        cost = compute_expected_cost(
            y_true=y_true,
            y_prob=y_prob,
            threshold=th,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            ead=ead,
            normalize=True,
        )
        if cost < min_cost:
            min_cost = cost
            best_thresh = float(th)

    return best_thresh, min_cost


def sensitivity_analysis_cost_ratios(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_ratios: Tuple[float, ...] = (3.0, 5.0, 7.0, 9.0, 10.0),
    cost_fp: float = 0.05,
    ead: Optional[np.ndarray] = None,
) -> List[Dict[str, float]]:
    """Phân tích độ nhạy của ngưỡng tối ưu theo tỷ lệ chi phí FN:FP từ 3:1 đến 10:1."""
    results = []
    for ratio in cost_ratios:
        cost_fn = cost_fp * ratio
        opt_thresh, min_cost = find_optimal_threshold(
            y_true=y_true,
            y_prob=y_prob,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            ead=ead,
        )
        conf = compute_confusion_matrix_metrics(y_true, y_prob, threshold=opt_thresh)
        results.append({
            "fn_fp_ratio": float(ratio),
            "cost_fn": float(cost_fn),
            "cost_fp": float(cost_fp),
            "optimal_threshold": float(opt_thresh),
            "expected_cost": float(min_cost),
            "flag_rate": conf["flag_rate"],
            "recall": conf["recall"],
            "precision": conf["precision"],
        })
    return results


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = compute_roc_auc,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Ước lượng khoảng tin cậy (Bootstrap Confidence Interval) cho một chỉ số.
    
    Trả về:
    - point_estimate: Giá trị chỉ số đo trên toàn bộ tập dữ liệu gốc.
    - ci_lower: Ngưỡng dưới khoảng tin cậy.
    - ci_upper: Ngưỡng trên khoảng tin cậy.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Độ dài y_true ({len(y_true)}) và y_prob ({len(y_prob)}) phải bằng nhau."
        )

    n_samples = len(y_true)
    point_estimate = float(metric_fn(y_true, y_prob))

    rng = np.random.RandomState(random_state)
    bootstrapped_scores: List[float] = []

    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n_samples, n_samples)
        # Bỏ qua các lần rút mẫu nếu chỉ xuất hiện 1 nhãn duy nhất
        if len(np.unique(y_true[boot_idx])) < 2:
            continue
        score = metric_fn(y_true[boot_idx], y_prob[boot_idx])
        bootstrapped_scores.append(score)

    if not bootstrapped_scores:
        return point_estimate, point_estimate, point_estimate

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(bootstrapped_scores, 100.0 * alpha))
    ci_upper = float(np.percentile(bootstrapped_scores, 100.0 * (1.0 - alpha)))

    return point_estimate, ci_lower, ci_upper


def bootstrap_auc_diff_ci(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Union[float, bool]]:
    """Ước lượng khoảng tin cậy Bootstrap cho chênh lệch ROC-AUC (AUC_A - AUC_B).
    
    Phục vụ KPI chính: Bootstrap 95% CI của chênh lệch AUC không chứa 0.
    """
    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)

    if len(y_true) != len(y_prob_a) or len(y_true) != len(y_prob_b):
        raise ValueError(
            f"Độ dài y_true ({len(y_true)}), y_prob_a ({len(y_prob_a)}) và y_prob_b ({len(y_prob_b)}) phải bằng nhau."
        )

    auc_a = compute_roc_auc(y_true, y_prob_a)
    auc_b = compute_roc_auc(y_true, y_prob_b)
    diff_point = auc_a - auc_b

    n_samples = len(y_true)
    rng = np.random.RandomState(random_state)
    bootstrapped_diffs: List[float] = []

    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[boot_idx])) < 2:
            continue
        diff = compute_roc_auc(y_true[boot_idx], y_prob_a[boot_idx]) - compute_roc_auc(
            y_true[boot_idx], y_prob_b[boot_idx]
        )
        bootstrapped_diffs.append(diff)

    if not bootstrapped_diffs:
        ci_lower = diff_point
        ci_upper = diff_point
        is_significant = False
    else:
        alpha = (1.0 - ci) / 2.0
        ci_lower = float(np.percentile(bootstrapped_diffs, 100.0 * alpha))
        ci_upper = float(np.percentile(bootstrapped_diffs, 100.0 * (1.0 - alpha)))
        is_significant = bool(ci_lower > 0 or ci_upper < 0)

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "diff": float(diff_point),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "is_significant": is_significant,
    }


def _compute_delong_structural_components(
    pos: np.ndarray,
    neg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tính các thành phần cấu trúc V10 và V01 bằng thuật toán xếp hạng O((m+n)log(m+n))."""
    m = len(pos)
    n = len(neg)
    pooled = np.concatenate([pos, neg])
    ranks = stats.rankdata(pooled, method="average")

    pos_ranks_in_pooled = ranks[:m]
    neg_ranks_in_pooled = ranks[m:]

    pos_ranks_internal = stats.rankdata(pos, method="average")
    neg_ranks_internal = stats.rankdata(neg, method="average")

    v10 = (pos_ranks_in_pooled - pos_ranks_internal) / n
    v01 = 1.0 - (neg_ranks_in_pooled - neg_ranks_internal) / m
    return v10, v01


def delong_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Ước lượng phương sai và khoảng tin cậy của ROC-AUC bằng phương pháp giải tích DeLong."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Độ dài y_true ({len(y_true)}) và y_prob ({len(y_prob)}) phải bằng nhau."
        )

    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    m, n = len(pos), len(neg)

    if m == 0 or n == 0:
        raise ValueError("y_true phải có cả hai nhãn 0 và 1.")

    v10, v01 = _compute_delong_structural_components(pos, neg)
    auc = float(np.mean(v10))

    s10 = float(np.var(v10, ddof=1)) if m > 1 else 0.0
    s01 = float(np.var(v01, ddof=1)) if n > 1 else 0.0

    var_auc = (s10 / m) + (s01 / n)
    se_auc = float(np.sqrt(max(var_auc, 0.0)))

    z_crit = float(stats.norm.ppf(1.0 - (1.0 - ci) / 2.0))
    ci_lower = float(np.clip(auc - z_crit * se_auc, 0.0, 1.0))
    ci_upper = float(np.clip(auc + z_crit * se_auc, 0.0, 1.0))

    return auc, ci_lower, ci_upper


def delong_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    ci: float = 0.95,
) -> Dict[str, Union[float, bool]]:
    """Kiểm định DeLong so sánh hai mô hình phân loại trên cùng tập dữ liệu (paired samples).
    
    Phục vụ KPI chính: DeLong test p < 0.05 chứng minh sự khác biệt có ý nghĩa thống kê.
    
    Trả về dict:
    - auc_a, auc_b: AUC của từng mô hình
    - diff: Chênh lệch AUC (AUC_A - AUC_B)
    - z_stat: Giá trị thống kê kiểm định Z
    - p_value: p-value hai phía
    - ci_lower, ci_upper: Khoảng tin cậy DeLong của chênh lệch
    - is_significant: True nếu p_value < 0.05
    """
    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)

    if len(y_true) != len(y_prob_a) or len(y_true) != len(y_prob_b):
        raise ValueError(
            f"Độ dài y_true ({len(y_true)}), y_prob_a ({len(y_prob_a)}) và y_prob_b ({len(y_prob_b)}) phải bằng nhau."
        )

    pos_mask = y_true == 1
    neg_mask = y_true == 0

    pos_a, neg_a = y_prob_a[pos_mask], y_prob_a[neg_mask]
    pos_b, neg_b = y_prob_b[pos_mask], y_prob_b[neg_mask]

    m, n = len(pos_a), len(neg_a)
    if m == 0 or n == 0:
        raise ValueError("y_true phải có cả hai nhãn 0 và 1.")

    v10_a, v01_a = _compute_delong_structural_components(pos_a, neg_a)
    v10_b, v01_b = _compute_delong_structural_components(pos_b, neg_b)

    auc_a = float(np.mean(v10_a))
    auc_b = float(np.mean(v10_b))
    diff = auc_a - auc_b

    v10 = np.vstack([v10_a, v10_b])
    v01 = np.vstack([v01_a, v01_b])

    s10 = np.cov(v10) if m > 1 else np.zeros((2, 2))
    s01 = np.cov(v01) if n > 1 else np.zeros((2, 2))

    s = (s10 / m) + (s01 / n)
    var_diff = float(s[0, 0] + s[1, 1] - 2.0 * s[0, 1])

    if var_diff <= 1e-12:
        z_stat = 0.0
        p_val = 1.0
        ci_lower = diff
        ci_upper = diff
    else:
        se_diff = np.sqrt(var_diff)
        z_stat = float(diff / se_diff)
        p_val = float(2.0 * stats.norm.sf(np.abs(z_stat)))
        z_crit = float(stats.norm.ppf(1.0 - (1.0 - ci) / 2.0))
        ci_lower = float(diff - z_crit * se_diff)
        ci_upper = float(diff + z_crit * se_diff)

    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "diff": float(diff),
        "z_stat": z_stat,
        "p_value": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "is_significant": bool(p_val < 0.05),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    cost_fn: float = 0.45,
    cost_fp: float = 0.05,
    ead: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Hàm tổng hợp tính toàn bộ các chỉ số đánh giá của một mô hình."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    auc = compute_roc_auc(y_true, y_prob)
    gini = compute_gini(y_true, y_prob)
    ks = compute_ks(y_true, y_prob)
    pr_auc = compute_pr_auc(y_true, y_prob)
    brier = compute_brier_score(y_true, y_prob)

    conf_metrics = compute_confusion_matrix_metrics(y_true, y_prob, threshold=threshold)
    expected_cost = compute_expected_cost(
        y_true, y_prob, threshold=threshold, cost_fn=cost_fn, cost_fp=cost_fp, ead=ead
    )
    opt_thresh, min_cost = find_optimal_threshold(
        y_true, y_prob, cost_fn=cost_fn, cost_fp=cost_fp, ead=ead
    )

    return {
        "roc_auc": auc,
        "gini": gini,
        "ks": ks,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "expected_cost": expected_cost,
        "optimal_threshold": opt_thresh,
        "min_expected_cost": min_cost,
        **conf_metrics,
    }
