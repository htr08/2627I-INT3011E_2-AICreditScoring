"""Module hiệu chuẩn xác suất (Probability Calibration), tối ưu hóa ngưỡng chi phí và quy đổi điểm PDO.

Nhiệm vụ: Tuần 3 – T2 (Thành viên B - ML Engineer).
Quy ước kỹ thuật: reports/project_plan.md mục 3.5 và reports/project_charter.md mục 1.3, 7.

Bao gồm:
- Trích xuất log-odds / margin thô từ các mô hình (LightGBM, CatBoost, XGBoost, Logistic).
- Platt scaling: fit tay bằng Logistic Regression 1 chiều trên output_margin thô (bảo toàn tính tuyến tính và cộng tính cho SHAP).
- Isotonic regression: hiệu chuẩn phi tham số đơn điệu theo Brier score.
- Đánh giá hiệu chuẩn: Brier score, Log Loss, ROC-AUC, Expected Calibration Error (ECE) và Calibration Curve.
- CalibratedPipeline: đóng gói pipeline gốc + calibrator + ngưỡng chi phí + quy đổi điểm PDO.
- Tối ưu ngưỡng theo ma trận chi phí (LGD=0.45, FP=0.05) kèm phân tích độ nhạy (tỷ lệ 3:1 đến 10:1).
- Quy đổi PD đã hiệu chuẩn sang điểm tín dụng PDO (thang [300, 850], base score 600 tại odds 50:1).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import special
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from src.config import PROJECT_ROOT, load_config
from src.evaluate import (
    compute_confusion_matrix_metrics,
    compute_expected_cost,
    find_optimal_threshold,
    sensitivity_analysis_cost_ratios,
)
from src.scoring import (
    BASE_ODDS,
    BASE_SCORE,
    PDO,
    SCORE_CLIP,
    base_points,
    pd_to_score,
    pdo_params,
    shap_to_points,
)


# ---------------------------------------------------------------------------
# 1. Trích xuất Margin (Raw Log-Odds)
# ---------------------------------------------------------------------------

def predict_margin(estimator_or_pipeline: Any, X: Any) -> np.ndarray:
    """Trích xuất margin thô (raw log-odds / decision function) từ estimator hoặc pipeline.

    Hỗ trợ:
    - Pipeline: tiền xử lý qua các bước đầu, sau đó gọi hàm margin của estimator cuối.
    - LightGBM (LGBMClassifier / Booster): `predict(raw_score=True)`
    - CatBoost (CatBoostClassifier): `predict(prediction_type='RawFormulaVal')`
    - XGBoost (XGBClassifier): `predict(output_margin=True)`
    - Mô hình tuyến tính / Scorecard: `decision_function(X)`
    - Fallback nếu chỉ có predict_proba: `scipy.special.logit(clip(p))`
    """
    if isinstance(estimator_or_pipeline, Pipeline):
        steps = estimator_or_pipeline.steps
        X_trans = X
        for _, step in steps[:-1]:
            if step is not None:
                X_trans = step.transform(X_trans)
        clf = steps[-1][1]
    else:
        clf = estimator_or_pipeline
        X_trans = X

    # 1. LightGBM
    clf_type = type(clf).__name__
    if hasattr(clf, "predict") and "LGBM" in clf_type:
        try:
            return np.asarray(clf.predict(X_trans, raw_score=True), dtype=float)
        except TypeError:
            pass

    # 2. CatBoost
    if hasattr(clf, "predict") and "CatBoost" in clf_type:
        try:
            return np.asarray(clf.predict(X_trans, prediction_type="RawFormulaVal"), dtype=float)
        except (TypeError, ValueError):
            pass

    # 3. XGBoost
    if hasattr(clf, "predict") and "XGB" in clf_type:
        try:
            return np.asarray(clf.predict(X_trans, output_margin=True), dtype=float)
        except TypeError:
            pass

    # 4. Decision function (LogisticRegression, SignConstrainedLogisticRegression, v.v.)
    if hasattr(clf, "decision_function"):
        decision = clf.decision_function(X_trans)
        decision = np.asarray(decision, dtype=float)
        if decision.ndim == 2 and decision.shape[1] == 1:
            decision = decision.ravel()
        return decision

    # 5. Fallback qua logit(predict_proba)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_trans)
        p = proba[:, 1] if proba.ndim == 2 else proba
        p_clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
        return np.asarray(special.logit(p_clipped), dtype=float)

    raise ValueError(f"Không thể trích xuất margin từ estimator kiểu {clf_type!r}.")


# ---------------------------------------------------------------------------
# 2. Platt Calibrator (Fit tay trên output_margin)
# ---------------------------------------------------------------------------

class PlattCalibrator(BaseEstimator, ClassifierMixin):
    """Hiệu chuẩn Platt (Platt scaling) fit tay bằng Logistic Regression 1 chiều trên raw margin.

    PD = sigmoid(a * margin + b)

    Tại sao cần fit tay trên margin (project_plan.md mục 3.5e):
    - CalibratedClassifierCV của scikit-learn mặc định fit sigmoid lên xác suất đã qua sigmoid
      (predict_proba) chứ không phải lên log-odds thô, làm mất tính tuyến tính của logit.
    - Với TreeExplainer trên margin thô: sum(SHAP) = margin. Sau khi Platt scaling, nhân toàn bộ
      SHAP values với hệ số `a` thì:
        logit(PD_hiển thị) = (a * phi_0 + b) + sum_i (a * phi_i)
      tổng SHAP cộng đúng bằng logit của PD hiệu chuẩn!
    """

    def __init__(self, penalty: Optional[str] = None, C: float = 1.0, random_state: int = 42):
        self.penalty = penalty
        self.C = C
        self.random_state = random_state
        self.lr_: Optional[LogisticRegression] = None
        self.a_: Optional[float] = None
        self.b_: Optional[float] = None

    def fit(self, margin: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        margin = np.asarray(margin, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=int).ravel()

        lr = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver="lbfgs",
            random_state=self.random_state,
        )
        lr.fit(margin, y)
        self.lr_ = lr
        self.a_ = float(lr.coef_[0, 0])
        self.b_ = float(lr.intercept_[0])
        return self

    def transform(self, margin: np.ndarray) -> np.ndarray:
        """Trả về mảng 1D xác suất hiệu chuẩn P(Y=1)."""
        margin = np.asarray(margin, dtype=float).reshape(-1, 1)
        if self.lr_ is None:
            raise RuntimeError("PlattCalibrator chưa được fit.")
        return self.lr_.predict_proba(margin)[:, 1]

    def predict_proba(self, margin: np.ndarray) -> np.ndarray:
        """Trả về ma trận 2D shape (n, 2): [P(Y=0), P(Y=1)]."""
        p1 = self.transform(margin)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, margin: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p1 = self.transform(margin)
        return (p1 >= threshold).astype(int)

    def calibrated_logit(self, margin: np.ndarray) -> np.ndarray:
        """Logit sau hiệu chuẩn: a * margin + b."""
        if self.a_ is None or self.b_ is None:
            raise RuntimeError("PlattCalibrator chưa được fit.")
        return self.a_ * np.asarray(margin, dtype=float) + self.b_

    def scale_shap(self, shap_values: Any) -> Any:
        """Nhân SHAP values trên thang margin với hệ số `a` để chuyển sang thang logit đã hiệu chuẩn."""
        if self.a_ is None:
            raise RuntimeError("PlattCalibrator chưa được fit.")
        return shap_values * self.a_


# ---------------------------------------------------------------------------
# 3. Isotonic Calibrator
# ---------------------------------------------------------------------------

class IsotonicCalibrator(BaseEstimator, ClassifierMixin):
    """Hiệu chuẩn đơn điệu phi tham số (Isotonic Regression) theo Brier score.

    Fit trên xác suất thô (raw probability) hoặc margin thô.
    Lưu ý (project_plan.md mục 3.5e): Do là hàm từng đoạn (piecewise constant),
    Isotonic không bảo toàn tính cộng tính tuyến tính cho SHAP.
    """

    def __init__(self, y_min: float = 0.0, y_max: float = 1.0, out_of_bounds: str = "clip"):
        self.y_min = y_min
        self.y_max = y_max
        self.out_of_bounds = out_of_bounds
        self.iso_: Optional[IsotonicRegression] = None

    def fit(self, score_or_prob: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        score_or_prob = np.asarray(score_or_prob, dtype=float).ravel()
        y = np.asarray(y, dtype=int).ravel()

        iso = IsotonicRegression(
            y_min=self.y_min,
            y_max=self.y_max,
            out_of_bounds=self.out_of_bounds,
        )
        iso.fit(score_or_prob, y)
        self.iso_ = iso
        return self

    def transform(self, score_or_prob: np.ndarray) -> np.ndarray:
        if self.iso_ is None:
            raise RuntimeError("IsotonicCalibrator chưa được fit.")
        score_or_prob = np.asarray(score_or_prob, dtype=float).ravel()
        p = self.iso_.predict(score_or_prob)
        return np.clip(p, 0.0, 1.0)

    def predict_proba(self, score_or_prob: np.ndarray) -> np.ndarray:
        p1 = self.transform(score_or_prob)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, score_or_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p1 = self.transform(score_or_prob)
        return (p1 >= threshold).astype(int)


# ---------------------------------------------------------------------------
# 4. Chỉ số đánh giá hiệu chuẩn (Metrics)
# ---------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Tính Expected Calibration Error (ECE) - sai số hiệu chuẩn kỳ vọng qua các bin.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[0] -= 1e-8
        bins[-1] += 1e-8
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for i in range(len(bins) - 1):
        idx = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == len(bins) - 2:
            idx = idx | (y_prob == bins[i + 1])
        bin_count = int(np.sum(idx))
        if bin_count > 0:
            bin_acc = float(np.mean(y_true[idx]))
            bin_conf = float(np.mean(y_prob[idx]))
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Tổng hợp toàn bộ chỉ số đánh giá độ hiệu chuẩn và phân tách xác suất.

    Trả về:
    - brier_score: Brier score loss (càng nhỏ càng tốt, <= 0.25).
    - log_loss: Cross-entropy loss (càng nhỏ càng tốt).
    - roc_auc: Diện tích dưới đường ROC (đo độ phân tách).
    - ece: Expected Calibration Error.
    - prob_true, prob_pred: Tọa độ đường cong hiệu chuẩn (Calibration Curve).
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    p_clipped = np.clip(y_prob, 1e-7, 1.0 - 1e-7)

    prob_true, prob_pred = calibration_curve(y_true, p_clipped, n_bins=n_bins, strategy="uniform")
    brier = float(brier_score_loss(y_true, p_clipped))
    ll = float(log_loss(y_true, p_clipped))
    auc = float(roc_auc_score(y_true, p_clipped))
    ece = float(compute_ece(y_true, p_clipped, n_bins=n_bins))

    return {
        "brier_score": brier,
        "log_loss": ll,
        "roc_auc": auc,
        "ece": ece,
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "n_samples": len(y_true),
    }


# ---------------------------------------------------------------------------
# 5. So sánh Hiệu chuẩn trên tập Valid
# ---------------------------------------------------------------------------

def compare_calibrations_on_valid(
    base_pipeline: Pipeline,
    X_val: Any,
    y_val: np.ndarray,
    cost_fn: float = 0.45,
    cost_fp: float = 0.05,
    ead: Optional[np.ndarray] = None,
    n_bins: int = 10,
    cv_splits: int = 5,
    random_state: int = 42,
    ead_for_cv: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """So sánh hiệu chuẩn in-sample và out-of-fold trên các dự đoán của tập Valid.

    - Platt scaling: fit tay trên output_margin.
    - Isotonic regression: fit trên raw probabilities.
    - Trả về chỉ số fit/đo cùng mẫu trong ``methods`` để tương thích báo cáo cũ.
    - ``cv_methods`` chứa xác suất OOF từ StratifiedKFold cho đánh giá công bằng hơn.
    - ``cv_cost`` chọn ngưỡng trong phần train của từng fold rồi đánh giá chi phí trên
      phần giữ lại; đây là CV của toàn bộ quy trình chọn calibrator/ngưỡng.
    """
    y_val = np.asarray(y_val, dtype=int).ravel()
    if cv_splits < 2:
        raise ValueError("cv_splits phải từ 2 trở lên.")

    # Trích xuất margin và xác suất thô
    margin_val = predict_margin(base_pipeline, X_val)
    p_raw = base_pipeline.predict_proba(X_val)[:, 1]

    # 1. Fit Platt scaling trên margin
    platt = PlattCalibrator().fit(margin_val, y_val)
    p_platt = platt.transform(margin_val)

    # 2. Fit Isotonic regression trên p_raw
    iso = IsotonicCalibrator().fit(p_raw, y_val)
    p_iso = iso.transform(p_raw)

    methods = {
        "uncalibrated": p_raw,
        "platt": p_platt,
        "isotonic": p_iso,
    }

    results: Dict[str, Any] = {
        "platt_params": {"a": platt.a_, "b": platt.b_},
        "calibrators": {"platt": platt, "isotonic": iso},
        "methods": {},
    }

    for name, p in methods.items():
        cal_metrics = evaluate_calibration(y_val, p, n_bins=n_bins)
        opt_th, min_cost = find_optimal_threshold(
            y_val, p, cost_fn=cost_fn, cost_fp=cost_fp, ead=ead
        )
        cost_05 = compute_expected_cost(
            y_val, p, threshold=0.5, cost_fn=cost_fn, cost_fp=cost_fp, ead=ead
        )
        cm_opt = compute_confusion_matrix_metrics(y_val, p, threshold=opt_th)
        cm_05 = compute_confusion_matrix_metrics(y_val, p, threshold=0.5)

        results["methods"][name] = {
            "metrics": cal_metrics,
            "optimal_threshold": float(opt_th),
            "min_expected_cost": float(min_cost),
            "cost_at_05": float(cost_05),
            "cm_at_optimal": cm_opt,
            "cm_at_05": cm_05,
            "y_prob": p,
        }

    # OOF probabilities: each calibrator is fitted without the fold it predicts.
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    cv_probs = {name: np.empty_like(y_val, dtype=float) for name in methods}
    fold_costs: Dict[str, List[Dict[str, float]]] = {
        "platt": [], "isotonic": []
    }
    fold_costs_ead: Dict[str, List[Dict[str, float]]] = {
        "platt": [], "isotonic": []
    }
    fold_metrics: Dict[str, List[Dict[str, float]]] = {
        name: [] for name in methods
    }
    ead_arr = None if ead is None else np.asarray(ead, dtype=float).ravel()
    ead_cv_arr = None if ead_for_cv is None else np.asarray(ead_for_cv, dtype=float).ravel()
    if ead_cv_arr is not None and len(ead_cv_arr) != len(y_val):
        raise ValueError("ead_for_cv phải có cùng số phần tử với y_val.")
    for train_idx, test_idx in cv.split(margin_val, y_val):
        platt_fold = PlattCalibrator().fit(margin_val[train_idx], y_val[train_idx])
        iso_fold = IsotonicCalibrator().fit(p_raw[train_idx], y_val[train_idx])
        fold_predictions = {
            "uncalibrated": p_raw[test_idx],
            "platt": platt_fold.transform(margin_val[test_idx]),
            "isotonic": iso_fold.transform(p_raw[test_idx]),
        }
        for name, predictions in fold_predictions.items():
            cv_probs[name][test_idx] = predictions
            fold_metrics[name].append(
                evaluate_calibration(y_val[test_idx], predictions, n_bins=n_bins)
            )

        fold_ead_train = None if ead_arr is None else ead_arr[train_idx]
        fold_ead_test = None if ead_arr is None else ead_arr[test_idx]
        fold_ead_cv_train = None if ead_cv_arr is None else ead_cv_arr[train_idx]
        fold_ead_cv_test = None if ead_cv_arr is None else ead_cv_arr[test_idx]
        for name in ("platt", "isotonic"):
            p_train = platt_fold.transform(margin_val[train_idx]) if name == "platt" else iso_fold.transform(p_raw[train_idx])
            p_test = fold_predictions[name]
            threshold, _ = find_optimal_threshold(
                y_val[train_idx], p_train, cost_fn=cost_fn, cost_fp=cost_fp,
                ead=fold_ead_train,
            )
            heldout_cost = compute_expected_cost(
                y_val[test_idx], p_test, threshold=threshold,
                cost_fn=cost_fn, cost_fp=cost_fp, ead=fold_ead_test,
            )
            heldout_cost_05 = compute_expected_cost(
                y_val[test_idx], p_test, threshold=0.5,
                cost_fn=cost_fn, cost_fp=cost_fp, ead=fold_ead_test,
            )
            fold_costs[name].append({
                "threshold": float(threshold),
                "heldout_cost": float(heldout_cost),
                "heldout_cost_at_05": float(heldout_cost_05),
            })

            if ead_cv_arr is not None:
                threshold_ead, _ = find_optimal_threshold(
                    y_val[train_idx], p_train, cost_fn=cost_fn, cost_fp=cost_fp,
                    ead=fold_ead_cv_train,
                )
                heldout_cost_ead = compute_expected_cost(
                    y_val[test_idx], p_test, threshold=threshold_ead,
                    cost_fn=cost_fn, cost_fp=cost_fp, ead=fold_ead_cv_test,
                )
                fold_costs_ead[name].append({
                    "threshold": float(threshold_ead),
                    "heldout_cost": float(heldout_cost_ead),
                    "heldout_cost_at_05": float(compute_expected_cost(
                        y_val[test_idx], p_test, threshold=0.5,
                        cost_fn=cost_fn, cost_fp=cost_fp, ead=fold_ead_cv_test,
                    )),
                })

    metric_names = ("brier_score", "log_loss", "roc_auc", "ece")
    results["cv_methods"] = {}
    for name, p in cv_probs.items():
        fold_values = fold_metrics[name]
        results["cv_methods"][name] = {
            "metrics": {
                metric: float(np.mean([fold[metric] for fold in fold_values]))
                for metric in metric_names
            },
            "metric_std": {
                metric: float(np.std([fold[metric] for fold in fold_values], ddof=1))
                for metric in metric_names
            },
            "fold_metrics": fold_values,
            "y_prob": p,
        }
    results["cv_cost"] = {
        name: {
            "folds": values,
            "threshold_mean": float(np.mean([v["threshold"] for v in values])),
            "threshold_std": float(np.std([v["threshold"] for v in values], ddof=1)),
            "heldout_cost_mean": float(np.mean([v["heldout_cost"] for v in values])),
            "heldout_cost_std": float(np.std([v["heldout_cost"] for v in values], ddof=1)),
            "heldout_cost_at_05_mean": float(np.mean([v["heldout_cost_at_05"] for v in values])),
            "heldout_cost_at_05_std": float(np.std([v["heldout_cost_at_05"] for v in values], ddof=1)),
        }
        for name, values in fold_costs.items()
    }
    results["cv_cost_ead"] = {
        name: {
            "folds": values,
            "threshold_mean": float(np.mean([v["threshold"] for v in values])),
            "threshold_std": float(np.std([v["threshold"] for v in values], ddof=1)),
            "heldout_cost_mean": float(np.mean([v["heldout_cost"] for v in values])),
            "heldout_cost_std": float(np.std([v["heldout_cost"] for v in values], ddof=1)),
            "heldout_cost_at_05_mean": float(np.mean([v["heldout_cost_at_05"] for v in values])),
            "heldout_cost_at_05_std": float(np.std([v["heldout_cost_at_05"] for v in values], ddof=1)),
        }
        for name, values in fold_costs_ead.items()
    } if ead_cv_arr is not None else {}

    return results


# ---------------------------------------------------------------------------
# 6. CalibratedPipeline (Đóng gói hoàn chỉnh phục vụ Model Registry & Demo)
# ---------------------------------------------------------------------------

class CalibratedPipeline(BaseEstimator, ClassifierMixin):
    """Đóng gói Pipeline gốc + Calibrator đã fit trên Valid + Ngưỡng tối ưu + PDO Scoring.

    Giữ nguyên mô hình gốc được huấn luyện trên Train (không retrain).
    Calibrator được fit riêng trên Valid.
    Cung cấp các phương thức dự đoán:
    - predict_proba: xác suất vỡ nợ đã hiệu chuẩn.
    - predict: phân loại nhị phân theo ngưỡng tối ưu (hoặc ngưỡng tùy chọn).
    - predict_score: điểm tín dụng PDO trong thang [300, 850].
    """

    def __init__(
        self,
        base_pipeline: Pipeline,
        method: str = "platt",
        threshold: float = 0.5,
        base_score: float = BASE_SCORE,
        base_odds: float = BASE_ODDS,
        pdo: float = PDO,
        score_clip: Tuple[int, int] = SCORE_CLIP,
    ):
        self.base_pipeline = base_pipeline
        self.method = method
        self.threshold = threshold
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.score_clip = score_clip

        self.calibrator_: Optional[Union[PlattCalibrator, IsotonicCalibrator]] = None
        self.optimal_threshold_: Optional[float] = None
        self.platt_a_: Optional[float] = None
        self.platt_b_: Optional[float] = None

    def fit_calibrator(
        self,
        X_val: Any,
        y_val: np.ndarray,
        optimize_thresh: bool = True,
        cost_fn: float = 0.45,
        cost_fp: float = 0.05,
        ead: Optional[np.ndarray] = None,
    ) -> "CalibratedPipeline":
        """Fit calibrator trên tập Valid và tùy chọn tối ưu ngưỡng theo ma trận chi phí."""
        y_val = np.asarray(y_val, dtype=int).ravel()

        if self.method == "platt":
            margin_val = predict_margin(self.base_pipeline, X_val)
            calibrator = PlattCalibrator().fit(margin_val, y_val)
            self.platt_a_ = calibrator.a_
            self.platt_b_ = calibrator.b_
        elif self.method == "isotonic":
            p_raw = self.base_pipeline.predict_proba(X_val)[:, 1]
            calibrator = IsotonicCalibrator().fit(p_raw, y_val)
        else:
            raise ValueError(f"Phương pháp hiệu chuẩn không hợp lệ: {self.method!r}. Chọn 'platt' hoặc 'isotonic'.")

        self.calibrator_ = calibrator

        if optimize_thresh:
            p_cal = self.predict_proba(X_val)[:, 1]
            opt_th, _ = find_optimal_threshold(
                y_val, p_cal, cost_fn=cost_fn, cost_fp=cost_fp, ead=ead
            )
            self.optimal_threshold_ = float(opt_th)
            self.threshold = float(opt_th)

        return self

    def get_margin(self, X: Any) -> np.ndarray:
        """Trích xuất margin thô từ pipeline gốc."""
        return predict_margin(self.base_pipeline, X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Dự đoán xác suất đã hiệu chuẩn. Trả về shape (n, 2)."""
        if self.calibrator_ is None:
            return self.base_pipeline.predict_proba(X)

        if isinstance(self.calibrator_, PlattCalibrator):
            margin = self.get_margin(X)
            return self.calibrator_.predict_proba(margin)
        elif isinstance(self.calibrator_, IsotonicCalibrator):
            p_raw = self.base_pipeline.predict_proba(X)[:, 1]
            return self.calibrator_.predict_proba(p_raw)
        else:
            raise RuntimeError("Calibrator không xác định.")

    def predict(self, X: Any, threshold: Optional[float] = None) -> np.ndarray:
        """Dự đoán nhãn nhị phân theo ngưỡng (mặc định dùng self.threshold)."""
        th = self.threshold if threshold is None else threshold
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= th).astype(int)

    def predict_score(
        self,
        X: Any,
        clip: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Quy đổi PD dự đoán đã hiệu chuẩn sang điểm tín dụng theo PDO."""
        effective_clip = self.score_clip if clip is None else clip
        p1 = self.predict_proba(X)[:, 1]
        return pd_to_score(
            p1,
            base_score=self.base_score,
            base_odds=self.base_odds,
            pdo=self.pdo,
            clip=effective_clip,
        )

    def scale_shap(self, shap_values: Any) -> Any:
        """Chuyển đổi SHAP values trên thang margin sang thang logit đã hiệu chuẩn."""
        if isinstance(self.calibrator_, PlattCalibrator) and self.platt_a_ is not None:
            return shap_values * self.platt_a_
        return shap_values


# ---------------------------------------------------------------------------
# 7. Hàm hỗ trợ chạy toàn bộ quy trình Hiệu chuẩn & Phân tích (Runner)
# ---------------------------------------------------------------------------

def run_calibration_experiments(
    candidates: Sequence[str] = ("lightgbm_monotone", "catboost"),
    include_baseline: bool = True,
    save_figures: bool = True,
    reports_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Chạy toàn bộ thí nghiệm hiệu chuẩn, chọn ngưỡng tối ưu và quy đổi PDO trên tập Valid.

    Trả về từ điển chứa đầy đủ kết quả số liệu để ghi báo cáo và vẽ đồ thị.
    """
    from sklearn.linear_model import LogisticRegression
    from src.data_split import load_split_data
    from src.features import build_features
    from src.pipelines import make_pipeline
    from src.tune import fit_tuned_pipeline

    reports_dir = reports_dir or (PROJECT_ROOT / "reports")
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    cost_cfg = cfg.get("cost_matrix", {})
    cost_fn = float(cost_cfg.get("cost_fn", 0.45))
    cost_fp = float(cost_cfg.get("cost_fp", 0.05))
    cost_ratios = tuple(cost_cfg.get("fn_fp_ratios", [3.0, 5.0, 7.0, 9.0, 10.0]))

    scoring_cfg = cfg.get("scoring", {})
    base_score = float(scoring_cfg.get("base_score", BASE_SCORE))
    base_odds = float(scoring_cfg.get("base_odds", BASE_ODDS))
    pdo = float(scoring_cfg.get("pdo", PDO))
    clip_bounds = (int(scoring_cfg.get("clip_min", 300)), int(scoring_cfg.get("clip_max", 850)))

    train_df, val_df, _ = load_split_data()
    X_train, y_train = build_features(train_df)
    X_val, y_val = build_features(val_df)
    ead_val = np.maximum(val_df["BILL_AMT1"].to_numpy(dtype=float), 0.0)

    models_to_run: Dict[str, Pipeline] = {}
    for study in candidates:
        models_to_run[study] = fit_tuned_pipeline(study, X_train, y_train)

    if include_baseline:
        lr_pipe = make_pipeline(
            LogisticRegression(max_iter=1000, random_state=cfg.get("random_state", 42))
        ).fit(X_train, y_train)
        models_to_run["logistic_regression"] = lr_pipe

    experiments: Dict[str, Any] = {}
    cv_metric_rows: List[Dict[str, Any]] = []
    cv_cost_rows: List[Dict[str, Any]] = []
    for name, pipe in models_to_run.items():
        comp = compare_calibrations_on_valid(
            pipe,
            X_val,
            y_val,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            ead=None,
            ead_for_cv=ead_val,
        )
        for method_name, cv_result in comp["cv_methods"].items():
            cv_metric_rows.append({
                "model": name,
                "method": method_name,
                **cv_result["metrics"],
                **{f"{metric}_std": value for metric, value in cv_result["metric_std"].items()},
            })
        for method_name, cost_result in comp["cv_cost"].items():
            cv_cost_rows.append({"model": name, "method": method_name, "weighting": "unit", **{
                key: value for key, value in cost_result.items() if key != "folds"
            }})
        for method_name, cost_result in comp["cv_cost_ead"].items():
            cv_cost_rows.append({"model": name, "method": method_name, "weighting": "EAD", **{
                key: value for key, value in cost_result.items() if key != "folds"
            }})
        # Bổ sung tính toán có EAD cho phương pháp Platt và Uncalibrated
        p_platt = comp["methods"]["platt"]["y_prob"]
        th_ead, min_cost_ead = find_optimal_threshold(
            y_val, p_platt, cost_fn=cost_fn, cost_fp=cost_fp, ead=ead_val
        )
        cost_ead_05 = compute_expected_cost(
            y_val, p_platt, threshold=0.5, cost_fn=cost_fn, cost_fp=cost_fp, ead=ead_val
        )
        comp["ead_analysis"] = {
            "optimal_threshold": float(th_ead),
            "min_expected_cost": float(min_cost_ead),
            "cost_at_05": float(cost_ead_05),
        }

        # Phân tích độ nhạy theo tỷ lệ chi phí
        comp["sensitivity"] = sensitivity_analysis_cost_ratios(
            y_val,
            p_platt,
            cost_ratios=cost_ratios,
            cost_fp=cost_fp,
        )

        # Tính điểm tín dụng PDO
        cal_pipe = CalibratedPipeline(
            base_pipeline=pipe,
            method="platt",
            threshold=comp["methods"]["platt"]["optimal_threshold"],
            base_score=base_score,
            base_odds=base_odds,
            pdo=pdo,
            score_clip=clip_bounds,
        )
        cal_pipe.fit_calibrator(X_val, y_val, optimize_thresh=True, cost_fn=cost_fn, cost_fp=cost_fp)
        scores = cal_pipe.predict_score(X_val)
        cutoff_score = pd_to_score(
            cal_pipe.optimal_threshold_,
            base_score=base_score,
            base_odds=base_odds,
            pdo=pdo,
            clip=clip_bounds,
        )

        comp["scoring"] = {
            "scores": scores,
            "cutoff_score": float(cutoff_score),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "score_median": float(np.median(scores)),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "calibrated_pipeline": cal_pipe,
        }

        experiments[name] = comp

    if cv_metric_rows:
        pd.DataFrame(cv_metric_rows).to_csv(reports_dir / "calibration_cv_metrics.csv", index=False)
    if cv_cost_rows:
        pd.DataFrame(cv_cost_rows).to_csv(reports_dir / "calibration_cv_cost.csv", index=False)

    if save_figures:
        _plot_calibration_figures(experiments, fig_dir, y_val)

    return experiments


def _plot_calibration_figures(experiments: Dict[str, Any], fig_dir: Path, y_val: np.ndarray) -> None:
    """Vẽ và lưu các biểu đồ hiệu chuẩn, đường cong chi phí và phân phối điểm số."""
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Calibration Curves
    fig, axes = plt.subplots(1, len(experiments), figsize=(6 * len(experiments), 5), squeeze=False)
    for ax, (name, comp) in zip(axes[0], experiments.items()):
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
        for method_name, style, col in [
            ("uncalibrated", "s-", "#e74c3c"),
            ("platt", "o-", "#2ecc71"),
            ("isotonic", "^-", "#3498db"),
        ]:
            m = comp["methods"][method_name]["metrics"]
            brier = m["brier_score"]
            ax.plot(
                m["prob_pred"],
                m["prob_true"],
                style,
                label=f"{method_name.capitalize()} (Brier={brier:.4f})",
                color=col,
                linewidth=2,
                markersize=6,
            )
        ax.set_title(f"Calibration: {name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Fraction of Positives", fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(fig_dir / "calibration_curves.png", dpi=200)
    plt.close(fig)

    # 2. Expected Cost Curves
    fig, ax = plt.subplots(figsize=(8, 5))
    thresholds = np.linspace(0.02, 0.98, 97)
    for name, comp in experiments.items():
        p = comp["methods"]["platt"]["y_prob"]
        costs = [
            compute_expected_cost(y_val, p, threshold=th, cost_fn=0.45, cost_fp=0.05, normalize=True)
            for th in thresholds
        ]
        opt_th = comp["methods"]["platt"]["optimal_threshold"]
        min_cost = comp["methods"]["platt"]["min_expected_cost"]
        ax.plot(thresholds, costs, label=f"{name} (Opt Th={opt_th:.2f}, Cost={min_cost:.4f})", linewidth=2)
        ax.scatter([opt_th], [min_cost], s=60, zorder=5)

    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="Default Th=0.50")
    ax.set_title("Expected Unit Cost vs. Decision Threshold (Platt-Calibrated)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decision Threshold", fontsize=10)
    ax.set_ylabel("Expected Cost per Customer", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(fig_dir / "cost_curves.png", dpi=200)
    plt.close(fig)

    # 3. Score Distribution (Primary Candidate)
    primary = "lightgbm_monotone" if "lightgbm_monotone" in experiments else list(experiments.keys())[0]
    scores = experiments[primary]["scoring"]["scores"]
    cutoff = experiments[primary]["scoring"]["cutoff_score"]

    fig, ax = plt.subplots(figsize=(8, 5))
    n_bins = 40
    counts, bins, _ = ax.hist(scores, bins=n_bins, color="#3498db", edgecolor="white", alpha=0.85, label="Hồ sơ Valid")
    ax.axvline(cutoff, color="#e74c3c", linestyle="--", linewidth=2.5, label=f"Cutoff Ngưỡng Tối ưu: {cutoff:.1f} điểm")
    ax.set_title(f"Phân phối Điểm Tín dụng PDO ({primary}) - Thang [300, 850]", fontsize=12, fontweight="bold")
    ax.set_xlabel("Điểm tín dụng (Score)", fontsize=10)
    ax.set_ylabel("Số lượng khách hàng", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(fig_dir / "score_distribution.png", dpi=200)
    plt.close(fig)
