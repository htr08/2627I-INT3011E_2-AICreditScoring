"""Adapter nối mô hình cuối (LightGBM monotonic + Platt) với app demo.

Mô hình lưu ở `models/final_model.pkl` là `CalibratedPipeline`: pipeline `preprocess -> LGBMClassifier` fit trên Train,
calibrator Platt (PD = sigmoid(a·margin + b)) fit trên Valid. Giải thích dùng `shap.TreeExplainer` trên margin thô
(project_plan.md mục 3.5e): SHAP nhân hệ số a thì tổng đóng góp của mọi cột cộng với base calibrated
(a·phi0 + b) đúng bằng logit(PD hiển thị). Đóng góp được cộng về nhóm đặc trưng cấu hình ở `feature_groups`.
"""

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.explanation import Explanation
from app.validation import record_to_frame
from src.calibrate import DEFAULT_MODEL_PATH, CalibratedPipeline, PlattCalibrator, load_calibrated_pipeline
from src.explain import ShapResult, load_feature_groups

NAME = "LightGBM monotonic + Platt (mô hình cuối)"


class FinalModel:
    is_mock = False
    is_final = True
    name = NAME

    def __init__(self, calibrated: CalibratedPipeline):
        if not isinstance(calibrated.calibrator_, PlattCalibrator):
            raise TypeError("Adapter chỉ hỗ trợ calibrator Platt (SHAP chỉ cộng tính với phép biến đổi tuyến tính).")
        steps = getattr(calibrated.base_pipeline, "named_steps", {})
        clf = steps.get("clf")
        if "preprocess" not in steps or not hasattr(clf, "booster_"):
            raise TypeError("Adapter chỉ hỗ trợ Pipeline 'preprocess' -> LGBMClassifier.")

        import shap

        self.calibrated = calibrated
        self.threshold = float(calibrated.threshold)
        self._a, self._b = float(calibrated.platt_a_), float(calibrated.platt_b_)
        self._preprocess, self._clf = steps["preprocess"], clf
        self._explainer = shap.TreeExplainer(clf)
        col_tf = self._preprocess.named_steps["preprocessor"]
        self._columns = [name.split("__", 1)[1] for name in col_tf.get_feature_names_out()]
        self._categorical_vars = list(col_tf.transformers_[0][2])
        self._groups = load_feature_groups()

    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "FinalModel":
        return cls(load_calibrated_pipeline(path or DEFAULT_MODEL_PATH))

    def to_frame(self, record: Mapping[str, float]) -> pd.DataFrame:
        return record_to_frame(record)

    def score(self, record: Mapping[str, float]) -> float:
        """Điểm PDO của hồ sơ (đã clip theo cấu hình mô hình)."""
        return float(self.calibrated.predict_score(self.to_frame(record))[0])

    def _shap(self, record: Mapping[str, float]) -> ShapResult:
        Z = np.asarray(self._preprocess.transform(self.to_frame(record)), dtype=float)
        with warnings.catch_warnings():  # shap 0.46 cảnh báo định dạng đầu ra của LightGBM nhị phân; đã xử lý bên dưới
            warnings.simplefilter("ignore", UserWarning)
            values = self._explainer.shap_values(Z)
        if isinstance(values, list):  # phiên bản shap cũ trả một mảng cho mỗi lớp
            values = values[-1]
        base = float(np.ravel(self._explainer.expected_value)[-1])
        margin = self._clf.predict(Z, raw_score=True)
        return ShapResult(np.asarray(values), base, np.asarray(margin, dtype=float), self._columns, self._categorical_vars)

    def explain(self, record: Mapping[str, float]) -> Explanation:
        shap_result = self._shap(record)
        by_group = shap_result.by_group(self._groups).iloc[0] * self._a  # thang logit đã hiệu chuẩn
        pd_ = float(self.calibrated.predict_proba(self.to_frame(record))[0, 1])
        return Explanation(
            pd=pd_,
            base_logit=self._a * shap_result.base + self._b,
            contributions={group: float(value) for group, value in by_group.items()},
            variable_contributions={
                var: float(value) for var, value in (shap_result.by_variable().iloc[0] * self._a).items()
            },
        )
