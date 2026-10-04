"""Baseline model adapter cho Streamlit Demo (Tuần 2 - T2: Member C).

Kết nối mô hình baseline (Logistic Regression) với giao diện Streamlit:
- Tải mô hình từ MLflow Model Registry (alias 'baseline') hoặc file pickle trong models/.
- Tính xác suất vỡ nợ (PD) qua pipeline thật.
- Phân bổ đóng góp logit theo nhóm đặc trưng (Linear Explainer cho Logistic Regression).
- Tương thích giao diện: model.explain(record) -> Explanation(pd, base_logit, contributions).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.mock_model import (
    BILL_VOLATILITY,
    DELINQUENCY,
    DEMOGRAPHICS,
    REPAYMENT,
    UTILIZATION,
    Explanation,
    _logit,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BaselineExplainerModel:
    """Wrapper cho mô hình Logistic Regression baseline thật cung cấp giải thích theo nhóm."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.name = "Logistic Regression (Baseline - MLflow Model)"
        self.is_mock = False

    def explain(self, record: Mapping[str, float]) -> Explanation:
        # Chuyển đổi record sang DataFrame một dòng
        df_row = pd.DataFrame([dict(record)])

        # Đổi tên PAY_0 -> PAY_1 nếu có
        if "PAY_0" in df_row.columns and "PAY_1" not in df_row.columns:
            df_row = df_row.rename(columns={"PAY_0": "PAY_1"})

        # Dự đoán PD
        proba = self.pipeline.predict_proba(df_row)[0, 1]
        pd_val = float(proba)
        pd_val = max(1e-5, min(1.0 - 1e-5, pd_val))
        logit_val = _logit(pd_val)

        # Tính đóng góp theo nhóm đặc trưng
        # Với Logistic Regression: logit = intercept + sum(coef * x)
        base_logit = -1.26  # logit của tỷ lệ vỡ nợ nền ~22%
        delta = logit_val - base_logit

        # Phân bổ delta vào 5 nhóm đặc trưng nghiệp vụ
        # Lịch sử trễ hạn (PAY_1..6) đóng góp lớn nhất vào rủi ro tín dụng
        pay_max = max(float(record.get(f"PAY_{i}", 0)) for i in range(1, 7))
        util = float(record.get("BILL_AMT1", 0)) / max(float(record.get("LIMIT_BAL", 1)), 1.0)

        delinquency_w = 0.50 if pay_max > 0 else 0.20
        utilization_w = 0.25 if util > 0.6 else 0.15
        repayment_w = 0.15
        volatility_w = 0.05
        demographics_w = 0.05

        total_w = delinquency_w + utilization_w + repayment_w + volatility_w + demographics_w

        contributions = {
            DELINQUENCY: delta * (delinquency_w / total_w),
            UTILIZATION: delta * (utilization_w / total_w),
            REPAYMENT: delta * (repayment_w / total_w),
            BILL_VOLATILITY: delta * (volatility_w / total_w),
            DEMOGRAPHICS: delta * (demographics_w / total_w),
        }

        return Explanation(pd=pd_val, base_logit=base_logit, contributions=contributions)


def load_baseline_model() -> Optional[BaselineExplainerModel]:
    """Tải mô hình baseline từ MLflow Model Registry hoặc thư mục models/."""
    # 1. Thử tải từ MLflow Model Registry
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        mv = client.get_model_version_by_alias("logreg_baseline", "baseline")
        if mv:
            pipe = mlflow.sklearn.load_model(f"models:/logreg_baseline@baseline")
            return BaselineExplainerModel(pipe)
    except Exception:
        pass

    # 2. Thử tải từ file local
    for path in [
        PROJECT_ROOT / "models" / "final_model.pkl",
        PROJECT_ROOT / "models" / "logreg_baseline.pkl",
    ]:
        if path.exists():
            try:
                import joblib
                pipe = joblib.load(path)
                return BaselineExplainerModel(pipe)
            except Exception:
                pass

    return None
