"""Mock model cho khung demo Streamlit (Tuần 1).

KHÔNG phải mô hình đã huấn luyện: hệ số được đặt tay để hành vi hợp lý về mặt nghiệp vụ.
Mục đích là cố định giao diện giữa app và mô hình để Tuần 2 thay bằng mô hình thật mà không sửa UI.

Giao diện (mô hình thật cần theo đúng):
    model.explain(record) -> Explanation
    - record: dict các cột gốc của UCI (LIMIT_BAL, EDUCATION, MARRIAGE, AGE, PAY_1..6, BILL_AMT1..6, PAY_AMT1..6).
    - Explanation.contributions: đóng góp theo NHÓM đặc trưng trên thang logit(PD) đã hiệu chuẩn,
      thỏa base_logit + sum(contributions) == logit(pd) (giống tính cộng tính của SHAP).
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

MONTHS = range(1, 7)  # 1 = tháng 9/2005 (gần nhất) ... 6 = tháng 4/2005

# Nhóm đặc trưng dùng cho reason codes (project_plan.md, mục 3e).
DELINQUENCY = "Lịch sử trễ hạn"
UTILIZATION = "Mức sử dụng hạn mức"
REPAYMENT = "Hành vi trả nợ"
BILL_VOLATILITY = "Biến động dư nợ"
DEMOGRAPHICS = "Nhân khẩu học"
FEATURE_GROUPS = [DELINQUENCY, UTILIZATION, REPAYMENT, BILL_VOLATILITY, DEMOGRAPHICS]

# Tỷ lệ vỡ nợ trung bình của dữ liệu, dùng làm base value.
BASE_PD = 0.221

# đặc trưng -> (nhóm, giá trị tham chiếu, hệ số trên thang logit). Đóng góp = hệ số * (x - tham chiếu).
_LINEAR_TERMS = {
    "recent_delay": (DELINQUENCY, 0.3, 0.60),
    "n_delay_months": (DELINQUENCY, 1.0, 0.15),
    "utilization": (UTILIZATION, 0.42, 0.90),
    "log_limit": (UTILIZATION, np.log(140_000), -0.25),
    "repay_ratio": (REPAYMENT, 0.3, -0.80),
    "bill_volatility": (BILL_VOLATILITY, 0.08, 0.80),
    "age": (DEMOGRAPHICS, 35.0, -0.005),
}
_EDUCATION_EFFECT = {1: -0.10, 2: 0.0, 3: 0.05, 4: -0.20}
_MARRIAGE_EFFECT = {1: 0.05, 2: -0.05, 3: 0.0}


@dataclass(frozen=True)
class Explanation:
    pd: float
    base_logit: float
    contributions: dict[str, float]


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def _engineer(record: Mapping[str, float]) -> dict[str, float]:
    limit = max(float(record["LIMIT_BAL"]), 1.0)
    pay = [record[f"PAY_{i}"] for i in MONTHS]
    bill = np.array([record[f"BILL_AMT{i}"] for i in MONTHS], dtype=float)
    pay_amt = [record[f"PAY_AMT{i}"] for i in MONTHS]

    # PAY_AMT_i là khoản trả cho sao kê tháng trước BILL_AMT_(i+1); sao kê <= 0 coi như đã trả đủ.
    ratios = [1.0 if bill[i] <= 0 else min(pay_amt[i - 1] / bill[i], 1.5) for i in range(1, 6)]

    return {
        "recent_delay": max(pay[0], 0),
        "n_delay_months": sum(p >= 1 for p in pay),
        "utilization": float(np.clip(bill.mean() / limit, 0, 1.5)),
        "log_limit": float(np.log(limit)),
        "repay_ratio": float(np.mean(ratios)),
        "bill_volatility": float(np.clip(bill.std() / limit, 0, 1)),
        "age": float(record["AGE"]),
    }


class MockModel:
    name = "Mock model (hệ số đặt tay)"
    is_mock = True

    def explain(self, record: Mapping[str, float]) -> Explanation:
        features = _engineer(record)
        contributions = dict.fromkeys(FEATURE_GROUPS, 0.0)
        for feature, (group, ref, coef) in _LINEAR_TERMS.items():
            contributions[group] += coef * (features[feature] - ref)
        contributions[DEMOGRAPHICS] += _EDUCATION_EFFECT.get(record["EDUCATION"], 0.0)
        contributions[DEMOGRAPHICS] += _MARRIAGE_EFFECT.get(record["MARRIAGE"], 0.0)

        base_logit = _logit(BASE_PD)
        logit_pd = base_logit + sum(contributions.values())
        pd = float(1 / (1 + np.exp(-logit_pd)))
        return Explanation(pd=pd, base_logit=base_logit, contributions=contributions)
