"""Hợp đồng giao diện giữa app demo và mô hình (reports/demo_wireframe.md, mục 3).

Mô hình dùng trong app cung cấp:
    model.name: str
    model.is_mock: bool
    model.explain(record) -> Explanation
"""

from dataclasses import dataclass

# Nhóm đặc trưng dùng cho reason codes (project_plan.md, mục 3e).
DELINQUENCY = "Lịch sử trễ hạn"
UTILIZATION = "Mức sử dụng hạn mức"
REPAYMENT = "Hành vi trả nợ"
BILL_VOLATILITY = "Biến động dư nợ"
DEMOGRAPHICS = "Nhân khẩu học"
FEATURE_GROUPS = [DELINQUENCY, UTILIZATION, REPAYMENT, BILL_VOLATILITY, DEMOGRAPHICS]


@dataclass(frozen=True)
class Explanation:
    """pd: PD; contributions: đóng góp theo nhóm trên thang logit(PD), base_logit + sum == logit(pd)."""

    pd: float
    base_logit: float
    contributions: dict[str, float]
