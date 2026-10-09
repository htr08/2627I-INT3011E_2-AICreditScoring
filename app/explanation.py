"""Hợp đồng giao diện giữa app demo và mô hình (reports/demo_wireframe.md, mục 3).

Mô hình dùng trong app cung cấp:
    model.name: str
    model.is_mock: bool
    model.explain(record) -> Explanation
"""

from dataclasses import dataclass

from src.explain import load_feature_groups

# Nhóm đặc trưng dùng cho reason codes (project_plan.md mục 3.5e), đọc từ configs/config.yaml
# (khóa feature_groups) để demo và phân tích SHAP dùng chung một định nghĩa.
_GROUPS = load_feature_groups()
DELINQUENCY = _GROUPS["lich_su_tre_han"]["label"]
UTILIZATION = _GROUPS["muc_su_dung_han_muc"]["label"]
REPAYMENT = _GROUPS["hanh_vi_tra_no"]["label"]
BILL_VOLATILITY = _GROUPS["bien_dong_du_no"]["label"]
DEMOGRAPHICS = _GROUPS["nhan_khau_hoc"]["label"]
FEATURE_GROUPS = [spec["label"] for spec in _GROUPS.values()]


@dataclass(frozen=True)
class Explanation:
    """pd: PD; contributions: đóng góp theo nhóm trên thang logit(PD), base_logit + sum == logit(pd)."""

    pd: float
    base_logit: float
    contributions: dict[str, float]
