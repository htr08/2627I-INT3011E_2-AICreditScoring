"""Phân tích lỗi phân loại (FN/FP) của mô hình trên một tập đánh giá: gán nhóm, hồ sơ trung bình, so sánh nhóm.

Định nghĩa theo Project Charter mục 7 (công cụ cảnh báo sớm cho danh mục hiện hữu):
- FN: khách hàng vỡ nợ thực tế nhưng không bị gắn cờ (bỏ sót, không kịp can thiệp).
- FP: khách hàng không vỡ nợ nhưng bị gắn cờ (cảnh báo hoặc giảm hạn mức oan).
- TP / TN: gắn cờ đúng khách vỡ nợ / không gắn cờ đúng khách không vỡ nợ.

Module chỉ chứa phần tính toán. Việc huấn luyện mô hình và ghi báo cáo nằm ở scripts/analyze_errors.py.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.features import FeatureEngineeringTransformer
from src.preprocessing import AbnormalCodeTransformer

GROUPS: Tuple[str, ...] = ("FN", "TP", "FP", "TN")

# Đặc trưng dùng để mô tả hồ sơ: cột gốc cần so sánh và các đặc trưng dẫn xuất của feature freeze v1.
# PAY_1_LATE (PAY_1 >= 1) là tỷ lệ hồ sơ trễ hạn ở tháng gần nhất, bổ sung cho PAY_1 vì PAY_1 là mã trạng thái
# (-2, -1, 0, 1, ...) nên trung bình của nó chỉ mang tính tham khảo.
PROFILE_FEATURES: Tuple[str, ...] = (
    "LIMIT_BAL",
    "AGE",
    "PAY_1",
    "PAY_1_LATE",
    "PAY_MEAN",
    "PAY_LATE_2PLUS_COUNT",
    "UTIL_MEAN",
    "PAY_RATIO_MEAN",
)

# Cột gốc bắt buộc để dựng PROFILE_FEATURES (PAY_0 hoặc PAY_1 đều được chấp nhận cho tháng gần nhất).
_REQUIRED_RAW = (
    ["LIMIT_BAL", "AGE"]
    + [f"PAY_{i}" for i in range(2, 7)]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


def assign_error_groups(y_true: Sequence[int], y_prob: Sequence[float], threshold: float) -> np.ndarray:
    """Nhãn FN/FP/TP/TN cho từng hồ sơ; hồ sơ bị gắn cờ khi PD >= threshold."""
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(y_prob, dtype=float) >= threshold
    if len(y_true) != len(flagged):
        raise ValueError(f"Độ dài y_true ({len(y_true)}) và y_prob ({len(flagged)}) phải bằng nhau.")
    actual = y_true == 1
    return np.select([actual & ~flagged, ~actual & flagged, actual & flagged], ["FN", "FP", "TP"], default="TN")


def group_counts(groups: np.ndarray) -> pd.Series:
    """Số hồ sơ của từng nhóm theo thứ tự GROUPS (nhóm vắng mặt có giá trị 0)."""
    return pd.Series(groups).value_counts().reindex(GROUPS, fill_value=0).astype(int)


def build_profile_frame(X_raw: pd.DataFrame) -> pd.DataFrame:
    """Cột gốc của CSV -> các đặc trưng mô tả hồ sơ (PROFILE_FEATURES), kể cả đặc trưng dẫn xuất.

    build_features chỉ trả cột gốc, nên PAY_MEAN, UTIL_MEAN, PAY_RATIO_MEAN... được tính ở đây bằng
    AbnormalCodeTransformer và FeatureEngineeringTransformer. Hai bước này không học tham số từ dữ liệu
    nên áp dụng trực tiếp lên tập Valid không gây leakage.

    Raises:
        ValueError: nếu thiếu cột gốc hoặc thiếu đặc trưng mô tả sau khi dựng.
    """
    has_recent_pay = "PAY_0" in X_raw.columns or "PAY_1" in X_raw.columns
    missing_raw = [c for c in _REQUIRED_RAW if c not in X_raw.columns] + ([] if has_recent_pay else ["PAY_0/PAY_1"])
    if missing_raw:
        raise ValueError(f"Thiếu cột gốc để dựng hồ sơ lỗi: {missing_raw}")

    frame = FeatureEngineeringTransformer().transform(AbnormalCodeTransformer().transform(X_raw))
    frame["PAY_1_LATE"] = (frame["PAY_1"] >= 1).astype(float)
    missing = [c for c in PROFILE_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Thiếu đặc trưng để tổng hợp hồ sơ lỗi: {missing}")
    return frame.loc[:, list(PROFILE_FEATURES)]


def profile_table(profile: pd.DataFrame, groups: np.ndarray, pd_values: Sequence[float]) -> pd.DataFrame:
    """Giá trị trung bình của từng đặc trưng (và PD) theo nhóm FN/TP/FP/TN; cột cuối `ALL` là toàn tập.

    Hàng: PROFILE_FEATURES + `PD`. NaN (vd PAY_RATIO_MEAN khi dư nợ <= 0) được bỏ qua khi lấy trung bình.
    """
    if not (len(profile) == len(groups) == len(pd_values)):
        raise ValueError("profile, groups và pd_values phải có cùng số dòng.")
    frame = profile.copy()
    frame["PD"] = np.asarray(pd_values, dtype=float)
    table = frame.groupby(np.asarray(groups)).mean().T.reindex(columns=list(GROUPS))
    table["ALL"] = frame.mean()
    return table


def nearest_group_counts(table: pd.DataFrame, target: str, candidates: Sequence[str]) -> Dict[str, int]:
    """Với mỗi đặc trưng (trừ PD), nhóm trong `candidates` có trung bình gần `target` nhất; trả về số đặc trưng
    của từng nhóm. Hòa giá trị thì không tính cho nhóm nào."""
    features = [f for f in table.index if f != "PD"]
    counts = {c: 0 for c in candidates}
    for feature in features:
        distances = {c: abs(table.loc[feature, target] - table.loc[feature, c]) for c in candidates}
        if any(np.isnan(v) for v in distances.values()):
            continue
        best = min(distances.values())
        winners = [c for c, v in distances.items() if np.isclose(v, best)]
        if len(winners) == 1:
            counts[winners[0]] += 1
    return counts


def largest_gaps(
    table: pd.DataFrame,
    a: str,
    b: str,
    spread: pd.Series,
    k: int = 3,
) -> List[Tuple[str, float, float]]:
    """k đặc trưng (trừ PD) có chênh lệch trung bình giữa nhóm a và b lớn nhất, chuẩn hóa theo `spread`
    (độ lệch chuẩn toàn tập) để so sánh được giữa các đặc trưng khác thang đo. Trả về (đặc trưng, giá trị a, giá trị b)."""
    features = [f for f in table.index if f != "PD" and spread.get(f, 0) > 0]
    gaps = {f: abs(table.loc[f, a] - table.loc[f, b]) / spread[f] for f in features}
    gaps = {f: g for f, g in gaps.items() if not np.isnan(g)}
    top = sorted(gaps, key=gaps.get, reverse=True)[:k]
    return [(f, float(table.loc[f, a]), float(table.loc[f, b])) for f in top]


def group_overlap(groups_a: np.ndarray, groups_b: np.ndarray, label: str) -> Tuple[int, int, int]:
    """Số hồ sơ thuộc nhóm `label` ở mô hình A, ở mô hình B và ở cả hai (cùng thứ tự dòng)."""
    in_a, in_b = np.asarray(groups_a) == label, np.asarray(groups_b) == label
    return int(in_a.sum()), int(in_b.sum()), int((in_a & in_b).sum())
