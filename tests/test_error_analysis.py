"""Unit tests cho src/error_analysis.py và scripts/analyze_errors.py (phân tích lỗi FN/FP trên tập Valid)."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw_credit_df
from scripts.analyze_errors import build_report, error_findings, fmt_value, lean, markdown_table
from src.error_analysis import (
    GROUPS,
    PROFILE_FEATURES,
    assign_error_groups,
    build_profile_frame,
    group_counts,
    group_overlap,
    largest_gaps,
    nearest_group_counts,
    profile_table,
)
from src.features import build_features


# ---------------------------------------------------------------------------
# Gán nhóm
# ---------------------------------------------------------------------------

def test_assign_error_groups_follows_charter_definition():
    y = [1, 1, 0, 0, 1, 0]
    p = [0.9, 0.1, 0.8, 0.2, 0.5, 0.5]
    # PD >= ngưỡng thì bị gắn cờ: 0.5 >= 0.5 là gắn cờ
    groups = assign_error_groups(y, p, threshold=0.5)
    assert groups.tolist() == ["TP", "FN", "FP", "TN", "TP", "FP"]


def test_assign_error_groups_length_mismatch_raises():
    with pytest.raises(ValueError, match="phải bằng nhau"):
        assign_error_groups([1, 0], [0.5], threshold=0.5)


def test_group_counts_includes_empty_groups_in_fixed_order():
    counts = group_counts(np.array(["FN", "FN", "TN"]))
    assert list(counts.index) == list(GROUPS)
    assert counts.to_dict() == {"FN": 2, "TP": 0, "FP": 0, "TN": 1}


# ---------------------------------------------------------------------------
# Hồ sơ mô tả
# ---------------------------------------------------------------------------

def test_build_profile_frame_computes_derived_features_from_raw_columns():
    df = make_raw_credit_df(n=120)
    X, _ = build_features(df)
    assert "PAY_MEAN" not in X.columns  # build_features chỉ trả cột gốc

    frame = build_profile_frame(X)
    assert list(frame.columns) == list(PROFILE_FEATURES)
    assert len(frame) == len(X)
    # PAY_0 của CSV gốc được đổi tên thành PAY_1; PAY_1_LATE là cờ trễ hạn tháng gần nhất
    np.testing.assert_array_equal(frame["PAY_1_LATE"], (X["PAY_0"] >= 1).astype(float))
    assert frame["UTIL_MEAN"].notna().all()


def test_build_profile_frame_raises_when_raw_column_missing():
    X, _ = build_features(make_raw_credit_df(n=50))
    with pytest.raises(ValueError, match="BILL_AMT3"):
        build_profile_frame(X.drop(columns=["BILL_AMT3"]))
    with pytest.raises(ValueError, match="PAY_0/PAY_1"):
        build_profile_frame(X.drop(columns=["PAY_0"]))


def test_profile_table_means_per_group_and_overall():
    profile = pd.DataFrame({"LIMIT_BAL": [10.0, 30.0, 50.0, 70.0], "AGE": [20.0, 40.0, 60.0, 80.0]})
    groups = np.array(["FN", "FN", "TN", "TP"])
    table = profile_table(profile, groups, [0.1, 0.3, 0.5, 0.7])

    assert list(table.columns) == [*GROUPS, "ALL"]
    assert table.loc["LIMIT_BAL", "FN"] == pytest.approx(20.0)
    assert table.loc["AGE", "TN"] == pytest.approx(60.0)
    assert table.loc["PD", "ALL"] == pytest.approx(0.4)
    assert np.isnan(table.loc["LIMIT_BAL", "FP"])  # nhóm vắng mặt


def test_profile_table_rejects_length_mismatch():
    with pytest.raises(ValueError):
        profile_table(pd.DataFrame({"A": [1.0, 2.0]}), np.array(["TN"]), [0.1, 0.2])


# ---------------------------------------------------------------------------
# So sánh nhóm
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_table():
    # FN gần TN ở A, B và gần TP ở C; D cách đều TN và TP nên không tính cho nhóm nào.
    return pd.DataFrame(
        {
            "FN": [1.0, 10.0, 90.0, 5.0, 0.2],
            "TP": [9.0, 50.0, 100.0, 9.0, 0.7],
            "FP": [8.0, 45.0, 99.0, 5.0, 0.6],
            "TN": [0.0, 12.0, 20.0, 1.0, 0.1],
            "ALL": [4.0, 30.0, 70.0, 5.0, 0.4],
        },
        index=["A", "B", "C", "D", "PD"],
    )


def test_nearest_group_counts_skips_pd_and_counts_closest(toy_table):
    # A, B gần TN; C gần TP; D hòa (|5-1| = |5-9| = 4) và PD không được tính
    assert nearest_group_counts(toy_table, "FN", ["TN", "TP"]) == {"TN": 2, "TP": 1}
    assert nearest_group_counts(toy_table, "FP", ["TP", "TN"]) == {"TP": 3, "TN": 0}


def test_largest_gaps_are_standardized_by_spread(toy_table):
    spread = pd.Series({"A": 1.0, "B": 100.0, "C": 1.0, "D": 1.0})
    top = largest_gaps(toy_table, "FN", "TP", spread, k=2)
    # |FN-TP|/spread: A=8, B=0.4, C=10, D=4 -> C rồi A
    assert [f for f, _, _ in top] == ["C", "A"]
    assert top[0][1:] == (90.0, 100.0)


def test_group_overlap_counts_shared_members():
    a = np.array(["FN", "FN", "TN", "FN"])
    b = np.array(["FN", "TN", "TN", "FN"])
    assert group_overlap(a, b, "FN") == (3, 2, 2)


# ---------------------------------------------------------------------------
# Script: định dạng và nhận xét sinh từ số liệu
# ---------------------------------------------------------------------------

def test_fmt_value_by_feature_type():
    assert fmt_value("LIMIT_BAL", 167339.4) == "167,339"
    assert fmt_value("PAY_1_LATE", 0.2234) == "22.3%"
    assert fmt_value("PD", 0.0644) == "0.064"
    assert fmt_value("UTIL_MEAN", float("nan")) == "–"


def test_lean_requires_three_quarters_of_features():
    assert lean(6, 2, 8) == "first"
    assert lean(2, 6, 8) == "second"
    assert lean(5, 3, 8) == "mixed"


def test_markdown_table_alignment_row():
    md = markdown_table(["a", "b"], [[1, 2]], ["l", "r"])
    assert md.splitlines() == ["| a | b |", "| :--- | ---: |", "| 1 | 2 |"]


def test_error_findings_quote_numbers_from_table(toy_table):
    counts = pd.Series({"FN": 4, "TP": 6, "FP": 3, "TN": 7})
    spread = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0})
    lines = error_findings(toy_table, counts, spread)
    assert len(lines) == 2
    fn_line, fp_line = lines
    assert "n = 4, 40.0% số hồ sơ vỡ nợ" in fn_line
    assert "FN gần TN hơn TP ở 2/4 đặc trưng, gần TP hơn TN ở 1/4" in fn_line
    assert "`B` 10.00 so với 50.00" in fn_line  # chênh lệch chuẩn hóa lớn nhất (40)
    assert "n = 3, 30.0% số hồ sơ không vỡ nợ" in fp_line
    assert "FP gần TP hơn TN ở 3/4 đặc trưng, gần TN hơn TP ở 0/4" in fp_line


def test_error_findings_handles_empty_error_group(toy_table):
    counts = pd.Series({"FN": 0, "TP": 6, "FP": 0, "TN": 7})
    lines = error_findings(toy_table, counts, pd.Series({"A": 1.0}))
    assert lines[0].startswith("**FN**: 0 hồ sơ")
    assert lines[1].startswith("**FP**: 0 hồ sơ")


def test_build_report_end_to_end_on_synthetic_data():
    df = make_raw_credit_df(n=600)
    X, y = build_features(df)
    rng = np.random.default_rng(0)
    base = 1.0 / (1.0 + np.exp(-(0.8 * X["PAY_0"].to_numpy() - 1.5)))
    data = {
        "X_valid": X,
        "y_valid": y.to_numpy(),
        "pd": {
            "Mô hình A": np.clip(base + rng.normal(0, 0.02, len(X)), 0.001, 0.999),
            "Mô hình B": np.clip(base + rng.normal(0, 0.05, len(X)), 0.001, 0.999),
        },
        "platt": (1.01, 0.04),
    }
    report = build_report(data, (0.09, 0.5))

    assert report.startswith("# Báo cáo Phân tích Lỗi")
    for section in ("## 1. Phạm vi", "## 2. Phân bố", "## 3. Hồ sơ", "## 4. Đối chiếu", "## 5. Kết luận", "## 6. Hạn chế"):
        assert section in report
    # Mọi đặc trưng mô tả đều có trong bảng hồ sơ, kể cả đặc trưng dẫn xuất mà build_features không trả về
    for feature in PROFILE_FEATURES:
        assert f"| `{feature}` |" in report
    # Thuật ngữ theo Charter mục 7, không dùng cách gọi của bài toán xét duyệt khoản vay mới
    assert "bỏ sót" in report
    assert "Duyệt nhầm" not in report and "Từ chối nhầm" not in report
