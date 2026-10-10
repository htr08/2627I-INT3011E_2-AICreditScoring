"""Unit tests cho scripts/compare_models.py."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.compare_models import (
    CV_MODEL_RESULTS,
    build_full_comparison_df,
    build_summary_table_df,
    check_charter_kpis,
    compute_statistical_tests,
    df_to_markdown,
    generate_markdown_report,
    main,
)


def test_build_summary_table_df():
    """Kiểm tra cấu trúc bảng tóm tắt chuẩn theo Mục 5 project_plan.md."""
    df = build_summary_table_df()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = [
        "Mô hình",
        "AUC CV (mean ± std)",
        "AUC Test (95% CI)",
        "Gini",
        "KS",
        "PR-AUC",
        "Brier (thô)",
        "Thời gian train",
        "Khả năng giải thích",
    ]
    assert list(df.columns) == expected_cols

    # Kiểm tra có đủ các mô hình chủ chốt và dòng MLP tuỳ chọn theo Mục 5
    model_names = df["Mô hình"].tolist()
    assert any("Logistic Regression" in m for m in model_names)
    assert any("Logistic Scorecard" in m for m in model_names)
    assert any("Decision Tree" in m for m in model_names)
    assert any("Random Forest" in m for m in model_names)
    assert any("XGBoost" in m for m in model_names)
    assert any("LightGBM" in m for m in model_names)
    assert any("CatBoost" in m for m in model_names)
    assert any("MLP" in m for m in model_names)


def test_build_full_comparison_df():
    """Kiểm tra bảng tổng hợp đầy đủ 17 cấu hình."""
    df = build_full_comparison_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(CV_MODEL_RESULTS)
    assert "ROC-AUC (mean ± std)" in df.columns
    assert "F1-Score" in df.columns
    assert "Recall@0.5" in df.columns
    assert "Precision@0.5" in df.columns

    # Kiểm tra metric cập nhật chuẩn MLflow của LR và Scorecard
    lr_row = df[df["Mô hình"] == "Logistic Regression (baseline)"].iloc[0]
    assert lr_row["PR-AUC"] == "0.5470"
    assert lr_row["Recall@0.5"] == "0.3619"
    assert lr_row["F1-Score"] == "0.4714"

    sc_row = df[df["Mô hình"] == "Logistic Scorecard (WoE)"].iloc[0]
    assert sc_row["Recall@0.5"] == "0.3744"
    assert sc_row["Precision@0.5"] == "0.6596"
    assert sc_row["F1-Score"] == "0.4777"


def test_compute_statistical_tests():
    """Kiểm tra tính toán kiểm định t-test cặp đôi trên các fold."""
    df = compute_statistical_tests()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "So sánh cặp" in df.columns
    assert "p-value (paired)" in df.columns
    assert "t-statistic" in df.columns
    assert "95% CI (Δ AUC)" in df.columns

    # CatBoost tuned vs baseline phải có p < 0.01 và CI không chứa 0
    cb_row = df[df["So sánh cặp"] == "CatBoost tuned vs Logistic baseline"].iloc[0]
    assert float(cb_row["p-value (paired)"]) < 0.01
    assert "+0." in cb_row["95% CI (Δ AUC)"]

    # LightGBM monotonic vs baseline phải có p < 0.01 và CI không chứa 0
    lgb_row = df[df["So sánh cặp"] == "LightGBM monotonic vs Logistic baseline"].iloc[0]
    assert float(lgb_row["p-value (paired)"]) < 0.01
    assert "+0." in lgb_row["95% CI (Δ AUC)"]

    # Scorecard vs baseline phải không có ý nghĩa thống kê (p > 0.05)
    sc_row = df[df["So sánh cặp"] == "Logistic Scorecard vs Logistic baseline"].iloc[0]
    assert float(sc_row["p-value (paired)"]) > 0.05
    assert sc_row["Ý nghĩa (p < 0.05)"] == "Không"


def test_check_charter_kpis():
    """Kiểm tra đối chiếu mốc KPI trong Project Charter."""
    df = check_charter_kpis()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert any("LightGBM" in m for m in df["Mô hình"])
    assert any("CatBoost" in m for m in df["Mô hình"])


def test_df_to_markdown():
    """Kiểm tra hàm chuyển đổi DataFrame sang Markdown table không phụ thuộc tabulate."""
    sample = pd.DataFrame({
        "Model": ["A", "B"],
        "AUC": [0.78, 0.79],
    })
    md = df_to_markdown(sample)
    assert "| Model | AUC |" in md
    assert "| --- | --- |" in md
    assert "| A | 0.78 |" in md
    assert "| B | 0.79 |" in md


def test_generate_markdown_report():
    """Kiểm tra nội dung báo cáo markdown hoàn chỉnh."""
    content = generate_markdown_report()
    assert "# Báo cáo Tổng hợp: Bảng So sánh Sơ bộ Các Mô hình" in content
    assert "Tuần 2 – T6: Thành viên B (ML Engineer)" in content
    assert "LightGBM tuned + Monotonic" in content
    assert "CatBoost tuned" in content
    assert "Kiểm định paired t-test" in content


def test_compare_models_main(tmp_path, monkeypatch):
    """Kiểm tra main() ghi đúng các file artifacts."""
    monkeypatch.setattr("scripts.compare_models.PROJECT_ROOT", tmp_path)
    main()

    csv_file = tmp_path / "reports" / "model_comparison.csv"
    md_file = tmp_path / "reports" / "model_comparison.md"

    assert csv_file.exists()
    assert md_file.exists()
    assert len(csv_file.read_text(encoding="utf-8-sig")) > 100
    assert len(md_file.read_text(encoding="utf-8")) > 500
