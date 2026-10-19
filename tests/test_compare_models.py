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
    main,
)


def test_build_summary_table_df():
    """Kiểm tra cấu trúc bảng tóm tắt (chỉ số CV) theo mục 5 project_plan.md."""
    df = build_summary_table_df()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = [
        "Mô hình",
        "AUC CV (mean ± std)",
        "Gini (CV)",
        "KS (CV)",
        "PR-AUC (CV)",
        "Brier thô (CV)",
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

    # Kiểm tra metric đồng bộ của LightGBM default và CatBoost default
    lgb_row = df[df["Mô hình"] == "LightGBM (default)"].iloc[0]
    assert lgb_row["ROC-AUC (mean ± std)"] == "0.7805 ± 0.0077"
    assert lgb_row["Gini"] == "0.5610"

    cb_row = df[df["Mô hình"] == "CatBoost (default)"].iloc[0]
    assert cb_row["ROC-AUC (mean ± std)"] == "0.7844 ± 0.0074"
    assert cb_row["Gini"] == "0.5688"


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

    # CatBoost default vs baseline: p < 0.05, mean diff +0.0099
    cb_def_row = df[df["So sánh cặp"] == "CatBoost default vs Logistic baseline"].iloc[0]
    assert float(cb_def_row["p-value (paired)"]) < 0.05
    assert cb_def_row["Fold 1"] == "+0.0077"
    assert cb_def_row["Fold 2"] == "+0.0021"
    assert cb_def_row["Δ AUC TB"] == "+0.0099"

    # LightGBM default vs baseline: p > 0.05, mean diff +0.0060
    lgb_def_row = df[df["So sánh cặp"] == "LightGBM default vs Logistic baseline"].iloc[0]
    assert float(lgb_def_row["p-value (paired)"]) > 0.05
    assert lgb_def_row["Fold 1"] == "+0.0005"
    assert lgb_def_row["Fold 2"] == "-0.0007"
    assert lgb_def_row["Δ AUC TB"] == "+0.0060"

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


def test_compare_models_main(tmp_path, monkeypatch):
    """main() chỉ ghi bảng máy đọc và không động vào báo cáo model_comparison.md viết tay."""
    monkeypatch.setattr("scripts.compare_models.PROJECT_ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    md_file = reports / "model_comparison.md"
    md_file.write_text("báo cáo viết tay, có kết quả Test", encoding="utf-8")

    main()

    csv_file = reports / "model_comparison.csv"
    assert csv_file.exists()
    assert len(csv_file.read_text(encoding="utf-8-sig")) > 100
    assert md_file.read_text(encoding="utf-8") == "báo cáo viết tay, có kết quả Test"
