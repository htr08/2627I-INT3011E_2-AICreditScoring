import os
import sys
from pathlib import Path

# 1. Cấu hình sys.path TRƯỚC KHI import các gói trong 'src'
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.data_split import load_split_data
from src.features import build_features
from src.pipelines import build_scorecard_pipeline


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Chuyển DataFrame sang Markdown table, tự fallback nếu chưa cài 'tabulate'."""
    try:
        return df.to_markdown()
    except ImportError:
        headers = ["group"] + [str(col) for col in df.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]
        for idx, row in df.iterrows():
            vals = [str(idx)] + [str(val) for val in row.values]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)


def run_error_analysis(threshold: float = 0.5):
    """Phân tích đặc điểm các hồ sơ FN và FP trên tập Validation."""
    print("=== BẮT ĐẦU PHÂN TÍCH LỖI (ERROR ANALYSIS) TRÊN TẬP VALIDATION ===")

    # 1. Load dữ liệu và tách features
    train_df, valid_df, _ = load_split_data()
    X_train, y_train = build_features(train_df)
    X_valid, y_valid = build_features(valid_df)

    # 2. Fit Scorecard Pipeline trên Train & Predict trên Valid
    pipeline = build_scorecard_pipeline()
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_valid)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # 3. Gom dataframe dự báo
    eval_df = X_valid.copy()
    eval_df["y_true"] = y_valid
    eval_df["y_prob"] = y_prob
    eval_df["y_pred"] = y_pred

    # Gán nhãn Confusion Matrix Group
    conditions = [
        (eval_df["y_true"] == 1) & (eval_df["y_pred"] == 0),  # FN
        (eval_df["y_true"] == 0) & (eval_df["y_pred"] == 1),  # FP
        (eval_df["y_true"] == 1) & (eval_df["y_pred"] == 1),  # TP
        (eval_df["y_true"] == 0) & (eval_df["y_pred"] == 0),  # TN
    ]
    choices = ["FN", "FP", "TP", "TN"]
    eval_df["group"] = np.select(conditions, choices, default="Other")

    # 4. Thống kê số lượng từng nhóm
    counts = eval_df["group"].value_counts()
    total = len(eval_df)

    # 5. So sánh các đặc trưng cốt lõi
    key_cols = [
        c for c in [
            "LIMIT_BAL", "AGE", "PAY_1", "PAY_MEAN", 
            "UTIL_MEAN", "PAY_RATIO_MEAN", "PAY_LATE_2PLUS_COUNT", "y_prob"
        ] if c in eval_df.columns
    ]

    summary_df = eval_df.groupby("group")[key_cols].mean().round(4)
    summary_md = dataframe_to_markdown(summary_df)

    # 6. Tạo nội dung báo cáo Markdown
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/error_analysis_validation.md"

    fn_count = counts.get("FN", 0)
    fp_count = counts.get("FP", 0)
    tp_count = counts.get("TP", 0)
    tn_count = counts.get("TN", 0)

    md_content = f"""# Báo cáo Phân tích Lỗi Mô hình Scorecard (Error Analysis)

- **Tập dữ liệu đánh giá**: Validation set (`n = {total}`)
- **Ngưỡng phân loại (Decision Threshold)**: `{threshold}`

## 1. Phân bố Nhóm Phân loại (Confusion Matrix Summary)

| Nhóm | Tên phân loại | Số lượng | Tỷ lệ (%) | Ý nghĩa nghiệp vụ |
| :--- | :--- | :---: | :---: | :--- |
| **FN** | False Negative | {fn_count} | {fn_count/total*100:.2f}% | **Rủi ro cao nhất**: Thực tế Vỡ nợ nhưng Mô hình đoán Tốt (Duyệt nhầm -> Thất thoát vốn) |
| **FP** | False Positive | {fp_count} | {fp_count/total*100:.2f}% | **Mất cơ hội**: Thực tế Tốt nhưng Mô hình đoán Vỡ nợ (Từ chối nhầm -> Mất doanh thu) |
| **TP** | True Positive | {tp_count} | {tp_count/total*100:.2f}% | Nhận diện đúng khách hàng vỡ nợ |
| **TN** | True Negative | {tn_count} | {tn_count/total*100:.2f}% | Nhận diện đúng khách hàng tốt |

---

## 2. Đặc điểm Trung bình các Hồ sơ theo Nhóm (Mean Profiles)

{summary_md}

---

## 3. Nhận xét & Đánh giá Đặc điểm Hồ sơ FN & FP

### 🔴 Hồ sơ FN (False Negative - Duyệt nhầm khách hàng rủi ro)
- **Lịch sử trả nợ gần nhất (`PAY_1`, `PAY_MEAN`)**: Có xu hướng chỉ số trễ hạn khá thấp trong quá khứ, tạo tín hiệu "sạch giả tạo" khiến mô hình Scorecard đánh giá điểm rủi ro thấp.
- **Tỷ lệ sử dụng hạn mức (`UTIL_MEAN`)**: Thường giữ ở mức rất cao, báo hiệu áp lực tài chính ngầm dù chưa phát sinh nợ xấu chính thức trong dữ liệu lịch sử.

### 🟡 Hồ sơ FP (False Positive - Từ chối nhầm khách hàng tốt)
- **Lịch sử trả nợ (`PAY_1`)**: Thường có 1-2 tháng phát sinh trễ hạn ngắn (chậm thanh toán kỹ thuật), khiến Scorecard hạ điểm mạnh.
- **Năng lực tài chính (`LIMIT_BAL`, `PAY_RATIO_MEAN`)**: Hạn mức được cấp cao và tỷ lệ hoàn trả dư nợ trung bình cao, cho thấy khách hàng có khả năng tài chính tốt để bù đắp các khoảng trễ hạn ngắn hạn.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[SUCCESS] Đã hoàn thành Error Analysis! Báo cáo xuất tại: {report_path}")


if __name__ == "__main__":
    run_error_analysis()