import os
import sys
from pathlib import Path

# Cấu hình sys.path trước khi import gói 'src'
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data_split import load_split_data
from src.features import build_features
from src.pipelines import build_scorecard_pipeline


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Chuyển DataFrame sang bảng Markdown, tự fallback nếu thiếu 'tabulate'."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        headers = [str(col) for col in df.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]
        for _, row in df.iterrows():
            vals = [str(val) for val in row.values]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)


def assign_age_group(age: float) -> str:
    """Phân nhóm tuổi theo yêu cầu nghiệp vụ."""
    if age < 30:
        return "<30"
    elif age <= 50:
        return "30–50"
    else:
        return ">50"


def calculate_group_metrics(df: pd.DataFrame, group_col: str, threshold: float = 0.5) -> pd.DataFrame:
    """Tính Flag Rate, AUC và Disparate Impact Ratio (DIR) theo nhóm."""
    results = []
    overall_flag_rate = (df["y_pred"] == 1).mean()

    for group_val, group_data in df.groupby(group_col):
        n_samples = len(group_data)
        flag_rate = (group_data["y_pred"] == 1).mean()
        dir_ratio = flag_rate / overall_flag_rate if overall_flag_rate > 0 else np.nan

        if group_data["y_true"].nunique() > 1:
            auc = roc_auc_score(group_data["y_true"], group_data["y_prob"])
        else:
            auc = np.nan

        warning = "⚠️ CẢNH BÁO (DIR < 0.8)" if (dir_ratio < 0.8 or dir_ratio > 1.25) else "✅ Đạt"

        results.append({
            "Subgroup": str(group_val),
            "N": n_samples,
            "Flag Rate": round(flag_rate, 4),
            "Disparate Impact Ratio": round(dir_ratio, 4) if not np.isnan(dir_ratio) else "N/A",
            "AUC": round(auc, 4) if not np.isnan(auc) else "N/A",
            "Fairness Status": warning
        })

    return pd.DataFrame(results)


def run_fairness_evaluation(threshold: float = 0.5):
    print("=== BẮT ĐẦU ĐÁNH GIÁ TÍNH CÔNG BẰNG (FAIRNESS CHECK) TRÊN TẬP VALIDATION ===")

    train_df, valid_df, _ = load_split_data()

    sex_map = {1: "Nam (1)", 2: "Nữ (2)"}
    valid_sex = valid_df["SEX"].map(sex_map).values if "SEX" in valid_df.columns else ["Unknown"] * len(valid_df)
    valid_age_group = valid_df["AGE"].apply(assign_age_group).values

    # 1. Mô hình Mặc định (KHÔNG CÓ SEX)
    X_tr_no_sex, y_tr_no_sex = build_features(train_df, include_sex=False)
    X_va_no_sex, y_va_no_sex = build_features(valid_df, include_sex=False)

    pipe_no_sex = build_scorecard_pipeline(drop_sensitive=True)
    pipe_no_sex.fit(X_tr_no_sex, y_tr_no_sex)

    prob_no_sex = pipe_no_sex.predict_proba(X_va_no_sex)[:, 1]
    pred_no_sex = (prob_no_sex >= threshold).astype(int)

    df_eval_no_sex = pd.DataFrame({
        "y_true": y_va_no_sex.values,
        "y_prob": prob_no_sex,
        "y_pred": pred_no_sex,
        "SEX": valid_sex,
        "AGE_GROUP": valid_age_group
    })

    # 2. Mô hình Đối chiếu (CÓ SEX)
    X_tr_sex, y_tr_sex = build_features(train_df, include_sex=True)
    X_va_sex, y_va_sex = build_features(valid_df, include_sex=True)

    pipe_sex = build_scorecard_pipeline(drop_sensitive=False)
    pipe_sex.fit(X_tr_sex, y_tr_sex)

    prob_sex = pipe_sex.predict_proba(X_va_sex)[:, 1]
    pred_sex = (prob_sex >= threshold).astype(int)

    df_eval_sex = pd.DataFrame({
        "y_true": y_va_sex.values,
        "y_prob": prob_sex,
        "y_pred": pred_sex,
        "SEX": valid_sex,
        "AGE_GROUP": valid_age_group
    })

    # 3. Tính chỉ số Fairness
    sex_fairness_no_sex = calculate_group_metrics(df_eval_no_sex, "SEX", threshold)
    sex_fairness_sex = calculate_group_metrics(df_eval_sex, "SEX", threshold)
    age_fairness_no_sex = calculate_group_metrics(df_eval_no_sex, "AGE_GROUP", threshold)
    age_fairness_sex = calculate_group_metrics(df_eval_sex, "AGE_GROUP", threshold)

    auc_total_no_sex = roc_auc_score(y_va_no_sex, prob_no_sex)
    auc_total_sex = roc_auc_score(y_va_sex, prob_sex)

    # 4. Xuất Báo cáo Markdown
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/fairness_evaluation.md"

    md_content = f"""# Báo cáo Đánh giá Tính Công bằng (Fairness Check Report)

- **Tập dữ liệu**: Validation set (`n = {len(valid_df)}`)
- **Ngưỡng quyết định (Decision Threshold)**: `{threshold}`
- **Ngưỡng cảnh báo Disparate Impact (DIR)**: `< 0.8` hoặc `> 1.25` (Quy tắc 80%)

---

## 1. Tổng quan So sánh Hiệu năng

| Phiên bản Mô hình | Cột nhạy cảm `SEX` | Overall AUC | Flag Rate Tổng thể |
| :--- | :---: | :---: | :---: |
| **Mô hình Mặc định (No SEX)** | Bị loại bỏ | **{auc_total_no_sex:.4f}** | {(pred_no_sex == 1).mean():.4f} |
| **Mô hình Đối chiếu (With SEX)** | Được giữ lại | **{auc_total_sex:.4f}** | {(pred_sex == 1).mean():.4f} |

---

## 2. Phân tích Theo Giới tính (SEX)

### A. Mô hình Mặc định (Không có `SEX`)
{dataframe_to_markdown(sex_fairness_no_sex)}

### B. Mô hình Đối chiếu (Có `SEX`)
{dataframe_to_markdown(sex_fairness_sex)}

---

## 3. Phân tích Theo Nhóm Tuổi (AGE)

### A. Mô hình Mặc định (Không có `SEX`)
{dataframe_to_markdown(age_fairness_no_sex)}

### B. Mô hình Đối chiếu (Có `SEX`)
{dataframe_to_markdown(age_fairness_sex)}

---

## 4. Kết luận & Khuyến nghị Nghiệp vụ

1. **Tác động của `SEX`**: Việc loại bỏ biến `SEX` khỏi mô hình chính thức đảm bảo tính tuân thủ pháp lý và phòng ngừa rủi ro phân biệt đối xử trực tiếp.
2. **Đánh giá Disparate Impact (DIR)**: Các sub-group được kiểm soát bằng chỉ số DIR (ngưỡng 0.8) để phát hiện và xử lý kịp thời chênh lệch can thiệp rủi ro.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[SUCCESS] Đã hoàn thành Fairness Check! Báo cáo xuất tại: {report_path}")


if __name__ == "__main__":
    run_fairness_evaluation()