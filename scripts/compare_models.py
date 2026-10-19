"""Tổng hợp bảng so sánh sơ bộ (CV mean ± std) của các mô hình.

Task: Tuần 2 – T6: Thành viên B (ML Engineer).
Đầu ra: in bảng so sánh và kiểm định cặp ra console; ghi bảng máy đọc reports/model_comparison.csv.
Không ghi reports/model_comparison.md: báo cáo đó được viết tay (gồm kết quả Test ở mục 2.1) và script không được ghi đè.

Usage:
    python scripts/compare_models.py
    python scripts/run_pipeline.py --mode summary
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Đảm bảo console Windows in tiếng Việt UTF-8 không lỗi
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Bộ dữ liệu kết quả thực nghiệm 5-fold Stratified CV trên Train (18.000 mẫu, 22.1% default rate, random_state=42)
# thu thập từ các giai đoạn T1 -> T5 (Tuần 1 & Tuần 2).
# Tập Test không nằm trong bảng này; kết quả Test: reports/test_evaluation/ và reports/model_comparison.md mục 2.1.
CV_MODEL_RESULTS: List[Dict] = [
    {
        "model_key": "logreg_baseline",
        "name": "Logistic Regression (baseline)",
        "category": "Baseline",
        "roc_auc_mean": 0.7745,
        "roc_auc_std": 0.0116,
        "gini_mean": 0.5490,
        "ks_mean": 0.4282,
        "pr_auc_mean": 0.5470,
        "brier_mean": 0.1354,
        "recall_05": 0.3619,
        "precision_05": 0.6760,
        "f1_05": 0.4714,
        "train_time": "~2.5s",
        "interpretability": "Rất cao (hệ số odds ratio trực tiếp)",
        "is_main_candidate": False,
        "note": "Baseline chuẩn Tuần 1",
    },
    {
        "model_key": "dt_baseline",
        "name": "Decision Tree (max_depth=5)",
        "category": "Baseline",
        "roc_auc_mean": 0.7672,
        "roc_auc_std": 0.0120,
        "gini_mean": 0.5345,
        "ks_mean": 0.4162,
        "pr_auc_mean": 0.5242,
        "brier_mean": 0.1367,
        "recall_05": 0.3531,
        "precision_05": 0.6629,
        "f1_05": 0.4595,
        "train_time": "~1.8s",
        "interpretability": "Cao (cây quyết định if-then rõ ràng)",
        "is_main_candidate": False,
        "note": "Baseline cây quyết định",
    },
    {
        "model_key": "logistic_scorecard",
        "name": "Logistic Scorecard (WoE)",
        "category": "Scorecard",
        "roc_auc_mean": 0.7790,
        "roc_auc_std": 0.0083,
        "gini_mean": 0.5580,
        "ks_mean": 0.4326,
        "pr_auc_mean": 0.5432,
        "brier_mean": 0.1351,
        "recall_05": 0.3744,
        "precision_05": 0.6596,
        "f1_05": 0.4777,
        "train_time": "~4.2s",
        "interpretability": "Rất cao (bảng điểm PDO, chuẩn ngân hàng)",
        "is_main_candidate": True,
        "note": "Scorecard chuẩn mực (coarse classing + ràng buộc dấu)",
    },
    {
        "model_key": "rf_default",
        "name": "Random Forest (default)",
        "category": "Tree Ensembles",
        "roc_auc_mean": 0.7667,
        "roc_auc_std": 0.0097,
        "gini_mean": 0.5333,
        "ks_mean": 0.4107,
        "pr_auc_mean": 0.5319,
        "brier_mean": 0.1388,
        "recall_05": 0.3870,
        "precision_05": 0.6313,
        "f1_05": 0.4797,
        "train_time": "~8.5s",
        "interpretability": "Trung bình (SHAP / Feature Importance)",
        "is_main_candidate": False,
        "note": "Tree ensemble mặc định T2",
    },
    {
        "model_key": "xgboost_default",
        "name": "XGBoost (default)",
        "category": "Tree Ensembles",
        "roc_auc_mean": 0.7597,
        "roc_auc_std": 0.0073,
        "gini_mean": 0.5195,
        "ks_mean": 0.4007,
        "pr_auc_mean": 0.5258,
        "brier_mean": 0.1440,
        "recall_05": 0.3669,
        "precision_05": 0.6077,
        "f1_05": 0.4575,
        "train_time": "~5.2s",
        "interpretability": "Trung bình (SHAP TreeExplainer)",
        "is_main_candidate": False,
        "note": "XGBoost mặc định T2",
    },
    {
        "model_key": "lightgbm_default",
        "name": "LightGBM (default)",
        "category": "Gradient Boosting (T3)",
        "roc_auc_mean": 0.7805,
        "roc_auc_std": 0.0077,
        "gini_mean": 0.5610,
        "ks_mean": 0.4276,
        "pr_auc_mean": 0.5512,
        "brier_mean": 0.1350,
        "recall_05": 0.3767,
        "precision_05": 0.6677,
        "f1_05": 0.4816,
        "train_time": "~3.5s",
        "interpretability": "Trung bình (SHAP TreeExplainer)",
        "is_main_candidate": False,
        "note": "Boosting mặc định (đồng bộ fold-cache Optuna T4)",
    },
    {
        "model_key": "catboost_default",
        "name": "CatBoost (default)",
        "category": "Gradient Boosting (T3)",
        "roc_auc_mean": 0.7844,
        "roc_auc_std": 0.0074,
        "gini_mean": 0.5688,
        "ks_mean": 0.4359,
        "pr_auc_mean": 0.5584,
        "brier_mean": 0.1342,
        "recall_05": 0.3784,
        "precision_05": 0.6696,
        "f1_05": 0.4836,
        "train_time": "~28.0s",
        "interpretability": "Trung bình (SHAP TreeExplainer)",
        "is_main_candidate": False,
        "note": "Boosting mặc định (đồng bộ fold-cache Optuna T4)",
    },
    {
        "model_key": "lightgbm_balanced",
        "name": "LightGBM Balanced (class_weight)",
        "category": "Imbalance Experiments",
        "roc_auc_mean": 0.7809,
        "roc_auc_std": 0.0081,
        "gini_mean": 0.5617,
        "ks_mean": 0.4278,
        "pr_auc_mean": 0.5505,
        "brier_mean": 0.1682,
        "recall_05": 0.6110,
        "precision_05": 0.4758,
        "f1_05": 0.5350,
        "train_time": "~3.8s",
        "interpretability": "Trung bình (SHAP, méo xác suất thô)",
        "is_main_candidate": False,
        "note": "Recall cao nhưng Brier tăng mạnh",
    },
    {
        "model_key": "catboost_balanced",
        "name": "CatBoost Balanced (auto_class_weights)",
        "category": "Imbalance Experiments",
        "roc_auc_mean": 0.7806,
        "roc_auc_std": 0.0081,
        "gini_mean": 0.5611,
        "ks_mean": 0.4341,
        "pr_auc_mean": 0.5520,
        "brier_mean": 0.1659,
        "recall_05": 0.6072,
        "precision_05": 0.4872,
        "f1_05": 0.5406,
        "train_time": "~30.0s",
        "interpretability": "Trung bình (SHAP, méo xác suất thô)",
        "is_main_candidate": False,
        "note": "Recall cao nhưng Brier tăng mạnh",
    },
    {
        "model_key": "lightgbm_smote",
        "name": "LightGBM + SMOTE (in-fold)",
        "category": "Imbalance Experiments",
        "roc_auc_mean": 0.7776,
        "roc_auc_std": 0.0058,
        "gini_mean": 0.5551,
        "ks_mean": 0.4216,
        "pr_auc_mean": 0.5440,
        "brier_mean": 0.1383,
        "recall_05": 0.4309,
        "precision_05": 0.6222,
        "f1_05": 0.5092,
        "train_time": "~7.2s",
        "interpretability": "Trung bình (SHAP)",
        "is_main_candidate": False,
        "note": "Thí nghiệm phụ SMOTE: AUC giảm",
    },
    {
        "model_key": "catboost_smote",
        "name": "CatBoost + SMOTE (in-fold)",
        "category": "Imbalance Experiments",
        "roc_auc_mean": 0.7747,
        "roc_auc_std": 0.0057,
        "gini_mean": 0.5493,
        "ks_mean": 0.4192,
        "pr_auc_mean": 0.5410,
        "brier_mean": 0.1386,
        "recall_05": 0.4156,
        "precision_05": 0.6130,
        "f1_05": 0.4953,
        "train_time": "~54.0s",
        "interpretability": "Trung bình (SHAP)",
        "is_main_candidate": False,
        "note": "Thí nghiệm phụ SMOTE: AUC giảm",
    },
    {
        "model_key": "lightgbm_tuned",
        "name": "LightGBM tuned (Optuna)",
        "category": "Optuna Tuned (T4)",
        "roc_auc_mean": 0.7898,
        "roc_auc_std": 0.0077,
        "gini_mean": 0.5796,
        "ks_mean": 0.4439,
        "pr_auc_mean": 0.5650,
        "brier_mean": 0.1327,
        "recall_05": 0.3774,
        "precision_05": 0.6696,
        "f1_05": 0.4828,
        "train_time": "~12.5s",
        "interpretability": "Trung bình (SHAP TreeExplainer)",
        "is_main_candidate": False,
        "note": "Tuned 37 trials, plateau stopping",
    },
    {
        "model_key": "lightgbm_monotone_tuned",
        "name": "LightGBM tuned + Monotonic",
        "category": "Optuna Tuned (T4) - Candidate",
        "roc_auc_mean": 0.7906,
        "roc_auc_std": 0.0074,
        "gini_mean": 0.5813,
        "ks_mean": 0.4453,
        "pr_auc_mean": 0.5624,
        "brier_mean": 0.1328,
        "recall_05": 0.3774,
        "precision_05": 0.6748,
        "f1_05": 0.4839,
        "train_time": "~14.0s",
        "interpretability": "Trung bình - Cao (SHAP + Đơn điệu nghiệp vụ)",
        "is_main_candidate": True,
        "note": "ỨNG VIÊN CHÍNH: Nhanh, bảo đảm chiều tác động, AUC đạt mục tiêu",
    },
    {
        "model_key": "catboost_tuned",
        "name": "CatBoost tuned (Optuna)",
        "category": "Optuna Tuned (T4) - Candidate",
        "roc_auc_mean": 0.7911,
        "roc_auc_std": 0.0073,
        "gini_mean": 0.5822,
        "ks_mean": 0.4452,
        "pr_auc_mean": 0.5675,
        "brier_mean": 0.1327,
        "recall_05": 0.3674,
        "precision_05": 0.6754,
        "f1_05": 0.4759,
        "train_time": "~66.0s",
        "interpretability": "Trung bình (SHAP TreeExplainer)",
        "is_main_candidate": True,
        "note": "ỨNG VIÊN ĐỐI CHỨNG: AUC CV cao nhất toàn dự án (0.7911)",
    },
    {
        "model_key": "catboost_monotone_tuned",
        "name": "CatBoost tuned + Monotonic (đối chiếu)",
        "category": "Optuna Tuned (T4)",
        "roc_auc_mean": 0.7910,
        "roc_auc_std": 0.0077,
        "gini_mean": 0.5820,
        "ks_mean": 0.4434,
        "pr_auc_mean": 0.5632,
        "brier_mean": 0.1328,
        "recall_05": 0.3722,
        "precision_05": 0.6675,
        "f1_05": 0.4780,
        "train_time": "~318.0s",
        "interpretability": "Trung bình (SHAP + Đơn điệu nghiệp vụ)",
        "is_main_candidate": False,
        "note": "Thời gian train lâu ~5x so với CatBoost thường",
    },
    {
        "model_key": "lightgbm_monotone_tuned_with_sex",
        "name": "LightGBM tuned + Monotonic (with SEX)",
        "category": "Fairness Comparison (T5)",
        "roc_auc_mean": 0.7907,
        "roc_auc_std": 0.0075,
        "gini_mean": 0.5814,
        "ks_mean": 0.4465,
        "pr_auc_mean": 0.5634,
        "brier_mean": 0.1327,
        "recall_05": 0.3769,
        "precision_05": 0.6708,
        "f1_05": 0.4828,
        "train_time": "~14.5s",
        "interpretability": "Trung bình (Chỉ dùng đối chiếu Fairness)",
        "is_main_candidate": False,
        "note": "SEX chỉ chiếm 0.17% feature importance; delta AUC +0.0001",
    },
    {
        "model_key": "catboost_tuned_with_sex",
        "name": "CatBoost tuned (with SEX)",
        "category": "Fairness Comparison (T5)",
        "roc_auc_mean": 0.7912,
        "roc_auc_std": 0.0079,
        "gini_mean": 0.5824,
        "ks_mean": 0.4461,
        "pr_auc_mean": 0.5681,
        "brier_mean": 0.1326,
        "recall_05": 0.3692,
        "precision_05": 0.6723,
        "f1_05": 0.4767,
        "train_time": "~68.0s",
        "interpretability": "Trung bình (Chỉ dùng đối chiếu Fairness)",
        "is_main_candidate": False,
        "note": "SEX chỉ chiếm 0.54% feature importance; delta AUC +0.0001",
    },
]

# Chênh lệch AUC từng fold so với baseline Logistic Regression
# Thống nhất số liệu Boosting default tính trên fold-cache Optuna T4 (LGBM: 0.7805, CB: 0.7844)
PAIRED_FOLD_DIFFS = {
    "CatBoost tuned vs Logistic baseline": [0.0145, 0.0091, 0.0196, 0.0236, 0.0162],
    "LightGBM monotonic vs Logistic baseline": [0.0144, 0.0077, 0.0185, 0.0244, 0.0157],
    "CatBoost default vs Logistic baseline": [0.0077, 0.0021, 0.0125, 0.0176, 0.0098],
    "LightGBM default vs Logistic baseline": [0.0005, -0.0007, 0.0075, 0.0109, 0.0116],
    "Logistic Scorecard vs Logistic baseline": [-0.0022, -0.0051, 0.0040, 0.0191, 0.0068],
    "CatBoost tuned vs LightGBM monotonic": [0.0001, 0.0015, 0.0011, -0.0008, 0.0004],
}


def build_summary_table_df() -> pd.DataFrame:
    """Tạo bảng so sánh sơ bộ (chỉ số CV) theo mẫu mục 5 project_plan.md; cột Test xem reports/model_comparison.md mục 2.1."""
    rows = []
    # Chọn danh sách các mô hình tiêu biểu xuất hiện trong mẫu Section 5
    key_models = [
        "logreg_baseline",
        "logistic_scorecard",
        "dt_baseline",
        "rf_default",
        "xgboost_default",
        "lightgbm_default",
        "lightgbm_monotone_tuned",
        "catboost_default",
        "catboost_tuned",
    ]

    for item in CV_MODEL_RESULTS:
        if item["model_key"] in key_models:
            rows.append({
                "Mô hình": item["name"],
                "AUC CV (mean ± std)": f"{item['roc_auc_mean']:.4f} ± {item['roc_auc_std']:.4f}",
                "Gini (CV)": f"{item['gini_mean']:.4f}",
                "KS (CV)": f"{item['ks_mean']:.4f}",
                "PR-AUC (CV)": f"{item['pr_auc_mean']:.4f}",
                "Brier thô (CV)": f"{item['brier_mean']:.4f}",
                "Khả năng giải thích": item["interpretability"],
            })

    # Dòng cho MLP tuỳ chọn theo đúng mẫu Section 5 project_plan.md
    rows.append({
        "Mô hình": "MLP (tuỳ chọn)",
        "AUC CV (mean ± std)": "Chưa triển khai (tùy chọn)",
        "Gini (CV)": "–",
        "KS (CV)": "–",
        "PR-AUC (CV)": "–",
        "Brier thô (CV)": "–",
        "Khả năng giải thích": "Thấp (KernelExplainer)",
    })

    return pd.DataFrame(rows)


def build_full_comparison_df() -> pd.DataFrame:
    """Tạo bảng đối chiếu toàn bộ các mô hình và thử nghiệm kỹ thuật."""
    rows = []
    for item in CV_MODEL_RESULTS:
        rows.append({
            "Mô hình": item["name"],
            "Nhóm": item["category"],
            "ROC-AUC (mean ± std)": f"{item['roc_auc_mean']:.4f} ± {item['roc_auc_std']:.4f}",
            "Gini": f"{item['gini_mean']:.4f}",
            "KS": f"{item['ks_mean']:.4f}",
            "PR-AUC": f"{item['pr_auc_mean']:.4f}",
            "Brier Score": f"{item['brier_mean']:.4f}",
            "Recall@0.5": f"{item['recall_05']:.4f}",
            "Precision@0.5": f"{item['precision_05']:.4f}",
            "F1-Score": f"{item['f1_05']:.4f}",
            "Ghi chú": item["note"],
        })
    return pd.DataFrame(rows)


def compute_statistical_tests() -> pd.DataFrame:
    """Tính kiểm định t-test cặp đôi và 95% CI trên các fold so với baseline."""
    rows = []
    for comp_name, diffs in PAIRED_FOLD_DIFFS.items():
        s = pd.Series(diffs)
        mean_diff = s.mean()
        std_diff = s.std(ddof=1)
        t_stat, p_val = stats.ttest_1samp(diffs, 0)
        se = stats.sem(diffs)
        ci = stats.t.interval(0.95, df=len(diffs) - 1, loc=mean_diff, scale=se)
        ci_str = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"

        rows.append({
            "So sánh cặp": comp_name,
            "Fold 1": f"{diffs[0]:+.4f}",
            "Fold 2": f"{diffs[1]:+.4f}",
            "Fold 3": f"{diffs[2]:+.4f}",
            "Fold 4": f"{diffs[3]:+.4f}",
            "Fold 5": f"{diffs[4]:+.4f}",
            "Δ AUC TB": f"{mean_diff:+.4f}",
            "Std Δ": f"{std_diff:.4f}",
            "95% CI (Δ AUC)": ci_str,
            "t-statistic": f"{t_stat:.3f}",
            "p-value (paired)": f"{p_val:.5f}",
            "Ý nghĩa (p < 0.05)": "Có ý nghĩa (***)" if p_val < 0.01 else ("Có ý nghĩa (*)" if p_val < 0.05 else "Không"),
        })
    return pd.DataFrame(rows)


def check_charter_kpis() -> pd.DataFrame:
    """Đối chiếu hiệu năng CV với mốc KPI trong Project Charter."""
    # Mốc KPI Charter (project_charter.md / project_plan.md mục 1.3):
    # Sàn: ROC-AUC >= 0.78, Gini >= 0.57, KS >= 0.45, PR-AUC >= 0.56
    # Mục tiêu: ROC-AUC >= 0.79, Gini >= 0.59, KS >= 0.47, PR-AUC >= 0.58
    candidates = [m for m in CV_MODEL_RESULTS if m["model_key"] in ("logreg_baseline", "logistic_scorecard", "lightgbm_monotone_tuned", "catboost_tuned")]
    rows = []
    for m in candidates:
        auc = m["roc_auc_mean"]
        gini = m["gini_mean"]
        ks = m["ks_mean"]
        pr = m["pr_auc_mean"]

        def eval_metric(val, floor_val, target_val):
            if val >= target_val:
                return f"Đạt mục tiêu ({val:.4f} ≥ {target_val})"
            elif val >= floor_val:
                return f"Đạt sàn ({val:.4f} ≥ {floor_val})"
            else:
                return f"Chưa đạt sàn ({val:.4f} < {floor_val})"

        rows.append({
            "Mô hình": m["name"],
            "ROC-AUC (Sàn 0.78 / Mục tiêu 0.79)": eval_metric(auc, 0.78, 0.79),
            "Gini (Sàn 0.57 / Mục tiêu 0.59)": eval_metric(gini, 0.57, 0.59),
            "KS (Sàn 0.45 / Mục tiêu 0.47)": eval_metric(ks, 0.45, 0.47),
            "PR-AUC (Sàn 0.56 / Mục tiêu 0.58)": eval_metric(pr, 0.56, 0.58),
        })
    return pd.DataFrame(rows)


def main():
    """In bảng so sánh sơ bộ, kiểm định cặp, đối chiếu KPI và ghi bảng máy đọc."""
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_df = build_summary_table_df()
    full_df = build_full_comparison_df()
    stat_df = compute_statistical_tests()
    kpi_df = check_charter_kpis()

    rule = "-" * 80
    for title, table in [
        ("BẢNG SO SÁNH SƠ BỘ CÁC MÔ HÌNH (5-FOLD CV TRÊN TRAIN) — TUẦN 2 T6", summary_df),
        ("KIỂM ĐỊNH THỐNG KÊ (PAIRED T-TEST) SO VỚI BASELINE LOGISTIC REGRESSION", stat_df),
        ("ĐỐI CHIẾU KPI CHARTER (CV)", kpi_df),
    ]:
        print(f"{rule}\n {title}\n{rule}\n")
        print(table.to_string(index=False))
        print()

    csv_path = reports_dir / "model_comparison.csv"
    full_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Đã xuất dữ liệu máy đọc ra: {csv_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
