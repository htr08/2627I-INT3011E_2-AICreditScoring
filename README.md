# AI Credit Scoring: Dự đoán xác suất vỡ nợ (2627I-INT3011E_2)

Pipeline dự đoán xác suất vỡ nợ (Probability of Default, PD) trong tháng kế tiếp cho danh mục thẻ tín dụng hiện hữu, trên bộ dữ liệu Default of Credit Card Clients (Đài Loan, 2005; 30.000 hồ sơ, tỷ lệ vỡ nợ 22.12%). Dự án so sánh nhiều thuật toán có kiểm định thống kê, hiệu chuẩn PD, chọn ngưỡng theo chi phí, giải thích bằng SHAP và cung cấp ứng dụng demo Streamlit.

Mô hình dùng làm công cụ cảnh báo sớm và quản lý danh mục hiện hữu, không phải công cụ xét duyệt tín dụng mới, và không sẵn sàng triển khai thực tế (dữ liệu một ngân hàng, một snapshot năm 2005).

## Kết quả chính

| Hạng mục | Kết quả |
|---|---|
| Mô hình cuối | LightGBM tuned + ràng buộc đơn điệu, hiệu chuẩn Platt, không dùng `SEX` |
| ROC-AUC trên Test | 0.7825 (bootstrap 95% CI 0.7681–0.7973); baseline Logistic Regression 0.7609 |
| So với baseline | ΔAUC = 0.0215, DeLong p = 1.86 × 10⁻⁹ (KPI chính đạt) |
| ROC-AUC 5-fold CV (Train) | 0.7906 ± 0.0074 |
| Ngưỡng vận hành | PD ≥ 0.09: recall 92.46%, precision 28.02%, gắn cờ 72.98% hồ sơ Test |
| Phân tầng demo | PD < 9% Bình thường; 9% ≤ PD < 27% Theo dõi; PD ≥ 27% Cảnh báo cao |

Gini, KS và PR-AUC trên Test dưới mốc sàn tham chiếu của Charter (khác biệt nằm trong khoảng tin cậy). Chi tiết và các hạn chế: [reports/model_card.md](reports/model_card.md), [reports/final_report.md](reports/final_report.md).

## Cấu trúc repository

```
├── app/          # demo Streamlit (mô hình cuối, baseline, mock), phân tầng rủi ro, kiểm tra đầu vào
├── configs/      # config.yaml (chia dữ liệu, chi phí, PDO, ngưỡng, nhóm đặc trưng), feature_freeze_v1.yaml, tuned_params.yaml
├── data/
│   ├── raw/      # dữ liệu gốc (không commit; chỉ có file checksum)
│   └── splits/   # chỉ số dòng Train/Valid/Test (commit)
├── docs/         # hướng dẫn chạy chi tiết (GETTING_STARTED.md)
├── models/       # final_model.pkl: mô hình cuối cho demo
├── notebooks/    # EDA, mô hình hóa, SHAP, calibration, phân tích lỗi và fairness
├── reports/      # báo cáo, Model Card, kết quả đánh giá Test
├── scripts/      # chạy pipeline, tải và chia dữ liệu, phân tích lỗi, fairness, reason codes
├── src/          # preprocessing, features, train, tune, calibrate, scoring, evaluate, explain, fairness
└── tests/        # unit test (pytest)
```

## Cài đặt

Yêu cầu Python 3.11 hoặc 3.12 (các phiên bản pin trong `requirements.txt` chưa hỗ trợ 3.13). Mỗi thành viên làm việc trong một virtual environment riêng; không cài thư viện vào Python hệ thống.

```bash
python -m venv venv
# Windows: venv\Scripts\activate    |    Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Chạy mọi lệnh dưới đây từ thư mục gốc repo.

## Chạy lại dự án từ bản clone sạch

```bash
# 1. Tải dữ liệu vào data/raw/ (kiểm tra checksum SHA-256 với file .sha256 đã commit)
python scripts/download_data.py

# 2. Chỉ số chia dữ liệu đã có trong data/splits/splits.json; chỉ chạy lại khi cần tái tạo
python scripts/split_data.py

# 3. Huấn luyện baseline (Logistic Regression, Decision Tree), log MLflow vào mlruns/
python scripts/run_pipeline.py

# 4. Xuất mô hình cuối (chỉ dùng Train và Valid) tại models/final_model.pkl
python scripts/run_pipeline.py --mode final-model

# 5. Phân tích lỗi và fairness trên Valid
python scripts/analyze_errors.py
python scripts/analyze_fairness.py

# 6. Fairness trên Test, đọc dự đoán đã xuất (không chấm lại Test)
python scripts/analyze_fairness.py --split test

# 7. Demo
streamlit run app/streamlit_app.py

# 8. Kiểm thử
pytest
```

Các chế độ của `scripts/run_pipeline.py` (`--mode`):

| Chế độ | Nội dung |
|---|---|
| `baseline` (mặc định) | Logistic Regression và Decision Tree, 5-fold CV |
| `advanced`, `boosting` | Random Forest, XGBoost; LightGBM, CatBoost (tham số mặc định) |
| `imbalance` | `class_weight` và SMOTE trong từng fold |
| `woe`, `scorecard` | IV, tương quan, độ ổn định; Logistic Scorecard (WoE) |
| `compare`, `summary` | So sánh Scorecard với mô hình black-box; in bảng so sánh sơ bộ (CV) và ghi `reports/model_comparison.csv` |
| `tune` | Optuna cho LightGBM (có và không monotonic) và CatBoost |
| `tuned` | Huấn luyện lại mô hình đã tuning; thêm `--include-sex` cho bản đối chiếu fairness |
| `calibrate` | So sánh Platt/Isotonic, chọn ngưỡng theo chi phí, điểm PDO, lưu mô hình |
| `final-model` | Chỉ lưu `models/final_model.pkl`, không ghi đè các bảng calibration |

Thời gian tham khảo: `tune` khoảng 1 giờ, `tuned` khoảng 5 phút, `final-model` dưới 1 phút.

Kết quả MLflow xem bằng `mlflow ui --backend-store-uri ./mlruns` (thư mục `mlruns/` không commit, mỗi người tự chạy lại). Hướng dẫn từng bước và cách dùng server MLflow chung: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

**Tập Test chỉ được đánh giá một lần** (`scripts/evaluate_final_test.py`, có marker `reports/test_evaluation/test_evaluation_started.json` chặn chạy lại). Kết quả nghiệm thu và dự đoán từng hồ sơ nằm trong `reports/test_evaluation/`; không chạy lại script này.

## Demo Streamlit

App nạp mô hình cuối từ `models/final_model.pkl` và giải thích bằng `shap.TreeExplainer`; không cần `data/raw`. Người dùng nhập hồ sơ hoặc chọn hồ sơ mẫu (Bình thường, Ca biên, Cảnh báo cao); app kiểm tra đầu vào, trả PD, điểm tín dụng (thang PDO: 600 điểm tại odds 50:1, PDO = 20), mức cảnh báo kèm hành động đề xuất và đóng góp theo nhóm đặc trưng. Khi không nạp được mô hình cuối, app dùng baseline Logistic Regression từ MLflow Model Registry, rồi mock model kèm cảnh báo. Biến môi trường `DEMO_MODEL=final|baseline|mock` buộc dùng đúng mô hình. Bố cục và hợp đồng giao diện: [reports/demo_wireframe.md](reports/demo_wireframe.md).

## Báo cáo

| Nội dung | Tài liệu |
|---|---|
| Kế hoạch, Charter | [reports/project_plan.md](reports/project_plan.md), [reports/project_charter.md](reports/project_charter.md) |
| Báo cáo tổng kết | [reports/final_report.md](reports/final_report.md) |
| Model Card | [reports/model_card.md](reports/model_card.md) |
| Lọc đặc trưng, Scorecard | [reports/feature_selection_decision.md](reports/feature_selection_decision.md), [reports/logistic_scorecard.md](reports/logistic_scorecard.md) |
| So sánh mô hình, tuning | [reports/model_comparison.md](reports/model_comparison.md), [reports/experiments_optuna_tuning.md](reports/experiments_optuna_tuning.md), [reports/week2_review.md](reports/week2_review.md) |
| Hiệu chuẩn, ngưỡng, điểm | [reports/calibration_threshold.md](reports/calibration_threshold.md) |
| Đánh giá Test | [reports/test_evaluation/test_evaluation.md](reports/test_evaluation/test_evaluation.md) |
| XAI và reason codes | [reports/xai_report.md](reports/xai_report.md), [reports/shap_agreement.md](reports/shap_agreement.md) |
| Phân tích lỗi, fairness | [reports/error_analysis_validation.md](reports/error_analysis_validation.md), [reports/fairness_evaluation.md](reports/fairness_evaluation.md), [reports/fairness_evaluation_test.md](reports/fairness_evaluation_test.md) |

## Quy ước làm việc

Quy ước Git, quản lý tệp dữ liệu, đặt tên, tái lập và đánh giá Test: [reports/project_plan.md](reports/project_plan.md) mục 8.
