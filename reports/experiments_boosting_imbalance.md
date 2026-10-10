# Báo cáo Thực nghiệm: Huấn luyện Gradient Boosting & Xử lý Mất cân bằng

_Tuần 2 – T3: Thành viên B (ML Engineer)_

---

## 1. Mục tiêu và Phạm vi Thực nghiệm

Theo kế hoạch tại `reports/project_plan.md` (mục 3.3, 3.5c và phân công Tuần 2 – T3):
1. **Huấn luyện Gradient Boosting nâng cao:** Xây dựng pipeline và huấn luyện LightGBM và CatBoost với tham số mặc định trên 5-fold Stratified CV (tập Train 18.000 mẫu, feature freeze v1).
2. **Thực nghiệm Cân bằng Lớp (Cost/Weight-based):** Thử nghiệm cơ chế gán trọng số lớp (`class_weight="balanced"` trên LightGBM và `auto_class_weights="Balanced"` trên CatBoost).
3. **Thí nghiệm phụ Resampling (SMOTE in-fold):** Tích hợp SMOTE bên trong từng fold cross-validation thông qua `imblearn.pipeline.Pipeline`, đảm bảo:
   - SMOTE chỉ `fit_resample` trên dữ liệu huấn luyện của từng fold.
   - Tập validation trong từng fold hoàn toàn không bị resample hay rò rỉ dữ liệu (không data leakage).

---

## 2. Bảng Tổng hợp Kết quả Đánh giá (5-fold Stratified CV)

Tất cả các mô hình được đánh giá trên cùng tập Train 18.000 mẫu (22.1% vỡ nợ), 5-fold Stratified CV cố định `random_state = 42`. Pipeline tiền xử lý tuân thủ chuẩn `feature_freeze_v1` (mã bất thường được làm sạch, biến phân loại one-hot, AGE binning theo IV, loại bỏ biến nhạy cảm `SEX` và `ID`).

| Nhóm | Mô hình / Kỹ thuật | ROC-AUC (mean ± std) | KS (mean) | Gini (mean) | Brier Score | Recall | Precision | F1-Score |
|---|---|---|---|---|---|---|---|---|
| **Baseline** | Logistic Regression (`logreg_baseline`) | 0.7745 ± 0.0116 | 0.4282 | 0.5490 | 0.1354 | 0.3235 | 0.6728 | 0.4369 |
| | Decision Tree, max_depth=5 (`dt_baseline`) | 0.7672 ± 0.0120 | 0.4162 | 0.5345 | 0.1367 | 0.3541 | 0.6385 | 0.4555 |
| **Scorecard**| Logistic Scorecard WoE ¹ | 0.7790 ± 0.0083 | 0.4326 | 0.5580 | 0.1351 | 0.3744 | 0.6596 | 0.4777 |
| **Tree Ensembles (T2)**| Random Forest (`rf_default`) | 0.7667 ± 0.0097 | 0.4107 | 0.5333 | 0.1388 | 0.3644 | 0.6321 | 0.4624 |
| | XGBoost (`xgboost_default`) | 0.7597 ± 0.0073 | 0.4007 | 0.5195 | 0.1440 | 0.3601 | 0.6120 | 0.4531 |
| **Gradient Boosting (T3)**| **LightGBM Default** (`lightgbm_default`) | **0.7813 ± 0.0064** | **0.4276** | **0.5626** | **0.1350** | 0.3767 | 0.6677 | 0.4816 |
| | **CatBoost Default** (`catboost_default`) | **0.7853 ± 0.0077** | **0.4359** | **0.5706** | **0.1342** | 0.3784 | 0.6696 | 0.4836 |
| **Class Weighting (T3)**| LightGBM Balanced (`lightgbm_balanced`) | 0.7809 ± 0.0081 | 0.4278 | 0.5617 | 0.1682 | **0.6110** | 0.4758 | **0.5350** |
| | CatBoost Balanced (`catboost_balanced`) | 0.7806 ± 0.0081 | 0.4341 | 0.5611 | 0.1659 | **0.6072** | 0.4872 | **0.5406** |
| **SMOTE In-Fold (T3)**| LightGBM + SMOTE (`lightgbm_smote`) | 0.7776 ± 0.0058 | 0.4216 | 0.5551 | 0.1383 | 0.4309 | 0.6222 | 0.5092 |
| | CatBoost + SMOTE (`catboost_smote`) | 0.7747 ± 0.0057 | 0.4192 | 0.5493 | 0.1386 | 0.4156 | 0.6130 | 0.4953 |

¹ Cập nhật theo bản Scorecard chính thức (ràng buộc dấu + coarse classing, số liệu từ MLflow). Số 0.7809 ghi ở thời điểm T3 là của cấu hình LR 5 bin trước khi chốt.

---

## 3. Phân tích Chi tiết và Đánh giá Chuyên môn

### 3.1. Hiệu năng vượt trội của CatBoost và LightGBM
- **CatBoost Default thiết lập kỷ lục mới:** Đạt ROC-AUC **0.7853 ± 0.0077** và KS **0.4359**, trở thành mô hình có năng lực phân tách rủi ro cao nhất toàn bộ dự án tới thời điểm này, vượt qua Logistic Scorecard WoE (0.7790) và baseline Logistic Regression (0.7745).
- **Vượt mốc sàn KPI Charter:** Cả CatBoost (0.7853) và LightGBM (0.7813) đều vượt mốc sàn ROC-AUC $\ge 0.78$ ngay từ bộ tham số mặc định trước khi tuning.
- **Brier Score tối ưu:** CatBoost Default đạt Brier score thấp nhất (0.1342), cho thấy độ chính xác dự báo xác suất thô tốt hơn so với baseline và XGBoost.

### 3.2. Ảnh hưởng của Cơ chế Cân bằng Lớp (`class_weight`)
- **Tăng vọt Recall:** Ở ngưỡng phân loại mặc định 0.5, việc kích hoạt trọng số cân bằng lớp làm Recall tăng từ ~37.7% lên **61.1%** (LightGBM) và **60.7%** (CatBoost) — tức tăng hơn 60% khả năng bắt trúng các hồ sơ vỡ nợ. F1-score cũng tăng từ ~0.48 lên **0.54**.
- **ROC-AUC được bảo toàn:** ROC-AUC hầu như không bị ảnh hưởng tiêu cực (LightGBM: 0.7813 → 0.7809; CatBoost: 0.7853 → 0.7806, chênh lệch $< 0.005$, hoàn toàn trong phạm vi độ lệch chuẩn).
- **Méo xác suất đầu ra (Brier Score tăng):** Brier score tăng mạnh từ 0.134 lên 0.166–0.168. Đúng như phân tích tại `project_plan.md` mục 3.5c: gán trọng số đẩy phân phối xác suất về phía 0.5, làm xác suất đầu ra không còn phản ánh đúng tỷ lệ vỡ nợ thực tế (22%). **Hệ quả kỹ thuật:** Nếu sử dụng mô hình có class_weight, bắt buộc phải qua bước Probability Calibration (Platt Scaling hoặc Isotonic) trên tập Valid ở Tuần 3.

### 3.3. Đánh giá Thí nghiệm phụ SMOTE in-fold
- **Không cải thiện khả năng phân tách:** Cả 2 mô hình đều bị giảm ROC-AUC khi dùng SMOTE (LightGBM giảm từ 0.7813 xuống 0.7776; CatBoost giảm từ 0.7853 xuống 0.7747).
- **Hiệu quả phát hiện kém hơn Class Weighting:** Recall của SMOTE chỉ đạt 41–43% (kém xa mức 61% của `class_weight`).
- **Chi phí tính toán cao:** SMOTE nhân đôi số lượng mẫu thiểu số trong từng fold, khiến thời gian huấn luyện tăng gần 2 lần.
- **Kết luận kiến trúc:** Xác nhận định hướng trong `project_plan.md`: với dữ liệu bảng tín dụng và mức mất cân bằng nhẹ (~22%), **SMOTE không mang lại lợi ích** do tạo mẫu nội suy trong không gian nhiều chiều làm mờ biên quyết định. Không chọn SMOTE cho mô hình chính thức.

---

## 4. Kiến trúc Mã nguồn đã Triển khai

1. **`src/pipelines.py`:**
   - Cập nhật hàm `make_pipeline(estimator, include_sex=False, sampler=None)` tích hợp `imblearn.pipeline.Pipeline` khi có `sampler`.
   - Đảm bảo tính đóng gói không rò rỉ: sampler chỉ thực thi trong bước huấn luyện fold, tập validation giữ nguyên cấu trúc chuẩn.
2. **`src/train.py`:**
   - Bổ sung factory `get_boosting_default_models()` (LightGBM, CatBoost default).
   - Bổ sung factory `get_imbalance_weighted_models()` (LightGBM balanced, CatBoost balanced).
   - Nâng cấp `_train_and_log()` hỗ trợ sampler, ghi nhận tham số `sampler` và tag `imbalance_strategy` vào MLflow.
   - Thêm 2 hàm thực thi: `train_boosting_default()` và `train_imbalance_experiments()`.
3. **`scripts/run_pipeline.py`:**
   - Mở rộng CLI runner với các mode:
     - `python scripts/run_pipeline.py --mode boosting`
     - `python scripts/run_pipeline.py --mode imbalance`
     - `python scripts/run_pipeline.py --mode all`
4. **`tests/test_train.py`:**
   - Thêm 5 test case toàn diện, kiểm thử toàn bộ factory, tích hợp pipeline imblearn, và ghi nhận MLflow đúng định dạng.

---

## 5. Khuyến nghị cho Tuần 2 – T4 & T5 (Tuning & Lựa chọn Mô hình cuối)

1. **Lựa chọn 2 mô hình tốt nhất để Tuning bằng Optuna (Tuần 2 – T4):**
   - **Ứng viên số 1:** `CatBoost` (AUC 0.7853) — Khả năng xử lý biến phân loại và độ ổn định tốt nhất.
   - **Ứng viên số 2:** `LightGBM` (AUC 0.7813) — Tốc độ huấn luyện cực nhanh, AUC cao, tương thích tốt với monotonic constraints.
2. **Chiến lược xác suất cho Demo & XAI:**
   - Huấn luyện bản unweighted hoặc kết hợp Platt scaling trên `output_margin=True` để phục vụ giải thích SHAP tuyến tính chính xác và điểm tín dụng PDO theo chuẩn Charter.
