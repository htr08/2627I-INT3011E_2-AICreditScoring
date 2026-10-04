# PROJECT CHARTER

_Dự đoán xác suất vỡ nợ (Probability of Default) – So sánh mô hình & Giải thích bằng SHAP_

> Chốt tại Tuần 1 – T2. Mọi thay đổi nội dung Charter sau khi chốt phải được cả 3 thành viên thống nhất. Tham chiếu đầy đủ: [project_plan.md](project_plan.md).

## 1. Mục tiêu dự án

Xây dựng pipeline dự đoán xác suất vỡ nợ (PD) đã được hiệu chuẩn, so sánh nhiều thuật toán học máy một cách có kiểm định thống kê, áp dụng SHAP để giải thích mô hình và triển khai một ứng dụng demo hoàn chỉnh.

## 2. Phạm vi và sản phẩm đầu ra (Scope & Deliverables)

- **Source code:** Repository GitHub hoàn chỉnh, tái chạy được từ đầu (reproducible), bao gồm script tải dữ liệu tự động.
- **Bảng so sánh:** Hiệu năng các mô hình theo các chỉ số chuẩn hóa, kèm độ lệch chuẩn qua CV và khoảng tin cậy trên tập Test.
- **Báo cáo XAI:** Phân tích giải thích mô hình bằng SHAP (toàn cục, cục bộ, reason codes).
- **Model Card:** Mô tả mục đích sử dụng, dữ liệu, hiệu năng, kết quả fairness và các hạn chế của mô hình.
- **Ứng dụng Demo:** Web app tương tác bằng Streamlit.
- **Slide thuyết trình:** Bộ slide tổng kết dự án.

### 2.1. Scope tối thiểu bắt buộc (MVP)

**Logistic Regression + XGBoost + SHAP + Calibration.** Các mô hình và kỹ thuật còn lại (Decision Tree, Random Forest, LightGBM, CatBoost, MLP, SMOTE, monotonic constraints...) là mở rộng — ưu tiên thấp hơn khi thiếu thời gian.

## 3. Chỉ số thành công (KPI)

### 3.1. KPI chính (tương đối)

Mô hình cuối vượt baseline Logistic Regression về ROC-AUC với khác biệt có ý nghĩa thống kê:

- Bootstrap 95% CI của chênh lệch AUC không chứa 0, **hoặc**
- DeLong test p < 0.05.

Baseline Logistic Regression là run `logreg_baseline` (alias `baseline` trong MLflow Model Registry). Baseline này chưa encode PAY_*, EDUCATION, MARRIAGE (Option A) nên khá yếu — Decision Tree baseline đã vượt 0.023 AUC. Vì vậy KPI chính chỉ là điều kiện cần; mốc sàn ở mục 3.2 dùng để xác nhận mô hình cuối vượt cả baseline mạnh nhất.

### 3.2. KPI tham chiếu (tuyệt đối)

> Điều chỉnh tại Tuần 1 theo kết quả baseline thực tế (PR #16). Mức cũ: ROC-AUC ≥ 0.77 (Gini ≥ 0.54), KS ≥ 0.35.

**Baseline thực tế** — 5-fold Stratified CV trên Train (18.000 mẫu, 23 đặc trưng, Option A, tỷ lệ vỡ nợ 22.1%), mean ± std (ddof=1):

| Mô hình | ROC-AUC | Gini | KS | PR-AUC | Brier (chưa calib.) |
|---|---|---|---|---|---|
| Logistic Regression (`logreg_baseline`) | 0.7278 ± 0.0130 | 0.4557 ± 0.0261 | 0.3907 ± 0.0261 | 0.5109 ± 0.0106 | 0.1439 ± 0.0019 |
| Decision Tree, max_depth=5 (`dt_baseline`) | 0.7506 ± 0.0107 | 0.5011 ± 0.0215 | 0.4136 ± 0.0216 | 0.5088 ± 0.0137 | 0.1372 ± 0.0026 |

**Mốc tham chiếu mới** (đo trên tập Test):

| Chỉ số | Mốc sàn | Mốc mục tiêu |
|---|---|---|
| ROC-AUC | ≥ 0.76 | ≥ 0.77 |
| Gini | ≥ 0.52 | ≥ 0.54 |
| KS | ≥ 0.43 | ≥ 0.45 |
| PR-AUC | ≥ 0.52 | ≥ 0.53 |

Cách xác định: mốc sàn = giá trị baseline tốt nhất của từng chỉ số + 1 std CV; mốc mục tiêu = + 2 std; làm tròn xuống 2 chữ số thập phân.

Lý do điều chỉnh:

- **KS:** mức cũ 0.35 thấp hơn cả LR baseline (0.39) nên không còn phân biệt được mô hình tốt/kém → nâng lên 0.43 / 0.45.
- **ROC-AUC:** giữ 0.77 làm mốc mục tiêu (cao hơn DT baseline ≈ 1.8 std, cao hơn LR baseline 0.042). Bổ sung mốc sàn 0.76 để có một mức "chắc chắn vượt baseline tốt nhất", thay vì chỉ một con số gần trần.
- **PR-AUC (bổ sung):** dữ liệu mất cân bằng; hai baseline gần như bằng nhau ở PR-AUC (~0.51) dù ROC-AUC chênh 0.023, nên cần theo dõi riêng.

Lưu ý:

- Mốc được suy ra từ CV trên Train nhưng áp dụng trên Test (chỉ chạy 1 lần ở Tuần 3). Chênh lệch CV–Test cỡ 1 std là bình thường, vì vậy luôn báo cáo kèm 95% CI bootstrap.
- Baseline Option A vẫn dùng SEX và AGE liên tục (chưa theo mục 8). Chạy lại baseline bỏ SEX: ROC-AUC LR 0.7273, DT 0.7502 (giảm ≤ 0.0005) → không cần đổi mốc.
- KPI tham chiếu vẫn chỉ là mốc so sánh, không bắt buộc (xem mục 4).

## 4. Definition of Done (DoD)

Một mô hình/hạng mục được coi là hoàn thành khi:

- Đạt KPI chính (bắt buộc) — KPI tham chiếu chỉ là mốc so sánh, không bắt buộc tuyệt đối.
- Đã calibrate trên tập Valid và có Brier score + Calibration Curve báo cáo.
- Đã tính SHAP (Global + Local) cho mô hình được chọn làm mô hình cuối.
- Đã fit đúng quy trình (xem mục 6 — không leakage: mọi bước dùng target chỉ fit trong Train/từng fold).
- Test set chỉ được chạy đúng 1 lần ở Tuần 3 để báo cáo kết quả cuối.
- Code đã qua review (PR có ít nhất 1 approval) và CI chạy thành công.

## 5. Điều kiện dừng tuning

Dừng tuning (Optuna) khi xảy ra một trong hai điều kiện:

- Không cải thiện CV ROC-AUC quá 0.002 sau 30 trial liên tiếp, **hoặc**
- Hết ngân sách thời gian: tối đa 2 giờ tính toán cho mỗi mô hình.

## 6. Chính sách kỹ thuật 

- **Chia dữ liệu:** Stratified theo target, random_state = 42, Train 60% / Valid 20% / Test 20%. Chỉ số dòng lưu chung cho cả nhóm.
- **Không data leakage:** Binning/WoE, IV, lọc đặc trưng theo IV, target encoding, SMOTE đều chỉ fit trên Train, đặt trong Pipeline (sklearn/imblearn) để fit lại trong từng fold CV.
- **Test set:** Chỉ chạy đánh giá 1 lần duy nhất, ở Tuần 3.
- **Mô hình cuối:** Giữ nguyên bản fit trên Train; calibrator fit trên Valid. Không retrain trên Train + Valid.

## 7. Ma trận chi phí (Cost Matrix)

| Loại lỗi | Ý nghĩa | Công thức chi phí | Giả định |
|---|---|---|---|
| False Negative (FN) | Bỏ sót khách sắp vỡ nợ, không kịp can thiệp | ≈ LGD × EAD | LGD = 0.45; EAD = dư nợ sao kê gần nhất |
| False Positive (FP) | Giảm hạn mức/cảnh báo oan một khách hàng tốt | ≈ lãi và phí bị mất do giảm hạn mức | 5% × EAD |

Báo cáo kèm phân tích độ nhạy với tỷ lệ chi phí **FN:FP trong khoảng 3:1 đến 10:1**.

## 8. Chính sách biến nhạy cảm (Fairness Policy)

- **SEX:** Không được dùng trong mô hình chính thức.
- **AGE:** Được phép dùng, nhưng chỉ dưới dạng binning (không dùng giá trị liên tục thô).
- Huấn luyện thêm **một phiên bản mô hình có SEX** để đối chiếu hiệu năng và fairness (không dùng để triển khai).
- Ngưỡng cảnh báo disparate impact ratio: **< 0.8**.

## 9. Phân công trách nhiệm

| TV | Vai trò chính | Trách nhiệm chi tiết |
|---|---|---|
| A | Data Engineer / Analyst | Thu thập, làm sạch dữ liệu; EDA; Feature Engineering; Binning/WoE và Logistic Scorecard; chống Data Leakage; phân tích lỗi; đánh giá fairness. |
| B | ML Engineer | Chiến lược chia dữ liệu; module đánh giá; baseline, huấn luyện, tuning; calibration; chọn ngưỡng theo chi phí; quy đổi PD sang điểm; đánh giá cuối trên Test. |
| C | Explainability & Product | Project Charter; phân tích SHAP; reason codes; Demo Streamlit; Model Card; tổng hợp báo cáo. |

Chi tiết phân công theo từng ngày/tuần: xem mục 4 trong [project_plan.md](project_plan.md).

## 10. Rủi ro trọng yếu (tóm tắt)

| Rủi ro | Phương án xử lý chính |
|---|---|
| Data Leakage | Mọi bước fit dùng target đặt trong Pipeline, fit lại theo từng fold CV. |
| Tuning không điểm dừng | Áp dụng điều kiện dừng ở mục 5; dùng KPI tương đối thay vì chỉ một con số tuyệt đối. |
| Xác suất không hiệu chuẩn | Calibration bắt buộc trên Valid; đánh giá bằng Brier score và Calibration Curve. |
| Trễ tiến độ | Giữ đúng MVP (mục 2.1); áp dụng mốc feature freeze cuối T3 Tuần 2. |
| Giới hạn dữ liệu (snapshot 2005, một ngân hàng) | Nêu rõ trong Model Card; không diễn giải như mô hình sẵn sàng triển khai thực tế. |

Danh sách rủi ro đầy đủ: xem mục 7 trong [project_plan.md](project_plan.md).

