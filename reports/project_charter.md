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

Baseline Logistic Regression là run `logreg_baseline` (alias `baseline` trong MLflow Model Registry), huấn luyện trên bộ đặc trưng feature freeze v1 (`configs/feature_freeze_v1.yaml`): mã không tài liệu hóa được chuẩn hóa, các biến phân loại được one-hot và đặc trưng T2 được bổ sung. Đây là mô hình có ROC-AUC CV cao nhất trong nhóm Logistic Regression, Decision Tree, Random Forest và XGBoost tham số mặc định (mục 3.2). Mốc sàn ở mục 3.2 bổ sung một ngưỡng tuyệt đối bên cạnh KPI tương đối.

### 3.2. KPI tham chiếu (tuyệt đối)

> Điều chỉnh lần 2 tại Tuần 2 – T3 theo feature freeze v1. Lịch sử mốc tham chiếu:
>
> - Chốt ban đầu (Tuần 1 – T2): ROC-AUC ≥ 0.77, Gini ≥ 0.54, KS ≥ 0.35.
> - Lần 1 (Tuần 1 – T6, PR #16, baseline Option A), mốc sàn / mốc mục tiêu: ROC-AUC 0.76 / 0.77, Gini 0.52 / 0.54, KS 0.43 / 0.45, PR-AUC 0.52 / 0.53.

**Baseline thực tế** — 5-fold Stratified CV trên Train (18.000 mẫu, tỷ lệ vỡ nợ 22.1%), mean ± std (ddof=1). Pipeline `src.pipelines.make_pipeline`: làm sạch mã (PAY_0 → PAY_1, EDUCATION 0/5/6 → 4, MARRIAGE 0 → 3) → đặc trưng T2 → binning AGE theo IV (AGE_BIN) → bỏ SEX, ID → one-hot 9 biến phân loại (gồm AGE_BIN) + impute/scale 40 biến số.

| Mô hình | ROC-AUC | Gini | KS | PR-AUC | Brier (chưa calib.) |
|---|---|---|---|---|---|
| Logistic Regression (`logreg_baseline`) | 0.7745 ± 0.0116 | 0.5490 ± 0.0233 | 0.4282 ± 0.0231 | 0.5470 ± 0.0179 | 0.1354 ± 0.0031 |
| Decision Tree, max_depth=5 (`dt_baseline`) | 0.7672 ± 0.0120 | 0.5345 ± 0.0240 | 0.4162 ± 0.0184 | 0.5242 ± 0.0152 | 0.1367 ± 0.0031 |

Tham khảo (không dùng để đặt mốc) — mô hình tham số mặc định cùng pipeline; Logistic Scorecard được đánh giá trên cùng bộ fold, hiện mới có ROC-AUC:

| Mô hình | ROC-AUC | Gini | KS | PR-AUC | Brier (chưa calib.) |
|---|---|---|---|---|---|
| Random Forest (`rf_default`) | 0.7667 ± 0.0097 | 0.5333 ± 0.0193 | 0.4107 ± 0.0240 | 0.5319 ± 0.0200 | 0.1388 ± 0.0035 |
| XGBoost (`xgboost_default`) | 0.7597 ± 0.0073 | 0.5195 ± 0.0145 | 0.4007 ± 0.0160 | 0.5258 ± 0.0125 | 0.1440 ± 0.0031 |
| Logistic Scorecard WoE (`build_scorecard_pipeline`) | 0.7809 | – | – | – | – |

**Mốc tham chiếu** (đo trên tập Test):

| Chỉ số | Mốc sàn | Mốc mục tiêu |
|---|---|---|
| ROC-AUC | ≥ 0.78 | ≥ 0.79 |
| Gini | ≥ 0.57 | ≥ 0.59 |
| KS | ≥ 0.45 | ≥ 0.47 |
| PR-AUC | ≥ 0.56 | ≥ 0.58 |

Cách xác định: mốc sàn = giá trị baseline tốt nhất của từng chỉ số (đều là LR) + 1 std CV; mốc mục tiêu = + 2 std; làm tròn xuống 2 chữ số thập phân.

Lý do điều chỉnh:

- **Tiền xử lý và đặc trưng:** ROC-AUC của LR tăng từ 0.7278 lên 0.7745 (+0.0467) sau khi mã không tài liệu hóa được chuẩn hóa, PAY_*, EDUCATION, MARRIAGE được one-hot và đặc trưng T2 được bổ sung. Mốc sàn lần 1 (0.76) thấp hơn baseline mới nên không còn phân biệt được mô hình tốt với mô hình kém.
- **Baseline mạnh nhất:** Thứ hạng đảo so với lần 1: LR (0.7745) vượt DT (0.7672), nên các mốc lấy từ LR. Random Forest (0.7667) và XGBoost (0.7597) tham số mặc định chưa vượt LR; khoảng cách đến mốc sàn thuộc phạm vi tuning (Tuần 2 – T4 trở đi).
- **Logistic Scorecard (WoE):** ROC-AUC 0.7809, cao hơn baseline LR 0.0064 (khoảng 0.5 std CV) và xấp xỉ mốc sàn 0.78.

Lưu ý:

- Mốc được suy ra từ CV trên Train nhưng áp dụng trên Test (chỉ chạy 1 lần ở Tuần 3). Chênh lệch CV–Test cỡ 1 std là bình thường, vì vậy luôn báo cáo kèm 95% CI bootstrap.
- Mô hình chính thức không dùng SEX (mục 8). Bản đối chiếu fairness có SEX (run `*_with_sex`, không đăng ký vào Model Registry) đạt ROC-AUC LR 0.7749 và DT 0.7669, chênh không quá 0.0004 so với bản không SEX; SEX gần như không đóng góp vào khả năng phân biệt.
- AGE được đưa vào mọi mô hình dưới dạng binning theo IV (AGE_BIN, 3–8 bin, fit lại trong từng fold) theo mục 8, thay cho AGE liên tục. So với AGE liên tục, ROC-AUC của LR giảm 0.0007 và của DT tăng 0.0007, đều trong phạm vi 0.1 std CV; các mốc tham chiếu không đổi.
- Mốc mục tiêu ROC-AUC 0.79 cao hơn mọi mô hình đã đánh giá (cao nhất là Logistic Scorecard, 0.7809); khả năng đạt mốc phụ thuộc kết quả tuning.
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
- **Feature freeze v1:** Bộ đặc trưng được chốt tại `configs/feature_freeze_v1.yaml` (cột đầu vào, đặc trưng T2, AGE dạng binning, các cột loại bỏ SEX và ID, tham số binning/WoE và ngưỡng IV ≥ 0.02 của Scorecard). Từ Tuần 2 – T4, tuning chỉ thực hiện trên bộ đặc trưng này; mọi thay đổi cần cả nhóm thống nhất.
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

