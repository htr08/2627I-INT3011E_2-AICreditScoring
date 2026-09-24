# KẾ HOẠCH DỰ ÁN

_Dự đoán xác suất vỡ nợ (Probability of Default) – So sánh mô hình & Giải thích bằng SHAP_

## 1. Mục tiêu và sản phẩm đầu ra

### 1.1. Mục tiêu

Xây dựng pipeline dự đoán xác suất vỡ nợ (PD) đã được hiệu chuẩn, so sánh nhiều thuật toán học máy một cách có kiểm định thống kê, áp dụng SHAP để giải thích mô hình và triển khai một ứng dụng demo hoàn chỉnh.

### 1.2. Sản phẩm cuối (Deliverables)

- **Source code:** Repository GitHub hoàn chỉnh, tái chạy được từ đầu (reproducible), bao gồm script tải dữ liệu tự động.
- **Bảng so sánh:** Hiệu năng các mô hình theo các chỉ số chuẩn hóa, kèm độ lệch chuẩn qua CV và khoảng tin cậy trên tập Test.
- **Báo cáo XAI:** Phân tích giải thích mô hình bằng SHAP (toàn cục, cục bộ, reason codes).
- **Model Card:** Mô tả mục đích sử dụng, dữ liệu, hiệu năng, kết quả fairness và các hạn chế của mô hình.
- **Ứng dụng Demo:** Web app tương tác bằng Streamlit.
- **Slide thuyết trình:** Bộ slide tổng kết dự án.

### 1.3. Project Charter – KPI và Definition of Done

Project Charter được chốt tại Tuần 1 – T2, gồm các nội dung sau:

- **KPI chính (tương đối):** Mô hình cuối vượt baseline Logistic Regression về ROC-AUC với khác biệt có ý nghĩa thống kê (bootstrap 95% CI của chênh lệch AUC không chứa 0, hoặc DeLong test p < 0.05).
- **KPI tham chiếu (tuyệt đối):** ROC-AUC ≥ 0.77 (tương đương Gini ≥ 0.54) và KS ≥ 0.35 trên tập Test. Đây là mức gần trần của bộ dữ liệu, nên được điều chỉnh sau khi có baseline Tuần 1.
- **Điều kiện dừng tuning:** Dừng khi Optuna không cải thiện CV ROC-AUC quá 0.002 sau 30 trial liên tiếp, hoặc hết ngân sách thời gian (tối đa 2 giờ tính toán cho mỗi mô hình).
- **Ma trận chi phí (chốt từ Tuần 1):** Chi phí FN (bỏ sót khách sắp vỡ nợ, không kịp can thiệp) ≈ LGD × EAD (giả định LGD = 0.45, EAD = dư nợ sao kê gần nhất); chi phí FP (giảm hạn mức/cảnh báo oan một khách hàng tốt) ≈ lãi và phí bị mất do giảm hạn mức (giả định 5% × EAD). Báo cáo kèm phân tích độ nhạy với tỷ lệ chi phí FN:FP trong khoảng 3:1 đến 10:1.
- **Chính sách biến nhạy cảm:** SEX không được dùng trong mô hình chính thức. AGE được phép dùng dưới dạng binning. Huấn luyện thêm một phiên bản có SEX để đối chiếu hiệu năng và fairness.
- **Scope tối thiểu bắt buộc (MVP):** Logistic Regression + XGBoost + SHAP + Calibration. Các phần còn lại là mở rộng.

## 2. Phân công và nguyên tắc phân vai

### 2.1. Phân công trách nhiệm

| TV | Vai trò chính | Trách nhiệm chi tiết |
|---|---|---|
| A | Data Engineer / Analyst | Thu thập, làm sạch dữ liệu; EDA; Feature Engineering; Binning/WoE và Logistic Scorecard; đảm bảo chống Data Leakage; phân tích lỗi; đánh giá công bằng (Fairness). |
| B | ML Engineer | Chiến lược chia dữ liệu; module đánh giá; baseline, huấn luyện, tuning; calibration; lựa chọn ngưỡng theo chi phí; quy đổi PD sang điểm; đánh giá cuối trên tập Test. |
| C | Explainability & Product | Project Charter; phân tích SHAP; reason codes; xây dựng Demo Streamlit; Model Card; tổng hợp báo cáo. |

### 2.2. Quy định phối hợp chung

- Cả 3 thành viên tham gia review code chéo thông qua Pull Request (PR).
- Mỗi thành viên tự soạn nội dung báo cáo và slide tương ứng với phạm vi công việc được giao; thành viên C chịu trách nhiệm hợp nhất và thống nhất văn phong.

## 3. Lựa chọn kỹ thuật và công cụ (Stack)

### 3.1. Dữ liệu

- **Default of Credit Card Clients (Taiwan):** 30.000 dòng, 23 đặc trưng, tỷ lệ vỡ nợ khoảng 22%. Nhãn là "vỡ nợ trong tháng kế tiếp" trên một snapshot duy nhất năm 2005.
- **Nguồn:** https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset (hoặc UCI Machine Learning Repository). Dữ liệu được tải qua script `scripts/download_data.py` kèm kiểm tra checksum, không commit lên GitHub.
- **Đặc điểm cần xử lý:**
  - Dữ liệu gần như không có missing value; vấn đề chính là các mã không được tài liệu hóa: EDUCATION có giá trị 0, 5, 6; MARRIAGE có giá trị 0; PAY_x có giá trị −2 và 0.
  - Cột trạng thái trả nợ được đặt tên PAY_0, PAY_2, …, PAY_6 (không có PAY_1); đổi tên thống nhất thành PAY_1…PAY_6 ngay ở bước tiền xử lý.
  - Chỉ số 1 là tháng gần nhất (tháng 9), chỉ số 6 là tháng xa nhất (tháng 4).

### 3.2. Công nghệ sử dụng

- **Ngôn ngữ & Thư viện:** Python, pandas, scikit-learn, imbalanced-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP.
- **Quản lý thí nghiệm & Triển khai:** MLflow (Optuna log dạng nested runs), Streamlit, GitHub, GitHub Actions, pytest.

### 3.3. Danh sách mô hình so sánh

- Logistic Regression (baseline) và Logistic Scorecard truyền thống dùng WoE.
- Decision Tree và Random Forest.
- Gradient Boosting: XGBoost, LightGBM, CatBoost (có thử nghiệm monotonic constraints cho các biến trễ hạn và tỷ lệ sử dụng hạn mức).
- Tùy chọn bổ sung (nếu còn thời gian): MLP (Multi-Layer Perceptron).

### 3.4. Chỉ số đánh giá (Evaluation Metrics)

- **Phân tách:** ROC-AUC, Gini, Kolmogorov-Smirnov (KS), PR-AUC. Báo cáo mean ± std qua 5-fold CV và 95% CI (bootstrap) trên tập Test.
- **So sánh cặp mô hình:** DeLong test hoặc bootstrap CI của chênh lệch AUC.
- **Hiệu chuẩn:** Brier score (trước và sau calibration), Calibration Curve.
- **Theo ngưỡng:** Precision, Recall, F1-score, flag rate (tỷ lệ khách hàng bị gắn cờ can thiệp) tại ngưỡng được chọn.
- **Kinh doanh:** Chi phí kỳ vọng theo ma trận chi phí FN/FP đã chốt trong Project Charter.

### 3.5. Quy ước kỹ thuật chính

#### a) Chia dữ liệu

- Chia stratified theo target, random_state = 42: **Train 60% / Valid 20% / Test 20%**. Lưu chỉ số dòng của từng tập vào file để mọi thành viên dùng chung.
- **Train:** huấn luyện và tuning bằng 5-fold Stratified CV.
- **Valid:** fit calibration, chọn ngưỡng theo chi phí, phân tích lỗi, kiểm tra fairness sơ bộ.
- **Test:** chỉ chạy một lần duy nhất ở Tuần 3 để báo cáo kết quả cuối.
- Mô hình cuối giữ nguyên bản fit trên Train; calibrator fit trên Valid. Không retrain trên Train + Valid để đảm bảo xác suất đã hiệu chuẩn nhất quán.

#### b) Các bước dùng target phải fit trong từng fold

Binning/WoE, tính IV, lọc đặc trưng theo IV, age binning theo IV, target encoding và SMOTE (nếu dùng) đều sử dụng thông tin target. Các bước này chỉ được fit trên dữ liệu train và phải được đặt bên trong Pipeline (sklearn.pipeline hoặc imblearn.pipeline) để được fit lại trong từng fold khi chạy CV và Optuna.

#### c) Xử lý mất cân bằng

Tỷ lệ vỡ nợ khoảng 22% không phải mất cân bằng nặng. Hướng chính là dùng class_weight / scale_pos_weight hoặc không xử lý, sau đó bắt buộc calibration. SMOTE chỉ là thí nghiệm phụ để đối chiếu, do resampling làm méo xác suất đầu ra, trong khi PD đã hiệu chuẩn là sản phẩm chính của dự án.

#### d) Quy đổi PD sang điểm tín dụng

Sử dụng phương pháp PDO (Points to Double the Odds): Score = Offset + Factor × ln(odds tốt/xấu), với Factor = PDO / ln(2) và Offset = Base Score − Factor × ln(Base Odds). Tham số đề xuất: Base Score = 600 tại odds 50:1, PDO = 20. Kết quả được giới hạn (clip) trong thang 300–850 để hiển thị trên demo. Công thức và tham số được ghi rõ trong báo cáo.

#### e) Lưu ý khi dùng SHAP

- TreeExplainer giải thích đầu ra thô (log-odds/margin) của mô hình chưa hiệu chuẩn. Ưu tiên Platt scaling, fit tay bằng LogisticRegression 1 chiều trên `output_margin=True` (không dùng `CalibratedClassifierCV(method="sigmoid")` mặc định của sklearn, vì với estimator chỉ có predict_proba như XGBoost/LightGBM/CatBoost, hàm này fit sigmoid lên xác suất đã qua sigmoid chứ không phải lên log-odds thô, làm mất tính tuyến tính cần thiết). Khi PD = sigmoid(a·margin + b), nhân SHAP values (đang ở thang margin) với hệ số a để được SHAP trên thang logit-đã-hiệu-chuẩn, tổng cộng đúng bằng logit(PD hiển thị). Nếu dùng Isotonic, demo phải ghi chú rằng tổng SHAP không cộng đúng bằng PD cuối.
- Các nhóm biến BILL_AMT1–6 và PAY_1–6 tương quan rất cao, khiến SHAP phân bổ đóng góp giữa chúng khá tùy ý. Reason codes và so sánh đồng thuận giữa các mô hình được thực hiện trên **nhóm đặc trưng** (lịch sử trễ hạn, mức sử dụng hạn mức, hành vi trả nợ, biến động dư nợ, nhân khẩu học) thay vì từng biến riêng lẻ.
- Mô hình cây dùng TreeExplainer; Logistic dùng LinearExplainer; không dùng KernelExplainer trong demo live.

## 4. Kế hoạch triển khai theo từng tuần

### 4.1. Tuần 1: Dữ liệu, EDA và Baseline

**Mục tiêu tuần:** Làm sạch dữ liệu, hiểu đặc trưng, chốt Project Charter và thiết lập mô hình baseline.

| Ngày | Thành viên A (Data) | Thành viên B (ML) | Thành viên C (XAI & Product) |
|---|---|---|---|
| T2 | Kickoff, chốt dataset. Viết `scripts/download_data.py` tải dữ liệu tự động kèm checksum. | Kickoff, dựng cấu trúc repository, cấu hình môi trường (`requirements.txt`, MLflow). Thiết lập CI tối thiểu (GitHub Actions chạy pytest). | Kickoff, viết Project Charter: mục tiêu, phạm vi, KPI, điều kiện dừng tuning, ma trận chi phí FN/FP, chính sách biến nhạy cảm (mục 1.3). |
| T3 | Data profiling: kiểu dữ liệu, outlier, phân phối target. Xác định cách xử lý các mã không tài liệu hóa (EDUCATION, MARRIAGE, PAY_x); đổi tên PAY_0 → PAY_1. | Chia dữ liệu Train/Valid/Test 60/20/20 stratified, random_state = 42; lưu file chỉ số dòng dùng chung. | Nghiên cứu SHAP (TreeExplainer, LinearExplainer, KernelExplainer), Logistic Scorecard và phương pháp PDO. |
| T4 | EDA đơn biến và hai biến với target; tính Correlation và Information Value (IV) trên tập Train. | Viết `src/evaluate.py`: AUC, Gini, KS, PR-AUC, Brier, Confusion Matrix, bootstrap CI, DeLong test, hàm chi phí kỳ vọng; kèm unit test. | Viết notebook mẫu thực thi SHAP trên tập dữ liệu nhỏ để hoàn thiện luồng xử lý. |
| T5 | Xây dựng pipeline tiền xử lý (xử lý mã bất thường, Encoding, Scaling) bằng sklearn.pipeline trong `src/preprocessing.py`; kèm unit test. | Huấn luyện baseline: Logistic Regression và Decision Tree với 5-fold CV trên Train. | Phác thảo wireframe demo; dựng khung Streamlit chạy với mock model (form nhập, hiển thị PD, điểm, giải thích). |
| T6 | Hoàn thiện báo cáo EDA tổng hợp các insights chính. | Log toàn bộ kết quả baseline vào MLflow. | Tổ chức họp review Tuần 1 cùng nhóm; điều chỉnh KPI tham chiếu theo baseline thực tế. |

**Deliverables Tuần 1:** Project Charter, script tải dữ liệu, notebook EDA, `src/preprocessing.py`, `src/evaluate.py` kèm unit test, file chỉ số chia dữ liệu, kết quả baseline trên MLflow, wireframe và khung Streamlit.

### 4.2. Tuần 2: Feature Engineering, Mô hình nâng cao và SHAP

**Mục tiêu tuần:** Chốt bộ đặc trưng, huấn luyện và tối ưu các mô hình mạnh, thực hiện phân tích SHAP toàn cục và cục bộ.

**Mốc quan trọng:** Feature freeze v1 vào cuối ngày T3. Từ T4, thành viên B chỉ tuning trên bộ đặc trưng đã đóng băng; mọi thay đổi đặc trưng sau mốc này phải được cả nhóm thống nhất.

| Ngày | Thành viên A (Data) | Thành viên B (ML) | Thành viên C (XAI & Product) |
|---|---|---|---|
| T2 | Tạo đặc trưng mới: <br>• Tỷ lệ sử dụng hạn mức: BILL_AMT_i / LIMIT_BAL (và trung bình 6 tháng). <br>• Xu hướng trễ hạn: trung bình, max, độ dốc của PAY_1 → PAY_6. <br>• Số tháng trễ hạn liên tiếp; số lần trễ ≥ 2 tháng trong 6 tháng. <br>• Tỷ lệ trả nợ/dư nợ: PAY_AMT_i / BILL_AMT_(i+1) (khoản trả tháng i dành cho sao kê tháng trước); xử lý riêng trường hợp BILL_AMT ≤ 0. <br>• Biến động dư nợ (std/delta BILL_AMT qua các tháng). <br>• Cờ "trả mức tối thiểu": PAY_AMT rất nhỏ so với BILL_AMT. <br>• Age binning theo IV (fit trên train). | Huấn luyện Random Forest và XGBoost với tham số mặc định trên dữ liệu đã tiền xử lý. | Kết nối khung Streamlit với mô hình baseline thật thay cho mock model. |
| T3 | Binning và WoE cho Scorecard; lọc đặc trưng theo ngưỡng IV (tất cả đặt trong pipeline để fit trong từng fold). **Chốt feature freeze v1.** | Huấn luyện LightGBM, CatBoost với tham số mặc định. Thử class_weight / scale_pos_weight. Thí nghiệm phụ: SMOTE đặt bên trong từng fold bằng `imblearn.pipeline.Pipeline`, không resample trước CV. | Trích xuất SHAP Global cho XGBoost (mẫu 2.000–5.000 dòng): Summary Plot, Bar Plot mean(\|SHAP\|). |
| T4 | Lọc đặc trưng: loại biến tương quan cao; kiểm tra độ ổn định IV và tầm quan trọng đặc trưng qua các fold CV. Đóng gói `src/features.py`. | Tuning bằng Optuna với 5-fold Stratified CV cho 2 mô hình tốt nhất, có thử monotonic constraints; log MLflow dạng nested runs; áp dụng điều kiện dừng trong Charter. | SHAP Dependence Plot cho Top 5–10 đặc trưng để phát hiện tương tác. Định nghĩa các nhóm đặc trưng dùng cho reason codes. |
| T5 | Xây dựng Logistic Scorecard (WoE) làm cơ sở so sánh với các mô hình dạng black-box. | Tiếp tục tuning; huấn luyện thêm phiên bản mô hình tốt nhất có biến SEX để phục vụ đối chiếu fairness. | SHAP Local: Waterfall/Force plot cho các trường hợp Duyệt, Từ chối và Ca biên. |
| T6 | Rà soát toàn bộ pipeline để đảm bảo không có data leakage và lỗi logic. | Tổng hợp bảng so sánh sơ bộ (CV mean ± std) của các mô hình. | Tổ chức họp review Tuần 2 cùng nhóm, thống nhất chọn 1–2 mô hình cuối. |

**Deliverables Tuần 2:** `src/features.py` hoàn chỉnh, 6–7 mô hình lưu trên MLflow, Logistic Scorecard, bảng so sánh sơ bộ, notebook phân tích SHAP.

### 4.3. Tuần 3: Đánh giá sâu, Hoàn thiện Demo và Báo cáo

**Mục tiêu tuần:** Hiệu chuẩn và chốt mô hình cuối, đánh giá một lần trên Test, hoàn thiện ứng dụng Streamlit, báo cáo và thuyết trình.

| Ngày | Thành viên A (Data) | Thành viên B (ML) | Thành viên C (XAI & Product) |
|---|---|---|---|
| T2 | Phân tích lỗi (Error Analysis) trên tập Valid: đặc điểm các hồ sơ FN và FP. | Calibration trên tập Valid (so sánh Platt — fit tay trên `output_margin=True` — và Isotonic theo Brier score). Chọn ngưỡng tối ưu theo ma trận chi phí, kèm phân tích độ nhạy. Quy đổi PD sang điểm theo PDO, giới hạn thang 420–630. | So sánh mức đồng thuận về tầm quan trọng đặc trưng (theo nhóm) giữa XGBoost, LightGBM và Logistic qua SHAP. |
| T3 | Fairness check: flag rate (tỷ lệ bị gắn cờ can thiệp), disparate impact ratio và AUC theo giới tính và nhóm tuổi (<30 / 30–50 / >50) tại ngưỡng đã chọn; ngưỡng cảnh báo disparate impact < 0.8. So sánh phiên bản có và không có SEX. | **Đánh giá cuối trên tập Test (chạy 1 lần duy nhất):** ROC, PR, KS, Calibration Curve, bootstrap 95% CI, DeLong test giữa mô hình cuối và baseline. Cung cấp số liệu Test cho A cập nhật phần fairness. | Tích hợp mô hình chính thức vào Streamlit: nhập thông tin hoặc chọn hồ sơ mẫu có sẵn (Bình thường / Cảnh báo cao / Ca biên), validate input, trả về PD, điểm, mức độ rủi ro & hành động đề xuất (theo dõi / giảm hạn mức / chuyển thu hồi nợ) và SHAP Waterfall Plot. Dùng TreeExplainer nếu mô hình cuối là cây; LinearExplainer nếu là Logistic. |
| T4 | Viết báo cáo: Dữ liệu, EDA, Feature Engineering, Fairness. | Viết báo cáo: Phương pháp, Kết quả so sánh; refactor code, cập nhật README. | Viết báo cáo: XAI và hệ thống lý do cảnh báo (reason codes theo nhóm đặc trưng); soạn Model Card kèm mục "Mục đích sử dụng: công cụ cảnh báo sớm/quản lý danh mục hiện hữu, không phải công cụ xét duyệt tín dụng mới" và mục Hạn chế. |
| T5 | Soạn slide phần Dữ liệu. | Soạn slide phần Mô hình; kiểm tra chạy lại dự án từ một bản clone sạch theo README. | Soạn slide phần XAI và Demo. |
| T6 | Tổng duyệt thuyết trình toàn đội, hoàn tất nộp sản phẩm. | Tổng duyệt thuyết trình toàn đội, hoàn tất nộp sản phẩm. | Tổng duyệt thuyết trình toàn đội, hoàn tất nộp sản phẩm. |

**Deliverables Tuần 3:** Báo cáo tổng kết, Model Card, repository hoàn chỉnh, demo Streamlit ổn định, bộ slide.

## 5. Mẫu bảng so sánh mô hình

Sử dụng thống nhất cho báo cáo. Cột CV lấy từ 5-fold CV trên Train; các cột còn lại đo trên tập Test.

| Mô hình | AUC CV (mean ± std) | AUC Test (95% CI) | Gini | KS | PR-AUC | Brier (sau calib.) | Thời gian train | Khả năng giải thích |
|---|---|---|---|---|---|---|---|---|
| Logistic (baseline) | Điền | Điền | Điền | Điền | Điền | Điền | Điền | Cao (hệ số trực tiếp) |
| Logistic Scorecard (WoE) | Điền | Điền | Điền | Điền | Điền | Điền | Điền | Cao (bảng điểm) |
| Decision Tree | Điền | Điền | Điền | Điền | Điền | Điền | Điền | Cao (luật rõ ràng) |
| Random Forest | Điền | Điền | Điền | Điền | Điền | Điền | Điền | Trung bình (SHAP) |
| XGBoost | Điền | Điền | Điền | Điền | Điền | Điền | Điền | Trung bình (SHAP) |
| LightGBM | Điền | Điền | Điền | Điền | Điền | Điền | Điền | Trung bình (SHAP) |
| CatBoost (tùy chọn) | Điền | Điền | Điền | Điền | Điền | Điền | Điền | Trung bình (SHAP) |
| MLP (tuỳ chọn) | Điền | Điền | Điền | Điền | Điền | Điền | Điền | Thấp (KernelExplainer) |

Kèm bảng phụ: chênh lệch AUC của từng mô hình so với baseline Logistic, kèm 95% CI hoặc p-value DeLong; chi phí kỳ vọng tại ngưỡng tối ưu.

## 6. Cấu trúc Repository chuẩn

```
credit-scoring/
├── .github/workflows/ci.yml
├── configs/
│   └── config.yaml           # random_state, split, chi phí, PDO, nhóm đặc trưng
├── data/
│   ├── raw/                  # gitignore
│   ├── processed/            # gitignore
│   └── splits/                # commit: chỉ số dòng Train/Valid/Test
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_shap.ipynb
│   ├── 04_calibration_threshold.ipynb
│   └── 05_error_fairness.ipynb
├── scripts/
│   ├── download_data.py
│   └── run_pipeline.py       # train → calibrate → xuất model cuối
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── calibrate.py
│   ├── scoring.py            # quy đổi PD → điểm (PDO)
│   ├── evaluate.py
│   ├── fairness.py
│   └── explain.py
├── tests/
├── app/
│   └── streamlit_app.py
├── models/                   # gitignore, trừ model cuối cho demo
├── reports/
│   ├── figures/
│   ├── model_card.md
│   └── final_report.md
├── .gitignore                # thêm mlruns/
├── pyproject.toml
├── requirements.txt          # pin version cụ thể
└── README.md
```

CI pipeline (GitHub Actions) chạy pytest cho các module trong `src/` ở mỗi Pull Request để phát hiện lỗi trước khi merge vào main.

## 7. Quản lý rủi ro và phương án xử lý

| Rủi ro | Mô tả | Phương án xử lý |
|---|---|---|
| Rủi ro Data Leakage | Thông tin từ tập validation/test, hoặc từ target, rò rỉ vào quá trình huấn luyện. | Mọi bước fit (Scaler, Encoder, Binning/WoE, lọc theo IV, SMOTE) chỉ thực hiện trên train và được đặt trong Pipeline để fit lại trong từng fold CV. Thành viên A chịu trách nhiệm chính kiểm tra tính toàn vẹn. |
| Rủi ro tuning không điểm dừng | KPI tuyệt đối gần trần của bộ dữ liệu khiến nhóm tiếp tục tuning để đạt con số. | Dùng KPI tương đối so với baseline và điều kiện dừng tuning trong Project Charter; báo cáo khoảng tin cậy thay vì chỉ một con số. |
| Rủi ro xác suất không hiệu chuẩn | Class weighting hoặc resampling làm PD đầu ra lệch so với tỷ lệ vỡ nợ thực tế. | Calibration bắt buộc trên tập Valid; đánh giá bằng Brier score và Calibration Curve; SMOTE chỉ là thí nghiệm phụ. |
| Rủi ro lệch độ đo do mất cân bằng | Accuracy cao nhưng không phản ánh khả năng phát hiện vỡ nợ. | Loại Accuracy khỏi danh sách chỉ số chính; tập trung vào ROC-AUC, PR-AUC, Gini, KS và chi phí kỳ vọng. |
| Rủi ro chi phí tính toán SHAP | Tính SHAP trên toàn bộ dữ liệu mất nhiều thời gian. | Dùng TreeExplainer cho mô hình cây; lấy mẫu ngẫu nhiên 2.000–5.000 dòng khi phân tích SHAP Global. |
| Rủi ro SHAP chậm trong demo trực tiếp | KernelExplainer có thể khiến Streamlit bị đơ khi thuyết trình. | Ưu tiên mô hình cuối dạng cây (TreeExplainer) hoặc Logistic (LinearExplainer). |
| Rủi ro trễ tiến độ | Một số hạng mục kéo dài hơn dự kiến, đặc biệt khối lượng mô hình ở Tuần 2. | Giữ MVP bắt buộc (Logistic Regression, XGBoost, SHAP, Calibration); Random Forest, CatBoost, MLP và SMOTE là mục tùy chọn; áp dụng mốc feature freeze để tránh làm lại. |
| Rủi ro về giới hạn dữ liệu | Dữ liệu một ngân hàng, năm 2005, một snapshot; nhãn là vỡ nợ tháng kế tiếp chứ không phải PD 12 tháng; không thể kiểm định out-of-time. | Nêu rõ trong Model Card và báo cáo; không diễn giải kết quả như một mô hình sẵn sàng triển khai thực tế. |

## 8. Quy trình làm việc và quy tắc phát triển

### 8.1. Quy định sử dụng Git

- **Cấu trúc branch:** Đặt tên theo cú pháp `<role>/<task>` (ví dụ: `data/eda`, `ml/xgboost-tuning`, `xai/shap-global`). Không thao tác trực tiếp trên nhánh main.
- **Push/Pull:** Mỗi cá nhân chỉ push trên branch của mình; bắt buộc pull/rebase từ main đầu mỗi ngày làm việc.
- **Merge:** Mọi thay đổi vào main phải qua Pull Request, được ít nhất 1 thành viên khác approve và CI chạy thành công.
- **Quy mô PR:** PR nhỏ, mỗi PR giải quyết một tính năng.
- **Quản lý tập tin:** Không commit dữ liệu thô, dữ liệu trung gian hay file mô hình; đưa vào `.gitignore`. Xóa output của notebook trước khi commit, hoặc chuyển code chuẩn sang `src/`.
- **Merge conflict:** Người tạo ra xung đột tự xử lý trước khi yêu cầu review lại.

### 8.2. Quy định code và quản lý thí nghiệm

- **Tính tái lập:** Cố định random_state = 42 trên toàn bộ thư viện và file code; dùng chung file chỉ số chia dữ liệu.
- **Đánh giá tập Test:** Tập Test chỉ chạy một lần duy nhất ở Tuần 3 (T3). Calibration, chọn ngưỡng, phân tích lỗi đều thực hiện trên tập Valid.
- **Theo dõi thí nghiệm:** Mọi lượt huấn luyện đều được log tham số và kết quả trên MLflow; các trial Optuna log dạng nested runs dưới một run cha.
- **Commit message:** Theo chuẩn Conventional Commits (ví dụ: `feat: add IV-based feature selection`, `fix: leakage in scaler fit`).
- **Tài liệu:** README.md luôn phản ánh đúng luồng chạy thực tế, đảm bảo bất kỳ ai cũng chạy lại được dự án từ một bản clone sạch.

### 8.3. Cơ chế giao tiếp và quản lý nhóm

- **Họp review:** Thứ 6 hàng tuần họp nghiệm thu tiến độ, đối chiếu kết quả với kế hoạch và điều chỉnh nhiệm vụ nếu phát sinh.
- **Xử lý blocker:** Khi bị nghẽn tiến độ, thành viên thông báo ngay trên kênh chat chung để phối hợp giải quyết, không giữ vấn đề đến buổi họp định kỳ.
- **Integration Owner:** Luân phiên theo tuần (Tuần 1: B, Tuần 2: A, Tuần 3: C). Người đảm nhiệm rà soát các PR cuối tuần, merge và đảm bảo nhánh main luôn chạy được.
