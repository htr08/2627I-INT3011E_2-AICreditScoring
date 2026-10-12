# Báo cáo Kỹ thuật: Hiệu chuẩn Xác suất, Tối ưu hóa Ngưỡng Chi phí & Quy đổi Điểm Tín dụng (PDO)

_Tuần 3 – T2: Thành viên B (ML Engineer)_

---

## 1. Mục tiêu và Cơ sở Lý luận

Theo phân công nhiệm vụ tại `reports/project_plan.md` (mục 4.3 – Tuần 3, T2) và `reports/project_charter.md` (mục 1.3, 7):
1. **Probability Calibration trên tập Valid:**
   - So sánh hai phương pháp hiệu chuẩn xác suất: **Platt scaling** (fit tay bằng mô hình hồi quy Logistic 1 chiều trên raw margin / `output_margin=True`) và **Isotonic regression** (hồi quy đơn điệu phi tham số) dựa trên thước đo chính là **Brier score loss**, đi kèm Log Loss, ROC-AUC và Expected Calibration Error (ECE).
   - Đảm bảo tính nhất quán toán học với phân tích giải thích mô hình SHAP (mục 3.5e): Giữ nguyên quan hệ tuyến tính giữa margin thô và logit của xác suất đã hiệu chuẩn.
2. **Tối ưu hóa ngưỡng quyết định theo ma trận chi phí:**
   - Xác định ngưỡng cắt xác suất tối ưu ($p^*$) thiểu hóa hàm chi phí kinh doanh kỳ vọng theo ma trận chi phí chốt từ Charter:
     - Chi phí FN (bỏ sót khách sắp vỡ nợ, không kịp can thiệp sớm) $\approx \text{LGD} \times \text{EAD}$ với giả định $\text{LGD} = 0.45$.
     - Chi phí FP (cảnh báo oan, giảm hạn mức của khách hàng tốt) $\approx 5\% \times \text{EAD}$.
     - Tỷ lệ chi phí chuẩn $\text{Cost(FN)} : \text{Cost(FP)} = 0.45 : 0.05 = 9:1$.
   - Thực hiện phân tích độ nhạy (Sensitivity Analysis) toàn diện với tỷ lệ chi phí FN:FP dao động từ **3:1 đến 10:1**.
3. **Quy đổi PD sang điểm tín dụng theo phương pháp PDO (Points to Double the Odds):**
   - Chuyển đổi xác suất vỡ nợ đã hiệu chuẩn $PD$ sang thang điểm tín dụng chuẩn ngân hàng giới hạn trong khoảng **[300, 850]** với tham số $\text{Base Score} = 600$ tại $\text{Base Odds} = 50:1$ và $\text{PDO} = 20$.
   - Xác định ngưỡng điểm cắt (cutoff score) phân loại hồ sơ rủi ro tương ứng với ngưỡng chi phí tối ưu.
4. **Tuân thủ kỷ luật phân chia dữ liệu (Data Split Integrity):**
   - **Tập Train (60% - 18.000 dòng):** Huấn luyện mô hình cơ sở với siêu tham số đã đóng băng.
   - **Tập Valid (20% - 6.000 dòng):** Dùng để fit calibrator, tối ưu hóa ngưỡng chi phí và khảo sát phân phối điểm số.
   - **Tập Test (20% - 6.000 dòng):** Đóng băng 100%, tuyệt đối không can thiệp, bảo lưu cho ngày Tuần 3 – T3 chạy nghiệm thu duy nhất 1 lần.

---

## 2. Bảng Tổng hợp Kết quả Hiệu chuẩn trên tập Valid

Thực nghiệm được thực hiện trên 3 mô hình tiêu biểu từ kết quả Tuần 2:
- **Ứng viên chính:** `LightGBM tuned + Monotonic` (tham số Optuna tốt nhất, tuân thủ ràng buộc đơn điệu nghiệp vụ).
- **Ứng viên đối chứng:** `CatBoost tuned` (mô hình có ROC-AUC CV cao nhất toàn dự án).
- **Mô hình cơ sở:** `Logistic Regression (baseline)` (chuẩn so sánh từ Tuần 1).

### 2.1. Bảng số liệu Hiệu chuẩn Chi tiết (Tập Valid – 6.000 mẫu)

| Mô hình | Phương pháp | Brier Score | Log Loss | ROC-AUC | ECE | Platt $a$ (dốc) | Platt $b$ (hệ số tự do) | Ghi chú kỹ thuật |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM tuned + Monotonic** | **Thô (Uncalibrated)** | 0.13312 | 0.42387 | 0.79030 | 0.01633 | – | – | Tỷ lệ dự đoán TB = 21.81% (thực tế 22.12%) |
| | **Platt Scaling (Fit tay)** | **0.13310** | **0.42382** | **0.79030** | **0.01509** | **1.0129** | **+0.0367** | **Bảo toàn SHAP additivity ($a \approx 1.01$), AUC giữ nguyên** |
| | **Isotonic Regression** | 0.13190 | 0.41973 | 0.79325 | 0.00941 | – | – | Brier thấp nhất nhưng mất tính tuyến tính SHAP |
| **CatBoost tuned** | **Thô (Uncalibrated)** | 0.13351 | 0.42550 | 0.78734 | 0.01756 | – | – | Tỷ lệ dự đoán TB = 21.77% |
| | **Platt Scaling (Fit tay)** | **0.13348** | **0.42543** | **0.78734** | **0.01584** | **1.0142** | **+0.0415** | Bảo toàn SHAP additivity, AUC giữ nguyên |
| | **Isotonic Regression** | 0.13221 | 0.42142 | 0.79046 | 0.01025 | – | – | Phi tham số, Brier giảm sâu |
| **Logistic Regression (baseline)** | **Thô (Uncalibrated)** | 0.13689 | 0.43468 | 0.77123 | 0.02104 | – | – | Baseline Tuần 1 |
| | **Platt Scaling (Fit tay)** | 0.13685 | 0.43459 | 0.77123 | 0.01892 | 0.9854 | +0.0215 | Điều chỉnh nhẹ độ lệch |
| | **Isotonic Regression** | 0.13576 | 0.43085 | 0.77410 | 0.01180 | – | – | Giảm Brier score |

---

## 3. Phân tích Kỹ thuật: So sánh Platt Scaling vs. Isotonic Regression

### 3.1. Tại sao Platt scaling fit tay trên raw margin là phương án tối ưu cho XAI / SHAP?

Theo quy ước kỹ thuật tại `project_plan.md` mục 3.5e:
- Thuật toán `TreeExplainer` của SHAP giải thích đầu ra thô của mô hình trên không gian **log-odds (margin)**:
  $$\text{margin}(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$
- Khi áp dụng mô hình hồi quy Logistic 1 chiều fit tay trên margin:
  $$\text{logit}(PD) = a \cdot \text{margin}(x) + b$$
- Thay triển khai SHAP vào công thức hiệu chuẩn:
  $$\text{logit}(PD) = (a \cdot \phi_0 + b) + \sum_{i=1}^{M} (a \cdot \phi_i(x))$$
- **Hệ quả cực kỳ quan trọng:** Mọi giá trị đóng góp SHAP $\phi_i$ chỉ cần nhân với hệ số vô hướng $a$ ($a = 1.0129$ đối với LightGBM) thì tổng đóng góp của tất cả các đặc trưng sẽ cộng đúng chính xác bằng logit của PD hiển thị trên giao diện người dùng!
- Ngược lại, nếu dùng `CalibratedClassifierCV(method="sigmoid")` của Scikit-learn, hàm này sẽ fit sigmoid lên xác suất đã qua sigmoid ($p$), phá vỡ quan hệ tuyến tính.
- Nếu dùng **Isotonic Regression**, do đây là hàm từng đoạn không liên tục (piecewise constant step-function), không tồn tại phép biến đổi tuyến tính tương đương, khiến tổng SHAP không còn khớp với PD cuối cùng.

### 3.2. Hiệu quả cải thiện chất lượng xác suất
- Cả `LightGBM tuned + Monotonic` và `CatBoost tuned` đều có hệ số Platt $a \approx 1.01$ và $b \approx +0.04$, chứng tỏ mô hình sau quá trình Optuna tuning đã có margin thô rất gần với log-odds thực tế (rất ít bị overconfidence).
- Chỉ số sai số kỳ vọng **ECE (Expected Calibration Error)** giảm từ **0.01633 xuống 0.01509** trên LightGBM và từ **0.01756 xuống 0.01584** trên CatBoost.
- Độ phân tách **ROC-AUC** được bảo toàn nguyên vẹn 100% qua Platt scaling ($0.79030$ trên LightGBM và $0.78734$ trên CatBoost) do phép biến đổi $z \mapsto \sigma(az+b)$ với $a > 0$ là hàm đồng biến chặt.

![Biểu đồ đường cong hiệu chuẩn (Reliability Diagrams)](figures/calibration_curves.png)

---

## 4. Tối ưu hóa Ngưỡng Quyết định theo Ma trận Chi phí

### 4.1. Hàm chi phí kinh doanh kỳ vọng

Chi phí bình quân trên mỗi khách hàng (Expected Cost per Customer) được tính theo công thức:
$$\text{Expected Cost} = \frac{1}{N} \left[ \text{Cost}_{FN} \times \sum_{i \in FN} \text{EAD}_i + \text{Cost}_{FP} \times \sum_{i \in FP} \text{EAD}_i \right]$$

1. **Trường hợp Đơn vị (Unit Cost - không trọng số EAD):**
   - $\text{Cost}_{FN} = 0.45$ (LGD = 0.45), $\text{Cost}_{FP} = 0.05$. Tỷ lệ chi phí $FN:FP = 9:1$.
2. **Trường hợp Có trọng số Dư nợ thực tế (EAD-weighted):**
   - $\text{EAD}_i = \max(\text{BILL\_AMT1}_i, 0)$ (dư nợ sao kê gần nhất).

### 4.2. Bảng So sánh Hiệu quả Giảm thiểu Chi phí tại Ngưỡng Tối ưu ($p^*$) vs. Ngưỡng Mặc định ($0.50$)

| Mô hình (Đã hiệu chuẩn Platt) | Ngưỡng Tối ưu ($p^*$) | Chi phí Kỳ vọng Đơn vị ($p^*$) | Chi phí tại Ngưỡng $0.50$ | **Mức giảm Chi phí (%)** | Recall ($p^*$) | Flag Rate ($p^*$) | Precision ($p^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM tuned + Monotonic** | **0.11** | **0.03322** | 0.06471 | **-48.66%** | **89.22%** | **64.72%** | 30.49% |
| **CatBoost tuned** | **0.11** | **0.03348** | 0.06520 | **-48.65%** | **88.62%** | **64.55%** | 30.36% |
| **Logistic baseline** | **0.13** | 0.03584 | 0.06647 | **-46.08%** | 86.89% | 61.42% | 31.28% |

### 4.3. Phân tích Trường hợp Trọng số Dư nợ EAD thực tế
- Khi tính toán có trọng số dư nợ thực tế $\text{EAD} = \max(\text{BILL\_AMT1}, 0)$:
  - Ngưỡng tối ưu dịch chuyển về **$p^* = 0.08$** do các khoản nợ lớn cần được bảo vệ khắt khe hơn.
  - Chi phí tổn thất bình quân kỳ vọng giảm từ **2.856,93 USD/khách hàng** (tại ngưỡng 0.50) xuống còn **1.632,74 USD/khách hàng** (tại ngưỡng 0.08) — **giảm thiểu 42.85% tổng thiệt hại tài chính**!
  - Tại ngưỡng $p^* = 0.11$ (ngưỡng tối ưu đơn vị), chi phí EAD bình quân là 1.684,20 USD/khách hàng, chỉ chênh lệch 3.1% so với điểm cực trị tuyệt đối, chứng minh ngưỡng $p^* = 0.11$ có tính vững rất cao.

![Đường cong Chi phí Kỳ vọng theo Ngưỡng](figures/cost_curves.png)

---

## 5. Phân tích Độ nhạy (Sensitivity Analysis): Tỷ lệ Chi phí FN:FP từ 3:1 đến 10:1

Khảo sát sự dịch chuyển của ngưỡng tối ưu khi khẩu vị rủi ro và chính sách biên lợi nhuận của tổ chức tín dụng thay đổi:

| Tỷ lệ Chi phí FN:FP | Chi phí FN | Chi phí FP | Ngưỡng Tối ưu ($p^*$) | Chi phí Kỳ vọng | Tỷ lệ Gắn cờ (Flag Rate) | Khả năng Phát hiện (Recall) | Độ chính xác (Precision) | Khuyến nghị Ứng dụng |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **3 : 1** | 0.15 | 0.05 | **0.27** | 0.01970 | 25.80% | 59.61% | 51.07% | Giai đoạn mở rộng tín dụng, hạn chế làm phiền khách tốt |
| **5 : 1** | 0.25 | 0.05 | **0.15** | 0.02613 | 49.72% | 81.54% | 36.29% | Trạng thái thị trường ổn định, cân bằng giữa kiểm soát và trải nghiệm |
| **7 : 1** | 0.35 | 0.05 | **0.13** | 0.03021 | 56.98% | 85.61% | 33.24% | Thắt chặt tín dụng khi kinh tế vĩ mô có biến động tiêu cực |
| **9 : 1 (Chuẩn Charter)** | **0.45** | **0.05** | **0.11** | **0.03322** | **64.72%** | **89.22%** | **30.49%** | **Chuẩn mực theo Charter: LGD = 45%, Mất lãi = 5%** |
| **10 : 1** | 0.50 | 0.05 | **0.11** | 0.03444 | 64.72% | 89.22% | 30.49% | Khẩu vị rủi ro cực kỳ thận trọng (khách hàng có nợ lớn) |

### Nhận xét về độ nhạy:
1. Khi tỷ lệ FN:FP tăng từ 3:1 lên 9:1, ngưỡng quyết định hạ từ **0.27 xuống 0.11**, phản ánh trực giác nghiệp vụ: cái giá của việc bỏ sót khách vỡ nợ càng đắt thì hệ thống càng chủ động hạ thấp ngưỡng cảnh báo sớm để can thiệp kịp thời.
2. Tại dải tỷ lệ cao (từ 7:1 đến 10:1), ngưỡng tối ưu ổn định vững chắc trong biên hẹp **[0.11, 0.13]**, cho thấy ngưỡng **$p^* = 0.11$** là lựa chọn bền vững, không bị nhạy cảm quá mức trước sai số ước lượng chi phí.

---

## 6. Quy đổi Xác suất vỡ nợ (PD) sang Điểm Tín dụng theo Phương pháp PDO

### 6.1. Công thức và Tham số Chuẩn hóa
Theo quy định tại `reports/project_plan.md` mục 3.5d và `reports/research_shap_scorecard_pdo.md` mục 4:
- Công thức quy đổi tuyến tính từ odds tốt/xấu:
  $$\text{Score} = \text{Offset} + \text{Factor} \times \ln\left(\frac{1 - \text{PD}}{\text{PD}}\right)$$
- Trong đó các tham số chuẩn mực:
  - $\text{Base Score} = 600$ tại $\text{Base Odds} = 50:1$
  - $\text{PDO} = 20$ (Points to Double the Odds)
  - $\text{Factor} = \frac{\text{PDO}}{\ln 2} = \frac{20}{\ln 2} \approx 28.8539$
  - $\text{Offset} = \text{Base Score} - \text{Factor} \times \ln(\text{Base Odds}) = 600 - 28.8539 \times \ln 50 \approx 487.1229$
- Giới hạn hiển thị: $\text{Score} \in [300, 850]$.

### 6.2. Phân phối Điểm số và Ngưỡng Cắt (Cutoff Score) trên tập Valid

Áp dụng trên mô hình chính thức `LightGBM tuned + Monotonic` (đã hiệu chuẩn Platt):

| Chỉ số Phân phối | Giá trị | Ý nghĩa Nghiệp vụ |
| :--- | :---: | :--- |
| **Điểm trung bình (Mean)** | **534.2 điểm** | Tương ứng PD trung bình khoảng 18.5% |
| **Độ lệch chuẩn (Std)** | **36.5 điểm** | Độ phân tán của danh mục |
| **Trung vị (Median)** | **537.1 điểm** | Phân vị 50% |
| **Điểm thấp nhất (Min)** | **423.4 điểm** | Thuộc về khách hàng có nguy cơ vỡ nợ cao nhất (PD = 89.9%) |
| **Điểm cao nhất (Max)** | **627.1 điểm** | Thuộc về khách hàng an toàn nhất (PD = 0.8%) |
| **Tỷ lệ bị Clip [300, 850]** | **0.00%** | Toàn bộ 6.000 mẫu nằm hoàn toàn trong thang 300–850 mà không bị bão hòa |
| **Cutoff Ngưỡng Tối ưu ($p^* = 0.11$)** | **547.5 điểm** | Ngưỡng phân định phân tầng rủi ro |

![Phân phối Điểm Tín dụng PDO](figures/score_distribution.png)

### 6.3. Bảng Phân tầng Rủi ro & Hành động Đề xuất (Action Tiers)

| Nhóm Phân loại | Dải Điểm PDO | Khoảng Xác suất PD | Tỷ trọng Mẫu Valid | Khuyến nghị Hành động Nghiệp vụ trên App Demo |
| :--- | :---: | :---: | :---: | :--- |
| **Đạt chuẩn (Bình thường)** | **> 547.5 điểm** | **PD < 11.0%** | **35.28%** | Duy trì hạn mức, tiếp tục theo dõi định kỳ bình thường |
| **Theo dõi / Cảnh báo nhẹ** | **515.8 – 547.5 điểm** | **11.0% ≤ PD < 27.0%** | **38.92%** | Gửi thông báo nhắc nợ tự động, tạm dừng nâng hạn mức tín dụng |
| **Cảnh báo cao (Can thiệp sớm)** | **< 515.8 điểm** | **PD ≥ 27.0%** | **25.80%** | Giảm hạn mức khả dụng, chuyển sang bộ phận thu hồi nợ sớm can thiệp |

---

## 7. Kết luận và Bàn giao Tiến độ

1. **Hoàn thành trọn vẹn nhiệm vụ Tuần 3 – T2 của Thành viên B:**
   - Hoàn thành module `src/calibrate.py` bao gồm đầy đủ `PlattCalibrator`, `IsotonicCalibrator`, `CalibratedPipeline` và các hàm phân tích chi phí, ECE.
   - Thêm phương thức `decision_function` cho `SignConstrainedLogisticRegression` trong `src/pipelines.py` bảo đảm tính tương thích đồng bộ.
   - Bổ sung cấu hình `calibration`, `cost_matrix`, `scoring` chuẩn hóa vào `configs/config.yaml`.
   - Xây dựng thành công notebook tương tác `notebooks/04_calibration_threshold.ipynb` tái hiện toàn bộ chu trình.
   - Viết trọn bộ 9 unit test trong `tests/test_calibrate.py` và 1 unit test CLI trong `tests/test_run_pipeline.py` (100% tests passed).
   - Tích hợp cờ `--mode calibrate` vào `scripts/run_pipeline.py`.
2. **Quyết định Kỹ thuật then chốt:**
   - **Mô hình chính thức:** Chọn `LightGBM tuned + Monotonic` kết hợp **Platt scaling fit tay trên margin**.
   - **Tham số Platt Scaling:** $a = 1.0129$, $b = 0.0367$.
   - **Ngưỡng quyết định chính thức:** $p^* = 0.11$ (tương ứng Cutoff $\text{Score}^* = 547.5$ điểm).
   - **Tối ưu kinh doanh:** Giảm 48.66% chi phí kỳ vọng so với ngưỡng 0.50, nâng Recall phát hiện vỡ nợ lên 89.22%.
3. **Kế hoạch tiếp theo (Tuần 3 – T3):**
   - Chạy nghiệm thu cuối cùng trên tập Test (duy nhất 1 lần) để hoàn thiện bảng so sánh và báo cáo tổng kết.
   - Bàn giao `CalibratedPipeline` và các ngưỡng phân tầng cho Thành viên C tích hợp vào giao diện demo Streamlit.
