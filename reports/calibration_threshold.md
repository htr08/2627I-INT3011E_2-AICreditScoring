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

### 2.1. Kết quả fit và đánh giá trên cùng tập Valid (6.000 mẫu; in-sample)

**Lưu ý review:** Các chỉ số dưới đây là kết quả fit calibrator và đánh giá trên cùng 6.000 mẫu Valid, nên Isotonic có lợi thế in-sample. Bảng 2.1 không dùng để xếp hạng phương pháp; xem kết quả 5-fold OOF tại mục 2.2. Tập Test vẫn dành cho nghiệm thu cuối cùng.

| Mô hình | Phương pháp | Brier Score | Log Loss | ROC-AUC | ECE | Platt $a$ (dốc) | Platt $b$ (hệ số tự do) | Ghi chú kỹ thuật |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM tuned + Monotonic** | **Thô (Uncalibrated)** | 0.13312 | 0.42387 | 0.79030 | 0.01633 | – | – | Tỷ lệ dự đoán TB = 21.81% (thực tế 22.12%) |
| | **Platt Scaling (Fit tay)** | **0.13310** | **0.42382** | **0.79030** | **0.01509** | **1.0129** | **+0.0367** | **Bảo toàn SHAP additivity ($a \approx 1.01$), AUC giữ nguyên** |
| | **Isotonic Regression** | 0.13190 | 0.41973 | 0.79325 | 0.00941 | – | – | Brier thấp nhất trên cùng mẫu fit/đo; không bảo toàn tính tuyến tính SHAP |
| **CatBoost tuned** | **Thô (Uncalibrated)** | 0.13351 | 0.42550 | 0.78734 | 0.01756 | – | – | Tỷ lệ dự đoán TB = 21.77% |
| | **Platt Scaling (Fit tay)** | **0.13348** | **0.42543** | **0.78734** | **0.01584** | **1.0142** | **+0.0415** | Bảo toàn SHAP additivity, AUC giữ nguyên |
| | **Isotonic Regression** | 0.13221 | 0.42142 | 0.79046 | 0.01025 | – | – | Brier thấp hơn in-sample; không bảo toàn tuyến tính SHAP |
| **Logistic Regression (baseline)** | **Thô (Uncalibrated)** | 0.13689 | 0.43468 | 0.77123 | 0.02104 | – | – | Baseline Tuần 1 |
| | **Platt Scaling (Fit tay)** | 0.13685 | 0.43459 | 0.77123 | 0.01892 | 0.9854 | +0.0215 | Điều chỉnh nhẹ độ lệch |
| | **Isotonic Regression** | 0.13576 | 0.43085 | 0.77410 | 0.01180 | – | – | Brier thấp hơn in-sample |

### 2.2. So sánh 5-fold OOF trên Valid

Trong từng fold, calibrator chỉ fit trên bốn fold còn lại; chỉ số được tính trên fold giữ lại. Bảng ghi trung bình ± độ lệch chuẩn giữa năm fold. Các giá trị được sinh bởi `--mode calibrate` và lưu tại `reports/calibration_cv_metrics.csv`.

| Mô hình | Phương pháp | Brier | Log Loss | ROC-AUC | ECE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| LightGBM tuned + Monotonic | Thô | 0.13312 ± 0.00761 | 0.42387 ± 0.01915 | 0.79017 ± 0.02368 | 0.02479 ± 0.00568 |
| | Platt | 0.13326 ± 0.00755 | 0.42434 ± 0.01906 | 0.79017 ± 0.02368 | 0.02732 ± 0.00839 |
| | Isotonic | 0.13404 ± 0.00766 | 0.43297 ± 0.01610 | 0.78825 ± 0.02479 | 0.02825 ± 0.01013 |
| CatBoost tuned | Thô | 0.13351 ± 0.00767 | 0.42550 ± 0.01932 | 0.78739 ± 0.02379 | 0.02504 ± 0.00361 |
| | Platt | 0.13364 ± 0.00760 | 0.42598 ± 0.01921 | 0.78739 ± 0.02379 | 0.02504 ± 0.00601 |
| | Isotonic | 0.13410 ± 0.00771 | 0.43217 ± 0.01919 | 0.78629 ± 0.02400 | 0.02802 ± 0.00699 |
| Logistic baseline | Thô | 0.13641 ± 0.00762 | 0.43733 ± 0.01885 | 0.76600 ± 0.01952 | 0.02781 ± 0.00607 |
| | Platt | 0.13654 ± 0.00725 | 0.43755 ± 0.01785 | 0.76600 ± 0.01952 | 0.02593 ± 0.00742 |
| | Isotonic | 0.13688 ± 0.00765 | 0.44196 ± 0.02298 | 0.76411 ± 0.02034 | 0.02824 ± 0.00793 |

OOF không cho thấy Brier của Platt hoặc Isotonic tốt hơn xác suất thô trên ba mô hình; với LightGBM, Brier tăng nhẹ từ 0.13312 lên 0.13326 (Platt) và 0.13404 (Isotonic). Vì vậy không kết luận hiệu chuẩn cải thiện chất lượng xác suất từ kết quả hiện có. Platt vẫn có ưu điểm giữ quan hệ tuyến tính với margin để diễn giải SHAP, nhưng quyết định dùng nó cần cân nhắc ưu điểm giải thích với chất lượng PD đo ngoài mẫu.

---

## 3. Phân tích Kỹ thuật: So sánh Platt Scaling vs. Isotonic Regression

### 3.1. Tại sao Platt scaling fit tay trên raw margin là phương án tối ưu cho XAI / SHAP?

Platt là lựa chọn **tương thích** với việc giữ tính cộng tính SHAP trên logit sau hiệu chuẩn; từ “tối ưu” ở đây chỉ nói về tính chất giải thích này, không khẳng định chất lượng PD tốt hơn.

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

### 3.2. Diễn giải kết quả hiệu chuẩn
- In-sample, Brier Platt chỉ giảm **0.00002** trên LightGBM; OOF, Brier Platt lại cao hơn xác suất thô **0.00013**. Cải thiện in-sample không khái quát sang kết quả OOF trong lần đánh giá này.
- Brier OOF của Isotonic cũng cao hơn xác suất thô trên cả ba ứng viên; mức thấp 0.13190 trong bảng in-sample là lạc quan do fit/đo cùng Valid.
- Platt với $a > 0$ giữ thứ hạng trong mỗi fold và AUC trung bình bằng raw đến độ chính xác hiển thị; đó là tính chất đơn điệu, không đồng nghĩa hiệu chuẩn tốt hơn.

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

### 4.2. Ước lượng in-sample tại ngưỡng được chọn trên Valid

Ngưỡng và chi phí dưới đây được chọn và đo trên cùng tập Valid, vì vậy **mức giảm chi phí là lạc quan và không phải ước lượng hiệu quả triển khai**. Dùng `cv_cost` từ chế độ `--mode calibrate` để báo cáo chi phí held-out theo fold; chỉ Test mới là đánh giá cuối theo kế hoạch.

| Mô hình (Đã hiệu chuẩn Platt) | Ngưỡng Tối ưu ($p^*$) | Chi phí Kỳ vọng Đơn vị ($p^*$) | Chi phí tại Ngưỡng $0.50$ | **Mức giảm Chi phí (%)** | Recall ($p^*$) | Flag Rate ($p^*$) | Precision ($p^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM tuned + Monotonic** | **0.11** | **0.03322** | 0.06471 | **-48.66%** | **89.22%** | **64.72%** | 30.49% |
| **CatBoost tuned** | **0.11** | **0.03348** | 0.06520 | **-48.65%** | **88.62%** | **64.55%** | 30.36% |
| **Logistic baseline** | **0.13** | 0.03584 | 0.06647 | **-46.08%** | 86.89% | 61.42% | 31.28% |

### 4.3. Chi phí held-out 5-fold trên Valid

Mỗi fold chọn ngưỡng trên bốn fold huấn luyện rồi đo chi phí trên fold giữ lại. Đây là ước lượng CV cho quy trình chọn ngưỡng; không phải kết quả Test. Đơn vị chi phí không trọng số là chi phí giả định trên mỗi hồ sơ; EAD-weighted cost dùng `BILL_AMT1` theo tiền gốc **NT$ (TWD)**. Số liệu và độ lệch chuẩn fold nằm trong `reports/calibration_cv_cost.csv`.

| LightGBM tuned + Monotonic | Cách tính | Ngưỡng train-fold TB ± SD | Chi phí held-out tại ngưỡng ± SD | Chi phí held-out tại 0.50 ± SD | Giảm so với 0.50 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Platt | Unit | 0.122 ± 0.013 | 0.03409 ± 0.00260 | 0.06463 ± 0.00424 | 47.3% |
| Isotonic | Unit | 0.096 ± 0.005 | 0.03396 ± 0.00258 | 0.06499 ± 0.00488 | 47.8% |
| Platt | EAD-weighted (NT$/khách hàng) | 0.088 ± 0.008 | 1,645.96 ± 275.43 | 2,854.55 ± 394.33 | 42.3% |
| Isotonic | EAD-weighted (NT$/khách hàng) | 0.090 ± 0.000 | 1,794.02 ± 323.63 | 2,862.51 ± 408.58 | 37.3% |

Ngưỡng unit-cost OOF của Platt trung bình 0.122 (SD 0.013), phù hợp với vùng ứng viên khoảng 0.09–0.13 hơn là một ngưỡng chính xác 0.11. EAD-weighted OOF cho Platt trung bình 0.088 (SD 0.008). Mức giảm chi phí CV tốt hơn mốc 0.50 trong các fold này, nhưng việc chốt ngưỡng và hiệu quả cuối cùng vẫn phải dựa trên Test một lần theo kế hoạch.

### 4.4. Phân tích Trường hợp Trọng số Dư nợ EAD thực tế (in-sample)
- Khi tính toán có trọng số dư nợ thực tế $\text{EAD} = \max(\text{BILL\_AMT1}, 0)$:
  - Ngưỡng tối ưu dịch chuyển về **$p^* = 0.08$** do các khoản nợ lớn cần được bảo vệ khắt khe hơn.
  - EAD dùng `BILL_AMT1` theo đơn vị tiền gốc của bộ dữ liệu UCI, **NT$ (TWD)**; số tiền trong báo cáo được ghi theo NT$/khách hàng.
  - Các con số chi phí và mức chênh giữa ngưỡng 0.08, 0.11 và 0.50 đều là in-sample; chưa đủ cơ sở để kết luận ngưỡng 0.11 vững.

![Đường cong Chi phí Kỳ vọng theo Ngưỡng](figures/cost_curves.png)

---

## 5. Phân tích Độ nhạy (Sensitivity Analysis): Tỷ lệ Chi phí FN:FP từ 3:1 đến 10:1

Khảo sát sự dịch chuyển của ngưỡng tối ưu khi khẩu vị rủi ro và chính sách biên lợi nhuận của tổ chức tín dụng thay đổi:

Các số trong bảng độ nhạy bên dưới là tối ưu in-sample trên Valid. Chúng mô tả ảnh hưởng của giả định chi phí, không phải ngưỡng đã xác nhận ngoài mẫu.

| Tỷ lệ Chi phí FN:FP | Chi phí FN | Chi phí FP | Ngưỡng Tối ưu ($p^*$) | Chi phí Kỳ vọng | Tỷ lệ Gắn cờ (Flag Rate) | Khả năng Phát hiện (Recall) | Độ chính xác (Precision) | Khuyến nghị Ứng dụng |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **3 : 1** | 0.15 | 0.05 | **0.27** | 0.01970 | 25.80% | 59.61% | 51.07% | Giai đoạn mở rộng tín dụng, hạn chế làm phiền khách tốt |
| **5 : 1** | 0.25 | 0.05 | **0.15** | 0.02613 | 49.72% | 81.54% | 36.29% | Trạng thái thị trường ổn định, cân bằng giữa kiểm soát và trải nghiệm |
| **7 : 1** | 0.35 | 0.05 | **0.13** | 0.03021 | 56.98% | 85.61% | 33.24% | Thắt chặt tín dụng khi kinh tế vĩ mô có biến động tiêu cực |
| **9 : 1 (Chuẩn Charter)** | **0.45** | **0.05** | **0.11** | **0.03322** | **64.72%** | **89.22%** | **30.49%** | **Chuẩn mực theo Charter: LGD = 45%, Mất lãi = 5%** |
| **10 : 1** | 0.50 | 0.05 | **0.11** | 0.03444 | 64.72% | 89.22% | 30.49% | Khẩu vị rủi ro cực kỳ thận trọng (khách hàng có nợ lớn) |

### Nhận xét về độ nhạy:
1. Khi tỷ lệ FN:FP tăng từ 3:1 lên 9:1, ngưỡng quyết định hạ từ **0.27 xuống 0.11**, phản ánh trực giác nghiệp vụ: cái giá của việc bỏ sót khách vỡ nợ càng đắt thì hệ thống càng chủ động hạ thấp ngưỡng cảnh báo sớm để can thiệp kịp thời.
2. Đường chi phí khá phẳng trong vùng xấp xỉ **0.08–0.14** (chênh lệch chi phí khoảng 2% theo kết quả hiện có). Vì vậy, nếu cần mô tả vùng ứng viên thì dùng **0.09–0.13**, không coi 0.11 là điểm tối ưu chính xác hay ngưỡng đã được xác nhận. Mức ổn định cần được kiểm tra lại bằng CV/Bootstrap và cuối cùng trên Test.

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

Kết quả in-sample áp dụng trên ứng viên `LightGBM tuned + Monotonic` (đã hiệu chuẩn Platt):

| Chỉ số Phân phối | Giá trị | Ý nghĩa Nghiệp vụ |
| :--- | :---: | :--- |
| **Điểm trung bình (Mean)** | **534.2 điểm** | Không quy đổi trực tiếp thành PD trung bình vì phép PDO là phi tuyến |
| **Độ lệch chuẩn (Std)** | **36.5 điểm** | Độ phân tán của danh mục |
| **Trung vị (Median)** | **537.1 điểm** | Phân vị 50% |
| **Điểm thấp nhất (Min)** | **423.4 điểm** | Thuộc về khách hàng có nguy cơ vỡ nợ cao nhất (PD = 89.9%) |
| **Điểm cao nhất (Max)** | **627.1 điểm** | Thuộc về khách hàng an toàn nhất (PD = 0.8%) |
| **Tỷ lệ bị Clip [300, 850]** | **0.00%** | Toàn bộ 6.000 mẫu nằm hoàn toàn trong thang 300–850 mà không bị bão hòa |
| **Cutoff Ngưỡng Tối ưu ($p^* = 0.11$)** | **547.5 điểm** | Ngưỡng phân định phân tầng rủi ro |

![Phân phối Điểm Tín dụng PDO](figures/score_distribution.png)

PD trung bình của các hồ sơ Valid cần được tính trực tiếp từ `scores`/xác suất hiệu chuẩn, không suy ra từ điểm trung bình. Báo cáo hiện ghi tỷ lệ vỡ nợ thực tế Valid là **22,12%**; nếu nêu PD trung bình dự đoán, hãy ghi riêng giá trị được tính trực tiếp và phân biệt với tỷ lệ thực tế.

### 6.3. Bảng Phân tầng Rủi ro & Hành động Đề xuất (Action Tiers)

| Nhóm Phân loại | Dải Điểm PDO | Khoảng Xác suất PD | Tỷ trọng Mẫu Valid | Khuyến nghị Hành động Nghiệp vụ trên App Demo |
| :--- | :---: | :---: | :---: | :--- |
| **Đạt chuẩn (Bình thường)** | **> 547.5 điểm** | **PD < 11.0%** | **35.28%** | Duy trì hạn mức, tiếp tục theo dõi định kỳ bình thường |
| **Theo dõi / Cảnh báo nhẹ** | **515.8 – 547.5 điểm** | **11.0% ≤ PD < 27.0%** | **38.92%** | Gửi thông báo nhắc nợ tự động, tạm dừng nâng hạn mức tín dụng |
| **Cảnh báo cao (Can thiệp sớm)** | **< 515.8 điểm** | **PD ≥ 27.0%** | **25.80%** | Giảm hạn mức khả dụng, chuyển sang bộ phận thu hồi nợ sớm can thiệp |

Ngưỡng 0.11 gắn cờ **64,72%** hồ sơ trong kết quả in-sample, trong khi precision chỉ **30,49%**. Đây là đánh đổi theo chi phí FN/FP giả định; tỷ lệ cảnh báo cao cần được đối chiếu với năng lực xử lý và chính sách can thiệp. Không nên coi đây là khuyến nghị vận hành trước khi có kết quả OOF/Test và xác nhận sức chứa của quy trình nghiệp vụ.

Đánh giá cuối theo ngưỡng đã khóa 0.09 trên Test cho flag rate **72,98%**, precision **28,02%**, recall **92,46%**; EAD-weighted cost là **NT$1.634,49/khách hàng** so với NT$2.830,29 tại ngưỡng 0.50. Kết quả này chỉ dùng báo cáo nghiệm thu, không dùng để điều chỉnh ngưỡng. Xem [báo cáo Test](test_evaluation/test_evaluation.md).

---

## 7. Kết luận và Bàn giao Tiến độ

1. **Hoàn thành trọn vẹn nhiệm vụ Tuần 3 – T2 của Thành viên B:**
   - Hoàn thành module `src/calibrate.py` bao gồm đầy đủ `PlattCalibrator`, `IsotonicCalibrator`, `CalibratedPipeline` và các hàm phân tích chi phí, ECE.
   - Thêm phương thức `decision_function` cho `SignConstrainedLogisticRegression` trong `src/pipelines.py` bảo đảm tính tương thích đồng bộ.
   - Bổ sung cấu hình `calibration`, `cost_matrix`, `scoring` chuẩn hóa vào `configs/config.yaml`.
   - Xây dựng thành công notebook tương tác `notebooks/04_calibration_threshold.ipynb` tái hiện toàn bộ chu trình.
   - Có bộ kiểm thử cho calibrator và CLI; cần chạy lại kiểm thử sau các thay đổi review.
   - Tích hợp cờ `--mode calibrate` vào `scripts/run_pipeline.py`.
2. **Quyết định Kỹ thuật then chốt:**
   - OOF chưa cho thấy Platt hoặc Isotonic cải thiện Brier so với xác suất thô. Nếu vẫn cần hiệu chuẩn để hỗ trợ diễn giải SHAP, Platt là phương án giữ được tính cộng tính nhưng phải chấp nhận rằng lợi ích PD chưa được chứng minh.
   - **Tham số Platt Scaling:** $a = 1.0129$, $b = 0.0367$.
   - **Ngưỡng 0.11 và cutoff 547.5** hiện chỉ là kết quả chọn/đo in-sample trên Valid, chưa được chốt làm ngưỡng chính thức. Vùng ứng viên cần xem xét là 0.09–0.13 và phải xác nhận lại bằng CV/Test.
   - Mức giảm chi phí 48,66% và Recall 89,22% cũng là số in-sample; không diễn giải như hiệu quả ngoài mẫu.
3. **Nghiệm thu Tuần 3 – T3:**
   - Đã chạy Test một lần; LightGBM Platt đạt ROC-AUC 0.7825 (bootstrap 95% CI 0.7681–0.7973), cao hơn baseline Logistic 0.7609; DeLong p = 1.86 × 10⁻⁹.
   - Bàn giao `CalibratedPipeline` và kết quả Test cho tích hợp demo; giữ nguyên ngưỡng/calibrator đã khóa trước Test.
