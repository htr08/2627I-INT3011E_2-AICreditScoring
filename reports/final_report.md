# Báo cáo Tổng kết: Dự đoán Xác suất Vỡ nợ, So sánh Mô hình và Giải thích bằng SHAP

_Tổng hợp từ các báo cáo thành phần trong `reports/`; số liệu và nguồn được dẫn ở từng mục._

---

## 1. Tóm tắt

Dự án xây dựng pipeline dự đoán xác suất vỡ nợ (PD) trong tháng kế tiếp cho danh mục thẻ tín dụng hiện hữu trên bộ dữ liệu Default of Credit Card Clients (30.000 hồ sơ, tỷ lệ vỡ nợ 22.12%). Mô hình cuối là LightGBM tuned với ràng buộc đơn điệu trên 11 biến trễ hạn và tỷ lệ sử dụng hạn mức, hiệu chuẩn bằng Platt scaling, không dùng `SEX`.

| Kết quả | Giá trị |
|---|---|
| ROC-AUC trên Test | 0.7825 (bootstrap 95% CI 0.7681–0.7973) |
| So với baseline Logistic Regression (0.7609) | ΔAUC = 0.0215, DeLong p = 1.86 × 10⁻⁹: KPI chính đạt |
| ROC-AUC 5-fold CV | 0.7906 ± 0.0074 |
| Ngưỡng vận hành | PD ≥ 0.09 (recall 92.46%, precision 28.02%, gắn cờ 72.98% hồ sơ Test) |
| Fairness | DIR theo flag rate (chỉ số Charter) của mọi nhóm giới tính và tuổi nằm trong dải 0.8–1.25 ở ngưỡng vận hành; chỉ số bổ sung DIR không bị flag của nhóm >50 ngoài dải (mục 9) |
| Hạn chế chính | Dữ liệu một ngân hàng, năm 2005, một snapshot; Gini, KS, PR-AUC trên Test dưới mốc sàn tham chiếu; ngưỡng gắn cờ khoảng 73% danh mục |

Mô hình là công cụ cảnh báo sớm và quản lý danh mục hiện hữu, không phải công cụ xét duyệt tín dụng mới, và không sẵn sàng triển khai thực tế.

---

## 2. Mục tiêu, phạm vi và tiêu chí thành công

Mục tiêu: PD đã hiệu chuẩn, so sánh nhiều thuật toán có kiểm định thống kê, giải thích bằng SHAP và ứng dụng demo Streamlit (`reports/project_plan.md`). Tiêu chí theo Project Charter (`reports/project_charter.md`):

| Tiêu chí | Nội dung | Kết quả |
|---|---|---|
| KPI chính | Vượt baseline Logistic Regression về ROC-AUC có ý nghĩa thống kê | Đạt (mục 7) |
| KPI tham chiếu (Test) | ROC-AUC ≥ 0.78 / 0.79; Gini ≥ 0.57 / 0.59; KS ≥ 0.45 / 0.47; PR-AUC ≥ 0.56 / 0.58 (sàn / mục tiêu) | ROC-AUC đạt sàn; ba chỉ số còn lại dưới sàn, khoảng tin cậy chứa mốc sàn (mục 7) |
| Calibration | Hiệu chuẩn trên Valid, báo cáo Brier và Calibration Curve | Có (mục 6) |
| SHAP | Global và Local cho mô hình cuối | Có (mục 8) |
| Chống rò rỉ | Mọi bước dùng nhãn fit trong từng fold | Có (mục 3, 4) |
| Test | Chỉ đánh giá một lần | Một lượt đánh giá, kèm một lượt xuất dự đoán cho fairness (mục 7) |

Scope tối thiểu (Logistic Regression, XGBoost, SHAP, Calibration) được giữ và mở rộng bằng Decision Tree, Random Forest, LightGBM, CatBoost, Logistic Scorecard, monotonic constraints và SMOTE; MLP không thực hiện. Mô hình cuối là LightGBM chứ không phải XGBoost (mục 5).

---

## 3. Dữ liệu và khám phá

| Hạng mục | Nội dung |
|---|---|
| Chia dữ liệu | Stratified, `random_state = 42`: Train 18.000, Valid 6.000, Test 6.000 hồ sơ; tỷ lệ vỡ nợ 22.12% ở cả ba tập (`data/splits/`). |
| Mã không tài liệu hóa | `EDUCATION` 0/5/6 gộp vào 4; `MARRIAGE` 0 gộp vào 3; `PAY_0` đổi tên `PAY_1`; mã `PAY_x` = −2 và 0 giữ là trạng thái riêng. |
| Tín hiệu mạnh nhất | Trạng thái trả nợ: tỷ lệ vỡ nợ theo `PAY_1` (Train) là 12.5% (−2), 16.5% (−1), 12.6% (0), 34.4% (1), 69.5% (2), 78.7% (3). |
| Quan hệ khác | Tỷ lệ vỡ nợ giảm theo tứ phân vị hạn mức (32.3%, 24.7%, 17.5%, 13.2%); Nam 24.0% so với Nữ 20.9%. |
| Information Value (Train) | `PAY_1` 0.9155, `PAY_LATE_2PLUS_COUNT` 0.8644, `PAY_LATE_CONSECUTIVE` 0.8474, `PAY_MAX` 0.8200, `PAY_MEAN` 0.7567; `BILL_AMT1`–`BILL_AMT6` và `MARRIAGE` dưới 0.02. |

Nguồn: `notebooks/01_eda.ipynb`, `configs/feature_freeze_v1.yaml`, `reports/experiments_optuna_tuning.md` mục 8.2.

---

## 4. Feature Engineering và chọn đặc trưng

Bộ đặc trưng feature freeze v1 (chốt 2026-10-07, `configs/feature_freeze_v1.yaml`) gồm cột gốc, đặc trưng dẫn xuất và `AGE_BIN`:

- Tỷ lệ sử dụng hạn mức (`UTIL_1`–`UTIL_6`, `UTIL_MEAN`); xu hướng trễ hạn (`PAY_MEAN`, `PAY_MAX`, `PAY_SLOPE`, `PAY_LATE_CONSECUTIVE`, `PAY_LATE_2PLUS_COUNT`).
- Tỷ lệ trả nợ trên dư nợ (`PAY_RATIO_*`), biến động dư nợ (`BILL_STD`, `BILL_DELTA`), cờ trả mức tối thiểu (`MIN_PAY_FLAG_*`).
- `AGE` thay bằng `AGE_BIN` (binning theo IV, 3–8 bin, fit lại trong từng fold); loại `SEX` và `ID`.

Lọc tương quan (Spearman > 0.9) và các phương án mã hóa khác (thứ tự thô, tách `PAY_i` thành số tháng trễ và cờ trạng thái) được thử sau tuning: không phương án nào cải thiện ROC-AUC (chênh lệch từ −0.0011 đến 0.0000, nhỏ hơn độ lệch chuẩn giữa các fold), nên feature freeze v1 được giữ nguyên và lọc tương quan chỉ là tùy chọn cho thí nghiệm Scorecard (`reports/feature_selection_decision.md`, `reports/experiments_optuna_tuning.md` mục 8).

---

## 5. Mô hình và so sánh

### 5.1. Kết quả 5-fold CV trên Train

| Mô hình | ROC-AUC (mean ± std) |
|---|---|
| Logistic Regression (baseline) | 0.7745 ± 0.0116 |
| Decision Tree (độ sâu 5) | 0.7672 ± 0.0120 |
| Logistic Scorecard (WoE) | 0.7790 ± 0.0083 |
| Random Forest, XGBoost (tham số mặc định) | 0.7667 ± 0.0097; 0.7597 ± 0.0073 |
| LightGBM, CatBoost (tham số mặc định) | 0.7805 ± 0.0077; 0.7844 ± 0.0074 |
| LightGBM tuned + monotonic | 0.7906 ± 0.0074 |
| CatBoost tuned | 0.7911 ± 0.0073 |

Nguồn: `reports/model_comparison.md` (kèm Gini, KS, PR-AUC, Brier của từng cấu hình). Kiểm định cặp theo fold: LightGBM monotonic và CatBoost tuned đều vượt baseline trên 5/5 fold (ΔAUC +0.0161, p = 0.004; +0.0166, p = 0.002); hai mô hình không phân biệt được với nhau (+0.0005, p = 0.31); Scorecard không vượt baseline có ý nghĩa (+0.0045, p = 0.34).

### 5.2. Các thí nghiệm bổ sung

- **Tuning (Optuna, 5-fold CV):** ba study (LightGBM, LightGBM monotonic, CatBoost) đều dừng theo điều kiện plateau của Charter sau 34–37 trial (10–40 phút); tham số lưu ở `configs/tuned_params.yaml`.
- **Mất cân bằng:** `class_weight` đưa recall lên khoảng 61% ở ngưỡng 0.5 nhưng làm Brier tăng từ 0.133 lên 0.168; SMOTE trong từng fold không cải thiện AUC (0.7747–0.7776). Cả hai không dùng cho mô hình chính thức (`reports/experiments_boosting_imbalance.md`).
- **Logistic Scorecard:** ràng buộc dấu hệ số và coarse classing cho bảng điểm 24 biến, 156 thuộc tính, điểm của mọi biến liên tục đơn điệu theo giá trị; đánh đổi khoảng 0.012 AUC so với mô hình boosting (`reports/logistic_scorecard.md`).
- **Biến `SEX`:** thêm vào mô hình đổi AUC trung bình +0.0001 và chiếm 0.17% (LightGBM) đến 0.54% (CatBoost) tầm quan trọng, nên loại khỏi mô hình chính thức không làm giảm hiệu năng.

### 5.3. Lựa chọn mô hình cuối

LightGBM tuned + monotonic được chọn làm mô hình chính vì AUC tương đương CatBoost, ràng buộc đơn điệu nhất quán với nghiệp vụ, thời gian huấn luyện ngắn hơn 5–25 lần và tương thích `TreeExplainer` cho demo. CatBoost là mô hình đối chứng, Scorecard là mô hình tham chiếu minh bạch; XGBoost và Random Forest không vào nhóm cuối vì ở cấu hình mặc định thấp hơn baseline (`reports/week2_review.md`).

---

## 6. Hiệu chuẩn, ngưỡng và điểm tín dụng

- **Hiệu chuẩn:** Platt scaling fit tay trên margin thô, fit trên Valid (a = 1.0128, b = 0.0364). Platt giữ quan hệ tuyến tính với margin nên SHAP cộng đúng về logit(PD). Trên Valid theo 5-fold OOF, Platt và Isotonic đều không cải thiện Brier so với xác suất thô (LightGBM: 0.13326 và 0.13404 so với 0.13312); trên Test cũng vậy (0.1348 so với 0.1347). Lợi ích của hiệu chuẩn vì vậy là tính cộng tính của giải thích, không phải chất lượng PD.
- **Ma trận chi phí (Charter mục 7):** FN ≈ 0.45 × EAD, FP ≈ 0.05 × EAD, tỷ lệ FN:FP = 9:1; độ nhạy từ 3:1 (ngưỡng tối ưu 0.27) đến 10:1 (0.11), đường chi phí phẳng trong vùng 0.08–0.14.
- **Ngưỡng vận hành 0.09:** trung bình OOF của ngưỡng tối ưu có trọng số EAD là 0.088 ± 0.008, làm tròn; khóa trước khi đánh giá Test (`reports/calibration_threshold.md` mục 4.5).
- **Điểm PDO:** 600 điểm tại odds 50:1, PDO = 20; cutoff 553.9 điểm tương ứng PD 0.09.
- **Phân tầng cho demo:** PD < 9% Bình thường (27.5% hồ sơ Valid, tỷ lệ vỡ nợ thực tế 6.2%); 9% ≤ PD < 27% Theo dõi (46.65%, 15.5%); PD ≥ 27% Cảnh báo cao (25.8%, 51.0%).

---

## 7. Đánh giá cuối trên Test

Test được đánh giá một lần bằng mô hình, calibrator và ngưỡng đã khóa (`reports/test_evaluation/test_evaluation.md`).

| Chỉ số | LightGBM + Platt | Logistic baseline |
|---|---|---|
| ROC-AUC (bootstrap 95% CI) | 0.7825 (0.7681, 0.7973) | 0.7609 (0.7461, 0.7765) |
| Gini | 0.5649 | 0.5219 |
| KS | 0.4290 | 0.4046 |
| PR-AUC | 0.5600 | 0.5236 |
| Brier | 0.1348 | 0.1384 (chưa hiệu chuẩn) |

- **KPI chính:** ΔAUC = 0.0215, DeLong 95% CI [0.0145, 0.0285], p = 1.86 × 10⁻⁹; bootstrap CI [0.0143, 0.0286]. Đạt.
- **KPI tham chiếu:** ROC-AUC đạt mốc sàn 0.78 nhưng chưa tới mục tiêu 0.79; Gini (0.5649 so với 0.57), KS (0.4290 so với 0.45) và PR-AUC (0.55996 so với 0.56) dưới mốc sàn. Khoảng tin cậy 95% của cả bốn chỉ số chứa mốc sàn. ROC-AUC CV cao hơn Test khoảng một độ lệch chuẩn.
- **Tại ngưỡng 0.09:** recall 92.46%, precision 28.02%, flag rate 72.98%; chi phí kỳ vọng có trọng số EAD NT$1,634.49 mỗi khách hàng so với NT$2,830.29 ở ngưỡng 0.50.
- **Quy trình:** Test được chấm hai lượt: đánh giá cuối và xuất dự đoán từng hồ sơ cho phần fairness (không tính lại chỉ số, không đổi quyết định nào).

---

## 8. Giải thích bằng SHAP

`shap.TreeExplainer` trên margin thô, nhân hệ số Platt để tổng đóng góp cộng đúng về logit(PD), cộng về 5 nhóm đặc trưng và quy đổi sang điểm (`reports/xai_report.md`).

- **Toàn cục:** lịch sử trễ hạn chiếm 48.4% mức độ quan trọng, mức sử dụng hạn mức 19.0%, dư nợ 16.4%, hành vi trả nợ 13.4%, nhân khẩu học 2.8%. Bốn biến đầu (`PAY_MAX`, `PAY_1`, `PAY_LATE_CONSECUTIVE`, `PAY_LATE_2PLUS_COUNT`) đều thuộc nhóm trễ hạn.
- **Hình dạng quan hệ:** `PAY_1` không đơn điệu (one-hot, không ràng buộc được) nhưng không ảnh hưởng đóng góp theo nhóm; tương tác yếu (cặp mạnh nhất 0.048).
- **Đồng thuận giữa mô hình:** nhóm trễ hạn đứng đầu và nhân khẩu học đứng cuối ở mọi cấu hình; reason code đầu tiên trùng 96% hồ sơ rủi ro cao giữa LightGBM và XGBoost `max_depth=4`, 76% với Logistic (`reports/shap_agreement.md`).
- **Reason codes:** tối đa 3 nhóm làm giảm điểm (đóng góp ≤ −1 điểm) và 3 yếu tố tích cực. Ở mức Cảnh báo cao, lịch sử trễ hạn là lý do đầu tiên của 78.7% hồ sơ; ở mức Theo dõi, lý do phân tán giữa sử dụng hạn mức (44.5%), dư nợ (33.1%) và trễ hạn (9.9%).
- **Giới hạn:** SHAP không phải quan hệ nhân quả; các biến trễ hạn và dư nợ tương quan cao nên chỉ đóng góp theo nhóm được dùng làm lý do.

---

## 9. Phân tích lỗi và công bằng

### 9.1. Phân tích lỗi (Valid, `reports/error_analysis_validation.md`)

| Ngưỡng | Bỏ sót khách vỡ nợ (FN) | Cảnh báo oan (FP) |
|---|---|---|
| 0.09 (vận hành) | 102 hồ sơ (7.7% số hồ sơ vỡ nợ) | 3,123 hồ sơ (66.8% số hồ sơ không vỡ nợ) |
| 0.5 (đối chiếu) | 836 hồ sơ (63.0%) | 233 hồ sơ (5.0%) |

Hồ sơ bị bỏ sót gần nhóm khách hàng không vỡ nợ hơn nhóm vỡ nợ được phát hiện ở 7/8 đặc trưng mô tả (không trễ hạn ở tháng gần nhất, mức sử dụng hạn mức thấp, tỷ lệ trả nợ cao), nên các đặc trưng hiện có không tách được chúng. Ở ngưỡng 0.5, hồ sơ cảnh báo oan gần nhóm vỡ nợ được phát hiện ở 7/8 đặc trưng. Mẫu lỗi tương tự ở Scorecard, và 96.7% FN ở ngưỡng 0.5 trùng giữa hai mô hình.

### 9.2. Công bằng (`reports/fairness_evaluation.md`, `reports/fairness_evaluation_test.md`)

Chính sách: `SEX` không dùng trong mô hình; `AGE` chỉ dùng dạng binning. Disparate Impact Ratio (DIR) là flag rate của nhóm chia flag rate của nhóm đông nhất, cảnh báo khi ngoài 0.8–1.25.

| Kết quả | Ngưỡng 0.09 | Ngưỡng 0.5 |
|---|---|---|
| Valid, giới tính (Nam so với Nữ) | DIR 1.06, trong dải | DIR 1.28, ngoài dải |
| Test, giới tính (Nam so với Nữ) | DIR 1.01, trong dải | DIR 1.05, trong dải |
| Valid, nhóm tuổi | mọi nhóm trong dải | mọi nhóm trong dải |
| Test, nhóm tuổi | mọi nhóm trong dải | nhóm <30 ngoài dải (DIR 1.33) |

- Thêm `SEX` vào mô hình đổi AUC Valid +0.0002; việc bỏ `SEX` không làm giảm hiệu năng nhưng cũng không triệt tiêu chênh lệch flag rate trên Valid, phù hợp với chênh lệch tỷ lệ vỡ nợ thực tế giữa hai nhóm.
- AUC thấp nhất ở nhóm >50 trên Valid (0.759, N = 448) nhưng khoảng tin cậy chồng lấn với nhóm 30–50.
- Ở ngưỡng 0.09 mô hình gắn cờ khoảng 73% hồ sơ nên DIR theo flag rate bị nén; DIR của tỷ lệ không bị flag (chỉ số bổ sung, không thuộc Charter) của nhóm >50 là 0.70 (Valid) và 0.64 (Test), ngoài dải. Chỉ số theo Charter là DIR theo flag rate; chỉ số bổ sung chỉ để tham chiếu.

---

## 10. Ứng dụng demo

Ứng dụng Streamlit (`app/streamlit_app.py`) nạp `models/final_model.pkl`, nhận hồ sơ nhập tay hoặc hồ sơ mẫu (Bình thường, Ca biên, Cảnh báo cao), kiểm tra đầu vào và trả PD, điểm tín dụng, mức cảnh báo kèm hành động đề xuất (theo dõi, giảm hạn mức, chuyển thu hồi sớm), đóng góp theo nhóm đặc trưng dạng waterfall và 10 biến lớn nhất. Ba hồ sơ mẫu cho PD 4.0%, 26.0% và 73.3%, tương ứng ba mức cảnh báo. Khi không nạp được mô hình cuối, app dùng baseline từ MLflow Registry rồi mock model (`reports/demo_wireframe.md`).

---

## 11. Hạn chế và rủi ro

- **Dữ liệu:** một ngân hàng, năm 2005, một snapshot; nhãn là vỡ nợ tháng kế tiếp chứ không phải PD 12 tháng; không kiểm định được out-of-time.
- **Ngưỡng vận hành:** gắn cờ khoảng 73% hồ sơ với precision 28% theo ma trận chi phí giả định (LGD 0.45, FN:FP = 9:1); cần đối chiếu với năng lực xử lý của quy trình can thiệp.
- **KPI tuyệt đối:** Gini, KS và PR-AUC trên Test dưới mốc sàn (khác biệt nằm trong khoảng tin cậy); KS cũng chưa đạt mốc sàn ở CV.
- **Hiệu chuẩn:** Platt không cải thiện Brier ngoài mẫu.
- **Giải thích:** SHAP không phải quan hệ nhân quả; ràng buộc đơn điệu chưa áp dụng được cho `PAY_1`–`PAY_6`; reason codes chưa đáp ứng yêu cầu pháp lý về thông báo hành động bất lợi.
- **Công bằng:** DIR không hiệu chỉnh theo chênh lệch tỷ lệ vỡ nợ giữa các nhóm; recall và tỷ lệ cảnh báo oan theo nhóm chưa được báo cáo; nhóm >50 chỉ có 448 hồ sơ Valid và 471 hồ sơ Test.
- **Đánh giá Test:** CatBoost, Scorecard và bản có `SEX` không được chấm trên Test; Brier của baseline là số chưa hiệu chuẩn.
- **Tái lập:** LightGBM đa luồng có thể lệch ở chữ số thập phân cuối giữa các máy; ở lần chạy lại, flag rate trên Test ở ngưỡng 0.09 lệch 5 hồ sơ.

---

## 12. Kết luận và hướng phát triển

Mô hình cuối vượt baseline với ý nghĩa thống kê, cho PD hiệu chuẩn có thể giải thích theo nhóm đặc trưng và đã được tích hợp vào demo kèm phân tầng rủi ro. Trần hiệu năng của bộ dữ liệu (ROC-AUC khoảng 0.78–0.79) không bị phá bởi tuning hay thay đổi đặc trưng, nên cải thiện tiếp cần thêm thông tin, không phải thêm tuning.

Hướng phát triển: (1) dữ liệu nhiều giai đoạn để kiểm định out-of-time và PD 12 tháng; (2) chọn ngưỡng theo sức chứa của quy trình can thiệp thay vì chỉ theo chi phí giả định; (3) bổ sung đánh giá cơ hội bình đẳng theo nhóm (recall và tỷ lệ cảnh báo oan); (4) giám sát độ lệch phân phối, hiệu chuẩn và công bằng khi triển khai.

---

## 13. Tài liệu và tái lập

| Nội dung | Tài liệu |
|---|---|
| Kế hoạch, Charter | `project_plan.md`, `project_charter.md` |
| Đặc trưng, Scorecard | `feature_selection_decision.md`, `logistic_scorecard.md` |
| So sánh, tuning, chọn mô hình | `model_comparison.md`, `experiments_boosting_imbalance.md`, `experiments_optuna_tuning.md`, `week2_review.md` |
| Hiệu chuẩn, ngưỡng, Test | `calibration_threshold.md`, `test_evaluation/test_evaluation.md` |
| XAI | `xai_report.md`, `shap_agreement.md` |
| Lỗi, công bằng | `error_analysis_validation.md`, `fairness_evaluation.md`, `fairness_evaluation_test.md` |
| Model Card, demo | `model_card.md`, `demo_wireframe.md` |

Lệnh tái lập và cấu trúc repository: `README.md` ở thư mục gốc.
