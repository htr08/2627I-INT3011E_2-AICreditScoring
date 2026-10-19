# Model Card: Dự đoán Xác suất Vỡ nợ Thẻ tín dụng (LightGBM monotonic + Platt)

_Tuần 3 – T4: Thành viên C (Explainability & Product). Số liệu lấy từ các báo cáo trong `reports/`; nguồn ghi ở từng mục._

---

## 1. Thông tin mô hình

| Hạng mục | Nội dung |
| :--- | :--- |
| Mô hình | LightGBM (`lightgbm_monotone`) với ràng buộc đơn điệu tăng trên 11 biến trễ hạn và tỷ lệ sử dụng hạn mức; tham số trong `configs/tuned_params.yaml` (Optuna, 37 trial, dừng theo điều kiện plateau). |
| Đầu ra | Xác suất vỡ nợ trong tháng kế tiếp (PD), hiệu chuẩn bằng Platt scaling (PD = sigmoid(a·margin + b), a = 1.0128, b = 0.0364) fit trên tập Valid; quy đổi sang điểm PDO (600 điểm tại odds 50:1, PDO = 20). |
| Quy trình | Fit trên Train (18.000 hồ sơ), calibrator fit trên Valid (6.000), đánh giá cuối trên Test (6.000). Không retrain trên Train + Valid. Chia stratified, `random_state = 42`. |
| Đặc trưng | Bộ đặc trưng feature freeze v1 (`configs/feature_freeze_v1.yaml`): cột gốc, đặc trưng dẫn xuất (tỷ lệ sử dụng hạn mức, xu hướng trễ hạn, tỷ lệ trả nợ, biến động dư nợ, cờ trả tối thiểu) và `AGE_BIN`. Không dùng `SEX` và `ID`. |
| Tệp mô hình | `models/final_model.pkl` (`CalibratedPipeline`), xuất bằng `python scripts/run_pipeline.py --mode final-model`. |
| Phiên bản và tái lập | Thư viện pin trong `requirements.txt`. LightGBM đa luồng có thể lệch ở chữ số thập phân cuối giữa các máy: ở lần chạy lại, flag rate trên Test ở ngưỡng 0.09 lệch 5 hồ sơ (72.90% so với 72.98%). |

---

## 2. Mục đích sử dụng

**Mục đích sử dụng:** công cụ cảnh báo sớm và quản lý danh mục thẻ tín dụng hiện hữu, không phải công cụ xét duyệt tín dụng mới.

| | Nội dung |
| :--- | :--- |
| Người dùng dự kiến | Bộ phận quản lý rủi ro và thu hồi nợ của tổ chức phát hành thẻ, dùng để xếp hạng khách hàng hiện hữu theo nguy cơ vỡ nợ tháng kế tiếp và chọn hành động theo dõi, giảm hạn mức hoặc chuyển thu hồi sớm. |
| Ngoài phạm vi | Xét duyệt hoặc từ chối cấp thẻ mới; quyết định tín dụng tự động không có người xem xét; dự đoán PD 12 tháng; áp dụng cho thị trường, giai đoạn hoặc sản phẩm khác dữ liệu huấn luyện (Đài Loan, 2005, thẻ tín dụng). |
| Dạng sử dụng | Ứng dụng demo Streamlit (`app/streamlit_app.py`): nhập hoặc chọn hồ sơ mẫu, trả PD, điểm, mức cảnh báo, hành động đề xuất và đóng góp theo nhóm đặc trưng. |

---

## 3. Dữ liệu

| Hạng mục | Nội dung |
| :--- | :--- |
| Nguồn | Default of Credit Card Clients (UCI/Kaggle): 30.000 hồ sơ, 23 đặc trưng, tỷ lệ vỡ nợ 22.12%. Tải bằng `scripts/download_data.py` kèm checksum. |
| Nhãn | Vỡ nợ trong tháng kế tiếp, trên một snapshot duy nhất năm 2005 của một ngân hàng tại Đài Loan. |
| Xử lý mã không tài liệu hóa | `EDUCATION` 0/5/6 → 4; `MARRIAGE` 0 → 3; `PAY_0` đổi tên `PAY_1`; mã `PAY_x` −2 và 0 giữ nguyên là các trạng thái riêng. |
| Chia dữ liệu | Train 60%, Valid 20%, Test 20% (tỷ lệ vỡ nợ 22.12% ở cả ba); chỉ số dòng lưu ở `data/splits/splits.json`. |
| Chống rò rỉ | Mọi bước dùng nhãn (age binning) nằm trong Pipeline và fit lại theo từng fold. Test chỉ dùng cho nghiệm thu; được chấm hai lượt: đánh giá cuối và xuất dự đoán cho phần fairness (`reports/test_evaluation/test_evaluation.md`). |

---

## 4. Hiệu năng

### 4.1. Trên Test (nguồn: `reports/test_evaluation/test_evaluation.md`)

| Chỉ số | LightGBM + Platt | Logistic Regression baseline |
| :--- | :---: | :---: |
| ROC-AUC (bootstrap 95% CI) | 0.7825 (0.7681, 0.7973) | 0.7609 (0.7461, 0.7765) |
| Gini | 0.5649 | 0.5219 |
| KS | 0.4290 (0.4048, 0.4603) | 0.4046 (0.3771, 0.4341) |
| PR-AUC | 0.5600 (0.5315, 0.5887) | 0.5236 (0.4960, 0.5546) |
| Brier | 0.1348 (0.1288, 0.1405) | 0.1384 (chưa hiệu chuẩn) |
| ECE | 0.0163 | 0.0185 (chưa hiệu chuẩn) |

- **KPI chính đạt:** ΔAUC so với baseline là 0.0215, DeLong 95% CI [0.0145, 0.0285], p = 1.86 × 10⁻⁹.
- **KPI tham chiếu (Charter mục 3.2):** ROC-AUC đạt mốc sàn 0.78 nhưng chưa tới mục tiêu 0.79; Gini (0.5649), KS (0.4290) và PR-AUC (0.55996) dưới mốc sàn. Khoảng tin cậy 95% của cả bốn chỉ số đều chứa mốc sàn.
- ROC-AUC 5-fold CV trên Train là 0.7906 ± 0.0074, cao hơn Test khoảng một độ lệch chuẩn.

### 4.2. Hiệu chuẩn

Platt không cải thiện Brier ngoài mẫu so với xác suất thô (OOF trên Valid: 0.13326 so với 0.13312; Test: 0.1348 so với 0.1347). Platt được giữ để SHAP cộng đúng về logit của PD hiển thị (`reports/calibration_threshold.md` mục 3 và 7). PD trung bình dự đoán trên Test là 22.37%, tỷ lệ vỡ nợ thực tế 22.12%.

### 4.3. Ngưỡng và phân tầng (nguồn: `reports/calibration_threshold.md` mục 4.5 và 6.3)

| Mức | Khoảng PD | Điểm PDO | Hành động đề xuất | Tỷ lệ vỡ nợ thực tế (Valid) |
| :--- | :--- | :--- | :--- | ---: |
| Bình thường | PD < 9% | > 553.9 | Duy trì hạn mức, theo dõi định kỳ | 6.17% |
| Theo dõi | 9% ≤ PD < 27% | 515.8 – 553.9 | Nhắc nợ tự động, tạm dừng nâng hạn mức | 15.54% |
| Cảnh báo cao | PD ≥ 27% | < 515.8 | Giảm hạn mức, chuyển thu hồi sớm | 51.00% |

Ngưỡng vận hành PD ≥ 0.09 được khóa trước khi đánh giá Test (trung bình OOF của ngưỡng tối ưu có trọng số EAD là 0.088, tỷ lệ chi phí FN:FP = 9:1). Trên Test, ngưỡng này cho recall 92.46%, precision 28.02%, flag rate 72.98%; chi phí kỳ vọng có trọng số EAD là NT$1,634.49 mỗi khách hàng so với NT$2,830.29 ở ngưỡng 0.50. Ranh giới 27% trùng ngưỡng tối ưu khi tỷ lệ chi phí FN:FP = 3:1.

---

## 5. Khả năng giải thích

SHAP `TreeExplainer` trên margin, nhân hệ số Platt `a` để tổng đóng góp cộng đúng về logit(PD); đóng góp được cộng về 5 nhóm đặc trưng và quy đổi sang điểm. Nhóm lịch sử trễ hạn chiếm 48.4% mức độ quan trọng, mức sử dụng hạn mức 19.0%, dư nợ 16.4%, hành vi trả nợ 13.4%, nhân khẩu học 2.8%. Reason codes là các nhóm làm giảm điểm từ 1 điểm trở lên. Chi tiết: `reports/xai_report.md`, `reports/shap_agreement.md`.

---

## 6. Công bằng (nguồn: `reports/fairness_evaluation.md`, `reports/fairness_evaluation_test.md`)

- **Chính sách:** `SEX` không được dùng; `AGE` chỉ dùng dưới dạng binning. Giới tính và nhóm tuổi (<30, 30–50, >50) được dùng để đánh giá. Ngưỡng cảnh báo của Disparate Impact Ratio (DIR, flag rate của nhóm chia flag rate của nhóm đông nhất) là ngoài khoảng 0.8–1.25.
- **Tác động của SEX:** thêm SEX vào mô hình đổi AUC Valid +0.0002 và SEX chiếm 0.17% tầm quan trọng; kết quả CV cùng chiều (+0.0001). Việc bỏ SEX không làm giảm hiệu năng đo được.

| Kết quả | Ngưỡng 0.09 | Ngưỡng 0.5 |
| :--- | :--- | :--- |
| Valid, giới tính (Nam so với Nữ) | DIR 1.06, trong dải | DIR 1.28, ngoài dải (CI [1.11, 1.46]) |
| Test, giới tính (Nam so với Nữ) | DIR 1.01, trong dải | DIR 1.05, trong dải |
| Valid, nhóm tuổi | mọi nhóm trong dải | mọi nhóm trong dải |
| Test, nhóm tuổi | mọi nhóm trong dải | <30 ngoài dải (DIR 1.33, CI [1.16, 1.54]) |

- Chênh lệch Nam/Nữ trên Valid không lặp lại trên Test; tỷ lệ vỡ nợ thực tế của Nam cao hơn Nữ ở cả hai tập (Valid 25.5% so với 19.9%; Test 23.4% so với 21.3%).
- AUC theo nhóm tuổi: nhóm >50 có AUC thấp nhất trên Valid (0.759, N = 448) nhưng khoảng tin cậy chồng lấn với nhóm 30–50; trên Test không có nhóm nào khác biệt rõ.
- Ở ngưỡng 0.09, mô hình gắn cờ khoảng 73% hồ sơ nên DIR theo flag rate bị nén. Chỉ số bổ sung DIR của tỷ lệ không bị flag của nhóm >50 là 0.70 (Valid) và 0.64 (Test), ngoài dải. Định nghĩa DIR cần nhóm thống nhất; chỉ số bổ sung này không thuộc Charter.

---

## 7. Hạn chế

- **Dữ liệu:** một ngân hàng, năm 2005, một snapshot; nhãn là vỡ nợ tháng kế tiếp chứ không phải PD 12 tháng; không kiểm định được out-of-time. Mô hình không sẵn sàng triển khai thực tế.
- **Ngưỡng vận hành gắn cờ khoảng 73% danh mục** với precision 28%: đánh đổi theo ma trận chi phí giả định (LGD 0.45, mất lãi 5% EAD, FN:FP = 9:1). Cần đối chiếu với năng lực xử lý của quy trình can thiệp; đường chi phí phẳng trong vùng 0.08–0.14.
- **KPI tuyệt đối:** Gini, KS và PR-AUC trên Test dưới mốc sàn của Charter (khác biệt nằm trong khoảng tin cậy); KS chưa đạt mốc sàn ở cả CV (0.4453) và Test.
- **Hiệu chuẩn:** Platt không cải thiện Brier ngoài mẫu; PD đã hiệu chuẩn chủ yếu phục vụ tính cộng tính của giải thích.
- **Ràng buộc đơn điệu:** áp dụng cho 11 biến số; `PAY_1`–`PAY_6` (one-hot) không được ràng buộc và `PAY_1` không đơn điệu.
- **Giải thích:** SHAP không phải quan hệ nhân quả; reason codes theo nhóm vì các biến trễ hạn và dư nợ tương quan cao (`reports/xai_report.md` mục 6).
- **Công bằng:** DIR không hiệu chỉnh theo chênh lệch tỷ lệ vỡ nợ thực tế giữa các nhóm; recall và tỷ lệ cảnh báo oan theo nhóm chưa được báo cáo; các biến còn lại có thể là biến đại diện cho giới tính; nhóm >50 chỉ có 448 hồ sơ Valid và 471 hồ sơ Test nên khoảng tin cậy rộng.
- **Đánh giá Test:** Test không đánh giá CatBoost, Scorecard hay bản có SEX; Brier và ECE của baseline là số chưa hiệu chuẩn nên không hoàn toàn ngang hàng với mô hình cuối.

---

## 8. Cân nhắc đạo đức và quản trị

- Mô hình hỗ trợ quyết định của con người; hành động giảm hạn mức hoặc chuyển thu hồi cần được người có thẩm quyền xem xét.
- Không dùng thông tin nhạy cảm (`SEX`) làm đầu vào; việc giám sát công bằng cần lặp lại khi có dữ liệu mới.
- Reason codes chưa đáp ứng yêu cầu pháp lý về thông báo hành động bất lợi; không dùng để giải thích quyết định từ chối tín dụng cho khách hàng.
- Khi đưa vào sử dụng thật cần theo dõi độ lệch phân phối, hiệu chuẩn (PD trung bình so với tỷ lệ vỡ nợ thực tế) và công bằng theo nhóm, và huấn luyện lại bằng dữ liệu hiện hành.

---

## 9. Tài liệu liên quan

`reports/project_charter.md`, `reports/week2_review.md`, `reports/calibration_threshold.md`, `reports/test_evaluation/test_evaluation.md`, `reports/shap_agreement.md`, `reports/xai_report.md`, `reports/fairness_evaluation.md`, `reports/fairness_evaluation_test.md`, `reports/error_analysis_validation.md`.
