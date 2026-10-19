# Báo cáo Đánh giá Tính Công bằng (Fairness Check) trên tập Valid

_Tuần 3 – T3: Thành viên A (Data Engineer / Analyst)_

---

## 1. Phạm vi và thiết lập

| Hạng mục | Nội dung |
| :--- | :--- |
| Tập dữ liệu | Valid, n = 6000 (1327 hồ sơ vỡ nợ, 22.12%). Tập Test không được sử dụng. |
| Mô hình chính | LightGBM tuned + monotonic (`lightgbm_monotone`), không dùng SEX, huấn luyện trên Train; Platt scaling fit trên Valid (a = 1.0128, b = 0.0364). Đây là mô hình cuối theo `week2_review.md`. |
| Mô hình đối chiếu | Logistic Scorecard (WoE), không dùng SEX: mô hình tham chiếu minh bạch (`week2_review.md` mục 6) và là mô hình của bản đánh giá công bằng ban đầu. |
| Ngưỡng vận hành | PD ≥ 0.09: ngưỡng do thành viên B khóa trước khi đánh giá Test, làm tròn từ trung bình OOF EAD-weighted (0.088) với tỷ lệ chi phí FN:FP = 9:1 theo Charter mục 7 (`reports/test_evaluation/test_evaluation.md`). Mô hình chính gắn cờ 72.5% hồ sơ Valid. |
| Ngưỡng đối chiếu | PD ≥ 0.5: ngưỡng phân loại mặc định, mô hình chính gắn cờ 12.1% hồ sơ Valid. Dùng để kiểm tra độ nhạy của kết luận theo ngưỡng, vì ngưỡng vận hành gắn cờ phần lớn danh mục. |
| Nhóm | Giới tính (SEX = 1 Nam, 2 Nữ) và nhóm tuổi <30, 30–50, >50 (30 và 50 thuộc nhóm 30–50). Hai thuộc tính này chỉ dùng để đánh giá; mô hình không dùng SEX, AGE được đưa vào dưới dạng `AGE_BIN` (Charter mục 8). |
| Flag rate | Tỷ lệ hồ sơ bị gắn cờ can thiệp (PD ≥ ngưỡng). |
| DIR | Disparate Impact Ratio = flag rate của nhóm / flag rate của nhóm tham chiếu. Nhóm tham chiếu là nhóm đông nhất của từng thuộc tính: **Nữ** (giới tính) và **30–50** (nhóm tuổi). Cờ là kết quả bất lợi, nên cảnh báo khi DIR ngoài [0.8, 1.25] (ngưỡng 0.8 của Charter mục 8 và nghịch đảo 1 / 0.8 cho chiều ngược lại). CI 95% bằng bootstrap 1000 lần trong từng nhóm. |
| DIR không bị flag | (1 − flag rate của nhóm) / (1 − flag rate của tham chiếu): quy tắc 4/5 áp dụng cho kết quả có lợi (không bị can thiệp). Chỉ số bổ sung, không thuộc Charter; dấu * đánh dấu giá trị ngoài [0.8, 1.25]. |
| AUC | AUC của PD trong từng nhóm, CI 95% bằng bootstrap 1000 lần. |

---

## 2. Tác động của biến SEX

| Mô hình | SEX trong đầu vào | AUC Valid | Flag Nam / Nữ (0.09) | Nam/Nữ (0.09) | Flag Nam / Nữ (0.5) | Nam/Nữ (0.5) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| LightGBM monotonic + Platt | Không | 0.7903 | 74.8% / 70.9% | 1.06 | 13.9% / 10.8% | 1.28 |
| LightGBM monotonic + Platt (có SEX) | Có | 0.7905 | 75.6% / 70.2% | 1.08 | 14.2% / 10.7% | 1.32 |
| Logistic Scorecard (WoE) | Không | 0.7706 | 78.6% / 76.1% | 1.03 | 14.2% / 10.6% | 1.34 |
| Logistic Scorecard (WoE, có SEX) | Có | 0.7706 | 78.6% / 76.1% | 1.03 | 14.2% / 10.6% | 1.34 |

- Bản Scorecard có SEX giữ SEX trong đầu vào nhưng bước WoE loại SEX vì IV = 0.0078 < ngưỡng 0.02. PD của hai bản Scorecard trùng nhau, nên số liệu của hai bản giống hệt nhau và không phản ánh tác động của SEX. Tác động của SEX được đo trên mô hình LightGBM, nơi SEX thực sự là đầu vào.
- Với mô hình chính, thêm SEX đổi AUC Valid +0.0002 và SEX chiếm 0.17% tầm quan trọng (gain) của mô hình có SEX.
- Kết quả CV 5-fold của thành viên B (`model_comparison.md` mục 6.4) cùng chiều: thêm SEX đổi AUC trung bình +0.0001; tỷ trọng tầm quan trọng của SEX là LightGBM tuned + monotonic 0.17%; CatBoost tuned 0.54%.

---

## 3. Kết quả mô hình chính (không dùng SEX)

### 3.1. Ngưỡng vận hành (0.09)

| Thuộc tính | Nhóm | N | Vỡ nợ thực tế | Flag rate | DIR [95% CI] | Trạng thái | DIR không bị flag | AUC [95% CI] |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| Giới tính | Nam | 2385 | 25.5% | 74.8% | 1.06 [1.02, 1.09] | Đạt | 0.86 | 0.781 [0.759, 0.802] |
| Giới tính | Nữ | 3615 | 19.9% | 70.9% | 1.00 (tham chiếu) | – | – | 0.796 [0.776, 0.815] |
| Nhóm tuổi | <30 | 1878 | 21.5% | 75.9% | 1.09 [1.05, 1.12] | Đạt | 0.80 | 0.801 [0.777, 0.825] |
| Nhóm tuổi | 30–50 | 3674 | 21.8% | 69.9% | 1.00 (tham chiếu) | – | – | 0.790 [0.772, 0.809] |
| Nhóm tuổi | >50 | 448 | 27.0% | 79.0% | 1.13 [1.07, 1.19] | Đạt | 0.70 * | 0.759 [0.705, 0.808] |

### 3.2. Ngưỡng đối chiếu (0.5)

| Thuộc tính | Nhóm | N | Vỡ nợ thực tế | Flag rate | DIR [95% CI] | Trạng thái | DIR không bị flag | AUC [95% CI] |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| Giới tính | Nam | 2385 | 25.5% | 13.9% | 1.28 [1.11, 1.46] | Vượt ngưỡng | 0.97 | 0.781 [0.759, 0.802] |
| Giới tính | Nữ | 3615 | 19.9% | 10.8% | 1.00 (tham chiếu) | – | – | 0.796 [0.776, 0.815] |
| Nhóm tuổi | <30 | 1878 | 21.5% | 13.3% | 1.18 [1.02, 1.37] | Đạt | 0.98 | 0.801 [0.777, 0.825] |
| Nhóm tuổi | 30–50 | 3674 | 21.8% | 11.3% | 1.00 (tham chiếu) | – | – | 0.790 [0.772, 0.809] |
| Nhóm tuổi | >50 | 448 | 27.0% | 13.4% | 1.19 [0.91, 1.53] | Đạt | 0.98 | 0.759 [0.705, 0.808] |

---

## 4. Đối chiếu với Logistic Scorecard (không dùng SEX)

### 4.1. Ngưỡng vận hành (0.09)

| Thuộc tính | Nhóm | N | Vỡ nợ thực tế | Flag rate | DIR [95% CI] | Trạng thái | DIR không bị flag | AUC [95% CI] |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| Giới tính | Nam | 2385 | 25.5% | 78.6% | 1.03 [1.01, 1.06] | Đạt | 0.90 | 0.759 [0.735, 0.781] |
| Giới tính | Nữ | 3615 | 19.9% | 76.1% | 1.00 (tham chiếu) | – | – | 0.778 [0.757, 0.797] |
| Nhóm tuổi | <30 | 1878 | 21.5% | 83.6% | 1.14 [1.11, 1.17] | Đạt | 0.61 * | 0.782 [0.757, 0.808] |
| Nhóm tuổi | 30–50 | 3674 | 21.8% | 73.1% | 1.00 (tham chiếu) | – | – | 0.772 [0.754, 0.793] |
| Nhóm tuổi | >50 | 448 | 27.0% | 83.0% | 1.14 [1.08, 1.19] | Đạt | 0.63 * | 0.729 [0.672, 0.781] |

### 4.2. Ngưỡng đối chiếu (0.5)

| Thuộc tính | Nhóm | N | Vỡ nợ thực tế | Flag rate | DIR [95% CI] | Trạng thái | DIR không bị flag | AUC [95% CI] |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| Giới tính | Nam | 2385 | 25.5% | 14.2% | 1.34 [1.17, 1.54] | Vượt ngưỡng | 0.96 | 0.759 [0.735, 0.781] |
| Giới tính | Nữ | 3615 | 19.9% | 10.6% | 1.00 (tham chiếu) | – | – | 0.778 [0.757, 0.797] |
| Nhóm tuổi | <30 | 1878 | 21.5% | 12.9% | 1.15 [0.99, 1.34] | Đạt | 0.98 | 0.782 [0.757, 0.808] |
| Nhóm tuổi | 30–50 | 3674 | 21.8% | 11.2% | 1.00 (tham chiếu) | – | – | 0.772 [0.754, 0.793] |
| Nhóm tuổi | >50 | 448 | 27.0% | 14.7% | 1.31 [1.02, 1.67] | Vượt ngưỡng | 0.96 | 0.729 [0.672, 0.781] |

---

## 5. Kết luận

1. Giới tính (mô hình chính, không dùng SEX), flag rate: ngưỡng 0.09: Nam 74.8% so với Nữ 70.9% (DIR Nam/Nữ = 1.06, CI 95% [1.02, 1.09], trong dải 0.8–1.25); ngưỡng 0.5: Nam 13.9% so với Nữ 10.8% (DIR Nam/Nữ = 1.28, CI 95% [1.11, 1.46], ngoài dải 0.8–1.25). Tỷ lệ vỡ nợ thực tế trên Valid: Nam 25.5%, Nữ 19.9%. Nhóm bị flag nhiều hơn (Nam) cũng là nhóm có tỷ lệ vỡ nợ thực tế cao hơn.
2. Nhóm tuổi (mô hình chính), DIR ngoài dải: ngưỡng 0.09: mọi nhóm trong dải; ngưỡng 0.5: mọi nhóm trong dải. AUC thấp nhất ở nhóm >50: 0.759 (N = 448, CI 95% [0.705, 0.808]); khoảng tin cậy chồng lấn với nhóm tham chiếu 30–50 (0.790, [0.772, 0.809]), nên chưa đủ bằng chứng về khác biệt AUC giữa hai nhóm.
3. Vai trò của SEX: thêm SEX vào mô hình chính đổi AUC Valid +0.0002 và SEX chiếm 0.17% tầm quan trọng (gain); tỷ lệ flag Nam/Nữ 1.06 (không SEX) so với 1.08 (có SEX) ở ngưỡng 0.09, 1.28 (không SEX) so với 1.32 (có SEX) ở ngưỡng 0.5. Bỏ SEX không làm giảm hiệu năng đo được, nên chính sách không dùng SEX (Charter mục 8) được giữ. Chênh lệch flag rate giữa Nam và Nữ vẫn còn khi không dùng SEX; loại SEX không triệt tiêu chênh lệch này.
4. Ở ngưỡng vận hành 0.09, mô hình chính gắn cờ 72.5% hồ sơ Valid nên flag rate của mọi nhóm đều cao. Cùng một chênh lệch tuyệt đối, DIR theo flag rate lệch gần 1 hơn DIR không bị flag; ví dụ nhóm >50: flag rate 79.0% so với 69.9% của nhóm 30–50 cho DIR = 1.13, còn tỷ lệ không bị flag 21.0% so với 30.1% cho DIR không bị flag = 0.70. DIR không bị flag (quy tắc 4/5 áp dụng cho kết quả có lợi) nằm ngoài 0.8–1.25 ở: >50 (0.70). Kết luận về công bằng ở ngưỡng này phụ thuộc vào định nghĩa DIR, cần nhóm thống nhất.
5. Logistic Scorecard (WoE) (đối chiếu), DIR ngoài dải: ngưỡng 0.09: mọi nhóm trong dải; ngưỡng 0.5: Nam (DIR = 1.34, CI [1.17, 1.54]), >50 (DIR = 1.31, CI [1.02, 1.67]). AUC thấp nhất ở nhóm >50: 0.729 (N = 448, CI 95% [0.672, 0.781]); khoảng tin cậy chồng lấn với nhóm tham chiếu 30–50 (0.772, [0.754, 0.793]), nên chưa đủ bằng chứng về khác biệt AUC giữa hai nhóm.

---

## 6. Hạn chế

- Kết quả trên Valid. Kết quả trên Test (dự đoán xuất sau lượt nghiệm thu của thành viên B): `fairness_evaluation_test.md`.
- DIR đo chênh lệch tỷ lệ gắn cờ giữa các nhóm, không hiệu chỉnh theo chênh lệch tỷ lệ vỡ nợ thực tế giữa các nhóm. Recall và tỷ lệ cảnh báo oan theo nhóm (equalized odds) không nằm trong phạm vi báo cáo.
- Nhóm >50 có 448 hồ sơ nên CI của DIR và AUC rộng; chênh lệch nằm trong CI chưa đủ bằng chứng là khác biệt thực.
- Calibrator Platt (hai tham số a, b) được fit trên chính tập Valid dùng để đánh giá.
- Mô hình được huấn luyện lại từ `configs/tuned_params.yaml` (`random_state = 42`); chữ số thập phân cuối của PD có thể lệch nhẹ so với số liệu trong `calibration_threshold.md`.

## 7. Tái lập

```bash
python scripts/analyze_fairness.py
```

Yêu cầu `data/raw/UCI_Credit_Card.csv`, `data/splits/splits.json` và `src/calibrate.py`.
