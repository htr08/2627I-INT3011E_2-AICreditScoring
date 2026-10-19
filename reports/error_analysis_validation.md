# Báo cáo Phân tích Lỗi (Error Analysis) trên tập Valid

_Tuần 3 – T2: Thành viên A (Data Engineer / Analyst)_

---

## 1. Phạm vi và thiết lập

| Hạng mục | Nội dung |
| :--- | :--- |
| Tập dữ liệu | Valid, n = 6000 (1327 hồ sơ vỡ nợ, 22.12%). Tập Test không được sử dụng. |
| Mô hình chính | LightGBM tuned + monotonic (`lightgbm_monotone`, tham số trong `configs/tuned_params.yaml`), huấn luyện trên Train, Platt scaling fit trên Valid (a = 1.0128, b = 0.0364). Đây là mô hình cuối theo `week2_review.md`. |
| Mô hình đối chiếu | Logistic Scorecard (WoE): mô hình tham chiếu minh bạch (`week2_review.md` mục 6) và là mô hình của bản phân tích lỗi ban đầu. Dùng để kiểm tra mẫu lỗi có phụ thuộc loại mô hình hay không. |
| Ngưỡng vận hành | PD ≥ 0.09: ngưỡng do thành viên B khóa trước khi đánh giá Test, làm tròn từ trung bình OOF EAD-weighted (0.088) với tỷ lệ chi phí FN:FP = 9:1 theo Charter mục 7 (`reports/test_evaluation/test_evaluation.md`). Gắn cờ 72.5% hồ sơ Valid ở mô hình chính. |
| Ngưỡng đối chiếu | PD ≥ 0.5: ngưỡng phân loại mặc định, gắn cờ 12.1% hồ sơ Valid ở mô hình chính. Do ngưỡng vận hành gắn cờ phần lớn danh mục nên FP ở đó gồm nhiều hồ sơ rủi ro thấp; ngưỡng 0.5 chỉ giữ các ca PD cao để quan sát lỗi ở vùng tự tin. |
| Định nghĩa nhóm | Theo Charter mục 7 (công cụ cảnh báo sớm cho danh mục hiện hữu). **FN**: khách hàng vỡ nợ thực tế nhưng không bị gắn cờ (bỏ sót, không kịp can thiệp). **FP**: khách hàng không vỡ nợ nhưng bị gắn cờ (cảnh báo hoặc giảm hạn mức oan). **TP**, **TN**: gắn cờ đúng khách vỡ nợ, không gắn cờ đúng khách không vỡ nợ. |
| Hồ sơ mô tả | Cột gốc của Valid sau làm sạch mã bất thường, cùng các đặc trưng dẫn xuất T2 của `src/features.py` (tính trực tiếp trên Valid, không học tham số từ dữ liệu). Giá trị là trung bình theo nhóm; `PD` là PD trung bình của mô hình tương ứng. |

Đặc trưng mô tả hồ sơ:

| Đặc trưng | Ý nghĩa |
| :--- | :--- |
| `LIMIT_BAL` | Hạn mức tín dụng (NT$) |
| `AGE` | Tuổi |
| `PAY_1` | Trạng thái trả nợ tháng gần nhất (mã -2 … 8) |
| `PAY_1_LATE` | Tỷ lệ hồ sơ trễ hạn ở tháng gần nhất (PAY_1 ≥ 1) |
| `PAY_MEAN` | Trung bình trạng thái trả nợ 6 tháng |
| `PAY_LATE_2PLUS_COUNT` | Số tháng trễ từ 2 kỳ trở lên trong 6 tháng |
| `UTIL_MEAN` | Tỷ lệ sử dụng hạn mức trung bình 6 tháng (dư nợ / hạn mức) |
| `PAY_RATIO_MEAN` | Tỷ lệ trả nợ / dư nợ trung bình (bỏ qua tháng dư nợ ≤ 0) |

---

## 2. Phân bố nhóm phân loại

| Mô hình | Ngưỡng PD | FN | FP | TP | TN | Recall | Precision | Flag rate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM monotonic + Platt | 0.09 | 102 | 3123 | 1225 | 1550 | 92.3% | 28.2% | 72.5% |
| LightGBM monotonic + Platt | 0.5 | 836 | 233 | 491 | 4440 | 37.0% | 67.8% | 12.1% |
| Logistic Scorecard (WoE) | 0.09 | 103 | 3402 | 1224 | 1271 | 92.2% | 26.5% | 77.1% |
| Logistic Scorecard (WoE) | 0.5 | 847 | 241 | 480 | 4432 | 36.2% | 66.6% | 12.0% |

Recall = TP / (TP + FN); Precision = TP / (TP + FP); Flag rate = (TP + FP) / n.

---

## 3. Hồ sơ trung bình theo nhóm: mô hình chính

### 3.1. Ngưỡng vận hành (0.09)

| Đặc trưng | FN | TP | FP | TN | Toàn Valid |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `LIMIT_BAL` | 269,510 | 119,404 | 137,168 | 259,290 | 167,339 |
| `AGE` | 36.53 | 36.31 | 35.22 | 35.57 | 35.56 |
| `PAY_1` | -0.75 | 0.75 | -0.08 | -0.47 | -0.02 |
| `PAY_1_LATE` | 0.0% | 54.4% | 21.4% | 0.4% | 22.3% |
| `PAY_MEAN` | -0.69 | 0.42 | -0.28 | -0.46 | -0.19 |
| `PAY_LATE_2PLUS_COUNT` | 0.00 | 1.94 | 0.58 | 0.00 | 0.69 |
| `UTIL_MEAN` | 0.20 | 0.48 | 0.41 | 0.25 | 0.38 |
| `PAY_RATIO_MEAN` | 0.64 | 0.25 | 0.47 | 0.45 | 0.42 |
| `PD` | 0.064 | 0.426 | 0.227 | 0.059 | 0.221 |

- **FN** (n = 102, 7.7% số hồ sơ vỡ nợ). So với TP (cùng vỡ nợ thực tế, được mô hình phát hiện), chênh lệch lớn nhất ở `PAY_LATE_2PLUS_COUNT` 0.00 so với 1.94; `PAY_1_LATE` 0.0% so với 54.4%; `PAY_1` -0.75 so với 0.75 (FN so với TP). FN gần TN hơn TP ở 7/8 đặc trưng, gần TP hơn TN ở 1/8. Hồ sơ lỗi giống nhóm TN hơn nhóm TP trên hầu hết đặc trưng.
- **FP** (n = 3123, 66.8% số hồ sơ không vỡ nợ). So với TN (cùng không vỡ nợ, không bị gắn cờ), chênh lệch lớn nhất ở `LIMIT_BAL` 137,168 so với 259,290; `PAY_1_LATE` 21.4% so với 0.4%; `UTIL_MEAN` 0.41 so với 0.25 (FP so với TN). FP gần TP hơn TN ở 2/8 đặc trưng, gần TN hơn TP ở 6/8. Hồ sơ lỗi giống nhóm TN hơn nhóm TP trên hầu hết đặc trưng.

### 3.2. Ngưỡng đối chiếu (0.5)

| Đặc trưng | FN | TP | FP | TN | Toàn Valid |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `LIMIT_BAL` | 150,120 | 98,289 | 91,227 | 182,212 | 167,339 |
| `AGE` | 36.65 | 35.76 | 34.92 | 35.36 | 35.56 |
| `PAY_1` | -0.16 | 1.99 | 1.94 | -0.32 | -0.02 |
| `PAY_1_LATE` | 22.1% | 98.2% | 95.3% | 10.2% | 22.3% |
| `PAY_MEAN` | -0.30 | 1.41 | 1.31 | -0.43 | -0.19 |
| `PAY_LATE_2PLUS_COUNT` | 0.61 | 3.79 | 3.49 | 0.22 | 0.69 |
| `UTIL_MEAN` | 0.36 | 0.62 | 0.62 | 0.34 | 0.38 |
| `PAY_RATIO_MEAN` | 0.41 | 0.07 | 0.09 | 0.48 | 0.42 |
| `PD` | 0.230 | 0.685 | 0.656 | 0.145 | 0.221 |

- **FN** (n = 836, 63.0% số hồ sơ vỡ nợ). So với TP (cùng vỡ nợ thực tế, được mô hình phát hiện), chênh lệch lớn nhất ở `PAY_LATE_2PLUS_COUNT` 0.61 so với 3.79; `PAY_1` -0.16 so với 1.99; `PAY_1_LATE` 22.1% so với 98.2% (FN so với TP). FN gần TN hơn TP ở 7/8 đặc trưng, gần TP hơn TN ở 1/8. Hồ sơ lỗi giống nhóm TN hơn nhóm TP trên hầu hết đặc trưng.
- **FP** (n = 233, 5.0% số hồ sơ không vỡ nợ). So với TN (cùng không vỡ nợ, không bị gắn cờ), chênh lệch lớn nhất ở `PAY_LATE_2PLUS_COUNT` 3.49 so với 0.22; `PAY_1_LATE` 95.3% so với 10.2%; `PAY_1` 1.94 so với -0.32 (FP so với TN). FP gần TP hơn TN ở 7/8 đặc trưng, gần TN hơn TP ở 1/8. Hồ sơ lỗi giống nhóm TP hơn nhóm TN trên hầu hết đặc trưng.

---

## 4. Đối chiếu với Logistic Scorecard

### 4.1. Ngưỡng vận hành (0.09)

| Đặc trưng | FN | TP | FP | TN | Toàn Valid |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `LIMIT_BAL` | 302,039 | 116,544 | 136,413 | 288,120 | 167,339 |
| `AGE` | 37.09 | 36.26 | 35.05 | 36.11 | 35.56 |
| `PAY_1` | -0.73 | 0.75 | -0.08 | -0.55 | -0.02 |
| `PAY_1_LATE` | 0.0% | 54.5% | 19.8% | 0.0% | 22.3% |
| `PAY_MEAN` | -0.73 | 0.42 | -0.27 | -0.54 | -0.19 |
| `PAY_LATE_2PLUS_COUNT` | 0.00 | 1.94 | 0.53 | 0.00 | 0.69 |
| `UTIL_MEAN` | 0.26 | 0.47 | 0.38 | 0.28 | 0.38 |
| `PAY_RATIO_MEAN` | 0.72 | 0.24 | 0.44 | 0.53 | 0.42 |
| `PD` | 0.073 | 0.406 | 0.207 | 0.072 | 0.217 |

- **FN** (n = 103, 7.8% số hồ sơ vỡ nợ). So với TP (cùng vỡ nợ thực tế, được mô hình phát hiện), chênh lệch lớn nhất ở `LIMIT_BAL` 302,039 so với 116,544; `PAY_LATE_2PLUS_COUNT` 0.00 so với 1.94; `PAY_1_LATE` 0.0% so với 54.5% (FN so với TP). FN gần TN hơn TP ở 7/8 đặc trưng, gần TP hơn TN ở 1/8. Hồ sơ lỗi giống nhóm TN hơn nhóm TP trên hầu hết đặc trưng.
- **FP** (n = 3402, 72.8% số hồ sơ không vỡ nợ). So với TN (cùng không vỡ nợ, không bị gắn cờ), chênh lệch lớn nhất ở `LIMIT_BAL` 136,413 so với 288,120; `PAY_1_LATE` 19.8% so với 0.0%; `PAY_1` -0.08 so với -0.55 (FP so với TN). FP gần TP hơn TN ở 2/8 đặc trưng, gần TN hơn TP ở 6/8. Hồ sơ lỗi giống nhóm TN hơn nhóm TP trên hầu hết đặc trưng.

### 4.2. Ngưỡng đối chiếu (0.5)

| Đặc trưng | FN | TP | FP | TN | Toàn Valid |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `LIMIT_BAL` | 154,935 | 88,604 | 84,589 | 182,737 | 167,339 |
| `AGE` | 36.58 | 35.86 | 35.46 | 35.33 | 35.56 |
| `PAY_1` | -0.15 | 2.03 | 2.05 | -0.33 | -0.02 |
| `PAY_1_LATE` | 22.1% | 100.0% | 100.0% | 9.7% | 22.3% |
| `PAY_MEAN` | -0.28 | 1.41 | 1.23 | -0.43 | -0.19 |
| `PAY_LATE_2PLUS_COUNT` | 0.64 | 3.81 | 3.33 | 0.22 | 0.69 |
| `UTIL_MEAN` | 0.36 | 0.62 | 0.63 | 0.34 | 0.38 |
| `PAY_RATIO_MEAN` | 0.40 | 0.09 | 0.12 | 0.48 | 0.42 |
| `PD` | 0.215 | 0.673 | 0.648 | 0.144 | 0.217 |

- **FN** (n = 847, 63.8% số hồ sơ vỡ nợ). So với TP (cùng vỡ nợ thực tế, được mô hình phát hiện), chênh lệch lớn nhất ở `PAY_LATE_2PLUS_COUNT` 0.64 so với 3.81; `PAY_1` -0.15 so với 2.03; `PAY_1_LATE` 22.1% so với 100.0% (FN so với TP). FN gần TN hơn TP ở 7/8 đặc trưng, gần TP hơn TN ở 1/8. Hồ sơ lỗi giống nhóm TN hơn nhóm TP trên hầu hết đặc trưng.
- **FP** (n = 241, 5.2% số hồ sơ không vỡ nợ). So với TN (cùng không vỡ nợ, không bị gắn cờ), chênh lệch lớn nhất ở `PAY_1_LATE` 100.0% so với 9.7%; `PAY_LATE_2PLUS_COUNT` 3.33 so với 0.22; `PAY_1` 2.05 so với -0.33 (FP so với TN). FP gần TP hơn TN ở 7/8 đặc trưng, gần TN hơn TP ở 1/8. Hồ sơ lỗi giống nhóm TP hơn nhóm TN trên hầu hết đặc trưng.

### 4.3. Mức trùng lặp lỗi giữa hai mô hình

| Ngưỡng PD | Nhóm | Mô hình chính | Scorecard | Thuộc nhóm ở cả hai mô hình | Tỷ lệ so với mô hình chính |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 0.09 | FN | 102 | 103 | 70 | 68.6% |
| 0.09 | FP | 3123 | 3402 | 2904 | 93.0% |
| 0.5 | FN | 836 | 847 | 808 | 96.7% |
| 0.5 | FP | 233 | 241 | 210 | 90.1% |

---

## 5. Kết luận

1. Ở ngưỡng vận hành 0.09, mô hình chính bỏ sót 102 hồ sơ vỡ nợ (7.7% số hồ sơ vỡ nợ) và gắn cờ oan 3123 hồ sơ (66.8% số hồ sơ không vỡ nợ). FN gần TN hơn TP ở 7/8 đặc trưng: hồ sơ vỡ nợ bị bỏ sót có đặc điểm giống khách hàng không vỡ nợ trên hầu hết đặc trưng, nên các đặc trưng này không tách được chúng khỏi nhóm TN.
2. Ở ngưỡng đối chiếu 0.5, FP của mô hình chính gần TP hơn TN ở 7/8 đặc trưng (ngưỡng 0.09: 2/8): khách hàng không vỡ nợ bị gắn cờ ở PD cao có đặc điểm giống nhóm vỡ nợ, nên các đặc trưng này không tách được họ khỏi nhóm TP.
3. Với Scorecard, FN gần TN hơn TP ở 7/8 đặc trưng (ngưỡng 0.09) và FP gần TP hơn TN ở 7/8 (ngưỡng 0.5). Số FN của mô hình chính cũng là FN của Scorecard: 70/102 (68.6%) ở ngưỡng 0.09, 808/836 (96.7%) ở ngưỡng 0.5.

---

## 6. Hạn chế

- Calibrator Platt (hai tham số a, b) được fit trên chính tập Valid dùng để phân tích, nên số liệu không phải đánh giá ngoài mẫu hoàn toàn. Kết quả chưa cập nhật theo tập Test.
- Mô tả theo giá trị trung bình; `LIMIT_BAL`, `UTIL_MEAN`, `PAY_RATIO_MEAN` có phân phối lệch phải nên trung bình chịu ảnh hưởng của đuôi. Khoảng cách giữa các nhóm cho biết xu hướng, không kèm kiểm định thống kê.
- "Nhóm gần nhất" xét riêng từng đặc trưng, không phản ánh khoảng cách đa chiều giữa các hồ sơ. Các đặc trưng về trễ hạn (`PAY_1`, `PAY_1_LATE`, `PAY_MEAN`, `PAY_LATE_2PLUS_COUNT`) tương quan cao nên số đặc trưng gần một nhóm không độc lập với nhau.
- Mô hình được huấn luyện lại từ `configs/tuned_params.yaml` (`random_state = 42`); chữ số thập phân cuối của PD có thể lệch nhẹ so với số liệu trong `calibration_threshold.md`.

## 7. Tái lập

```bash
python scripts/analyze_errors.py
```

Yêu cầu `data/raw/UCI_Credit_Card.csv`, `data/splits/splits.json` và `src/calibrate.py`.
