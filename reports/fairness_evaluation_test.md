# Báo cáo Đánh giá Tính Công bằng (Fairness Check) trên tập Test

_Tuần 3 – T3: Thành viên A (Data Engineer / Analyst)_

---

## 1. Phạm vi và thiết lập

| Hạng mục | Nội dung |
| :--- | :--- |
| Tập dữ liệu | Test, n = 6000 (1327 hồ sơ vỡ nợ, 22.12%). Dự đoán lấy từ `reports/test_evaluation/test_predictions.csv`, xuất sau lượt đánh giá Test; báo cáo này không chấm điểm lại Test. |
| Mô hình chính | LightGBM tuned + monotonic, Platt fit trên Valid, không dùng SEX. |
| Mô hình đối chiếu | Logistic Regression baseline, mô hình đối chiếu của lượt đánh giá Test (không qua Platt). |
| Ngưỡng | PD ≥ 0.09 (ngưỡng vận hành đã khóa, mô hình chính gắn cờ 72.9% hồ sơ Test) và PD ≥ 0.5 (đối chiếu, gắn cờ 12.4%). |
| Chỉ số | Flag rate, DIR (so với nhóm đông nhất của từng thuộc tính, dải [0.8, 1.25]), DIR không bị flag (chỉ số bổ sung, dấu * là ngoài dải), AUC theo nhóm; CI 95% bootstrap 1000 lần. Định nghĩa đầy đủ: `fairness_evaluation.md` mục 1. |

---

## 2. Kết quả mô hình chính (không dùng SEX)

### 2.1. Ngưỡng vận hành (0.09)

| Thuộc tính | Nhóm | N | Vỡ nợ thực tế | Flag rate | DIR [95% CI] | Trạng thái | DIR không bị flag | AUC [95% CI] |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| Giới tính | Nam | 2402 | 23.4% | 73.2% | 1.01 [0.98, 1.04] | Đạt | 0.98 | 0.775 [0.752, 0.798] |
| Giới tính | Nữ | 3598 | 21.3% | 72.7% | 1.00 (tham chiếu) | – | – | 0.787 [0.768, 0.805] |
| Nhóm tuổi | <30 | 1912 | 22.3% | 75.8% | 1.08 [1.04, 1.12] | Đạt | 0.82 | 0.787 [0.761, 0.812] |
| Nhóm tuổi | 30–50 | 3617 | 21.7% | 70.3% | 1.00 (tham chiếu) | – | – | 0.779 [0.760, 0.797] |
| Nhóm tuổi | >50 | 471 | 24.4% | 81.1% | 1.15 [1.09, 1.21] | Đạt | 0.64 * | 0.788 [0.739, 0.835] |

### 2.2. Ngưỡng đối chiếu (0.5)

| Thuộc tính | Nhóm | N | Vỡ nợ thực tế | Flag rate | DIR [95% CI] | Trạng thái | DIR không bị flag | AUC [95% CI] |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| Giới tính | Nam | 2402 | 23.4% | 12.8% | 1.05 [0.92, 1.20] | Đạt | 0.99 | 0.775 [0.752, 0.798] |
| Giới tính | Nữ | 3598 | 21.3% | 12.1% | 1.00 (tham chiếu) | – | – | 0.787 [0.768, 0.805] |
| Nhóm tuổi | <30 | 1912 | 22.3% | 14.6% | 1.33 [1.16, 1.54] | Vượt ngưỡng | 0.96 | 0.787 [0.761, 0.812] |
| Nhóm tuổi | 30–50 | 3617 | 21.7% | 11.0% | 1.00 (tham chiếu) | – | – | 0.779 [0.760, 0.797] |
| Nhóm tuổi | >50 | 471 | 24.4% | 13.6% | 1.23 [0.94, 1.57] | Đạt | 0.97 | 0.788 [0.739, 0.835] |

---

## 3. Đối chiếu với Logistic Regression baseline

### 3.1. Ngưỡng vận hành (0.09)

| Thuộc tính | Nhóm | N | Vỡ nợ thực tế | Flag rate | DIR [95% CI] | Trạng thái | DIR không bị flag | AUC [95% CI] |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| Giới tính | Nam | 2402 | 23.4% | 82.7% | 0.99 [0.97, 1.02] | Đạt | 1.03 | 0.762 [0.737, 0.785] |
| Giới tính | Nữ | 3598 | 21.3% | 83.2% | 1.00 (tham chiếu) | – | – | 0.760 [0.739, 0.780] |
| Nhóm tuổi | <30 | 1912 | 22.3% | 84.0% | 1.03 [1.00, 1.06] | Đạt | 0.87 | 0.775 [0.747, 0.799] |
| Nhóm tuổi | 30–50 | 3617 | 21.7% | 81.7% | 1.00 (tham chiếu) | – | – | 0.750 [0.729, 0.770] |
| Nhóm tuổi | >50 | 471 | 24.4% | 88.7% | 1.09 [1.05, 1.13] | Đạt | 0.61 * | 0.790 [0.739, 0.838] |

### 3.2. Ngưỡng đối chiếu (0.5)

| Thuộc tính | Nhóm | N | Vỡ nợ thực tế | Flag rate | DIR [95% CI] | Trạng thái | DIR không bị flag | AUC [95% CI] |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| Giới tính | Nam | 2402 | 23.4% | 12.2% | 1.07 [0.92, 1.22] | Đạt | 0.99 | 0.762 [0.737, 0.785] |
| Giới tính | Nữ | 3598 | 21.3% | 11.4% | 1.00 (tham chiếu) | – | – | 0.760 [0.739, 0.780] |
| Nhóm tuổi | <30 | 1912 | 22.3% | 13.9% | 1.32 [1.14, 1.52] | Vượt ngưỡng | 0.96 | 0.775 [0.747, 0.799] |
| Nhóm tuổi | 30–50 | 3617 | 21.7% | 10.5% | 1.00 (tham chiếu) | – | – | 0.750 [0.729, 0.770] |
| Nhóm tuổi | >50 | 471 | 24.4% | 12.3% | 1.17 [0.90, 1.50] | Đạt | 0.98 | 0.790 [0.739, 0.838] |

---

## 4. Kết luận

1. Giới tính (mô hình chính, không dùng SEX), flag rate: ngưỡng 0.09: Nam 73.2% so với Nữ 72.7% (DIR Nam/Nữ = 1.01, CI 95% [0.98, 1.04], trong dải 0.8–1.25); ngưỡng 0.5: Nam 12.8% so với Nữ 12.1% (DIR Nam/Nữ = 1.05, CI 95% [0.92, 1.20], trong dải 0.8–1.25). Tỷ lệ vỡ nợ thực tế trên Test: Nam 23.4%, Nữ 21.3%. Nhóm bị flag nhiều hơn (Nam) cũng là nhóm có tỷ lệ vỡ nợ thực tế cao hơn.
2. Nhóm tuổi (mô hình chính), DIR ngoài dải: ngưỡng 0.09: mọi nhóm trong dải; ngưỡng 0.5: <30 (DIR = 1.33, CI [1.16, 1.54]). AUC thấp nhất ở nhóm 30–50: 0.779 (N = 3617, CI 95% [0.760, 0.797]).
3. Ở ngưỡng vận hành 0.09, mô hình chính gắn cờ 72.9% hồ sơ Test nên flag rate của mọi nhóm đều cao. Cùng một chênh lệch tuyệt đối, DIR theo flag rate lệch gần 1 hơn DIR không bị flag; ví dụ nhóm >50: flag rate 81.1% so với 70.3% của nhóm 30–50 cho DIR = 1.15, còn tỷ lệ không bị flag 18.9% so với 29.7% cho DIR không bị flag = 0.64. DIR không bị flag (quy tắc 4/5 áp dụng cho kết quả có lợi) nằm ngoài 0.8–1.25 ở: >50 (0.64). Chỉ số theo Charter là DIR theo flag rate; DIR không bị flag là chỉ số bổ sung, chỉ để tham chiếu.
4. Logistic Regression baseline (đối chiếu), DIR ngoài dải: ngưỡng 0.09: mọi nhóm trong dải; ngưỡng 0.5: <30 (DIR = 1.32, CI [1.14, 1.52]). AUC thấp nhất ở nhóm 30–50: 0.750 (N = 3617, CI 95% [0.729, 0.770]).

---

## 5. Hạn chế

- Test chỉ có dự đoán của mô hình chính và baseline; bản có SEX và Scorecard không được chấm trên Test, nên tác động của SEX được đánh giá trên Valid (`fairness_evaluation.md` mục 2).
- Mô hình được huấn luyện lại để xuất dự đoán; flag rate ở ngưỡng 0.09 lệch 5 hồ sơ so với báo cáo nghiệm thu (`test_evaluation/test_evaluation.md`).
- DIR đo chênh lệch tỷ lệ gắn cờ, không hiệu chỉnh theo chênh lệch tỷ lệ vỡ nợ thực tế giữa các nhóm; nhóm >50 có 471 hồ sơ nên CI rộng.

## 6. Tái lập

```bash
python scripts/analyze_fairness.py --split test
```

Yêu cầu `reports/test_evaluation/test_predictions.csv` (không commit) và `data/raw/UCI_Credit_Card.csv`.
