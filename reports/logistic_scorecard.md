# Logistic Scorecard (WoE)

_Tuần 2 – T5: Thành viên A (Data Engineer / Analyst)_

---

## 1. Mục tiêu

Xây dựng Logistic Scorecard truyền thống dựa trên Weight of Evidence (WoE) làm cơ sở so sánh với các mô hình dạng black-box (project_plan mục 3.3, phân công Tuần 2 – T5). Sản phẩm gồm mô hình đánh giá bằng 5-fold Stratified CV, đăng ký trên MLflow, và bảng điểm theo thang PDO dùng chung với demo.

---

## 2. Thiết kế

Pipeline `build_scorecard_pipeline()` (`src/pipelines.py`), mọi bước học tham số được fit lại trong từng fold:

1. Làm sạch mã bất thường, đổi `PAY_0` → `PAY_1`; tạo biến dẫn xuất T2; `AGE` → `AGE_BIN` theo IV; bỏ `SEX`, `ID`.
2. **WoE + lọc IV** (`WoEIVTransformer`): biến liên tục chia 5 bin theo phân vị (biên ngoài mở rộng ±∞, NaN vào bin riêng); biến ≤ 12 giá trị (PAY_x, EDUCATION, cờ 0/1) mỗi giá trị một bin. Giữ biến có IV ≥ 0.02.
3. **Logistic Regression ràng buộc dấu** (`SignConstrainedLogisticRegression`): với WoE = ln(%good / %bad), mọi hệ số phải âm. Mô hình loại dần biến có hệ số dương lớn nhất rồi fit lại đến khi mọi hệ số cùng dấu.

### 2.1. Lý do ràng buộc dấu

Khi fit Logistic Regression thông thường trên 42 biến WoE, **17/42 biến có hệ số dương**, gồm `PAY_2`–`PAY_6`, `PAY_MAX`, `PAY_LATE_CONSECUTIVE`, `UTIL_1`, `UTIL_3`, `UTIL_5` và các cờ `MIN_PAY_FLAG_*`. Nguyên nhân là đa cộng tuyến giữa các biến trễ hạn và giữa các biến sử dụng hạn mức: mô hình bù trừ hệ số giữa các biến gần trùng nhau. Hệ quả trên bảng điểm là thuộc tính rủi ro cao hơn nhận nhiều điểm hơn (vd `PAY_MAX` ở mức trễ cao được cộng điểm), trái với logic nghiệp vụ và làm mất giá trị giải thích của scorecard.

| Cấu hình | ROC-AUC (5-fold) | KS | Số biến trong mô hình | Hệ số sai dấu |
|---|---|---|---|---|
| Logistic Regression thường (cấu hình T3) | 0.7809 ± 0.0071 | 0.4392 | 42 | 17 |
| **Ràng buộc dấu (chính thức)** | **0.7789 ± 0.0078** | 0.4355 | 21–27 (24 trên toàn Train) | 0 |

Ràng buộc dấu làm AUC giảm 0.0020 và giảm số biến gần một nửa, đổi lại toàn bộ bảng điểm nhất quán với nghiệp vụ. Với vai trò mô hình giải thích được để đối chiếu, đây là đánh đổi phù hợp.

---

## 3. Kết quả

### 3.1. Hiệu năng (5-fold Stratified CV trên Train)

| Mô hình | ROC-AUC | Gini | KS | PR-AUC | Brier |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.7745 ± 0.0116 | 0.5490 | 0.4282 | – | 0.1354 |
| **Logistic Scorecard (WoE)** | **0.7789 ± 0.0078** | 0.5578 ± 0.0156 | 0.4355 ± 0.0240 | 0.5453 | 0.1351 |
| LightGBM tuned + monotonic | 0.7906 ± 0.0074 | 0.5813 | 0.4453 | 0.5624 | 0.1328 |
| CatBoost tuned | 0.7911 ± 0.0073 | 0.5822 | 0.4452 | 0.5675 | 0.1327 |

Scorecard vượt baseline Logistic Regression (+0.0044 AUC) và thấp hơn các mô hình boosting đã tuning khoảng 0.012 AUC. Đây là chi phí hiệu năng của việc rời rạc hoá biến liên tục và giới hạn ở quan hệ cộng tính trên thang log-odds.

### 3.2. Bảng điểm

Thang điểm dùng chung với `src/scoring.py` (project_plan mục 3.5d): 600 điểm tại odds 50:1, PDO = 20, Factor = 28.85.

Score = Offset − Factor × logit(PD) = **điểm nền** + Σ **điểm thuộc tính**, với điểm nền = Offset − Factor × a và điểm thuộc tính = −Factor × b_j × WoE_ij. Tổng điểm của một hồ sơ (chưa clip) bằng `pd_to_score(PD)` của chính scorecard (kiểm thử tại `tests/test_scoring.py`).

- **Điểm nền:** 523.2.
- **24 biến** trong scorecard cuối (fit trên toàn bộ Train). Biên độ điểm (chênh lệch giữa thuộc tính cao nhất và thấp nhất) của các biến chính:

| Biến | Biên độ điểm | Điểm thấp nhất → cao nhất |
|---|---|---|
| `PAY_1` | 56.4 | −45.0 → +11.4 |
| `PAY_LATE_2PLUS_COUNT` | 27.0 | −21.1 → +5.9 |
| `EDUCATION` | 14.7 | −1.7 → +13.0 |
| `BILL_DELTA` | 10.7 | −5.1 → +5.6 |
| `PAY_MAX` | 9.3 | −7.1 → +2.2 |
| `PAY_AMT2` | 8.2 | −3.7 → +4.5 |
| `UTIL_2` | 8.1 | −3.5 → +4.6 |

- **Ví dụ `PAY_1`:** trả đúng hạn hoặc không phát sinh dư nợ (−2, −1, 0) nhận +6.0 đến +11.4 điểm; trễ 1 tháng −10.1; trễ 2 tháng −34.4; trễ 3 tháng −42.3. Các mức trễ ≥ 4 tháng có ít mẫu (< 50 hồ sơ mỗi mức), nên điểm dao động (−18.4 đến −45.0) chủ yếu do nhiễu thống kê; gộp các mức này thành một bin là hướng cải thiện nếu bảng điểm được dùng cho hiển thị.

Bảng điểm đầy đủ được log tại artifact `scorecard_points.csv` của run `logistic_scorecard` trên MLflow.

---

## 4. Tái lập

```
python scripts/run_pipeline.py --mode scorecard                 # CV + MLflow + bảng điểm; đăng ký logistic_scorecard@scorecard
python scripts/run_pipeline.py --mode scorecard --include-sex   # bản đối chiếu fairness, không đăng ký
```

---

## 5. Hạn chế

- Bin của biến liên tục được chia theo phân vị cố định (5 bin), chưa tối ưu theo IV hay ràng buộc đơn điệu của WoE; một số biến có WoE không đơn điệu theo giá trị.
- Các mức PAY_x hiếm (≥ 4 tháng trễ) chưa được gộp bin, nên điểm của các mức này kém ổn định.
- Ràng buộc dấu dùng thủ tục tham lam (loại một biến mỗi vòng), không bảo đảm tập biến tối ưu toàn cục.
