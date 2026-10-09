# Logistic Scorecard (WoE)

_Tuần 2 – T5: Thành viên A (Data Engineer / Analyst)_

---

## 1. Mục tiêu

Xây dựng Logistic Scorecard truyền thống dựa trên Weight of Evidence (WoE) làm cơ sở so sánh với các mô hình dạng black-box (project_plan mục 3.3, phân công Tuần 2 – T5). Sản phẩm gồm mô hình đánh giá bằng 5-fold Stratified CV, đăng ký trên MLflow, và bảng điểm theo thang PDO dùng chung với demo.

---

## 2. Thiết kế

Pipeline `build_scorecard_pipeline()` (`src/pipelines.py`), mọi bước học tham số được fit lại trong từng fold:

1. Làm sạch mã bất thường, đổi `PAY_0` → `PAY_1`; tạo biến dẫn xuất T2; `AGE` → `AGE_BIN` theo IV; bỏ `SEX`, `ID`.
2. **WoE + lọc IV với coarse classing** (`WoEIVTransformer`, mục 2.2): biến liên tục chia 20 bin theo phân vị rồi gộp bin; biến ≤ 12 giá trị (PAY_x, EDUCATION, cờ 0/1) bắt đầu từ mỗi giá trị một bin rồi gộp bin thưa. Biên ngoài mở rộng ±∞, NaN vào bin riêng. Giữ biến có IV ≥ 0.02.
3. **Logistic Regression ràng buộc dấu** (`SignConstrainedLogisticRegression`): với WoE = ln(%good / %bad), mọi hệ số phải âm. Mô hình loại dần biến có hệ số dương lớn nhất rồi fit lại đến khi mọi hệ số cùng dấu.

### 2.1. Lý do ràng buộc dấu

Khi fit Logistic Regression thông thường trên 42 biến WoE, **17/42 biến có hệ số dương**, gồm `PAY_2`–`PAY_6`, `PAY_MAX`, `PAY_LATE_CONSECUTIVE`, `UTIL_1`, `UTIL_3`, `UTIL_5` và các cờ `MIN_PAY_FLAG_*`. Nguyên nhân là đa cộng tuyến giữa các biến trễ hạn và giữa các biến sử dụng hạn mức: mô hình bù trừ hệ số giữa các biến gần trùng nhau. Hệ quả trên bảng điểm là thuộc tính rủi ro cao hơn nhận nhiều điểm hơn (vd `PAY_MAX` ở mức trễ cao được cộng điểm), trái với logic nghiệp vụ và làm mất giá trị giải thích của scorecard.

| Cấu hình | ROC-AUC (5-fold) | KS | Số biến trong mô hình | Hệ số sai dấu |
|---|---|---|---|---|
| Logistic Regression thường, 5 bin (cấu hình T3) | 0.7809 ± 0.0071 | 0.4392 | 42 | 17 |
| Ràng buộc dấu, 5 bin | 0.7789 ± 0.0078 | 0.4355 | 23.4 (trung bình fold) | 0 |
| **Ràng buộc dấu + coarse classing (chính thức)** | **0.7790 ± 0.0083** | 0.4326 | 22.8 (trung bình fold), 24 trên toàn Train | 0 |

Ràng buộc dấu làm AUC giảm 0.0020 và giảm số biến gần một nửa, đổi lại toàn bộ hệ số nhất quán với nghiệp vụ. Với vai trò mô hình giải thích được để đối chiếu, đây là đánh đổi phù hợp.

### 2.2. Coarse classing

Ràng buộc dấu bảo đảm chiều của từng biến, nhưng chưa bảo đảm điểm thay đổi hợp lý giữa các bin của cùng một biến. Với 5 bin phân vị và mỗi giá trị rời rạc một bin, bảng điểm có hai vấn đề:

- **Bin quá thưa:** 29 bin của biến rời rạc có dưới 5% mẫu, nhiều bin chỉ 5–20 hồ sơ (vd `PAY_1` = 5, 6, 7, 8). Điểm của các bin này chủ yếu phản ánh nhiễu: trễ 2 tháng bị −34.4 điểm nhưng trễ 5 tháng chỉ −18.4 và trễ 8 tháng −20.8.
- **WoE không đơn điệu:** 9/24 biến liên tục (`UTIL_MEAN`, `PAY_MEAN`, `PAY_SLOPE`, `PAY_RATIO_*`, ...) có WoE lên xuống giữa các bin liền kề, nên điểm đi zig-zag theo giá trị biến.

`WoEIVTransformer(min_bin_share, monotonic)` thực hiện coarse classing trong từng fold:

1. **Gộp bin thưa:** bin nhỏ nhất có dưới 5% mẫu được gộp với bin kề có tỷ lệ bad gần hơn, lặp đến khi mọi bin đạt ≥ 5%.
2. **WoE đơn điệu cho biến liên tục:** gộp cặp bin kề vi phạm thứ tự tỷ lệ bad, theo chiều tương quan Spearman của biến với nhãn, đến khi đơn điệu. Biến rời rạc không bị ép đơn điệu: với `PAY_x`, các mã −2 (không phát sinh dư nợ), −1 (trả đủ), 0 (trả tối thiểu) là ba trạng thái khác nhau, có tỷ lệ vỡ nợ 12.5% / 16.5% / 12.6% trên Train. Khi ép đơn điệu cả biến rời rạc, AUC giảm 0.002–0.005 trên cả 5 fold.
3. **Giá trị chưa gặp:** biến rời rạc được chia theo khoảng giữa các giá trị, nên giá trị không có trong dữ liệu fit (vd `PAY_1` = 9) rơi vào bin gần nhất. Trước đây các giá trị này nhận WoE = 0, tức điểm trung tính.

| Cấu hình bin (đều có ràng buộc dấu) | ROC-AUC | KS | Bin < 5% mẫu | Biến liên tục có WoE không đơn điệu |
|---|---|---|---|---|
| 5 bin phân vị, mỗi giá trị rời rạc một bin | 0.7789 ± 0.0078 | 0.4355 | 29 | 9 |
| Bin ≥ 5% | 0.7780 ± 0.0073 | 0.4314 | 0 | – |
| **20 bin → bin ≥ 5% + WoE đơn điệu (chính thức)** | **0.7790 ± 0.0083** | 0.4326 | 0 | 0 |

Coarse classing giữ nguyên AUC (+0.0001) và loại bỏ toàn bộ bin thưa cùng các điểm đảo chiều. Báo cáo IV của `--mode woe` vẫn dùng cấu hình 5 bin để tái lập đúng `configs/feature_freeze_v1.yaml`.

---

## 3. Kết quả

### 3.1. Hiệu năng (5-fold Stratified CV trên Train)

| Mô hình | ROC-AUC | Gini | KS | PR-AUC | Brier |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.7745 ± 0.0116 | 0.5490 | 0.4282 | – | 0.1354 |
| **Logistic Scorecard (WoE)** | **0.7790 ± 0.0083** | 0.5580 ± 0.0165 | 0.4326 ± 0.0224 | 0.5432 | 0.1351 |
| LightGBM tuned + monotonic | 0.7906 ± 0.0074 | 0.5813 | 0.4453 | 0.5624 | 0.1328 |
| CatBoost tuned | 0.7911 ± 0.0073 | 0.5822 | 0.4452 | 0.5675 | 0.1327 |

Scorecard vượt baseline Logistic Regression (+0.0045 AUC) và thấp hơn các mô hình boosting đã tuning khoảng 0.012 AUC. Đây là chi phí hiệu năng của việc rời rạc hoá biến liên tục và giới hạn ở quan hệ cộng tính trên thang log-odds.

### 3.2. Bảng điểm

Thang điểm dùng chung với `src/scoring.py` (project_plan mục 3.5d): 600 điểm tại odds 50:1, PDO = 20, Factor = 28.85.

Score = Offset − Factor × logit(PD) = **điểm nền** + Σ **điểm thuộc tính**, với điểm nền = Offset − Factor × a và điểm thuộc tính = −Factor × b_j × WoE_ij. Tổng điểm của một hồ sơ (chưa clip) bằng `pd_to_score(PD)` của chính scorecard (kiểm thử tại `tests/test_scoring.py`).

- **Điểm nền:** 523.2.
- **24 biến, 156 thuộc tính** trong scorecard cuối (fit trên toàn bộ Train). Mọi bin có ≥ 5% mẫu; điểm của mọi biến liên tục thay đổi đơn điệu theo giá trị. Biên độ điểm (chênh lệch giữa thuộc tính cao nhất và thấp nhất) của các biến chính:

| Biến | Số bin | Biên độ điểm | Điểm thấp nhất → cao nhất |
|---|---|---|---|
| `PAY_1` | 5 | 49.4 | −37.3 → +12.2 |
| `PAY_AMT2` | 8 | 12.0 | −3.7 → +8.2 |
| `PAY_LATE_CONSECUTIVE` | 3 | 10.8 | −7.5 → +3.3 |
| `BILL_STD` | 10 | 10.1 | −5.5 → +4.6 |
| `UTIL_2` | 6 | 8.3 | −5.5 → +2.7 |
| `PAY_MAX` | 5 | 7.8 | −4.4 → +3.5 |
| `PAY_AMT3` | 8 | 7.1 | −2.3 → +4.8 |

- **Ví dụ `PAY_1`:** không phát sinh dư nợ (−2) +12.2 điểm, trả đủ (−1) +6.4, trả tối thiểu (0) +11.9, trễ 1 tháng −10.8, trễ từ 2 tháng trở lên −37.3. Các mức trễ ≥ 3 tháng (1.5% mẫu Train) được gộp vào bin "≥ 2 tháng".
- **Ví dụ `UTIL_2`:** điểm giảm dần theo tỷ lệ sử dụng hạn mức, từ +2.7 (≤ 29%) xuống −5.5 (> 101%).

Bảng điểm đầy đủ được log tại artifact `scorecard_points.csv` của run `logistic_scorecard` trên MLflow.

---

## 4. Tái lập

```
python scripts/run_pipeline.py --mode scorecard                 # CV + MLflow + bảng điểm; đăng ký logistic_scorecard@scorecard
python scripts/run_pipeline.py --mode scorecard --include-sex   # bản đối chiếu fairness, không đăng ký
```

---

## 5. Hạn chế

- Coarse classing dùng thủ tục gộp tham lam (ngưỡng 5% mẫu, đơn điệu theo chiều Spearman), không tối ưu IV toàn cục; biến có quan hệ hình chữ U thật sẽ bị gộp thành quan hệ đơn điệu.
- Biến rời rạc `PAY_x` giữ quan hệ không đơn điệu ở vùng −2 … 0 theo dữ liệu; điểm của các mã này khác nhau ít (6–12 điểm với `PAY_1`).
- Ràng buộc dấu dùng thủ tục tham lam (loại một biến mỗi vòng), không bảo đảm tập biến tối ưu toàn cục.
