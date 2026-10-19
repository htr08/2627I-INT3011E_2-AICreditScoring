# Tuần 3 – T3: Đánh giá cuối trên tập Test

Test được đánh giá một lần. Mô hình, calibrator và ngưỡng đã khóa trước khi đánh giá: LightGBM tuned + monotonic, Platt fit trên Valid, ngưỡng 0.09 (làm tròn từ trung bình OOF EAD-weighted 0.088). Không điều chỉnh tham số theo kết quả Test.

Tỷ lệ default thực tế trên Test: **22.12%** (6,000 hồ sơ).

## Chỉ số phân tách và hiệu chuẩn

| Mô hình/đầu ra | ROC-AUC (bootstrap 95% CI) | Gini | KS (bootstrap 95% CI) | PR-AUC (bootstrap 95% CI) | Brier (bootstrap 95% CI) | Log Loss | ECE | Mean PD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM tuned + Monotonic (raw) | 0.7825 (0.7681, 0.7973) | 0.5649 | 0.4290 (0.4048, 0.4603) | 0.5600 (0.5315, 0.5887) | 0.1347 (0.1288, 0.1405) | 0.4284 | 0.0139 | 22.07% |
| LightGBM tuned + Monotonic (Platt) | 0.7825 (0.7681, 0.7973) | 0.5649 | 0.4290 (0.4048, 0.4603) | 0.5600 (0.5315, 0.5887) | 0.1348 (0.1288, 0.1405) | 0.4285 | 0.0163 | 22.37% |
| Logistic Regression baseline | 0.7609 (0.7461, 0.7765) | 0.5219 | 0.4046 (0.3771, 0.4341) | 0.5236 (0.4960, 0.5546) | 0.1384 (0.1322, 0.1446) | 0.4410 | 0.0185 | 22.01% |

## So sánh cuối với baseline Logistic Regression

- DeLong paired test: ΔAUC (LightGBM Platt − baseline) = **0.02150**, 95% CI [0.01449, 0.02852], p = **1.8616e-09**.
- Paired bootstrap 95% CI cho ΔAUC: [0.01434, 0.02857].
- Kết luận mức 5%: **có khác biệt có ý nghĩa**.

## Kết quả tại ngưỡng đã khóa trên Valid

Ngưỡng PD = **0.09**; cutoff PDO = **553.9** điểm.

| Chỉ số | Ngưỡng 0.09 | Ngưỡng 0.50 |
|---|---:|---:|
| Precision | 28.02% | 65.63% |
| Recall | 92.46% | 36.70% |
| Flag rate | 72.98% | 12.37% |
| Unit expected cost/customer | 0.03377 | 0.06513 |
| EAD-weighted cost (NT$/customer) | 1634.49 | 2830.29 |

## Đồ thị

![ROC, PR và calibration curves](test_curves.png)

Chi tiết số liệu và khoảng tin cậy: `test_metrics.csv` và `test_comparison.csv`. Kết quả Test chỉ dùng nghiệm thu, không dùng đổi calibrator, ngưỡng hoặc mô hình.
