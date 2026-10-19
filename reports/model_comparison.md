# Báo cáo Tổng hợp: Bảng So sánh Sơ bộ Các Mô hình

_Tuần 2 – T6: Thành viên B (ML Engineer)_

---

## 1. Mục tiêu và Bối cảnh Thực hiện

Theo phân công nhiệm vụ tại `reports/project_plan.md` (mục 4.2 – Tuần 2, T6 và mục 1.2, 5):
1. **Tổng hợp bảng so sánh sơ bộ (CV mean ± std):** Tập hợp kết quả 5-fold Stratified Cross-Validation trên tập Train (18.000 mẫu, 22.12% default rate) của tất cả các mô hình đã huấn luyện và tối ưu từ Tuần 1 đến hết Tuần 2.
2. **Tuân thủ quy ước dữ liệu (mục 3.5a & 8.2):**
   - **Tập Train (60%):** Huấn luyện và đánh giá sơ bộ qua 5-fold Stratified CV cố định `random_state = 42`.
   - **Tập Valid (20%):** Dành cho bước Probability Calibration, chọn ngưỡng chi phí và fairness check ở Tuần 3 – T2.
   - **Tập Test (20%):** Đã dùng một lần duy nhất ở Tuần 3 – T3 cho mô hình cuối và baseline Logistic; kết quả nghiệm thu ở mục 2.1. Các mô hình khác không được đánh giá trên Test trong lượt này.
3. **Chuẩn bị cơ sở cho cuộc họp Review Tuần 2 (T6):** Cung cấp các bằng chứng thực nghiệm, kiểm định thống kê và phân tích trade-off để nhóm thống nhất lựa chọn **1–2 mô hình cuối cùng** bước vào giai đoạn hoàn thiện sản phẩm.

---

## 2. Bảng Tổng hợp So sánh Sơ bộ Các Mô hình Tiêu biểu

Bảng dưới đây tuân thủ cấu trúc chuẩn theo **Mục 5. Mẫu bảng so sánh mô hình** trong `reports/project_plan.md`:

| Mô hình | AUC CV (mean ± std) | AUC Test (95% CI) | Gini (Test) | KS (Test) | PR-AUC (Test) | Brier (Test) | Khả năng giải thích |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression (baseline) | 0.7745 ± 0.0116 | 0.7609 (0.7461, 0.7765) | 0.5219 | 0.4046 | 0.5236 | 0.1384 (thô) | Rất cao (hệ số odds ratio trực tiếp) |
| Decision Tree (max_depth=5) | 0.7672 ± 0.0120 | Không đánh giá | – | – | – | – | Cao (cây quyết định if-then rõ ràng) |
| Logistic Scorecard (WoE) | 0.7790 ± 0.0083 | Không đánh giá | – | – | – | – | Rất cao (bảng điểm PDO, chuẩn ngân hàng) |
| Random Forest (default) | 0.7667 ± 0.0097 | Không đánh giá | – | – | – | – | Trung bình (SHAP / Feature Importance) |
| XGBoost (default) | 0.7597 ± 0.0073 | Không đánh giá | – | – | – | – | Trung bình (SHAP TreeExplainer) |
| LightGBM (default) | 0.7805 ± 0.0077 | Không đánh giá | – | – | – | – | Trung bình (SHAP TreeExplainer) |
| CatBoost (default) | 0.7844 ± 0.0074 | Không đánh giá | – | – | – | – | Trung bình (SHAP TreeExplainer) |
| **LightGBM tuned + Monotonic (mô hình cuối)** | 0.7906 ± 0.0074 | **0.7825 (0.7681, 0.7973)** | 0.5649 | 0.4290 | 0.5600 | 0.1348 (Platt) | Trung bình - Cao (SHAP + Đơn điệu nghiệp vụ) |
| CatBoost tuned (Optuna) | 0.7911 ± 0.0073 | Không đánh giá | – | – | – | – | Trung bình (SHAP TreeExplainer) |
| MLP (tuỳ chọn) | Chưa triển khai (tùy chọn) | – | – | – | – | – | Thấp (KernelExplainer) |

> **Ghi chú kỹ thuật & Phương pháp tính:**
> - **Nguồn số liệu:** Cột AUC CV đo bằng 5-fold Stratified CV trên tập Train (`splits.json`, 18.000 dòng, `random_state = 42`). Các cột (Test) đo trên tập Test 6.000 hồ sơ, chỉ có cho mô hình cuối và baseline Logistic (mục 2.1); các mô hình còn lại không được đánh giá trên Test. Gini, KS, PR-AUC và Brier theo CV của mọi mô hình nằm ở bảng phụ (mục 3).
> - **Ngưỡng & cách tính chỉ số (Recall, Precision, F1):** Đo tại ngưỡng cố định `threshold = 0.5` độc lập trên từng fold kiểm định (validation fold), sau đó lấy trung bình (`mean`) và độ lệch chuẩn mẫu (`std`, `ddof=1`) qua 5 fold. Cách tính này khớp hoàn toàn với quy trình `run_cv` và `summarize_folds` log vào MLflow (không gộp chung toàn bộ OOF dự đoán trước khi tính).
> - **Lược bỏ cột thời gian train:** Cột thời gian huấn luyện đã được lược bỏ khỏi các bảng so sánh. Nguyên nhân: MLflow không lưu trữ đồng nhất một metric thời gian cho tất cả các mô hình và thời gian chạy wall-clock phụ thuộc cấu hình phần cứng cục bộ. Với quy mô 18.000 mẫu (mọi mô hình đều hoàn thành dưới 1 phút), thời gian train không phải là tiêu chí đánh đổi hay KPI chọn mô hình.
> - **Thống nhất số liệu Boosting default:** Bảng chính, bảng phụ và bảng kiểm định paired t-test sử dụng đồng bộ số liệu tính trên cùng fold-cache Optuna T4 (LightGBM default: 0.7805 ± 0.0077; CatBoost default: 0.7844 ± 0.0074). Bộ số này bảo đảm khớp toán học hoàn toàn giữa chênh lệch từng fold trong bảng paired và điểm CV trung bình.
> - **Brier score:** Brier Test của mô hình cuối là Brier sau Platt scaling. Brier Test của baseline là Brier thô vì baseline không được hiệu chuẩn, nên hai giá trị không hoàn toàn ngang hàng; AUC, Gini, KS và PR-AUC không phụ thuộc hiệu chuẩn. Brier CV (thô) của mọi mô hình: mục 3.
> - **Đóng băng tập Test:** Đã chạy đúng một lượt ở Tuần 3 – T3 cho LightGBM tuned + Monotonic và baseline Logistic; xem mục 2.1. Không dùng Test để chọn giữa các mô hình phụ.

### 2.1. Kết quả nghiệm thu cuối trên Test (Tuần 3 – T3)

Tập Test được đánh giá một lần bằng `scripts/evaluate_final_test.py`. Mô hình cuối và ngưỡng được khóa trước khi đánh giá: LightGBM tuned + monotonic, Platt fit trên Valid, ngưỡng 0.09 (làm tròn từ trung bình OOF EAD-weighted 0.088). Lượt Test so sánh mô hình cuối với baseline Logistic Regression; các mô hình còn lại không được đưa vào lượt nghiệm thu cuối.

| Mô hình/đầu ra | ROC-AUC Test (bootstrap 95% CI) | Gini | KS (bootstrap 95% CI) | PR-AUC (bootstrap 95% CI) | Brier (bootstrap 95% CI) | Log Loss | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM tuned + Monotonic (raw) | 0.7825 (0.7681, 0.7973) | 0.5649 | 0.4290 (0.4048, 0.4603) | 0.5600 (0.5315, 0.5887) | 0.1347 (0.1288, 0.1405) | 0.4284 | 0.0139 |
| **LightGBM tuned + Monotonic (Platt, final)** | **0.7825 (0.7681, 0.7973)** | **0.5649** | **0.4290 (0.4048, 0.4603)** | **0.5600 (0.5315, 0.5887)** | 0.1348 (0.1288, 0.1405) | 0.4285 | 0.0163 |
| Logistic Regression baseline | 0.7609 (0.7461, 0.7765) | 0.5219 | 0.4046 (0.3771, 0.4341) | 0.5236 (0.4960, 0.5546) | 0.1384 (0.1322, 0.1446) | 0.4410 | 0.0185 |

Paired DeLong cho LightGBM Platt so với Logistic baseline: **ΔAUC = 0.02150**, 95% CI **[0.01449, 0.02852]**, **p = 1.86 × 10⁻⁹**. Paired bootstrap 95% CI của ΔAUC là **[0.01434, 0.02857]**; cả hai cách đều cho thấy AUC cao hơn baseline có ý nghĩa thống kê, đáp ứng KPI chính. Platt không cải thiện calibration trên Test so với raw LightGBM (Brier 0.1348 so với 0.1347; ECE 0.0163 so với 0.0139), phù hợp với kết quả OOF; Platt vẫn được giữ cho đầu ra cuối theo lựa chọn đã khóa trên Valid để bảo toàn phép biến đổi tuyến tính cho SHAP. Brier, Log Loss và ECE của baseline tính trên PD chưa hiệu chuẩn.

Đối chiếu KPI tham chiếu (Charter mục 3.2) trên Test: ROC-AUC đạt mốc sàn (0.7825 ≥ 0.78) nhưng chưa tới mục tiêu 0.79; Gini (0.5649), KS (0.4290) và PR-AUC (0.55996) dưới mốc sàn, trong đó PR-AUC thiếu 0.00004. Khoảng tin cậy 95% của cả bốn chỉ số đều chứa mốc sàn. Bảng đầy đủ nằm trong [báo cáo nghiệm thu Test](test_evaluation/test_evaluation.md).

Tại ngưỡng 0.09 trên Test, recall là **92.46%**, precision **28.02%**, flag rate **72.98%**; EAD-weighted cost là **NT$1,634.49/khách hàng**, so với **NT$2,830.29** tại 0.50. Tỷ lệ cảnh báo cao cần được cân nhắc về năng lực vận hành. Chi tiết ROC/PR/calibration curves và toàn bộ chỉ số nằm trong báo cáo nghiệm thu Test, cùng các bảng máy đọc `test_evaluation/test_metrics.csv` và `test_evaluation/test_comparison.csv`.

---

## 3. Bảng phụ: Phân tích Chi tiết Toàn bộ Thử nghiệm (Tuần 1 & Tuần 2)

Tổng hợp toàn bộ 17 cấu hình mô hình và kỹ thuật đã thử nghiệm (bao gồm Baseline, Tree Ensembles, Gradient Boosting, Xử lý mất cân bằng, Optuna Tuning, và Đối chiếu Fairness):

| Mô hình | Nhóm | ROC-AUC (mean ± std) | Gini | KS | PR-AUC | Brier Score | Recall@0.5 | Precision@0.5 | F1-Score | Ghi chú |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression (baseline) | Baseline | 0.7745 ± 0.0116 | 0.5490 | 0.4282 | 0.5470 | 0.1354 | 0.3619 | 0.6760 | 0.4714 | Baseline chuẩn Tuần 1 |
| Decision Tree (max_depth=5) | Baseline | 0.7672 ± 0.0120 | 0.5345 | 0.4162 | 0.5242 | 0.1367 | 0.3531 | 0.6629 | 0.4595 | Baseline cây quyết định |
| Logistic Scorecard (WoE) | Scorecard | 0.7790 ± 0.0083 | 0.5580 | 0.4326 | 0.5432 | 0.1351 | 0.3744 | 0.6596 | 0.4777 | Scorecard chuẩn mực (coarse classing + ràng buộc dấu) |
| Random Forest (default) | Tree Ensembles | 0.7667 ± 0.0097 | 0.5333 | 0.4107 | 0.5319 | 0.1388 | 0.3870 | 0.6313 | 0.4797 | Tree ensemble mặc định T2 |
| XGBoost (default) | Tree Ensembles | 0.7597 ± 0.0073 | 0.5195 | 0.4007 | 0.5258 | 0.1440 | 0.3669 | 0.6077 | 0.4575 | XGBoost mặc định T2 |
| LightGBM (default) | Gradient Boosting (T3) | 0.7805 ± 0.0077 | 0.5610 | 0.4276 | 0.5512 | 0.1350 | 0.3767 | 0.6677 | 0.4816 | Boosting mặc định (đồng bộ fold-cache Optuna T4) |
| CatBoost (default) | Gradient Boosting (T3) | 0.7844 ± 0.0074 | 0.5688 | 0.4359 | 0.5584 | 0.1342 | 0.3784 | 0.6696 | 0.4836 | Boosting mặc định (đồng bộ fold-cache Optuna T4) |
| LightGBM Balanced (class_weight) | Imbalance Experiments | 0.7809 ± 0.0081 | 0.5617 | 0.4278 | 0.5505 | 0.1682 | 0.6110 | 0.4758 | 0.5350 | Recall cao nhưng Brier tăng mạnh |
| CatBoost Balanced (auto_class_weights) | Imbalance Experiments | 0.7806 ± 0.0081 | 0.5611 | 0.4341 | 0.5520 | 0.1659 | 0.6072 | 0.4872 | 0.5406 | Recall cao nhưng Brier tăng mạnh |
| LightGBM + SMOTE (in-fold) | Imbalance Experiments | 0.7776 ± 0.0058 | 0.5551 | 0.4216 | 0.5440 | 0.1383 | 0.4309 | 0.6222 | 0.5092 | Thí nghiệm phụ SMOTE: AUC giảm |
| CatBoost + SMOTE (in-fold) | Imbalance Experiments | 0.7747 ± 0.0057 | 0.5493 | 0.4192 | 0.5410 | 0.1386 | 0.4156 | 0.6130 | 0.4953 | Thí nghiệm phụ SMOTE: AUC giảm |
| LightGBM tuned (Optuna) | Optuna Tuned (T4) | 0.7898 ± 0.0077 | 0.5796 | 0.4439 | 0.5650 | 0.1327 | 0.3774 | 0.6696 | 0.4828 | Tuned 37 trials, plateau stopping |
| LightGBM tuned + Monotonic | Optuna Tuned (T4) - Candidate | 0.7906 ± 0.0074 | 0.5813 | 0.4453 | 0.5624 | 0.1328 | 0.3774 | 0.6748 | 0.4839 | Ứng viên chính: Nhanh, bảo đảm chiều tác động, AUC đạt mục tiêu |
| CatBoost tuned (Optuna) | Optuna Tuned (T4) - Candidate | 0.7911 ± 0.0073 | 0.5822 | 0.4452 | 0.5675 | 0.1327 | 0.3674 | 0.6754 | 0.4759 | Ứng viên đối chứng: AUC CV cao nhất toàn dự án (0.7911) |
| CatBoost tuned + Monotonic (đối chiếu) | Optuna Tuned (T4) | 0.7910 ± 0.0077 | 0.5820 | 0.4434 | 0.5632 | 0.1328 | 0.3722 | 0.6675 | 0.4780 | Thời gian train lâu ~5x so với CatBoost thường |
| LightGBM tuned + Monotonic (with SEX) | Fairness Comparison (T5) | 0.7907 ± 0.0075 | 0.5814 | 0.4465 | 0.5634 | 0.1327 | 0.3769 | 0.6708 | 0.4828 | SEX chỉ chiếm 0.17% feature importance; delta AUC +0.0001 |
| CatBoost tuned (with SEX) | Fairness Comparison (T5) | 0.7912 ± 0.0079 | 0.5824 | 0.4461 | 0.5681 | 0.1326 | 0.3692 | 0.6723 | 0.4767 | SEX chỉ chiếm 0.54% feature importance; delta AUC +0.0001 |

---

## 4. Kiểm định Thống kê & So sánh Cặp theo Fold

Vì tất cả các mô hình đều được huấn luyện trên cùng 5 fold Stratified CV (`random_state=42`), chênh lệch AUC theo từng fold ($\Delta \text{AUC}$) phản ánh khác biệt thực chất của thuật toán và loại trừ phương sai do việc chia mẫu.

Kiểm định paired t-test trên 5 fold đối chiếu với Baseline Logistic Regression:

| So sánh cặp | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Δ AUC TB | Std Δ | 95% CI (Δ AUC) | t-statistic | p-value (paired) | Ý nghĩa (p < 0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CatBoost tuned vs Logistic baseline | +0.0145 | +0.0091 | +0.0196 | +0.0236 | +0.0162 | +0.0166 | 0.0055 | [+0.0098, +0.0234] | 6.810 | 0.00243 | Có ý nghĩa (***) |
| LightGBM monotonic vs Logistic baseline | +0.0144 | +0.0077 | +0.0185 | +0.0244 | +0.0157 | +0.0161 | 0.0061 | [+0.0086, +0.0237] | 5.928 | 0.00406 | Có ý nghĩa (***) |
| CatBoost default vs Logistic baseline | +0.0077 | +0.0021 | +0.0125 | +0.0176 | +0.0098 | +0.0099 | 0.0057 | [+0.0028, +0.0171] | 3.873 | 0.01795 | Có ý nghĩa (*) |
| LightGBM default vs Logistic baseline | +0.0005 | -0.0007 | +0.0075 | +0.0109 | +0.0116 | +0.0060 | 0.0058 | [-0.0012, +0.0131] | 2.313 | 0.08173 | Không |
| Logistic Scorecard vs Logistic baseline | -0.0022 | -0.0051 | +0.0040 | +0.0191 | +0.0068 | +0.0045 | 0.0094 | [-0.0072, +0.0162] | 1.072 | 0.34420 | Không |
| CatBoost tuned vs LightGBM monotonic | +0.0001 | +0.0015 | +0.0011 | -0.0008 | +0.0004 | +0.0005 | 0.0009 | [-0.0007, +0.0016] | 1.148 | 0.31501 | Không |

### Nhận xét kiểm định thống kê:
- **Cả CatBoost tuned và LightGBM tuned + monotonic đều vượt Baseline trên 5/5 fold:**
  - `CatBoost tuned`: Chênh lệch AUC trung bình **+0.0166** (dao động từ +0.0091 đến +0.0236), $t = 6.810$, $p = 0.00243 < 0.01$.
  - `LightGBM monotonic`: Chênh lệch AUC trung bình **+0.0161** (dao động từ +0.0077 đến +0.0244), $t = 5.928$, $p = 0.00406 < 0.01$.
- **Cả hai mô hình Boosting mặc định đều cải thiện AUC so với Baseline LR:**
  - `CatBoost default`: Chênh lệch AUC trung bình **+0.0099** (dao động từ +0.0021 đến +0.0176), $t = 3.873$, $p = 0.01795 < 0.05$ (có ý nghĩa thống kê (*), 95% CI [+0.0028, +0.0171]).
  - `LightGBM default`: Chênh lệch AUC trung bình **+0.0060** (dao động từ -0.0007 đến +0.0116), $t = 2.313$, $p = 0.08173 > 0.05$ (chưa đạt ý nghĩa thống kê ở mức 5% trên 5 fold, 95% CI [-0.0012, +0.0131]).
- **Scorecard WoE không vượt baseline có ý nghĩa thống kê:** Chênh lệch $+0.0045$ AUC trung bình có $t = 1.072$, $p = 0.34420 \approx 0.34 > 0.05$; khoảng tin cậy 95% $[-0.0072, +0.0162]$ chứa 0; và trên thực tế có 2/5 fold Scorecard có AUC thấp hơn Baseline LR.
- **Sự khác biệt giữa CatBoost tuned và LightGBM monotonic là không có ý nghĩa thống kê ($p = 0.31501 \approx 0.31 > 0.05$):** Chênh lệch giữa hai mô hình chỉ là $+0.0005$ AUC trung bình (95% CI $[-0.0007, +0.0016]$ chứa 0) và đảo dấu ở Fold 4 (-0.0008). Về mặt năng lực phân tách, hai mô hình này tương đương nhau.

---

## 5. Đối chiếu với Project Charter & KPI

Đối chiếu hiệu năng CV của các mô hình ứng viên với các mốc KPI đã cam kết trong `reports/project_charter.md`:

| Mô hình | ROC-AUC (Sàn 0.78 / Mục tiêu 0.79) | Gini (Sàn 0.57 / Mục tiêu 0.59) | KS (Sàn 0.45 / Mục tiêu 0.47) | PR-AUC (Sàn 0.56 / Mục tiêu 0.58) |
| --- | --- | --- | --- | --- |
| Logistic Regression (baseline) | Chưa đạt sàn (0.7745 < 0.78) | Chưa đạt sàn (0.5490 < 0.57) | Chưa đạt sàn (0.4282 < 0.45) | Chưa đạt sàn (0.5470 < 0.56) |
| Logistic Scorecard (WoE) | Chưa đạt sàn (0.7790 < 0.78) | Chưa đạt sàn (0.5580 < 0.57) | Chưa đạt sàn (0.4326 < 0.45) | Chưa đạt sàn (0.5432 < 0.56) |
| LightGBM tuned + Monotonic | Đạt mục tiêu (0.7906 ≥ 0.79) | Đạt sàn (0.5813 ≥ 0.57) | Chưa đạt sàn (0.4453 < 0.45) | Đạt sàn (0.5624 ≥ 0.56) |
| CatBoost tuned (Optuna) | Đạt mục tiêu (0.7911 ≥ 0.79) | Đạt sàn (0.5822 ≥ 0.57) | Chưa đạt sàn (0.4452 < 0.45) | Đạt sàn (0.5675 ≥ 0.56) |

### Đánh giá KPI Charter:
1. **KPI chính (tương đối):** Cả hai mô hình boosting sau tuning đều vượt baseline trên mọi fold với $p < 0.005$, đáp ứng đầy đủ tiêu chí vượt baseline có ý nghĩa thống kê.
2. **KPI tham chiếu (tuyệt đối):**
   - **ROC-AUC:** Cả CatBoost tuned (0.7911) và LightGBM monotonic (0.7906) đều **vượt mốc mục tiêu $\ge 0.79$**.
   - **Gini:** Đạt mốc sàn $\ge 0.57$ (CatBoost 0.5822, LightGBM 0.5813), tiệm cận mốc mục tiêu 0.59.
   - **PR-AUC:** Đạt mốc sàn $\ge 0.56$ (CatBoost 0.5675, LightGBM 0.5624).
   - **KS:** **Chưa đạt mốc sàn** ($0.4452–0.4453$ so với mốc sàn 0.45, thiếu khoảng ~0.005; bảng đối chiếu KPI phân loại 'Chưa đạt sàn'). Theo kết luận tuning tại Tuần 2 – T4, đây là trần tự nhiên của bộ dữ liệu snapshot 2005 trên mô hình dự đoán xác suất không leak dữ liệu.
   - **Trên tập Test (mô hình cuối, mục 2.1):** ROC-AUC 0.7825 đạt mốc sàn, chưa tới mục tiêu; Gini 0.5649, KS 0.4290 và PR-AUC 0.55996 dưới mốc sàn. ROC-AUC CV cao hơn Test khoảng một độ lệch chuẩn (0.7906 so với 0.7825), phù hợp ghi chú ở Charter mục 3.2.

3. **Kế hoạch Chi phí Kỳ vọng & Lựa chọn Ngưỡng Tối ưu (Kế hoạch Tuần 3 – T2):**
   - **Ma trận chi phí (Charter mục 1.3):** Chi phí $FN \approx 0.45 \times EAD$ (với giả định $LGD = 0.45$), chi phí $FP \approx 0.05 \times EAD$ (tỷ lệ $FN:FP = 9:1$, kèm phân tích độ nhạy $3:1$ đến $10:1$).
   - **Nguyên tắc chống rò rỉ (No Leakage / Overfitting):** Ngưỡng tối ưu không được chọn trên tập Train hay từng fold CV (tránh ước lượng chi phí lạc quan quá mức như phân tích tại PR #16 và `src/train.py`).
   - **Lộ trình triển khai:** Calibrator và ngưỡng được chọn trên Valid ở Tuần 3 – T2; chi phí độc lập trên Test đã được đo tại Tuần 3 – T3 và báo cáo ở mục 2.1.

---

## 6. Tổng kết Đánh giá Kỹ thuật theo Từng Nhóm Mô hình

### 6.1. Nhóm Baseline & Scorecard Truyền thống
- **Logistic Regression baseline (0.7745 ± 0.0116):** Hoạt động ổn định, độ phân tách khá, thời gian train tức thì (~2.5s). Là mốc chuẩn tin cậy.
- **Logistic Scorecard WoE (0.7790 ± 0.0083):**
  - Về mặt số học, AUC trung bình cao hơn baseline Logistic thông thường một lượng nhỏ (+0.0045 AUC, 0.7790 vs 0.7745), tuy nhiên **chênh lệch này không có ý nghĩa thống kê** ($t = 1.072, p = 0.344 > 0.05$; khoảng tin cậy 95% $[-0.0072, +0.0162]$ chứa giá trị 0; và trên thực tế có 2/5 fold Scorecard thấp hơn LR).
  - Ưu thế cốt lõi của Scorecard không nằm ở việc vượt trội về năng lực phân tách, mà nằm ở **tính minh bạch và chuẩn mực quản trị rủi ro ngân hàng**: Nhờ coarse classing và ràng buộc dấu, hệ số có dấu phù hợp logic nghiệp vụ ngân hàng và điểm số của biến liên tục đơn điệu theo giá trị.
  - Điểm số PDO tính trực tiếp từ bảng điểm thuộc tính nên truy vết được từng thành phần; là mốc so sánh minh bạch cho các mô hình học máy phức tạp.

### 6.2. Nhóm Tree Ensembles Mặc định (RF & XGBoost)
- **Random Forest (0.7667) và XGBoost (0.7597):** Khi chưa tinh chỉnh tham số, cả hai mô hình cây đều có AUC thấp hơn Logistic Regression. Nguyên nhân có thể nằm ở cấu hình mặc định: XGBoost `max_depth=4` đạt ROC-AUC CV khoảng 0.788 (`shap_agreement.md`).

### 6.3. Nhóm Gradient Boosting Hiện đại (LightGBM & CatBoost)
- **Năng lực phân tách cao ngay từ bản Default:** LightGBM default (0.7805) và CatBoost default (0.7844) đều vượt mốc sàn Charter 0.78 khi chưa tuning.
- **Sau khi tuning Optuna:**
  - Cả hai mô hình hội tụ về kiến trúc: cây nông (num_leaves 10–11, depth 6), learning rate nhỏ (0.005–0.011), regularization mạnh (`reg_lambda` 8.87, `l2_leaf_reg` 16.76).
  - Cải thiện ROC-AUC lên **0.7906 – 0.7911**, mức cao nhất của dự án.

### 6.4. Bài học từ các Thử nghiệm Kỹ thuật Mở rộng
1. **Ràng buộc đơn điệu (Monotonic constraints):**
   - Trên LightGBM, monotonic constraints không làm giảm AUC (0.7906 vs 0.7898), thời gian train gần như không đổi (~14s). Đảm bảo tính đơn điệu nghiệp vụ: khách hàng trễ hạn nhiều hơn hoặc dùng hạn mức cao hơn sẽ không bao giờ bị giảm xác suất rủi ro.
   - Trên CatBoost, monotonic constraints làm thời gian huấn luyện tăng gấp 5 lần (~318s) mà không tăng thêm AUC.
2. **Xử lý mất cân bằng (Class Weighting vs SMOTE):**
   - `class_weight`: Recall tăng lên ~61% ở ngưỡng 0.5, nhưng Brier score tăng từ 0.133 lên 0.168 (xác suất bị méo).
   - `SMOTE in-fold`: Không cải thiện AUC (0.774–0.777), làm tăng thời gian huấn luyện gấp đôi. Phù hợp kết luận Charter: không dùng SMOTE cho mô hình chính thức.
3. **Đối chiếu Biến nhạy cảm SEX:**
   - Việc thêm biến SEX vào mô hình chỉ làm thay đổi AUC trung bình $+0.0001$, tỷ trọng đóng góp của SEX trong feature importance chỉ đạt $0.17\%$ (LightGBM) và $0.54\%$ (CatBoost).
   - Kết quả phù hợp chính sách Charter: **loại bỏ biến SEX khỏi mô hình chính thức mà không làm suy giảm hiệu năng**.

---

## 7. Khuyến nghị cho Cuộc họp Review Tuần 2 (Chốt 1–2 Mô hình Cuối)

Thành viên B (ML Engineer) đề xuất với nhóm tại cuộc họp Review Tuần 2:

1. **Ứng viên Mô hình Chính (Primary Model): `LightGBM tuned + Monotonic constraints`**
   - **Lý do lựa chọn:**
     - Hiệu năng phân tách cao: ROC-AUC **0.7906 ± 0.0074**, vượt mốc mục tiêu Charter 0.79.
     - Tính nhất quán nghiệp vụ: 11 biến trễ hạn và tỷ lệ sử dụng hạn mức được bảo đảm đơn điệu tăng tuyệt đối.
     - Tốc độ huấn luyện: 5 fold mất **14 giây** (nhanh hơn CatBoost 5 lần và nhanh hơn CatBoost monotonic 22 lần).
     - Tương thích giải thích SHAP: Tương thích với `shap.TreeExplainer`, thời gian tính SHAP đủ ngắn cho demo Streamlit.
2. **Ứng viên Mô hình Đối chứng (Benchmark / Secondary): `CatBoost tuned`**
   - Giữ làm mô hình đối chứng năng lực thuật toán (AUC cao nhất 0.7911) và kiểm tra độ ổn định trên tập Valid/Test.
3. **Mô hình Chuẩn mực Nghiệp vụ: `Logistic Scorecard (WoE)`**
   - Giữ nguyên trong pipeline demo để người dùng có thể so sánh song song giữa mô hình Black-box hiện đại (LightGBM) và Bảng điểm truyền thống chuẩn ngân hàng (Scorecard).

---

## 8. Lệnh Tái lập & Kiểm thử

Bảng máy đọc `reports/model_comparison.csv` và các bảng so sánh, kiểm định cặp, đối chiếu KPI (in ra console) được sinh bằng:
```bash
python scripts/compare_models.py
# hoặc
python scripts/run_pipeline.py --mode summary
```

Lệnh này chỉ ghi `reports/model_comparison.csv`; báo cáo này được viết tay (gồm kết quả Test ở mục 2.1) nên script không ghi đè. Kết quả Test lấy từ `scripts/evaluate_final_test.py`, đã chạy một lần (xem `reports/test_evaluation/`).
