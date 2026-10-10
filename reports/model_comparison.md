# Báo cáo Tổng hợp: Bảng So sánh Sơ bộ Các Mô hình

_Tuần 2 – T6: Thành viên B (ML Engineer)_

---

## 1. Mục tiêu và Bối cảnh Thực hiện

Theo phân công nhiệm vụ tại `reports/project_plan.md` (mục 4.2 – Tuần 2, T6 và mục 1.2, 5):
1. **Tổng hợp bảng so sánh sơ bộ (CV mean ± std):** Tập hợp kết quả 5-fold Stratified Cross-Validation trên tập Train (18.000 mẫu, 22.12% default rate) của tất cả các mô hình đã huấn luyện và tối ưu từ Tuần 1 đến hết Tuần 2.
2. **Tuân thủ quy ước dữ liệu (mục 3.5a & 8.2):**
   - **Tập Train (60%):** Huấn luyện và đánh giá sơ bộ qua 5-fold Stratified CV cố định `random_state = 42`.
   - **Tập Valid (20%):** Dành cho bước Probability Calibration, chọn ngưỡng chi phí và fairness check ở Tuần 3 – T2.
   - **Tập Test (20%):** Giữ nguyên vẹn, **chỉ chạy một lần duy nhất ở Tuần 3 – T3** để báo cáo kết quả cuối cùng. Do đó, các cột đo trên tập Test trong bảng mẫu sẽ được điền ở Tuần 3.
3. **Chuẩn bị cơ sở cho cuộc họp Review Tuần 2 (T6):** Cung cấp các bằng chứng thực nghiệm, kiểm định thống kê và phân tích trade-off để nhóm thống nhất lựa chọn **1–2 mô hình cuối cùng** bước vào giai đoạn hoàn thiện sản phẩm.

---

## 2. Bảng Tổng hợp So sánh Sơ bộ Các Mô hình Tiêu biểu

Bảng dưới đây tuân thủ cấu trúc chuẩn theo **Mục 5. Mẫu bảng so sánh mô hình** trong `reports/project_plan.md`:

| Mô hình | AUC CV (mean ± std) | AUC Test (95% CI) | Gini | KS | PR-AUC | Brier (thô) | Thời gian train | Khả năng giải thích |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression (baseline) | 0.7745 ± 0.0116 | Dành cho Tuần 3 – T3 | 0.5490 | 0.4282 | 0.5362 | 0.1354 | ~2.5s | Rất cao (hệ số odds ratio trực tiếp) |
| Decision Tree (max_depth=5) | 0.7672 ± 0.0120 | Dành cho Tuần 3 – T3 | 0.5345 | 0.4162 | 0.5284 | 0.1367 | ~1.8s | Cao (cây quyết định if-then rõ ràng) |
| Logistic Scorecard (WoE) | 0.7790 ± 0.0083 | Dành cho Tuần 3 – T3 | 0.5580 | 0.4326 | 0.5432 | 0.1351 | ~4.2s | Rất cao (bảng điểm PDO, chuẩn ngân hàng) |
| Random Forest (default) | 0.7667 ± 0.0097 | Dành cho Tuần 3 – T3 | 0.5333 | 0.4107 | 0.5301 | 0.1388 | ~8.5s | Trung bình (SHAP / Feature Importance) |
| XGBoost (default) | 0.7597 ± 0.0073 | Dành cho Tuần 3 – T3 | 0.5195 | 0.4007 | 0.5218 | 0.1440 | ~5.2s | Trung bình (SHAP TreeExplainer) |
| LightGBM (default) | 0.7813 ± 0.0064 | Dành cho Tuần 3 – T3 | 0.5626 | 0.4276 | 0.5512 | 0.1350 | ~3.5s | Trung bình (SHAP TreeExplainer) |
| CatBoost (default) | 0.7853 ± 0.0077 | Dành cho Tuần 3 – T3 | 0.5706 | 0.4359 | 0.5584 | 0.1342 | ~28.0s | Trung bình (SHAP TreeExplainer) |
| LightGBM tuned + Monotonic | 0.7906 ± 0.0074 | Dành cho Tuần 3 – T3 | 0.5813 | 0.4453 | 0.5624 | 0.1328 | ~14.0s | Trung bình - Cao (SHAP + Đơn điệu nghiệp vụ) |
| CatBoost tuned (Optuna) | 0.7911 ± 0.0073 | Dành cho Tuần 3 – T3 | 0.5822 | 0.4452 | 0.5675 | 0.1327 | ~66.0s | Trung bình (SHAP TreeExplainer) |

> **Ghi chú kỹ thuật:**
> - Chỉ số CV đo bằng 5-fold Stratified CV trên tập Train (`splits.json`, 18.000 dòng).
> - Brier score ở bảng trên là Brier thô (chưa qua calibration). Brier sau hiệu chuẩn sẽ được cập nhật ở Tuần 3 sau bước Platt scaling trên tập Valid.
> - Cột Test (ROC-AUC 95% CI, Gini Test, KS Test...) tuân thủ quy tắc đóng băng tập Test, chỉ thực hiện tại Tuần 3 – T3.

---

## 3. Bảng phụ: Phân tích Chi tiết Toàn bộ Thử nghiệm (Tuần 1 & Tuần 2)

Tổng hợp toàn bộ 17 cấu hình mô hình và kỹ thuật đã thử nghiệm (bao gồm Baseline, Tree Ensembles, Gradient Boosting, Xử lý mất cân bằng, Optuna Tuning, và Đối chiếu Fairness):

| Mô hình | Nhóm | ROC-AUC (mean ± std) | Gini | KS | PR-AUC | Brier Score | Recall@0.5 | Precision@0.5 | F1-Score | Thời gian | Ghi chú |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression (baseline) | Baseline | 0.7745 ± 0.0116 | 0.5490 | 0.4282 | 0.5362 | 0.1354 | 0.3235 | 0.6728 | 0.4369 | ~2.5s | Baseline chuẩn Tuần 1 |
| Decision Tree (max_depth=5) | Baseline | 0.7672 ± 0.0120 | 0.5345 | 0.4162 | 0.5284 | 0.1367 | 0.3541 | 0.6385 | 0.4555 | ~1.8s | Baseline cây quyết định |
| Logistic Scorecard (WoE) | Scorecard | 0.7790 ± 0.0083 | 0.5580 | 0.4326 | 0.5432 | 0.1351 | 0.3340 | 0.6750 | 0.4470 | ~4.2s | Scorecard chuẩn mực (coarse classing + ràng buộc dấu) |
| Random Forest (default) | Tree Ensembles | 0.7667 ± 0.0097 | 0.5333 | 0.4107 | 0.5301 | 0.1388 | 0.3644 | 0.6321 | 0.4624 | ~8.5s | Tree ensemble mặc định T2 |
| XGBoost (default) | Tree Ensembles | 0.7597 ± 0.0073 | 0.5195 | 0.4007 | 0.5218 | 0.1440 | 0.3601 | 0.6120 | 0.4531 | ~5.2s | XGBoost mặc định T2 |
| LightGBM (default) | Gradient Boosting (T3) | 0.7813 ± 0.0064 | 0.5626 | 0.4276 | 0.5512 | 0.1350 | 0.3767 | 0.6677 | 0.4816 | ~3.5s | Boosting mặc định T3 |
| CatBoost (default) | Gradient Boosting (T3) | 0.7853 ± 0.0077 | 0.5706 | 0.4359 | 0.5584 | 0.1342 | 0.3784 | 0.6696 | 0.4836 | ~28.0s | Boosting mặc định T3, AUC cao nhất mốc default |
| LightGBM Balanced (class_weight) | Imbalance Experiments | 0.7809 ± 0.0081 | 0.5617 | 0.4278 | 0.5505 | 0.1682 | 0.6110 | 0.4758 | 0.5350 | ~3.8s | Recall cao nhưng Brier tăng mạnh |
| CatBoost Balanced (auto_class_weights) | Imbalance Experiments | 0.7806 ± 0.0081 | 0.5611 | 0.4341 | 0.5520 | 0.1659 | 0.6072 | 0.4872 | 0.5406 | ~30.0s | Recall cao nhưng Brier tăng mạnh |
| LightGBM + SMOTE (in-fold) | Imbalance Experiments | 0.7776 ± 0.0058 | 0.5551 | 0.4216 | 0.5440 | 0.1383 | 0.4309 | 0.6222 | 0.5092 | ~7.2s | Thí nghiệm phụ SMOTE: AUC giảm |
| CatBoost + SMOTE (in-fold) | Imbalance Experiments | 0.7747 ± 0.0057 | 0.5493 | 0.4192 | 0.5410 | 0.1386 | 0.4156 | 0.6130 | 0.4953 | ~54.0s | Thí nghiệm phụ SMOTE: AUC giảm |
| LightGBM tuned (Optuna) | Optuna Tuned (T4) | 0.7898 ± 0.0077 | 0.5796 | 0.4439 | 0.5650 | 0.1327 | 0.3774 | 0.6696 | 0.4828 | ~12.5s | Tuned 37 trials, plateau stopping |
| LightGBM tuned + Monotonic | Optuna Tuned (T4) - Candidate | 0.7906 ± 0.0074 | 0.5813 | 0.4453 | 0.5624 | 0.1328 | 0.3774 | 0.6748 | 0.4839 | ~14.0s | ỨNG VIÊN CHÍNH: Nhanh, bảo đảm chiều tác động, AUC đạt mục tiêu |
| CatBoost tuned (Optuna) | Optuna Tuned (T4) - Candidate | 0.7911 ± 0.0073 | 0.5822 | 0.4452 | 0.5675 | 0.1327 | 0.3674 | 0.6754 | 0.4759 | ~66.0s | ỨNG VIÊN ĐỐI CHỨNG: AUC CV cao nhất toàn dự án (0.7911) |
| CatBoost tuned + Monotonic (đối chiếu) | Optuna Tuned (T4) | 0.7910 ± 0.0077 | 0.5820 | 0.4434 | 0.5632 | 0.1328 | 0.3722 | 0.6675 | 0.4780 | ~318.0s | Thời gian train lâu ~5x so với CatBoost thường |
| LightGBM tuned + Monotonic (with SEX) | Fairness Comparison (T5) | 0.7907 ± 0.0075 | 0.5814 | 0.4465 | 0.5634 | 0.1327 | 0.3769 | 0.6708 | 0.4828 | ~14.5s | SEX chỉ chiếm 0.17% feature importance; delta AUC +0.0001 |
| CatBoost tuned (with SEX) | Fairness Comparison (T5) | 0.7912 ± 0.0079 | 0.5824 | 0.4461 | 0.5681 | 0.1326 | 0.3692 | 0.6723 | 0.4767 | ~68.0s | SEX chỉ chiếm 0.54% feature importance; delta AUC +0.0001 |

---

## 4. Kiểm định Thống kê & So sánh Cặp theo Fold

Vì tất cả các mô hình đều được huấn luyện trên cùng 5 fold Stratified CV (`random_state=42`), chênh lệch AUC theo từng fold ($\Delta \text{AUC}$) phản ánh khác biệt thực chất của thuật toán và loại trừ phương sai do việc chia mẫu.

Kiểm định paired t-test trên 5 fold đối chiếu với Baseline Logistic Regression:

| So sánh cặp | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Δ AUC TB | Std Δ | t-statistic | p-value (paired) | Ý nghĩa (p < 0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CatBoost tuned vs Logistic baseline | +0.0145 | +0.0091 | +0.0196 | +0.0236 | +0.0162 | +0.0166 | 0.0055 | 6.810 | 0.00243 | Có ý nghĩa (***) |
| LightGBM monotonic vs Logistic baseline | +0.0144 | +0.0077 | +0.0185 | +0.0244 | +0.0157 | +0.0161 | 0.0061 | 5.928 | 0.00406 | Có ý nghĩa (***) |
| CatBoost default vs Logistic baseline | +0.0076 | +0.0021 | +0.0125 | +0.0176 | +0.0098 | +0.0099 | 0.0057 | 3.859 | 0.01817 | Có ý nghĩa (*) |
| LightGBM default vs Logistic baseline | +0.0023 | -0.0013 | +0.0090 | +0.0134 | +0.0062 | +0.0059 | 0.0057 | 2.315 | 0.08155 | Không |
| CatBoost tuned vs LightGBM monotonic | +0.0001 | +0.0015 | +0.0011 | -0.0008 | +0.0004 | +0.0005 | 0.0009 | 1.148 | 0.31501 | Không |

### Nhận xét kiểm định thống kê:
- **Cả CatBoost tuned và LightGBM tuned + monotonic đều vượt trội hơn hẳn Baseline trên 100% các fold (5/5 fold dương):**
  - `CatBoost tuned`: Chênh lệch AUC trung bình **+0.0166** (dao động từ +0.0091 đến +0.0236), $t = 6.810$, $p = 0.00243 < 0.01$.
  - `LightGBM monotonic`: Chênh lệch AUC trung bình **+0.0161** (dao động từ +0.0077 đến +0.0244), $t = 5.928$, $p = 0.00406 < 0.01$.
- **Sự khác biệt giữa CatBoost tuned và LightGBM monotonic là không đáng kể ($p = 0.505$):** Chênh lệch giữa hai mô hình chỉ là $+0.0005$ AUC trung bình và đảo dấu ở Fold 4 (-0.0008). Về mặt năng lực phân tách, hai mô hình này tương đương nhau.

---

## 5. Đối chiếu với Project Charter & KPI

Đối chiếu hiệu năng CV của các mô hình ứng viên với các mốc KPI đã cam kết trong `reports/project_charter.md`:

| Mô hình | ROC-AUC (Sàn 0.78 / Mục tiêu 0.79) | Gini (Sàn 0.57 / Mục tiêu 0.59) | KS (Sàn 0.45 / Mục tiêu 0.47) | PR-AUC (Sàn 0.56 / Mục tiêu 0.58) |
| --- | --- | --- | --- | --- |
| Logistic Regression (baseline) | Chưa đạt sàn (0.7745 < 0.78) | Chưa đạt sàn (0.5490 < 0.57) | Chưa đạt sàn (0.4282 < 0.45) | Chưa đạt sàn (0.5362 < 0.56) |
| Logistic Scorecard (WoE) | Chưa đạt sàn (0.7790 < 0.78) | Chưa đạt sàn (0.5580 < 0.57) | Chưa đạt sàn (0.4326 < 0.45) | Chưa đạt sàn (0.5432 < 0.56) |
| LightGBM tuned + Monotonic | Đạt mục tiêu (0.7906 ≥ 0.79) | Đạt sàn (0.5813 ≥ 0.57) | Chưa đạt sàn (0.4453 < 0.45) | Đạt sàn (0.5624 ≥ 0.56) |
| CatBoost tuned (Optuna) | Đạt mục tiêu (0.7911 ≥ 0.79) | Đạt sàn (0.5822 ≥ 0.57) | Chưa đạt sàn (0.4452 < 0.45) | Đạt sàn (0.5675 ≥ 0.56) |

### Đánh giá KPI Charter:
1. **KPI chính (tương đối):** Cả hai mô hình boosting sau tuning đều vượt baseline vượt bậc trên mọi fold với $p < 0.005$, đáp ứng đầy đủ tiêu chí vượt baseline có ý nghĩa thống kê.
2. **KPI tham chiếu (tuyệt đối):**
   - **ROC-AUC:** Cả CatBoost tuned (0.7911) và LightGBM monotonic (0.7906) đều **vượt mốc mục tiêu $\ge 0.79$**.
   - **Gini:** Đạt mốc sàn $\ge 0.57$ (CatBoost 0.5822, LightGBM 0.5813), tiệm cận mốc mục tiêu 0.59.
   - **PR-AUC:** Đạt mốc sàn $\ge 0.56$ (CatBoost 0.5675, LightGBM 0.5624).
   - **KS:** Đạt ~0.445 (tiệm cận mốc sàn 0.45, thiếu khoảng 0.005). Theo kết luận tuning tại Tuần 2 – T4, đây là trần tự nhiên của bộ dữ liệu snapshot 2005.

---

## 6. Tổng kết Đánh giá Kỹ thuật theo Từng Nhóm Mô hình

### 6.1. Nhóm Baseline & Scorecard Truyền thống
- **Logistic Regression baseline (0.7745 ± 0.0116):** Hoạt động ổn định, độ phân tách khá, thời gian train tức thì (~2.5s). Là mốc chuẩn tin cậy.
- **Logistic Scorecard WoE (0.7790 ± 0.0083):**
  - Vượt baseline Logistic thông thường (+0.0045 AUC), độ lệch chuẩn thấp hơn (0.0083 vs 0.0116).
  - Nhờ coarse classing và ràng buộc dấu, toàn bộ hệ số và điểm số hoàn toàn tuân thủ logic nghiệp vụ ngân hàng (không bị đảo điểm phi lý).
  - Điểm số PDO tính toán trực tiếp từ bảng điểm thuộc tính, minh bạch 100%. Là chuẩn mực so sánh tuyệt vời cho các mô hình học máy phức tạp.

### 6.2. Nhóm Tree Ensembles Mặc định (RF & XGBoost)
- **Random Forest (0.7667) và XGBoost (0.7597):** Khi chưa tinh chỉnh tham số, cả hai mô hình cây đều có AUC thấp hơn Logistic Regression. Nguyên nhân do XGBoost mặc định dễ overfit trên dữ liệu bảng tín dụng có nhiều biến nhiễu.

### 6.3. Nhóm Gradient Boosting Hiện đại (LightGBM & CatBoost)
- **Năng lực phân tách vượt trội ngay từ bản Default:** LightGBM default (0.7813) và CatBoost default (0.7853) đều tự động vượt mốc sàn Charter 0.78 ngay khi chưa tuning.
- **Sau khi tuning Optuna:**
  - Cả hai mô hình hội tụ về kiến trúc: cây nông (num_leaves 10–11, depth 6), learning rate nhỏ (0.005–0.011), regularization mạnh (`reg_lambda` 8.87, `l2_leaf_reg` 16.76).
  - Cải thiện ROC-AUC lên **0.7906 – 0.7911**, thiết lập đỉnh hiệu năng mới.

### 6.4. Bài học từ các Thử nghiệm Kỹ thuật Mở rộng
1. **Ràng buộc đơn điệu (Monotonic constraints):**
   - Trên LightGBM, monotonic constraints không làm giảm AUC (0.7906 vs 0.7898), thời gian train gần như không đổi (~14s). Đảm bảo tính đơn điệu nghiệp vụ: khách hàng trễ hạn nhiều hơn hoặc dùng hạn mức cao hơn sẽ không bao giờ bị giảm xác suất rủi ro.
   - Trên CatBoost, monotonic constraints làm thời gian huấn luyện tăng gấp 5 lần (~318s) mà không tăng thêm AUC.
2. **Xử lý mất cân bằng (Class Weighting vs SMOTE):**
   - `class_weight`: Giúp Recall vọt lên ~61% ở ngưỡng 0.5, nhưng làm Brier score tăng vọt từ 0.133 lên 0.168 (méo xác suất nghiêm trọng).
   - `SMOTE in-fold`: Không cải thiện AUC (0.774–0.777), làm tăng thời gian huấn luyện gấp đôi. Khẳng định kết luận Charter: không dùng SMOTE cho mô hình chính thức.
3. **Đối chiếu Biến nhạy cảm SEX:**
   - Việc thêm biến SEX vào mô hình chỉ làm thay đổi AUC trung bình $+0.0001$, tỷ trọng đóng góp của SEX trong feature importance chỉ đạt $0.17\%$ (LightGBM) và $0.54\%$ (CatBoost).
   - Khẳng định tính đúng đắn của chính sách Charter: **loại bỏ biến SEX khỏi mô hình chính thức mà không làm suy giảm hiệu năng**.

---

## 7. Khuyến nghị cho Cuộc họp Review Tuần 2 (Chốt 1–2 Mô hình Cuối)

Thành viên B (ML Engineer) đề xuất với nhóm tại cuộc họp Review Tuần 2:

1. **Ứng viên Mô hình Chính (Primary Model): `LightGBM tuned + Monotonic constraints`**
   - **Lý do lựa chọn:**
     - Hiệu năng phân tách xuất sắc: ROC-AUC **0.7906 ± 0.0074**, vượt mốc mục tiêu Charter 0.79.
     - Tính nhất quán nghiệp vụ: 11 biến trễ hạn và tỷ lệ sử dụng hạn mức được bảo đảm đơn điệu tăng tuyệt đối.
     - Tốc độ vượt trội: Thời gian huấn luyện 5 fold chỉ mất **14 giây** (nhanh hơn CatBoost 5 lần và nhanh hơn CatBoost monotonic 22 lần).
     - Tương thích giải thích SHAP: Kiến trúc tương thích hoàn hảo với `shap.TreeExplainer` cho live demo Streamlit mà không lo giật lag.
2. **Ứng viên Mô hình Đối chứng (Benchmark / Secondary): `CatBoost tuned`**
   - Giữ làm mô hình đối chứng năng lực thuật toán (AUC cao nhất 0.7911) và kiểm tra độ ổn định trên tập Valid/Test.
3. **Mô hình Chuẩn mực Nghiệp vụ: `Logistic Scorecard (WoE)`**
   - Giữ nguyên trong pipeline demo để người dùng có thể so sánh song song giữa mô hình Black-box hiện đại (LightGBM) và Bảng điểm truyền thống chuẩn ngân hàng (Scorecard).

---

## 8. Lệnh Tái lập & Kiểm thử

Để hiển thị lại bảng so sánh sơ bộ hoặc xuất lại báo cáo:
```bash
python scripts/compare_models.py
# hoặc
python scripts/run_pipeline.py --mode compare
```

File dữ liệu máy đọc (machine-readable) được lưu tại: `reports/model_comparison.csv`.
