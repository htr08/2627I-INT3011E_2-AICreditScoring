# Review Tuần 2: Lựa chọn mô hình cuối

_Tuần 2 – T6. Số liệu CV lấy từ MLflow._

## 1. Mục đích

Tài liệu tổng hợp kết quả Tuần 2 làm cơ sở thống nhất 1–2 mô hình cuối chuyển sang Tuần 3 (calibration, chọn ngưỡng, đánh giá Test, tích hợp Streamlit). Tập Test chưa được sử dụng ở bất kỳ bước nào.

## 2. Mô hình ứng viên

5-fold Stratified CV trên Train (18.000 dòng), mean ± std. Nguồn: MLflow (`logreg_baseline`, `logistic_scorecard`, `optuna_lightgbm_monotone`, `optuna_catboost`).

| Mô hình | ROC-AUC | Gini | KS | PR-AUC | Brier (thô) |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.7745 ± 0.0116 | 0.5490 | 0.4282 | 0.5470 | 0.1354 |
| Logistic Scorecard (WoE) | 0.7790 ± 0.0083 | 0.5580 | 0.4326 | 0.5432 | 0.1351 |
| LightGBM tuned + monotonic | 0.7906 ± 0.0074 | 0.5813 | 0.4453 | 0.5624 | 0.1328 |
| CatBoost tuned | 0.7911 ± 0.0073 | 0.5822 | 0.4452 | 0.5675 | 0.1327 |

## 3. So sánh cặp theo fold

Chênh lệch ROC-AUC trên cùng 5 fold, kiểm định paired t-test. Cỡ mẫu nhỏ và các fold không độc lập hoàn toàn, do đó kết quả mang tính sơ bộ; kết luận chính thức dựa trên bootstrap CI và DeLong test trên tập Test.

| Cặp | Δ AUC trung bình | Ghi chú | p |
|---|---|---|---|
| CatBoost tuned − LR | +0.0166 | 5/5 fold dương | 0.002 |
| LightGBM monotonic − LR | +0.0161 | 5/5 fold dương | 0.004 |
| CatBoost tuned − LightGBM monotonic | +0.0005 | đổi dấu ở 1/5 fold | 0.31 |
| Scorecard − LR | +0.0045 | 2/5 fold âm | 0.34 |

Hai mô hình boosting sau tuning vượt baseline rõ rệt và không phân biệt được với nhau về khả năng phân tách. Scorecard không vượt LR có ý nghĩa thống kê.

### 3.1. XGBoost và Random Forest

Hai mô hình chỉ được huấn luyện với tham số mặc định; bước tuning Tuần 2 – T4 áp dụng cho hai mô hình tốt nhất sau T3.

| Cặp | Δ AUC trung bình | Số fold dương | p |
|---|---|---|---|
| Random Forest − LR | −0.0078 | 0/5 | 0.024 |
| XGBoost − LR | −0.0148 | 0/5 | 0.006 |
| LightGBM default − XGBoost default | +0.0207 | 5/5 | 0.001 |
| CatBoost default − XGBoost default | +0.0247 | 5/5 | < 0.001 |
| LightGBM default − Random Forest default | +0.0138 | 5/5 | 0.003 |

Ở cấu hình mặc định, XGBoost (0.7597) và Random Forest (0.7667) thấp hơn LR trên mọi fold. Khoảng cách với LightGBM và CatBoost cùng cấu hình (0.014–0.025) lớn hơn nhiều độ lệch chuẩn giữa các fold (0.007–0.010). Do hai mô hình này chưa được tuning, phép so sánh với các mô hình đã tuning chưa ngang bằng. Tác động của tuning lên XGBoost chưa được đo.

## 4. Đối chiếu KPI tham chiếu (CV)

| Chỉ số (sàn / mục tiêu) | LightGBM monotonic | CatBoost tuned |
|---|---|---|
| ROC-AUC (0.78 / 0.79) | Đạt mục tiêu | Đạt mục tiêu |
| Gini (0.57 / 0.59) | Đạt sàn | Đạt sàn |
| PR-AUC (0.56 / 0.58) | Đạt sàn | Đạt sàn |
| KS (0.45 / 0.47) | Chưa đạt sàn (0.4453) | Chưa đạt sàn (0.4452) |

KPI chính (vượt baseline có ý nghĩa thống kê) được xác nhận bằng bootstrap CI trên Test ở Tuần 3.

## 5. Tiêu chí XAI và sản phẩm

| Tiêu chí | LightGBM monotonic | CatBoost tuned |
|---|---|---|
| Giải thích SHAP | TreeExplainer | TreeExplainer |
| Thời gian CV 5-fold (báo cáo tuning) | ~12–17 giây | ~66 giây |
| Thời gian toàn bộ study Optuna (MLflow) | 628 giây | 2.383 giây |
| Ràng buộc đơn điệu | Có, 11 biến (4 biến trễ hạn tổng hợp, 7 biến `UTIL`) | Không |
| Calibration Platt trên `output_margin` | Áp dụng được | Áp dụng được |

Ràng buộc đơn điệu có giới hạn: `PAY_1`–`PAY_6` là biến one-hot nên không được ràng buộc, và SHAP dependence ghi nhận `PAY_1` không đơn điệu.

Phân tích SHAP trên LightGBM monotonic: Global và dependence tại `03_shap_final_model.ipynb`, Local tại `03_shap_local.ipynb`. Kết quả chính:

- Nhóm lịch sử trễ hạn chiếm 48,7% mức độ quan trọng (XGBoost mặc định: 38,5%); nhóm hành vi trả nợ giảm từ hạng 2 xuống hạng 4.
- Tương tác yếu (cặp mạnh nhất `PAY_1` – `PAY_MAX` 0,048, XGBoost 0,233), đóng góp theo nhóm gần như cộng tính.
- Điểm không đơn điệu của `PAY_1` không ảnh hưởng reason codes theo nhóm: với `PAY_1` ≥ 4, đóng góp ròng của nhóm trễ hạn vẫn ở mức +1,4 đến +1,6 (log-odds).
- Thứ hạng theo biến khác đáng kể so với XGBoost (Spearman 0,73), nhóm đứng đầu không đổi; kết quả củng cố thiết kế reason codes theo nhóm.

## 6. Đề xuất

1. **Mô hình chính: LightGBM tuned + monotonic.** AUC tương đương CatBoost, thời gian huấn luyện thấp hơn, có bảo đảm đơn điệu cho reason codes, phù hợp với demo Streamlit.
2. **Mô hình đối chứng: CatBoost tuned.** AUC CV cao nhất, dùng để kiểm tra độ ổn định trên Valid và Test.
3. **Logistic Scorecard (WoE): mô hình tham chiếu minh bạch** cho so sánh trong demo và báo cáo, không thuộc nhóm mô hình cuối.
4. **XGBoost và Random Forest: không đưa vào nhóm mô hình cuối,** giữ trong bảng so sánh ở cấu hình mặc định kèm ghi chú chưa tuning. XGBoost thuộc MVP (Charter mục 2.1) và đã được huấn luyện, so sánh, dùng cho phân tích SHAP; MVP không quy định XGBoost là mô hình cuối. Tuning bổ sung cho XGBoost là tùy chọn.
5. **Bản có SEX** chỉ dùng cho đối chiếu fairness (chênh AUC +0.0001; SEX chiếm 0.17% và 0.54% tầm quan trọng); mô hình chính thức không dùng SEX.