# Báo cáo Thực nghiệm: Tuning Siêu tham số bằng Optuna

_Tuần 2 – T4: Thành viên B (ML Engineer)_

---

## 1. Mục tiêu và Phạm vi

Theo phân công Tuần 2 – T4 trong `reports/project_plan.md`:

1. Tuning siêu tham số bằng Optuna với 5-fold Stratified CV cho 2 mô hình tốt nhất sau T3: **CatBoost** (CV AUC 0.7853) và **LightGBM** (0.7813).
2. Thử nghiệm **monotonic constraints** cho các biến trễ hạn và tỷ lệ sử dụng hạn mức.
3. Log MLflow dạng **nested runs** (một run cha cho mỗi study, mỗi trial là một run con).
4. Áp dụng **điều kiện dừng** trong Project Charter mục 1.3: dừng khi CV ROC-AUC không cải thiện quá 0.002 sau 30 trial liên tiếp, hoặc hết ngân sách 2 giờ cho mỗi mô hình.

Toàn bộ thực nghiệm dùng tập Train (18.000 mẫu, 22,1% vỡ nợ), bộ đặc trưng `feature_freeze_v1` (không có SEX), cùng bộ chia fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` với T1–T3. Tập Valid và Test không được sử dụng.

---

## 2. Thiết lập Thực nghiệm

### 2.1. Không gian tìm kiếm

| Mô hình | Siêu tham số | Miền giá trị |
|---|---|---|
| LightGBM | `n_estimators` | 100–2000 (bước 50) |
| | `learning_rate` | 0.005–0.2 (log) |
| | `num_leaves` | 8–128 (log) |
| | `min_child_samples` | 5–300 (log) |
| | `subsample` (`subsample_freq=1`) | 0.5–1.0 |
| | `colsample_bytree` | 0.3–1.0 |
| | `reg_alpha`, `reg_lambda` | 1e-8–10 (log) |
| CatBoost | `iterations` | 200–1500 (bước 100) |
| | `learning_rate` | 0.01–0.2 (log) |
| | `depth` | 4–8 |
| | `l2_leaf_reg` | 1–30 (log) |
| | `random_strength` | 0.01–10 (log) |
| | `subsample` (MVS bootstrap) | 0.5–1.0 |

- **Sampler:** `TPESampler(seed=42)`. Trial đầu tiên của mỗi study được enqueue bằng tham số mặc định để làm mốc trong cùng study.
- **Pruner:** `MedianPruner(n_startup_trials=10, n_warmup_steps=2)`, giá trị trung gian là AUC trung bình tích lũy sau mỗi fold. Trial bị prune được tính là trial không cải thiện trong điều kiện dừng.
- **Không dùng class weighting:** kết quả T3 cho thấy `class_weight` không thay đổi ROC-AUC (chênh lệch < 0.005) nhưng làm Brier score tăng từ 0.134 lên 0.166–0.168. Mục tiêu tuning là ROC-AUC, và mô hình không trọng số giữ xác suất thô gần với tỷ lệ vỡ nợ thực tế, thuận lợi cho bước Platt scaling ở Tuần 3.
- **Không dùng early stopping trên fold val:** số cây được tuning trực tiếp như một siêu tham số, tránh việc fold val vừa dùng để dừng huấn luyện vừa dùng để chấm điểm.

### 2.2. Monotonic constraints

Ràng buộc đơn điệu tăng (PD không giảm khi giá trị biến tăng) áp dụng cho 11 biến numeric, khai báo tại `configs/config.yaml` (`tuning.monotone_increasing`):

- **Trễ hạn:** `PAY_MEAN`, `PAY_MAX`, `PAY_LATE_CONSECUTIVE`, `PAY_LATE_2PLUS_COUNT`.
- **Tỷ lệ sử dụng hạn mức:** `UTIL_1` … `UTIL_6`, `UTIL_MEAN`.

`PAY_1` … `PAY_6` không áp dụng được ràng buộc do được mã hoá one-hot theo `feature_freeze_v1`. `PAY_SLOPE` không được ràng buộc vì chiều tác động của độ dốc trễ hạn không có cơ sở nghiệp vụ rõ ràng.

Kế hoạch chạy được thiết kế theo chi phí tính toán của từng mô hình:

| Study | Monotonic constraints |
|---|---|
| `optuna_lightgbm` | Không; sau study chạy thêm 1 lượt CV bộ tham số tốt nhất có ràng buộc để đối chiếu |
| `optuna_lightgbm_monotone` | Có, áp dụng cho mọi trial |
| `optuna_catboost` | Không; sau study chạy thêm 1 lượt CV bộ tham số tốt nhất có ràng buộc để đối chiếu |

CatBoost không có study riêng với ràng buộc vì monotonic constraints làm thời gian huấn luyện tăng khoảng 5 lần (66 giây/trial → 318 giây/lượt CV), một study đầy đủ sẽ vượt ngân sách 2 giờ trước khi đạt điều kiện dừng.

### 2.3. Cache tiền xử lý theo fold

Pipeline tiền xử lý (làm sạch mã bất thường, tạo đặc trưng, age binning theo IV, one-hot, scaling) không có siêu tham số cần tuning. Pipeline này được fit một lần trên phần train của mỗi fold, transform phần val, và kết quả được dùng lại cho mọi trial. Fold val chỉ được transform nên không phát sinh data leakage; kết quả tương đương với việc fit lại toàn bộ Pipeline trong mỗi trial.

Số cột one-hot thay đổi giữa các fold (115–119 cột, do một số mức hiếm của `PAY_x` không xuất hiện trong mọi fold), nên vector monotonic constraints được dựng theo tên cột đầu ra của từng fold thay vì theo vị trí cố định.

---

## 3. Kết quả

### 3.1. Tổng quan các study

| Study | Số trial (prune) | Trial tốt nhất | Thời gian | Lý do dừng |
|---|---|---|---|---|
| `optuna_lightgbm` | 37 (4) | #28 | 13,7 phút | plateau (30 trial không cải thiện > 0.002) |
| `optuna_lightgbm_monotone` | 37 (3) | #23 | 10,5 phút | plateau |
| `optuna_catboost` | 34 (4) | #14 | 39,7 phút | plateau |

Cả ba study dừng theo điều kiện plateau, không study nào chạm ngân sách 2 giờ.

### 3.2. Hiệu năng CV (5-fold, bộ tham số tốt nhất)

| Mô hình | ROC-AUC (mean ± std) | Gini | KS | PR-AUC | Brier | Recall@0.5 | Precision@0.5 |
|---|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.7745 ± 0.0116 | 0.5490 | 0.4282 | – | 0.1354 | 0.3235 | 0.6728 |
| LightGBM default ¹ | 0.7805 ± 0.0077 | – | – | – | – | – | – |
| CatBoost default ¹ | 0.7844 ± 0.0074 | – | – | – | – | – | – |
| **LightGBM tuned** | 0.7898 ± 0.0077 | 0.5796 | 0.4439 | 0.5650 | 0.1327 | 0.3774 | 0.6696 |
| **LightGBM tuned + monotonic** | 0.7906 ± 0.0074 | 0.5813 | 0.4453 | 0.5624 | 0.1328 | 0.3774 | 0.6748 |
| **CatBoost tuned** | **0.7911 ± 0.0073** | **0.5822** | 0.4452 | **0.5675** | 0.1327 | 0.3674 | 0.6754 |
| CatBoost tuned + monotonic (đối chiếu) | 0.7910 ± 0.0077 | 0.5820 | 0.4434 | 0.5632 | 0.1328 | 0.3722 | 0.6675 |
| LightGBM tuned + monotonic (đối chiếu) ² | 0.7900 ± 0.0074 | 0.5800 | 0.4453 | 0.5633 | 0.1328 | 0.3717 | 0.6729 |

¹ Tính lại trên cùng fold cache để so sánh từng cặp fold. Kết quả lệch ≤ 0.001 so với số liệu báo cáo T3 (0.7813 và 0.7853); Logistic Regression khớp hoàn toàn (0.7745).
² Bộ tham số tốt nhất của study không ràng buộc, thêm monotonic constraints.

### 3.3. So sánh từng cặp theo fold

Do mọi mô hình dùng chung 5 fold, chênh lệch AUC theo từng fold phản ánh khác biệt giữa các mô hình chính xác hơn so với việc so sánh mean ± std độc lập.

| So sánh | Chênh lệch AUC theo fold | Trung bình | Nhỏ nhất |
|---|---|---|---|
| CatBoost tuned − CatBoost default | +0.0069, +0.0070, +0.0071, +0.0060, +0.0064 | +0.0067 | +0.0060 |
| LightGBM tuned − LightGBM default | +0.0121, +0.0090, +0.0095, +0.0110, +0.0050 | +0.0093 | +0.0050 |
| LightGBM monotonic − LightGBM default | +0.0139, +0.0084, +0.0110, +0.0135, +0.0042 | +0.0102 | +0.0042 |
| CatBoost tuned − Logistic baseline | +0.0145, +0.0091, +0.0196, +0.0236, +0.0162 | +0.0166 | +0.0091 |
| LightGBM monotonic − Logistic baseline | +0.0144, +0.0077, +0.0185, +0.0244, +0.0157 | +0.0161 | +0.0077 |
| CatBoost tuned − LightGBM monotonic | +0.0001, +0.0015, +0.0011, −0.0008, +0.0004 | +0.0005 | −0.0008 |
| CatBoost tuned − LightGBM tuned | +0.0019, +0.0008, +0.0026, +0.0016, −0.0004 | +0.0013 | −0.0004 |

### 3.4. Bộ tham số tốt nhất

| Tham số | LightGBM | LightGBM + monotonic | Tham số | CatBoost |
|---|---|---|---|---|
| `n_estimators` | 1450 | 1000 | `iterations` | 1100 |
| `learning_rate` | 0.0050 | 0.0078 | `learning_rate` | 0.0111 |
| `num_leaves` | 11 | 10 | `depth` | 6 |
| `min_child_samples` | 24 | 12 | `l2_leaf_reg` | 16.76 |
| `subsample` | 0.546 | 0.737 | `random_strength` | 3.475 |
| `colsample_bytree` | 0.729 | 0.498 | `subsample` | 0.997 |
| `reg_alpha` | 1.4e-7 | 1.6e-3 | | |
| `reg_lambda` | 8.2e-5 | 8.87 | | |

Bộ tham số giá trị đầy đủ (không làm tròn) được lưu tại tham số `best_*` của run cha trên MLflow.

---

## 4. Phân tích

### 4.1. Hiệu quả của tuning

- Tuning cải thiện ROC-AUC trên **cả 5 fold** cho cả hai mô hình: CatBoost +0.0067 (dao động 0.0060–0.0071), LightGBM +0.0093 (0.0050–0.0121) so với tham số mặc định.
- Mức cải thiện nhỏ hơn độ lệch chuẩn giữa các fold (~0.007), nhưng tính nhất quán dương trên mọi fold cho thấy cải thiện không do nhiễu chia fold.
- Hai mô hình hội tụ về cùng một dạng cấu hình: **learning rate thấp (0.005–0.011), nhiều cây (1000–1450), cây nông** (LightGBM 10–11 lá; CatBoost depth 6) và **regularization mạnh** (CatBoost `l2_leaf_reg` ≈ 17, LightGBM monotonic `reg_lambda` ≈ 8.9). Điều này phù hợp với đặc điểm dữ liệu: tín hiệu tập trung ở nhóm biến trễ hạn, phần còn lại nhiễu cao, mô hình phức tạp dễ overfit.
- Trial tốt nhất xuất hiện ở trial #14–#28; 30 trial tiếp theo không cải thiện quá 0.002, xác nhận vùng tham số hiện tại gần mức trần của bộ đặc trưng `feature_freeze_v1`.

### 4.2. Monotonic constraints

- **Không làm giảm khả năng phân tách:** LightGBM study có ràng buộc đạt 0.7906, cao hơn bản không ràng buộc (0.7898). Khi áp ràng buộc lên bộ tham số tốt nhất đã có, CatBoost giữ 0.7910 (so với 0.7911) và LightGBM đạt 0.7900 (so với 0.7898). Các chênh lệch đều dưới 0.001.
- **Lợi ích về khả năng giải thích:** ràng buộc bảo đảm PD không giảm khi mức trễ hạn hoặc tỷ lệ sử dụng hạn mức tăng (đã kiểm tra trong `tests/test_tune.py::test_monotone_constraint_respected`). Hồ sơ trễ hạn nặng hơn sẽ không nhận PD thấp hơn ở các biến được ràng buộc, đáp ứng yêu cầu nhất quán nghiệp vụ của reason codes và tránh các điểm đảo chiều khó giải thích.
- **Giới hạn:** điểm không đơn điệu của `PAY_1` ghi nhận trong phân tích SHAP dependence (`notebooks/03_shap_dependence.ipynb`) không được xử lý, do `PAY_1`–`PAY_6` là biến one-hot. Phương án mã hoá lại `PAY_1`–`PAY_6` để áp được ràng buộc đã được đánh giá ở mục 8: không cải thiện AUC.
- **Chi phí:** với CatBoost, ràng buộc làm thời gian huấn luyện tăng khoảng 5 lần; với LightGBM, thời gian không tăng đáng kể.

### 4.3. So sánh CatBoost và LightGBM

- CatBoost tuned đạt AUC trung bình cao nhất (0.7911), nhưng chênh lệch với LightGBM monotonic chỉ +0.0005 và đổi dấu ở 1/5 fold. **Hai mô hình không phân biệt được về khả năng phân tách** ở mức CV.
- Brier score của các mô hình tuned gần như bằng nhau (0.1327–0.1328) và thấp hơn baseline Logistic (0.1354).
- Chi phí tính toán khác biệt rõ: LightGBM ~12–17 giây/lượt 5-fold, CatBoost ~66 giây (không ràng buộc) và ~318 giây (có ràng buộc).

### 4.4. Đối chiếu KPI trong Project Charter

KPI chính thức được đo trên tập Test (Tuần 3 – T3); bảng sau chỉ mang tính tham khảo trên CV.

| Chỉ số | Mốc sàn / mục tiêu | CatBoost tuned | LightGBM + monotonic | Đánh giá (CV) |
|---|---|---|---|---|
| ROC-AUC | ≥ 0.78 / 0.79 | 0.7911 | 0.7906 | Đạt mục tiêu |
| Gini | ≥ 0.57 / 0.59 | 0.5822 | 0.5813 | Đạt sàn, chưa đạt mục tiêu |
| KS | ≥ 0.45 / 0.47 | 0.4452 | 0.4453 | Chưa đạt sàn (thiếu ~0.005) |
| PR-AUC | ≥ 0.56 / 0.58 | 0.5675 | 0.5624 | Đạt sàn, chưa đạt mục tiêu |

- **KPI tương đối** (vượt baseline Logistic có ý nghĩa thống kê): cả hai mô hình cao hơn baseline trên cả 5 fold (+0.0077 đến +0.0244). Kiểm định chính thức bằng DeLong test / bootstrap CI sẽ được thực hiện trên tập Test.
- KS chưa đạt mốc sàn trên CV (thiếu ~0.005), trong khi độ lệch chuẩn của KS giữa các fold là ~0.013–0.015. Theo điều kiện dừng trong Charter, tuning tiếp không được kỳ vọng cải thiện chỉ số này; các phương án thay đổi bộ đặc trưng (mục 8) cũng không cải thiện KS vượt mức nhiễu.

---

## 5. Kiến trúc Mã nguồn

1. **`src/tune.py`** (mới):
   - `suggest_lightgbm`, `suggest_catboost`, `DEFAULT_TRIAL_PARAMS`: không gian tìm kiếm và trial mốc.
   - `build_estimator`: tạo estimator từ bộ tham số, nhận vector monotonic constraints.
   - `monotone_vector`: ánh xạ danh sách biến ràng buộc sang vector theo tên cột đầu ra ColumnTransformer.
   - `prepare_folds`, `cv_score`: cache tiền xử lý theo fold và đánh giá CV với báo cáo giá trị trung gian cho pruner.
   - `PlateauStopper`: callback Optuna thực thi điều kiện dừng của Charter.
   - `tune_model`, `tune_best_models`: chạy study, log MLflow nested runs, fit lại mô hình tốt nhất trên toàn bộ Train và đăng ký vào Model Registry.
2. **`configs/config.yaml`**: thêm mục `tuning` (patience, min_delta, timeout, cấu hình pruner, danh sách biến ràng buộc đơn điệu).
3. **`scripts/run_pipeline.py`**: thêm `--mode tune` cùng các tuỳ chọn `--tune-model`, `--monotone`, `--max-trials`, `--timeout`.
   **`scripts/run_feature_experiment.py`** (mới): thí nghiệm bộ đặc trưng ở mục 8.
4. **`tests/test_tune.py`** (12 test): trial mặc định nằm trong không gian tìm kiếm, ánh xạ ràng buộc chỉ khớp khối numeric, tính đơn điệu của dự đoán với LightGBM và CatBoost, hành vi `PlateauStopper`, cấu trúc nested runs và Model Registry, CLI dispatch.

**Tổ chức trên MLflow** (experiment `credit_scoring`):

- Run cha `optuna_<study>`: tham số cấu hình study, `best_*`, metrics CV mean/std của trial tốt nhất, `roc_auc_fold` theo step, `n_trials`, `n_pruned`, `elapsed_seconds`, tag `stop_reason`, artifact `optuna_trials.csv` và mô hình đã fit trên toàn bộ Train.
- Run con `trial_NNN`: tham số trial, metrics CV mean/std (trial hoàn tất), `fit_seconds`, tag `trial_state`.
- Run con `best_params_monotone_check`: lượt đối chiếu monotonic constraints.
- Model Registry: `lightgbm_tuned`, `lightgbm_monotone_tuned`, `catboost_tuned`, alias `tuned`. Bước `clf` của Pipeline là estimator gốc (LGBMClassifier / CatBoostClassifier), dùng trực tiếp được với `shap.TreeExplainer`.

Lệnh tái lập: `python scripts/run_pipeline.py --mode tune` (khoảng 70 phút trên CPU 8 luồng).

---

## 6. Hạn chế

- **Trial mốc của CatBoost không phải tham số mặc định thực tế:** trial #0 dùng `learning_rate = 0.05`, trong khi CatBoost tự chọn 0.032 khi không khai báo. Trial #0 đạt 0.7793, thấp hơn CatBoost default thực tế (0.7844). So sánh tuned − default ở mục 3.3 sử dụng mô hình default thực tế.
- **Đánh giá trên CV, cùng dữ liệu dùng để chọn tham số:** AUC của trial tốt nhất mang thiên lệch lạc quan nhẹ do được chọn từ 34–37 trial. Mức thiên lệch này nhỏ so với mức cải thiện quan sát được trên mọi fold, và sẽ được kiểm chứng trên Valid (Tuần 3 – T2) và Test (Tuần 3 – T3).
- **Một seed cho sampler và chia fold:** kết quả chưa được kiểm tra độ ổn định với seed khác.

---

## 7. Khuyến nghị cho Tuần 2 – T5 và Tuần 3

1. **Ứng viên mô hình cuối:** đề xuất **LightGBM + monotonic constraints** (`lightgbm_monotone_tuned`) làm ứng viên chính. Lý do: AUC tương đương CatBoost (chênh lệch +0.0005, không phân biệt được), có ràng buộc đơn điệu nhất quán với nghiệp vụ, huấn luyện nhanh hơn ~5–25 lần và tương thích tốt với `TreeExplainer` cho demo. **CatBoost tuned** (`catboost_tuned`) giữ vai trò ứng viên đối chứng. Quyết định chọn 1–2 mô hình cuối thuộc buổi họp review Tuần 2 (T6).
2. **T5 – phiên bản có SEX:** huấn luyện lại ứng viên chính với cùng bộ tham số tốt nhất và `include_sex=True` để phục vụ đối chiếu fairness; không tuning lại.
3. **SHAP (thành viên C):** chạy lại `03_shap.ipynb` và `03_shap_dependence.ipynb` trên mô hình tuned, do cấu hình cây nông và ràng buộc đơn điệu có thể thay đổi dạng quan hệ đã quan sát trên XGBoost default.
4. **Calibration (Tuần 3 – T2):** Platt scaling fit tay trên `raw_score=True` (LightGBM) hoặc `prediction_type="RawFormulaVal"` (CatBoost), tương ứng với `output_margin=True` trong quy ước mục 3.5e.
5. **Bộ đặc trưng:** giữ feature freeze v1 (mục 8). Không cần tuning lại hay chạy lại SHAP do thay đổi đặc trưng.

---

## 8. Thí nghiệm Bộ đặc trưng (sau tuning)

### 8.1. Mục tiêu

Kiểm tra liệu thay đổi bộ đặc trưng có cải thiện hiệu năng vượt mức của tuning hay không, làm căn cứ cho quyết định giữ hoặc điều chỉnh feature freeze v1. Các phương án được chọn theo các điểm yếu đã ghi nhận: `PAY_1`–`PAY_6` mã hoá one-hot không áp được monotonic constraints và cho dạng quan hệ không đơn điệu trong SHAP dependence; nhóm biến tương quan cao trong phân tích lọc đặc trưng của thành viên A.

### 8.2. Thiết lập

Cùng 5 fold, cùng bộ tham số tốt nhất ở mục 3.4 (LightGBM + monotonic; CatBoost không ràng buộc), chỉ thay đổi bộ đặc trưng:

| Cấu hình | Mô tả |
|---|---|
| `v1_onehot` | Feature freeze v1 (chính thức) |
| `v1_decorr` | v1 bỏ 11 biến tương quan cao (Spearman > 0.9): `PAY_LATE_CONSECUTIVE`, `BILL_STD`, `UTIL_1`–`UTIL_5`, `BILL_AMT1`, `BILL_AMT3`, `BILL_AMT4`, `BILL_AMT5` |
| `ordinal_raw` | `PAY_1`–`PAY_6` dạng số thứ tự thô (−2 … 8) |
| `split` | `PAY_i` tách thành `PAY_i_DELAY` = max(`PAY_i`, 0) (ràng buộc đơn điệu tăng) và 2 cờ trạng thái −2 (không phát sinh dư nợ), −1 (trả đủ) |
| `split_new` | `split` + 6 đặc trưng mới: số tháng kể từ lần trễ gần nhất, `PAY_1` − `PAY_2`, số tháng không phát sinh dư nợ, số tháng trả đủ, trung bình `PAY_AMT` / `LIMIT_BAL`, `UTIL_1` − `UTIL_6` |

Cấu hình `split` xuất phát từ tỷ lệ vỡ nợ theo `PAY_1` trên Train: 12,5% (−2), 16,5% (−1), 12,6% (0), 34,4% (1), 69,5% (2), 78,7% (3). Quan hệ không đơn điệu ở vùng −2 … 0 và tăng mạnh từ mức 1, nên mã hoá thứ tự thô với ràng buộc đơn điệu sẽ sai ở vùng −2 … 0; tách thành số tháng trễ (đơn điệu) và cờ trạng thái giữ được cả hai dạng quan hệ.

### 8.3. Kết quả (ROC-AUC, 5-fold)

| Cấu hình | LightGBM + monotonic | Δ theo fold so với v1 | CatBoost | Δ theo fold so với v1 |
|---|---|---|---|---|
| `v1_onehot` | 0.7905 ± 0.0076 | – | 0.7906 ± 0.0080 | – |
| `v1_decorr` | 0.7894 ± 0.0070 | −0.0011 (5/5 fold thấp hơn) | 0.7896 ± 0.0072 | −0.0010 (4/5 fold thấp hơn) |
| `ordinal_raw` | 0.7897 ± 0.0069 | −0.0008 | 0.7899 ± 0.0080 | −0.0007 |
| `split` | 0.7905 ± 0.0073 | +0.0000 | 0.7903 ± 0.0079 | −0.0003 |
| `split_new` | 0.7900 ± 0.0079 | −0.0005 | 0.7904 ± 0.0078 | −0.0002 |

KS của mọi cấu hình nằm trong khoảng 0.4445–0.4472, PR-AUC trong khoảng 0.5625–0.5670; chênh lệch so với v1 đều nhỏ hơn độ lệch chuẩn giữa các fold.

### 8.4. Kết luận

- **Không phương án nào cải thiện khả năng phân tách.** Mọi chênh lệch AUC nằm trong khoảng −0.0011 đến 0.0000, nhỏ hơn nhiều so với độ lệch chuẩn giữa các fold (~0.007). Bộ đặc trưng v1 đã khai thác gần hết tín hiệu của dữ liệu; các biến dẫn xuất từ T2 (`PAY_MAX`, `PAY_MEAN`, `PAY_LATE_*`) đã chứa thông tin thứ tự mà mã hoá one-hot của `PAY_x` bỏ qua.
- **Lọc tương quan làm giảm AUC nhất quán** với mô hình cây, nên không áp dụng vào pipeline chính thức (thống nhất với `reports/feature_selection_decision.md`).
- `split` đạt AUC tương đương v1 với ít cột hơn (73 so với 115–119) và cho phép ràng buộc đơn điệu trên số tháng trễ của `PAY_1`–`PAY_6`. Đây là lợi ích về khả năng giải thích, không phải hiệu năng; áp dụng đòi hỏi tuning lại, chạy lại SHAP và cập nhật `feature_groups`. Với chi phí này và hiệu năng không đổi, **feature freeze v1 được giữ nguyên**.

Lệnh tái lập: `python scripts/run_feature_experiment.py --model lightgbm` và `--model catboost` (khoảng 3 và 9 phút).
