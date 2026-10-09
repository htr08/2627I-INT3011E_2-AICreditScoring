# Quyết định Lọc Đặc trưng — Feature Freeze v1

_Tuần 2 – T4: Thành viên A (Data Engineer / Analyst)_

---

## 1. Kết luận

**Feature freeze v1 được giữ nguyên làm bộ đặc trưng chính thức.** Lọc tương quan và các phương án mã hoá / đặc trưng bổ sung đã được đánh giá trên cùng 5 fold Stratified CV của tập Train; không phương án nào cải thiện khả năng phân tách, và lọc tương quan làm giảm AUC của mô hình cây trên mọi fold. Lọc tương quan được giữ ở dạng tuỳ chọn (`--feature-selection`) cho thí nghiệm scorecard, không áp dụng vào pipeline chính thức.

| Hạng mục | Quyết định |
|---|---|
| `ID` | Loại (mã định danh) |
| `SEX` | Loại khỏi mô hình chính thức; chỉ dùng ở bản đối chiếu fairness (Charter mục 1.3) |
| Biến gốc + biến dẫn xuất T2 | Giữ theo feature freeze v1 (`configs/feature_freeze_v1.yaml`) |
| Lọc IV < 0.02 | Chỉ áp dụng trong Logistic Scorecard (bước `woe_iv`), fit trong từng fold |
| Lọc tương quan > 0.9 | Không áp dụng vào pipeline chính thức; tuỳ chọn cho thí nghiệm scorecard |

---

## 2. Phương pháp

- Toàn bộ phân tích dùng tập Train (18.000 mẫu) và `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, cùng fold với các mô hình khác.
- Bộ đặc trưng được dựng bằng `build_feature_frame_pipeline` (làm sạch mã bất thường → biến dẫn xuất T2 → AGE_BIN theo IV → bỏ SEX/ID), fit lại trên phần train của từng fold để AGE_BIN không dùng dữ liệu ngoài fold.
- **Tương quan:** Spearman, ngưỡng |ρ| > 0.9; trong mỗi cặp giữ biến có IV cao hơn. Pearson không được dùng do các biến tỷ lệ (`PAY_RATIO_*`, `UTIL_*`) có đuôi rất dài: một số ít giá trị cực lớn đẩy Pearson của `PAY_RATIO_1` – `PAY_RATIO_3` lên 0.9997, trong khi quan hệ thứ hạng giữa hai biến không chặt như vậy.
- **Độ ổn định IV:** IV của từng biến trên phần train của từng fold (`reports/iv_stability.csv`).
- **Độ ổn định tầm quan trọng:** |hệ số| Logistic Regression trên biến numeric đã chuẩn hoá, sau lọc tương quan trong từng fold (`reports/feature_importance.csv`); cột `n_folds_selected` ghi số fold biến được giữ.

---

## 3. Kết quả

### 3.1. Cặp biến tương quan cao (Spearman > 0.9, toàn bộ Train)

| Biến bị loại | Giữ lại | ρ |
|---|---|---|
| `PAY_LATE_CONSECUTIVE` | `PAY_LATE_2PLUS_COUNT` | 0.999 |
| `BILL_STD` | `BILL_DELTA` | 0.972 |
| `UTIL_1`, `UTIL_2`, `UTIL_3`, `UTIL_4`, `UTIL_5` | `UTIL_MEAN` | 0.908–0.942 |
| `BILL_AMT1`, `BILL_AMT3` | `BILL_AMT2` | 0.904–0.910 |
| `BILL_AMT4` | `BILL_AMT5` | 0.903 |
| `BILL_AMT5` | `BILL_AMT6` | 0.900 |

Danh sách đầy đủ: `reports/correlation_drop.csv`. Các cặp trùng lặp này khớp với các cặp tương tác mạnh trong phân tích SHAP dependence (`notebooks/03_shap_dependence.ipynb`).

### 3.2. Độ ổn định IV qua 5 fold

- 49 biến; hệ số biến thiên của IV (std/mean) có trung vị 0.061, lớn nhất 0.301. Thứ hạng IV của nhóm biến mạnh không đổi giữa các fold (`PAY_1` 0.88–0.94, `PAY_LATE_2PLUS_COUNT` 0.84–0.89, `PAY_MAX` 0.80–0.84).
- 7 biến có IV < 0.02 ở cả 5 fold: `BILL_AMT1`–`BILL_AMT6`, `MARRIAGE`. Đây là các biến bị scorecard loại ổn định ở mọi fold.
- Một biến nằm sát ngưỡng: `MIN_PAY_FLAG_1` (IV 0.019–0.023), bị loại ở 1/5 fold. Quyết định giữ/loại biến này không ảnh hưởng đáng kể do IV thấp.

### 3.3. Độ ổn định tầm quan trọng (Logistic Regression)

- Các biến dẫn đầu ổn định trên cả 5 fold: `PAY_LATE_2PLUS_COUNT` (0.91 ± 0.01), `PAY_MAX` (0.34 ± 0.02), `LIMIT_BAL` (0.23 ± 0.02), `PAY_1`–`PAY_4` (0.17–0.19).
- Nhóm `BILL_AMT*` không ổn định: `BILL_AMT1`, `BILL_AMT3` chỉ được giữ ở 1/5 fold, `BILL_AMT5` ở 2/5 fold; hệ số dao động mạnh (vd `BILL_AMT2` 0.14–0.30). Đây là hệ quả của đa cộng tuyến trong nhóm dư nợ, phù hợp với khuyến nghị giải thích theo nhóm đặc trưng (project_plan mục 3.5e).

### 3.4. Ảnh hưởng của lọc đặc trưng đến hiệu năng (5-fold CV, ROC-AUC)

**Logistic Scorecard** (`python scripts/run_correlation_experiment.py`):

| Cấu hình | AUC (mean ± std) | KS | Số biến sau lọc | Chênh lệch AUC so với freeze v1 |
|---|---|---|---|---|
| freeze v1 (WoE + IV ≥ 0.02) | 0.7790 ± 0.0083 | 0.4326 | 41.0 | – |
| Lọc IV (FeatureSelectionTransformer) | 0.7790 ± 0.0083 | 0.4328 | 41.8 | 0.0000 |
| Lọc IV + tương quan | 0.7781 ± 0.0081 | 0.4342 | 34.8 | −0.0009 |

**Mô hình cây** (bộ tham số đã tuning ở T4 của thành viên B; `python scripts/run_feature_experiment.py`, chi tiết tại `reports/experiments_optuna_tuning.md` mục 8):

| Cấu hình | LightGBM + monotonic | CatBoost |
|---|---|---|
| freeze v1 | 0.7905 ± 0.0076 | 0.7906 ± 0.0080 |
| Bỏ 11 biến tương quan cao (mục 3.1) | 0.7894 (−0.0011, thấp hơn ở 5/5 fold) | 0.7896 (−0.0010, thấp hơn ở 4/5 fold) |

Với scorecard, lọc tương quan giảm 7 biến mà AUC gần như không đổi. Với mô hình cây, lọc tương quan làm giảm AUC nhất quán: các biến tương quan cao về thứ hạng vẫn mang thông tin bổ sung mà cây khai thác được (vd `UTIL_1` – tháng gần nhất so với `UTIL_MEAN`).

---

## 4. Tái lập

```
python scripts/run_pipeline.py --mode woe                     # IV, tương quan, độ ổn định → reports/*.csv
python scripts/run_correlation_experiment.py                  # bảng mục 3.4 (scorecard)
python scripts/run_pipeline.py --mode scorecard --feature-selection   # scorecard có lọc IV + tương quan, log MLflow
```

`reports/correlation_drop.csv`, `reports/iv_stability.csv`, `reports/feature_importance.csv` được commit; `reports/correlation_experiment_auc.csv` và `reports/correlation_feature_selection.csv` sinh lại bằng lệnh trên.
