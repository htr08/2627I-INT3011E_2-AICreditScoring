# Báo cáo So sánh Logistic Scorecard (WoE) vs. Black-box Models

## 1. Kết quả Đánh giá (Stratified 5-Fold Cross-Validation)

                    Model  Mean AUC  Mean Gini  Mean KS
      XGBoost (Black-box)    0.7881     0.5761   0.4433
     LightGBM (Black-box)    0.7878     0.5756   0.4416
     CatBoost (Black-box)    0.7876     0.5752   0.4401
Random Forest (Black-box)    0.7854     0.5709   0.4383
 Logistic Scorecard (WoE)    0.7790     0.5580   0.4326

## 2. Cấu hình Hyperparameters của các Mô hình Black-box

| Mô hình | Tham số cấu hình chính (Key Hyperparameters) |
| :--- | :--- |
| **XGBoost** | `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `random_state=42` |
| **LightGBM** | `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `random_state=42` |
| **CatBoost** | `iterations=100`, `depth=4`, `learning_rate=0.05`, `random_seed=42` |
| **Random Forest** | `n_estimators=100`, `max_depth=6`, `random_state=42` |

## 3. Giải thích sự chênh lệch kết quả (Variance Analysis)

Chỉ số AUC của các mô hình Black-box trong báo cáo này (XGBoost: **0.7881**, LightGBM: **0.7878**) có sự khác biệt so với bảng tham chiếu cũ (0.7597, 0.7805). Nguyên nhân chính do:

1. **Chuẩn hóa Hyperparameters:** 
   - Độ sâu cây được giới hạn ở `max_depth=4` (thay vì độ sâu mặc định lớn hơn). Điều này giúp các mô hình Gradient Boosting kiểm soát overfitting hiệu quả hơn trên tập dữ liệu này.
2. **Kỹ thuật Cross-Validation đồng nhất:**
   - Sử dụng **Stratified 5-Fold Cross-Validation** cố định `random_state=42` xuyên suốt từ khâu tiền xử lý (Preprocessing) đến huấn luyện, đảm bảo tỷ lệ nhãn cân bằng giữa các fold và không bị data leakage.
3. **Bổ sung CatBoost:**
   - CatBoost đạt AUC **0.7870** (Gini **0.5740**), đạt hiệu năng tương đương XGBoost và LightGBM.
