# Báo cáo So sánh Logistic Scorecard (WoE) vs. Black-box Models

Kết quả sơ bộ trên 5-fold Stratified CV (tập Train, random_state=42, feature freeze v1). Kết quả các mô hình tham số mặc định và sau tuning: `experiments_boosting_imbalance.md`, `experiments_optuna_tuning.md`.

## 1. Kết quả Đánh giá

| Mô hình | AUC (mean ± std) | Gini | KS |
| :--- | :--- | :--- | :--- |
| XGBoost (Black-box) | 0.7881 ± 0.0067 | 0.5761 | 0.4433 |
| LightGBM (Black-box) | 0.7881 ± 0.0062 | 0.5761 | 0.4421 |
| CatBoost (Black-box) | 0.7883 ± 0.0067 | 0.5766 | 0.4413 |
| Random Forest (Black-box) | 0.7854 ± 0.0077 | 0.5709 | 0.4383 |
| Logistic Scorecard (WoE) | 0.7790 ± 0.0074 | 0.5580 | 0.4326 |

## 2. Cấu hình Hyperparameters của các Mô hình Black-box

Các mô hình black-box dùng bộ tham số cố định (cây nông, learning rate thấp), khác với cấu hình mặc định của thư viện và với bộ tham số tối ưu bằng Optuna.

| Mô hình | Tham số cấu hình chính (Key Hyperparameters) |
| :--- | :--- |
| **XGBoost** | `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `random_state=42` |
| **LightGBM** | `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `random_state=42` |
| **CatBoost** | `iterations=100`, `depth=4`, `learning_rate=0.05`, `random_seed=42` |
| **Random Forest** | `n_estimators=100`, `max_depth=6`, `random_state=42` |

## 3. Đối chiếu với kết quả mô hình tham số mặc định

| Mô hình | AUC báo cáo này | AUC tham số mặc định |
| :--- | :--- | :--- |
| XGBoost | 0.7881 | 0.7597 |
| LightGBM | 0.7881 | 0.7813 |
| CatBoost | 0.7883 | 0.7853 |
| Random Forest | 0.7854 | 0.7667 |

Chênh lệch AUC xuất phát từ cấu hình tham số: các mô hình mặc định dùng default thư viện (XGBoost `learning_rate=0.3`, cây sâu hơn; RF không giới hạn độ sâu), còn cấu hình trong báo cáo này dùng `max_depth=4`, `learning_rate=0.05` (RF `max_depth=6`). Hai bảng ứng với hai cấu hình khác nhau nên không thể so trực tiếp. Kết quả sau tuning Optuna: `experiments_optuna_tuning.md`.
