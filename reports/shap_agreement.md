# Mức đồng thuận về tầm quan trọng đặc trưng giữa XGBoost, LightGBM và Logistic Regression

Phân tích tại `notebooks/03_shap_agreement.ipynb`; hàm tính tại `src/explain.py`. Tập Test không được sử dụng._

## 1. Tóm tắt

- **Hai đầu bảng xếp hạng nhóm đặc trưng thống nhất ở cả bốn cấu hình mô hình.** Lịch sử trễ hạn luôn đứng đầu (38,5% đến 55,7% tổng mean(|SHAP|)) và nhân khẩu học luôn đứng cuối (2,3% đến 7,4%). Thứ tự ba nhóm giữa không thống nhất.
- **Mức đồng thuận phụ thuộc vào mức điều chuẩn của XGBoost.** XGBoost `max_depth=4` gần LightGBM (Spearman theo biến 0,92; tương quan PD 0,992), XGBoost mặc định lệch rõ hơn (0,73; 0,897) và vi phạm chiều tác động nghiệp vụ ở `UTIL_MEAN`.
- **Logistic Regression đồng thuận với mô hình cây ở mức nhóm (Spearman 0,9) nhưng không ở mức biến (0,30).** Dấu tác động của nhiều biến đảo ngược, và đóng góp của nhóm dư nợ tương quan âm với mô hình cây.
- **Reason code đầu tiên trùng nhau ở 96% hồ sơ rủi ro cao giữa LightGBM và XGBoost `max_depth=4`**, 76% với Logistic và 74% với XGBoost mặc định.
- Hệ quả: reason codes nên tính theo nhóm đặc trưng và trên LightGBM có ràng buộc đơn điệu; không nên diễn giải theo từng biến trên Logistic Regression.

## 2. Phạm vi và phương pháp

| Hạng mục | Giá trị |
|---|---|
| Mô hình | LightGBM tuned + monotonic (`lightgbm_monotone`, mô hình chính); XGBoost mặc định (`xgboost_default`); XGBoost `max_depth=4` (`build_blackbox_pipeline`); Logistic Regression (`logreg_baseline`) |
| Dữ liệu giải thích | 3.000 dòng ngẫu nhiên của tập Valid, `random_state = 42` |
| Explainer | `TreeExplainer` (`tree_path_dependent`) cho mô hình cây; công thức tuyến tính chính xác `φ = coef·(z − mean_Train)` cho Logistic, trùng với `shap.LinearExplainer` (sai lệch 0) |
| Thang | Log-odds chưa hiệu chuẩn; so sánh bằng tỷ trọng, thứ hạng và tương quan nên không phụ thuộc thang |
| Nhóm đặc trưng | Năm nhóm của `configs/config.yaml` (khóa `feature_groups`) |
| Kiểm tra | Sai số cộng tính `base + Σφ − đầu ra mô hình` ≤ 3,3·10⁻⁶ ở cả bốn mô hình |

Các chỉ số: tỷ trọng mean(|SHAP|) theo nhóm kèm khoảng tin cậy bootstrap 95% (500 lần, theo khách hàng); Spearman và Kendall tau của thứ hạng nhóm và thứ hạng 49 biến gốc; số biến trùng Top 10; tương quan Pearson giữa PD của hai mô hình; Spearman theo khách hàng giữa đóng góp của từng nhóm; tỷ lệ trùng nhóm làm tăng rủi ro mạnh nhất (reason code đầu tiên), trên toàn mẫu và trên 25% khách hàng có PD (LightGBM) cao nhất (750 hồ sơ); chiều tác động của biến (Spearman giữa giá trị biến và đóng góp SHAP của biến đó).

Platt scaling nhân mọi đóng góp SHAP với cùng một hệ số `a` (≈ 1,01), nên các tỷ trọng và thứ hạng dưới đây không đổi sau hiệu chuẩn.

## 3. Kết quả

### 3.1. Tỷ trọng theo nhóm đặc trưng

Tỷ trọng mean(|SHAP|), trong ngoặc là hạng; khoảng tin cậy 95% của mọi ô hẹp (rộng không quá 2 điểm phần trăm).

| Nhóm | LightGBM | XGBoost (mặc định) | XGBoost (`max_depth=4`) | Logistic |
|---|---|---|---|---|
| Lịch sử trễ hạn | 48,7% (1) | 38,5% (1) | 55,7% (1) | 54,5% (1) |
| Mức sử dụng hạn mức | 18,7% (2) | 16,7% (3) | 15,8% (2) | 18,0% (2) |
| Dư nợ và biến động dư nợ | 16,4% (3) | 15,4% (4) | 12,7% (4) | 8,3% (4) |
| Hành vi trả nợ | 13,5% (4) | 24,1% (2) | 13,6% (3) | 11,8% (3) |
| Nhân khẩu học | 2,8% (5) | 5,3% (5) | 2,3% (5) | 7,4% (5) |

- Thứ hạng của XGBoost `max_depth=4` và Logistic giống hệt nhau (Spearman nhóm = 1,0); LightGBM hoán đổi hạng 3 và 4 so với hai mô hình này.
- XGBoost mặc định gán cho hành vi trả nợ 24,1%, gần gấp đôi LightGBM (13,5%).
- Logistic gán cho nhân khẩu học 7,4%, cao hơn các mô hình cây (2,3% đến 5,3%).

### 3.2. Đồng thuận toàn cục theo cặp

| Cặp mô hình | Spearman (nhóm) | Kendall (nhóm) | Spearman (biến) | Trùng Top 10 biến | Tương quan PD |
|---|---|---|---|---|---|
| LightGBM – XGBoost (mặc định) | 0,70 | 0,60 | 0,73 | 4 | 0,897 |
| LightGBM – XGBoost (`max_depth=4`) | 0,90 | 0,80 | 0,92 | 8 | 0,992 |
| LightGBM – Logistic | 0,90 | 0,80 | 0,30 | 6 | 0,959 |
| XGBoost (mặc định) – XGBoost (`max_depth=4`) | 0,90 | 0,80 | 0,82 | 6 | 0,903 |
| XGBoost (mặc định) – Logistic | 0,90 | 0,80 | 0,02 | 2 | 0,856 |
| XGBoost (`max_depth=4`) – Logistic | 1,00 | 1,00 | 0,22 | 5 | 0,956 |

Với năm nhóm, Spearman và Kendall chỉ nhận số ít giá trị (một cặp hạng kề đổi chỗ làm Spearman giảm 0,1), nên chỉ số theo biến và theo khách hàng được dùng bổ sung.

### 3.3. Đồng thuận trên từng khách hàng

Spearman theo khách hàng giữa đóng góp của từng nhóm (3.000 hồ sơ):

| Nhóm | LGBM – XGB mặc định | LGBM – XGB `d=4` | LGBM – Logistic | XGB mặc định – Logistic | XGB `d=4` – Logistic |
|---|---|---|---|---|---|
| Lịch sử trễ hạn | 0,95 | 0,98 | 0,90 | 0,86 | 0,89 |
| Mức sử dụng hạn mức | 0,76 | 0,95 | 0,85 | 0,63 | 0,82 |
| Hành vi trả nợ | 0,76 | 0,95 | 0,54 | 0,45 | 0,56 |
| Nhân khẩu học | 0,68 | 0,88 | 0,82 | 0,53 | 0,66 |
| Dư nợ và biến động dư nợ | 0,67 | 0,94 | **−0,62** | **−0,40** | **−0,51** |

Tỷ lệ trùng reason code đầu tiên (nhóm làm tăng rủi ro mạnh nhất):

| Phạm vi | LGBM – XGB mặc định | LGBM – XGB `d=4` | LGBM – Logistic | XGB mặc định – Logistic | XGB `d=4` – Logistic |
|---|---|---|---|---|---|
| Toàn mẫu (3.000) | 66% | 88% | 49% | 44% | 51% |
| Rủi ro cao (750) | 74% | 96% | 76% | 70% | 78% |

Ở khách hàng rủi ro thấp, đóng góp dương của mọi nhóm đều nhỏ nên nhóm "mạnh nhất" ít ý nghĩa; tỷ lệ trùng trên nhóm rủi ro cao phản ánh đúng hơn mục đích sử dụng của reason codes.

### 3.4. Chiều tác động của biến

Spearman giữa giá trị biến và đóng góp SHAP của chính biến đó (dương: giá trị càng cao càng làm tăng rủi ro). Biến in đậm bị ràng buộc đơn điệu tăng trong LightGBM.

| Biến | LightGBM | XGBoost (mặc định) | XGBoost (`max_depth=4`) | Logistic |
|---|---|---|---|---|
| **`PAY_MAX`** | 0,68 | 0,66 | 0,68 | 1,0 |
| **`PAY_MEAN`** | 0,79 | −0,04 | 0,45 | −1,0 |
| **`PAY_LATE_2PLUS_COUNT`** | 0,79 | 0,43 | 0,30 | 1,0 |
| **`UTIL_MEAN`** | 0,81 | −0,33 | 0,63 | 1,0 |
| `LIMIT_BAL` | −0,95 | −0,77 | −0,95 | −1,0 |
| `BILL_AMT1` | −0,66 | −0,56 | −0,54 | 1,0 |
| `BILL_DELTA` | −0,64 | −0,31 | −0,77 | −1,0 |
| `PAY_AMT1` | −0,94 | −0,76 | −0,88 | −1,0 |
| `PAY_RATIO_MEAN` | −0,81 | −0,35 | −0,88 | 1,0 |

- LightGBM giữ chiều dương ở cả bốn biến bị ràng buộc (0,68 đến 0,81).
- XGBoost mặc định đảo chiều ở `UTIL_MEAN` (−0,33) và không có chiều rõ ràng ở `PAY_MEAN` (−0,04).
- Logistic đảo dấu so với mô hình cây ở `PAY_MEAN`, `BILL_AMT1` và `PAY_RATIO_MEAN`.

## 4. Nhận xét

1. **Kết luận về nhóm trễ hạn bền vững.** Nhóm này đứng đầu ở mọi mô hình, có Spearman theo khách hàng từ 0,86 đến 0,98 ở mọi cặp, và là nhóm chi phối trường hợp Cảnh báo cao được minh họa tại `03_shap_local.ipynb`.
2. **Sự khác biệt của XGBoost mặc định phần lớn đến từ siêu tham số.** Cùng thuật toán, `max_depth=4` đưa mức đồng thuận với LightGBM từ 0,73 lên 0,92 (Spearman theo biến) và từ 4/10 lên 8/10 (trùng Top 10). Việc XGBoost không thuộc nhóm mô hình cuối (`week2_review.md`) dựa trên cấu hình mặc định; bản `max_depth=4` đạt ROC-AUC CV khoảng 0,788 (gần LightGBM, 0,7906) nhưng chưa được tuning hay đăng ký trong registry.
3. **Logistic Regression không phù hợp để diễn giải theo biến.** Mức đồng thuận theo nhóm cao nhưng theo biến thấp, và dấu tác động đảo ngược ở các biến trong nhóm tương quan cao. Nguyên nhân khả dĩ là đa cộng tuyến (project_plan mục 3.5e), chưa được kiểm định riêng trong phân tích này.
4. **Nhóm dư nợ và biến động dư nợ là điểm bất đồng chính giữa họ tuyến tính và họ cây** (Spearman theo khách hàng từ −0,40 đến −0,62). Hướng đóng góp của nhóm này phụ thuộc mô hình, nên không được diễn giải như nguyên nhân độc lập trong reason codes.
5. **LightGBM có ràng buộc đơn điệu cho chiều tác động nhất quán với nghiệp vụ** ở các biến trễ hạn và mức sử dụng hạn mức, phù hợp làm mô hình giải thích trên demo.

## 5. Hạn chế

- Mẫu giải thích gồm 3.000 dòng của tập Valid; khoảng tin cậy chỉ phản ánh biến thiên theo khách hàng trong mẫu này, không phản ánh biến thiên theo cách chia dữ liệu hay theo hạt giống huấn luyện.
- Hai họ mô hình dùng hai định nghĩa kỳ vọng nền khác nhau (`tree_path_dependent` so với background Train), nên base value không so sánh trực tiếp; phép so sánh chỉ dùng tỷ trọng, thứ hạng và tương quan.
- Chiều tác động mô tả quan hệ trung bình trong mô hình, không phải quan hệ nhân quả, và chịu ảnh hưởng của tương quan giữa các biến.
- Định nghĩa "rủi ro cao" theo phân vị 75% của PD LightGBM, độc lập với ngưỡng cảnh báo cuối cùng (chưa chốt tại thời điểm viết).

## 6. Tái lập

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/03_shap_agreement.ipynb
pytest tests/test_explain.py
```

Notebook ghi các bảng và hình vào `reports/figures/` với tiền tố `shap_agreement_` (không commit). Các hàm `explain_pipeline`, `group_importance_share`, `bootstrap_share_ci`, `rank_agreement`, `top_k_overlap`, `contribution_agreement`, `top_group_agreement` và `effect_direction` nằm tại `src/explain.py`.
