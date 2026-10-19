# Báo cáo XAI và Hệ thống Lý do Cảnh báo (Reason Codes)

_Tuần 3 – T4: Thành viên C (Explainability & Product)_

---

## 1. Mục tiêu và phạm vi

Báo cáo tổng hợp cách mô hình cuối (LightGBM tuned + monotonic, Platt scaling) được giải thích bằng SHAP, định nghĩa hệ thống lý do cảnh báo (reason codes) hiển thị trên demo và các giới hạn khi diễn giải. Số liệu thống kê tính trên tập Valid (6.000 hồ sơ); tập Test không được sử dụng. Phân tích chi tiết nằm ở các notebook `03_shap_final_model.ipynb` (toàn cục, dependence), `03_shap_local.ipynb` (cục bộ) và `reports/shap_agreement.md` (đồng thuận giữa các mô hình).

---

## 2. Phương pháp giải thích

| Hạng mục | Nội dung |
| :--- | :--- |
| Explainer | `shap.TreeExplainer` (`tree_path_dependent`) trên margin thô của LightGBM; không dùng `KernelExplainer` (project_plan.md mục 3.5e). |
| Hiệu chuẩn | PD = sigmoid(a·margin + b), Platt fit trên Valid với a = 1.0128, b = 0.0364. Nhân SHAP với a và đặt base calibrated = a·φ₀ + b thì base + Σ đóng góp = logit(PD hiển thị). |
| Thang điểm | Điểm PDO (600 điểm tại odds 50:1, PDO = 20): đóng góp tính bằng −Δlogit × PDO / ln 2, tức 28.85 điểm cho mỗi đơn vị logit. Đóng góp âm làm giảm điểm, tức làm tăng rủi ro. Điểm nền (ứng với base value) là 532. |
| Đơn vị giải thích | Cột one-hot được cộng về biến gốc, biến gốc được cộng về 5 nhóm đặc trưng (`configs/config.yaml`, khóa `feature_groups`): Lịch sử trễ hạn, Mức sử dụng hạn mức, Hành vi trả nợ, Dư nợ và biến động dư nợ, Nhân khẩu học. |
| Kiểm tra | Sai số cộng tính tối đa 10⁻¹⁴ ở cấp cột và nhóm (notebooks); `tests/test_final_model.py` kiểm tra `base_logit + Σ đóng góp = logit(PD)` cho các hồ sơ mẫu. |
| Chi phí | Khoảng 2 giây cho 6.000 hồ sơ, dưới 1 ms mỗi hồ sơ; đủ nhanh cho giải thích trực tiếp trên demo. |

Platt chỉ nhân mọi đóng góp với cùng hệ số a (≈ 1.01) nên tỷ trọng và thứ hạng đóng góp không đổi so với thang margin. Calibrator Isotonic không được dùng cho mô hình cuối vì làm tổng SHAP không còn khớp PD.

---

## 3. Giải thích toàn cục

### 3.1. Theo nhóm đặc trưng

Tỷ trọng mean(|đóng góp|) trên Valid:

| Nhóm | Tỷ trọng |
| :--- | ---: |
| Lịch sử trễ hạn | 48.4% |
| Mức sử dụng hạn mức | 19.0% |
| Dư nợ và biến động dư nợ | 16.4% |
| Hành vi trả nợ | 13.4% |
| Nhân khẩu học | 2.8% |

Nhóm lịch sử trễ hạn đứng đầu và nhân khẩu học đứng cuối ở cả bốn cấu hình mô hình được so sánh trong `reports/shap_agreement.md` (tỷ trọng nhóm trễ hạn 38.5% đến 55.7%). Mức đóng góp thấp của nhân khẩu học nhất quán với chính sách không dùng SEX và chỉ dùng AGE dạng binning (Charter mục 8).

### 3.2. Theo biến

Bốn biến đầu đều thuộc nhóm trễ hạn: `PAY_MAX` 13.4%, `PAY_1` 9.9%, `PAY_LATE_CONSECUTIVE` 7.5%, `PAY_LATE_2PLUS_COUNT` 6.1%, tổng 36.8% mức độ quan trọng (`03_shap_final_model.ipynb`, mẫu 3.000 hồ sơ Valid).

### 3.3. Hình dạng quan hệ và tương tác

- 11 biến trễ hạn và tỷ lệ sử dụng hạn mức bị ràng buộc đơn điệu tăng, nên PD không giảm khi các biến này tăng. `PAY_1`–`PAY_6` là biến one-hot nên không ràng buộc được và `PAY_1` không đơn điệu; với `PAY_1` ≥ 4, đóng góp ròng của nhóm trễ hạn vẫn ở mức +1.4 đến +1.6 log-odds, nên reason codes theo nhóm không bị ảnh hưởng.
- Tương tác yếu: cặp mạnh nhất `PAY_1` – `PAY_MAX` có cường độ 0.048 (XGBoost mặc định: 0.233). Đóng góp theo nhóm gần như cộng tính.

### 3.4. Mức đồng thuận với mô hình khác

Reason code đầu tiên (nhóm làm tăng rủi ro mạnh nhất) trùng nhau ở 96% hồ sơ rủi ro cao giữa LightGBM và XGBoost `max_depth=4`, 76% với Logistic Regression và 74% với XGBoost mặc định. Ở cấp từng biến, Logistic đảo dấu nhiều biến tương quan cao (Spearman theo biến với LightGBM chỉ 0.30), nên reason codes dựa trên nhóm và trên mô hình cây có ràng buộc đơn điệu (`reports/shap_agreement.md`).

---

## 4. Hệ thống lý do cảnh báo (reason codes)

### 4.1. Định nghĩa

| Thành phần | Quy tắc |
| :--- | :--- |
| Đơn vị | Nhóm đặc trưng, không phải biến riêng lẻ: `PAY_1`–`PAY_6` và `BILL_AMT1`–`BILL_AMT6` tương quan rất cao nên SHAP phân bổ đóng góp giữa các biến trong nhóm khá tùy ý (project_plan.md mục 3.5e). |
| Thang | Điểm PDO, cộng đúng về điểm cuối: điểm nền + Σ đóng góp nhóm = điểm chưa clip. |
| Lý do làm giảm điểm | Tối đa 3 nhóm có đóng góp ≤ −1 điểm, xếp theo mức giảm nhiều nhất. |
| Yếu tố tích cực | Tối đa 3 nhóm có đóng góp ≥ +1 điểm, xếp theo mức tăng nhiều nhất. |
| Ngưỡng tối thiểu | Nhóm có |đóng góp| dưới 1 điểm không được coi là lý do (`MIN_REASON_POINTS` trong `app/explanation.py`). |
| Hành động | Theo mức cảnh báo của PD: Bình thường (PD < 9%), Theo dõi (9% ≤ PD < 27%), Cảnh báo cao (PD ≥ 27%); xem `reports/calibration_threshold.md` mục 6.3. |

### 4.2. Lý do đầu tiên theo mức cảnh báo (Valid)

Lý do đầu tiên là nhóm làm giảm điểm nhiều nhất của hồ sơ; tỷ lệ tính theo số hồ sơ của từng mức.

| | Bình thường | Theo dõi | Cảnh báo cao |
| :--- | ---: | ---: | ---: |
| Số hồ sơ | 1,652 | 2,799 | 1,549 |
| Tỷ lệ vỡ nợ thực tế | 6.2% | 15.5% | 51.0% |
| Số nhóm làm giảm điểm (trung bình) | 0.55 | 1.90 | 3.11 |
| Lịch sử trễ hạn | 0.0% | 9.9% | 78.7% |
| Mức sử dụng hạn mức | 9.1% | 44.5% | 2.9% |
| Dư nợ và biến động dư nợ | 9.3% | 33.1% | 18.3% |
| Hành vi trả nợ | 10.5% | 10.5% | 0.1% |
| Nhân khẩu học | 18.6% | 1.2% | 0.0% |
| Không có lý do làm giảm điểm | 52.5% | 0.8% | 0.0% |

- Ở mức Cảnh báo cao, lịch sử trễ hạn là lý do đầu tiên của 78.7% hồ sơ và mỗi hồ sơ trung bình có 3.11 nhóm làm giảm điểm.
- Ở mức Theo dõi, lý do đầu tiên phân tán: mức sử dụng hạn mức (44.5%) và dư nợ (33.1%) nhiều hơn trễ hạn (9.9%). Nhóm này gồm nhiều khách hàng chưa trễ hạn nhưng sử dụng hạn mức cao.
- Ở mức Bình thường, 52.5% hồ sơ không có nhóm nào làm giảm điểm từ 1 điểm trở lên, nên demo hiển thị "không có nhóm nào làm giảm điểm".

Đóng góp trung bình theo nhóm (điểm; âm là làm tăng rủi ro):

| Nhóm | Bình thường | Theo dõi | Cảnh báo cao |
| :--- | ---: | ---: | ---: |
| Lịch sử trễ hạn | +17.3 | +10.2 | −34.4 |
| Mức sử dụng hạn mức | +6.7 | −3.2 | −2.1 |
| Dư nợ và biến động dư nợ | +5.5 | −1.2 | −3.7 |
| Hành vi trả nợ | +6.7 | −0.9 | −5.3 |
| Nhân khẩu học | +0.3 | −0.1 | −0.2 |

---

## 5. Giải thích cục bộ: ba hồ sơ mẫu của demo

Đóng góp theo nhóm (điểm). Điểm nền 532.

| Nhóm | Bình thường | Ca biên | Cảnh báo cao |
| :--- | ---: | ---: | ---: |
| Lịch sử trễ hạn | +12.4 | −11.3 | −59.5 |
| Mức sử dụng hạn mức | +12.1 | −5.3 | −8.5 |
| Hành vi trả nợ | +14.6 | −0.6 | −8.0 |
| Dư nợ và biến động dư nợ | +7.0 | +4.6 | +2.2 |
| Nhân khẩu học | +1.0 | −1.8 | +0.1 |
| **PD / điểm / mức** | 4.0% / 579 / Bình thường | 26.0% / 517 / Theo dõi | 73.3% / 458 / Cảnh báo cao |

- **Bình thường:** không nhóm nào làm giảm điểm; bốn nhóm cộng từ +7 đến +15 điểm nên hồ sơ nằm trên điểm nền.
- **Ca biên:** lịch sử trễ hạn (−11.3) và mức sử dụng hạn mức (−5.3) là hai lý do chính, bị bù một phần bởi dư nợ (+4.6). PD 26.0% sát ranh giới Cảnh báo cao (27%).
- **Cảnh báo cao:** lịch sử trễ hạn (−59.5) chi phối; theo biến, `PAY_1` (−19.6), `PAY_MAX` (−15.0), `PAY_LATE_2PLUS_COUNT` (−9.8) và `PAY_LATE_CONSECUTIVE` (−9.3) cùng góp.
- **Lý do dùng nhóm thay vì biến:** ở Ca biên, `PAY_MAX` đóng góp −11.7 điểm còn `PAY_1` đóng góp +9.8 điểm. Hai biến cùng nhóm bù trừ nhau, còn tổng nhóm (−11.3) ổn định và dễ diễn giải hơn.

---

## 6. Hạn chế

- **SHAP không phải quan hệ nhân quả.** Đóng góp mô tả cách mô hình dùng biến, chịu ảnh hưởng của tương quan giữa các biến; không dùng để kết luận nguyên nhân vỡ nợ.
- **Phân bổ trong nhóm không ổn định.** Các biến trễ hạn và dư nợ tương quan cao nên đóng góp từng biến có thể đổi dấu giữa các mô hình; chỉ đóng góp theo nhóm được dùng làm lý do.
- **Ràng buộc đơn điệu chưa đầy đủ.** `PAY_1`–`PAY_6` không được ràng buộc; một số mức `PAY_1` có đóng góp không đơn điệu.
- **Ngưỡng 1 điểm là quy ước.** Mức 1 điểm (khoảng 0.035 logit) được chọn để loại các đóng góp không đáng kể; không có cơ sở thống kê riêng.
- **Mức Theo dõi chiếm 46.65% hồ sơ Valid** và có lý do đầu tiên phân tán; lý do chỉ gợi ý hướng xem xét, không đủ để quyết định can thiệp riêng lẻ.
- **Không dùng làm thông báo từ chối tín dụng.** Mô hình là công cụ cảnh báo sớm cho danh mục hiện hữu; reason codes chưa được thiết kế theo yêu cầu pháp lý về thông báo hành động bất lợi.
- **Dữ liệu giới hạn.** Một ngân hàng, năm 2005, một snapshot; nhãn là vỡ nợ tháng kế tiếp.

---

## 7. Tái lập

```bash
python scripts/run_pipeline.py --mode final-model   # models/final_model.pkl
python scripts/analyze_reason_codes.py              # bảng ở mục 4.2
pytest tests/test_final_model.py                    # tính cộng tính, mức cảnh báo của hồ sơ mẫu
streamlit run app/streamlit_app.py                  # demo
```
