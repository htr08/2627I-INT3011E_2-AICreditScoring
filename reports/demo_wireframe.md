# Wireframe Demo Streamlit

## Tóm tắt

Tài liệu mô tả ứng dụng demo `app/streamlit_app.py`: bố cục giao diện, nguồn dữ liệu cho từng thành phần và hợp đồng giao diện giữa app và mô hình. App chạy với mô hình cuối LightGBM monotonic + Platt; baseline Logistic Regression và mock model là phương án dự phòng.

Ứng dụng khởi chạy từ thư mục gốc repo bằng lệnh:

```bash
streamlit run app/streamlit_app.py
```

## 1. Bố cục

Bố cục với mô hình cuối, hồ sơ mẫu Ca biên.

```
┌──────────────────────────┬──────────────────────────────────────────────────────────────────┐
│ SIDEBAR                  │ 💳 AI Credit Scoring                                             │
│                          │ Công cụ cảnh báo sớm / quản lý danh mục hiện hữu, không phải     │
│ Mô hình                  │ công cụ xét duyệt tín dụng mới.                                  │
│  LightGBM monotonic +    ├──────────────────────────────────────────────────────────────────┤
│  Platt (mô hình cuối)    │ FORM NHẬP  [Thông tin chung] [Lịch sử trả nợ] [Dư nợ & thanh toán]│
│  ℹ PD đã hiệu chuẩn,     │                                                                  │
│    ngưỡng đã khóa        │  Thông tin chung:  Hạn mức | Tuổi | Học vấn | Hôn nhân           │
│  PDO: 600 @ 50:1, 20     │  Lịch sử trả nợ:   T9  T8  T7  T6  T5  T4   (PAY_1..6)           │
│                          │  Dư nợ & thanh toán: 6 cột × (Dư nợ, Đã trả)                     │
│ Hồ sơ mẫu                │                                            [ Chấm điểm ]          │
│  [Ca biên        ▼]      ├──────────────────────────────────────────────────────────────────┤
│  [Nạp hồ sơ]             │ KẾT QUẢ                                                          │
│                          │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐              │
│ Phân tầng rủi ro         │  │ PD   26.0%   │ │ Điểm   517   │ │ 🟡 Theo dõi  │              │
│  Mức        │ Hành động  │  └──────────────┘ └──────────────┘ └──────────────┘              │
│  🟢 Bình thường │ ...    │  [██████████░░░░░░░░░░░░░] vị trí trên thang 420–640             │
│  🟡 Theo dõi    │ ...    │  ┌────────────────────────────────────────────────────────────┐  │
│  🔴 Cảnh báo cao│ ...    │  │ Hành động đề xuất: Gửi nhắc nợ tự động, tạm dừng nâng hạn  │  │
│                          │  └────────────────────────────────────────────────────────────┘  │
│                          │  Waterfall (điểm)                  │ Lý do chính làm giảm điểm   │
│                          │  Điểm nền        ████████ 532      │  - Lịch sử trễ hạn: −11     │
│                          │  Lịch sử trễ hạn     ▐██ −11.3     │  - Mức sử dụng hạn mức: −5  │
│                          │  Mức sử dụng HM      ▐█ −5.3       │  - Nhân khẩu học: −2        │
│                          │  Dư nợ, biến động  ▌    +4.6       │ Yếu tố tích cực             │
│                          │  ...                               │  - Dư nợ, biến động: +5     │
│                          │  Điểm cuối       ███████ 517       │                             │
│                          │                                                                  │
│                          │  ▸ Đóng góp theo biến (10 biến lớn nhất, điểm)                   │
│                          │  ▸ Chi tiết kỹ thuật (điểm nền + đóng góp, logit, dữ liệu vào)   │
└──────────────────────────┴──────────────────────────────────────────────────────────────────┘
```

Với baseline và mock model, sidebar thay bảng phân tầng bằng thanh trượt "Ngưỡng cảnh báo PD" (mặc định 0.30) và không có khung hành động đề xuất.

## 2. Thành phần và nguồn dữ liệu

| Thành phần | Nội dung | Nguồn |
|---|---|---|
| Form nhập | Cột gốc UCI: `LIMIT_BAL`, `AGE`, `EDUCATION`, `MARRIAGE`, `PAY_1..6`, `BILL_AMT1..6`, `PAY_AMT1..6`. Không nhập `SEX` (mô hình chính không dùng biến này). | Người dùng / hồ sơ mẫu (`app/sample_profiles.py`) |
| Kiểm tra đầu vào | Chặn chấm điểm khi thiếu trường, sai kiểu hoặc ngoài phạm vi; cảnh báo khi dư nợ vượt 3 lần hạn mức. | `app/validation.py` |
| PD | Xác suất vỡ nợ đã hiệu chuẩn | `model.explain(record).pd` |
| Điểm | PDO 600 tại odds 50:1, PDO = 20, clip 300–850; thanh vị trí hiển thị thang 420–640 (`SCORE_DISPLAY_RANGE` trong app) | `src.scoring.pd_to_score` |
| Mức cảnh báo | Mô hình cuối: 🟢 PD < 9%, 🟡 9% ≤ PD < 27%, 🔴 PD ≥ 27%. Baseline và mock: 🔴 PD ≥ ngưỡng, 🟡 PD ≥ ngưỡng/2 | `app/risk_tiers.py` (`calibration.threshold`, `risk_tiers.alert_threshold`); ngưỡng ở sidebar chỉ cho baseline và mock |
| Hành động đề xuất | Theo mức cảnh báo (chỉ mô hình cuối): duy trì hạn mức; nhắc nợ tự động và tạm dừng nâng hạn mức; giảm hạn mức và chuyển thu hồi sớm | `app/risk_tiers.py` |
| Waterfall | Điểm nền → đóng góp từng nhóm → điểm cuối (chưa clip) | `base_points`, `shap_to_points` |
| Reason codes | Tối đa 3 nhóm làm giảm điểm (≤ −1 điểm) và 3 yếu tố tích cực (≥ +1 điểm) | `MIN_REASON_POINTS` trong `app/explanation.py` |
| Đóng góp theo biến | 10 biến có |đóng góp| lớn nhất, đơn vị điểm (chỉ mô hình cuối) | `Explanation.variable_contributions` |

Giải thích trình bày theo **nhóm đặc trưng** (lịch sử trễ hạn, mức sử dụng hạn mức, hành vi trả nợ, biến động dư nợ, nhân khẩu học) chứ không theo từng biến, vì `PAY_x` và `BILL_AMT_x` tương quan cao (`reports/project_plan.md` mục 3.5e). Định nghĩa reason codes: `reports/xai_report.md` mục 4.

## 3. Giao diện với mô hình

App chỉ gọi `model.explain(record)`, với đối tượng `Explanation` (định nghĩa trong `app/explanation.py`) gồm:

- `pd: float`: PD (đã hiệu chuẩn với mô hình cuối).
- `base_logit: float`: base value của SHAP trên thang logit(PD).
- `contributions: dict[str, float]`: tổng SHAP theo từng nhóm, cùng thang logit, thỏa `base_logit + sum(contributions) == logit(pd)`.
- `variable_contributions: dict[str, float] | None`: đóng góp theo biến gốc, cùng thang logit; chỉ có với mô hình cuối.

Mô hình có thêm thuộc tính `name` và `is_mock`; mô hình cuối có `is_final = True` để app dùng phân tầng cố định.

### 3.1. Adapter cho baseline Logistic Regression

`app/baseline_model.py` nạp Pipeline `preprocess → LogisticRegression` (`src.pipelines.make_pipeline`, feature freeze v1) từ MLflow Model Registry (`models:/logreg_baseline@baseline`, do `scripts/run_pipeline.py` đăng ký). Bước `preprocess` gồm làm sạch mã (EDUCATION, MARRIAGE, PAY_0 → PAY_1), đặc trưng T2, binning `AGE` theo IV (`AGE_BIN`), bỏ `SEX`/`ID`, one-hot 9 biến phân loại và impute/scale 40 biến số. Adapter là phương án dự phòng khi không nạp được mô hình cuối (mục 3.2).

- **Giải thích:** với mô hình tuyến tính, SHAP interventional trên thang logit tính chính xác bằng `phi_k = coef_k × (z_k − mean_k)`, với `z` là vector sau tiền xử lý và `mean_k` là trung bình trên tập Train (background); base value bằng trung bình logit trên Train. Kết quả trùng với `shap.LinearExplainer`, do đó adapter không phụ thuộc thư viện `shap`. PD do adapter tính khớp với `predict_proba` của pipeline. Background được tính khi nạp mô hình, do đó adapter yêu cầu `data/raw` và `data/splits`.
- **Ánh xạ nhóm:** mỗi cột sau one-hot được quy về cột gốc rồi gán nhóm theo `feature_groups` trong `configs/config.yaml` (định nghĩa chung với phân tích SHAP, `notebooks/03_shap_dependence.ipynb` mục 4). `PAY_1..6`, `PAY_MEAN`, `PAY_MAX`, `PAY_SLOPE`, `PAY_LATE_*` → Lịch sử trễ hạn; `LIMIT_BAL`, `UTIL_*` → Mức sử dụng hạn mức; `PAY_AMT*`, `PAY_RATIO_*`, `MIN_PAY_FLAG_*` → Hành vi trả nợ; `BILL_AMT*`, `BILL_STD`, `BILL_DELTA` → Dư nợ và biến động dư nợ; `AGE_BIN`, `EDUCATION`, `MARRIAGE` → Nhân khẩu học.
- **Cột đầu vào:** form và mô hình dùng cùng tên cột (`PAY_1..6`); mô hình chính thức không dùng `SEX` (Charter mục 8) nên form không nhập biến này.
- **Hiệu chuẩn:** PD của baseline chưa hiệu chuẩn trên tập Valid.

Kết quả với 3 hồ sơ mẫu trên baseline (feature freeze v1): Bình thường PD 7.4% (560 điểm), Ca biên 19.1% (529 điểm), Cảnh báo cao 71.1% (461 điểm).

### 3.2. Adapter cho mô hình cuối

`app/final_model.py` nạp `models/final_model.pkl` (`CalibratedPipeline`: pipeline `preprocess → LGBMClassifier` fit trên Train, Platt fit trên Valid, ngưỡng 0.09), xuất bởi `python scripts/run_pipeline.py --mode final-model`.

- **Giải thích:** `shap.TreeExplainer` trên margin thô; SHAP nhân hệ số Platt `a` và base calibrated bằng `a·φ₀ + b`, nên `base_logit + sum(contributions) == logit(PD)` (kiểm thử tại `tests/test_final_model.py`). Demo không cần dữ liệu Train.
- **Phân tầng và hành động:** PD < 9%: Bình thường (duy trì hạn mức, theo dõi định kỳ); 9% ≤ PD < 27%: Theo dõi (nhắc nợ tự động, tạm dừng nâng hạn mức); PD ≥ 27%: Cảnh báo cao (giảm hạn mức, chuyển thu hồi sớm). Ngưỡng lấy từ `configs/config.yaml`, khớp `reports/calibration_threshold.md` mục 6.3.
- **Kiểm tra đầu vào:** tuổi 21–79, hạn mức 10.000–1.000.000 NT$, mã `PAY_x` từ −2 đến 9, `PAY_AMT` không âm, mã học vấn và hôn nhân hợp lệ.
- **Chọn mô hình:** mặc định nạp mô hình cuối, lỗi thì dùng baseline từ MLflow Registry, rồi mock model kèm thông báo lỗi ở sidebar. Biến môi trường `DEMO_MODEL=final|baseline|mock` buộc dùng đúng mô hình.
- **Hồ sơ mẫu:** Bình thường PD 4.0% (579 điểm), Ca biên 26.0% (517 điểm, sát ranh giới cảnh báo cao), Cảnh báo cao PD 73.3% (458 điểm).
- **Giới hạn:** adapter chỉ hỗ trợ LightGBM với calibrator Platt; calibrator Isotonic làm tổng SHAP không còn khớp PD (`reports/project_plan.md` mục 3.5e). Ngưỡng 0.09 gắn cờ khoảng 73% hồ sơ nên cần đối chiếu với năng lực xử lý của quy trình can thiệp (`reports/model_card.md` mục 7).

### 3.3. Mock model

`app/mock_model.py` dùng hệ số đặt tay, chỉ để minh họa giao diện khi không có mô hình thật; app hiển thị cảnh báo ở sidebar khi dùng mock.
