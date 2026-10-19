# Wireframe Demo Streamlit

## Tóm tắt

Tài liệu này mô tả wireframe của ứng dụng demo `app/streamlit_app.py` (dựng ở Tuần 1 với mock model, nay chạy với mô hình cuối LightGBM monotonic + Platt; baseline Logistic Regression và mock model là phương án dự phòng): bố cục giao diện, nguồn dữ liệu cho từng thành phần, hợp đồng giao diện với mô hình để thay thế mock, và các việc còn mở.

Ứng dụng khởi chạy từ thư mục gốc repo bằng lệnh:

```bash
streamlit run app/streamlit_app.py
```

## 1. Bố cục

```
┌──────────────────────┬──────────────────────────────────────────────────────────────────┐
│ SIDEBAR              │ 💳 AI Credit Scoring                                             │
│                      │ Công cụ cảnh báo sớm / quản lý danh mục hiện hữu, không phải     │
│ Mô hình              │ công cụ xét duyệt tín dụng mới.                                  │
│  <tên mô hình>       ├──────────────────────────────────────────────────────────────────┤
│  ⚠ Mock model        │ FORM NHẬP  [Thông tin chung] [Lịch sử trả nợ] [Dư nợ & thanh toán]│
│  PDO: 600 @ 50:1, 20 │                                                                  │
│                      │  Thông tin chung:  Hạn mức | Tuổi | Học vấn | Hôn nhân           │
│ Hồ sơ mẫu            │  Lịch sử trả nợ:   T9  T8  T7  T6  T5  T4   (PAY_1..6)           │
│  [Ca biên        ▼]  │  Dư nợ & thanh toán: 6 cột × (Dư nợ, Đã trả)                     │
│  [Nạp hồ sơ]         │                                            [ Chấm điểm ]          │
│                      ├──────────────────────────────────────────────────────────────────┤
│ Ngưỡng               │ KẾT QUẢ                                                          │
│  PD ──●────── 0.30   │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐              │
│                      │  │ PD   26.3%   │ │ Điểm   517   │ │ 🟡 Theo dõi  │              │
│                      │  └──────────────┘ └──────────────┘ └──────────────┘              │
│                      │  [██████████░░░░░░░░░░░░░] vị trí trên thang 420–640             │
│                      │                                                                  │
│                      │  Waterfall (điểm)                  │ Lý do chính làm giảm điểm   │
│                      │  Điểm nền        ████████ 523      │  - Mức sử dụng hạn mức: −11 │
│                      │  Mức sử dụng HM      ▐██ −10.7     │  - Hành vi trả nợ: −6       │
│                      │  Lịch sử trễ hạn  ██▌    +9.5      │ Yếu tố tích cực             │
│                      │  Hành vi trả nợ      ▐█ −5.9       │  - Lịch sử trễ hạn: +10     │
│                      │  ...                               │  - Dư nợ, biến động: +1     │
│                      │  Điểm cuối       ███████ 517       │                             │
│                      │                                                                  │
│                      │  ▸ Chi tiết kỹ thuật (điểm nền + đóng góp, logit, dữ liệu vào)   │
└──────────────────────┴──────────────────────────────────────────────────────────────────┘
```

## 2. Thành phần và nguồn dữ liệu

| Thành phần | Nội dung | Nguồn |
|---|---|---|
| Form nhập | Cột gốc UCI: `LIMIT_BAL`, `AGE`, `EDUCATION`, `MARRIAGE`, `PAY_1..6`, `BILL_AMT1..6`, `PAY_AMT1..6`. Không nhập `SEX` (mô hình chính không dùng biến này). | Người dùng / hồ sơ mẫu (`app/sample_profiles.py`) |
| PD | Xác suất vỡ nợ đã hiệu chuẩn | `model.explain(record).pd` |
| Điểm | PDO 600 tại odds 50:1, PDO = 20, clip 300–850 | `src.scoring.pd_to_score` |
| Mức cảnh báo | Mô hình cuối: 🟢 PD < 9%, 🟡 9% ≤ PD < 27%, 🔴 PD ≥ 27%, kèm hành động đề xuất. Baseline và mock: 🔴 PD ≥ ngưỡng, 🟡 PD ≥ ngưỡng/2 | `app/risk_tiers.py` (`calibration.threshold`, `risk_tiers.alert_threshold`); ngưỡng ở sidebar chỉ cho baseline và mock |
| Waterfall | Điểm nền → đóng góp từng nhóm → điểm cuối (chưa clip) | `base_points`, `shap_to_points` |
| Reason codes | Top 3 nhóm làm giảm điểm / tăng điểm | Như trên |

Giải thích trình bày theo **nhóm đặc trưng** (lịch sử trễ hạn, mức sử dụng hạn mức, hành vi trả nợ, biến động dư nợ, nhân khẩu học) chứ không theo từng biến, vì `PAY_x` và `BILL_AMT_x` tương quan cao (project_plan.md, mục 3e).

## 3. Giao diện với mô hình (hợp đồng để thay mock)

App chỉ gọi `model.explain(record)`, với đối tượng `Explanation` (định nghĩa trong `app/explanation.py`) gồm:

- `pd: float`: PD (đã hiệu chuẩn với mô hình cuối).
- `base_logit: float`: base value của SHAP trên thang logit(PD).
- `contributions: dict[str, float]`: tổng SHAP theo từng nhóm, cùng thang logit, thỏa `base_logit + sum(contributions) == logit(pd)`.

### 3.1. Adapter cho baseline Logistic Regression

`app/baseline_model.py` nạp Pipeline `preprocess → LogisticRegression` (`src.pipelines.make_pipeline`, feature freeze v1) từ MLflow Model Registry (`models:/logreg_baseline@baseline`, do `scripts/run_pipeline.py` đăng ký). Bước `preprocess` gồm làm sạch mã (EDUCATION, MARRIAGE, PAY_0 → PAY_1), đặc trưng T2, binning `AGE` theo IV (`AGE_BIN`), bỏ `SEX`/`ID`, one-hot 9 biến phân loại và impute/scale 40 biến số. `load_model()` ưu tiên mô hình này; nếu không nạp được (registry trống, chưa train, thiếu dữ liệu Train), app dùng mock model và hiển thị lỗi ở sidebar. Biến môi trường `DEMO_MODEL=mock` buộc dùng mock model.

- **Giải thích:** với mô hình tuyến tính, SHAP interventional trên thang logit tính chính xác bằng `phi_k = coef_k × (z_k − mean_k)`, với `z` là vector sau tiền xử lý và `mean_k` là trung bình trên tập Train (background); base value bằng trung bình logit trên Train. Kết quả trùng với `shap.LinearExplainer`, do đó adapter không phụ thuộc thư viện `shap`. PD do adapter tính khớp với `predict_proba` của pipeline. Background được tính khi nạp mô hình, do đó app yêu cầu `data/raw` và `data/splits`.
- **Ánh xạ nhóm:** mỗi cột sau one-hot được quy về cột gốc rồi gán nhóm theo `feature_groups` trong `configs/config.yaml` (định nghĩa chung với phân tích SHAP, `notebooks/03_shap_dependence.ipynb` mục 4). `PAY_1..6`, `PAY_MEAN`, `PAY_MAX`, `PAY_SLOPE`, `PAY_LATE_*` → Lịch sử trễ hạn; `LIMIT_BAL`, `UTIL_*` → Mức sử dụng hạn mức; `PAY_AMT*`, `PAY_RATIO_*`, `MIN_PAY_FLAG_*` → Hành vi trả nợ; `BILL_AMT*`, `BILL_STD`, `BILL_DELTA` → Dư nợ và biến động dư nợ; `AGE_BIN`, `EDUCATION`, `MARRIAGE` → Nhân khẩu học.
- **Cột đầu vào:** form và mô hình dùng cùng tên cột (`PAY_1..6`); mô hình chính thức không dùng `SEX` (Charter mục 8) nên form không nhập biến này.
- **Hiệu chuẩn:** PD của baseline chưa hiệu chuẩn trên tập Valid. Khi có mô hình cuối đã calibrate: với Platt scaling, SHAP (thang margin) cần nhân với hệ số `a` trước khi cộng theo nhóm; với Isotonic, tính cộng tính không còn đúng và cần ghi chú rõ trong app.

Kết quả với 3 hồ sơ mẫu trên baseline thật (feature freeze v1): Bình thường PD 7.4% (560 điểm), Ca biên 19.1% (529 điểm), Cảnh báo cao 71.1% (461 điểm). Hồ sơ Ca biên có một tháng trễ 2 kỳ (T8), nằm trong vùng 🟡 Theo dõi ở ngưỡng 0,30 trên cả baseline và mock model.

### 3.2. Adapter cho mô hình cuối

`app/final_model.py` nạp `models/final_model.pkl` (`CalibratedPipeline`: pipeline `preprocess → LGBMClassifier` fit trên Train, Platt fit trên Valid, ngưỡng 0.09), xuất bởi `python scripts/run_pipeline.py --mode final-model`.

- **Giải thích:** `shap.TreeExplainer` trên margin thô; SHAP nhân hệ số Platt `a` và base calibrated bằng `a·φ₀ + b`, nên `base_logit + sum(contributions) == logit(PD)` (kiểm thử tại `tests/test_final_model.py`). Demo không cần dữ liệu Train. Đóng góp được cộng về 5 nhóm đặc trưng; mục "Đóng góp theo biến" liệt kê 10 biến lớn nhất theo điểm.
- **Phân tầng và hành động:** PD < 9%: Bình thường (duy trì hạn mức, theo dõi định kỳ); 9% ≤ PD < 27%: Theo dõi (nhắc nợ tự động, tạm dừng nâng hạn mức); PD ≥ 27%: Cảnh báo cao (giảm hạn mức, chuyển thu hồi sớm). Ngưỡng lấy từ `configs/config.yaml`, khớp `reports/calibration_threshold.md` mục 6.3.
- **Kiểm tra đầu vào:** `app/validation.py` chặn chấm điểm khi thiếu trường, sai kiểu hoặc ngoài phạm vi (tuổi 21–79, hạn mức 10.000–1.000.000 NT$, mã `PAY_x` từ −2 đến 9, `PAY_AMT` không âm, mã học vấn và hôn nhân hợp lệ); cảnh báo khi dư nợ vượt 3 lần hạn mức.
- **Chọn mô hình:** mặc định nạp mô hình cuối, lỗi thì dùng baseline từ MLflow Registry, rồi mock model. Biến môi trường `DEMO_MODEL=final|baseline|mock` buộc dùng đúng mô hình.
- **Hồ sơ mẫu:** Bình thường PD 4.0% (579 điểm), Ca biên 26.0% (517 điểm, sát ranh giới cảnh báo cao), Cảnh báo cao PD 73.3% (458 điểm).

## 4. Việc còn mở

- Ngưỡng 0.09 gắn cờ khoảng 73% hồ sơ (`reports/calibration_threshold.md` mục 6.3); cần đối chiếu với năng lực xử lý của quy trình can thiệp trước khi triển khai thực tế.
- Dải hiển thị thanh điểm (420–640) và tham số PDO cần nhóm thống nhất trước khi đưa vào `configs/config.yaml`.
- Các con số trên wireframe ở mục 1 lấy từ mock model, chỉ để minh họa bố cục.
- Adapter mô hình cuối chỉ hỗ trợ LightGBM với calibrator Platt; calibrator Isotonic làm tổng SHAP không còn khớp PD (project_plan.md mục 3.5e).
