# Wireframe Demo Streamlit

## Tóm tắt

Tài liệu này mô tả wireframe của ứng dụng demo `app/streamlit_app.py` (Tuần 1, chạy với mock model): bố cục giao diện, nguồn dữ liệu cho từng thành phần, hợp đồng giao diện với mô hình để thay thế mock, và các việc còn mở.

Lệnh chạy từ thư mục gốc repo:

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
│                      │  ...                               │  - Biến động dư nợ: +1      │
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
| Mức cảnh báo | 🔴 PD ≥ ngưỡng; 🟡 PD ≥ ngưỡng/2; 🟢 còn lại | Ngưỡng ở sidebar |
| Waterfall | Điểm nền → đóng góp từng nhóm → điểm cuối (chưa clip) | `base_points`, `shap_to_points` |
| Reason codes | Top 3 nhóm làm giảm điểm / tăng điểm | Như trên |

Giải thích trình bày theo **nhóm đặc trưng** (lịch sử trễ hạn, mức sử dụng hạn mức, hành vi trả nợ, biến động dư nợ, nhân khẩu học) chứ không theo từng biến, vì `PAY_x` và `BILL_AMT_x` tương quan cao (project_plan.md, mục 3e).

## 3. Giao diện với mô hình (hợp đồng để thay mock)

App chỉ gọi `load_model().explain(record)`, với đối tượng trả về gồm:

- `pd: float`: PD đã hiệu chuẩn.
- `base_logit: float`: base value của SHAP trên thang logit(PD) đã hiệu chuẩn.
- `contributions: dict[str, float]`: tổng SHAP theo từng nhóm, cùng thang logit, thỏa `base_logit + sum(contributions) == logit(pd)`.

Khi mô hình baseline thật được kết nối (Tuần 2), việc thay thế chỉ cần một lớp adapter trả về `Explanation` (thay cho `app/mock_model.py`) và đổi `load_model()` để trỏ tới lớp đó. Với Platt scaling, SHAP (thang margin) cần nhân với hệ số `a` trước khi cộng theo nhóm; với Isotonic, tính cộng tính không còn đúng và cần được ghi chú rõ trong app.

## 4. Việc còn mở

- Ngưỡng cảnh báo mặc định 0,30 là tạm thời; thay bằng ngưỡng tối ưu theo ma trận chi phí (Tuần 3).
- Dải hiển thị thanh điểm (420–640) và tham số PDO cần nhóm thống nhất trước khi đưa vào `configs/config.yaml`.
- Mock model dùng hệ số đặt tay, không phản ánh dữ liệu thật; mọi con số trên wireframe chỉ để minh họa.
