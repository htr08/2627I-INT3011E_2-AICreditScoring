# Khai báo sử dụng AI và phân chia đóng góp

_Tài liệu ghi nhận công cụ AI đã được dùng, phần do thành viên tự thực hiện và cách kiểm chứng đầu ra. Mỗi thành viên có một mục theo cùng mẫu; số liệu và nguồn dẫn theo các báo cáo trong `reports/`._

---

## 1. Phạm vi và nguyên tắc ghi nhận

- Khai báo theo từng thành viên và từng nhóm công việc theo phân công trong `project_charter.md` mục 9.
- Mỗi dòng nêu công cụ, việc được hỗ trợ và phần quyết định do thành viên đảm nhiệm.
- Các quyết định cốt lõi (chọn bài toán, thiết kế đánh giá, chọn ngưỡng, chính sách biến nhạy cảm, diễn giải kết quả) thuộc về thành viên; AI không được dùng để thay thế các quyết định này.
- Đầu ra do AI hỗ trợ chỉ được đưa vào repo sau khi qua kiểm chứng nêu ở từng mục (unit test, CI, đối chiếu giữa các báo cáo).

---

## 2. Thành viên C (Hiền Trang): Explainability & Product

Phạm vi theo Charter: Project Charter, phân tích SHAP, reason codes, demo Streamlit, Model Card, tổng hợp báo cáo.

### 2.1. Công cụ AI và mục đích sử dụng

| Công cụ | Việc được hỗ trợ | Ghi chú |
|---|---|---|
| Claude | Rà soát code và PR: chỉ ra lỗi, điểm thiếu, đề xuất hướng sửa | Rà soát ở dạng nhận xét; thành viên tự quyết định có sửa hay không và sửa như thế nào |
| Claude | Biên tập tài liệu trong `reports/` và `README.md`: soát tính nhất quán số liệu giữa các báo cáo, văn phong, định dạng bảng | Số liệu gốc lấy từ kết quả chạy trong repo, không do AI sinh |

### 2.2. Phần do thành viên tự thực hiện và quyết định

| Nội dung | Quyết định | Nguồn |
|---|---|---|
| Gộp đóng góp SHAP theo 5 nhóm đặc trưng (lịch sử trễ hạn, sử dụng hạn mức, dư nợ, hành vi trả nợ, nhân khẩu học) | Các biến trễ hạn và dư nợ tương quan cao nên giải thích theo biến đơn lẻ không ổn định; chỉ đóng góp theo nhóm được dùng làm lý do | `xai_report.md`, `final_report.md` mục 8 |
| Quy tắc reason codes | Tối đa 3 nhóm làm giảm điểm (đóng góp ≤ −1 điểm) và 3 yếu tố tích cực (đóng góp ≥ +1 điểm) | `xai_report.md` mục 4 |
| Quy đổi SHAP về logit(PD) sau hiệu chuẩn | `TreeExplainer` trên margin thô, nhân hệ số Platt để tổng đóng góp cộng đúng về logit(PD) | `final_report.md` mục 8 |
| Phân tầng rủi ro và hành động đề xuất của demo | PD < 9% Bình thường; 9% ≤ PD < 27% Theo dõi; PD ≥ 27% Cảnh báo cao; hành động gồm theo dõi, giảm hạn mức, chuyển thu hồi sớm | `demo_wireframe.md`, `app/risk_tiers.py` |
| Chiến lược dự phòng của demo | Mô hình cuối, rồi baseline từ MLflow Registry, rồi mock model kèm cảnh báo | `demo_wireframe.md`, `app/streamlit_app.py` |
| Phạm vi sử dụng và giới hạn trong Model Card | Công cụ cảnh báo sớm cho danh mục hiện hữu; không dùng để xét duyệt thẻ mới, không dùng làm thông báo từ chối tín dụng | `model_card.md` mục 2, 7, 8 |
| Chính sách biến nhạy cảm, KPI và Definition of Done trong Charter | Loại `SEX`; `AGE` chỉ dùng dạng binning; KPI chính theo kiểm định thống kê so với baseline | `project_charter.md` mục 3, 4, 8 |
| Tổng hợp báo cáo cuối | Chọn cấu trúc, đối chiếu số liệu với các báo cáo thành phần, nêu đầy đủ hạn chế (KPI tham chiếu chưa đạt, Test chấm hai lượt, DIR không hiệu chỉnh) | `final_report.md` |

### 2.3. Kiểm chứng đầu ra

| Cơ chế | Nội dung | Bằng chứng |
|---|---|---|
| Unit test cho phần XAI và demo | 34 test: `test_explain.py` (13), `test_final_model.py` (6), `test_demo_app.py` (7), `test_validation.py` (5), `test_risk_tiers.py` (3); kiểm tra tính cộng của đóng góp SHAP, bảo toàn tổng khi gộp nhóm, tách reason code theo hướng và ngưỡng | `tests/` |
| CI | `pytest --cov=src` chạy trên mỗi push và PR vào `main` | `.github/workflows/ci.yml` |
| Đối chiếu chéo giải thích giữa mô hình | Nhóm trễ hạn đứng đầu ở mọi cấu hình; reason code đầu tiên trùng 96% giữa LightGBM và XGBoost `max_depth=4` (hồ sơ rủi ro cao), 76% với Logistic Regression | `shap_agreement.md`, `xai_report.md` mục 3 |
| Truy vết số liệu | Các mục của báo cáo cuối dẫn nguồn đến file kết quả; chỉ số Test lấy nguyên từ `test_evaluation.md`, không tính lại | `final_report.md` |

---

## 3. Thành viên A: Data Engineer / Analyst

Bổ sung theo cùng mẫu mục 2.

## 4. Thành viên B: ML Engineer

Bổ sung theo cùng mẫu mục 2.
