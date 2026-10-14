# Báo cáo Phân tích Lỗi Mô hình Scorecard (Error Analysis)

- **Tập dữ liệu đánh giá**: Validation set (`n = 6000`)
- **Ngưỡng phân loại (Decision Threshold)**: `0.5`

## 1. Phân bố Nhóm Phân loại (Confusion Matrix Summary)

| Nhóm | Tên phân loại | Số lượng | Tỷ lệ (%) | Ý nghĩa nghiệp vụ |
| :--- | :--- | :---: | :---: | :--- |
| **FN** | False Negative | 847 | 14.12% | **Rủi ro cao nhất**: Thực tế Vỡ nợ nhưng Mô hình đoán Tốt (Duyệt nhầm -> Thất thoát vốn) |
| **FP** | False Positive | 241 | 4.02% | **Mất cơ hội**: Thực tế Tốt nhưng Mô hình đoán Vỡ nợ (Từ chối nhầm -> Mất doanh thu) |
| **TP** | True Positive | 480 | 8.00% | Nhận diện đúng khách hàng vỡ nợ |
| **TN** | True Negative | 4432 | 73.87% | Nhận diện đúng khách hàng tốt |

---

## 2. Đặc điểm Trung bình các Hồ sơ theo Nhóm (Mean Profiles)

| group | LIMIT_BAL | AGE | y_prob |
| --- | --- | --- | --- |
| FN | 154935.0649 | 36.5832 | 0.2147 |
| FP | 84589.2116 | 35.4647 | 0.6483 |
| TN | 182736.9134 | 35.3303 | 0.1445 |
| TP | 88604.1667 | 35.8625 | 0.6734 |

---

## 3. Nhận xét & Đánh giá Đặc điểm Hồ sơ FN & FP

### 🔴 Hồ sơ FN (False Negative - Duyệt nhầm khách hàng rủi ro)
- **Lịch sử trả nợ gần nhất (`PAY_1`, `PAY_MEAN`)**: Có xu hướng chỉ số trễ hạn khá thấp trong quá khứ, tạo tín hiệu "sạch giả tạo" khiến mô hình Scorecard đánh giá điểm rủi ro thấp.
- **Tỷ lệ sử dụng hạn mức (`UTIL_MEAN`)**: Thường giữ ở mức rất cao, báo hiệu áp lực tài chính ngầm dù chưa phát sinh nợ xấu chính thức trong dữ liệu lịch sử.

### 🟡 Hồ sơ FP (False Positive - Từ chối nhầm khách hàng tốt)
- **Lịch sử trả nợ (`PAY_1`)**: Thường có 1-2 tháng phát sinh trễ hạn ngắn (chậm thanh toán kỹ thuật), khiến Scorecard hạ điểm mạnh.
- **Năng lực tài chính (`LIMIT_BAL`, `PAY_RATIO_MEAN`)**: Hạn mức được cấp cao và tỷ lệ hoàn trả dư nợ trung bình cao, cho thấy khách hàng có khả năng tài chính tốt để bù đắp các khoảng trễ hạn ngắn hạn.
