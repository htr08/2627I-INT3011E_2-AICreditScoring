# Feature Selection Decision — Feature Freeze v1

## 1. Quyết định lựa chọn đặc trưng

- **ID:** Loại khỏi tập đặc trưng vì chỉ là mã định danh khách hàng.
- **SEX:** Loại khỏi scorecard chính thức vì là biến nhạy cảm.
- **Biến gốc và biến dẫn xuất hợp lệ:** Giữ theo cấu hình Feature Freeze v1.
- **Biến có IV thấp:** Chỉ loại trong thí nghiệm feature selection khi IV dưới ngưỡng 0.02.
- **Biến tương quan cao:** Chỉ loại trong thí nghiệm khi hệ số tương quan vượt ngưỡng 0.90. Ưu tiên giữ biến có IV cao hơn; nếu IV bằng nhau, áp dụng quy tắc phá hòa đã cài đặt trong code.

## 2. Phạm vi áp dụng

Feature Freeze v1 tiếp tục là cấu hình đặc trưng chính thức của dự án.

Lọc tương quan chỉ được sử dụng trong thí nghiệm, không tự động áp dụng vào pipeline chính thức.

Kết quả so sánh AUC được lưu tại `reports/correlation_experiment_auc.csv`.

Chi tiết các đặc trưng bị loại trong thí nghiệm được lưu tại `reports/correlation_feature_selection.csv`.

## 3. Kết luận

Kết quả thí nghiệm được sử dụng để đánh giá ảnh hưởng của bước lọc tương quan đến hiệu năng mô hình. Danh sách đặc trưng chính thức chỉ thay đổi khi có quyết định cập nhật Feature Freeze.