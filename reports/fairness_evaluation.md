# Báo cáo Đánh giá Tính Công bằng (Fairness Check Report)

- **Tập dữ liệu**: Validation set (`n = 6000`)
- **Ngưỡng quyết định (Decision Threshold)**: `0.5`
- **Ngưỡng cảnh báo Disparate Impact (DIR)**: `< 0.8` hoặc `> 1.25` (Quy tắc 80%)

---

## 1. Tổng quan So sánh Hiệu năng

| Phiên bản Mô hình | Cột nhạy cảm `SEX` | Overall AUC | Flag Rate Tổng thể |
| :--- | :---: | :---: | :---: |
| **Mô hình Mặc định (No SEX)** | Bị loại bỏ | **0.7706** | 0.1202 |
| **Mô hình Đối chiếu (With SEX)** | Được giữ lại | **0.7706** | 0.1202 |

---

## 2. Phân tích Theo Giới tính (SEX)

### A. Mô hình Mặc định (Không có `SEX`)
| Subgroup | N | Flag Rate | Disparate Impact Ratio | AUC | Fairness Status |
| --- | --- | --- | --- | --- | --- |
| Nam (1) | 2385 | 0.1417 | 1.1794 | 0.7589 | ✅ Đạt |
| Nữ (2) | 3615 | 0.1059 | 0.8817 | 0.7778 | ✅ Đạt |

### B. Mô hình Đối chiếu (Có `SEX`)
| Subgroup | N | Flag Rate | Disparate Impact Ratio | AUC | Fairness Status |
| --- | --- | --- | --- | --- | --- |
| Nam (1) | 2385 | 0.1417 | 1.1794 | 0.7589 | ✅ Đạt |
| Nữ (2) | 3615 | 0.1059 | 0.8817 | 0.7778 | ✅ Đạt |

---

## 3. Phân tích Theo Nhóm Tuổi (AGE)

### A. Mô hình Mặc định (Không có `SEX`)
| Subgroup | N | Flag Rate | Disparate Impact Ratio | AUC | Fairness Status |
| --- | --- | --- | --- | --- | --- |
| 30–50 | 3674 | 0.1124 | 0.9355 | 0.7723 | ✅ Đạt |
| <30 | 1878 | 0.1289 | 1.0723 | 0.7824 | ✅ Đạt |
| >50 | 448 | 0.1473 | 1.226 | 0.7292 | ✅ Đạt |

### B. Mô hình Đối chiếu (Có `SEX`)
| Subgroup | N | Flag Rate | Disparate Impact Ratio | AUC | Fairness Status |
| --- | --- | --- | --- | --- | --- |
| 30–50 | 3674 | 0.1124 | 0.9355 | 0.7723 | ✅ Đạt |
| <30 | 1878 | 0.1289 | 1.0723 | 0.7824 | ✅ Đạt |
| >50 | 448 | 0.1473 | 1.226 | 0.7292 | ✅ Đạt |

---

## 4. Kết luận & Khuyến nghị Nghiệp vụ

1. **Tác động của `SEX`**: Việc loại bỏ biến `SEX` khỏi mô hình chính thức đảm bảo tính tuân thủ pháp lý và phòng ngừa rủi ro phân biệt đối xử trực tiếp.
2. **Đánh giá Disparate Impact (DIR)**: Các sub-group được kiểm soát bằng chỉ số DIR (ngưỡng 0.8) để phát hiện và xử lý kịp thời chênh lệch can thiệp rủi ro.
