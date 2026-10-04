"""Xây dựng feature matrix X và vector nhãn y từ DataFrame thô.

Strategy (Option A - Baseline):
  - Dùng toàn bộ đặc trưng số, chỉ loại bỏ cột định danh (ID) và cột nhãn.
  - Chuẩn hoá bằng StandardScaler bên trong Pipeline để tránh data leakage.

Giới hạn của Option A:
  - Các biến EDUCATION, MARRIAGE, SEX và PAY_* là biến danh mục/thứ bậc nhưng đang
    được coi trực tiếp như biến số liên tục (chưa áp dụng One-Hot hay WoE encoding).
  - Đây là lý do chính khiến Logistic Regression (AUC ~0.728) kém hơn Decision Tree
    (AUC ~0.751), do cây quyết định có khả năng phân tách ngưỡng phi tuyến trên các
    giá trị rời rạc tốt hơn mô hình tuyến tính.
  - StandardScaler là thừa đối với Decision Tree (các phép chia nhánh của cây bất biến
    với phép co giãn đơn điệu), nhưng được giữ trong Pipeline để đảm bảo giao diện đồng
    nhất giữa các mô hình baseline.
"""

from typing import List, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config


def _get_default_target_col() -> str:
    try:
        cfg = load_config()
        return cfg.get("data", {}).get("target_col", "default.payment.next.month")
    except Exception:
        return "default.payment.next.month"


TARGET_COL = _get_default_target_col()
DROP_COLS = ["ID"]


def build_features(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Tách X và y từ DataFrame thô (Option A: dùng toàn bộ features trừ ID).

    Args:
        df: DataFrame đã được lọc theo split (train / valid / test).
        target_col: Tên cột nhãn (nếu None sẽ đọc từ config data.target_col).
        drop_cols: Danh sách cột bổ sung cần bỏ (mặc định: ["ID"]).

    Returns:
        (X, y): DataFrame đặc trưng và Series nhãn nhị phân (int).

    Raises:
        KeyError: Nếu target_col không tồn tại trong df.
    """
    if target_col is None:
        target_col = TARGET_COL
    if drop_cols is None:
        drop_cols = DROP_COLS

    if target_col not in df.columns:
        raise KeyError(f"Không tìm thấy cột nhãn '{target_col}' trong DataFrame.")

    cols_to_drop = [c for c in drop_cols if c in df.columns] + [target_col]
    X = df.drop(columns=cols_to_drop)
    y = df[target_col].astype(int)
    return X, y


def make_pipeline(estimator) -> Pipeline:
    """Tạo Pipeline: StandardScaler → estimator.

    Scaler chỉ fit trên train, transform trên valid/test — tránh data leakage.
    Lưu ý: StandardScaler về mặt lý thuyết là thừa với Decision Tree (bất biến với
    biến đổi đơn điệu), nhưng được dùng chung để chuẩn hóa giao diện giữa các mô hình.

    Args:
        estimator: Scikit-learn estimator (LogisticRegression, DecisionTreeClassifier, …).

    Returns:
        sklearn.pipeline.Pipeline sẵn sàng gọi .fit() / .predict_proba().
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", estimator),
    ])
