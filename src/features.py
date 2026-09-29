"""Xây dựng feature matrix X và vector nhãn y từ DataFrame thô.

Strategy (Option A - Baseline):
  - Dùng toàn bộ đặc trưng số, chỉ loại bỏ cột định danh (ID) và cột nhãn.
  - Chuẩn hoá bằng StandardScaler bên trong Pipeline để tránh data leakage.
"""

from typing import List, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COL = "default.payment.next.month"
DROP_COLS = ["ID"]


def build_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Tách X và y từ DataFrame thô (Option A: dùng toàn bộ features trừ ID).

    Args:
        df: DataFrame đã được lọc theo split (train / valid / test).
        target_col: Tên cột nhãn.
        drop_cols: Danh sách cột bổ sung cần bỏ (mặc định: ["ID"]).

    Returns:
        (X, y): DataFrame đặc trưng và Series nhãn nhị phân (int).

    Raises:
        KeyError: Nếu target_col không tồn tại trong df.
    """
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

    Args:
        estimator: Scikit-learn estimator (LogisticRegression, DecisionTreeClassifier, …).

    Returns:
        sklearn.pipeline.Pipeline sẵn sàng gọi .fit() / .predict_proba().
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", estimator),
    ])
