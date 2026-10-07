import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Chiều import một chiều: preprocessing -> features. features.py không được import preprocessing.
from src.features import ENGINEERED_COLUMNS, FeatureEngineeringTransformer

# Danh sách biến phân loại cho mô hình chính thức (không bao gồm SEX theo Charter mục 1.3 & 8)
CATEGORICAL_COLUMNS = [
    "EDUCATION",
    "MARRIAGE",
    "PAY_1",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]

# Danh sách biến phân loại đầy đủ (bao gồm SEX cho mô hình đối chiếu fairness)
CATEGORICAL_COLUMNS_WITH_SEX = ["SEX"] + CATEGORICAL_COLUMNS

# Các cột nhạy cảm / không phải đặc trưng dự đoán
SENSITIVE_COLUMNS = ["SEX", "ID"]

NUMERIC_COLUMNS = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


class AbnormalCodeTransformer(BaseEstimator, TransformerMixin):
    """Xử lý các mã không được tài liệu hóa và chuẩn hóa tên cột theo project_plan.md mục 3.1:
    - PAY_0 -> PAY_1 (thống nhất chuỗi tháng 1..6)
    - EDUCATION: 0, 5, 6 -> 4 ('Khác')
    - MARRIAGE: 0 -> 3 ('Khác')
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Đổi tên PAY_0 -> PAY_1 nếu có
        if "PAY_0" in X.columns and "PAY_1" not in X.columns:
            X = X.rename(columns={"PAY_0": "PAY_1"})

        # Merge mã EDUCATION bất thường vào mã 4 ("Khác")
        if "EDUCATION" in X.columns:
            X["EDUCATION"] = X["EDUCATION"].replace({
                0: 4,
                5: 4,
                6: 4,
            })

        # Merge mã MARRIAGE 0 vào mã 3 ("Khác")
        if "MARRIAGE" in X.columns:
            X["MARRIAGE"] = X["MARRIAGE"].replace({0: 3})

        return X


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Transformer loại bỏ các cột chỉ định (ví dụ SEX, ID)."""

    def __init__(self, columns=None):
        self.columns = columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.columns, errors="ignore")


def build_preprocessing_pipeline(drop_sensitive: bool = True, engineered: bool = True):
    """Xây dựng pipeline tiền xử lý (feature freeze v1).

    Thứ tự: làm sạch mã & đổi tên PAY_0 -> tạo đặc trưng T2 -> bỏ SEX/ID -> encode/scale.

    Args:
        drop_sensitive: Nếu True, loại bỏ SEX và ID (mô hình chính thức).
                        Nếu False, giữ SEX để đối chiếu fairness (Charter mục 1.3).
        engineered: Nếu True, thêm các đặc trưng T2 (ENGINEERED_COLUMNS).
    """
    cat_cols = CATEGORICAL_COLUMNS if drop_sensitive else CATEGORICAL_COLUMNS_WITH_SEX
    num_cols = NUMERIC_COLUMNS + (ENGINEERED_COLUMNS if engineered else [])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Một số đặc trưng T2 có NaN (vd PAY_RATIO khi dư nợ <= 0) nên cần impute trước khi scale.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        [
            ("categorical", categorical_pipeline, cat_cols),
            ("numeric", numeric_pipeline, num_cols),
        ],
        remainder="drop",
    )

    steps = [("abnormal_codes", AbnormalCodeTransformer())]
    if engineered:
        steps.append(("feature_eng", FeatureEngineeringTransformer()))
    if drop_sensitive:
        steps.append(("drop_sensitive", DropColumnsTransformer(columns=SENSITIVE_COLUMNS)))
    steps.append(("preprocessor", preprocessor))

    return Pipeline(steps)