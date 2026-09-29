import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_COLUMNS = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_1",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]

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
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "EDUCATION" in X.columns:
            X["EDUCATION"] = X["EDUCATION"].replace({
                0: 4,
                5: 4,
                6: 4
            })

        return X

def build_preprocessing_pipeline():
    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS)
    ])

    pipeline = Pipeline([
        ("abnormal_codes", AbnormalCodeTransformer()),
        ("preprocessor", preprocessor)
    ])

    return pipeline