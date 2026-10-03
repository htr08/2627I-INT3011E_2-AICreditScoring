import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    AgeBinningTransformer,
    WoEIVTransformer
)


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

        if "PAY_0" in X.columns:
            X = X.rename(columns={"PAY_0": "PAY_1"})

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


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        return X.drop(
            columns=self.columns,
            errors="ignore"
        )


def build_scorecard_pipeline():
    return Pipeline([
        (
            "drop_sensitive",
            DropColumnsTransformer(
                columns=["SEX", "ID"]
            )
        ),
        (
            "age_binning",
            AgeBinningTransformer(
                min_bins=3,
                max_bins=8
            )
        ),
        (
            "woe_iv",
            WoEIVTransformer(
                n_bins=5,
                iv_threshold=0.02
            )
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])