"""Adapter nối mô hình baseline thật (Logistic Regression, feature freeze v1) với app demo.

Mô hình được nạp từ MLflow Model Registry (`models:/logreg_baseline@baseline`), là Pipeline
`preprocess -> clf` do `src.pipelines.make_pipeline` tạo: làm sạch mã (PAY_0 -> PAY_1, EDUCATION,
MARRIAGE) -> đặc trưng T2 -> bỏ SEX/ID -> one-hot 8 biến phân loại + impute/scale biến số -> LogisticRegression.

Giải thích: với mô hình tuyến tính, SHAP (interventional, background = tập Train) trên thang logit là
chính xác: phi_k = coef_k * (z_k - mean_k), với z là vector sau tiền xử lý và mean_k là trung bình của z_k
trên Train; base value = intercept + coef · mean = trung bình logit trên Train. Cộng phi_k theo nhóm đặc
trưng cho ra `Explanation`, thỏa base_logit + sum == logit(pd). Kết quả trùng với shap.LinearExplainer.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.explanation import DEMOGRAPHICS, Explanation
from src.explain import load_feature_groups, variable_to_group
from src.preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

REGISTRY_NAME = "logreg_baseline"
REGISTRY_ALIAS = "baseline"
# Cột đầu vào của mô hình chính thức (cột gốc, không có SEX/ID); form dùng đúng các tên này.
INPUT_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


# Biến -> nhóm theo configs/config.yaml (feature_groups). AGE (cột gốc trước binning) và SEX (chỉ có ở
# phiên bản đối chiếu fairness) không có trong feature freeze v1 nên được gán thêm vào nhóm nhân khẩu học.
_GROUP_LABELS = {key: spec["label"] for key, spec in load_feature_groups().items()}
_COLUMN_TO_GROUP = {
    **{var: _GROUP_LABELS[key] for var, key in variable_to_group(load_feature_groups()).items()},
    "AGE": DEMOGRAPHICS,
    "SEX": DEMOGRAPHICS,
}


def feature_group(column: str) -> str:
    """Cột trước khi encode (cột gốc hoặc đặc trưng T2) -> nhóm đặc trưng cho reason codes."""
    if column in _COLUMN_TO_GROUP:
        return _COLUMN_TO_GROUP[column]
    raise ValueError(f"Chưa gán nhóm đặc trưng cho cột '{column}'.")


def _source_columns(preprocess: Pipeline) -> list[str]:
    """Tên cột gốc của từng cột đầu ra sau ColumnTransformer (one-hot 'PAY_1_2' -> 'PAY_1')."""
    ct = preprocess.named_steps["preprocessor"]
    inputs = sorted(ct.feature_names_in_, key=len, reverse=True)
    sources = []
    for name in ct.get_feature_names_out():
        name = name.split("__", 1)[1]
        if name in ct.feature_names_in_:
            sources.append(name)
        else:
            sources.append(next(c for c in inputs if name.startswith(f"{c}_")))
    return sources


class BaselineModel:
    is_mock = False

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame, name: str = "Logistic Regression baseline"):
        preprocess, clf = pipeline.named_steps.get("preprocess"), pipeline.named_steps.get("clf")
        if not isinstance(preprocess, Pipeline) or not isinstance(clf, LogisticRegression):
            raise TypeError("Adapter chỉ hỗ trợ Pipeline gồm 'preprocess' và 'clf' (LogisticRegression).")
        if len(clf.classes_) != 2:
            raise ValueError("Adapter chỉ hỗ trợ phân loại nhị phân.")
        self.pipeline = pipeline
        self.name = name
        self._preprocess = preprocess
        self._coef = clf.coef_[0]
        self._mean = np.asarray(preprocess.transform(background), dtype=float).mean(axis=0)
        self._base_logit = float(clf.intercept_[0] + self._coef @ self._mean)
        self._groups = [feature_group(c) for c in _source_columns(preprocess)]

    @classmethod
    def from_registry(cls, name: str = REGISTRY_NAME, alias: str = REGISTRY_ALIAS) -> "BaselineModel":
        """Nạp mô hình từ registry; background là tập Train (cần data/raw và data/splits)."""
        import mlflow.sklearn

        from src.data_split import load_split_data
        from src.features import build_features
        from src.tracking import get_tracking_uri

        mlflow.set_tracking_uri(get_tracking_uri())
        pipeline = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
        train_df, _, _ = load_split_data()
        background, _ = build_features(train_df)
        return cls(pipeline, background, name=f"Logistic Regression baseline ({name}@{alias})")

    def to_frame(self, record: Mapping[str, float]) -> pd.DataFrame:
        """Record của form -> DataFrame 1 dòng với các cột đầu vào của mô hình."""
        missing = [c for c in INPUT_COLUMNS if c not in record]
        if missing:
            raise KeyError(f"Thiếu trường {missing} trong hồ sơ đầu vào.")
        return pd.DataFrame([{c: record[c] for c in INPUT_COLUMNS}], columns=INPUT_COLUMNS)

    def explain(self, record: Mapping[str, float]) -> Explanation:
        z = np.asarray(self._preprocess.transform(self.to_frame(record)), dtype=float)[0]
        phi = self._coef * (z - self._mean)
        contributions: dict[str, float] = {}
        for group, value in zip(self._groups, phi):
            contributions[group] = contributions.get(group, 0.0) + float(value)
        logit_pd = self._base_logit + float(phi.sum())
        pd_ = float(1 / (1 + np.exp(-logit_pd)))
        return Explanation(pd=pd_, base_logit=self._base_logit, contributions=contributions)
