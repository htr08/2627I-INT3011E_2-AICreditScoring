"""Module xây dựng đặc trưng cho Credit Scoring.

Bao gồm:
- Baseline: build_features() tách X, y và make_pipeline() chuẩn hóa.
- Tuần 2 - T2 (Feature Engineering):
    - create_utilization_features: tỷ lệ sử dụng hạn mức (UTIL_1..6, UTIL_MEAN)
    - create_payment_trend_features: xu hướng trễ hạn (PAY_MEAN, PAY_MAX, PAY_SLOPE, PAY_LATE_CONSECUTIVE, PAY_LATE_2PLUS_COUNT)
    - create_payment_ratio_features: tỷ lệ thanh toán / dư nợ (PAY_RATIO_1..5, PAY_RATIO_MEAN)
    - create_bill_variation_features: biến động dư nợ (BILL_STD, BILL_DELTA)
    - create_min_payment_features: cờ trả tối thiểu (MIN_PAY_FLAG_1..6, MIN_PAY_FLAG_COUNT)
    - fit_age_bins / transform_age_bins / create_age_bin_features: age binning theo IV
    - FeatureEngineeringTransformer: gói các hàm trên thành một bước Pipeline
- Tuần 2 - T3 (WoE & IV Feature Selection):
    - AgeBinningTransformer: Transformer binned AGE dựa trên IV fit từ train
    - WoEIVTransformer: Transformer tính WoE và lọc theo ngưỡng IV
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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

    Args:
        estimator: Scikit-learn estimator (LogisticRegression, DecisionTreeClassifier, …).

    Returns:
        sklearn.pipeline.Pipeline sẵn sàng gọi .fit() / .predict_proba().
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", estimator),
    ])


# Create credit utilization features from monthly bill amounts.
def create_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for i in range(1, 7):
        df[f"UTIL_{i}"] = np.where(
            df["LIMIT_BAL"] > 0,
            df[f"BILL_AMT{i}"] / df["LIMIT_BAL"],
            np.nan
        )

    util_cols = [f"UTIL_{i}" for i in range(1, 7)]
    df["UTIL_MEAN"] = df[util_cols].mean(axis=1)

    return df


# Calculate the slope of monthly payment status.
def calculate_slope(values):
    x = np.arange(len(values))

    if np.all(pd.isna(values)):
        return np.nan

    return np.polyfit(x, values, 1)[0]


# Calculate the maximum number of consecutive late-payment months.
def max_consecutive_late(values):
    max_count = 0
    current_count = 0

    for value in values:
        if value >= 2:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0

    return max_count


# Create features describing payment delay trends.
def create_payment_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # PAY_6 is the oldest month, PAY_1 is the most recent month.
    pay_cols = [f"PAY_{i}" for i in range(6, 0, -1)]

    df["PAY_MEAN"] = df[pay_cols].mean(axis=1)
    df["PAY_MAX"] = df[pay_cols].max(axis=1)
    df["PAY_SLOPE"] = df[pay_cols].apply(calculate_slope, axis=1)

    df["PAY_LATE_CONSECUTIVE"] = df[pay_cols].apply(
        max_consecutive_late,
        axis=1
    )

    df["PAY_LATE_2PLUS_COUNT"] = (df[pay_cols] >= 2).sum(axis=1)

    return df


# Create payment-to-bill ratio features.
def create_payment_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for i in range(1, 6):
        bill_col = f"BILL_AMT{i + 1}"
        pay_col = f"PAY_AMT{i}"

        df[f"PAY_RATIO_{i}"] = np.where(
            df[bill_col] > 0,
            df[pay_col] / df[bill_col],
            np.nan
        )

    ratio_cols = [f"PAY_RATIO_{i}" for i in range(1, 6)]
    df["PAY_RATIO_MEAN"] = df[ratio_cols].mean(axis=1)

    return df


# Create features measuring monthly bill balance variation.
def create_bill_variation_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]

    df["BILL_STD"] = df[bill_cols].std(axis=1)

    bill_delta = df[bill_cols].diff(axis=1)
    df["BILL_DELTA"] = bill_delta.iloc[:, 1:].abs().mean(axis=1)

    return df


# Create flags for months with very small payment amounts.
def create_min_payment_features(
    df: pd.DataFrame,
    threshold: float = 0.1
) -> pd.DataFrame:
    df = df.copy()

    for i in range(1, 7):
        bill_col = f"BILL_AMT{i}"
        pay_col = f"PAY_AMT{i}"

        df[f"MIN_PAY_FLAG_{i}"] = np.where(
            df[bill_col] > 0,
            (df[pay_col] / df[bill_col]) <= threshold,
            0
        )

    min_pay_cols = [f"MIN_PAY_FLAG_{i}" for i in range(1, 7)]

    df["MIN_PAY_FLAG_COUNT"] = df[min_pay_cols].sum(axis=1)

    return df


# Tên các đặc trưng do FeatureEngineeringTransformer tạo ra (feature freeze v1).
ENGINEERED_COLUMNS = (
    [f"UTIL_{i}" for i in range(1, 7)] + ["UTIL_MEAN"]
    + ["PAY_MEAN", "PAY_MAX", "PAY_SLOPE", "PAY_LATE_CONSECUTIVE", "PAY_LATE_2PLUS_COUNT"]
    + [f"PAY_RATIO_{i}" for i in range(1, 6)] + ["PAY_RATIO_MEAN"]
    + ["BILL_STD", "BILL_DELTA"]
    + [f"MIN_PAY_FLAG_{i}" for i in range(1, 7)] + ["MIN_PAY_FLAG_COUNT"]
)


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Gói các hàm tạo đặc trưng Tuần 2 - T2 thành một bước của Pipeline.

    Các hàm này không học tham số từ dữ liệu (stateless) nên không gây leakage.
    Yêu cầu dữ liệu đã đổi tên PAY_0 -> PAY_1 (AbnormalCodeTransformer chạy trước).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = create_utilization_features(X)
        X = create_payment_trend_features(X)
        X = create_payment_ratio_features(X)
        X = create_bill_variation_features(X)
        X = create_min_payment_features(X)
        return X


MISSING_BIN = "__MISSING__"


def _label_missing(bins: pd.Series) -> pd.Series:
    """Gán NaN (vd UTIL khi LIMIT_BAL = 0, PAY_RATIO khi dư nợ <= 0) vào bin riêng MISSING_BIN
    để có WoE riêng, thay vì bị bỏ khi fit và bị coi là trung tính (WoE = 0) khi transform."""
    if not bins.isna().any():
        return bins
    return bins.astype(object).where(bins.notna(), MISSING_BIN)


def _open_edges(edges: np.ndarray) -> np.ndarray:
    """Mở rộng biên ngoài cùng thành -inf/+inf để giá trị ngoài khoảng Train
    rơi vào bin đầu/cuối thay vì thành NaN (WoE = 0)."""
    edges = np.asarray(edges, dtype=float).copy()
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


# Calculate Information Value (IV) for a categorical or binned feature.
def calculate_iv(
    feature: pd.Series,
    target: pd.Series
) -> float:
    data = pd.DataFrame({
        "feature": feature,
        "target": target
    }).dropna()

    grouped = data.groupby("feature", observed=False)["target"]

    good = grouped.apply(lambda x: (x == 0).sum())
    bad = grouped.apply(lambda x: (x == 1).sum())

    good_total = good.sum()
    bad_total = bad.sum()

    good_dist = (good + 0.5) / (good_total + 0.5 * len(good))
    bad_dist = (bad + 0.5) / (bad_total + 0.5 * len(bad))

    woe = np.log(good_dist / bad_dist)

    iv = ((good_dist - bad_dist) * woe).sum()

    return iv


# Find the best age bin edges based on IV using training data only.
def fit_age_bins(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    min_bins: int = 3,
    max_bins: int = 8
) -> np.ndarray:
    age = X_train["AGE"]

    best_iv = -np.inf
    best_edges = None

    for n_bins in range(min_bins, max_bins + 1):
        try:
            _, edges = pd.qcut(
                age,
                q=n_bins,
                retbins=True,
                duplicates="drop"
            )

            edges = np.unique(edges)

            if len(edges) < 3:
                continue

            bins = pd.cut(
                age,
                bins=edges,
                include_lowest=True
            )

            iv = calculate_iv(bins, y_train)

            if iv > best_iv:
                best_iv = iv
                best_edges = edges

        except ValueError:
            continue

    if best_edges is None:
        raise ValueError("Unable to create age bins from training data.")

    return _open_edges(best_edges)


# Apply age bins learned from the training data.
def transform_age_bins(
    df: pd.DataFrame,
    age_edges: np.ndarray
) -> pd.DataFrame:
    df = df.copy()

    df["AGE_BIN"] = pd.cut(
        df["AGE"],
        bins=age_edges,
        include_lowest=True
    )

    return df


# Fit age bins on training data and apply the same bins to the dataset.
def create_age_bin_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    df: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    age_edges = fit_age_bins(
        X_train,
        y_train
    )

    df = transform_age_bins(
        df,
        age_edges
    )

    return df, age_edges


class AgeBinningTransformer(BaseEstimator, TransformerMixin):
    """Transformer tự động phân nhóm AGE tối ưu theo Information Value (IV)."""

    def __init__(self, min_bins: int = 3, max_bins: int = 8):
        self.min_bins = min_bins
        self.max_bins = max_bins

    def fit(self, X, y=None):
        X = X.copy()
        if y is None:
            self.age_edges_ = None
            return self

        y = pd.Series(y, index=X.index)

        # Some unit tests use data without AGE.
        if "AGE" not in X.columns:
            self.age_edges_ = None
            return self

        self.age_edges_ = fit_age_bins(
            X,
            y,
            min_bins=self.min_bins,
            max_bins=self.max_bins
        )
        return self

    def transform(self, X):
        X = X.copy()
        if self.age_edges_ is None or "AGE" not in X.columns:
            return X

        X["AGE_BIN"] = pd.cut(
            X["AGE"],
            bins=self.age_edges_,
            include_lowest=True
        )
        X = X.drop(columns=["AGE"])
        return X


class WoEIVTransformer(BaseEstimator, TransformerMixin):
    """Transformer mã hóa Weight of Evidence (WoE) và lọc theo Information Value (IV).

    Biến số có <= max_categories giá trị khác nhau (cờ 0/1, EDUCATION, PAY_*) được xử lý như
    biến phân loại (mỗi giá trị một bin) vì qcut sẽ gộp chúng thành quá ít bin.
    NaN của biến liên tục được gán vào bin MISSING_BIN riêng (có WoE riêng).
    """

    def __init__(self, n_bins: int = 5, iv_threshold: float = 0.02, max_categories: int = 12):
        self.n_bins = n_bins
        self.iv_threshold = iv_threshold
        self.max_categories = max_categories

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y, index=X.index)

        self.bin_edges_ = {}
        self.category_maps_ = {}
        self.woe_maps_ = {}
        self.iv_values_ = {}

        for col in X.columns:
            try:
                is_continuous = (
                    pd.api.types.is_numeric_dtype(X[col])
                    and X[col].nunique() > self.max_categories
                )
                if is_continuous:
                    _, edges = pd.qcut(
                        X[col],
                        q=self.n_bins,
                        retbins=True,
                        duplicates="drop"
                    )
                    edges = np.unique(edges)
                    if len(edges) < 2:
                        continue
                    edges = _open_edges(edges)
                    bins = _label_missing(pd.cut(
                        X[col],
                        bins=edges,
                        include_lowest=True
                    ))
                    self.bin_edges_[col] = edges
                else:
                    bins = X[col].astype(str)
                    self.category_maps_[col] = bins.unique().tolist()

                data = pd.DataFrame({"bin": bins, "target": y})
                grouped = data.groupby("bin", observed=False, sort=False)["target"]
                good = grouped.apply(lambda x: (x == 0).sum())
                bad = grouped.apply(lambda x: (x == 1).sum())

                good_dist = (good + 0.5) / (good.sum() + 0.5 * len(good))
                bad_dist = (bad + 0.5) / (bad.sum() + 0.5 * len(bad))

                woe = np.log(good_dist / bad_dist)
                iv = ((good_dist - bad_dist) * woe).sum()

                self.woe_maps_[col] = woe.to_dict()
                self.iv_values_[col] = iv

            except (TypeError, ValueError):
                continue

        self.selected_features_ = [
            col for col, iv in self.iv_values_.items()
            if iv >= self.iv_threshold
        ]
        return self

    def transform(self, X):
        X = X.copy()
        result = pd.DataFrame(index=X.index)

        for col in self.selected_features_:
            if col in self.bin_edges_:
                bins = _label_missing(pd.cut(
                    X[col],
                    bins=self.bin_edges_[col],
                    include_lowest=True
                ))
            else:
                bins = X[col].astype(str)

            mapped = bins.map(self.woe_maps_[col])
            result[col] = pd.to_numeric(mapped, errors="coerce").fillna(0.0)

        return result
