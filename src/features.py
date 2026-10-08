"""Module xây dựng đặc trưng cho Credit Scoring.

Bao gồm:
- build_features() tách X, y (Pipeline mô hình nằm ở src.pipelines.make_pipeline).
- Tuần 2 - T2 (Feature Engineering):
    - create_utilization_features: tỷ lệ sử dụng hạn mức (UTIL_1..6, UTIL_MEAN)
    - create_payment_trend_features: xu hướng trễ hạn (PAY_MEAN, PAY_MAX, PAY_SLOPE,
      PAY_LATE_CONSECUTIVE, PAY_LATE_2PLUS_COUNT)
    - create_payment_ratio_features: tỷ lệ thanh toán / dư nợ
      (PAY_RATIO_1..5, PAY_RATIO_MEAN)
    - create_bill_variation_features: biến động dư nợ (BILL_STD, BILL_DELTA)
    - create_min_payment_features: cờ trả tối thiểu
      (MIN_PAY_FLAG_1..6, MIN_PAY_FLAG_COUNT)
    - fit_age_bins / transform_age_bins / create_age_bin_features:
      age binning theo IV
    - FeatureEngineeringTransformer: gói các hàm trên thành một bước Pipeline
- Tuần 2 - T3 (WoE & IV Feature Selection):
    - AgeBinningTransformer: Transformer binned AGE dựa trên IV fit từ train
    - WoEIVTransformer: Transformer tính WoE và lọc theo ngưỡng IV
    - FeatureSelectionTransformer: lọc feature theo IV và tương quan cao
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import load_config


def _get_default_target_col() -> str:
    try:
        cfg = load_config()
        return cfg.get(
            "data",
            {}
        ).get(
            "target_col",
            "default.payment.next.month"
        )
    except Exception:
        return "default.payment.next.month"


TARGET_COL = _get_default_target_col()
DROP_COLS = ["ID"]


def build_features(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    drop_cols: Optional[List[str]] = None,
    include_sex: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Tách X và y từ DataFrame thô.

    Args:
        df: DataFrame đã được lọc theo split.
        target_col: Tên cột nhãn.
        drop_cols: Danh sách cột cần bỏ.
        include_sex: Nếu False thì bỏ SEX.

    Returns:
        (X, y)
    """
    if target_col is None:
        target_col = TARGET_COL

    if drop_cols is None:
        drop_cols = DROP_COLS

    if not include_sex:
        drop_cols = list(drop_cols) + ["SEX"]

    if target_col not in df.columns:
        raise KeyError(
            f"Không tìm thấy cột nhãn '{target_col}' trong DataFrame."
        )

    cols_to_drop = [
        c for c in drop_cols
        if c in df.columns
    ] + [target_col]

    X = df.drop(columns=cols_to_drop)
    y = df[target_col].astype(int)

    return X, y


# Create credit utilization features from monthly bill amounts.
def create_utilization_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()

    for i in range(1, 7):
        df[f"UTIL_{i}"] = np.where(
            df["LIMIT_BAL"] > 0,
            df[f"BILL_AMT{i}"] / df["LIMIT_BAL"],
            np.nan
        )

    util_cols = [
        f"UTIL_{i}"
        for i in range(1, 7)
    ]

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
            max_count = max(
                max_count,
                current_count
            )
        else:
            current_count = 0

    return max_count


# Create features describing payment delay trends.
def create_payment_trend_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()

    # PAY_6 is the oldest month, PAY_1 is the most recent month.
    pay_cols = [
        f"PAY_{i}"
        for i in range(6, 0, -1)
    ]

    df["PAY_MEAN"] = df[pay_cols].mean(axis=1)
    df["PAY_MAX"] = df[pay_cols].max(axis=1)
    df["PAY_SLOPE"] = df[pay_cols].apply(
        calculate_slope,
        axis=1
    )

    df["PAY_LATE_CONSECUTIVE"] = df[pay_cols].apply(
        max_consecutive_late,
        axis=1
    )

    df["PAY_LATE_2PLUS_COUNT"] = (
        df[pay_cols] >= 2
    ).sum(axis=1)

    return df


# Create payment-to-bill ratio features.
def create_payment_ratio_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()

    for i in range(1, 6):
        bill_col = f"BILL_AMT{i + 1}"
        pay_col = f"PAY_AMT{i}"

        df[f"PAY_RATIO_{i}"] = np.where(
            df[bill_col] > 0,
            df[pay_col] / df[bill_col],
            np.nan
        )

    ratio_cols = [
        f"PAY_RATIO_{i}"
        for i in range(1, 6)
    ]

    df["PAY_RATIO_MEAN"] = df[ratio_cols].mean(axis=1)

    return df


# Create features measuring monthly bill balance variation.
def create_bill_variation_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()

    bill_cols = [
        f"BILL_AMT{i}"
        for i in range(1, 7)
    ]

    df["BILL_STD"] = df[bill_cols].std(axis=1)

    bill_delta = df[bill_cols].diff(axis=1)

    df["BILL_DELTA"] = (
        bill_delta.iloc[:, 1:]
        .abs()
        .mean(axis=1)
    )

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

    min_pay_cols = [
        f"MIN_PAY_FLAG_{i}"
        for i in range(1, 7)
    ]

    df["MIN_PAY_FLAG_COUNT"] = (
        df[min_pay_cols].sum(axis=1)
    )

    return df


# Tên các đặc trưng do FeatureEngineeringTransformer tạo ra.
ENGINEERED_COLUMNS = (
    [f"UTIL_{i}" for i in range(1, 7)]
    + ["UTIL_MEAN"]
    + [
        "PAY_MEAN",
        "PAY_MAX",
        "PAY_SLOPE",
        "PAY_LATE_CONSECUTIVE",
        "PAY_LATE_2PLUS_COUNT",
    ]
    + [f"PAY_RATIO_{i}" for i in range(1, 6)]
    + ["PAY_RATIO_MEAN"]
    + ["BILL_STD", "BILL_DELTA"]
    + [f"MIN_PAY_FLAG_{i}" for i in range(1, 7)]
    + ["MIN_PAY_FLAG_COUNT"]
)


class FeatureEngineeringTransformer(
    BaseEstimator,
    TransformerMixin
):
    """Gói các hàm tạo đặc trưng Tuần 2 - T2."""

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


def _label_missing(
    bins: pd.Series
) -> pd.Series:
    """Gán NaN vào bin riêng MISSING_BIN."""
    if not bins.isna().any():
        return bins

    return bins.astype(object).where(
        bins.notna(),
        MISSING_BIN
    )


def _open_edges(
    edges: np.ndarray
) -> np.ndarray:
    """Mở rộng biên ngoài cùng thành -inf/+inf."""
    edges = np.asarray(
        edges,
        dtype=float
    ).copy()

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


# Calculate Information Value (IV).
def calculate_iv(
    feature: pd.Series,
    target: pd.Series
) -> float:
    data = pd.DataFrame({
        "feature": feature,
        "target": target
    }).dropna()

    grouped = data.groupby(
        "feature",
        observed=False
    )["target"]

    good = grouped.apply(
        lambda x: (x == 0).sum()
    )

    bad = grouped.apply(
        lambda x: (x == 1).sum()
    )

    good_total = good.sum()
    bad_total = bad.sum()

    good_dist = (
        good + 0.5
    ) / (
        good_total + 0.5 * len(good)
    )

    bad_dist = (
        bad + 0.5
    ) / (
        bad_total + 0.5 * len(bad)
    )

    woe = np.log(
        good_dist / bad_dist
    )

    iv = (
        (good_dist - bad_dist) * woe
    ).sum()

    return iv


# Find the best age bin edges based on IV.
def fit_age_bins(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    min_bins: int = 3,
    max_bins: int = 8
) -> np.ndarray:
    age = X_train["AGE"]

    best_iv = -np.inf
    best_edges = None

    for n_bins in range(
        min_bins,
        max_bins + 1
    ):
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

            iv = calculate_iv(
                bins,
                y_train
            )

            if iv > best_iv:
                best_iv = iv
                best_edges = edges

        except ValueError:
            continue

    if best_edges is None:
        raise ValueError(
            "Unable to create age bins from training data."
        )

    return _open_edges(best_edges)


# Apply age bins learned from training data.
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


# Fit age bins on training data and apply them.
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


class AgeBinningTransformer(
    BaseEstimator,
    TransformerMixin
):
    """Transformer tự động phân nhóm AGE tối ưu theo IV."""

    def __init__(
        self,
        min_bins: int = 3,
        max_bins: int = 8
    ):
        self.min_bins = min_bins
        self.max_bins = max_bins

    def fit(self, X, y=None):
        X = X.copy()

        if y is None:
            self.age_edges_ = None
            return self

        y = pd.Series(
            y,
            index=X.index
        )

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

        if (
            self.age_edges_ is None
            or "AGE" not in X.columns
        ):
            return X

        X["AGE_BIN"] = pd.cut(
            X["AGE"],
            bins=self.age_edges_,
            include_lowest=True
        )

        X = X.drop(
            columns=["AGE"]
        )

        return X


class WoEIVTransformer(
    BaseEstimator,
    TransformerMixin
):
    """Transformer mã hóa Weight of Evidence (WoE)
    và lọc theo Information Value (IV).
    """

    def __init__(
        self,
        n_bins: int = 5,
        iv_threshold: float = 0.02,
        max_categories: int = 12
    ):
        self.n_bins = n_bins
        self.iv_threshold = iv_threshold
        self.max_categories = max_categories

    def fit(self, X, y):
        X = X.copy()

        y = pd.Series(
            y,
            index=X.index
        )

        self.bin_edges_ = {}
        self.category_maps_ = {}
        self.woe_maps_ = {}
        self.iv_values_ = {}

        for col in X.columns:
            try:
                is_continuous = (
                    pd.api.types.is_numeric_dtype(X[col])
                    and X[col].nunique()
                    > self.max_categories
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

                    bins = _label_missing(
                        pd.cut(
                            X[col],
                            bins=edges,
                            include_lowest=True
                        )
                    )

                    self.bin_edges_[col] = edges

                else:
                    bins = X[col].astype(str)

                    self.category_maps_[col] = (
                        bins.unique().tolist()
                    )

                data = pd.DataFrame({
                    "bin": bins,
                    "target": y
                })

                grouped = data.groupby(
                    "bin",
                    observed=False,
                    sort=False
                )["target"]

                good = grouped.apply(
                    lambda x: (x == 0).sum()
                )

                bad = grouped.apply(
                    lambda x: (x == 1).sum()
                )

                good_dist = (
                    good + 0.5
                ) / (
                    good.sum()
                    + 0.5 * len(good)
                )

                bad_dist = (
                    bad + 0.5
                ) / (
                    bad.sum()
                    + 0.5 * len(bad)
                )

                woe = np.log(
                    good_dist / bad_dist
                )

                iv = (
                    (good_dist - bad_dist)
                    * woe
                ).sum()

                self.woe_maps_[col] = (
                    woe.to_dict()
                )

                self.iv_values_[col] = iv

            except (
                TypeError,
                ValueError
            ):
                continue

        self.selected_features_ = [
            col
            for col, iv in self.iv_values_.items()
            if iv >= self.iv_threshold
        ]

        return self

    def transform(self, X):
        X = X.copy()

        result = pd.DataFrame(
            index=X.index
        )

        for col in self.selected_features_:
            if col in self.bin_edges_:
                bins = _label_missing(
                    pd.cut(
                        X[col],
                        bins=self.bin_edges_[col],
                        include_lowest=True
                    )
                )
            else:
                bins = X[col].astype(str)

            mapped = bins.map(
                self.woe_maps_[col]
            )

            result[col] = pd.to_numeric(
                mapped,
                errors="coerce"
            ).fillna(0.0)

        return result


class FeatureSelectionTransformer(
    BaseEstimator,
    TransformerMixin
):
    """Lọc feature theo IV và tương quan cao.

    Quy trình:
    1. Tính IV trên tập train.
    2. Loại feature có IV < iv_threshold.
    3. Với các feature số có |correlation| > correlation_threshold,
       giữ feature có IV cao hơn.
    4. Lưu selected_features_ để dùng cho transform.

    Transformer này được thiết kế để đặt trong sklearn Pipeline.
    Khi Pipeline được dùng trong cross-validation, fit() chỉ chạy
    trên training fold nên tránh leakage.
    """

    def __init__(
        self,
        iv_threshold: float = 0.02,
        correlation_threshold: float = 0.9,
        n_bins: int = 5,
    ):
        self.iv_threshold = iv_threshold
        self.correlation_threshold = correlation_threshold
        self.n_bins = n_bins

    def fit(self, X, y):
        X = X.copy()

        y = pd.Series(
            y,
            index=X.index
        )

        # Tính IV cho toàn bộ feature.
        iv_transformer = WoEIVTransformer(
            n_bins=self.n_bins,
            iv_threshold=0.0,
        )

        iv_transformer.fit(
            X,
            y
        )

        self.iv_values_ = (
            iv_transformer.iv_values_.copy()
        )

        # Bước 1: lọc theo IV.
        selected = [
            column
            for column in X.columns
            if self.iv_values_.get(
                column,
                0.0
            ) >= self.iv_threshold
        ]

        # Bước 2: xử lý tương quan cao.
        # Chỉ xét các feature numeric.
        numeric_selected = X[
            selected
        ].select_dtypes(
            include=[np.number]
        )

        if numeric_selected.shape[1] >= 2:
            to_drop = find_high_correlation_features(
                numeric_selected,
                y,
                threshold=self.correlation_threshold,
            )

            selected = [
                column
                for column in selected
                if column not in to_drop
            ]

        self.selected_features_ = selected

        return self

    def transform(self, X):
        X = X.copy()

        return X.loc[
            :,
            [
                column
                for column in self.selected_features_
                if column in X.columns
            ]
        ]


def prepare_feature_selection_data(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    preprocessed: bool = False,
    drop_sensitive: bool = True,
) -> pd.DataFrame:
    """Chuẩn bị dữ liệu theo đúng preprocessing của scorecard.

    Nếu preprocessed=True, dữ liệu được giữ nguyên.

    Với dữ liệu không phải credit-scoring dataset, giữ nguyên dữ liệu
    để các hàm feature selection có thể dùng với dữ liệu test/toy.

    Với dữ liệu credit-scoring raw, áp dụng:
    1. AbnormalCodeTransformer
    2. FeatureEngineeringTransformer
    3. AgeBinningTransformer
    4. DropColumnsTransformer cho SEX và ID
    """
    X = X.copy()

    if preprocessed:
        return X

    # Dữ liệu toy/test hoặc dữ liệu không phải credit scoring:
    # giữ nguyên để các hàm feature selection vẫn hoạt động.
    is_credit_dataset = (
        "LIMIT_BAL" in X.columns
        and "BILL_AMT1" in X.columns
    )

    if not is_credit_dataset:
        return X

    if y is None:
        raise ValueError(
            "y is required when preparing raw credit-scoring data."
        )

    from src.preprocessing import (
        AbnormalCodeTransformer,
        DropColumnsTransformer,
        SENSITIVE_COLUMNS,
    )

    # 1. Xử lý mã bất thường và PAY_0 -> PAY_1.
    X = AbnormalCodeTransformer().fit_transform(
        X,
        y,
    )

    # 2. Feature engineering.
    X = FeatureEngineeringTransformer().fit_transform(
        X,
        y,
    )

    # 3. Binning AGE.
    X = AgeBinningTransformer(
        min_bins=3,
        max_bins=8,
    ).fit_transform(
        X,
        y,
    )

    # 4. Loại biến nhạy cảm giống scorecard pipeline.
    if drop_sensitive:
        X = DropColumnsTransformer(
            columns=SENSITIVE_COLUMNS,
        ).fit_transform(
            X,
            y,
        )

    return X

def find_high_correlation_features(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.9,
) -> list[str]:
    """Tìm các feature tương quan cao và loại feature có IV thấp hơn.

    Với mỗi cặp feature có |correlation| > threshold:
    - giữ feature có IV cao hơn;
    - loại feature có IV thấp hơn;
    - nếu IV bằng nhau, dùng tên feature để quyết định ổn định.
    """
    X = X.copy()

    y = pd.Series(
        y
    ).reset_index(
        drop=True
    )

    X = X.reset_index(
        drop=True
    )

    numeric_X = X.select_dtypes(
        include=[np.number]
    )

    if numeric_X.shape[1] < 2:
        return []

    # Tính IV trên đúng tập feature hiện tại.
    iv_transformer = WoEIVTransformer(
        n_bins=5,
        iv_threshold=0.0,
    )

    iv_transformer.fit(
        X,
        y
    )

    iv_values = (
        iv_transformer.iv_values_
    )

    corr_matrix = (
        numeric_X
        .corr()
        .abs()
    )

    pairs = []
    columns = numeric_X.columns

    for i in range(len(columns)):
        for j in range(
            i + 1,
            len(columns)
        ):
            feature_a = columns[i]
            feature_b = columns[j]

            correlation = corr_matrix.loc[
                feature_a,
                feature_b
            ]

            if (
                pd.notna(correlation)
                and correlation > threshold
            ):
                pairs.append(
                    (
                        correlation,
                        feature_a,
                        feature_b
                    )
                )

    # Xét cặp có correlation cao nhất trước.
    pairs.sort(
        reverse=True
    )

    to_drop = set()

    for (
        _,
        feature_a,
        feature_b
    ) in pairs:
        if (
            feature_a in to_drop
            or feature_b in to_drop
        ):
            continue

        iv_a = iv_values.get(
            feature_a,
            0.0
        )

        iv_b = iv_values.get(
            feature_b,
            0.0
        )

        if iv_a < iv_b:
            to_drop.add(
                feature_a
            )

        elif iv_b < iv_a:
            to_drop.add(
                feature_b
            )

        else:
            # Tie-break ổn định,
            # không phụ thuộc thứ tự cột.
            to_drop.add(
                max(
                    feature_a,
                    feature_b
                )
            )

    return sorted(to_drop)


def compute_iv_stability(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Tính IV của từng feature trên từng fold CV.

    Với dữ liệu credit-scoring raw, toàn bộ preprocessing có học tham số
    được fit riêng trên training fold để tránh data leakage.
    """
    from sklearn.model_selection import StratifiedKFold

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_ivs = []

    for fold, (train_idx, _) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        X_fold = X.iloc[train_idx].copy()
        y_fold = y.iloc[train_idx].copy()

        X_fold = prepare_feature_selection_data(
            X_fold,
            y=y_fold,
        )

        transformer = WoEIVTransformer(
            n_bins=5,
            iv_threshold=0.0,
        )

        transformer.fit(
            X_fold,
            y_fold,
        )

        fold_iv = transformer.iv_values_.copy()

        for feature, iv in fold_iv.items():
            fold_ivs.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "iv": iv,
                }
            )

    fold_df = pd.DataFrame(fold_ivs)

    result = (
        fold_df
        .groupby("feature")["iv"]
        .agg(
            iv_mean="mean",
            iv_std="std",
            iv_min="min",
            iv_max="max",
        )
        .reset_index()
    )

    return result


def compute_feature_importance_stability(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    correlation_threshold: float = 0.9,
) -> pd.DataFrame:
    """Đánh giá độ ổn định của feature importance qua các fold CV.

    Toàn bộ preprocessing và feature selection được thực hiện
    riêng trên training fold để tránh data leakage.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_importances = []

    for fold, (train_idx, _) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()

        # Preprocessing chỉ fit trên training fold.
        X_train = prepare_feature_selection_data(
            X_train,
            y=y_train,
        )

        # Chỉ sử dụng biến số.
        X_numeric = X_train.select_dtypes(
            include=[np.number]
        )

        # Xử lý giá trị vô hạn và missing.
        X_numeric = X_numeric.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X_numeric = X_numeric.fillna(
            X_numeric.median()
        )

        # Lọc biến tương quan cao trên chính training fold.
        # Giữ feature có IV cao hơn.
        to_drop = find_high_correlation_features(
            X_train,
            y_train,
            threshold=correlation_threshold,
        )

        X_numeric = X_numeric.drop(
            columns=to_drop,
            errors="ignore",
        )

        # Chuẩn hóa trên training fold.
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_numeric
        )

        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        )

        model.fit(
            X_train_scaled,
            y_train,
        )

        importance = np.abs(
            model.coef_[0]
        )

        fold_importances.append(
            pd.Series(
                importance,
                index=X_numeric.columns,
                name=f"fold_{fold}",
            )
        )

    importance_df = pd.DataFrame(
        fold_importances
    )

    result = pd.DataFrame({
        "feature": importance_df.columns,
        "importance_mean": (
            importance_df
            .mean(axis=0)
            .values
        ),
        "importance_std": (
            importance_df
            .std(axis=0)
            .fillna(0.0)
            .values
        ),
        "importance_min": (
            importance_df
            .min(axis=0)
            .values
        ),
        "importance_max": (
            importance_df
            .max(axis=0)
            .values
        ),
    })

    return (
        result
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )