import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


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


# Calculate Information Value (IV) for a categorical or binned feature.
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
    
    good = grouped.apply(lambda x: (x == 0).sum())
    bad = grouped.apply(lambda x: (x == 1).sum())

    good_total = good.sum()
    bad_total = bad.sum()

    good_dist = (good + 0.5) / (
        good_total + 0.5 * len(good)
    )

    bad_dist = (bad + 0.5) / (
        bad_total + 0.5 * len(bad)
    )

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
        raise ValueError(
            "Unable to create age bins from training data."
        )

    return best_edges


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
    def __init__(self, min_bins=3, max_bins=8):
        self.min_bins = min_bins
        self.max_bins = max_bins

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y, index=X.index)

        # Some unit tests use data without AGE.
        # In that case, skip age binning.
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

        # Skip age binning if AGE was not available during fitting.
        if self.age_edges_ is None:
            return X

        X["AGE_BIN"] = pd.cut(
            X["AGE"],
            bins=self.age_edges_,
            include_lowest=True
        )

        X = X.drop(columns=["AGE"])

        return X


# Transform numerical and categorical features into Weight of Evidence (WoE)
# and select features based on Information Value (IV).
class WoEIVTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5, iv_threshold=0.02):
        self.n_bins = n_bins
        self.iv_threshold = iv_threshold

    # Learn binning rules, WoE values, and IV from training data.
    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y, index=X.index)

        self.bin_edges_ = {}
        self.category_maps_ = {}
        self.woe_maps_ = {}
        self.iv_values_ = {}

        # Process each feature independently.
        for col in X.columns:
            try:
                if pd.api.types.is_numeric_dtype(X[col]):
                    # Create quantile-based bins for numerical features.
                    _, edges = pd.qcut(
                        X[col],
                        q=self.n_bins,
                        retbins=True,
                        duplicates="drop"
                    )

                    edges = np.unique(edges)

                    if len(edges) < 2:
                        continue

                    bins = pd.cut(
                        X[col],
                        bins=edges,
                        include_lowest=True
                    )

                    self.bin_edges_[col] = edges

                else:
                    # Treat each category as a separate bin.
                    bins = X[col].astype(str)
                    self.category_maps_[col] = bins.unique().tolist()

                data = pd.DataFrame({
                    "bin": bins,
                    "target": y
                })

                # Count good (target=0) and bad (target=1) samples.
                grouped = data.groupby(
                    "bin",
                    observed=False
                )["target"]

                good = grouped.apply(
                    lambda x: (x == 0).sum()
                )

                bad = grouped.apply(
                    lambda x: (x == 1).sum()
                )

                # Calculate smoothed distributions.
                good_dist = (good + 0.5) / (
                    good.sum() + 0.5 * len(good)
                )

                bad_dist = (bad + 0.5) / (
                    bad.sum() + 0.5 * len(bad)
                )

                # Calculate WoE.
                woe = np.log(
                    good_dist / bad_dist
                )

                # Calculate IV.
                iv = (
                    (good_dist - bad_dist) * woe
                ).sum()

                self.woe_maps_[col] = woe.to_dict()
                self.iv_values_[col] = iv

            except (TypeError, ValueError):
                continue

        # Keep only features whose IV reaches the threshold.
        self.selected_features_ = [
            col
            for col, iv in self.iv_values_.items()
            if iv >= self.iv_threshold
        ]

        return self

    # Apply the learned binning and WoE transformation.
    def transform(self, X):
        X = X.copy()

        result = pd.DataFrame(index=X.index)

        for col in self.selected_features_:
            if col in self.bin_edges_:
                # Transform numerical features using learned bins.
                bins = pd.cut(
                    X[col],
                    bins=self.bin_edges_[col],
                    include_lowest=True
                )
            else:
                # Transform categorical features using learned categories.
                bins = X[col].astype(str)

            mapped = bins.map(
                self.woe_maps_[col]
            )

            result[col] = pd.to_numeric(
                mapped,
                errors="coerce"
            ).fillna(0.0)

        return result