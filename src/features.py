import numpy as np
import pandas as pd


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

    pay_cols = [f"PAY_{i}" for i in range(1, 7)]

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

    grouped = data.groupby("feature")["target"]

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