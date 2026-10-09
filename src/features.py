"""Module xây dựng đặc trưng cho Credit Scoring.

Bao gồm:
- build_features() tách X, y (Pipeline mô hình nằm ở src.pipelines.make_pipeline).
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
- Tuần 2 - T4 (Lọc đặc trưng & độ ổn định):
    - find_high_correlation_features: lọc tương quan (Spearman), giữ biến IV cao hơn
    - FeatureSelectionTransformer: lọc theo IV + tương quan trong từng fold
    - compute_iv_stability / compute_feature_importance_stability: độ ổn định qua các fold CV
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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
    include_sex: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Tách X và y từ DataFrame thô (cột gốc; tiền xử lý nằm trong src.pipelines.make_pipeline).

    Args:
        df: DataFrame đã được lọc theo split (train / valid / test).
        target_col: Tên cột nhãn (nếu None sẽ đọc từ config data.target_col).
        drop_cols: Danh sách cột bổ sung cần bỏ (mặc định: ["ID"]).
        include_sex: Nếu False (mô hình chính thức) bỏ SEX; True để đối chiếu fairness.

    Returns:
        (X, y): DataFrame đặc trưng và Series nhãn nhị phân (int).

    Raises:
        KeyError: Nếu target_col không tồn tại trong df.
    """
    if target_col is None:
        target_col = TARGET_COL
    if drop_cols is None:
        drop_cols = DROP_COLS
    if not include_sex:
        drop_cols = list(drop_cols) + ["SEX"]

    if target_col not in df.columns:
        raise KeyError(f"Không tìm thấy cột nhãn '{target_col}' trong DataFrame.")

    cols_to_drop = [c for c in drop_cols if c in df.columns] + [target_col]
    X = df.drop(columns=cols_to_drop)
    y = df[target_col].astype(int)
    return X, y


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


def _bin_stats(x: np.ndarray, y: np.ndarray, edges: np.ndarray):
    """Số mẫu và tỷ lệ bad của từng bin (x không có NaN)."""
    idx = np.searchsorted(edges[1:-1], x, side="left")
    counts = np.bincount(idx, minlength=len(edges) - 1)
    bads = np.bincount(idx, weights=y, minlength=len(edges) - 1)
    return counts, bads / np.maximum(counts, 1)


def _coarse_edges(x, y, edges, min_bin_share=None, monotonic=False) -> np.ndarray:
    """Coarse classing: gộp bin liền kề cho đến khi mỗi bin có >= min_bin_share mẫu và
    (nếu monotonic) tỷ lệ bad đơn điệu theo chiều tương quan của biến với nhãn.

    edges là biên đã mở rộng ±inf; mỗi lần gộp bỏ một biên trong. Bin NaN xử lý riêng.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.asarray(edges, dtype=float)

    def merge(i):  # gộp bin i và i+1
        return np.delete(edges, i + 1)

    if min_bin_share:
        min_count = min_bin_share * len(x)
        while len(edges) > 2:
            counts, rates = _bin_stats(x, y, edges)
            i = int(np.argmin(counts))
            if counts[i] >= min_count:
                break
            if i == 0:
                edges = merge(0)
            elif i == len(counts) - 1:
                edges = merge(i - 1)
            else:  # gộp với bin kề có tỷ lệ bad gần hơn
                left = abs(rates[i] - rates[i - 1]) <= abs(rates[i] - rates[i + 1])
                edges = merge(i - 1 if left else i)

    if monotonic and len(edges) > 2:
        direction = np.sign(pd.Series(x).corr(pd.Series(y), method="spearman")) or 1.0
        while len(edges) > 2:
            _, rates = _bin_stats(x, y, edges)
            violations = np.where(direction * np.diff(rates) < 0)[0]
            if len(violations) == 0:
                break
            edges = merge(int(violations[0]))
    return edges


class WoEIVTransformer(BaseEstimator, TransformerMixin):
    """Transformer mã hóa Weight of Evidence (WoE) và lọc theo Information Value (IV).

    Biến số có <= max_categories giá trị khác nhau (cờ 0/1, EDUCATION, PAY_*) được xử lý như
    biến phân loại (mỗi giá trị một bin) vì qcut sẽ gộp chúng thành quá ít bin.
    NaN của biến liên tục được gán vào bin MISSING_BIN riêng (có WoE riêng).

    Coarse classing (Tuần 2 - T5, dùng cho Logistic Scorecard; mặc định tắt):
        min_bin_share: mỗi bin có tối thiểu tỷ lệ mẫu này; bin nhỏ gộp với bin kề. Biến số rời rạc
            khi bật coarse classing được chia theo khoảng giữa các giá trị, nên giá trị chưa gặp khi
            fit (vd PAY_x = 8 chỉ có ở Valid) rơi vào bin gần nhất thay vì nhận WoE = 0.
        monotonic: với biến liên tục, gộp bin kề đến khi tỷ lệ bad đơn điệu theo chiều tương quan
            Spearman với nhãn.
    """

    def __init__(
        self,
        n_bins: int = 5,
        iv_threshold: float = 0.02,
        max_categories: int = 12,
        min_bin_share: Optional[float] = None,
        monotonic: bool = False,
    ):
        self.n_bins = n_bins
        self.iv_threshold = iv_threshold
        self.max_categories = max_categories
        self.min_bin_share = min_bin_share
        self.monotonic = monotonic

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y, index=X.index)

        self.bin_edges_ = {}
        self.category_maps_ = {}
        self.woe_maps_ = {}
        self.iv_values_ = {}

        coarse = bool(self.min_bin_share) or self.monotonic

        for col in X.columns:
            try:
                is_numeric = pd.api.types.is_numeric_dtype(X[col])
                is_continuous = is_numeric and X[col].nunique() > self.max_categories
                if is_continuous or (coarse and is_numeric):
                    if is_continuous:
                        _, edges = pd.qcut(
                            X[col],
                            q=self.n_bins,
                            retbins=True,
                            duplicates="drop"
                        )
                        edges = np.unique(edges)
                    else:
                        # Biến rời rạc: mỗi giá trị một bin, biên đặt giữa hai giá trị liền kề.
                        values = np.sort(X[col].dropna().unique()).astype(float)
                        mids = (values[:-1] + values[1:]) / 2
                        edges = np.concatenate([[values[0]], mids, [values[-1]]])
                    if len(edges) < 2:
                        continue
                    edges = _open_edges(edges)
                    if coarse:
                        observed = X[col].notna()
                        # Chỉ ép đơn điệu cho biến liên tục: mã rời rạc như PAY_x (-2, -1, 0 là các
                        # trạng thái khác nhau, không phải thang tăng dần) giữ dạng quan hệ thực tế.
                        edges = _coarse_edges(
                            X.loc[observed, col],
                            y[observed],
                            edges,
                            self.min_bin_share,
                            self.monotonic and is_continuous,
                        )
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


# ---------------------------------------------------------------------------
# Tuần 2 - T4: Lọc đặc trưng (IV + tương quan) và độ ổn định qua các fold CV
# ---------------------------------------------------------------------------

def find_high_correlation_features(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.9,
    method: str = "spearman",
    return_details: bool = False,
):
    """Tìm biến tương quan cao (|corr| > threshold); trong mỗi cặp giữ biến có IV cao hơn.

    Mặc định dùng Spearman: các biến tỷ lệ (PAY_RATIO_*, UTIL_*) có đuôi rất dài, Pearson bị
    vài giá trị cực lớn chi phối (vd PAY_RATIO_1 – PAY_RATIO_3 đạt 0.9997 theo Pearson).
    Cặp có tương quan cao hơn được xử lý trước; IV bằng nhau thì loại biến có tên lớn hơn.

    Returns:
        List biến bị loại, hoặc (list, details) nếu return_details=True.
    """
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    numeric_X = X.select_dtypes(include=[np.number])

    if numeric_X.shape[1] < 2:
        return ([], []) if return_details else []

    iv_values = WoEIVTransformer(iv_threshold=0.0).fit(numeric_X, y).iv_values_
    corr_matrix = numeric_X.corr(method=method).abs()
    columns = numeric_X.columns

    pairs = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            corr = corr_matrix.iloc[i, j]
            if pd.notna(corr) and corr > threshold:
                pairs.append((float(corr), columns[i], columns[j]))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    to_drop = set()
    details = []
    for corr, feature_a, feature_b in pairs:
        if feature_a in to_drop or feature_b in to_drop:
            continue
        iv_a, iv_b = iv_values.get(feature_a, 0.0), iv_values.get(feature_b, 0.0)
        if iv_a != iv_b:
            dropped, kept = (feature_a, feature_b) if iv_a < iv_b else (feature_b, feature_a)
        else:
            dropped, kept = max(feature_a, feature_b), min(feature_a, feature_b)
        to_drop.add(dropped)
        details.append({
            "feature": dropped,
            "iv": iv_values.get(dropped, float("nan")),
            "reason": "Tương quan cao",
            "correlated_with": kept,
            "correlation": corr,
        })

    result = sorted(to_drop)
    return (result, details) if return_details else result


class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    """Lọc đặc trưng theo IV rồi theo tương quan, fit trên dữ liệu train của từng fold.

    passthrough_columns (vd SEX ở bản đối chiếu fairness) được giữ nguyên, không tham gia lọc.
    Thuộc tính sau fit: iv_values_, selected_features_, passthrough_features_,
    dropped_features_ (list dict: feature, iv, reason, correlated_with, correlation).
    """

    def __init__(
        self,
        iv_threshold: float = 0.02,
        correlation_threshold: float = 0.9,
        n_bins: int = 5,
        passthrough_columns: tuple = (),
        enable_correlation_filter: bool = True,
    ):
        self.iv_threshold = iv_threshold
        self.correlation_threshold = correlation_threshold
        self.n_bins = n_bins
        self.passthrough_columns = passthrough_columns
        self.enable_correlation_filter = enable_correlation_filter

    def fit(self, X, y):
        y = pd.Series(y, index=X.index)
        self.passthrough_features_ = [c for c in self.passthrough_columns if c in X.columns]
        X_selection = X.drop(columns=self.passthrough_features_)

        self.iv_values_ = (
            WoEIVTransformer(n_bins=self.n_bins, iv_threshold=0.0).fit(X_selection, y).iv_values_.copy()
        )
        selected = [c for c in X_selection.columns if self.iv_values_.get(c, 0.0) >= self.iv_threshold]
        dropped = [
            {
                "feature": c,
                "iv": self.iv_values_.get(c, float("nan")),
                "reason": "IV thấp",
                "correlated_with": "",
                "correlation": float("nan"),
            }
            for c in X_selection.columns
            if c not in selected
        ]

        if self.enable_correlation_filter:
            to_drop, details = find_high_correlation_features(
                X_selection[selected],
                y,
                threshold=self.correlation_threshold,
                return_details=True,
            )
            dropped.extend(details)
            selected = [c for c in selected if c not in to_drop]

        self.selected_features_ = selected
        self.dropped_features_ = dropped
        return self

    def transform(self, X):
        columns = self.selected_features_ + self.passthrough_features_
        return X.loc[:, [c for c in columns if c in X.columns]].copy()


def _fit_frame(frame_pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Fit bản sao frame_pipeline trên (X, y) của fold hiện tại; None → giữ nguyên X."""
    if frame_pipeline is None:
        return X
    from sklearn.base import clone

    return clone(frame_pipeline).fit_transform(X, y)


def compute_iv_stability(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    frame_pipeline=None,
) -> pd.DataFrame:
    """IV của từng biến trên phần train của từng fold CV: mean, std, min, max, cv (= std/mean).

    frame_pipeline: Pipeline chưa fit, biến cột gốc thành bộ đặc trưng cần đánh giá
    (src.pipelines.build_feature_frame_pipeline). Được clone và fit lại trong từng fold,
    nên các bước học từ nhãn (age binning) không dùng dữ liệu ngoài fold.
    """
    from sklearn.model_selection import StratifiedKFold

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for fold, (train_idx, _) in enumerate(cv.split(X, y), start=1):
        X_fold, y_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_fold = _fit_frame(frame_pipeline, X_fold, y_fold)
        iv_values = WoEIVTransformer(iv_threshold=0.0).fit(X_fold, y_fold).iv_values_
        rows += [{"fold": fold, "feature": f, "iv": iv} for f, iv in iv_values.items()]

    result = (
        pd.DataFrame(rows)
        .groupby("feature")["iv"]
        .agg(iv_mean="mean", iv_std="std", iv_min="min", iv_max="max")
        .reset_index()
    )
    result["iv_cv"] = result["iv_std"] / result["iv_mean"]
    return result.sort_values("iv_mean", ascending=False).reset_index(drop=True)


def compute_feature_importance_stability(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    correlation_threshold: float = 0.9,
    frame_pipeline=None,
) -> pd.DataFrame:
    """|hệ số| Logistic Regression (biến numeric đã chuẩn hoá) qua các fold CV.

    Lọc tương quan được thực hiện trong từng fold nên một biến có thể chỉ được chọn ở một số fold;
    n_folds_selected ghi nhận số fold đó, và std chỉ có nghĩa khi n_folds_selected >= 2.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_importances = []
    for fold, (train_idx, _) in enumerate(cv.split(X, y), start=1):
        X_fold, y_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_fold = _fit_frame(frame_pipeline, X_fold, y_fold)

        X_numeric = X_fold.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        X_numeric = X_numeric.fillna(X_numeric.median())
        to_drop = find_high_correlation_features(X_numeric, y_fold, threshold=correlation_threshold)
        X_numeric = X_numeric.drop(columns=to_drop)
        if X_numeric.empty:
            raise ValueError("Không còn biến numeric để tính feature importance.")

        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(StandardScaler().fit_transform(X_numeric), y_fold)
        fold_importances.append(pd.Series(np.abs(model.coef_[0]), index=X_numeric.columns, name=fold))

    importance_df = pd.DataFrame(fold_importances)
    result = pd.DataFrame({
        "feature": importance_df.columns,
        "n_folds_selected": importance_df.notna().sum(axis=0).values,
        "importance_mean": importance_df.mean(axis=0).values,
        "importance_std": importance_df.std(axis=0).values,
        "importance_min": importance_df.min(axis=0).values,
        "importance_max": importance_df.max(axis=0).values,
    })
    return result.sort_values("importance_mean", ascending=False).reset_index(drop=True)
