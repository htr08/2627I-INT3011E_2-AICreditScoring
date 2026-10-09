import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from src.features import (
    create_utilization_features,
    calculate_slope,
    max_consecutive_late,
    create_payment_trend_features,
    create_payment_ratio_features,
    create_bill_variation_features,
    create_min_payment_features,
    calculate_iv,
    fit_age_bins,
    transform_age_bins,
    create_age_bin_features,
    WoEIVTransformer,
    FeatureEngineeringTransformer,
    ENGINEERED_COLUMNS,
    find_high_correlation_features,
    compute_feature_importance_stability,
    FeatureSelectionTransformer,
)
from src.pipelines import build_feature_frame_pipeline, build_scorecard_pipeline

# Test credit utilization features.
def test_create_utilization_features():
    df = pd.DataFrame({
        "LIMIT_BAL": [100000, 200000],
        "BILL_AMT1": [50000, 100000],
        "BILL_AMT2": [40000, 80000],
        "BILL_AMT3": [30000, 60000],
        "BILL_AMT4": [20000, 40000],
        "BILL_AMT5": [10000, 20000],
        "BILL_AMT6": [5000, 10000],
    })

    result = create_utilization_features(df)

    assert result["UTIL_1"].iloc[0] == 0.5
    assert result["UTIL_MEAN"].iloc[0] == 0.25833333333333336


# Test payment trend features.
def test_create_payment_trend_features():
    df = pd.DataFrame({
        "PAY_1": [0],
        "PAY_2": [1],
        "PAY_3": [2],
        "PAY_4": [2],
        "PAY_5": [3],
        "PAY_6": [0],
    })

    result = create_payment_trend_features(df)

    assert result["PAY_MEAN"].iloc[0] == 1.3333333333333333
    assert result["PAY_MAX"].iloc[0] == 3
    assert result["PAY_LATE_CONSECUTIVE"].iloc[0] == 3
    assert result["PAY_LATE_2PLUS_COUNT"].iloc[0] == 3


# Test slope calculation.
def test_calculate_slope():
    values = pd.Series([0, 1, 2, 3, 4, 5])

    result = calculate_slope(values)

    assert np.isclose(result, 1.0)


# Test maximum consecutive late payments.
def test_max_consecutive_late():
    values = pd.Series([0, 2, 2, 1, 3, 3, 3])

    result = max_consecutive_late(values)

    assert result == 3


# Test payment-to-bill ratio features.
def test_create_payment_ratio_features():
    df = pd.DataFrame({
        "PAY_AMT1": [500],
        "PAY_AMT2": [1000],
        "PAY_AMT3": [1500],
        "PAY_AMT4": [2000],
        "PAY_AMT5": [2500],
        "BILL_AMT2": [1000],
        "BILL_AMT3": [2000],
        "BILL_AMT4": [3000],
        "BILL_AMT5": [4000],
        "BILL_AMT6": [5000],
    })

    result = create_payment_ratio_features(df)

    assert result["PAY_RATIO_1"].iloc[0] == 0.5
    assert result["PAY_RATIO_2"].iloc[0] == 0.5
    assert result["PAY_RATIO_3"].iloc[0] == 0.5


# Test bill balance variation features.
def test_create_bill_variation_features():
    df = pd.DataFrame({
        "BILL_AMT1": [100],
        "BILL_AMT2": [200],
        "BILL_AMT3": [300],
        "BILL_AMT4": [400],
        "BILL_AMT5": [500],
        "BILL_AMT6": [600],
    })

    result = create_bill_variation_features(df)

    assert result["BILL_STD"].iloc[0] > 0
    assert result["BILL_DELTA"].iloc[0] == 100


# Test minimum payment flags.
def test_create_min_payment_features():
    df = pd.DataFrame({
        "BILL_AMT1": [1000],
        "BILL_AMT2": [1000],
        "BILL_AMT3": [1000],
        "BILL_AMT4": [1000],
        "BILL_AMT5": [1000],
        "BILL_AMT6": [1000],
        "PAY_AMT1": [50],
        "PAY_AMT2": [100],
        "PAY_AMT3": [200],
        "PAY_AMT4": [20],
        "PAY_AMT5": [500],
        "PAY_AMT6": [80],
    })

    result = create_min_payment_features(df)

    assert result["MIN_PAY_FLAG_1"].iloc[0] == 1
    assert result["MIN_PAY_FLAG_2"].iloc[0] == 1
    assert result["MIN_PAY_FLAG_3"].iloc[0] == 0
    assert result["MIN_PAY_FLAG_COUNT"].iloc[0] == 4


# Test Information Value calculation.
def test_calculate_iv():
    feature = pd.Series(["A", "A", "B", "B"])
    target = pd.Series([0, 1, 0, 1])

    result = calculate_iv(feature, target)

    assert result >= 0
    assert np.isfinite(result)


# Test age bin fitting using training data.
def test_fit_age_bins():
    X_train = pd.DataFrame({
        "AGE": [20, 22, 25, 28, 30, 32, 35, 40, 45, 50]
    })

    y_train = pd.Series([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])

    edges = fit_age_bins(X_train, y_train)

    assert edges is not None
    assert len(edges) >= 3
    assert np.all(np.diff(edges) > 0)


# Test applying age bins.
def test_transform_age_bins():
    df = pd.DataFrame({
        "AGE": [21, 30, 45]
    })

    edges = np.array([19, 25, 35, 60])

    result = transform_age_bins(df, edges)

    assert "AGE_BIN" in result.columns
    assert result["AGE_BIN"].notna().all()


# Test complete age bin feature pipeline.
def test_create_age_bin_features():
    X_train = pd.DataFrame({
        "AGE": [20, 22, 25, 28, 30, 32, 35, 40, 45, 50]
    })

    y_train = pd.Series([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])

    df = pd.DataFrame({
        "AGE": [21, 27, 36, 48]
    })

    result, edges = create_age_bin_features(
        X_train,
        y_train,
        df
    )

    assert "AGE_BIN" in result.columns
    assert len(edges) >= 3
    assert result["AGE_BIN"].notna().all()

def test_woe_iv_transformer():
    X = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })

    y = pd.Series([
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    ])

    transformer = WoEIVTransformer(
        n_bins=2,
        iv_threshold=0.0
    )

    result = transformer.fit_transform(X, y)

    # Check that IV is calculated for the feature.
    assert "feature" in transformer.iv_values_

    # Check that the feature passes the IV threshold.
    assert "feature" in transformer.selected_features_

    # Check that the transformed data keeps the same shape.
    assert result.shape == X.shape

    # Check that the WoE values are numeric and not missing.
    assert result["feature"].notna().all()


def test_woe_iv_transformer_categorical():
    X = pd.DataFrame({
        "education": [
            "low", "low", "low", "medium", "medium",
            "high", "high", "high", "high", "high"
        ]
    })

    y = pd.Series([
        0, 0, 0, 0, 1,
        1, 1, 1, 1, 1
    ])

    transformer = WoEIVTransformer(
        iv_threshold=0.0
    )

    result = transformer.fit_transform(X, y)

    # Check that IV is calculated for the categorical feature.
    assert "education" in transformer.iv_values_

    # Check that the feature passes the IV threshold.
    assert "education" in transformer.selected_features_

    # Check that the transformed values are numeric.
    assert pd.api.types.is_numeric_dtype(
        result["education"]
    )

    # Check that the transformed values are not missing.
    assert result["education"].notna().all()


# Values outside the training range must fall into the first/last bin, not WoE = 0.
def test_woe_out_of_range_uses_edge_bins():
    X = pd.DataFrame({"feature": list(range(1, 21))})
    y = pd.Series([0] * 10 + [1] * 10)

    transformer = WoEIVTransformer(n_bins=2, iv_threshold=0.0).fit(X, y)
    edge_woe = transformer.transform(pd.DataFrame({"feature": [1, 20]}))["feature"].tolist()
    outside_woe = transformer.transform(pd.DataFrame({"feature": [-100, 1000]}))["feature"].tolist()

    assert outside_woe == edge_woe
    assert all(w != 0 for w in outside_woe)


# NaN must get its own WoE instead of being dropped when fitting / treated as neutral.
def test_woe_missing_values_get_own_bin():
    values = list(range(1, 41)) + [np.nan] * 10
    X = pd.DataFrame({"feature": values})
    y = pd.Series([0] * 40 + [1] * 10)

    transformer = WoEIVTransformer(n_bins=4, iv_threshold=0.0).fit(X, y)
    result = transformer.transform(pd.DataFrame({"feature": [np.nan, 5.0]}))["feature"]

    assert result.iloc[0] < 0          # NaN toàn nhãn 1 -> WoE âm rõ rệt
    assert result.iloc[0] != 0
    assert np.isfinite(result).all()


# Binary flags must not collapse into a single bin (IV = 0).
def test_woe_binary_flag_has_iv():
    X = pd.DataFrame({"flag": [0] * 10 + [1] * 10})
    y = pd.Series([0] * 8 + [1] * 2 + [0] * 2 + [1] * 8)

    transformer = WoEIVTransformer(iv_threshold=0.0).fit(X, y)

    assert transformer.iv_values_["flag"] > 0.5


def test_age_bins_cover_out_of_range_ages():
    X_train = pd.DataFrame({"AGE": [20, 22, 25, 28, 30, 32, 35, 40, 45, 50]})
    y_train = pd.Series([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])

    edges = fit_age_bins(X_train, y_train)
    result = transform_age_bins(pd.DataFrame({"AGE": [18, 80]}), edges)

    assert result["AGE_BIN"].notna().all()


def test_feature_engineering_transformer_adds_engineered_columns(raw_credit_df: DataFrame):
    from src.preprocessing import AbnormalCodeTransformer

    X = AbnormalCodeTransformer().transform(raw_credit_df)
    result = FeatureEngineeringTransformer().fit_transform(X)

    assert set(ENGINEERED_COLUMNS) <= set(result.columns)
    assert len(result) == len(X)


def _split_xy(df):
    return df.drop(columns=["default.payment.next.month"]), df["default.payment.next.month"]


def test_scorecard_pipeline_on_raw_columns(raw_credit_df: DataFrame):
    """Scorecard chạy được trên đúng tên cột CSV gốc (PAY_0) và dùng cả đặc trưng T2."""
    X, y = _split_xy(raw_credit_df)

    pipeline = build_scorecard_pipeline().fit(X, y)
    woe = pipeline.named_steps["woe_iv"]

    assert len(pipeline.predict_proba(X)) == len(y)
    assert "PAY_0" not in woe.iv_values_
    assert "PAY_1" in woe.iv_values_
    assert "PAY_MEAN" in woe.iv_values_

def test_scorecard_pipeline_drops_sex_and_id(raw_credit_df: DataFrame):
    X, y = _split_xy(raw_credit_df)

    # Pipeline chính thức: loại cả SEX và ID
    pipeline = build_scorecard_pipeline().fit(X, y)
    woe = pipeline.named_steps["woe_iv"]

    assert "SEX" not in woe.iv_values_
    assert "ID" not in woe.iv_values_

    # Pipeline fairness: giữ SEX nhưng vẫn loại ID
    fairness_pipeline = build_scorecard_pipeline(
        drop_sensitive=False
    ).fit(X, y)
    woe_with_sex = fairness_pipeline.named_steps["woe_iv"]

    assert "SEX" in woe_with_sex.iv_values_
    assert "ID" not in woe_with_sex.iv_values_

def test_feature_selection_enabled_for_both_fairness_modes(
    raw_credit_df: DataFrame,
):
    X, y = _split_xy(raw_credit_df)

    official = build_scorecard_pipeline(
        drop_sensitive=True,
        enable_feature_selection=True,
    )
    fairness = build_scorecard_pipeline(
        drop_sensitive=False,
        enable_feature_selection=True,
    )

    # Cả hai pipeline đều sử dụng cùng bước feature selection.
    assert "feature_selection" in official.named_steps
    assert "feature_selection" in fairness.named_steps

    # Lấy các bước trước feature selection.
    official_idx = [
        name for name, _ in official.steps
    ].index("feature_selection")

    fairness_idx = [
        name for name, _ in fairness.steps
    ].index("feature_selection")

    X_official = official[:official_idx].fit_transform(X, y)
    X_fairness = fairness[:fairness_idx].fit_transform(X, y)

    # ID bị loại ở cả hai pipeline.
    assert "ID" not in X_official.columns
    assert "ID" not in X_fairness.columns

    # Chỉ pipeline fairness giữ SEX trước bước chọn đặc trưng.
    assert "SEX" not in X_official.columns
    assert "SEX" in X_fairness.columns

def test_scorecard_pipeline_with_cv(raw_credit_df: DataFrame):
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X, y = _split_xy(raw_credit_df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(build_scorecard_pipeline(), X, y, cv=cv, scoring="roc_auc")

    assert len(scores) == 5
    assert np.isfinite(scores).all()

def test_find_high_correlation_features():
    X = pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5],
        "feature_b": [2, 4, 6, 8, 10],
        "feature_c": [5, 1, 4, 2, 3],
    })

    y = pd.Series([0, 0, 0, 1, 1])

    result = find_high_correlation_features(
        X,
        y,
        threshold=0.9,
    )

    # feature_a và feature_b tương quan hoàn hảo,
    # nên phải loại đúng một trong hai.
    assert ("feature_a" in result) ^ ("feature_b" in result)

    # feature_c không tương quan cao với hai biến trên.
    assert "feature_c" not in result

def test_compute_feature_importance_stability():
    from src.features import compute_feature_importance_stability

    X = pd.DataFrame({
        "feature_a": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1,
                      0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "feature_b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    })

    y = pd.Series([
        0, 0, 0, 0, 1, 1, 1, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    ])

    result = compute_feature_importance_stability(
        X,
        y,
        n_splits=5,
        random_state=42,
    )

    assert "feature" in result.columns
    assert "importance_mean" in result.columns
    assert "importance_std" in result.columns
    assert "importance_min" in result.columns
    assert "importance_max" in result.columns

    assert set(result["feature"]) == {"feature_a", "feature_b"}
    assert (result["importance_mean"] >= 0).all()
    assert (result["importance_std"] >= 0).all()

def test_feature_selection_transformer():
    X = pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5, 6, 7, 8],
        "feature_b": [2, 4, 6, 8, 10, 12, 14, 16],
        "feature_c": [0, 0, 0, 0, 1, 1, 1, 1],
    })

    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    transformer = FeatureSelectionTransformer(
        iv_threshold=0.0,
        correlation_threshold=0.9,
    )

    transformer.fit(X, y)

    X_selected = transformer.transform(X)

    assert hasattr(transformer, "iv_values_")
    assert hasattr(transformer, "selected_features_")
    assert X_selected.shape[0] == X.shape[0]
    assert list(X_selected.columns) == transformer.selected_features_

def test_compute_iv_stability():
    from src.features import compute_iv_stability

    X = pd.DataFrame({
        "strong_feature": [0] * 50 + [1] * 50,
        "weak_feature": [
            0, 1
        ] * 50,
    })

    y = pd.Series([0] * 50 + [1] * 50)

    result = compute_iv_stability(
        X,
        y,
        n_splits=5,
        random_state=42,
    )

    assert not result.empty
    assert {
        "feature",
        "iv_mean",
        "iv_std",
        "iv_min",
        "iv_max",
    }.issubset(result.columns)

    assert set(result["feature"]) == {
        "strong_feature",
        "weak_feature",
    }

    assert np.isfinite(
        result[["iv_mean", "iv_std", "iv_min", "iv_max"]]
    ).all().all()

    assert (result["iv_std"] >= 0).all()
    assert (result["iv_min"] <= result["iv_mean"]).all()
    assert (result["iv_mean"] <= result["iv_max"]).all()

def test_feature_selection_removes_low_iv_feature():
    X = pd.DataFrame({
        "strong_feature": [0] * 50 + [1] * 50,
        "weak_feature": [0, 1] * 50,
    })

    y = pd.Series([0] * 50 + [1] * 50)

    transformer = FeatureSelectionTransformer(
        iv_threshold=0.02,
        enable_correlation_filter=False,
    )

    transformer.fit(X, y)
    result = transformer.transform(X)

    assert "strong_feature" in result.columns
    assert "weak_feature" not in result.columns
    assert any(
        item["feature"] == "weak_feature"
        and item["reason"] == "IV thấp"
        for item in transformer.dropped_features_
)

def test_find_high_correlation_features_keeps_higher_iv():
    X = pd.DataFrame({
        "feature_a": [0] * 50 + [1] * 50,
        "feature_b": [0] * 50 + [1] * 50,
        "feature_c": [0, 1] * 50,
    })

    y = pd.Series([0] * 50 + [1] * 50)

    dropped = find_high_correlation_features(
        X,
        y,
        threshold=0.9,
    )

    # Hai biến tương quan hoàn hảo nên chỉ giữ một biến.
    assert ("feature_a" in dropped) ^ ("feature_b" in dropped)

    # Biến không tương quan cao không bị loại vì tương quan.
    assert "feature_c" not in dropped

def test_feature_selection_on_raw_credit_data(raw_credit_df: DataFrame):
    X, y = _split_xy(raw_credit_df)

    pipeline = build_scorecard_pipeline(
        drop_sensitive=True,
        enable_feature_selection=True,
    )

    pipeline.fit(X, y)
    predictions = pipeline.predict_proba(X)

    assert predictions.shape == (len(y), 2)
    assert np.isfinite(predictions).all()

    selected = pipeline.named_steps["feature_selection"]

    assert "ID" not in selected.selected_features_
    assert "SEX" not in selected.selected_features_

def test_find_high_correlation_uses_spearman_against_outliers():
    """Hai biến độc lập nhưng cùng có một giá trị cực lớn: Pearson ~1, Spearman thấp -> không loại."""
    rng = np.random.default_rng(0)
    a, b = rng.random(200), rng.random(200)
    a[0], b[0] = 1e6, 1e6
    X = pd.DataFrame({"ratio_a": a, "ratio_b": b})
    y = pd.Series(rng.integers(0, 2, 200))

    assert find_high_correlation_features(X, y, method="pearson") != []
    assert find_high_correlation_features(X, y) == []


def test_iv_stability_on_feature_frame(raw_credit_df: DataFrame):
    """Độ ổn định IV phải tính trên bộ đặc trưng freeze v1 (có biến dẫn xuất, AGE_BIN), không phải cột gốc."""
    from src.features import compute_iv_stability

    X, y = _split_xy(raw_credit_df)
    result = compute_iv_stability(X, y, n_splits=3, frame_pipeline=build_feature_frame_pipeline())

    features = set(result["feature"])
    assert {"PAY_MAX", "UTIL_MEAN", "AGE_BIN"} <= features
    assert not {"AGE", "SEX", "ID", "PAY_0"} & features
    assert (result["iv_min"] <= result["iv_max"]).all()


def test_feature_importance_reports_folds_selected():
    """Biến chỉ được chọn ở một phần các fold phải có n_folds_selected < n_splits, std không bị gán 0."""
    rng = np.random.default_rng(1)
    n = 300
    signal = rng.normal(size=n)
    X = pd.DataFrame({
        "signal": signal,
        "near_copy": signal + rng.normal(scale=0.05, size=n),
        "noise": rng.normal(size=n),
    })
    y = pd.Series((signal + rng.normal(scale=1.0, size=n) > 0).astype(int))

    result = compute_feature_importance_stability(X, y, n_splits=5, random_state=42).set_index("feature")

    assert result["n_folds_selected"].max() == 5
    assert result.loc[["signal", "near_copy"], "n_folds_selected"].sum() == 5
    single = result[result["n_folds_selected"] == 1]
    assert single["importance_std"].isna().all()


# Coarse classing (Tuần 2 - T5)
def _coarse_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    # Quan hệ tăng nhưng có nhiễu cục bộ để bin phân vị không đơn điệu
    p = 1 / (1 + np.exp(-(1.2 * x + 0.6 * np.sin(6 * x))))
    y = pd.Series((rng.random(n) < p).astype(int))
    codes = rng.choice([-2, -1, 0, 1, 2, 3, 4, 7], size=n, p=[0.2, 0.2, 0.4, 0.1, 0.07, 0.015, 0.01, 0.005])
    return pd.DataFrame({"cont": x, "code": codes}), y


def test_woe_default_unchanged_without_coarse_classing():
    """Mặc định (không coarse classing): biến rời rạc giữ mỗi giá trị một bin như cấu hình T3."""
    X, y = _coarse_data()
    woe = WoEIVTransformer().fit(X, y)
    assert "code" not in woe.bin_edges_
    assert len(woe.woe_maps_["code"]) == X["code"].nunique()


def test_coarse_classing_min_bin_share():
    """Mọi bin (kể cả biến rời rạc) có >= min_bin_share mẫu."""
    X, y = _coarse_data()
    woe = WoEIVTransformer(n_bins=20, min_bin_share=0.05).fit(X, y)
    for col in ["cont", "code"]:
        share = pd.cut(X[col], woe.bin_edges_[col], include_lowest=True).value_counts(normalize=True)
        assert share.min() >= 0.05


def test_coarse_classing_monotonic_only_for_continuous():
    """Biến liên tục có tỷ lệ bad đơn điệu theo bin; biến rời rạc không bị ép đơn điệu."""
    X, y = _coarse_data()
    woe = WoEIVTransformer(n_bins=20, min_bin_share=0.05, monotonic=True).fit(X, y)

    bins = pd.cut(X["cont"], woe.bin_edges_["cont"], include_lowest=True)
    rates = y.groupby(bins, observed=True).mean().to_numpy()
    assert np.all(np.diff(rates) >= 0)

    plain = WoEIVTransformer(n_bins=20, min_bin_share=0.05).fit(X, y)
    np.testing.assert_array_equal(woe.bin_edges_["code"], plain.bin_edges_["code"])


def test_coarse_classing_unseen_value_maps_to_nearest_bin():
    """Giá trị rời rạc chưa gặp khi fit rơi vào bin gần nhất thay vì nhận WoE = 0."""
    X, y = _coarse_data()
    woe = WoEIVTransformer(n_bins=20, min_bin_share=0.05, iv_threshold=0.0).fit(X, y)
    seen_max = X.loc[X["code"] == X["code"].max()].head(1)
    unseen = seen_max.assign(code=99)
    assert woe.transform(unseen)["code"].item() == pytest.approx(woe.transform(seen_max)["code"].item())
    assert woe.transform(unseen)["code"].item() != 0.0
