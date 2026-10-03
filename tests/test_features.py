import numpy as np
import pandas as pd

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
)

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

from src.preprocessing import build_scorecard_pipeline

def test_scorecard_pipeline():
    X = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })

    y = pd.Series([
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    ])

    pipeline = build_scorecard_pipeline()

    pipeline.fit(X, y)

    result = pipeline.predict(X)

    # Check that the pipeline produces predictions.
    assert len(result) == len(y)

    # Check that the WoE transformer is fitted inside the pipeline.
    assert hasattr(
        pipeline.named_steps["woe_iv"],
        "iv_values_"
    )

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

def test_scorecard_pipeline_drops_sex():
    X = pd.DataFrame({
        "SEX": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })

    y = pd.Series([
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    ])

    pipeline = build_scorecard_pipeline()

    pipeline.fit(X, y)

    # Check that SEX is removed before WoE and model training.
    woe = pipeline.named_steps["woe_iv"]
    assert "SEX" not in woe.iv_values_

def test_scorecard_pipeline_with_cv():
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X = pd.DataFrame({
        "feature_1": range(1, 21),
        "feature_2": [
            "low", "low", "low", "low", "low",
            "medium", "medium", "medium", "medium", "medium",
            "high", "high", "high", "high", "high",
            "high", "high", "high", "high", "high"
        ]
    })

    y = pd.Series([
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ])

    pipeline = build_scorecard_pipeline()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="roc_auc"
    )

    # Check that cross-validation runs successfully.
    assert len(scores) == 5

    # Check that all folds produce valid scores.
    assert np.isfinite(scores).all()