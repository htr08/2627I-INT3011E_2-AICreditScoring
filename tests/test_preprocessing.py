import pandas as pd
from src.preprocessing import AbnormalCodeTransformer, build_preprocessing_pipeline


def test_abnormal_education_codes_are_grouped():
    X = pd.DataFrame({
        "EDUCATION": [0, 1, 2, 3, 4, 5, 6],
    })

    transformer = AbnormalCodeTransformer()
    result = transformer.fit_transform(X)

    assert result["EDUCATION"].tolist() == [4, 1, 2, 3, 4, 4, 4]


def test_preprocessing_pipeline_runs():
    """Pipeline chính thức phải chạy được và tự drop SEX khỏi dữ liệu."""
    X = pd.DataFrame({
        "SEX": [1, 2, 1],
        "EDUCATION": [0, 2, 6],
        "MARRIAGE": [1, 0, 2],
        "PAY_0": [0, 2, -2],
        "PAY_2": [0, 1, 0],
        "PAY_3": [0, 0, 2],
        "PAY_4": [0, 0, 1],
        "PAY_5": [0, 0, 1],
        "PAY_6": [0, 0, 1],
        "LIMIT_BAL": [50000, 100000, 200000],
        "AGE": [25, 35, 45],
        "BILL_AMT1": [1000, 2000, 3000],
        "BILL_AMT2": [1000, 2000, 3000],
        "BILL_AMT3": [1000, 2000, 3000],
        "BILL_AMT4": [1000, 2000, 3000],
        "BILL_AMT5": [1000, 2000, 3000],
        "BILL_AMT6": [1000, 2000, 3000],
        "PAY_AMT1": [500, 1000, 1500],
        "PAY_AMT2": [500, 1000, 1500],
        "PAY_AMT3": [500, 1000, 1500],
        "PAY_AMT4": [500, 1000, 1500],
        "PAY_AMT5": [500, 1000, 1500],
        "PAY_AMT6": [500, 1000, 1500],
    })

    pipeline = build_preprocessing_pipeline()
    result = pipeline.fit_transform(X)

    assert result.shape[0] == 3
    assert result.shape[1] > 0


def test_special_codes_are_preserved():
    """Kiểm tra rằng mã hợp lệ giữ nguyên; mã bất thường MARRIAGE 0 được merge vào 3."""
    X = pd.DataFrame({
        "SEX": [1, 2],
        "EDUCATION": [1, 2],
        "MARRIAGE": [0, 1],
        "PAY_1": [-2, 0],
        "PAY_2": [0, -2],
        "PAY_3": [0, 0],
        "PAY_4": [0, 0],
        "PAY_5": [0, 0],
        "PAY_6": [0, 0],
        "LIMIT_BAL": [50000, 100000],
        "AGE": [25, 35],
        "BILL_AMT1": [1000, 2000],
        "BILL_AMT2": [1000, 2000],
        "BILL_AMT3": [1000, 2000],
        "BILL_AMT4": [1000, 2000],
        "BILL_AMT5": [1000, 2000],
        "BILL_AMT6": [1000, 2000],
        "PAY_AMT1": [500, 1000],
        "PAY_AMT2": [500, 1000],
        "PAY_AMT3": [500, 1000],
        "PAY_AMT4": [500, 1000],
        "PAY_AMT5": [500, 1000],
        "PAY_AMT6": [500, 1000],
    })

    transformer = AbnormalCodeTransformer()
    result = transformer.fit_transform(X)

    # MARRIAGE mã 0 được merge vào 3 ("Khác") theo codebook
    assert result["MARRIAGE"].tolist() == [3, 1]
    # Các giá trị đặc biệt hợp lệ khác giữ nguyên
    assert result["PAY_1"].tolist() == [-2, 0]
    assert result["PAY_2"].tolist() == [0, -2]


def test_pipeline_handles_unknown_category():
    X_train = pd.DataFrame({
        "SEX": [1, 2],
        "EDUCATION": [1, 2],
        "MARRIAGE": [1, 2],
        "PAY_1": [0, 2],
        "PAY_2": [0, 1],
        "PAY_3": [0, 0],
        "PAY_4": [0, 0],
        "PAY_5": [0, 0],
        "PAY_6": [0, 0],
        "LIMIT_BAL": [50000, 100000],
        "AGE": [25, 35],
        "BILL_AMT1": [1000, 2000],
        "BILL_AMT2": [1000, 2000],
        "BILL_AMT3": [1000, 2000],
        "BILL_AMT4": [1000, 2000],
        "BILL_AMT5": [1000, 2000],
        "BILL_AMT6": [1000, 2000],
        "PAY_AMT1": [500, 1000],
        "PAY_AMT2": [500, 1000],
        "PAY_AMT3": [500, 1000],
        "PAY_AMT4": [500, 1000],
        "PAY_AMT5": [500, 1000],
        "PAY_AMT6": [500, 1000],
    })

    X_valid = X_train.copy()
    X_valid.loc[0, "MARRIAGE"] = 99

    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X_train)

    result = pipeline.transform(X_valid)
    assert result.shape[0] == 2


def test_preprocessing_pipeline_fairness_mode():
    """Kiểm tra chế độ drop_sensitive=False giữ lại SEX để so sánh fairness."""
    X = pd.DataFrame({
        "SEX": [1, 2],
        "EDUCATION": [1, 2],
        "MARRIAGE": [1, 2],
        "PAY_1": [0, 2],
        "PAY_2": [0, 1],
        "PAY_3": [0, 0],
        "PAY_4": [0, 0],
        "PAY_5": [0, 0],
        "PAY_6": [0, 0],
        "LIMIT_BAL": [50000, 100000],
        "AGE": [25, 35],
        "BILL_AMT1": [1000, 2000],
        "BILL_AMT2": [1000, 2000],
        "BILL_AMT3": [1000, 2000],
        "BILL_AMT4": [1000, 2000],
        "BILL_AMT5": [1000, 2000],
        "BILL_AMT6": [1000, 2000],
        "PAY_AMT1": [500, 1000],
        "PAY_AMT2": [500, 1000],
        "PAY_AMT3": [500, 1000],
        "PAY_AMT4": [500, 1000],
        "PAY_AMT5": [500, 1000],
        "PAY_AMT6": [500, 1000],
    })

    pipe_no_sex = build_preprocessing_pipeline(drop_sensitive=True)
    pipe_with_sex = build_preprocessing_pipeline(drop_sensitive=False)

    res_no_sex = pipe_no_sex.fit_transform(X)
    res_with_sex = pipe_with_sex.fit_transform(X)

    # pipe_with_sex có thêm các feature one-hot của SEX
    assert res_with_sex.shape[1] > res_no_sex.shape[1]

def test_preprocessing_pipeline_engineered_features(raw_credit_df):
    """Pipeline chạy trên CSV gốc (PAY_0), có thêm đặc trưng T2 và không lỗi NaN."""
    import numpy as np

    X = raw_credit_df.drop(columns=["default.payment.next.month"])

    res_raw = build_preprocessing_pipeline(engineered=False).fit_transform(X)
    res_eng = build_preprocessing_pipeline(engineered=True).fit_transform(X)

    assert res_eng.shape[1] > res_raw.shape[1]
    assert np.isfinite(res_eng).all()
