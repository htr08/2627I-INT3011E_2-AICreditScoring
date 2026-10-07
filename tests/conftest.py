import numpy as np
import pandas as pd
import pytest


def make_raw_credit_df(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Dữ liệu giả lập đúng schema CSV gốc (có PAY_0, SEX, ID) với nhãn phụ thuộc PAY_0 & LIMIT_BAL."""
    rng = np.random.default_rng(seed)
    limit_bal = rng.integers(10, 50, n) * 10_000
    pay = {f"PAY_{i}": rng.integers(-2, 5, n) for i in (0, 2, 3, 4, 5, 6)}
    bill = {f"BILL_AMT{i}": rng.integers(-1_000, 100_000, n) for i in range(1, 7)}
    pay_amt = {f"PAY_AMT{i}": rng.integers(0, 20_000, n) for i in range(1, 7)}

    log_odds = -1.5 + 0.8 * pay["PAY_0"] - 2.0 * (limit_bal / 500_000)
    labels = (rng.random(n) < 1.0 / (1.0 + np.exp(-log_odds))).astype(int)

    return pd.DataFrame({
        "ID": np.arange(1, n + 1),
        "LIMIT_BAL": limit_bal,
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(0, 7, n),
        "MARRIAGE": rng.integers(0, 4, n),
        "AGE": rng.integers(21, 70, n),
        **pay,
        **bill,
        **pay_amt,
        "default.payment.next.month": labels,
    })


@pytest.fixture
def raw_credit_df():
    return make_raw_credit_df()
