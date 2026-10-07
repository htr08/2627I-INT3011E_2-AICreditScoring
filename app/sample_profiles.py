"""Hồ sơ mẫu để nạp nhanh vào form demo (dữ liệu giả, không lấy từ bộ dữ liệu)."""


def _monthly(prefix: str, values: list[int]) -> dict[str, int]:
    return {f"{prefix}{i}": v for i, v in enumerate(values, start=1)}


PROFILES = {
    "Rủi ro thấp": {
        "LIMIT_BAL": 400_000, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 34,
        **_monthly("PAY_", [-1, -1, -1, -1, -1, -1]),
        **_monthly("BILL_AMT", [12_000, 9_000, 15_000, 8_000, 11_000, 10_000]),
        **_monthly("PAY_AMT", [9_000, 15_000, 8_000, 11_000, 10_000, 9_500]),
    },
    "Ca biên": {
        "LIMIT_BAL": 80_000, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 41,
        **_monthly("PAY_", [0, 2, 0, 0, 0, 0]),
        **_monthly("BILL_AMT", [58_000, 56_500, 55_000, 54_000, 52_000, 50_000]),
        **_monthly("PAY_AMT", [2_500, 2_400, 2_300, 2_200, 2_100, 2_000]),
    },
    "Rủi ro cao": {
        "LIMIT_BAL": 50_000, "EDUCATION": 3, "MARRIAGE": 1, "AGE": 52,
        **_monthly("PAY_", [2, 2, 2, 0, 0, 0]),
        **_monthly("BILL_AMT", [49_500, 48_200, 47_000, 45_000, 44_000, 42_000]),
        **_monthly("PAY_AMT", [0, 0, 1_500, 1_600, 1_500, 1_500]),
    },
}
