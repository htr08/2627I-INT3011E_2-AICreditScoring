"""Kiểm tra hồ sơ nhập vào demo trước khi chấm điểm.

Lỗi (errors) chặn chấm điểm; cảnh báo (warnings) chỉ hiển thị. Phạm vi giá trị theo dữ liệu UCI
(Default of Credit Card Clients) và các nhóm mã đã chuẩn hóa ở src/preprocessing.py.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from src.preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

# Cột đầu vào của mô hình chính thức (cột gốc, không có SEX/ID); thứ tự khớp lúc huấn luyện.
INPUT_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

LIMIT_RANGE = (10_000, 1_000_000)
AGE_RANGE = (21, 79)
PAY_CODES = range(-2, 10)  # -2 .. 9
EDUCATION_CODES = (1, 2, 3, 4)
MARRIAGE_CODES = (1, 2, 3)
# Dư nợ vượt hạn mức quá mức này thường là lỗi nhập liệu (đơn vị tiền), chỉ cảnh báo.
BILL_LIMIT_WARN_RATIO = 3.0


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _number(record: Mapping, key: str, errors: list[str]):
    value = record.get(key)
    if value is None or isinstance(value, bool) or pd.isna(value):
        errors.append(f"Thiếu giá trị {key}.")
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        errors.append(f"{key} phải là số, nhận {value!r}.")
        return None


def validate_record(record: Mapping) -> ValidationResult:
    """Kiểm tra đủ trường, kiểu số, phạm vi và mã hợp lệ; cảnh báo dư nợ bất thường."""
    result = ValidationResult()
    values = {key: _number(record, key, result.errors) for key in INPUT_COLUMNS}
    if result.errors:
        return result

    limit = values["LIMIT_BAL"]
    if not LIMIT_RANGE[0] <= limit <= LIMIT_RANGE[1]:
        result.errors.append(f"LIMIT_BAL phải trong khoảng {LIMIT_RANGE[0]:,}–{LIMIT_RANGE[1]:,} NT$.")
    if not AGE_RANGE[0] <= values["AGE"] <= AGE_RANGE[1]:
        result.errors.append(f"AGE phải trong khoảng {AGE_RANGE[0]}–{AGE_RANGE[1]}.")
    if values["EDUCATION"] not in EDUCATION_CODES:
        result.errors.append(f"EDUCATION phải thuộc {EDUCATION_CODES}.")
    if values["MARRIAGE"] not in MARRIAGE_CODES:
        result.errors.append(f"MARRIAGE phải thuộc {MARRIAGE_CODES}.")
    for i in range(1, 7):
        if values[f"PAY_{i}"] not in PAY_CODES:
            result.errors.append(f"PAY_{i} phải là số nguyên từ {PAY_CODES[0]} đến {PAY_CODES[-1]}.")
        if values[f"PAY_AMT{i}"] < 0:
            result.errors.append(f"PAY_AMT{i} không được âm.")
    if result.errors:
        return result

    if any(values[f"BILL_AMT{i}"] > BILL_LIMIT_WARN_RATIO * limit for i in range(1, 7)):
        result.warnings.append(
            f"Có dư nợ sao kê vượt {BILL_LIMIT_WARN_RATIO:g} lần hạn mức; kiểm tra lại đơn vị tiền (NT$)."
        )
    return result


def record_to_frame(record: Mapping) -> pd.DataFrame:
    """Record của form -> DataFrame 1 dòng với đúng INPUT_COLUMNS; KeyError nếu thiếu trường."""
    missing = [c for c in INPUT_COLUMNS if c not in record]
    if missing:
        raise KeyError(f"Thiếu trường {missing} trong hồ sơ đầu vào.")
    return pd.DataFrame([{c: record[c] for c in INPUT_COLUMNS}], columns=INPUT_COLUMNS)
