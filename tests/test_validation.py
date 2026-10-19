import pytest

from app.sample_profiles import PROFILES
from app.validation import INPUT_COLUMNS, record_to_frame, validate_record


@pytest.mark.parametrize("profile", PROFILES)
def test_sample_profiles_are_valid(profile):
    result = validate_record(PROFILES[profile])
    assert result.ok and not result.warnings


def record(**overrides):
    return {**PROFILES["Ca biên"], **overrides}


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"AGE": 15}, "AGE"),
        ({"LIMIT_BAL": 5_000}, "LIMIT_BAL"),
        ({"EDUCATION": 0}, "EDUCATION"),
        ({"MARRIAGE": 7}, "MARRIAGE"),
        ({"PAY_3": 10}, "PAY_3"),
        ({"PAY_2": -3}, "PAY_2"),
        ({"PAY_AMT4": -1}, "PAY_AMT4"),
        ({"BILL_AMT1": "abc"}, "BILL_AMT1"),
        ({"AGE": None}, "AGE"),
    ],
)
def test_invalid_values_are_reported(overrides, fragment):
    result = validate_record(record(**overrides))
    assert not result.ok
    assert any(fragment in message for message in result.errors)


def test_missing_field_is_reported():
    partial = {k: v for k, v in PROFILES["Ca biên"].items() if k != "PAY_AMT6"}
    result = validate_record(partial)
    assert not result.ok and "PAY_AMT6" in result.errors[0]


def test_huge_bill_only_warns():
    result = validate_record(record(BILL_AMT2=900_000))
    assert result.ok
    assert result.warnings and "đơn vị" in result.warnings[0]


def test_record_to_frame_has_model_columns_without_sex():
    frame = record_to_frame(PROFILES["Ca biên"])
    assert list(frame.columns) == INPUT_COLUMNS and "SEX" not in frame.columns
    with pytest.raises(KeyError, match="LIMIT_BAL"):
        record_to_frame({k: v for k, v in PROFILES["Ca biên"].items() if k != "LIMIT_BAL"})
