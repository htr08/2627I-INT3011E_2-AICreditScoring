import numpy as np
import pytest

from src.config import load_config
from src.explain import (
    aggregate_by_group,
    aggregate_by_variable,
    check_groups_cover,
    load_feature_groups,
    model_variables_from_freeze,
    source_variable,
    variable_to_group,
)


def test_config_groups_match_feature_freeze():
    check_groups_cover(load_feature_groups(), model_variables_from_freeze())


def test_variable_in_two_groups_raises():
    groups = {"a": {"label": "A", "variables": ["X", "Y"]}, "b": {"label": "B", "variables": ["Y"]}}
    with pytest.raises(ValueError):
        variable_to_group(groups)


def test_check_groups_cover_reports_missing_and_extra():
    groups = {"a": {"label": "A", "variables": ["X", "Z"]}}
    with pytest.raises(ValueError, match=r"thiếu \['Y'\].*thừa \['Z'\]"):
        check_groups_cover(groups, ["X", "Y"])


def test_source_variable_prefers_longest_match():
    cats = ["PAY_1", "PAY_10", "AGE_BIN"]
    assert source_variable("PAY_1_2", cats) == "PAY_1"
    assert source_variable("PAY_10_1", cats) == "PAY_10"
    assert source_variable("AGE_BIN_(25.0, 28.0]", cats) == "AGE_BIN"
    assert source_variable("PAY_MAX", cats) == "PAY_MAX"


def test_aggregation_preserves_total_contribution():
    rng = np.random.default_rng(0)
    cols = ["PAY_1_0", "PAY_1_2", "LIMIT_BAL", "UTIL_1"]
    values = rng.normal(size=(20, len(cols)))
    by_var = aggregate_by_variable(values, cols, ["PAY_1"])
    assert set(by_var.columns) == {"PAY_1", "LIMIT_BAL", "UTIL_1"}
    groups = {
        "tre": {"label": "Trễ hạn", "variables": ["PAY_1"]},
        "han_muc": {"label": "Hạn mức", "variables": ["LIMIT_BAL", "UTIL_1"]},
    }
    by_group = aggregate_by_group(by_var, groups)
    assert set(by_group.columns) == {"Trễ hạn", "Hạn mức"}
    np.testing.assert_allclose(by_group.sum(axis=1), values.sum(axis=1))


def test_aggregate_by_group_rejects_unknown_variable():
    by_var = aggregate_by_variable(np.zeros((2, 1)), ["FOO"], [])
    with pytest.raises(ValueError):
        aggregate_by_group(by_var, load_config()["feature_groups"])
