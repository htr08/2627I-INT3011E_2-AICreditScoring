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


def test_reason_codes_split_by_direction_and_threshold():
    """reason_codes: tối đa k nhóm mỗi chiều, sắp theo độ lớn, bỏ nhóm có |đóng góp| <= min_abs."""
    import pandas as pd

    from src.explain import reason_codes

    contrib = pd.Series({"A": 0.8, "B": 0.3, "C": 0.04, "D": -0.5, "E": -0.2, "F": -0.01})
    codes = reason_codes(contrib, k=2, min_abs=0.05)

    assert list(codes["nhóm"]) == ["A", "B", "D", "E"]
    assert list(codes["chiều"]) == ["Tăng rủi ro", "Tăng rủi ro", "Giảm rủi ro", "Giảm rủi ro"]
    assert reason_codes(pd.Series({"A": -0.3}), k=2)["chiều"].tolist() == ["Giảm rủi ro"]


def test_group_importance_share_sums_to_one_and_ranks_by_magnitude():
    import pandas as pd

    from src.explain import group_importance_share

    by_group = pd.DataFrame({"A": [3.0, -3.0], "B": [1.0, 1.0], "C": [0.0, 0.0]})
    share = group_importance_share(by_group)
    assert share.sum() == pytest.approx(1.0)
    assert share.to_dict() == pytest.approx({"A": 0.75, "B": 0.25, "C": 0.0})


def test_bootstrap_share_ci_brackets_point_estimate_and_is_reproducible():
    import pandas as pd

    from src.explain import bootstrap_share_ci, group_importance_share

    rng = np.random.default_rng(1)
    by_group = pd.DataFrame(rng.normal(size=(300, 3)) * [3.0, 1.0, 0.5], columns=["A", "B", "C"])
    ci = bootstrap_share_ci(by_group, n_boot=200, seed=7)
    share = group_importance_share(by_group)
    assert ((ci["lo"] <= share) & (share <= ci["hi"])).all()
    pd.testing.assert_frame_equal(ci, bootstrap_share_ci(by_group, n_boot=200, seed=7))


def test_rank_agreement_and_top_k_overlap():
    import pandas as pd

    from src.explain import rank_agreement, top_k_overlap

    a = pd.Series({"x": 4.0, "y": 3.0, "z": 2.0, "w": 1.0})
    same = rank_agreement(a, a * 10)
    assert same == pytest.approx({"spearman": 1.0, "kendall": 1.0})
    reversed_ = rank_agreement(a, a.iloc[::-1].set_axis(a.index))
    assert reversed_["spearman"] == pytest.approx(-1.0)
    assert top_k_overlap(a, pd.Series({"x": 1.0, "y": 5.0, "z": 3.0, "w": 0.5}), k=2) == 1


def test_contribution_and_top_group_agreement():
    import pandas as pd

    from src.explain import contribution_agreement, top_group_agreement

    by_a = pd.DataFrame({"A": [3.0, 1.0, -1.0, 0.5], "B": [0.0, 2.0, 0.5, 0.2]})
    by_b = pd.DataFrame({"A": [6.0, 2.0, -2.0, 1.0], "B": [1.0, -1.0, 0.0, 0.5]})
    rho = contribution_agreement(by_a, by_b)
    assert rho["A"] == pytest.approx(1.0)
    assert rho["B"] < 1.0
    # nhóm làm tăng rủi ro mạnh nhất: a = [A, B, B, A]; b = [A, A, B, A]
    assert top_group_agreement(by_a, by_b) == pytest.approx(0.75)
    assert top_group_agreement(by_a, by_b, mask=[True, False, True, False]) == pytest.approx(1.0)


def test_effect_direction_sign_and_missing_values():
    import pandas as pd

    from src.explain import effect_direction

    values = pd.DataFrame({"U": [1.0, 2.0, 3.0, 4.0, np.nan], "L": [1.0, 2.0, 3.0, 4.0, 5.0]})
    by_var = pd.DataFrame({"U": [0.1, 0.2, 0.3, 0.4, 0.0], "L": [5.0, 4.0, 3.0, 2.0, 1.0]})
    direction = effect_direction(by_var, values, ["U", "L"])
    assert direction["U"] == pytest.approx(1.0)
    assert direction["L"] == pytest.approx(-1.0)


def test_explain_pipeline_is_additive_for_linear_and_tree_models(raw_credit_df):
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression

    from src.explain import explain_pipeline, load_feature_groups
    from src.features import build_features
    from src.pipelines import make_pipeline

    X, y = build_features(raw_credit_df)
    groups = load_feature_groups()
    lr = make_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    gbm = make_pipeline(LGBMClassifier(n_estimators=20, random_state=42, verbose=-1)).fit(X, y)

    for pipe, kwargs in ((lr, {"background": X}), (gbm, {})):
        result = explain_pipeline(pipe, X.iloc[:50], **kwargs)
        assert result.max_additivity_error < 1e-5
        by_group = result.by_group(groups)
        np.testing.assert_allclose(result.base + by_group.sum(axis=1), result.margin, atol=1e-5)

    with pytest.raises(ValueError, match="background"):
        explain_pipeline(lr, X.iloc[:5])
