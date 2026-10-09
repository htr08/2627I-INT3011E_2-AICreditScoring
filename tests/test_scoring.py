import numpy as np
import pytest

from src.scoring import base_points, pd_to_score, shap_to_points


def test_base_odds_maps_to_base_score():
    assert pd_to_score(1 / 51) == pytest.approx(600)


def test_doubling_odds_adds_pdo():
    assert pd_to_score(1 / 101) == pytest.approx(620)


def test_score_decreases_with_pd():
    scores = pd_to_score(np.linspace(0.01, 0.99, 50))
    assert np.all(np.diff(scores) < 0)


def test_clip_bounds():
    assert pd_to_score(1 - 1e-9) == 300
    assert pd_to_score(1e-9) == 850


def test_points_are_additive():
    base_logit, shap_logit = -1.2, np.array([0.8, -0.3, 0.15])
    pd = 1 / (1 + np.exp(-(base_logit + shap_logit.sum())))
    total = base_points(base_logit) + shap_to_points(shap_logit).sum()
    assert total == pytest.approx(pd_to_score(pd, clip=None))


def test_scorecard_points_sum_to_pdo_score(raw_credit_df):
    """Điểm nền + điểm từng thuộc tính của một hồ sơ = pd_to_score(PD của scorecard) chưa clip."""
    from src.features import build_features
    from src.pipelines import build_scorecard_pipeline
    from src.scoring import scorecard_points_table

    X, y = build_features(raw_credit_df)
    pipe = build_scorecard_pipeline().fit(X, y)
    table = scorecard_points_table(pipe)

    pd_hat = pipe.predict_proba(X.iloc[:20])[:, 1]
    woe_rows = pipe[:-1].transform(X.iloc[:20])
    model = pipe.named_steps["model"]
    factor = 20 / np.log(2)
    attribute_points = -factor * woe_rows[model.selected_features_].to_numpy() @ model.coef_[0]
    base = table.loc[table["feature"] == "(base)", "points"].item()

    np.testing.assert_allclose(base + attribute_points, pd_to_score(pd_hat, clip=None), rtol=1e-9)
    assert set(table["feature"]) - {"(base)"} == set(model.selected_features_)
    # Mọi hệ số WoE cùng dấu âm: thuộc tính rủi ro cao hơn (WoE thấp hơn) không được nhiều điểm hơn.
    assert (table.loc[table["feature"] != "(base)", "coef"] < 0).all()
