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
