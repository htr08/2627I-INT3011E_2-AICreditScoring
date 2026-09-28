import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.mock_model import FEATURE_GROUPS, MockModel
from app.sample_profiles import PROFILES


@pytest.mark.parametrize("profile", PROFILES)
def test_contributions_sum_to_logit_pd(profile):
    exp = MockModel().explain(PROFILES[profile])
    assert set(exp.contributions) == set(FEATURE_GROUPS)
    logit_pd = np.log(exp.pd / (1 - exp.pd))
    assert exp.base_logit + sum(exp.contributions.values()) == pytest.approx(logit_pd)


def test_sample_profiles_are_ordered_by_risk():
    model = MockModel()
    pds = [model.explain(PROFILES[p]).pd for p in ["Rủi ro thấp", "Ca biên", "Rủi ro cao"]]
    assert pds == sorted(pds)


def test_recent_delay_increases_pd():
    model = MockModel()
    record = dict(PROFILES["Ca biên"])
    before = model.explain(record).pd
    record["PAY_1"] = 3
    assert model.explain(record).pd > before


def test_app_runs_and_scores_profiles():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["Xác suất vỡ nợ (PD)", "Điểm tín dụng", "Mức cảnh báo"]

    at.sidebar.selectbox[0].set_value("Rủi ro cao").run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.metric[2].value == "🔴 Cảnh báo"
