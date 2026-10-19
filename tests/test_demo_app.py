import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.mock_model import FEATURE_GROUPS, MockModel
from app.sample_profiles import PROFILES
from src.calibrate import DEFAULT_MODEL_PATH


@pytest.mark.parametrize("profile", PROFILES)
def test_contributions_sum_to_logit_pd(profile):
    exp = MockModel().explain(PROFILES[profile])
    assert set(exp.contributions) == set(FEATURE_GROUPS)
    logit_pd = np.log(exp.pd / (1 - exp.pd))
    assert exp.base_logit + sum(exp.contributions.values()) == pytest.approx(logit_pd)


def test_sample_profiles_are_ordered_by_risk():
    model = MockModel()
    pds = [model.explain(PROFILES[p]).pd for p in ["Bình thường", "Ca biên", "Cảnh báo cao"]]
    assert pds == sorted(pds)


def test_recent_delay_increases_pd():
    model = MockModel()
    record = dict(PROFILES["Ca biên"])
    before = model.explain(record).pd
    record["PAY_1"] = 3
    assert model.explain(record).pd > before


@pytest.fixture(autouse=True)
def clear_model_cache():
    # load_model() dùng st.cache_resource, cache sống qua các lần chạy AppTest trong cùng process.
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def test_app_runs_and_scores_profiles(monkeypatch):
    monkeypatch.setenv("DEMO_MODEL", "mock")
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["Xác suất vỡ nợ (PD)", "Điểm tín dụng", "Mức cảnh báo"]

    at.sidebar.selectbox[0].set_value("Cảnh báo cao").run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.metric[2].value == "🔴 Cảnh báo"


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="chưa có models/final_model.pkl")
def test_app_uses_final_model_with_tiers_and_actions(monkeypatch):
    monkeypatch.delenv("DEMO_MODEL", raising=False)
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=120).run()
    assert not at.exception
    assert not at.sidebar.error
    assert "Platt" in at.sidebar.info[0].value
    assert not at.sidebar.slider  # ngưỡng đã khóa, không cho chỉnh

    expected = {"Bình thường": ("🟢 Bình thường", "Duy trì hạn mức"), "Ca biên": ("🟡 Theo dõi", "nhắc nợ"),
                "Cảnh báo cao": ("🔴 Cảnh báo cao", "Giảm hạn mức")}
    for profile, (label, action) in expected.items():
        at.sidebar.selectbox[0].set_value(profile).run()
        at.sidebar.button[0].click().run()
        assert not at.exception
        assert at.metric[2].value == label
        shown = [e.value for e in list(at.success) + list(at.warning) + list(at.error)]
        assert any(action in text for text in shown)


def test_app_falls_back_when_final_model_is_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("DEMO_MODEL", raising=False)
    monkeypatch.setattr("app.final_model.DEFAULT_MODEL_PATH", tmp_path / "missing.pkl")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tmp_path.as_uri())
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60).run()
    assert not at.exception
    assert "Không nạp được mô hình cuối" in at.sidebar.error[0].value
    assert "mock model" in at.sidebar.warning[0].value


def test_app_falls_back_to_mock_when_registry_is_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("DEMO_MODEL", "baseline")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tmp_path.as_uri())
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60).run()
    assert not at.exception
    assert "Không nạp được mô hình baseline" in at.sidebar.error[0].value
    assert "mock model" in at.sidebar.warning[0].value
    assert len(at.metric) == 3
