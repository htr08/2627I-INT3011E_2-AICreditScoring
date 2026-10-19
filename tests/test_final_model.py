"""Adapter mô hình cuối của demo; dùng models/final_model.pkl đã commit (xuất bởi --mode final-model)."""

import numpy as np
import pytest

from app.explanation import FEATURE_GROUPS
from app.final_model import FinalModel
from app.risk_tiers import ALERT, NORMAL, WATCH, classify, load_thresholds
from app.sample_profiles import PROFILES
from src.calibrate import DEFAULT_MODEL_PATH, CalibratedPipeline

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="chưa có models/final_model.pkl")


@pytest.fixture(scope="module")
def model() -> FinalModel:
    return FinalModel.from_file()


@pytest.mark.parametrize("profile", PROFILES)
def test_pd_matches_pipeline_and_contributions_are_additive(model, profile):
    record = PROFILES[profile]
    exp = model.explain(record)
    expected_pd = model.calibrated.predict_proba(model.to_frame(record))[0, 1]
    assert exp.pd == pytest.approx(expected_pd)
    # SHAP nhân hệ số a: base calibrated + tổng đóng góp = logit(PD hiển thị)
    logit_pd = np.log(exp.pd / (1 - exp.pd))
    assert exp.base_logit + sum(exp.contributions.values()) == pytest.approx(logit_pd, abs=1e-6)
    assert set(exp.contributions) == set(FEATURE_GROUPS)
    # Đóng góp theo biến cộng lại bằng đóng góp theo nhóm
    assert sum(exp.variable_contributions.values()) == pytest.approx(sum(exp.contributions.values()), abs=1e-6)


def test_sample_profiles_land_in_expected_tiers(model):
    watch, alert = load_thresholds()
    tiers = {name: classify(model.explain(PROFILES[name]).pd, watch, alert) for name in PROFILES}
    assert tiers == {"Bình thường": NORMAL, "Ca biên": WATCH, "Cảnh báo cao": ALERT}


def test_saved_threshold_is_the_locked_one(model):
    assert model.threshold == load_thresholds()[0]


def test_score_matches_pdo_of_pd(model):
    from src.scoring import pd_to_score

    exp = model.explain(PROFILES["Ca biên"])
    assert model.score(PROFILES["Ca biên"]) == pytest.approx(float(pd_to_score(exp.pd)))


def test_recent_delay_increases_pd(model):
    record = dict(PROFILES["Ca biên"])
    record["PAY_1"] = 4
    assert model.explain(record).pd > model.explain(PROFILES["Ca biên"]).pd


def test_rejects_non_platt_calibrator(model):
    isotonic = CalibratedPipeline(base_pipeline=model.calibrated.base_pipeline, method="isotonic")
    with pytest.raises(TypeError, match="Platt"):
        FinalModel(isotonic)
