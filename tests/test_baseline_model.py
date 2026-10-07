import numpy as np
import pytest
from conftest import make_raw_credit_df
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from app.baseline_model import INPUT_COLUMNS, BaselineModel, feature_group
from app.explanation import BILL_VOLATILITY, DELINQUENCY, DEMOGRAPHICS, FEATURE_GROUPS, REPAYMENT, UTILIZATION
from app.sample_profiles import PROFILES
from src.features import build_features
from src.pipelines import make_pipeline


@pytest.fixture(scope="module")
def train():
    """Train giả lập đúng schema CSV gốc (có PAY_0, SEX, ID) như dữ liệu mô hình thật được fit."""
    return build_features(make_raw_credit_df(n=600, seed=0))


@pytest.fixture(scope="module")
def model(train) -> BaselineModel:
    X, y = train
    pipe = make_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    return BaselineModel(pipe, background=X)


@pytest.mark.parametrize("profile", PROFILES)
def test_pd_matches_pipeline_and_contributions_are_additive(model, profile):
    record = PROFILES[profile]
    exp = model.explain(record)
    expected_pd = model.pipeline.predict_proba(model.to_frame(record))[0, 1]
    assert exp.pd == pytest.approx(expected_pd)
    logit_pd = np.log(exp.pd / (1 - exp.pd))
    assert exp.base_logit + sum(exp.contributions.values()) == pytest.approx(logit_pd)
    assert set(exp.contributions) == set(FEATURE_GROUPS)


def test_base_logit_is_mean_logit_on_background(model, train):
    X, _ = train
    assert model.explain(PROFILES["Ca biên"]).base_logit == pytest.approx(model.pipeline.decision_function(X).mean())


def test_form_record_has_no_sex_and_uses_pay_1(model):
    frame = model.to_frame(PROFILES["Ca biên"])
    assert "SEX" not in frame.columns
    assert "PAY_1" in frame.columns and "PAY_0" not in frame.columns
    assert list(frame.columns) == INPUT_COLUMNS


def test_recent_delay_increases_pd(model):
    record = dict(PROFILES["Ca biên"])
    record["PAY_1"] = 4
    assert model.explain(record).pd > model.explain(PROFILES["Ca biên"]).pd


def test_missing_required_field_raises(model):
    record = {k: v for k, v in PROFILES["Ca biên"].items() if k != "LIMIT_BAL"}
    with pytest.raises(KeyError, match="LIMIT_BAL"):
        model.explain(record)


def test_rejects_non_linear_pipeline(train):
    X, y = train
    pipe = make_pipeline(DecisionTreeClassifier(max_depth=3)).fit(X, y)
    with pytest.raises(TypeError):
        BaselineModel(pipe, background=X)


@pytest.mark.parametrize(
    ("column", "group"),
    [("PAY_1", DELINQUENCY), ("PAY_SLOPE", DELINQUENCY), ("PAY_AMT3", REPAYMENT), ("PAY_RATIO_2", REPAYMENT),
     ("MIN_PAY_FLAG_COUNT", REPAYMENT), ("BILL_AMT1", UTILIZATION), ("UTIL_MEAN", UTILIZATION),
     ("LIMIT_BAL", UTILIZATION), ("BILL_STD", BILL_VOLATILITY), ("AGE", DEMOGRAPHICS), ("AGE_BIN", DEMOGRAPHICS)],
)
def test_feature_group(column, group):
    assert feature_group(column) == group
