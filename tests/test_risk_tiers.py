import pytest

from app.risk_tiers import ALERT, NORMAL, WATCH, classify, load_thresholds


def test_classify_boundaries():
    assert classify(0.0899, 0.09, 0.27) is NORMAL
    assert classify(0.09, 0.09, 0.27) is WATCH  # chạm ngưỡng thì bị gắn cờ, khớp PD >= ngưỡng
    assert classify(0.2699, 0.09, 0.27) is WATCH
    assert classify(0.27, 0.09, 0.27) is ALERT


def test_load_thresholds_from_config_matches_locked_threshold():
    watch, alert = load_thresholds()
    assert (watch, alert) == (0.09, 0.27)


@pytest.mark.parametrize(
    "config",
    [
        {"calibration": {"threshold": 0.09}},
        {"risk_tiers": {"alert_threshold": 0.27}},
        {"calibration": {"threshold": 0.3}, "risk_tiers": {"alert_threshold": 0.27}},
    ],
)
def test_load_thresholds_rejects_missing_or_unordered(config):
    with pytest.raises(ValueError):
        load_thresholds(config)
