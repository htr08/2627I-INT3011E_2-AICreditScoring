"""Unit tests cho scripts/run_pipeline.py CLI runner."""

import sys
import pytest
from unittest.mock import MagicMock, patch

from scripts.run_pipeline import main


def test_run_pipeline_cli_dispatch_boosting(monkeypatch):
    """Kiểm tra CLI runner gọi đúng train_boosting_default khi --mode boosting."""
    mock_boosting = MagicMock()
    monkeypatch.setattr("scripts.run_pipeline.train_boosting_default", mock_boosting)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--mode", "boosting"])

    main()
    mock_boosting.assert_called_once_with(include_sex=False)


def test_run_pipeline_cli_dispatch_imbalance(monkeypatch):
    """Kiểm tra CLI runner gọi đúng train_imbalance_experiments khi --mode imbalance."""
    mock_imbalance = MagicMock()
    monkeypatch.setattr("scripts.run_pipeline.train_imbalance_experiments", mock_imbalance)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--mode", "imbalance", "--include-sex"])

    main()
    mock_imbalance.assert_called_once_with(include_sex=True)


def test_run_pipeline_cli_dispatch_all(monkeypatch):
    """Kiểm tra CLI runner gọi đủ cả 4 runner khi --mode all."""
    mock_base = MagicMock()
    mock_adv = MagicMock()
    mock_boost = MagicMock()
    mock_imb = MagicMock()

    monkeypatch.setattr("scripts.run_pipeline.train_baseline", mock_base)
    monkeypatch.setattr("scripts.run_pipeline.train_rf_xgboost_default", mock_adv)
    monkeypatch.setattr("scripts.run_pipeline.train_boosting_default", mock_boost)
    monkeypatch.setattr("scripts.run_pipeline.train_imbalance_experiments", mock_imb)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--mode", "all"])

    main()
    mock_base.assert_called_once_with(include_sex=False)
    mock_adv.assert_called_once_with(include_sex=False)
    mock_boost.assert_called_once_with(include_sex=False)
    mock_imb.assert_called_once_with(include_sex=False)


def test_run_pipeline_cli_dispatch_scorecard(monkeypatch):
    """--mode scorecard --feature-selection gọi train_scorecard với đúng tham số."""
    mock_scorecard = MagicMock()
    monkeypatch.setattr("scripts.run_pipeline.train_scorecard", mock_scorecard)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--mode", "scorecard", "--feature-selection"])

    main()
    mock_scorecard.assert_called_once_with(include_sex=False, enable_feature_selection=True)
