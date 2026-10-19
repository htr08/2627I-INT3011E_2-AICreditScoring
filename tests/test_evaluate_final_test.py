"""Unit tests cho scripts/evaluate_final_test.py: bảo vệ chạy Test một lần và định dạng báo cáo nghiệm thu.

Không test nào đọc hoặc chấm dữ liệu Test thật; mọi bước chạm dữ liệu đều bị thay bằng hàm báo lỗi.
"""

import numpy as np
import pytest

from scripts import evaluate_final_test as runner
from src.calibrate import locked_threshold


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Thư mục kết quả giả; mọi đường dẫn marker của runner trỏ vào đó."""
    paths = {
        "OUTPUT_DIR": tmp_path,
        "RUN_MARKER": tmp_path / "test_evaluation_started.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(runner, name, path)
    monkeypatch.setattr(runner, "RESULT_MARKERS", (
        paths["RUN_MARKER"], tmp_path / "test_evaluation.md", tmp_path / "test_metrics.csv",
    ))

    def forbidden(*args, **kwargs):
        raise AssertionError("runner đã đọc dữ liệu dù Test đã được đánh giá")

    monkeypatch.setattr(runner, "load_split_data", forbidden)
    return paths


def test_final_threshold_matches_config():
    # Ngưỡng của runner và ngưỡng lưu trong mô hình cuối phải là cùng một giá trị
    assert runner.FINAL_THRESHOLD == locked_threshold()


@pytest.mark.parametrize("existing", ["test_evaluation_started.json", "test_evaluation.md", "test_metrics.csv"])
def test_runner_refuses_when_any_result_marker_exists(sandbox, tmp_path, existing):
    (tmp_path / existing).write_text("{}", encoding="utf-8")
    before = sorted(p.name for p in tmp_path.iterdir())

    with pytest.raises(RuntimeError, match="do not rerun on Test"):
        runner.run_final_test_evaluation()

    # Không ghi marker mới và không đọc dữ liệu (load_split_data bị chặn trong fixture)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def _rows():
    base = {
        "roc_auc": 0.7825, "roc_auc_ci95_low": 0.768, "roc_auc_ci95_high": 0.797, "gini": 0.5649,
        "ks": 0.429, "ks_ci95_low": 0.405, "ks_ci95_high": 0.46, "pr_auc": 0.56,
        "pr_auc_ci95_low": 0.53, "pr_auc_ci95_high": 0.589, "brier_score": 0.1348,
        "brier_score_ci95_low": 0.129, "brier_score_ci95_high": 0.14, "log_loss": 0.4285, "ece": 0.0163,
        "mean_predicted_pd": 0.2237,
    }
    return [{"model": "Mô hình A", **base}, {"model": "Mô hình B", **base}]


def _comparison(significant=True):
    return {
        "diff": 0.0215, "ci_lower": 0.0145, "ci_upper": 0.0285, "p_value": 1.86e-9,
        "bootstrap_ci_lower": 0.0143, "bootstrap_ci_upper": 0.0286, "is_significant": significant,
    }


def _decision():
    cm = {"precision": 0.2802, "recall": 0.9246, "flag_rate": 0.7298}
    return {
        "cutoff_score": 553.9, "cm_threshold": cm, "cm_05": {"precision": 0.65, "recall": 0.37, "flag_rate": 0.12},
        "unit_cost": 0.03377, "unit_cost_05": 0.06513, "ead_cost": 1634.49, "ead_cost_05": 2830.29,
    }


def test_write_report_tables_have_consistent_columns_and_conclusion(sandbox, tmp_path):
    runner._write_report(_rows(), _comparison(), _decision(), np.array([0, 1, 0, 0]))
    report = (tmp_path / "test_evaluation.md").read_text(encoding="utf-8")

    header = next(line for line in report.splitlines() if line.startswith("| Mô hình/đầu ra"))
    delimiter = report.splitlines()[report.splitlines().index(header) + 1]
    assert header.count("|") == delimiter.count("|")
    assert "có khác biệt có ý nghĩa" in report and "p = **1.86e-09**" in report
    assert "553.9" in report


def test_write_report_states_when_difference_is_not_significant(sandbox, tmp_path):
    runner._write_report(_rows(), _comparison(significant=False), _decision(), np.array([0, 1]))
    report = (tmp_path / "test_evaluation.md").read_text(encoding="utf-8")
    assert "chưa có bằng chứng khác biệt có ý nghĩa" in report
