"""Unit tests cho scripts/export_test_predictions.py: bảo vệ chấm Test một lần và kiểm tra khớp mô hình đã đánh giá.

Không test nào đọc hoặc chấm dữ liệu Test thật; việc chấm điểm được thay bằng dữ liệu giả lập.
"""

import json

import numpy as np
import pandas as pd
import pytest

from scripts import export_test_predictions as export


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Thư mục kết quả giả: marker đánh giá của B đã hoàn tất, test_metrics.csv và test_evaluation.md mẫu."""
    paths = {
        "OUTPUT_DIR": tmp_path,
        "RUN_MARKER": tmp_path / "test_evaluation_started.json",
        "EXPORT_MARKER": tmp_path / "test_predictions_export.json",
        "PREDICTIONS_PATH": tmp_path / "test_predictions.csv",
        "METRICS_PATH": tmp_path / "test_metrics.csv",
        "REPORT_PATH": tmp_path / "test_evaluation.md",
    }
    for name, path in paths.items():
        monkeypatch.setattr(export, name, path)
    paths["RUN_MARKER"].write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    pd.DataFrame({
        "model": list(export.REFERENCE_MODELS.values()),
        "mean_predicted_pd": [0.2207, 0.2237, 0.2201],
    }).to_csv(paths["METRICS_PATH"], index=False)
    paths["REPORT_PATH"].write_text(
        "| Chỉ số | Ngưỡng 0.09 | Ngưỡng 0.50 |\n|---|---:|---:|\n| Flag rate | 70.00% | 12.37% |\n", encoding="utf-8"
    )
    return paths


def fake_scored(flag_rate=0.70, mean_final=0.2237, n=100):
    """Dự đoán giả lập có PD trung bình và flag rate (ngưỡng 0.09) theo yêu cầu."""
    n_flagged = int(round(flag_rate * n))
    pd_final = np.concatenate([np.full(n_flagged, 0.3), np.full(n - n_flagged, 0.05)])
    pd_final = pd_final + (mean_final - pd_final.mean())  # dịch để trung bình đúng bằng mean_final
    return {
        "ids": np.arange(1, n + 1),
        "y_true": np.tile([0, 1], n // 2),
        "predictions": {
            "pd_raw": np.full(n, 0.2207),
            "pd_final": pd_final,
            "pd_baseline": np.full(n, 0.2201),
        },
    }


# ---------------------------------------------------------------------------
# Điều kiện tiên quyết
# ---------------------------------------------------------------------------

def test_preconditions_pass_when_evaluation_completed_and_nothing_exported(sandbox):
    export.check_preconditions(sandbox["RUN_MARKER"], sandbox["EXPORT_MARKER"], sandbox["PREDICTIONS_PATH"])


def test_preconditions_refuse_without_completed_evaluation(sandbox):
    sandbox["RUN_MARKER"].unlink()
    with pytest.raises(RuntimeError, match="không dùng để chấm Test lần đầu"):
        export.check_preconditions(sandbox["RUN_MARKER"], sandbox["EXPORT_MARKER"], sandbox["PREDICTIONS_PATH"])

    sandbox["RUN_MARKER"].write_text(json.dumps({"status": "started"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="chưa hoàn tất"):
        export.check_preconditions(sandbox["RUN_MARKER"], sandbox["EXPORT_MARKER"], sandbox["PREDICTIONS_PATH"])


@pytest.mark.parametrize("existing", ["EXPORT_MARKER", "PREDICTIONS_PATH"])
def test_preconditions_refuse_rerun(sandbox, existing):
    sandbox[existing].write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="không chấm lại Test"):
        export.check_preconditions(sandbox["RUN_MARKER"], sandbox["EXPORT_MARKER"], sandbox["PREDICTIONS_PATH"])


# ---------------------------------------------------------------------------
# Kiểm tra khớp mô hình đã đánh giá
# ---------------------------------------------------------------------------

def test_read_reported_flag_rate(sandbox, tmp_path):
    assert export.read_reported_flag_rate(sandbox["REPORT_PATH"]) == pytest.approx(0.70)
    assert export.read_reported_flag_rate(tmp_path / "missing.md") is None
    other = tmp_path / "other.md"
    other.write_text("không có bảng", encoding="utf-8")
    assert export.read_reported_flag_rate(other) is None


def test_consistency_checks_pass_within_tolerance(sandbox):
    scored = fake_scored(flag_rate=0.70, mean_final=0.2237)
    checks = export.consistency_checks(scored["predictions"], pd.read_csv(sandbox["METRICS_PATH"]), 0.09, 0.70)
    assert [c["check"] for c in checks] == ["mean_pd_raw", "mean_pd_final", "mean_pd_baseline", "flag_rate_at_threshold"]
    assert all(c["ok"] for c in checks)


def test_consistency_checks_flag_deviation_beyond_tolerance(sandbox):
    scored = fake_scored(flag_rate=0.80, mean_final=0.2237)
    checks = {c["check"]: c for c in export.consistency_checks(
        scored["predictions"], pd.read_csv(sandbox["METRICS_PATH"]), 0.09, 0.70)}
    assert not checks["flag_rate_at_threshold"]["ok"]
    assert checks["mean_pd_raw"]["ok"]

    scored = fake_scored(mean_final=0.30)
    checks = {c["check"]: c for c in export.consistency_checks(
        scored["predictions"], pd.read_csv(sandbox["METRICS_PATH"]), 0.09, None)}
    assert not checks["mean_pd_final"]["ok"]
    assert "flag_rate_at_threshold" not in checks  # không có flag rate đã báo cáo thì bỏ qua


def test_build_predictions_frame_columns_and_length_check():
    scored = fake_scored(n=10)
    frame = export.build_predictions_frame(scored["ids"], scored["y_true"], scored["predictions"])
    assert list(frame.columns) == ["ID", "y_true", "pd_raw", "pd_final", "pd_baseline"]
    assert len(frame) == 10
    with pytest.raises(ValueError):
        export.build_predictions_frame(scored["ids"][:5], scored["y_true"][:5], scored["predictions"])


# ---------------------------------------------------------------------------
# Quy trình xuất
# ---------------------------------------------------------------------------

def test_export_writes_marker_before_scoring_then_predictions(sandbox, monkeypatch):
    seen = {}

    def fake_score_split(split, before_scoring=None):
        assert split == "test"
        before_scoring()
        # Marker phải tồn tại ngay khi bắt đầu chấm Test, và chưa có file kết quả
        seen["marker_status"] = json.loads(sandbox["EXPORT_MARKER"].read_text(encoding="utf-8"))["status"]
        seen["predictions_exist"] = sandbox["PREDICTIONS_PATH"].exists()
        return fake_scored()

    monkeypatch.setattr(export, "score_split", fake_score_split)
    path = export.export_test_predictions()

    assert seen == {"marker_status": "started", "predictions_exist": False}
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["ID", "y_true", "pd_raw", "pd_final", "pd_baseline"]
    assert len(frame) == 100
    marker = json.loads(sandbox["EXPORT_MARKER"].read_text(encoding="utf-8"))
    assert marker["status"] == "completed" and marker["n_test"] == 100
    assert all(c["ok"] for c in marker["checks"])


def test_export_cannot_run_twice(sandbox, monkeypatch):
    monkeypatch.setattr(export, "score_split", lambda split, before_scoring=None: (before_scoring(), fake_scored())[1])
    export.export_test_predictions()

    calls = []
    monkeypatch.setattr(export, "score_split", lambda *a, **k: calls.append(1))
    with pytest.raises(RuntimeError, match="không chấm lại Test"):
        export.export_test_predictions()
    assert calls == []  # lần chạy lại không chấm Test


def test_export_does_not_write_predictions_when_model_mismatch(sandbox, monkeypatch):
    monkeypatch.setattr(
        export, "score_split",
        lambda split, before_scoring=None: (before_scoring(), fake_scored(mean_final=0.40))[1],
    )
    with pytest.raises(RuntimeError, match="không khớp mô hình đã đánh giá"):
        export.export_test_predictions()

    assert not sandbox["PREDICTIONS_PATH"].exists()
    assert json.loads(sandbox["EXPORT_MARKER"].read_text(encoding="utf-8"))["status"] == "failed_consistency"


def test_training_error_before_scoring_does_not_block_rerun(sandbox, monkeypatch):
    def failing_score_split(split, before_scoring=None):
        raise ValueError("lỗi khi huấn luyện, trước khi chấm Test")

    monkeypatch.setattr(export, "score_split", failing_score_split)
    with pytest.raises(ValueError):
        export.export_test_predictions()
    assert not sandbox["EXPORT_MARKER"].exists()


def test_dry_run_scores_valid_only_and_writes_nothing(sandbox, monkeypatch):
    requested = []

    def fake_score_split(split, before_scoring=None):
        requested.append(split)
        return fake_scored()

    monkeypatch.setattr(export, "score_split", fake_score_split)
    summary = export.dry_run()

    assert requested == ["valid"]
    assert summary["n"] == 100 and summary["flag_rate_at_threshold"] == pytest.approx(0.70)
    assert not sandbox["EXPORT_MARKER"].exists() and not sandbox["PREDICTIONS_PATH"].exists()
