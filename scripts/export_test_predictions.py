"""Xuất dự đoán từng hồ sơ trên Test của mô hình đã khóa, để thành viên A cập nhật phần fairness (Tuần 3 – T3).

Đây không phải lần đánh giá Test mới. Script chỉ ghi lại đầu ra của đúng các mô hình mà
`scripts/evaluate_final_test.py` đã đánh giá (LightGBM thô, LightGBM + Platt, Logistic baseline), với cùng quy trình
huấn luyện, cùng calibrator (Platt fit trên Valid) và cùng ngưỡng. Không tính chỉ số hiệu năng, không thay đổi mô hình,
calibrator hay ngưỡng.

Điều kiện và bảo vệ:
- Chỉ chạy khi đánh giá Test của `evaluate_final_test.py` đã hoàn tất (marker `completed`).
- Chấm Test đúng một lần: marker `test_predictions_export.json` được ghi trước khi chấm; marker hoặc file kết quả đã
  tồn tại thì từ chối chạy lại.
- Sau khi chấm, kiểm tra mô hình huấn luyện lại khớp mô hình đã được đánh giá bằng đại lượng không dùng nhãn: PD trung bình
  của từng đầu ra (so với `test_metrics.csv`) và flag rate ở ngưỡng (so với `test_evaluation.md`). Lệch quá dung sai thì
  không ghi file kết quả.

Cột xuất (`reports/test_evaluation/test_predictions.csv`, bị gitignore theo quy ước không commit dữ liệu):
    ID, y_true, pd_raw, pd_final, pd_baseline
SEX và AGE lấy từ dữ liệu gốc theo `ID`.

Usage:
    python scripts/export_test_predictions.py            # chấm Test một lần
    python scripts/export_test_predictions.py --dry-run  # chạy thử cùng quy trình trên Valid, không đụng Test, không ghi file
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_final_test import FINAL_STUDY, FINAL_THRESHOLD, OUTPUT_DIR, RUN_MARKER  # noqa: E402

logger = logging.getLogger(__name__)

EXPORT_MARKER = OUTPUT_DIR / "test_predictions_export.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "test_metrics.csv"
REPORT_PATH = OUTPUT_DIR / "test_evaluation.md"

# Cột xuất -> tên dòng tương ứng trong test_metrics.csv (cột `model`).
REFERENCE_MODELS = {
    "pd_raw": "LightGBM tuned + Monotonic (raw)",
    "pd_final": "LightGBM tuned + Monotonic (Platt)",
    "pd_baseline": "Logistic Regression baseline",
}
# Dung sai khi so khớp mô hình huấn luyện lại với mô hình đã đánh giá (LightGBM đa luồng có thể lệch ở chữ số thập phân cuối).
MEAN_PD_TOLERANCE = 2e-3
FLAG_RATE_TOLERANCE = 5e-3


# ---------------------------------------------------------------------------
# Bảo vệ và kiểm tra
# ---------------------------------------------------------------------------

def check_preconditions(run_marker: Path, export_marker: Path, predictions_path: Path) -> None:
    """Báo lỗi nếu đánh giá Test chưa hoàn tất hoặc dự đoán Test đã được xuất."""
    if not run_marker.exists():
        raise RuntimeError(
            f"Chưa có marker đánh giá Test ({run_marker}). Script này chỉ xuất dự đoán sau khi "
            "evaluate_final_test.py hoàn tất, không dùng để chấm Test lần đầu."
        )
    status = json.loads(run_marker.read_text(encoding="utf-8")).get("status")
    if status != "completed":
        raise RuntimeError(f"Đánh giá Test chưa hoàn tất (status = {status!r}); không xuất dự đoán.")
    previous = next((path for path in (export_marker, predictions_path) if path.exists()), None)
    if previous is not None:
        raise RuntimeError(f"Dự đoán Test đã được xuất ({previous}); không chấm lại Test.")


def read_reported_flag_rate(report_path: Path) -> Optional[float]:
    """Flag rate ở ngưỡng đã khóa trong test_evaluation.md (cột đầu của dòng `| Flag rate | ... |`); None nếu không có."""
    if not report_path.exists():
        return None
    match = re.search(r"\|\s*Flag rate\s*\|\s*([\d.]+)%", report_path.read_text(encoding="utf-8"))
    return float(match.group(1)) / 100 if match else None


def consistency_checks(
    predictions: Dict[str, np.ndarray],
    recorded_metrics: pd.DataFrame,
    threshold: float,
    reported_flag_rate: Optional[float],
) -> List[dict]:
    """So khớp PD trung bình từng đầu ra và flag rate với kết quả đã ghi; không dùng nhãn."""
    checks = []
    for column, model in REFERENCE_MODELS.items():
        recorded = float(recorded_metrics.loc[recorded_metrics["model"] == model, "mean_predicted_pd"].iloc[0])
        observed = float(np.mean(predictions[column]))
        checks.append({
            "check": f"mean_{column}", "recorded": recorded, "observed": observed,
            "tolerance": MEAN_PD_TOLERANCE, "ok": bool(abs(observed - recorded) <= MEAN_PD_TOLERANCE),
        })
    if reported_flag_rate is not None:
        observed = float(np.mean(predictions["pd_final"] >= threshold))
        checks.append({
            "check": "flag_rate_at_threshold", "recorded": reported_flag_rate, "observed": observed,
            "tolerance": FLAG_RATE_TOLERANCE, "ok": bool(abs(observed - reported_flag_rate) <= FLAG_RATE_TOLERANCE),
        })
    return checks


def build_predictions_frame(ids: Sequence[int], y_true: Sequence[int], predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Bảng ID, y_true và các cột PD theo thứ tự REFERENCE_MODELS."""
    frame = pd.DataFrame({"ID": np.asarray(ids), "y_true": np.asarray(y_true).astype(int)})
    for column in REFERENCE_MODELS:
        frame[column] = np.asarray(predictions[column], dtype=float)
    if len(frame) != len(predictions["pd_final"]):
        raise ValueError("ID, y_true và dự đoán phải có cùng số dòng.")
    return frame


# ---------------------------------------------------------------------------
# Chấm điểm
# ---------------------------------------------------------------------------

def score_split(split: str, before_scoring: Optional[Callable[[], None]] = None) -> dict:
    """Huấn luyện mô hình đã khóa trên Train (cùng quy trình với evaluate_final_test.py), calibrate Platt trên Valid và
    chấm điểm `split` ('test' hoặc 'valid' cho chạy thử). Phần huấn luyện và calibrate chỉ dùng Train và Valid;
    `before_scoring` được gọi ngay trước khi chấm `split` (dùng để ghi marker khi sắp chạm Test)."""
    from sklearn.linear_model import LogisticRegression

    from src.calibrate import PlattCalibrator, predict_margin
    from src.config import load_config
    from src.data_split import load_split_data
    from src.features import build_features
    from src.pipelines import make_pipeline
    from src.tune import fit_tuned_pipeline

    cfg = load_config()
    target_col = cfg["data"]["target_col"]
    train_df, valid_df, test_df = load_split_data()
    scored_df = {"test": test_df, "valid": valid_df}[split]
    X_train, y_train = build_features(train_df, target_col=target_col)
    X_valid, y_valid = build_features(valid_df, target_col=target_col)

    final_model = fit_tuned_pipeline(FINAL_STUDY, X_train, y_train.to_numpy(dtype=int))
    baseline = make_pipeline(LogisticRegression(max_iter=1000, random_state=cfg.get("random_state", 42))).fit(
        X_train, y_train.to_numpy(dtype=int)
    )
    calibrator = PlattCalibrator().fit(predict_margin(final_model, X_valid), y_valid.to_numpy(dtype=int))

    if before_scoring is not None:
        before_scoring()
    X_scored, y_scored = build_features(scored_df, target_col=target_col)
    return {
        "ids": scored_df["ID"].to_numpy(),
        "y_true": y_scored.to_numpy(dtype=int),
        "predictions": {
            "pd_raw": final_model.predict_proba(X_scored)[:, 1],
            "pd_final": calibrator.transform(predict_margin(final_model, X_scored)),
            "pd_baseline": baseline.predict_proba(X_scored)[:, 1],
        },
    }


def _write_marker(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_test_predictions() -> Path:
    """Chấm Test một lần và ghi PREDICTIONS_PATH; ghi marker trước khi chấm để chặn chạy lại."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_preconditions(RUN_MARKER, EXPORT_MARKER, PREDICTIONS_PATH)

    marker = {
        "started_at_utc": None,
        "purpose": "Xuất dự đoán Test của mô hình đã khóa cho phần fairness; không tính chỉ số, không đổi mô hình/calibrator/ngưỡng.",
        "model": FINAL_STUDY, "calibration": "platt", "threshold": FINAL_THRESHOLD, "status": "started",
    }

    def mark_started() -> None:
        marker["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_marker(EXPORT_MARKER, marker)

    # Marker được ghi sau khi huấn luyện xong và ngay trước khi chấm Test, nên lỗi ở bước huấn luyện không chặn lần chạy sau.
    scored = score_split("test", before_scoring=mark_started)
    checks = consistency_checks(
        scored["predictions"], pd.read_csv(METRICS_PATH), FINAL_THRESHOLD, read_reported_flag_rate(REPORT_PATH)
    )
    marker.update({"checks": checks, "n_test": int(len(scored["ids"]))})
    if not all(c["ok"] for c in checks):
        marker["status"] = "failed_consistency"
        _write_marker(EXPORT_MARKER, marker)
        failed = [c["check"] for c in checks if not c["ok"]]
        raise RuntimeError(
            f"Mô hình huấn luyện lại không khớp mô hình đã đánh giá ({failed}); không ghi dự đoán. "
            f"Chi tiết trong {EXPORT_MARKER}."
        )

    build_predictions_frame(scored["ids"], scored["y_true"], scored["predictions"]).to_csv(
        PREDICTIONS_PATH, index=False, float_format="%.8f"
    )
    marker.update({
        "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "columns": ["ID", "y_true", *REFERENCE_MODELS],
        "artifacts": [PREDICTIONS_PATH.name],
    })
    _write_marker(EXPORT_MARKER, marker)
    logger.info("Đã xuất dự đoán Test (%d hồ sơ): %s", len(scored["ids"]), PREDICTIONS_PATH)
    return PREDICTIONS_PATH


def dry_run() -> dict:
    """Chạy cùng quy trình trên Valid để kiểm tra mã; không đọc nhãn/dự đoán của Test, không ghi file."""
    scored = score_split("valid")
    frame = build_predictions_frame(scored["ids"], scored["y_true"], scored["predictions"])
    summary = {
        "n": int(len(frame)),
        "default_rate": float(frame["y_true"].mean()),
        "mean_pd": {c: float(frame[c].mean()) for c in REFERENCE_MODELS},
        "flag_rate_at_threshold": float((frame["pd_final"] >= FINAL_THRESHOLD).mean()),
    }
    logger.info("Chạy thử trên Valid: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Xuất dự đoán Test của mô hình đã khóa (chấm Test một lần)")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử trên Valid, không đụng Test, không ghi file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    dry_run() if args.dry_run else export_test_predictions()


if __name__ == "__main__":
    main()
