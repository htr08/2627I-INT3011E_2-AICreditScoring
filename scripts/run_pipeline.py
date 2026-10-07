"""Pipeline runner for model training and feature evaluation.

Usage:
    python scripts/run_pipeline.py                    # Huấn luyện baseline models (LR & DT)
    python scripts/run_pipeline.py --mode advanced    # Huấn luyện RF & XGBoost (tham số mặc định)
    python scripts/run_pipeline.py --mode all         # Cả baseline và advanced
    python scripts/run_pipeline.py --include-sex      # Bản đối chiếu fairness (có SEX, không đăng ký model)
    python scripts/run_pipeline.py --mode woe    # Báo cáo WoE/IV trên Train
    python scripts/run_pipeline.py --mode woe --freeze   # Ghi configs/feature_freeze_v1.yaml
"""

import argparse
import hashlib
import logging
import sys
from datetime import date
from pathlib import Path

import yaml

# Đảm bảo đường dẫn gốc dự án luôn có trong sys.path khi chạy script trực tiếp
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_split import load_split_data
from src.features import ENGINEERED_COLUMNS, build_features
from src.pipelines import (
    AGE_MAX_BINS,
    AGE_MIN_BINS,
    SCORECARD_IV_THRESHOLD,
    SCORECARD_N_BINS,
    build_scorecard_pipeline,
)
from src.preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, SENSITIVE_COLUMNS
from src.train import train_baseline, train_rf_xgboost_default

FREEZE_PATH = PROJECT_ROOT / "configs" / "feature_freeze_v1.yaml"


def run_woe_pipeline(write_freeze: bool = False):
    """Fit scorecard pipeline trên toàn bộ Train để báo cáo IV (chỉ dùng xem báo cáo,
    đánh giá mô hình phải chạy CV với cùng pipeline)."""
    train_df, _, _ = load_split_data()
    X_train, y_train = build_features(train_df)
    print("Train:", X_train.shape)

    pipeline = build_scorecard_pipeline().fit(X_train, y_train)
    woe = pipeline.named_steps["woe_iv"]
    iv_sorted = sorted(woe.iv_values_.items(), key=lambda x: x[1], reverse=True)

    print("\nIV values (* = selected):")
    for feature, iv in iv_sorted:
        mark = "*" if feature in woe.selected_features_ else " "
        print(f"  {mark} {feature}: {iv:.4f}")

    if write_freeze:
        write_feature_freeze(iv_sorted, woe.selected_features_)


def write_feature_freeze(iv_sorted, selected):
    """Ghi danh sách đặc trưng chốt (feature freeze v1) ra configs/feature_freeze_v1.yaml."""
    splits_path = PROJECT_ROOT / "data" / "splits" / "splits.json"
    splits_hash = (
        hashlib.sha256(splits_path.read_bytes()).hexdigest()[:16] if splits_path.exists() else None
    )
    freeze = {
        "version": "v1",
        "frozen_on": date.today().isoformat(),
        "splits_hash": splits_hash,
        "excluded_columns": SENSITIVE_COLUMNS,
        "model_features": {
            "note": "Dùng cho LR/DT/RF/XGBoost qua src.preprocessing.build_preprocessing_pipeline",
            "categorical": CATEGORICAL_COLUMNS,
            "numeric_raw": NUMERIC_COLUMNS,
            "numeric_engineered": ENGINEERED_COLUMNS,
        },
        "scorecard": {
            "note": "Dùng cho src.pipelines.build_scorecard_pipeline; IV tính trên toàn bộ Train",
            "n_bins": SCORECARD_N_BINS,
            "age_bins": [AGE_MIN_BINS, AGE_MAX_BINS],
            "iv_threshold": SCORECARD_IV_THRESHOLD,
            "selected_features": list(selected),
            "iv": {feature: round(float(iv), 4) for feature, iv in iv_sorted},
        },
    }
    with open(FREEZE_PATH, "w", encoding="utf-8") as f:
        f.write("# Feature freeze v1 - sinh tự động bởi: python scripts/run_pipeline.py --mode woe --freeze\n")
        f.write("# Mọi thay đổi đặc trưng sau mốc này phải được cả nhóm thống nhất (project_plan.md).\n")
        yaml.safe_dump(freeze, f, allow_unicode=True, sort_keys=False)
    print(f"\nWrote {FREEZE_PATH.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Credit Scoring Pipeline Runner")
    parser.add_argument(
        "--mode",
        choices=["baseline", "advanced", "all", "woe"],
        default="baseline",
        help="Pipeline mode to run: 'baseline' (default), 'advanced' (RF & XGBoost), 'all' or 'woe'",
    )
    parser.add_argument(
        "--include-sex",
        action="store_true",
        help="Giữ SEX để đối chiếu fairness; run có hậu tố _with_sex và không đăng ký model",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="(mode woe) Ghi danh sách đặc trưng chốt ra configs/feature_freeze_v1.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.mode == "woe":
        run_woe_pipeline(write_freeze=args.freeze)
    else:
        if args.mode in ("baseline", "all"):
            train_baseline(include_sex=args.include_sex)
        if args.mode in ("advanced", "all"):
            train_rf_xgboost_default(include_sex=args.include_sex)


if __name__ == "__main__":
    main()
