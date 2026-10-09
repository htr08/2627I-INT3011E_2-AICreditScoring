
"""Pipeline runner for model training and feature evaluation.

Usage:
    python scripts/run_pipeline.py
        # Huấn luyện baseline models (LR & DT)
    python scripts/run_pipeline.py --mode advanced
        # Huấn luyện RF & XGBoost
    python scripts/run_pipeline.py --mode all
        # Huấn luyện baseline và advanced
    python scripts/run_pipeline.py --include-sex
        # Đối chiếu fairness
    python scripts/run_pipeline.py --mode woe
        # Báo cáo WoE/IV trên Train
    python scripts/run_pipeline.py --mode woe --freeze
        # Ghi configs/feature_freeze_v1.yaml
    python scripts/run_pipeline.py --mode scorecard --feature-selection
        # Scorecard với feature selection theo từng fold
"""

import argparse
import hashlib
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_split import load_split_data

from src.features import (
    ENGINEERED_COLUMNS,
    build_features,
    compute_feature_importance_stability,
    compute_iv_stability,
    find_high_correlation_features,
    prepare_feature_selection_data,
)

from src.pipelines import (
    AGE_MAX_BINS,
    AGE_MIN_BINS,
    SCORECARD_IV_THRESHOLD,
    SCORECARD_N_BINS,
    build_scorecard_pipeline,
)

from src.preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    SENSITIVE_COLUMNS,
)

from src.train import (
    train_baseline,
    train_rf_xgboost_default,
    train_scorecard,
)


FREEZE_PATH = PROJECT_ROOT / "configs" / "feature_freeze_v1.yaml"


def run_woe_pipeline(write_freeze: bool = False):
    """Đánh giá WoE/IV, correlation và độ ổn định feature trên Train."""

    train_df, _, _ = load_split_data()
    X_train, y_train = build_features(train_df)

    print("Train:", X_train.shape)

    # 1. Fit WoE/IV trên Train.
    pipeline = build_scorecard_pipeline().fit(
        X_train,
        y_train,
    )

    woe = pipeline.named_steps["woe_iv"]

    iv_sorted = sorted(
        woe.iv_values_.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    selected_features = set(woe.selected_features_)

    print("\nIV values (* = selected by IV threshold):")

    for feature, iv in iv_sorted:
        mark = "*" if feature in selected_features else " "
        print(f"  {mark} {feature}: {iv:.4f}")

    # 2. Correlation filtering.
    X_selection = prepare_feature_selection_data(
        X_train,
        y=y_train,
        preprocessed=True,
    )

    correlation_drop = find_high_correlation_features(
        X_selection,
        y_train,
        threshold=0.9,
    )

    print("\nHigh-correlation features to drop:")

    for feature in correlation_drop:
        print(f"  - {feature}")

    # 3. IV stability.
    # X_train đã qua build_features; không preprocessing lại.
    iv_stability = compute_iv_stability(
        X_train,
        y_train,
        n_splits=5,
        random_state=42,
        raw_credit_data=False,
    )

    # 4. Feature importance stability.
    importance_stability = compute_feature_importance_stability(
        X_train,
        y_train,
        n_splits=5,
        random_state=42,
        correlation_threshold=0.9,
        raw_credit_data=False,
    )

    # 5. Xuất báo cáo.
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    correlation_df = pd.DataFrame({
        "feature_to_drop": correlation_drop,
    })

    correlation_path = reports_dir / "correlation_woe.csv"
    correlation_df.to_csv(
        correlation_path,
        index=False,
        encoding="utf-8-sig",
    )

    iv_report = pd.DataFrame(
        [
            {
                "feature": feature,
                "iv": float(iv),
                "selected_by_iv_threshold": (
                    feature in selected_features
                ),
            }
            for feature, iv in iv_sorted
        ],
        columns=[
            "feature",
            "iv",
            "selected_by_iv_threshold",
        ],
    )

    iv_report_path = reports_dir / "information_value_woe.csv"
    iv_report.to_csv(
        iv_report_path,
        index=False,
        encoding="utf-8-sig",
    )

    importance_stability.to_csv(
        reports_dir / "feature_importance.csv",
        index=False,
    )

    iv_stability.to_csv(
        reports_dir / "iv_stability.csv",
        index=False,
    )

    print("\nReports written:")
    print("  reports/correlation_woe.csv")
    print("  reports/information_value_woe.csv")
    print("  reports/iv_stability.csv")
    print("  reports/feature_importance.csv")

    if write_freeze:
        write_feature_freeze(
            iv_sorted,
            woe.selected_features_,
        )


def write_feature_freeze(iv_sorted, selected):
    """Ghi danh sách đặc trưng chốt ra feature_freeze_v1.yaml."""

    splits_path = (
        PROJECT_ROOT
        / "data"
        / "splits"
        / "splits.json"
    )

    splits_hash = (
        hashlib.sha256(
            splits_path.read_bytes()
        ).hexdigest()[:16]
        if splits_path.exists()
        else None
    )

    freeze = {
        "version": "v1",
        "frozen_on": date.today().isoformat(),
        "splits_hash": splits_hash,
        "excluded_columns": SENSITIVE_COLUMNS,
        "model_features": {
            "note": (
                "Dùng cho LR/DT/RF/XGBoost qua "
                "src.preprocessing.build_preprocessing_pipeline"
            ),
            "categorical": CATEGORICAL_COLUMNS + ["AGE_BIN"],
            "numeric_raw": [
                column
                for column in NUMERIC_COLUMNS
                if column != "AGE"
            ],
            "age": (
                f"AGE_BIN: binning theo IV trên Train "
                f"({AGE_MIN_BINS}–{AGE_MAX_BINS} bin), "
                "one-hot; không dùng AGE liên tục"
            ),
            "numeric_engineered": ENGINEERED_COLUMNS,
        },
        "scorecard": {
            "note": (
                "Dùng cho src.pipelines.build_scorecard_pipeline; "
                "IV tính trên toàn bộ Train"
            ),
            "n_bins": SCORECARD_N_BINS,
            "age_bins": [
                AGE_MIN_BINS,
                AGE_MAX_BINS,
            ],
            "iv_threshold": SCORECARD_IV_THRESHOLD,
            "selected_features": list(selected),
            "iv": {
                feature: round(float(iv), 4)
                for feature, iv in iv_sorted
            },
        },
    }

    FREEZE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        FREEZE_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "# Feature freeze v1 - sinh tự động bởi: "
            "python scripts/run_pipeline.py --mode woe --freeze\n"
        )
        file.write(
            "# Mọi thay đổi đặc trưng sau mốc này phải được "
            "cả nhóm thống nhất (project_plan.md).\n"
        )

        yaml.safe_dump(
            freeze,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    print(
        f"\nWrote {FREEZE_PATH.relative_to(PROJECT_ROOT)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Credit Scoring Pipeline Runner"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "advanced",
            "all",
            "scorecard",
            "woe",
        ],
        default="baseline",
        help=(
            "Pipeline mode: baseline, advanced, all, "
            "scorecard hoặc woe"
        ),
    )

    parser.add_argument(
        "--include-sex",
        action="store_true",
        help=(
            "Giữ SEX để đối chiếu fairness; "
            "run có hậu tố _with_sex và không đăng ký model"
        ),
    )

    parser.add_argument(
        "--feature-selection",
        action="store_true",
        help=(
            "Bật feature selection trong scorecard; "
            "xuất báo cáo theo từng fold"
        ),
    )

    parser.add_argument(
        "--freeze",
        action="store_true",
        help=(
            "Ghi configs/feature_freeze_v1.yaml "
            "khi chạy mode woe"
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.mode == "woe":
        run_woe_pipeline(
            write_freeze=args.freeze,
        )
        return

    if args.freeze:
        parser.error("--freeze chỉ dùng với --mode woe")

    if args.feature_selection and args.mode != "scorecard":
        parser.error(
            "--feature-selection chỉ dùng với --mode scorecard"
        )

    if args.mode in ("baseline", "all"):
        train_baseline(
            include_sex=args.include_sex,
        )

    if args.mode in ("advanced", "all"):
        train_rf_xgboost_default(
            include_sex=args.include_sex,
        )

    if args.mode == "scorecard":
        train_scorecard(
            include_sex=args.include_sex,
            enable_feature_selection=args.feature_selection,
        )


if __name__ == "__main__":
    main()