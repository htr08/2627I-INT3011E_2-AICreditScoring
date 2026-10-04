"""Pipeline runner for model training and feature evaluation.

Usage:
    python scripts/run_pipeline.py              # Huấn luyện baseline models (LR & DT)
    python scripts/run_pipeline.py --mode woe    # Chạy trích xuất đặc trưng và WoE/IV
"""

import argparse
import logging
import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc dự án luôn có trong sys.path khi chạy script trực tiếp
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_split import load_split_data
from src.features import (
    AgeBinningTransformer,
    WoEIVTransformer,
    create_bill_variation_features,
    create_min_payment_features,
    create_payment_ratio_features,
    create_payment_trend_features,
    create_utilization_features,
)
from src.preprocessing import AbnormalCodeTransformer
from src.train import train_baseline


def run_woe_pipeline():
    """Chạy trích xuất đặc trưng mới và phân tích WoE / IV."""
    target_col = "default.payment.next.month"
    train_df, valid_df, test_df = load_split_data()

    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test :", test_df.shape)

    normalizer = AbnormalCodeTransformer()
    train_df = normalizer.fit_transform(train_df)
    valid_df = normalizer.transform(valid_df)
    test_df = normalizer.transform(test_df)

    train_df = create_utilization_features(train_df)
    train_df = create_payment_trend_features(train_df)
    train_df = create_payment_ratio_features(train_df)
    train_df = create_bill_variation_features(train_df)
    train_df = create_min_payment_features(train_df)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    age_binning = AgeBinningTransformer(min_bins=3, max_bins=8)
    X_train = age_binning.fit_transform(X_train, y_train)

    transformer = WoEIVTransformer(n_bins=5, iv_threshold=0.02)
    transformer.fit(X_train, y_train)

    print("\nSelected features:")
    for feature in transformer.selected_features_:
        print(f"  - {feature}")

    print("\nIV values:")
    for feature, iv in sorted(
        transformer.iv_values_.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {feature}: {iv:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Credit Scoring Pipeline Runner")
    parser.add_argument(
        "--mode",
        choices=["baseline", "woe"],
        default="baseline",
        help="Pipeline mode to run: 'baseline' (default) or 'woe'",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.mode == "woe":
        run_woe_pipeline()
    else:
        train_baseline()


if __name__ == "__main__":
    main()
