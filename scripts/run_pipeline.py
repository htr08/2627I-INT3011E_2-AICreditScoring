import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_split import load_split_data
from src.features import (
    create_utilization_features,
    create_payment_trend_features,
    create_payment_ratio_features,
    create_bill_variation_features,
    create_min_payment_features,
)
from src.preprocessing import (
    AbnormalCodeTransformer,
    AgeBinningTransformer,
    WoEIVTransformer,
)

TARGET = "default.payment.next.month"


def main():
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

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    # Fit age bins using training data only
    age_binning = AgeBinningTransformer(
        min_bins=3,
        max_bins=8
    )

    X_train = age_binning.fit_transform(
        X_train,
        y_train
    )

    # Fit WoE and IV using training data only
    transformer = WoEIVTransformer(
        n_bins=5,
        iv_threshold=0.02
    )

    transformer.fit(X_train, y_train)

    print("\nSelected features:")
    for feature in transformer.selected_features_:
        print(feature)

    print("\nIV values:")
    for feature, iv in sorted(
        transformer.iv_values_.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{feature}: {iv:.4f}")


if __name__ == "__main__":
    main()