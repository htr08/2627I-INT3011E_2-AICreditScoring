"""Lắp ráp các Pipeline mô hình từ preprocessing.py và features.py.

Tách riêng module này để tránh import vòng giữa preprocessing.py và features.py.
"""

from typing import Optional

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import (
    AgeBinningTransformer,
    FeatureEngineeringTransformer,
    FeatureSelectionTransformer,
    WoEIVTransformer,
)

from src.preprocessing import (
    SENSITIVE_COLUMNS,
    AbnormalCodeTransformer,
    DropColumnsTransformer,
    build_preprocessing_pipeline,
)

SCORECARD_N_BINS = 5
SCORECARD_IV_THRESHOLD = 0.02
AGE_MIN_BINS = 3
AGE_MAX_BINS = 8


def make_pipeline(
    estimator,
    include_sex: bool = False,
    sampler: Optional[object] = None,
) -> Pipeline:
    """Pipeline mô hình (feature freeze v1): tiền xử lý của preprocessing.py -> estimator.

    Nhận cột gốc của CSV (có PAY_0). Mọi bước có học tham số chỉ fit trên dữ liệu train
    của từng fold — tránh data leakage.

    Args:
        estimator: Scikit-learn estimator (LogisticRegression, DecisionTreeClassifier, …).
        include_sex: False cho mô hình chính thức; True để đối chiếu fairness (Charter mục 1.3).
        sampler: Đối tượng sampler (như SMOTE). Nếu được truyền, Pipeline từ imblearn sẽ được dùng
                 để chỉ resample dữ liệu huấn luyện trong từng fold khi fit, không ảnh hưởng fold val.
    """
    prep = build_preprocessing_pipeline(drop_sensitive=not include_sex)
    if sampler is None:
        return Pipeline([
            ("preprocess", prep),
            ("clf", estimator),
        ])

    steps = list(prep.steps) + [
        ("sampler", sampler),
        ("clf", estimator),
    ]
    return ImbPipeline(steps)

def build_scorecard_pipeline(
    drop_sensitive: bool = True,
    enable_feature_selection: bool = False,
    enable_correlation_filter: bool = True,
) -> Pipeline:
    """Pipeline Logistic Scorecard.

    enable_feature_selection=False: giữ nguyên feature freeze v1.
    enable_feature_selection=True: bật lọc biến cho thí nghiệm fairness.

    Hai chế độ chỉ khác nhau ở việc có loại SEX hay không.
    ID luôn bị loại ở cả hai chế độ.
    """
    steps = [
        ("abnormal_codes", AbnormalCodeTransformer()),
        ("feature_eng", FeatureEngineeringTransformer()),
        (
            "age_binning",
            AgeBinningTransformer(
                min_bins=AGE_MIN_BINS,
                max_bins=AGE_MAX_BINS,
            ),
        ),
        (
            "drop_id",
            DropColumnsTransformer(columns=["ID"]),
        ),
    ]

    if drop_sensitive:
        steps.append(
            ("drop_sex", DropColumnsTransformer(columns=["SEX"]))
        )

    if enable_feature_selection:
        steps.append(
            (
                "feature_selection",
                FeatureSelectionTransformer(
                    iv_threshold=SCORECARD_IV_THRESHOLD,
                    correlation_threshold=0.9,
                    n_bins=SCORECARD_N_BINS,
                    passthrough_columns=("SEX",),
                    enable_correlation_filter=enable_correlation_filter,
                )
            )
        )

    steps.extend([
        (
            "woe_iv",
            WoEIVTransformer(
                n_bins=SCORECARD_N_BINS,
                iv_threshold=(
                    0.0 if enable_feature_selection
                    else SCORECARD_IV_THRESHOLD
                ),
            ),
        ),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    return Pipeline(steps)