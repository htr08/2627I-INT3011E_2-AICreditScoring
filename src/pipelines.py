"""Lắp ráp các Pipeline mô hình từ preprocessing.py và features.py.

Tách riêng module này để tránh import vòng giữa preprocessing.py và features.py.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import AgeBinningTransformer, FeatureEngineeringTransformer, WoEIVTransformer
from src.preprocessing import SENSITIVE_COLUMNS, AbnormalCodeTransformer, DropColumnsTransformer

SCORECARD_N_BINS = 5
SCORECARD_IV_THRESHOLD = 0.02
AGE_MIN_BINS = 3
AGE_MAX_BINS = 8


def build_scorecard_pipeline(drop_sensitive: bool = True) -> Pipeline:
    """Pipeline Logistic Scorecard (WoE) theo kế hoạch Tuần 2 - T3 / T5.

    Mọi bước có học tham số (age binning, WoE, lọc IV) nằm trong Pipeline nên được fit lại
    trong từng fold khi chạy cross-validation.
    """
    steps = [
        ("abnormal_codes", AbnormalCodeTransformer()),
        ("feature_eng", FeatureEngineeringTransformer()),
    ]
    if drop_sensitive:
        steps.append(("drop_sensitive", DropColumnsTransformer(columns=SENSITIVE_COLUMNS)))
    steps += [
        ("age_binning", AgeBinningTransformer(min_bins=AGE_MIN_BINS, max_bins=AGE_MAX_BINS)),
        ("woe_iv", WoEIVTransformer(n_bins=SCORECARD_N_BINS, iv_threshold=SCORECARD_IV_THRESHOLD)),
        ("model", LogisticRegression(max_iter=1000)),
    ]
    return Pipeline(steps)
