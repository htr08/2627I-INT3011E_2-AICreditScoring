"""Lắp ráp các Pipeline mô hình từ preprocessing.py và features.py.

Tách riêng module này để tránh import vòng giữa preprocessing.py và features.py.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import AgeBinningTransformer, WoEIVTransformer
from src.preprocessing import DropColumnsTransformer


def build_scorecard_pipeline():
    """Xây dựng pipeline Logistic Scorecard (WoE) theo kế hoạch Tuần 2 - T3 / T5."""
    return Pipeline([
        ("drop_sensitive", DropColumnsTransformer(columns=["SEX", "ID"])),
        ("age_binning", AgeBinningTransformer(min_bins=3, max_bins=8)),
        ("woe_iv", WoEIVTransformer(n_bins=5, iv_threshold=0.02)),
        ("model", LogisticRegression(max_iter=1000)),
    ])
