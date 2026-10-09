"""Lắp ráp các Pipeline mô hình từ preprocessing.py và features.py.

Tách riêng module này để tránh import vòng giữa preprocessing.py và features.py.
"""

from typing import Optional

import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
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
SCORECARD_CORRELATION_THRESHOLD = 0.9
# Coarse classing cho bảng điểm (Tuần 2 - T5): 20 bin phân vị ban đầu, gộp đến khi mỗi bin
# có >= 5% mẫu và WoE của biến liên tục đơn điệu (reports/logistic_scorecard.md mục 2.2).
SCORECARD_FINE_BINS = 20
SCORECARD_MIN_BIN_SHARE = 0.05
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


class SignConstrainedLogisticRegression(BaseEstimator, ClassifierMixin):
    """Logistic Regression trên WoE, loại dần biến có hệ số sai dấu (Tuần 2 - T5).

    Với WoE = ln(%good / %bad), mọi hệ số phải âm (WoE cao -> rủi ro thấp). Biến tương quan mạnh
    (PAY_1..6, PAY_MAX, PAY_MEAN, ...) làm một phần hệ số đổi dấu, khiến bảng điểm cộng điểm cho
    thuộc tính xấu. Mỗi vòng loại biến có hệ số dương lớn nhất rồi fit lại, đến khi không còn
    hệ số dương. Thuộc tính sau fit: selected_features_, coef_, intercept_ (theo selected_features_).
    """

    def __init__(self, max_iter: int = 1000):
        self.max_iter = max_iter

    def fit(self, X, y):
        features = list(X.columns)
        while True:
            model = LogisticRegression(max_iter=self.max_iter).fit(X[features], y)
            if len(features) == 1 or not (model.coef_[0] > 0).any():
                break
            features.pop(int(np.argmax(model.coef_[0])))
        self.model_ = model
        self.selected_features_ = features
        self.dropped_features_ = [c for c in X.columns if c not in features]
        self.classes_ = model.classes_
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X[self.selected_features_])

    def predict(self, X):
        return self.model_.predict(X[self.selected_features_])


def build_feature_frame_pipeline(drop_sensitive: bool = True) -> Pipeline:
    """Cột gốc -> bộ đặc trưng feature freeze v1 dạng DataFrame (chưa encode/scale, chưa WoE).

    Dùng cho phân tích IV / tương quan / độ ổn định (Tuần 2 - T4). AGE được thay bằng AGE_BIN
    theo IV, nên pipeline phải được fit lại trên dữ liệu train của từng fold.
    """
    steps = [
        ("abnormal_codes", AbnormalCodeTransformer()),
        ("feature_eng", FeatureEngineeringTransformer()),
        ("age_binning", AgeBinningTransformer(min_bins=AGE_MIN_BINS, max_bins=AGE_MAX_BINS)),
        ("drop_sensitive", DropColumnsTransformer(columns=SENSITIVE_COLUMNS if drop_sensitive else ["ID"])),
    ]
    return Pipeline(steps)


def build_scorecard_pipeline(
    drop_sensitive: bool = True,
    enable_feature_selection: bool = False,
    enable_correlation_filter: bool = True,
    sign_constrained: bool = True,
    coarse_classing: bool = True,
) -> Pipeline:
    """Pipeline Logistic Scorecard (WoE) theo kế hoạch Tuần 2 - T3 / T5.

    Mọi bước có học tham số (age binning, WoE, lọc IV / tương quan) nằm trong Pipeline nên
    được fit lại trong từng fold khi chạy cross-validation.

    Args:
        drop_sensitive: True cho mô hình chính thức (bỏ SEX, ID); False giữ SEX để đối chiếu fairness.
        enable_feature_selection: False (mặc định) giữ feature freeze v1, WoEIVTransformer lọc theo
            ngưỡng IV. True thêm bước FeatureSelectionTransformer (IV + tương quan) — chỉ dùng cho
            thí nghiệm lọc đặc trưng; SEX (nếu có) đi thẳng qua bước lọc.
        enable_correlation_filter: (khi enable_feature_selection=True) bật/tắt lọc tương quan.
        sign_constrained: True (mặc định) dùng SignConstrainedLogisticRegression để mọi hệ số WoE
            cùng dấu, bảng điểm nhất quán; False dùng LogisticRegression thường (cấu hình T3).
        coarse_classing: True (mặc định) gộp bin WoE (mỗi bin >= 5% mẫu, WoE đơn điệu với biến
            liên tục); False dùng 5 bin phân vị, mỗi giá trị rời rạc một bin (cấu hình T3).
    """
    steps = [
        ("abnormal_codes", AbnormalCodeTransformer()),
        ("feature_eng", FeatureEngineeringTransformer()),
        ("age_binning", AgeBinningTransformer(min_bins=AGE_MIN_BINS, max_bins=AGE_MAX_BINS)),
        ("drop_sensitive", DropColumnsTransformer(columns=SENSITIVE_COLUMNS if drop_sensitive else ["ID"])),
    ]
    if enable_feature_selection:
        steps.append((
            "feature_selection",
            FeatureSelectionTransformer(
                iv_threshold=SCORECARD_IV_THRESHOLD,
                correlation_threshold=SCORECARD_CORRELATION_THRESHOLD,
                n_bins=SCORECARD_N_BINS,
                passthrough_columns=("SEX",),
                enable_correlation_filter=enable_correlation_filter,
            ),
        ))
    # Khi đã có bước feature_selection, WoE không lọc thêm theo IV (ngưỡng 0).
    iv_threshold = 0.0 if enable_feature_selection else SCORECARD_IV_THRESHOLD
    woe_params = (
        dict(n_bins=SCORECARD_FINE_BINS, min_bin_share=SCORECARD_MIN_BIN_SHARE, monotonic=True)
        if coarse_classing
        else dict(n_bins=SCORECARD_N_BINS)
    )
    steps += [
        ("woe_iv", WoEIVTransformer(**woe_params, iv_threshold=iv_threshold)),
        ("model", SignConstrainedLogisticRegression() if sign_constrained else LogisticRegression(max_iter=1000)),
    ]
    return Pipeline(steps)
