"""Quy đổi PD sang điểm tín dụng theo phương pháp PDO (reports/research_shap_scorecard_pdo.md, mục 4).

Score = Offset + Factor * ln(odds tốt/xấu), với Factor = PDO / ln(2), Offset = Base Score - Factor * ln(Base Odds).
Tham số mặc định: 600 điểm tại odds 50:1, PDO = 20 (chưa đưa vào configs/config.yaml, chờ kết quả Tuần 2 của B để thống nhất).
"""

import numpy as np

BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20
SCORE_CLIP = (300, 850)


def pdo_params(base_score=BASE_SCORE, base_odds=BASE_ODDS, pdo=PDO):
    """Trả về (factor, offset) của phép quy đổi tuyến tính log-odds -> điểm."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset


def pd_to_score(pd, base_score=BASE_SCORE, base_odds=BASE_ODDS, pdo=PDO, clip=SCORE_CLIP, eps=1e-6):
    """PD (xác suất vỡ nợ) -> điểm. PD càng cao điểm càng thấp; clip=None để lấy điểm chưa giới hạn."""
    factor, offset = pdo_params(base_score, base_odds, pdo)
    pd = np.clip(pd, eps, 1 - eps)
    score = offset + factor * np.log((1 - pd) / pd)
    return np.clip(score, *clip) if clip else score


def base_points(base_logit, base_score=BASE_SCORE, base_odds=BASE_ODDS, pdo=PDO):
    """Điểm nền ứng với base value của SHAP (thang logit(PD)): Offset - Factor * phi_0."""
    factor, offset = pdo_params(base_score, base_odds, pdo)
    return offset - factor * base_logit


def shap_to_points(shap_logit, pdo=PDO):
    """SHAP trên thang logit(PD) đã hiệu chuẩn -> điểm đóng góp (dương = tăng điểm)."""
    return -np.asarray(shap_logit) * pdo / np.log(2)


def scorecard_points_table(pipeline, base_score=BASE_SCORE, base_odds=BASE_ODDS, pdo=PDO):
    """Bảng điểm của Logistic Scorecard (build_scorecard_pipeline đã fit) theo cùng thang PDO.

    logit(PD) = a + sum_j b_j * WoE_j, nên Score = Offset - Factor * logit(PD) tách thành
    điểm nền (Offset - Factor * a) cộng điểm từng thuộc tính (-Factor * b_j * WoE_ij).
    Tổng điểm của một hồ sơ (chưa clip) bằng pd_to_score(PD, clip=None).
    Giá trị chưa gặp khi fit nhận WoE = 0, tức 0 điểm. Với SignConstrainedLogisticRegression,
    chỉ các biến còn lại sau khi loại hệ số sai dấu (model.selected_features_) có trong bảng.
    """
    import pandas as pd

    woe = pipeline.named_steps["woe_iv"]
    model = pipeline.named_steps["model"]
    factor, _ = pdo_params(base_score, base_odds, pdo)
    features = getattr(model, "selected_features_", woe.selected_features_)
    coefs = dict(zip(features, model.coef_[0]))

    rows = [{
        "feature": "(base)",
        "bin": "",
        "woe": np.nan,
        "coef": float(model.intercept_[0]),
        "points": float(base_points(model.intercept_[0], base_score, base_odds, pdo)),
    }]
    for feature in features:
        for bin_label, woe_value in woe.woe_maps_[feature].items():
            rows.append({
                "feature": feature,
                "bin": str(bin_label),
                "woe": float(woe_value),
                "coef": float(coefs[feature]),
                "points": float(-factor * coefs[feature] * woe_value),
            })
    return pd.DataFrame(rows)
