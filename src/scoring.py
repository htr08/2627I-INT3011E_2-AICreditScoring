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
