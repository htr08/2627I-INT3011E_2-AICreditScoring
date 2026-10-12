"""Quy đổi PD sang điểm tín dụng theo phương pháp PDO (reports/research_shap_scorecard_pdo.md, mục 4).

Score = Offset + Factor * ln(odds tốt/xấu), với Factor = PDO / ln(2), Offset = Base Score - Factor * ln(Base Odds).
Tham số mặc định: 600 điểm tại odds 50:1, PDO = 20 (chưa đưa vào configs/config.yaml, chờ kết quả Tuần 2 của B để thống nhất).
"""

import numpy as np
import pandas as pd

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


def _bin_order(item):
    """Thứ tự hiển thị bin: khoảng theo cận dưới, giá trị số theo độ lớn, bin thiếu (NaN) cuối cùng."""
    label = item[0]
    if hasattr(label, "left"):
        return (0, float(label.left), "")
    try:
        return (0, float(label), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(label))


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
        for bin_label, woe_value in sorted(woe.woe_maps_[feature].items(), key=_bin_order):
            rows.append({
                "feature": feature,
                "bin": str(bin_label),
                "woe": float(woe_value),
                "coef": float(coefs[feature]),
                "points": float(-factor * coefs[feature] * woe_value),
            })
    return pd.DataFrame(rows)

def calculate_ks_statistic(y_true, y_prob) -> float:
    """Tính chỉ số Kolmogorov-Smirnov (KS) statistic giữa 2 phân phối Good/Bad."""
    import pandas as pd

    df_eval = pd.DataFrame({"target": y_true, "prob": y_prob}).sort_values("prob", ascending=True)
    df_eval["good"] = (df_eval["target"] == 0).astype(int)
    df_eval["bad"] = (df_eval["target"] == 1).astype(int)

    total_good = df_eval["good"].sum()
    total_bad = df_eval["bad"].sum()

    if total_good == 0 or total_bad == 0:
        return 0.0

    df_eval["cum_good"] = df_eval["good"].cumsum() / total_good
    df_eval["cum_bad"] = df_eval["bad"].cumsum() / total_bad

    ks_stat = float(np.max(np.abs(df_eval["cum_good"] - df_eval["cum_bad"])))
    return ks_stat


def compare_models_performance(models_dict: dict, X_test, y_test) -> "pd.DataFrame":
    """So sánh hiệu năng giữa Logistic Scorecard và các mô hình Black-box.

    Args:
        models_dict: Dictionary dạng {"Tên mô hình": pipeline_da_fit}.
        X_test: Tập đặc trưng kiểm thử.
        y_test: Nhãn thực tế (1: Bad, 0: Good).

    Returns:
        DataFrame tổng hợp các chỉ số AUC, Gini, KS.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    results = []
    for model_name, pipeline in models_dict.items():
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            continue

        auc = float(roc_auc_score(y_test, y_prob))
        gini = float(2 * auc - 1)
        ks = calculate_ks_statistic(y_test, y_prob)

        results.append({
            "Model": model_name,
            "AUC": round(auc, 4),
            "Gini": round(gini, 4),
            "KS": round(ks, 4),
        })

    return pd.DataFrame(results)


def export_scorecard_to_dict(scorecard_df: "pd.DataFrame") -> dict:
    """Chuyển scorecard_points_table thành dạng Dictionary tra cứu điểm nhanh.

    Args:
        scorecard_df: DataFrame trả về từ `scorecard_points_table()`.
    """
    card = {}
    base_row = scorecard_df[scorecard_df["feature"] == "(base)"]
    base_pt = float(base_row["points"].values[0]) if not base_row.empty else 0.0

    card["base_points"] = base_pt
    card["features"] = {}

    for feature, group in scorecard_df[scorecard_df["feature"] != "(base)"].groupby("feature"):
        card["features"][feature] = {}
        for _, row in group.iterrows():
            card["features"][feature][row["bin"]] = {
                "woe": row["woe"],
                "points": row["points"],
            }

    return card

def calculate_ks_statistic(y_true, y_prob) -> float:
    """Tính chỉ số Kolmogorov-Smirnov (KS) statistic giữa 2 phân phối Good/Bad."""
    import pandas as pd

    df_eval = pd.DataFrame({"target": y_true, "prob": y_prob}).sort_values("prob", ascending=True)
    df_eval["good"] = (df_eval["target"] == 0).astype(int)
    df_eval["bad"] = (df_eval["target"] == 1).astype(int)

    total_good = df_eval["good"].sum()
    total_bad = df_eval["bad"].sum()

    if total_good == 0 or total_bad == 0:
        return 0.0

    df_eval["cum_good"] = df_eval["good"].cumsum() / total_good
    df_eval["cum_bad"] = df_eval["bad"].cumsum() / total_bad

    ks_stat = float(np.max(np.abs(df_eval["cum_good"] - df_eval["cum_bad"])))
    return ks_stat


def evaluate_model_metrics(y_true, y_prob) -> dict:
    """Tính toán bộ 3 chỉ số đánh giá tiêu chuẩn tín dụng: AUC, Gini, KS."""
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y_true, y_prob))
    gini = float(2 * auc - 1)
    ks = calculate_ks_statistic(y_true, y_prob)

    return {"AUC": auc, "Gini": gini, "KS": ks}
