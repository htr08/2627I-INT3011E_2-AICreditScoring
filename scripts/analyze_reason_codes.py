"""Thống kê reason codes của mô hình cuối trên tập Valid, theo mức cảnh báo (Tuần 3 – T4, Thành viên C).

Đóng góp theo nhóm đặc trưng ở thang điểm PDO (điểm âm là làm giảm điểm, tức tăng rủi ro). Dùng đúng adapter của demo
(`app/final_model.py`), nên số liệu khớp với những gì demo hiển thị. Tập Test không được sử dụng.

Usage:
    python scripts/analyze_reason_codes.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.explanation import MIN_REASON_POINTS  # noqa: E402
from app.final_model import FinalModel  # noqa: E402
from app.risk_tiers import TIERS, classify, load_thresholds  # noqa: E402
from src.data_split import load_split_data  # noqa: E402
from src.explain import group_importance_share  # noqa: E402
from src.features import build_features  # noqa: E402
from src.scoring import shap_to_points  # noqa: E402


def group_points_on_valid(model: FinalModel):
    """(DataFrame đóng góp theo nhóm ở thang điểm, PD, nhãn) cho toàn bộ tập Valid."""
    _, valid_df, _ = load_split_data()
    X, y = build_features(valid_df)
    Z = np.asarray(model._preprocess.transform(X), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        values = model._explainer.shap_values(Z)
    values = values[-1] if isinstance(values, list) else values
    from src.explain import ShapResult

    result = ShapResult(np.asarray(values), 0.0, np.zeros(len(X)), model._columns, model._categorical_vars)
    by_group_logit = result.by_group(model._groups) * model._a
    points = by_group_logit.apply(lambda col: shap_to_points(col.to_numpy()))
    pd_valid = model.calibrated.predict_proba(X)[:, 1]
    return points, pd_valid, y.to_numpy()


def main() -> None:
    model = FinalModel.from_file()
    watch, alert = load_thresholds()
    points, pd_valid, y = group_points_on_valid(model)
    tiers = np.array([classify(p, watch, alert).label for p in pd_valid])
    print(f"Valid: n = {len(y)}, ngưỡng theo dõi {watch:g}, cảnh báo cao {alert:g}\n")

    share = group_importance_share(points)
    print("Tỷ trọng mean(|điểm|) theo nhóm:")
    print((share * 100).round(1).sort_values(ascending=False).to_string(), "\n")

    rows = []
    for tier in TIERS:
        mask = tiers == tier.label
        sub = points[mask]
        # Lý do đầu tiên: nhóm làm giảm điểm nhiều nhất (điểm âm nhất); chỉ xét nếu đóng góp <= -MIN_REASON_POINTS.
        top = sub.idxmin(axis=1).where(sub.min(axis=1) <= -MIN_REASON_POINTS, "Không có lý do làm giảm điểm")
        shares = top.value_counts(normalize=True) * 100
        rows.append({"Mức": tier.label, "n": int(mask.sum()), "Tỷ lệ vỡ nợ thực tế (%)": round(100 * y[mask].mean(), 1),
                     "Số nhóm làm giảm điểm (TB)": round(float((sub <= -MIN_REASON_POINTS).sum(axis=1).mean()), 2),
                     **{g: round(float(shares.get(g, 0.0)), 1) for g in list(points.columns) + ["Không có lý do làm giảm điểm"]}})
    print("Lý do đầu tiên (nhóm làm giảm điểm nhiều nhất) theo mức cảnh báo, % hồ sơ:")
    print(pd.DataFrame(rows).set_index("Mức").T.to_string(), "\n")

    print("Đóng góp trung bình theo nhóm (điểm) theo mức cảnh báo:")
    print(pd.DataFrame({t.label: points[tiers == t.label].mean() for t in TIERS}).round(1).to_string())


if __name__ == "__main__":
    main()
