"""Đánh giá công bằng (fairness) của mô hình cuối trên tập Valid và ghi báo cáo reports/fairness_evaluation.md.

Task: Tuần 3 – T3 (Thành viên A). Tập Test không được sử dụng.

- Mô hình chính: LightGBM tuned + monotonic, Platt scaling fit trên Valid (cùng quy trình với
  `python scripts/run_pipeline.py --mode calibrate`, yêu cầu src/calibrate.py), không dùng SEX; kèm bản có SEX để đối chiếu.
- Mô hình đối chiếu: Logistic Scorecard (WoE), không dùng SEX; kèm bản có SEX.
- Ngưỡng: ngưỡng vận hành do thành viên B khóa trước khi đánh giá Test (mặc định 0.09) và ngưỡng đối chiếu 0.5.
- Nhóm: giới tính (Nam, Nữ) và nhóm tuổi (<30, 30–50, >50). Chỉ số tính ở src/fairness.py.

Usage:
    python scripts/analyze_fairness.py
    python scripts/analyze_fairness.py --threshold 0.09 --reference-threshold 0.5
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Đảm bảo console Windows in tiếng Việt UTF-8 không lỗi
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.fairness import (  # noqa: E402
    AGE_GROUP_ORDER,
    DIR_BAND,
    assign_age_groups,
    dir_within_band,
    group_fairness_table,
    label_sex,
    overall_summary,
    select_reference_group,
)

FINAL_STUDY = "lightgbm_monotone"
# Ngưỡng PD do thành viên B khóa trước khi chạy Test: làm tròn từ trung bình OOF EAD-weighted (0.088),
# tỷ lệ chi phí FN:FP = 9:1 (reports/test_evaluation/test_evaluation.md).
OPERATING_THRESHOLD = 0.09
REFERENCE_THRESHOLD = 0.5
REPORT_PATH = PROJECT_ROOT / "reports" / "fairness_evaluation.md"
N_BOOTSTRAP = 1000

# Kết quả CV 5-fold của thành viên B khi thêm SEX (reports/model_comparison.md mục 3 và 6.4), dùng làm
# bằng chứng độc lập với phân tích trên Valid: chênh lệch AUC trung bình và tỷ trọng tầm quan trọng của SEX.
CV_SEX_AUC_DELTA = 0.0001
CV_SEX_SHARE = {"LightGBM tuned + monotonic": 0.0017, "CatBoost tuned": 0.0054}
# Mức chênh AUC và tỷ trọng tầm quan trọng dưới đó coi là "không đáng kể" khi kết luận về SEX.
NEGLIGIBLE_AUC_DELTA = 0.001
NEGLIGIBLE_SHARE = 0.01

MAIN = "LightGBM monotonic + Platt"
MAIN_SEX = "LightGBM monotonic + Platt (có SEX)"
SCORECARD = "Logistic Scorecard (WoE)"
SCORECARD_SEX = "Logistic Scorecard (WoE, có SEX)"


# ---------------------------------------------------------------------------
# Định dạng
# ---------------------------------------------------------------------------

def markdown_table(headers: Sequence[str], rows: Sequence[Sequence], align: Sequence[str] = ()) -> str:
    """Bảng Markdown dựng tay (không phụ thuộc tabulate). align: 'l' hoặc 'r' cho từng cột."""
    align = list(align) or ["l"] * len(headers)
    marks = [":---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def fairness_markdown(tables: Dict[str, pd.DataFrame]) -> str:
    """Bảng công bằng gộp các thuộc tính (khóa: tên thuộc tính) với DIR kèm CI, trạng thái và AUC kèm CI."""
    rows = []
    for attribute, table in tables.items():
        for r in table.itertuples():
            if r.is_reference:
                dir_cell, status = "1.00 (tham chiếu)", "–"
            else:
                dir_cell = f"{r.dir:.2f} [{r.dir_low:.2f}, {r.dir_high:.2f}]"
                status = "Đạt" if r.dir_ok else "Vượt ngưỡng"
            nonflag = "–" if r.is_reference else f"{r.nonflag_dir:.2f}" + ("" if r.nonflag_ok else " *")
            auc = "–" if np.isnan(r.auc) else f"{r.auc:.3f} [{r.auc_low:.3f}, {r.auc_high:.3f}]"
            rows.append([
                attribute, r.group, r.n, f"{r.default_rate:.1%}", f"{r.flag_rate:.1%}",
                dir_cell, status, nonflag, auc,
            ])
    return markdown_table(
        ["Thuộc tính", "Nhóm", "N", "Vỡ nợ thực tế", "Flag rate", "DIR [95% CI]", "Trạng thái", "DIR không bị flag", "AUC [95% CI]"],
        rows, ["l", "l", "r", "r", "r", "r", "l", "r", "r"],
    )


# ---------------------------------------------------------------------------
# Chạy phân tích
# ---------------------------------------------------------------------------

def fit_predictions() -> Dict[str, object]:
    """Huấn luyện trên Train; trả PD trên Valid của bốn mô hình (chính, chính có SEX, Scorecard, Scorecard có SEX)
    cùng thông tin nhóm và các số liệu về vai trò của SEX."""
    from src.calibrate import CalibratedPipeline
    from src.data_split import load_split_data
    from src.features import build_features
    from src.pipelines import build_scorecard_pipeline
    from src.tune import fit_tuned_pipeline, sex_importance_share

    train_df, valid_df, _ = load_split_data()
    X_train, y_train = build_features(train_df, include_sex=False)
    X_valid, y_valid = build_features(valid_df, include_sex=False)
    Xs_train, _ = build_features(train_df, include_sex=True)
    Xs_valid, _ = build_features(valid_df, include_sex=True)

    def final_model(X_tr, X_va, include_sex):
        pipe = fit_tuned_pipeline(FINAL_STUDY, X_tr, y_train, include_sex=include_sex)
        cal = CalibratedPipeline(base_pipeline=pipe, method="platt").fit_calibrator(X_va, y_valid, optimize_thresh=False)
        return pipe, cal

    _, cal = final_model(X_train, X_valid, include_sex=False)
    pipe_sex, cal_sex = final_model(Xs_train, Xs_valid, include_sex=True)
    scorecard = build_scorecard_pipeline(drop_sensitive=True).fit(X_train, y_train)
    scorecard_sex = build_scorecard_pipeline(drop_sensitive=False).fit(Xs_train, y_train)
    woe = scorecard_sex.named_steps["woe_iv"]

    return {
        "y": y_valid.to_numpy(),
        "sex": label_sex(valid_df["SEX"]).to_numpy(),
        "age": assign_age_groups(valid_df["AGE"]).to_numpy(),
        "pd": {
            MAIN: cal.predict_proba(X_valid)[:, 1],
            MAIN_SEX: cal_sex.predict_proba(Xs_valid)[:, 1],
            SCORECARD: scorecard.predict_proba(X_valid)[:, 1],
            SCORECARD_SEX: scorecard_sex.predict_proba(Xs_valid)[:, 1],
        },
        "platt": (cal.platt_a_, cal.platt_b_),
        "sex_iv": float(woe.iv_values_["SEX"]),
        "sex_iv_threshold": float(woe.iv_threshold),
        "sex_selected_by_woe": "SEX" in woe.selected_features_,
        "sex_importance_share": float(sex_importance_share(pipe_sex)),
    }


def sex_gap(y: np.ndarray, sex: np.ndarray, p: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    """(flag rate Nam, flag rate Nữ, Nam / Nữ) tại ngưỡng."""
    flagged = p >= threshold
    men, women = flagged[sex == "Nam"].mean(), flagged[sex == "Nữ"].mean()
    return float(men), float(women), float(men / women) if women > 0 else float("nan")


def _reference_row(table: pd.DataFrame) -> pd.Series:
    return table[table["is_reference"]].iloc[0]


def _lowest_auc_text(age_table: pd.DataFrame) -> str:
    """Mô tả nhóm tuổi có AUC thấp nhất và mức chồng lấn CI với nhóm tham chiếu."""
    t = age_table.set_index("group")
    lowest, ref_age = t["auc"].idxmin(), _reference_row(age_table)["group"]
    text = (
        f"AUC thấp nhất ở nhóm {lowest}: {t.loc[lowest, 'auc']:.3f} (N = {int(t.loc[lowest, 'n'])}, "
        f"CI 95% [{t.loc[lowest, 'auc_low']:.3f}, {t.loc[lowest, 'auc_high']:.3f}])"
    )
    if lowest == ref_age:
        return text + "."
    overlap = t.loc[lowest, "auc_high"] >= t.loc[ref_age, "auc_low"] and t.loc[lowest, "auc_low"] <= t.loc[ref_age, "auc_high"]
    return text + (
        f"; khoảng tin cậy {'chồng lấn' if overlap else 'không chồng lấn'} với nhóm tham chiếu {ref_age} "
        f"({t.loc[ref_age, 'auc']:.3f}, [{t.loc[ref_age, 'auc_low']:.3f}, {t.loc[ref_age, 'auc_high']:.3f}])"
        + (", nên chưa đủ bằng chứng về khác biệt AUC giữa hai nhóm." if overlap else ".")
    )


def build_conclusions(
    data: Dict[str, object],
    tables: Dict[Tuple[str, float], Dict[str, pd.DataFrame]],
    thresholds: Tuple[float, float],
    reference_name: str = SCORECARD,
) -> Sequence[str]:
    """Kết luận sinh từ số liệu; câu định tính chỉ xuất hiện khi số liệu thỏa điều kiện tương ứng."""
    operating, reference = thresholds
    y, sex, pd_by_model = data["y"], data["sex"], data["pd"]
    lo_band, hi_band = DIR_BAND
    out = []

    # 1. Giới tính (mô hình chính, không dùng SEX)
    parts = []
    for th in thresholds:
        t = tables[(MAIN, th)]["Giới tính"]
        ref = _reference_row(t)
        for row in t[~t["is_reference"]].itertuples():
            parts.append(
                f"ngưỡng {th:g}: {row.group} {row.flag_rate:.1%} so với {ref.group} {ref.flag_rate:.1%} "
                f"(DIR {row.group}/{ref.group} = {row.dir:.2f}, CI 95% [{row.dir_low:.2f}, {row.dir_high:.2f}], "
                f"{'trong' if row.dir_ok else 'ngoài'} dải {lo_band:g}–{hi_band:g})"
            )
    sex_op = tables[(MAIN, operating)]["Giới tính"].set_index("group")
    text = (
        "Giới tính (mô hình chính, không dùng SEX), flag rate: " + "; ".join(parts) + ". "
        f"Tỷ lệ vỡ nợ thực tế trên {data.get('split', 'Valid')}: " + ", ".join(f"{g} {v:.1%}" for g, v in sex_op["default_rate"].items()) + "."
    )
    if sex_op["flag_rate"].idxmax() == sex_op["default_rate"].idxmax():
        text += f" Nhóm bị flag nhiều hơn ({sex_op['flag_rate'].idxmax()}) cũng là nhóm có tỷ lệ vỡ nợ thực tế cao hơn."
    out.append(text)

    # 2. Nhóm tuổi
    parts = []
    for th in thresholds:
        t = tables[(MAIN, th)]["Nhóm tuổi"]
        outside = t[~t["is_reference"] & ~t["dir_ok"]]
        parts.append(
            f"ngưỡng {th:g}: "
            + (
                ", ".join(f"{r.group} (DIR = {r.dir:.2f}, CI [{r.dir_low:.2f}, {r.dir_high:.2f}])" for r in outside.itertuples())
                if len(outside) else "mọi nhóm trong dải"
            )
        )
    text = "Nhóm tuổi (mô hình chính), DIR ngoài dải: " + "; ".join(parts) + ". "
    text += _lowest_auc_text(tables[(MAIN, operating)]["Nhóm tuổi"])
    out.append(text)

    # 3. Vai trò của SEX (chỉ khi có bản mô hình có SEX)
    if MAIN_SEX in pd_by_model:
        out.append(_sex_role_text(data, thresholds))

    # 4. Tính bão hòa của DIR theo flag rate ở ngưỡng vận hành
    flag_overall = overall_summary(y, pd_by_model[MAIN], operating)["flag_rate"]
    return _finish_conclusions(out, data, tables, thresholds, reference_name, flag_overall)


def _sex_role_text(data: Dict[str, object], thresholds: Tuple[float, float]) -> str:
    operating = thresholds[0]
    y, sex, pd_by_model = data["y"], data["sex"], data["pd"]
    lo_band, hi_band = DIR_BAND
    auc_main = overall_summary(y, pd_by_model[MAIN], operating)["auc"]
    auc_sex = overall_summary(y, pd_by_model[MAIN_SEX], operating)["auc"]
    delta, share = auc_sex - auc_main, data["sex_importance_share"]
    gaps = {name: [sex_gap(y, sex, pd_by_model[name], th)[2] for th in thresholds] for name in (MAIN, MAIN_SEX)}
    text = (
        f"Vai trò của SEX: thêm SEX vào mô hình chính đổi AUC Valid {delta:+.4f} và SEX chiếm {share:.2%} tầm quan trọng (gain); "
        "tỷ lệ flag Nam/Nữ "
        + ", ".join(
            f"{gaps[MAIN][i]:.2f} (không SEX) so với {gaps[MAIN_SEX][i]:.2f} (có SEX) ở ngưỡng {th:g}"
            for i, th in enumerate(thresholds)
        )
        + "."
    )
    if abs(delta) < NEGLIGIBLE_AUC_DELTA and share < NEGLIGIBLE_SHARE:
        text += " Bỏ SEX không làm giảm hiệu năng đo được, nên chính sách không dùng SEX (Charter mục 8) được giữ."
        if any(not dir_within_band(g) for g in gaps[MAIN]):
            text += " Chênh lệch flag rate giữa Nam và Nữ vẫn còn khi không dùng SEX; loại SEX không triệt tiêu chênh lệch này."
    else:
        text += " Mức chênh AUC hoặc tỷ trọng của SEX vượt ngưỡng coi là không đáng kể, cần nhóm thảo luận lại chính sách SEX."
    return text


def _finish_conclusions(out, data, tables, thresholds, reference_name, flag_overall) -> Sequence[str]:
    operating = thresholds[0]
    lo_band, hi_band = DIR_BAND
    rows = []
    for attribute in ("Giới tính", "Nhóm tuổi"):
        t = tables[(MAIN, operating)][attribute]
        rows += [(r, t) for r in t[~t["is_reference"]].itertuples()]
    outside = [f"{r.group} ({r.nonflag_dir:.2f})" for r, _ in rows if not r.nonflag_ok]
    text = f"Ở ngưỡng vận hành {operating:g}, mô hình chính gắn cờ {flag_overall:.1%} hồ sơ {data.get('split', 'Valid')} nên flag rate của mọi nhóm đều cao. "
    comparable = [item for item in rows if np.isfinite(item[0].nonflag_dir) and item[0].nonflag_dir > 0]
    if comparable and flag_overall > 0.5:
        widest, t = max(comparable, key=lambda item: abs(np.log(item[0].nonflag_dir)))
        ref = _reference_row(t)
        text += (
            f"Cùng một chênh lệch tuyệt đối, DIR theo flag rate lệch gần 1 hơn DIR không bị flag; ví dụ nhóm {widest.group}: "
            f"flag rate {widest.flag_rate:.1%} so với {ref.flag_rate:.1%} của nhóm {ref.group} cho DIR = {widest.dir:.2f}, "
            f"còn tỷ lệ không bị flag {1 - widest.flag_rate:.1%} so với {1 - ref.flag_rate:.1%} cho DIR không bị flag = {widest.nonflag_dir:.2f}. "
        )
    if outside:
        text += (
            f"DIR không bị flag (quy tắc 4/5 áp dụng cho kết quả có lợi) nằm ngoài {lo_band:g}–{hi_band:g} ở: "
            + ", ".join(outside) + ". Kết luận về công bằng ở ngưỡng này phụ thuộc vào định nghĩa DIR, cần nhóm thống nhất."
        )
    else:
        text += f"DIR không bị flag của mọi nhóm nằm trong {lo_band:g}–{hi_band:g}, cùng kết luận với DIR theo flag rate."
    out.append(text)

    # 5. Mô hình đối chiếu
    parts = []
    for th in thresholds:
        outs = []
        for attribute in ("Giới tính", "Nhóm tuổi"):
            t = tables[(reference_name, th)][attribute]
            outs += [
                f"{r.group} (DIR = {r.dir:.2f}, CI [{r.dir_low:.2f}, {r.dir_high:.2f}])"
                for r in t[~t["is_reference"] & ~t["dir_ok"]].itertuples()
            ]
        parts.append(f"ngưỡng {th:g}: " + (", ".join(outs) if outs else "mọi nhóm trong dải"))
    out.append(
        f"{reference_name} (đối chiếu), DIR ngoài dải: " + "; ".join(parts) + ". "
        + _lowest_auc_text(tables[(reference_name, operating)]["Nhóm tuổi"])
    )
    return out


def build_report(data: Dict[str, object], thresholds: Tuple[float, float]) -> str:
    y, sex, age = data["y"], data["sex"], data["age"]
    pd_by_model: Dict[str, np.ndarray] = data["pd"]
    a, b = data["platt"]
    operating, reference = thresholds
    n, n_pos = len(y), int(y.sum())

    # Bảng công bằng cho từng (mô hình, ngưỡng)
    tables: Dict[Tuple[str, float], Dict[str, pd.DataFrame]] = {}
    for name in (MAIN, SCORECARD):
        for th in thresholds:
            tables[(name, th)] = {
                "Giới tính": group_fairness_table(y, pd_by_model[name], sex, th, order=["Nam", "Nữ"], n_boot=N_BOOTSTRAP),
                "Nhóm tuổi": group_fairness_table(y, pd_by_model[name], age, th, order=list(AGE_GROUP_ORDER), n_boot=N_BOOTSTRAP),
            }

    ref_sex, ref_age = select_reference_group(sex), select_reference_group(age)
    flag_op = overall_summary(y, pd_by_model[MAIN], operating)["flag_rate"]
    flag_ref = overall_summary(y, pd_by_model[MAIN], reference)["flag_rate"]
    lo_band, hi_band = DIR_BAND

    # Bảng ảnh hưởng của SEX
    sex_rows = []
    for name in (MAIN, MAIN_SEX, SCORECARD, SCORECARD_SEX):
        cells = [name, "Có" if "có SEX" in name else "Không", f"{overall_summary(y, pd_by_model[name], operating)['auc']:.4f}"]
        for th in thresholds:
            men, women, ratio = sex_gap(y, sex, pd_by_model[name], th)
            cells += [f"{men:.1%} / {women:.1%}", f"{ratio:.2f}"]
        sex_rows.append(cells)
    sex_table = markdown_table(
        ["Mô hình", "SEX trong đầu vào", "AUC Valid",
         f"Flag Nam / Nữ ({operating:g})", f"Nam/Nữ ({operating:g})", f"Flag Nam / Nữ ({reference:g})", f"Nam/Nữ ({reference:g})"],
        sex_rows, ["l", "l", "r", "r", "r", "r", "r"],
    )

    identical = np.allclose(pd_by_model[SCORECARD], pd_by_model[SCORECARD_SEX], rtol=0, atol=1e-12)
    if data["sex_selected_by_woe"]:
        scorecard_text = (
            f"Bản Scorecard có SEX giữ SEX trong đầu vào và bước WoE chọn SEX (IV = {data['sex_iv']:.4f}), "
            "nên khác bản không SEX."
        )
    else:
        scorecard_text = (
            f"Bản Scorecard có SEX giữ SEX trong đầu vào nhưng bước WoE loại SEX vì IV = {data['sex_iv']:.4f} < ngưỡng "
            f"{data['sex_iv_threshold']:g}. PD của hai bản Scorecard {'trùng nhau' if identical else 'gần như trùng nhau'}, "
            "nên số liệu của hai bản giống hệt nhau và không phản ánh tác động của SEX. "
            "Tác động của SEX được đo trên mô hình LightGBM, nơi SEX thực sự là đầu vào."
        )
    cv_text = "; ".join(f"{m} {s:.2%}" for m, s in CV_SEX_SHARE.items())

    conclusions = build_conclusions(data, tables, thresholds)
    conclusion_md = "\n".join(f"{i}. {text}" for i, text in enumerate(conclusions, start=1))

    def block(name: str, th: float) -> str:
        return fairness_markdown(tables[(name, th)])

    return f"""# Báo cáo Đánh giá Tính Công bằng (Fairness Check) trên tập Valid

_Tuần 3 – T3: Thành viên A (Data Engineer / Analyst)_

---

## 1. Phạm vi và thiết lập

| Hạng mục | Nội dung |
| :--- | :--- |
| Tập dữ liệu | Valid, n = {n} ({n_pos} hồ sơ vỡ nợ, {n_pos / n:.2%}). Tập Test không được sử dụng. |
| Mô hình chính | LightGBM tuned + monotonic (`{FINAL_STUDY}`), không dùng SEX, huấn luyện trên Train; Platt scaling fit trên Valid (a = {a:.4f}, b = {b:.4f}). Đây là mô hình cuối theo `week2_review.md`. |
| Mô hình đối chiếu | Logistic Scorecard (WoE), không dùng SEX: mô hình tham chiếu minh bạch (`week2_review.md` mục 6) và là mô hình của bản đánh giá công bằng ban đầu. |
| Ngưỡng vận hành | PD ≥ {operating:g}: ngưỡng do thành viên B khóa trước khi đánh giá Test, làm tròn từ trung bình OOF EAD-weighted (0.088) với tỷ lệ chi phí FN:FP = 9:1 theo Charter mục 7 (`reports/test_evaluation/test_evaluation.md`). Mô hình chính gắn cờ {flag_op:.1%} hồ sơ Valid. |
| Ngưỡng đối chiếu | PD ≥ {reference:g}: ngưỡng phân loại mặc định, mô hình chính gắn cờ {flag_ref:.1%} hồ sơ Valid. Dùng để kiểm tra độ nhạy của kết luận theo ngưỡng, vì ngưỡng vận hành gắn cờ phần lớn danh mục. |
| Nhóm | Giới tính (SEX = 1 Nam, 2 Nữ) và nhóm tuổi <30, 30–50, >50 (30 và 50 thuộc nhóm 30–50). Hai thuộc tính này chỉ dùng để đánh giá; mô hình không dùng SEX, AGE được đưa vào dưới dạng `AGE_BIN` (Charter mục 8). |
| Flag rate | Tỷ lệ hồ sơ bị gắn cờ can thiệp (PD ≥ ngưỡng). |
| DIR | Disparate Impact Ratio = flag rate của nhóm / flag rate của nhóm tham chiếu. Nhóm tham chiếu là nhóm đông nhất của từng thuộc tính: **{ref_sex}** (giới tính) và **{ref_age}** (nhóm tuổi). Cờ là kết quả bất lợi, nên cảnh báo khi DIR ngoài [{lo_band:g}, {hi_band:g}] (ngưỡng 0.8 của Charter mục 8 và nghịch đảo 1 / 0.8 cho chiều ngược lại). CI 95% bằng bootstrap {N_BOOTSTRAP} lần trong từng nhóm. |
| DIR không bị flag | (1 − flag rate của nhóm) / (1 − flag rate của tham chiếu): quy tắc 4/5 áp dụng cho kết quả có lợi (không bị can thiệp). Chỉ số bổ sung, không thuộc Charter; dấu * đánh dấu giá trị ngoài [{lo_band:g}, {hi_band:g}]. |
| AUC | AUC của PD trong từng nhóm, CI 95% bằng bootstrap {N_BOOTSTRAP} lần. |

---

## 2. Tác động của biến SEX

{sex_table}

- {scorecard_text}
- Với mô hình chính, thêm SEX đổi AUC Valid {overall_summary(y, pd_by_model[MAIN_SEX], operating)['auc'] - overall_summary(y, pd_by_model[MAIN], operating)['auc']:+.4f} và SEX chiếm {data['sex_importance_share']:.2%} tầm quan trọng (gain) của mô hình có SEX.
- Kết quả CV 5-fold của thành viên B (`model_comparison.md` mục 6.4) cùng chiều: thêm SEX đổi AUC trung bình {CV_SEX_AUC_DELTA:+.4f}; tỷ trọng tầm quan trọng của SEX là {cv_text}.

---

## 3. Kết quả mô hình chính (không dùng SEX)

### 3.1. Ngưỡng vận hành ({operating:g})

{block(MAIN, operating)}

### 3.2. Ngưỡng đối chiếu ({reference:g})

{block(MAIN, reference)}

---

## 4. Đối chiếu với Logistic Scorecard (không dùng SEX)

### 4.1. Ngưỡng vận hành ({operating:g})

{block(SCORECARD, operating)}

### 4.2. Ngưỡng đối chiếu ({reference:g})

{block(SCORECARD, reference)}

---

## 5. Kết luận

{conclusion_md}

---

## 6. Hạn chế

- Kết quả trên Valid. Kết quả trên Test (dự đoán xuất sau lượt nghiệm thu của thành viên B): `fairness_evaluation_test.md`.
- DIR đo chênh lệch tỷ lệ gắn cờ giữa các nhóm, không hiệu chỉnh theo chênh lệch tỷ lệ vỡ nợ thực tế giữa các nhóm. Recall và tỷ lệ cảnh báo oan theo nhóm (equalized odds) không nằm trong phạm vi báo cáo.
- Nhóm {AGE_GROUP_ORDER[2]} có {int((age == AGE_GROUP_ORDER[2]).sum())} hồ sơ nên CI của DIR và AUC rộng; chênh lệch nằm trong CI chưa đủ bằng chứng là khác biệt thực.
- Calibrator Platt (hai tham số a, b) được fit trên chính tập Valid dùng để đánh giá.
- Mô hình được huấn luyện lại từ `configs/tuned_params.yaml` (`random_state = 42`); chữ số thập phân cuối của PD có thể lệch nhẹ so với số liệu trong `calibration_threshold.md`.

## 7. Tái lập

```bash
python scripts/analyze_fairness.py
```

Yêu cầu `data/raw/UCI_Credit_Card.csv`, `data/splits/splits.json` và `src/calibrate.py`.
"""


TEST_REPORT_PATH = PROJECT_ROOT / "reports" / "fairness_evaluation_test.md"
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "test_evaluation" / "test_predictions.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
BASELINE = "Logistic Regression baseline"


def load_test_predictions(predictions_path: Path = TEST_PREDICTIONS_PATH, raw_path: Path = RAW_DATA_PATH) -> Dict[str, object]:
    """Đọc dự đoán Test đã xuất (scripts/export_test_predictions.py) và ghép SEX, AGE từ dữ liệu gốc theo ID.

    Không chấm điểm lại Test: chỉ đọc file dự đoán. Raises ValueError nếu ID không khớp dữ liệu gốc.
    """
    preds = pd.read_csv(predictions_path)
    raw = pd.read_csv(raw_path, usecols=["ID", "SEX", "AGE"])
    merged = preds.merge(raw, on="ID", how="left", validate="one_to_one")
    if merged[["SEX", "AGE"]].isna().any().any():
        raise ValueError("Có ID trong file dự đoán không khớp dữ liệu gốc.")
    return {
        "split": "Test",
        "y": merged["y_true"].to_numpy(dtype=int),
        "sex": label_sex(merged["SEX"]).to_numpy(),
        "age": assign_age_groups(merged["AGE"]).to_numpy(),
        "pd": {MAIN: merged["pd_final"].to_numpy(), BASELINE: merged["pd_baseline"].to_numpy()},
    }


def build_test_report(data: Dict[str, object], thresholds: Tuple[float, float]) -> str:
    y, sex, age = data["y"], data["sex"], data["age"]
    pd_by_model: Dict[str, np.ndarray] = data["pd"]
    operating, reference = thresholds
    n, n_pos = len(y), int(y.sum())

    tables: Dict[Tuple[str, float], Dict[str, pd.DataFrame]] = {}
    for name in (MAIN, BASELINE):
        for th in thresholds:
            tables[(name, th)] = {
                "Giới tính": group_fairness_table(y, pd_by_model[name], sex, th, order=["Nam", "Nữ"], n_boot=N_BOOTSTRAP),
                "Nhóm tuổi": group_fairness_table(y, pd_by_model[name], age, th, order=list(AGE_GROUP_ORDER), n_boot=N_BOOTSTRAP),
            }
    flag_op = overall_summary(y, pd_by_model[MAIN], operating)["flag_rate"]
    flag_ref = overall_summary(y, pd_by_model[MAIN], reference)["flag_rate"]
    lo_band, hi_band = DIR_BAND
    conclusions = build_conclusions(data, tables, thresholds, reference_name=BASELINE)
    conclusion_md = "\n".join(f"{i}. {text}" for i, text in enumerate(conclusions, start=1))

    return f"""# Báo cáo Đánh giá Tính Công bằng (Fairness Check) trên tập Test

_Tuần 3 – T3: Thành viên A (Data Engineer / Analyst)_

---

## 1. Phạm vi và thiết lập

| Hạng mục | Nội dung |
| :--- | :--- |
| Tập dữ liệu | Test, n = {n} ({n_pos} hồ sơ vỡ nợ, {n_pos / n:.2%}). Dự đoán lấy từ `reports/test_evaluation/test_predictions.csv`, xuất sau lượt đánh giá Test của thành viên B; báo cáo này không chấm điểm lại Test. |
| Mô hình chính | LightGBM tuned + monotonic, Platt fit trên Valid, không dùng SEX. |
| Mô hình đối chiếu | Logistic Regression baseline, mô hình đối chiếu của lượt đánh giá Test (không qua Platt). |
| Ngưỡng | PD ≥ {operating:g} (ngưỡng vận hành đã khóa, mô hình chính gắn cờ {flag_op:.1%} hồ sơ Test) và PD ≥ {reference:g} (đối chiếu, gắn cờ {flag_ref:.1%}). |
| Chỉ số | Flag rate, DIR (so với nhóm đông nhất của từng thuộc tính, dải [{lo_band:g}, {hi_band:g}]), DIR không bị flag (chỉ số bổ sung, dấu * là ngoài dải), AUC theo nhóm; CI 95% bootstrap {N_BOOTSTRAP} lần. Định nghĩa đầy đủ: `fairness_evaluation.md` mục 1. |

---

## 2. Kết quả mô hình chính (không dùng SEX)

### 2.1. Ngưỡng vận hành ({operating:g})

{fairness_markdown(tables[(MAIN, operating)])}

### 2.2. Ngưỡng đối chiếu ({reference:g})

{fairness_markdown(tables[(MAIN, reference)])}

---

## 3. Đối chiếu với Logistic Regression baseline

### 3.1. Ngưỡng vận hành ({operating:g})

{fairness_markdown(tables[(BASELINE, operating)])}

### 3.2. Ngưỡng đối chiếu ({reference:g})

{fairness_markdown(tables[(BASELINE, reference)])}

---

## 4. Kết luận

{conclusion_md}

---

## 5. Hạn chế

- Test chỉ có dự đoán của mô hình chính và baseline; bản có SEX và Scorecard không được chấm trên Test, nên tác động của SEX được đánh giá trên Valid (`fairness_evaluation.md` mục 2).
- Mô hình được huấn luyện lại để xuất dự đoán; flag rate ở ngưỡng 0.09 lệch 5 hồ sơ so với báo cáo nghiệm thu của B (`test_evaluation/test_evaluation.md`).
- DIR đo chênh lệch tỷ lệ gắn cờ, không hiệu chỉnh theo chênh lệch tỷ lệ vỡ nợ thực tế giữa các nhóm; nhóm {AGE_GROUP_ORDER[2]} có {int((age == AGE_GROUP_ORDER[2]).sum())} hồ sơ nên CI rộng.

## 6. Tái lập

```bash
python scripts/analyze_fairness.py --split test
```

Yêu cầu `reports/test_evaluation/test_predictions.csv` (không commit) và `data/raw/UCI_Credit_Card.csv`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá công bằng theo giới tính và nhóm tuổi")
    parser.add_argument("--threshold", type=float, default=OPERATING_THRESHOLD, help="Ngưỡng vận hành (PD)")
    parser.add_argument("--reference-threshold", type=float, default=REFERENCE_THRESHOLD, help="Ngưỡng đối chiếu (PD)")
    parser.add_argument("--split", choices=["valid", "test"], default="valid",
                        help="valid: huấn luyện lại và đánh giá trên Valid; test: đọc dự đoán Test đã xuất, không chấm lại Test")
    parser.add_argument("--predictions", type=Path, default=TEST_PREDICTIONS_PATH, help="(split test) File dự đoán Test")
    parser.add_argument("--output", type=Path, help="Đường dẫn file báo cáo")
    args = parser.parse_args()

    thresholds = (args.threshold, args.reference_threshold)
    if args.split == "test":
        print("=== ĐÁNH GIÁ TÍNH CÔNG BẰNG TRÊN TẬP TEST (từ file dự đoán đã xuất) ===")
        report = build_test_report(load_test_predictions(args.predictions), thresholds)
        args.output = args.output or TEST_REPORT_PATH
    else:
        print("=== ĐÁNH GIÁ TÍNH CÔNG BẰNG TRÊN TẬP VALID ===")
        report = build_report(fit_predictions(), thresholds)
        args.output = args.output or REPORT_PATH
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Đã ghi báo cáo: {args.output}")


if __name__ == "__main__":
    main()
