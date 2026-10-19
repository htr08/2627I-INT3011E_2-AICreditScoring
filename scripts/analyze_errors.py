"""Phân tích lỗi (FN/FP) của mô hình cuối trên tập Valid và ghi báo cáo reports/error_analysis_validation.md.

Task: Tuần 3 – T2 (Thành viên A). Tập Test không được sử dụng.

- Mô hình chính: LightGBM tuned + monotonic, Platt scaling fit trên Valid (cùng quy trình với
  `python scripts/run_pipeline.py --mode calibrate`, yêu cầu src/calibrate.py).
- Mô hình đối chiếu: Logistic Scorecard (WoE).
- Ngưỡng: ngưỡng vận hành do thành viên B khóa trước khi đánh giá Test (mặc định 0.09) và ngưỡng đối chiếu 0.5.

Usage:
    python scripts/analyze_errors.py
    python scripts/analyze_errors.py --threshold 0.09 --reference-threshold 0.5
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

from src.error_analysis import (  # noqa: E402
    GROUPS,
    PROFILE_FEATURES,
    assign_error_groups,
    build_profile_frame,
    group_counts,
    group_overlap,
    largest_gaps,
    nearest_group_counts,
    profile_table,
)

FINAL_STUDY = "lightgbm_monotone"
# Ngưỡng PD do thành viên B khóa trước khi chạy Test: làm tròn từ trung bình OOF EAD-weighted (0.088),
# tỷ lệ chi phí FN:FP = 9:1 (reports/test_evaluation/test_evaluation.md).
OPERATING_THRESHOLD = 0.09
REFERENCE_THRESHOLD = 0.5
REPORT_PATH = PROJECT_ROOT / "reports" / "error_analysis_validation.md"
# Nhóm lỗi "nghiêng" về một nhóm khi nhóm đó gần hơn ở ít nhất 75% số đặc trưng.
LEAN_SHARE = 0.75
# Hai mô hình được coi là trùng lặp lỗi nhiều khi tỷ lệ FN chung đạt mức này.
HIGH_OVERLAP = 0.7

FEATURE_NOTES = {
    "LIMIT_BAL": "Hạn mức tín dụng (NT$)",
    "AGE": "Tuổi",
    "PAY_1": "Trạng thái trả nợ tháng gần nhất (mã -2 … 8)",
    "PAY_1_LATE": "Tỷ lệ hồ sơ trễ hạn ở tháng gần nhất (PAY_1 ≥ 1)",
    "PAY_MEAN": "Trung bình trạng thái trả nợ 6 tháng",
    "PAY_LATE_2PLUS_COUNT": "Số tháng trễ từ 2 kỳ trở lên trong 6 tháng",
    "UTIL_MEAN": "Tỷ lệ sử dụng hạn mức trung bình 6 tháng (dư nợ / hạn mức)",
    "PAY_RATIO_MEAN": "Tỷ lệ trả nợ / dư nợ trung bình (bỏ qua tháng dư nợ ≤ 0)",
}


# ---------------------------------------------------------------------------
# Định dạng
# ---------------------------------------------------------------------------

def fmt_value(feature: str, value: float) -> str:
    """Định dạng số theo bản chất của đặc trưng."""
    if value is None or np.isnan(value):
        return "–"
    if feature == "LIMIT_BAL":
        return f"{value:,.0f}"
    if feature == "PAY_1_LATE":
        return f"{value:.1%}"
    if feature == "PD":
        return f"{value:.3f}"
    return f"{value:.2f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence], align: Sequence[str] = ()) -> str:
    """Bảng Markdown dựng tay (không phụ thuộc tabulate). align: 'l' hoặc 'r' cho từng cột."""
    align = list(align) or ["l"] * len(headers)
    marks = [":---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Nhận xét sinh từ số liệu
# ---------------------------------------------------------------------------

def lean(n_first: int, n_second: int, n_features: int) -> str:
    """'first' / 'second' nếu nhóm đó gần hơn ở >= LEAN_SHARE số đặc trưng, ngược lại 'mixed'."""
    if n_first >= LEAN_SHARE * n_features:
        return "first"
    if n_second >= LEAN_SHARE * n_features:
        return "second"
    return "mixed"


def _verdict(side: str, first: str, second: str) -> str:
    if side == "first":
        return f"Hồ sơ lỗi giống nhóm {first} hơn nhóm {second} trên hầu hết đặc trưng."
    if side == "second":
        return f"Hồ sơ lỗi giống nhóm {second} hơn nhóm {first} trên hầu hết đặc trưng."
    return f"Hồ sơ lỗi nằm giữa hai nhóm {first} và {second}, không nghiêng rõ về nhóm nào."


def _gap_text(gaps: List[Tuple[str, float, float]], pair: Tuple[str, str]) -> str:
    items = "; ".join(f"`{f}` {fmt_value(f, a)} so với {fmt_value(f, b)}" for f, a, b in gaps)
    return f"{items} ({pair[0]} so với {pair[1]})"


def error_findings(table: pd.DataFrame, counts: pd.Series, spread: pd.Series) -> List[str]:
    """Nhận xét FN (so với TP, TN) và FP (so với TN, TP); mọi con số lấy từ `table` và `counts`.

    Nhóm gần nhất của từng đặc trưng được xác định theo khoảng cách tuyệt đối giữa các trung bình.
    """
    n_features = len([f for f in table.index if f != "PD"])
    n_pos, n_neg = counts["FN"] + counts["TP"], counts["FP"] + counts["TN"]
    lines: List[str] = []

    if counts["FN"] and counts["TP"]:
        near = nearest_group_counts(table, "FN", ["TN", "TP"])
        gaps = largest_gaps(table, "FN", "TP", spread)
        verdict = _verdict(lean(near["TN"], near["TP"], n_features), "TN", "TP")
        lines.append(
            f"**FN** (n = {counts['FN']}, {counts['FN'] / n_pos:.1%} số hồ sơ vỡ nợ). So với TP (cùng vỡ nợ thực tế, "
            f"được mô hình phát hiện), chênh lệch lớn nhất ở {_gap_text(gaps, ('FN', 'TP'))}. "
            f"FN gần TN hơn TP ở {near['TN']}/{n_features} đặc trưng, gần TP hơn TN ở {near['TP']}/{n_features}. {verdict}"
        )
    else:
        lines.append(f"**FN**: {counts['FN']} hồ sơ, không đủ dữ liệu so sánh với TP.")

    if counts["FP"] and counts["TN"]:
        near = nearest_group_counts(table, "FP", ["TP", "TN"])
        gaps = largest_gaps(table, "FP", "TN", spread)
        verdict = _verdict(lean(near["TP"], near["TN"], n_features), "TP", "TN")
        lines.append(
            f"**FP** (n = {counts['FP']}, {counts['FP'] / n_neg:.1%} số hồ sơ không vỡ nợ). So với TN (cùng không vỡ nợ, "
            f"không bị gắn cờ), chênh lệch lớn nhất ở {_gap_text(gaps, ('FP', 'TN'))}. "
            f"FP gần TP hơn TN ở {near['TP']}/{n_features} đặc trưng, gần TN hơn TP ở {near['TN']}/{n_features}. {verdict}"
        )
    else:
        lines.append(f"**FP**: {counts['FP']} hồ sơ, không đủ dữ liệu so sánh với TN.")
    return lines


def build_conclusions(results: Dict[tuple, dict], names: Tuple[str, str], thresholds: Tuple[float, float], n: int, n_pos: int) -> List[str]:
    """Kết luận tổng hợp; câu định tính chỉ xuất hiện khi số liệu thỏa điều kiện tương ứng."""
    main, ref = names
    operating, reference = thresholds
    n_features = len(PROFILE_FEATURES)
    c_op = results[(main, operating)]["counts"]

    def near(model, th, target, candidates):
        return nearest_group_counts(results[(model, th)]["table"], target, candidates)

    out = []

    fn_near = near(main, operating, "FN", ["TN", "TP"])
    side = lean(fn_near["TN"], fn_near["TP"], n_features)
    text = (
        f"Ở ngưỡng vận hành {operating:g}, mô hình chính bỏ sót {c_op['FN']} hồ sơ vỡ nợ ({c_op['FN'] / n_pos:.1%} số hồ sơ vỡ nợ) "
        f"và gắn cờ oan {c_op['FP']} hồ sơ ({c_op['FP'] / (n - n_pos):.1%} số hồ sơ không vỡ nợ). "
        f"FN gần TN hơn TP ở {fn_near['TN']}/{n_features} đặc trưng"
    )
    if side == "first":
        text += ": hồ sơ vỡ nợ bị bỏ sót có đặc điểm giống khách hàng không vỡ nợ trên hầu hết đặc trưng, nên các đặc trưng này không tách được chúng khỏi nhóm TN."
    elif side == "second":
        text += ": hồ sơ vỡ nợ bị bỏ sót giống nhóm vỡ nợ được phát hiện hơn, nên PD của chúng có thể nằm sát ngưỡng."
    else:
        text += ", không nghiêng rõ về một nhóm."
    out.append(text)

    fp_ref = near(main, reference, "FP", ["TP", "TN"])
    fp_op = near(main, operating, "FP", ["TP", "TN"])
    side = lean(fp_ref["TP"], fp_ref["TN"], n_features)
    text = (
        f"Ở ngưỡng đối chiếu {reference:g}, FP của mô hình chính gần TP hơn TN ở {fp_ref['TP']}/{n_features} đặc trưng "
        f"(ngưỡng {operating:g}: {fp_op['TP']}/{n_features})"
    )
    if side == "first":
        text += ": khách hàng không vỡ nợ bị gắn cờ ở PD cao có đặc điểm giống nhóm vỡ nợ, nên các đặc trưng này không tách được họ khỏi nhóm TP."
    else:
        text += "."
    out.append(text)

    sc_fn = near(ref, operating, "FN", ["TN", "TP"])
    sc_fp = near(ref, reference, "FP", ["TP", "TN"])
    overlaps = {}
    for th in thresholds:
        n_a, _, both = group_overlap(results[(main, th)]["groups"], results[(ref, th)]["groups"], "FN")
        overlaps[th] = (both, n_a)
    text = (
        f"Với Scorecard, FN gần TN hơn TP ở {sc_fn['TN']}/{n_features} đặc trưng (ngưỡng {operating:g}) và FP gần TP hơn TN ở "
        f"{sc_fp['TP']}/{n_features} (ngưỡng {reference:g}). Số FN của mô hình chính cũng là FN của Scorecard: "
        + ", ".join(
            f"{both}/{n_a} ({both / n_a:.1%}) ở ngưỡng {th:g}"
            for th, (both, n_a) in overlaps.items() if n_a
        )
        + "."
    )
    if all(n_a and both / n_a >= HIGH_OVERLAP for both, n_a in overlaps.values()):
        text += " Phần lớn FN trùng nhau giữa hai mô hình, nên đặc điểm lỗi chủ yếu đến từ thông tin trong dữ liệu hơn là loại mô hình."
    out.append(text)
    return out


# ---------------------------------------------------------------------------
# Chạy phân tích
# ---------------------------------------------------------------------------

def fit_predictions() -> Dict[str, object]:
    """Huấn luyện trên Train, trả PD trên Valid của mô hình chính (đã Platt) và Scorecard, cùng dữ liệu Valid."""
    from src.calibrate import CalibratedPipeline
    from src.data_split import load_split_data
    from src.features import build_features
    from src.pipelines import build_scorecard_pipeline
    from src.tune import fit_tuned_pipeline

    train_df, valid_df, _ = load_split_data()
    X_train, y_train = build_features(train_df)
    X_valid, y_valid = build_features(valid_df)

    final = CalibratedPipeline(base_pipeline=fit_tuned_pipeline(FINAL_STUDY, X_train, y_train), method="platt")
    final.fit_calibrator(X_valid, y_valid, optimize_thresh=False)

    scorecard = build_scorecard_pipeline().fit(X_train, y_train)
    return {
        "X_valid": X_valid,
        "y_valid": y_valid.to_numpy(),
        "pd": {
            "LightGBM monotonic + Platt": final.predict_proba(X_valid)[:, 1],
            "Logistic Scorecard (WoE)": scorecard.predict_proba(X_valid)[:, 1],
        },
        "platt": (final.platt_a_, final.platt_b_),
    }


def build_report(data: Dict[str, object], thresholds: Tuple[float, float]) -> str:
    X_valid, y = data["X_valid"], data["y_valid"]
    pd_by_model: Dict[str, np.ndarray] = data["pd"]
    a, b = data["platt"]
    operating, reference = thresholds
    main_name, ref_name = list(pd_by_model)

    profile = build_profile_frame(X_valid)
    spread = profile.std()
    n, n_pos = len(y), int(y.sum())

    # Nhóm lỗi, bảng hồ sơ và nhận xét cho từng (mô hình, ngưỡng)
    results: Dict[tuple, dict] = {}
    for name, p in pd_by_model.items():
        for th in thresholds:
            groups = assign_error_groups(y, p, th)
            table = profile_table(profile, groups, p)
            counts = group_counts(groups)
            results[(name, th)] = {
                "groups": groups, "table": table, "counts": counts,
                "findings": error_findings(table, counts, spread),
            }

    # Mục 2: phân bố nhóm
    dist_rows = []
    for (name, th), r in results.items():
        c = r["counts"]
        tp, fp, fn = c["TP"], c["FP"], c["FN"]
        dist_rows.append([
            name, f"{th:g}", c["FN"], c["FP"], c["TP"], c["TN"],
            f"{tp / (tp + fn):.1%}" if tp + fn else "–",
            f"{tp / (tp + fp):.1%}" if tp + fp else "–",
            f"{(tp + fp) / n:.1%}",
        ])
    dist_table = markdown_table(
        ["Mô hình", "Ngưỡng PD", "FN", "FP", "TP", "TN", "Recall", "Precision", "Flag rate"],
        dist_rows, ["l", "r", "r", "r", "r", "r", "r", "r", "r"],
    )
    flag_op = (results[(main_name, operating)]["counts"][["TP", "FP"]].sum()) / n
    flag_ref = (results[(main_name, reference)]["counts"][["TP", "FP"]].sum()) / n

    def profile_block(name: str, th: float) -> str:
        r = results[(name, th)]
        rows = [
            [f"`{f}`"] + [fmt_value(f, r["table"].loc[f, g]) for g in (*GROUPS, "ALL")]
            for f in (*PROFILE_FEATURES, "PD")
        ]
        md = markdown_table(["Đặc trưng", *GROUPS, "Toàn Valid"], rows, ["l"] + ["r"] * 5)
        bullets = "\n".join(f"- {line}" for line in r["findings"])
        return f"{md}\n\n{bullets}"

    # Mục 4.3: mức trùng lặp lỗi giữa hai mô hình
    overlap_rows = []
    for th in thresholds:
        for label in ("FN", "FP"):
            n_a, n_b, both = group_overlap(results[(main_name, th)]["groups"], results[(ref_name, th)]["groups"], label)
            overlap_rows.append([f"{th:g}", label, n_a, n_b, both, f"{both / n_a:.1%}" if n_a else "–"])
    overlap_table = markdown_table(
        ["Ngưỡng PD", "Nhóm", "Mô hình chính", "Scorecard", "Thuộc nhóm ở cả hai mô hình", "Tỷ lệ so với mô hình chính"],
        overlap_rows, ["r", "l", "r", "r", "r", "r"],
    )

    conclusions = build_conclusions(results, (main_name, ref_name), thresholds, n, n_pos)
    conclusion_md = "\n".join(f"{i}. {text}" for i, text in enumerate(conclusions, start=1))
    notes = "\n".join(f"| `{f}` | {desc} |" for f, desc in FEATURE_NOTES.items())

    return f"""# Báo cáo Phân tích Lỗi (Error Analysis) trên tập Valid

_Tuần 3 – T2: Thành viên A (Data Engineer / Analyst)_

---

## 1. Phạm vi và thiết lập

| Hạng mục | Nội dung |
| :--- | :--- |
| Tập dữ liệu | Valid, n = {n} ({n_pos} hồ sơ vỡ nợ, {n_pos / n:.2%}). Tập Test không được sử dụng. |
| Mô hình chính | LightGBM tuned + monotonic (`{FINAL_STUDY}`, tham số trong `configs/tuned_params.yaml`), huấn luyện trên Train, Platt scaling fit trên Valid (a = {a:.4f}, b = {b:.4f}). Đây là mô hình cuối theo `week2_review.md`. |
| Mô hình đối chiếu | Logistic Scorecard (WoE): mô hình tham chiếu minh bạch (`week2_review.md` mục 6) và là mô hình của bản phân tích lỗi ban đầu. Dùng để kiểm tra mẫu lỗi có phụ thuộc loại mô hình hay không. |
| Ngưỡng vận hành | PD ≥ {operating:g}: ngưỡng do thành viên B khóa trước khi đánh giá Test, làm tròn từ trung bình OOF EAD-weighted (0.088) với tỷ lệ chi phí FN:FP = 9:1 theo Charter mục 7 (`reports/test_evaluation/test_evaluation.md`). Gắn cờ {flag_op:.1%} hồ sơ Valid ở mô hình chính. |
| Ngưỡng đối chiếu | PD ≥ {reference:g}: ngưỡng phân loại mặc định, gắn cờ {flag_ref:.1%} hồ sơ Valid ở mô hình chính. Do ngưỡng vận hành gắn cờ phần lớn danh mục nên FP ở đó gồm nhiều hồ sơ rủi ro thấp; ngưỡng {reference:g} chỉ giữ các ca PD cao để quan sát lỗi ở vùng tự tin. |
| Định nghĩa nhóm | Theo Charter mục 7 (công cụ cảnh báo sớm cho danh mục hiện hữu). **FN**: khách hàng vỡ nợ thực tế nhưng không bị gắn cờ (bỏ sót, không kịp can thiệp). **FP**: khách hàng không vỡ nợ nhưng bị gắn cờ (cảnh báo hoặc giảm hạn mức oan). **TP**, **TN**: gắn cờ đúng khách vỡ nợ, không gắn cờ đúng khách không vỡ nợ. |
| Hồ sơ mô tả | Cột gốc của Valid sau làm sạch mã bất thường, cùng các đặc trưng dẫn xuất T2 của `src/features.py` (tính trực tiếp trên Valid, không học tham số từ dữ liệu). Giá trị là trung bình theo nhóm; `PD` là PD trung bình của mô hình tương ứng. |

Đặc trưng mô tả hồ sơ:

| Đặc trưng | Ý nghĩa |
| :--- | :--- |
{notes}

---

## 2. Phân bố nhóm phân loại

{dist_table}

Recall = TP / (TP + FN); Precision = TP / (TP + FP); Flag rate = (TP + FP) / n.

---

## 3. Hồ sơ trung bình theo nhóm: mô hình chính

### 3.1. Ngưỡng vận hành ({operating:g})

{profile_block(main_name, operating)}

### 3.2. Ngưỡng đối chiếu ({reference:g})

{profile_block(main_name, reference)}

---

## 4. Đối chiếu với Logistic Scorecard

### 4.1. Ngưỡng vận hành ({operating:g})

{profile_block(ref_name, operating)}

### 4.2. Ngưỡng đối chiếu ({reference:g})

{profile_block(ref_name, reference)}

### 4.3. Mức trùng lặp lỗi giữa hai mô hình

{overlap_table}

---

## 5. Kết luận

{conclusion_md}

---

## 6. Hạn chế

- Calibrator Platt (hai tham số a, b) được fit trên chính tập Valid dùng để phân tích, nên số liệu không phải đánh giá ngoài mẫu hoàn toàn. Kết quả chưa cập nhật theo tập Test.
- Mô tả theo giá trị trung bình; `LIMIT_BAL`, `UTIL_MEAN`, `PAY_RATIO_MEAN` có phân phối lệch phải nên trung bình chịu ảnh hưởng của đuôi. Khoảng cách giữa các nhóm cho biết xu hướng, không kèm kiểm định thống kê.
- "Nhóm gần nhất" xét riêng từng đặc trưng, không phản ánh khoảng cách đa chiều giữa các hồ sơ. Các đặc trưng về trễ hạn (`PAY_1`, `PAY_1_LATE`, `PAY_MEAN`, `PAY_LATE_2PLUS_COUNT`) tương quan cao nên số đặc trưng gần một nhóm không độc lập với nhau.
- Mô hình được huấn luyện lại từ `configs/tuned_params.yaml` (`random_state = 42`); chữ số thập phân cuối của PD có thể lệch nhẹ so với số liệu trong `calibration_threshold.md`.

## 7. Tái lập

```bash
python scripts/analyze_errors.py
```

Yêu cầu `data/raw/UCI_Credit_Card.csv`, `data/splits/splits.json` và `src/calibrate.py`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Phân tích lỗi FN/FP trên tập Valid")
    parser.add_argument("--threshold", type=float, default=OPERATING_THRESHOLD, help="Ngưỡng vận hành (PD)")
    parser.add_argument("--reference-threshold", type=float, default=REFERENCE_THRESHOLD, help="Ngưỡng đối chiếu (PD)")
    parser.add_argument("--output", type=Path, default=REPORT_PATH, help="Đường dẫn file báo cáo")
    args = parser.parse_args()

    print("=== PHÂN TÍCH LỖI TRÊN TẬP VALID ===")
    data = fit_predictions()
    report = build_report(data, (args.threshold, args.reference_threshold))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Đã ghi báo cáo: {args.output}")


if __name__ == "__main__":
    main()
