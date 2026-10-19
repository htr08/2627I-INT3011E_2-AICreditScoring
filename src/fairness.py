"""Chỉ số công bằng theo nhóm (giới tính, nhóm tuổi) cho mô hình cảnh báo rủi ro tín dụng.

Quy ước theo Project Charter mục 7–8 và project_plan.md mục 4.3 (Tuần 3 – T3):
- Flag rate: tỷ lệ khách hàng bị gắn cờ can thiệp, tức PD >= ngưỡng.
- Disparate Impact Ratio (DIR) = flag rate của nhóm / flag rate của nhóm tham chiếu. Cờ là kết quả bất lợi
  nên cảnh báo khi DIR nằm ngoài [0.8, 1.25] (cận trên là nghịch đảo của ngưỡng 0.8 trong Charter).
- Chỉ số bổ sung DIR không bị flag = (1 - flag rate nhóm) / (1 - flag rate tham chiếu): quy tắc 4/5 áp dụng cho
  kết quả có lợi. Hữu ích khi ngưỡng gắn cờ phần lớn danh mục: flag rate của mọi nhóm đều cao nên chênh lệch tương đối
  của DIR theo flag rate bị nén.
- AUC theo nhóm, kèm khoảng tin cậy bootstrap 95%.

Module chỉ chứa phần tính toán. Việc huấn luyện mô hình và ghi báo cáo nằm ở scripts/analyze_fairness.py.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.evaluate import bootstrap_metric_ci, compute_roc_auc

AGE_GROUP_ORDER: Tuple[str, ...] = ("<30", "30–50", ">50")
SEX_LABELS: Dict[int, str] = {1: "Nam", 2: "Nữ"}
# Dải DIR chấp nhận được: ngưỡng 0.8 của Charter và nghịch đảo 1 / 0.8 cho chiều ngược lại.
DIR_BAND: Tuple[float, float] = (0.8, 1.25)


def assign_age_groups(ages: Sequence[float]) -> pd.Series:
    """Nhóm tuổi theo plan: <30, 30–50 (gồm cả 30 và 50), >50."""
    ages = pd.Series(ages)
    labels = np.where(ages < 30, AGE_GROUP_ORDER[0], np.where(ages <= 50, AGE_GROUP_ORDER[1], AGE_GROUP_ORDER[2]))
    return pd.Series(labels, index=ages.index)


def label_sex(sex: Sequence[int]) -> pd.Series:
    """Mã SEX của CSV (1 = Nam, 2 = Nữ) -> nhãn nhóm; mã khác gây ValueError."""
    sex = pd.Series(sex)
    labels = sex.map(SEX_LABELS)
    if labels.isna().any():
        raise ValueError(f"Mã SEX ngoài {sorted(SEX_LABELS)}: {sorted(sex[labels.isna()].unique())}")
    return labels


def select_reference_group(groups: Sequence[str]) -> str:
    """Nhóm tham chiếu mặc định: nhóm đông nhất (hòa thì lấy nhóm đứng đầu theo thứ tự chữ cái)."""
    counts = pd.Series(list(groups)).value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def dir_within_band(ratio: float, band: Tuple[float, float] = DIR_BAND) -> bool:
    """True nếu DIR nằm trong dải chấp nhận; so sánh sau khi làm tròn 2 chữ số để khớp giá trị hiển thị
    trong báo cáo (NaN không đạt)."""
    if np.isnan(ratio):
        return False
    return bool(band[0] <= round(ratio, 2) <= band[1])


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _bootstrap_means(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Trung bình của n_boot mẫu bootstrap (rút có hoàn lại trong nhóm)."""
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return values[idx].mean(axis=1)


def group_fairness_table(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    groups: Sequence[str],
    threshold: float,
    reference: Optional[str] = None,
    order: Optional[Sequence[str]] = None,
    n_boot: int = 1000,
    seed: int = 42,
    band: Tuple[float, float] = DIR_BAND,
) -> pd.DataFrame:
    """Bảng chỉ số công bằng theo nhóm tại một ngưỡng PD.

    Mỗi dòng là một nhóm với các cột:
        group, n, default_rate, flag_rate,
        dir, dir_low, dir_high, dir_ok  : DIR theo flag rate so với nhóm tham chiếu, CI bootstrap 95%, trong dải `band`
                                          (so sánh sau khi làm tròn 2 chữ số)
        nonflag_dir, nonflag_ok         : DIR của tỷ lệ không bị flag
        auc, auc_low, auc_high          : AUC của nhóm và CI bootstrap 95% (NaN nếu nhóm chỉ có một nhãn)
        is_reference
    CI của DIR lấy mẫu bootstrap độc lập trong từng nhóm (kích thước nhóm là cố định).

    Raises:
        ValueError: độ dài đầu vào không khớp, hoặc `reference` không phải một nhóm.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    groups = np.asarray(groups)
    if not (len(y_true) == len(y_prob) == len(groups)):
        raise ValueError("y_true, y_prob và groups phải có cùng độ dài.")

    present = list(pd.unique(groups))
    names = [g for g in (order or present) if g in present] + [g for g in present if order and g not in order]
    reference = select_reference_group(groups) if reference is None else reference
    if reference not in present:
        raise ValueError(f"Nhóm tham chiếu {reference!r} không có trong dữ liệu: {sorted(map(str, present))}")

    flagged = (y_prob >= threshold).astype(float)
    rng = np.random.default_rng(seed)
    boot_flag = {g: _bootstrap_means(flagged[groups == g], n_boot, rng) for g in names}
    ref_rate, ref_boot = flagged[groups == reference].mean(), boot_flag[reference]

    rows = []
    for g in names:
        mask = groups == g
        rate = flagged[mask].mean()
        ratio = _ratio(rate, ref_rate)
        nonflag_ratio = _ratio(1 - rate, 1 - ref_rate)
        if g == reference:
            low = high = 1.0
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = boot_flag[g] / ref_boot
            ratios = ratios[np.isfinite(ratios)]
            low, high = (np.percentile(ratios, [2.5, 97.5]) if len(ratios) else (np.nan, np.nan))

        if len(np.unique(y_true[mask])) == 2:
            auc, auc_low, auc_high = bootstrap_metric_ci(
                y_true[mask], y_prob[mask], metric_fn=compute_roc_auc, n_bootstraps=n_boot, random_state=seed
            )
        else:
            auc = auc_low = auc_high = float("nan")

        rows.append({
            "group": g,
            "n": int(mask.sum()),
            "default_rate": float(y_true[mask].mean()),
            "flag_rate": float(rate),
            "dir": ratio,
            "dir_low": float(low),
            "dir_high": float(high),
            "dir_ok": dir_within_band(ratio, band),
            "nonflag_dir": nonflag_ratio,
            "nonflag_ok": dir_within_band(nonflag_ratio, band),
            "auc": float(auc),
            "auc_low": float(auc_low),
            "auc_high": float(auc_high),
            "is_reference": g == reference,
        })
    return pd.DataFrame(rows)


def overall_summary(y_true: Sequence[int], y_prob: Sequence[float], threshold: float) -> Dict[str, float]:
    """AUC và flag rate toàn tập tại ngưỡng PD."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    return {"auc": compute_roc_auc(y_true, y_prob), "flag_rate": float((y_prob >= threshold).mean())}
