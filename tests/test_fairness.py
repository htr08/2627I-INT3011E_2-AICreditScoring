"""Unit tests cho src/fairness.py và scripts/analyze_fairness.py (đánh giá công bằng theo giới tính, nhóm tuổi)."""

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_fairness
from scripts.analyze_fairness import (
    MAIN,
    MAIN_SEX,
    SCORECARD,
    SCORECARD_SEX,
    BASELINE,
    build_report,
    build_test_report,
    load_test_predictions,
    fairness_markdown,
    markdown_table,
    sex_gap,
)
from src.fairness import (
    AGE_GROUP_ORDER,
    assign_age_groups,
    dir_within_band,
    group_fairness_table,
    label_sex,
    overall_summary,
    select_reference_group,
)


def make_groups_frame():
    """Nhóm A: 100 hồ sơ, 20% bị flag; nhóm B: 100 hồ sơ, 40% bị flag. Mỗi nhóm có đủ hai nhãn."""
    groups = np.array(["A"] * 100 + ["B"] * 100)
    y = np.tile([0, 1], 100)
    p = np.concatenate([
        np.where(np.arange(100) < 20, 0.9, 0.1),
        np.where(np.arange(100) < 40, 0.9, 0.1),
    ])
    return y, p, groups


# ---------------------------------------------------------------------------
# Nhóm và dải DIR
# ---------------------------------------------------------------------------

def test_assign_age_groups_boundaries():
    groups = assign_age_groups([21, 29, 30, 50, 51, 79])
    assert groups.tolist() == ["<30", "<30", "30–50", "30–50", ">50", ">50"]
    assert set(groups) <= set(AGE_GROUP_ORDER)


def test_label_sex_maps_codes_and_rejects_unknown():
    assert label_sex([1, 2, 2]).tolist() == ["Nam", "Nữ", "Nữ"]
    with pytest.raises(ValueError, match="Mã SEX"):
        label_sex([1, 3])


def test_select_reference_group_prefers_largest_then_alphabetical():
    assert select_reference_group(["a", "b", "b"]) == "b"
    assert select_reference_group(["b", "a"]) == "a"


@pytest.mark.parametrize(
    "ratio, expected",
    [(0.75, False), (0.8, True), (0.799, True), (0.79, False), (1.0, True), (1.25, True), (1.254, True), (1.26, False), (np.nan, False)],
)
def test_dir_within_band_compares_rounded_values(ratio, expected):
    # So sánh sau khi làm tròn 2 chữ số để khớp giá trị hiển thị (0.799 hiển thị là 0.80)
    assert dir_within_band(ratio) is expected


# ---------------------------------------------------------------------------
# Bảng công bằng theo nhóm
# ---------------------------------------------------------------------------

def test_group_fairness_table_uses_group_vs_reference_ratio():
    y, p, groups = make_groups_frame()
    table = group_fairness_table(y, p, groups, threshold=0.5, reference="A", n_boot=50).set_index("group")

    assert table.loc["A", ["n", "flag_rate", "default_rate"]].tolist() == [100, 0.2, 0.5]
    assert table.loc["B", "flag_rate"] == pytest.approx(0.4)
    assert table.loc["A", "dir"] == pytest.approx(1.0) and table.loc["A", "is_reference"]
    # Flag rate nhóm B gấp đôi nhóm tham chiếu, không phải tỷ lệ so với flag rate toàn tập
    assert table.loc["B", "dir"] == pytest.approx(2.0)
    assert not table.loc["B", "dir_ok"]
    assert table.loc["B", "dir_low"] <= 2.0 <= table.loc["B", "dir_high"]
    # DIR không bị flag: (1 - 0.4) / (1 - 0.2) = 0.75
    assert table.loc["B", "nonflag_dir"] == pytest.approx(0.75)
    assert not table.loc["B", "nonflag_ok"]


def test_group_fairness_table_reference_change_inverts_ratio():
    y, p, groups = make_groups_frame()
    table = group_fairness_table(y, p, groups, threshold=0.5, reference="B", n_boot=50).set_index("group")
    assert table.loc["A", "dir"] == pytest.approx(0.5)
    assert table.loc["B", "is_reference"]


def test_group_fairness_table_defaults_to_largest_group_and_keeps_order():
    y, p, groups = make_groups_frame()
    groups = np.where(np.arange(200) < 150, "A", "B")  # A đông hơn
    table = group_fairness_table(y, p, groups, threshold=0.5, order=["B", "A"], n_boot=20)
    assert table["group"].tolist() == ["B", "A"]
    assert table.set_index("group").loc["A", "is_reference"]


def test_group_fairness_table_auc_nan_for_single_label_group():
    y = np.array([0, 1, 0, 1, 0, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.4])
    groups = np.array(["A", "A", "A", "A", "B", "B"])
    table = group_fairness_table(y, p, groups, threshold=0.5, n_boot=20).set_index("group")
    assert table.loc["A", "auc"] == pytest.approx(1.0)
    assert np.isnan(table.loc["B", "auc"])


def test_group_fairness_table_is_deterministic_for_same_seed():
    y, p, groups = make_groups_frame()
    first = group_fairness_table(y, p, groups, threshold=0.5, n_boot=30, seed=7)
    second = group_fairness_table(y, p, groups, threshold=0.5, n_boot=30, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_group_fairness_table_validates_inputs():
    y, p, groups = make_groups_frame()
    with pytest.raises(ValueError, match="cùng độ dài"):
        group_fairness_table(y[:-1], p, groups, threshold=0.5)
    with pytest.raises(ValueError, match="Nhóm tham chiếu"):
        group_fairness_table(y, p, groups, threshold=0.5, reference="Z")


def test_overall_summary():
    summary = overall_summary([0, 1, 0, 1], [0.1, 0.9, 0.6, 0.4], threshold=0.5)
    assert summary["flag_rate"] == 0.5
    assert summary["auc"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Script: định dạng và báo cáo
# ---------------------------------------------------------------------------

def test_markdown_table_alignment_row():
    md = markdown_table(["a", "b"], [[1, 2]], ["l", "r"])
    assert md.splitlines() == ["| a | b |", "| :--- | ---: |", "| 1 | 2 |"]


def test_fairness_markdown_marks_reference_status_and_nonflag_warning():
    y, p, groups = make_groups_frame()
    table = group_fairness_table(y, p, groups, threshold=0.5, reference="A", n_boot=20)
    md = fairness_markdown({"Thuộc tính X": table})
    rows = md.splitlines()[2:]
    assert "1.00 (tham chiếu)" in rows[0] and "| – |" in rows[0]
    assert "Vượt ngưỡng" in rows[1]
    assert "0.75 *" in rows[1]  # DIR không bị flag ngoài dải được đánh dấu


def test_sex_gap_returns_rates_and_ratio():
    sex = np.array(["Nam", "Nam", "Nữ", "Nữ"])
    men, women, ratio = sex_gap(np.zeros(4), sex, np.array([0.9, 0.9, 0.9, 0.1]), threshold=0.5)
    assert (men, women, ratio) == (1.0, 0.5, 2.0)


def synthetic_report_data(sex_selected_by_woe=False, n=900, seed=3):
    rng = np.random.default_rng(seed)
    sex = rng.choice(["Nam", "Nữ"], n, p=[0.4, 0.6])
    age = rng.choice(list(AGE_GROUP_ORDER), n, p=[0.3, 0.6, 0.1])
    risk = rng.normal(size=n) + 0.3 * (sex == "Nam")
    y = (risk + rng.normal(scale=1.5, size=n) > 0.9).astype(int)
    base = 1 / (1 + np.exp(-(risk - 1.5)))
    pd_by_model = {
        MAIN: base,
        MAIN_SEX: np.clip(base + 1e-6 * (sex == "Nam"), 0, 1),
        SCORECARD: np.clip(base + rng.normal(scale=0.03, size=n), 0.001, 0.999),
    }
    pd_by_model[SCORECARD_SEX] = pd_by_model[SCORECARD].copy()
    return {
        "y": y, "sex": sex, "age": age, "pd": pd_by_model, "platt": (1.01, 0.04),
        "sex_iv": 0.0078, "sex_iv_threshold": 0.02,
        "sex_selected_by_woe": sex_selected_by_woe, "sex_importance_share": 0.0017,
    }


def test_build_report_end_to_end_on_synthetic_data(monkeypatch):
    monkeypatch.setattr(analyze_fairness, "N_BOOTSTRAP", 30)
    report = build_report(synthetic_report_data(), (0.09, 0.5))

    assert report.startswith("# Báo cáo Đánh giá Tính Công bằng")
    for section in ("## 1. Phạm vi", "## 2. Tác động của biến SEX", "## 3. Kết quả mô hình chính",
                    "## 4. Đối chiếu", "## 5. Kết luận", "## 6. Hạn chế"):
        assert section in report
    # Định nghĩa DIR được nêu rõ cùng nhóm tham chiếu
    assert "Nhóm tham chiếu là nhóm đông nhất" in report and "**Nữ**" in report and "**30–50**" in report
    # Hai bản Scorecard trùng nhau vì SEX bị loại ở bước WoE: báo cáo phải giải thích
    assert "bước WoE loại SEX vì IV = 0.0078" in report
    # Kết luận có đủ 5 mục, có số liệu và nêu quyết định về SEX
    conclusions = report.split("## 5. Kết luận")[1].split("## 6. Hạn chế")[0]
    assert all(f"\n{i}. " in conclusions for i in range(1, 6))
    assert "Charter mục 8" in conclusions


def test_build_report_describes_sex_selected_by_woe(monkeypatch):
    monkeypatch.setattr(analyze_fairness, "N_BOOTSTRAP", 20)
    report = build_report(synthetic_report_data(sex_selected_by_woe=True), (0.09, 0.5))
    assert "bước WoE chọn SEX" in report
    assert "bước WoE loại SEX" not in report


def test_load_test_predictions_joins_sex_and_age_by_id(tmp_path):
    raw = pd.DataFrame({"ID": [1, 2, 3, 4], "SEX": [1, 2, 2, 1], "AGE": [25, 40, 55, 30]})
    raw.to_csv(tmp_path / "raw.csv", index=False)
    pd.DataFrame({"ID": [4, 2, 1, 3], "y_true": [0, 1, 0, 1], "pd_raw": [.1] * 4,
                  "pd_final": [.2, .3, .4, .5], "pd_baseline": [.6, .7, .8, .9]}).to_csv(tmp_path / "p.csv", index=False)

    data = load_test_predictions(tmp_path / "p.csv", tmp_path / "raw.csv")

    assert data["split"] == "Test"
    assert data["sex"].tolist() == ["Nam", "Nữ", "Nam", "Nữ"]
    assert data["age"].tolist() == ["30–50", "30–50", "<30", ">50"]
    assert data["pd"][MAIN].tolist() == [.2, .3, .4, .5]


def test_load_test_predictions_rejects_unknown_id(tmp_path):
    pd.DataFrame({"ID": [1], "SEX": [1], "AGE": [25]}).to_csv(tmp_path / "raw.csv", index=False)
    pd.DataFrame({"ID": [9], "y_true": [0], "pd_raw": [.1], "pd_final": [.1], "pd_baseline": [.1]}).to_csv(tmp_path / "p.csv", index=False)
    with pytest.raises(ValueError, match="không khớp"):
        load_test_predictions(tmp_path / "p.csv", tmp_path / "raw.csv")


def test_build_test_report_uses_test_wording(monkeypatch):
    monkeypatch.setattr(analyze_fairness, "N_BOOTSTRAP", 20)
    d = synthetic_report_data()
    data = {"split": "Test", "y": d["y"], "sex": d["sex"], "age": d["age"],
            "pd": {MAIN: d["pd"][MAIN], BASELINE: d["pd"][SCORECARD]}}
    report = build_test_report(data, (0.09, 0.5))
    assert "trên tập Test" in report and "hồ sơ Test" in report
    assert "hồ sơ Valid" not in report.split("## 4. Kết luận")[1].split("## 5.")[0]
    assert "Vai trò của SEX" not in report  # không có bản có SEX trên Test
