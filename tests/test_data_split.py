import numpy as np
import pandas as pd
import pytest
import json

from src.data_split import load_split_data, load_split_indices, save_splits, split_indices


@pytest.fixture
def dummy_data():
    """Tạo bộ dữ liệu giả lập 1.000 dòng với tỷ lệ class 1 là 20%."""
    np.random.seed(42)
    n = 1000
    target = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    return pd.DataFrame({
        "ID": np.arange(1, n + 1),
        "feature": np.random.randn(n),
        "target": target,
    })


def test_split_indices_proportions_and_sizes(dummy_data):
    splits = split_indices(dummy_data, target_col="target", random_state=42)
    assert len(splits["train"]) == 600
    assert len(splits["valid"]) == 200
    assert len(splits["test"]) == 200


def test_split_indices_disjoint_and_complete(dummy_data):
    splits = split_indices(dummy_data, target_col="target", random_state=42)
    train_set = set(splits["train"])
    valid_set = set(splits["valid"])
    test_set = set(splits["test"])

    # Không giao nhau giữa các tập
    assert len(train_set & valid_set) == 0
    assert len(train_set & test_set) == 0
    assert len(valid_set & test_set) == 0

    # Hợp lại đủ 100% dữ liệu gốc
    assert len(train_set | valid_set | test_set) == len(dummy_data)


def test_split_stratification(dummy_data):
    splits = split_indices(dummy_data, target_col="target", random_state=42)
    overall_mean = dummy_data["target"].mean()

    for name, idx in splits.items():
        subset_mean = dummy_data.iloc[idx]["target"].mean()
        # Tỷ lệ vỡ nợ không được lệch quá 1.5% so với tổng thể
        assert abs(subset_mean - overall_mean) < 0.015


def test_save_and_load_splits(dummy_data, tmp_path):
    splits = split_indices(dummy_data, target_col="target", random_state=42)
    save_splits(
        splits=splits,
        df=dummy_data,
        output_dir=tmp_path,
        target_col="target",
        random_state=42,
        raw_file="dummy.csv",
    )

    loaded = load_split_indices(tmp_path)
    assert loaded["train"] == splits["train"]
    assert loaded["valid"] == splits["valid"]
    assert loaded["test"] == splits["test"]

    summary_file = tmp_path / "splits_summary.json"
    assert summary_file.exists()

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["raw_file"] == "dummy.csv"
    assert summary["random_state"] == 42
    assert "default_rate" in summary["splits"]["train"]
def test_split_indices_reproducibility(dummy_data):
    """Kiểm tra tính tái lập: cùng random_state cho kết quả giống hệt, khác random_state cho kết quả khác."""
    splits_1 = split_indices(dummy_data, target_col="target", random_state=42)
    splits_2 = split_indices(dummy_data, target_col="target", random_state=42)
    assert splits_1 == splits_2

    splits_diff = split_indices(dummy_data, target_col="target", random_state=999)
    assert splits_1["train"] != splits_diff["train"]


def test_split_indices_invalid_ratios(dummy_data):
    """Kiểm tra ngoại lệ ValueError khi tổng tỷ lệ chia khác 1.0 (nhánh lỗi 1)."""
    with pytest.raises(ValueError, match="Tổng tỷ lệ chia phải bằng 1.0"):
        split_indices(dummy_data, target_col="target", train_ratio=0.7, valid_ratio=0.2, test_ratio=0.2)

    with pytest.raises(ValueError, match="Tổng tỷ lệ chia phải bằng 1.0"):
        split_indices(dummy_data, target_col="target", train_ratio=0.5, valid_ratio=0.2, test_ratio=0.2)


def test_split_indices_missing_target(dummy_data):
    """Kiểm tra ngoại lệ KeyError khi cột target không tồn tại trong DataFrame (nhánh lỗi 2)."""
    with pytest.raises(KeyError, match="Không tìm thấy cột mục tiêu"):
        split_indices(dummy_data, target_col="non_existent_column")


def test_load_split_data(dummy_data, tmp_path):
    """Kiểm tra nạp 3 DataFrame (train, valid, test) từ raw CSV và thư mục splits."""
    raw_path = tmp_path / "raw.csv"
    splits_dir = tmp_path / "splits"
    dummy_data.to_csv(raw_path, index=False)

    splits = split_indices(dummy_data, target_col="target", random_state=42)
    save_splits(splits, dummy_data, splits_dir)

    train_df, valid_df, test_df = load_split_data(raw_path=raw_path, splits_dir=splits_dir)

    # 1. Kiểm tra kích thước từng DataFrame
    assert len(train_df) == 600
    assert len(valid_df) == 200
    assert len(test_df) == 200

    # 2. Kiểm tra index đã được reset về 0..len-1
    assert train_df.index.tolist() == list(range(600))
    assert valid_df.index.tolist() == list(range(200))
    assert test_df.index.tolist() == list(range(200))

    # 3. Kiểm tra tính toàn vẹn dữ liệu: ID khớp chính xác với chỉ số đã chia
    np.testing.assert_array_equal(
        train_df["ID"].to_numpy(),
        dummy_data.iloc[splits["train"]]["ID"].to_numpy(),
    )
    np.testing.assert_array_equal(
        test_df["ID"].to_numpy(),
        dummy_data.iloc[splits["test"]]["ID"].to_numpy(),
    )
