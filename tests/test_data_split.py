import numpy as np
import pandas as pd
import pytest

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
    save_splits(splits, dummy_data, tmp_path)

    loaded = load_split_indices(tmp_path)
    assert loaded["train"] == splits["train"]
    assert loaded["valid"] == splits["valid"]
    assert loaded["test"] == splits["test"]

    assert (tmp_path / "train_indices.csv").exists()
    assert (tmp_path / "valid_indices.csv").exists()
    assert (tmp_path / "test_indices.csv").exists()
    assert (tmp_path / "splits_summary.json").exists()
