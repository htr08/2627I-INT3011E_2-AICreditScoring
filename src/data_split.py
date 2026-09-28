"""Module chia dữ liệu Train/Valid/Test (60/20/20) Stratified và quản lý file chỉ số."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np

from src.config import PROJECT_ROOT, load_config


def split_indices(
    df: pd.DataFrame,
    target_col: str = "default.payment.next.month",
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Dict[str, List[int]]:
    """Chia chỉ số dòng (0-indexed) thành Train/Valid/Test có phân tầng (stratified).
    
    Quy trình:
    1. Tách Test (test_ratio = 0.20) -> còn lại 80% (Train + Valid).
    2. Tách Valid từ phần 80% với tỷ lệ valid_ratio / (train_ratio + valid_ratio)
       = 0.2 / 0.8 = 0.25 (25%) -> thu được đúng 60% Train và 20% Valid của tổng thể.
    """
    total_ratio = train_ratio + valid_ratio + test_ratio
    if not (0.999 <= total_ratio <= 1.001):
        raise ValueError(f"Tổng tỷ lệ chia phải bằng 1.0, hiện tại là: {total_ratio}")

    if target_col not in df.columns:
        raise KeyError(f"Không tìm thấy cột mục tiêu '{target_col}' trong DataFrame.")

    indices = np.arange(len(df))
    targets = df[target_col].to_numpy()

    # Bước 1: Tách Test set
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=targets,
        random_state=random_state,
    )

    # Bước 2: Tách Valid set từ Train+Val
    val_relative_ratio = valid_ratio / (train_ratio + valid_ratio)
    train_val_targets = df[target_col].to_numpy()[train_val_idx]

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative_ratio,
        stratify=train_val_targets,
        random_state=random_state,
    )

    return {
        "train": sorted(train_idx.tolist()),
        "valid": sorted(val_idx.tolist()),
        "test": sorted(test_idx.tolist()),
    }


def save_splits(
    splits: Dict[str, List[int]],
    df: pd.DataFrame,
    output_dir: Path,
    target_col: str = None,
    random_state: int = None,
    raw_file: str = None,
) -> None:
    """Lưu danh sách chỉ số ra file JSON vào thư mục output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lưu file tổng hợp splits.json
    with open(output_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    # 2. Lưu file metadata/summary để tra cứu nhanh
    has_target = target_col is not None and target_col in df.columns
    summary = {
        "raw_file": Path(raw_file).name if raw_file else None,
        "random_state": random_state,
        "target_col": target_col,
        "total_samples": len(df),
    }
    if has_target:
        summary["overall_default_rate"] = round(float(df[target_col].mean()), 4)

    splits_info = {}
    for k, v in splits.items():
        info = {
            "count": len(v),
            "ratio": round(len(v) / len(df), 4),
        }
        if has_target:
            info["default_rate"] = round(float(df[target_col].to_numpy()[v].mean()),4)
        splits_info[k] = info

    summary["splits"] = splits_info

    with open(output_dir / "splits_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def load_split_indices(splits_dir: Path = None) -> Dict[str, List[int]]:
    """Đọc lại các chỉ số dòng đã lưu từ thư mục splits."""
    if splits_dir is None:
        config = load_config()
        splits_dir = PROJECT_ROOT / config["data"]["splits_dir"]

    json_path = splits_dir / "splits.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Chưa có file chỉ số tại '{json_path}'. Hãy chạy scripts/split_data.py trước."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split_data(
    raw_path: Path = None,
    splits_dir: Path = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hàm tiện ích nạp sẵn 3 DataFrame: Train, Valid, Test cho các thành viên khác."""
    if raw_path is None or splits_dir is None:
        config = load_config()
        raw_path = raw_path or (PROJECT_ROOT / config["data"]["raw_path"])
        splits_dir = splits_dir or (PROJECT_ROOT / config["data"]["splits_dir"])

    df = pd.read_csv(raw_path)
    splits = load_split_indices(splits_dir)

    train_df = df.iloc[splits["train"]].copy().reset_index(drop=True)
    valid_df = df.iloc[splits["valid"]].copy().reset_index(drop=True)
    test_df = df.iloc[splits["test"]].copy().reset_index(drop=True)

    return train_df, valid_df, test_df
