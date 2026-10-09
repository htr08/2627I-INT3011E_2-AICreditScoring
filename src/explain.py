"""Tiện ích giải thích mô hình: gộp giá trị SHAP theo biến gốc và theo nhóm đặc trưng.

Nhóm đặc trưng được khai báo trong configs/config.yaml (khóa `feature_groups`).
SHAP có tính cộng tính nên tổng giá trị SHAP của các cột trong cùng một biến/nhóm vẫn bảo toàn
tổng đóng góp: base + sum(nhóm) = đầu ra của mô hình.
"""

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from src.config import PROJECT_ROOT, load_config

FREEZE_PATH = PROJECT_ROOT / "configs" / "feature_freeze_v1.yaml"


def load_feature_groups(config: Optional[dict] = None) -> Dict[str, dict]:
    """Trả về {mã nhóm: {"label": ..., "variables": [...]}} từ config; báo lỗi nếu một biến thuộc nhiều nhóm."""
    config = config or load_config()
    groups = config["feature_groups"]
    variable_to_group(groups)
    return groups


def variable_to_group(groups: Dict[str, dict]) -> Dict[str, str]:
    """Ánh xạ biến gốc -> mã nhóm. Một biến thuộc nhiều nhóm sẽ gây ValueError."""
    mapping: Dict[str, str] = {}
    for key, spec in groups.items():
        for var in spec["variables"]:
            if var in mapping:
                raise ValueError(f"Biến {var} thuộc cả '{mapping[var]}' và '{key}'.")
            mapping[var] = key
    return mapping


def model_variables_from_freeze(path=FREEZE_PATH) -> List[str]:
    """Danh sách biến gốc đưa vào mô hình theo feature freeze (không gồm cột bị loại)."""
    with open(path, encoding="utf-8") as f:
        mf = yaml.safe_load(f)["model_features"]
    return list(mf["categorical"]) + list(mf["numeric_raw"]) + list(mf["numeric_engineered"])


def check_groups_cover(groups: Dict[str, dict], variables: Iterable[str]) -> None:
    """Kiểm tra mọi biến của mô hình đều có nhóm và không có biến thừa trong config."""
    in_groups = set(variable_to_group(groups))
    in_model = set(variables)
    missing, extra = in_model - in_groups, in_groups - in_model
    if missing or extra:
        raise ValueError(f"Nhóm đặc trưng không khớp feature freeze: thiếu {sorted(missing)}, thừa {sorted(extra)}.")


def source_variable(column: str, categorical_vars: Sequence[str]) -> str:
    """Tên biến gốc của một cột sau one-hot (vd 'PAY_1_2' -> 'PAY_1'); cột số giữ nguyên tên."""
    for var in sorted(categorical_vars, key=len, reverse=True):
        if column.startswith(var + "_"):
            return var
    return column


def aggregate_by_variable(values: np.ndarray, columns: Sequence[str], categorical_vars: Sequence[str]) -> pd.DataFrame:
    """Cộng giá trị SHAP của các cột one-hot về biến gốc. Shape (n, số biến gốc)."""
    col_to_var = pd.Series({c: source_variable(c, categorical_vars) for c in columns})
    return pd.DataFrame(values, columns=list(columns)).T.groupby(col_to_var).sum().T


def aggregate_by_group(shap_by_var: pd.DataFrame, groups: Dict[str, dict]) -> pd.DataFrame:
    """Cộng giá trị SHAP theo nhóm đặc trưng; cột kết quả mang nhãn hiển thị (label)."""
    var_to_group = variable_to_group(groups)
    unknown = set(shap_by_var.columns) - set(var_to_group)
    if unknown:
        raise ValueError(f"Biến chưa có nhóm: {sorted(unknown)}")
    labels = {key: spec["label"] for key, spec in groups.items()}
    by_group = shap_by_var.T.groupby(shap_by_var.columns.map(var_to_group)).sum().T
    return by_group.rename(columns=labels)
