"""Tiện ích giải thích mô hình: gộp giá trị SHAP theo biến gốc và theo nhóm đặc trưng.

Nhóm đặc trưng được khai báo trong configs/config.yaml (khóa `feature_groups`).
SHAP có tính cộng tính nên tổng giá trị SHAP của các cột trong cùng một biến/nhóm vẫn bảo toàn
tổng đóng góp: base + sum(nhóm) = đầu ra của mô hình.
"""

from dataclasses import dataclass
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


def reason_codes(contributions: pd.Series, k: int = 3, min_abs: float = 0.0) -> pd.DataFrame:
    """Reason codes của một khách hàng từ đóng góp SHAP theo nhóm (thang log-odds).

    Trả về tối đa k nhóm làm tăng rủi ro nhiều nhất (đóng góp dương) và k nhóm làm giảm rủi ro nhiều
    nhất (đóng góp âm), bỏ qua nhóm có |đóng góp| <= min_abs. Cột: nhóm, đóng góp, chiều.
    """
    rows = []
    for direction, part in (
        ("Tăng rủi ro", contributions[contributions > min_abs].sort_values(ascending=False)),
        ("Giảm rủi ro", contributions[contributions < -min_abs].sort_values()),
    ):
        rows += [{"nhóm": name, "đóng góp": float(value), "chiều": direction} for name, value in part.head(k).items()]
    return pd.DataFrame(rows, columns=["nhóm", "đóng góp", "chiều"])


# ---------------------------------------------------------------------------
# Giá trị SHAP cho một Pipeline và so sánh mức đồng thuận giữa các mô hình (Tuần 3 – T2)
# ---------------------------------------------------------------------------


@dataclass
class ShapResult:
    """Kết quả giải thích một Pipeline `preprocess -> clf` trên thang margin/logit chưa hiệu chuẩn.

    values: (n, số cột sau one-hot); base: base value; margin: đầu ra thô của mô hình trên cùng mẫu.
    Phép cộng base + values.sum(axis=1) == margin được kiểm tra bởi `max_additivity_error`.
    """

    values: np.ndarray
    base: float
    margin: np.ndarray
    columns: List[str]
    categorical_vars: List[str]

    @property
    def max_additivity_error(self) -> float:
        return float(np.abs(self.base + self.values.sum(axis=1) - self.margin).max())

    def by_variable(self) -> pd.DataFrame:
        return aggregate_by_variable(self.values, self.columns, self.categorical_vars)

    def by_group(self, groups: Dict[str, dict]) -> pd.DataFrame:
        return aggregate_by_group(self.by_variable(), groups)


def explain_pipeline(pipeline, X: pd.DataFrame, background: Optional[pd.DataFrame] = None) -> ShapResult:
    """SHAP của Pipeline `preprocess -> clf` (make_pipeline) cho các dòng của X.

    Mô hình cây dùng TreeExplainer (tree_path_dependent, không cần background). Logistic Regression dùng
    công thức chính xác của SHAP interventional cho mô hình tuyến tính, phi_k = coef_k * (z_k - mean_k),
    trong đó mean_k là trung bình trên `background` (bắt buộc), trùng với shap.LinearExplainer.
    """
    prep, clf = pipeline.named_steps["preprocess"], pipeline.named_steps["clf"]
    col_tf = prep.named_steps["preprocessor"]
    columns = [name.split("__", 1)[1] for name in col_tf.get_feature_names_out()]
    categorical_vars = list(col_tf.transformers_[0][2])
    Z = np.asarray(prep.transform(X), dtype=float)

    if hasattr(clf, "coef_"):
        if background is None:
            raise ValueError("Mô hình tuyến tính cần `background` (dữ liệu Train) để tính base value.")
        mean = np.asarray(prep.transform(background), dtype=float).mean(axis=0)
        values = clf.coef_[0] * (Z - mean)
        base = float(clf.intercept_[0] + clf.coef_[0] @ mean)
        margin = clf.decision_function(Z)
    else:
        import shap

        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(Z)
        if isinstance(values, list):  # phiên bản shap cũ trả về một mảng cho mỗi lớp
            values = values[-1]
        base = float(np.ravel(explainer.expected_value)[-1])
        name = type(clf).__name__
        if "LGBM" in name:
            margin = clf.predict(Z, raw_score=True)
        elif "XGB" in name:
            margin = clf.predict(Z, output_margin=True)
        else:
            raise ValueError(f"Chưa hỗ trợ tính margin cho {name}.")
    return ShapResult(np.asarray(values), base, np.asarray(margin, dtype=float), columns, categorical_vars)


def group_importance_share(by_group: pd.DataFrame) -> pd.Series:
    """Tỷ trọng mean(|SHAP|) của từng nhóm trong tổng; tổng bằng 1, so sánh được giữa các mô hình."""
    importance = by_group.abs().mean()
    return importance / importance.sum()


def bootstrap_share_ci(by_group: pd.DataFrame, n_boot: int = 500, level: float = 0.95, seed: int = 42) -> pd.DataFrame:
    """Khoảng tin cậy bootstrap theo khách hàng cho tỷ trọng nhóm; cột `lo`, `hi`."""
    rng = np.random.default_rng(seed)
    abs_values = by_group.abs().to_numpy()
    shares = np.empty((n_boot, abs_values.shape[1]))
    for i in range(n_boot):
        mean = abs_values[rng.integers(0, len(abs_values), len(abs_values))].mean(axis=0)
        shares[i] = mean / mean.sum()
    alpha = (1 - level) / 2 * 100
    lo, hi = np.percentile(shares, [alpha, 100 - alpha], axis=0)
    return pd.DataFrame({"lo": lo, "hi": hi}, index=by_group.columns)


def rank_agreement(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    """Spearman và Kendall tau giữa hai bảng mức độ quan trọng (căn theo chỉ mục chung)."""
    from scipy.stats import kendalltau, spearmanr

    a, b = a.align(b, join="inner")
    return {"spearman": float(spearmanr(a, b)[0]), "kendall": float(kendalltau(a, b)[0])}


def top_k_overlap(a: pd.Series, b: pd.Series, k: int = 10) -> int:
    """Số phần tử chung trong Top k của hai bảng mức độ quan trọng."""
    return len(set(a.nlargest(k).index) & set(b.nlargest(k).index))


def contribution_agreement(by_a: pd.DataFrame, by_b: pd.DataFrame) -> pd.Series:
    """Spearman giữa đóng góp của từng nhóm của hai mô hình, tính qua các khách hàng (cùng thứ tự dòng)."""
    from scipy.stats import spearmanr

    return pd.Series({col: float(spearmanr(by_a[col], by_b[col])[0]) for col in by_a.columns})


def top_group_agreement(by_a: pd.DataFrame, by_b: pd.DataFrame, mask=None) -> float:
    """Tỷ lệ khách hàng có cùng nhóm làm tăng rủi ro mạnh nhất (reason code đầu tiên) ở hai mô hình."""
    same = by_a.idxmax(axis=1).to_numpy() == by_b.idxmax(axis=1).to_numpy()
    return float(same[np.asarray(mask)].mean() if mask is not None else same.mean())


def effect_direction(by_var: pd.DataFrame, values: pd.DataFrame, variables: Sequence[str]) -> pd.Series:
    """Chiều tác động của biến số: Spearman giữa giá trị biến và đóng góp SHAP của chính biến đó.

    Dương: giá trị càng cao thì đóng góp càng làm tăng rủi ro. Bỏ qua dòng thiếu giá trị.
    """
    from scipy.stats import spearmanr

    return pd.Series({
        v: float(spearmanr(values[v], by_var[v], nan_policy="omit")[0]) for v in variables
    })
