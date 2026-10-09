"""Thí nghiệm bộ đặc trưng cho mô hình cây đã tuning (Tuần 2 – T4, sau feature freeze v1).

So sánh trên cùng 5 fold Stratified CV, cùng bộ tham số tốt nhất của study Optuna:
    v1_onehot      : feature freeze v1 (PAY_1..6 one-hot) — cấu hình chính thức
    v1_decorr      : v1 bỏ các biến tương quan cao (Spearman > 0.9, reports/feature_selection_decision.md)
    ordinal_raw    : PAY_1..6 dạng số thứ tự thô (-2..8)
    split          : PAY_i -> PAY_i_DELAY = max(PAY_i, 0) (ràng buộc đơn điệu) + cờ NOUSE (-2), PAIDFULL (-1)
    split_new      : split + 6 đặc trưng mới (MONTHS_SINCE_LATE, PAY_TREND_12, N_NOUSE, N_PAIDFULL,
                     PAY_AMT_LIMIT_MEAN, UTIL_TREND)

Usage:
    python scripts/run_feature_experiment.py --model lightgbm
    python scripts/run_feature_experiment.py --model catboost
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config
from src.data_split import load_split_data
from src.evaluate import compute_ks, compute_pr_auc, compute_roc_auc
from src.features import ENGINEERED_COLUMNS, AgeBinningTransformer, FeatureEngineeringTransformer, build_features
from src.preprocessing import NUMERIC_COLUMNS, SENSITIVE_COLUMNS, AbnormalCodeTransformer, DropColumnsTransformer
from src.train import get_cv_splitter
from src.tune import build_estimator, monotone_vector

PAY = [f"PAY_{i}" for i in range(1, 7)]
SPLIT_PAY = [f"{c}_{s}" for c in PAY for s in ("DELAY", "NOUSE", "PAIDFULL")]
NEW_FEATURES = ["MONTHS_SINCE_LATE", "PAY_TREND_12", "N_NOUSE", "N_PAIDFULL", "PAY_AMT_LIMIT_MEAN", "UTIL_TREND"]
# Biến bị loại theo lọc tương quan Spearman > 0.9 trên Train (reports/correlation_drop.csv).
CORRELATED = ["PAY_LATE_CONSECUTIVE", "BILL_STD", "UTIL_1", "UTIL_2", "UTIL_3", "UTIL_4", "UTIL_5",
              "BILL_AMT1", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5"]
VARIANTS = ["v1_onehot", "v1_decorr", "ordinal_raw", "split", "split_new"]

# Bộ tham số tốt nhất của study Optuna (reports/experiments_optuna_tuning.md mục 3.4).
BEST_PARAMS = {
    "lightgbm": dict(n_estimators=1000, learning_rate=0.007777559917183324, num_leaves=10, min_child_samples=12,
                     subsample=0.7370640101182702, colsample_bytree=0.4983640827521146,
                     reg_alpha=0.0016438437374599448, reg_lambda=8.873933824224165),
    "catboost": dict(iterations=1100, learning_rate=0.011058500546306235, depth=6, l2_leaf_reg=16.7613476745288,
                     random_strength=3.4752685500732023, subsample=0.9969814418709121),
}
# LightGBM dùng monotonic constraints (study tốt nhất); CatBoost không ràng buộc (chi phí ~5x).
USE_MONOTONE = {"lightgbm": True, "catboost": False}


class PayEncodingTransformer(BaseEstimator, TransformerMixin):
    """Tách PAY_i thành số tháng trễ (đơn điệu với rủi ro) và cờ trạng thái -2 / -1 (không đơn điệu)."""

    def __init__(self, new_features: bool = False):
        self.new_features = new_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in PAY:
            X[f"{c}_DELAY"] = X[c].clip(lower=0)
            X[f"{c}_NOUSE"] = (X[c] == -2).astype(int)
            X[f"{c}_PAIDFULL"] = (X[c] == -1).astype(int)
        if self.new_features:
            late = X[[f"{c}_DELAY" for c in PAY]].to_numpy() > 0
            X["MONTHS_SINCE_LATE"] = np.where(late.any(axis=1), late.argmax(axis=1), 6)
            X["PAY_TREND_12"] = X["PAY_1_DELAY"] - X["PAY_2_DELAY"]
            X["N_NOUSE"] = X[[f"{c}_NOUSE" for c in PAY]].sum(axis=1)
            X["N_PAIDFULL"] = X[[f"{c}_PAIDFULL" for c in PAY]].sum(axis=1)
            pay_amt = X[[f"PAY_AMT{i}" for i in range(1, 7)]].mean(axis=1)
            X["PAY_AMT_LIMIT_MEAN"] = pay_amt / X["LIMIT_BAL"].replace(0, np.nan)
            X["UTIL_TREND"] = X["UTIL_1"] - X["UTIL_6"]
        return X


def build_variant(variant: str) -> Pipeline:
    numeric = [c for c in NUMERIC_COLUMNS if c != "AGE"] + ENGINEERED_COLUMNS
    categorical = ["EDUCATION", "MARRIAGE", "AGE_BIN"]
    steps = [("abnormal_codes", AbnormalCodeTransformer()), ("feature_eng", FeatureEngineeringTransformer())]
    if variant in ("v1_onehot", "v1_decorr"):
        categorical += PAY
    elif variant == "ordinal_raw":
        numeric += PAY
    else:
        steps.append(("pay_encoding", PayEncodingTransformer(new_features=variant == "split_new")))
        numeric += SPLIT_PAY + (NEW_FEATURES if variant == "split_new" else [])
    if variant == "v1_decorr":
        numeric = [c for c in numeric if c not in CORRELATED]
    steps += [
        ("age_binning", AgeBinningTransformer(min_bins=3, max_bins=8)),
        ("drop_sensitive", DropColumnsTransformer(columns=SENSITIVE_COLUMNS)),
        ("preprocessor", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
             numeric),
        ])),
    ]
    return Pipeline(steps)


def monotone_features(variant: str) -> list:
    increasing = list(load_config()["tuning"]["monotone_increasing"])
    if variant.startswith("split"):
        increasing += [f"{c}_DELAY" for c in PAY]
    return increasing


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=list(BEST_PARAMS), default="lightgbm")
    args = parser.parse_args()

    train_df, _, _ = load_split_data()
    X, y = build_features(train_df)
    splits = list(get_cv_splitter(random_state=42).split(X, y))

    rows = []
    for variant in VARIANTS:
        start = time.time()
        for fold, (tr_idx, val_idx) in enumerate(splits, start=1):
            prep = build_variant(variant).fit(X.iloc[tr_idx], y.iloc[tr_idx])
            names = list(prep.named_steps["preprocessor"].get_feature_names_out())
            constraints = monotone_vector(names, monotone_features(variant)) if USE_MONOTONE[args.model] else None
            clf = build_estimator(args.model, BEST_PARAMS[args.model], 42, constraints)
            clf.fit(prep.transform(X.iloc[tr_idx]), y.iloc[tr_idx])
            y_val = y.iloc[val_idx].to_numpy()
            proba = clf.predict_proba(prep.transform(X.iloc[val_idx]))[:, 1]
            rows.append({
                "variant": variant, "fold": fold, "n_features": len(names),
                "roc_auc": compute_roc_auc(y_val, proba), "ks": compute_ks(y_val, proba),
                "pr_auc": compute_pr_auc(y_val, proba),
            })
        v = pd.DataFrame([r for r in rows if r["variant"] == variant])
        print(f"{variant:12s} AUC {v.roc_auc.mean():.4f} ± {v.roc_auc.std():.4f}  KS {v.ks.mean():.4f}  "
              f"PR-AUC {v.pr_auc.mean():.4f}  n_features {v.n_features.mean():.0f}  ({time.time() - start:.0f}s)")

    df = pd.DataFrame(rows)
    base = df[df.variant == "v1_onehot"].set_index("fold").roc_auc
    print("\nChênh lệch AUC theo fold so với v1_onehot:")
    for variant in VARIANTS[1:]:
        diff = df[df.variant == variant].set_index("fold").roc_auc - base
        print(f"  {variant:12s} {np.round(diff.to_numpy(), 4)}  mean {diff.mean():+.4f}")


if __name__ == "__main__":
    main()
