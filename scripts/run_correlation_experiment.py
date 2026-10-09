"""Thí nghiệm lọc đặc trưng cho Logistic Scorecard (Tuần 2 - T4).

So sánh 5-fold CV (cùng fold với các mô hình khác) của 3 cấu hình:
    freeze_v1         : WoE + lọc IV >= 0.02 (cấu hình chính thức)
    iv_only           : FeatureSelectionTransformer, chỉ lọc IV
    iv_correlation    : FeatureSelectionTransformer, lọc IV + tương quan (Spearman > 0.9)

Usage:
    python scripts/run_correlation_experiment.py

Kết quả: reports/correlation_experiment_auc.csv (AUC/KS theo cấu hình, chênh lệch theo fold so với
freeze_v1) và reports/correlation_feature_selection.csv (biến bị loại ở từng fold, kèm lý do).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.config import load_config
from src.data_split import load_split_data
from src.features import build_features
from src.pipelines import build_scorecard_pipeline
from src.train import N_SPLITS, run_cv, summarize_folds

CONFIGS = {
    "freeze_v1": dict(enable_feature_selection=False),
    "iv_only": dict(enable_feature_selection=True, enable_correlation_filter=False),
    "iv_correlation": dict(enable_feature_selection=True, enable_correlation_filter=True),
}


def main():
    cfg = load_config()
    target_col = cfg.get("data", {}).get("target_col", "default.payment.next.month")
    random_state = cfg.get("random_state", 42)

    train_df, _, _ = load_split_data()
    X, y = build_features(train_df, target_col=target_col)

    fold_auc = {}
    results = []
    selection_rows = []

    for name, kwargs in CONFIGS.items():
        n_selected = []

        def collect(fold, pipeline, name=name, n_selected=n_selected):
            selector = pipeline.named_steps.get("feature_selection")
            woe = pipeline.named_steps["woe_iv"]
            n_selected.append(len(woe.selected_features_))
            if selector is not None:
                selection_rows.extend({"config": name, "fold": fold, **d} for d in selector.dropped_features_)

        folds = run_cv(
            build_scorecard_pipeline(drop_sensitive=True, **kwargs),
            X,
            y,
            n_splits=N_SPLITS,
            random_state=random_state,
            on_fold_fit=collect,
        )
        summary = summarize_folds(folds)
        fold_auc[name] = np.array([m["roc_auc"] for m in folds])
        diff = fold_auc[name] - fold_auc["freeze_v1"]
        results.append({
            "config": name,
            "auc_mean": summary["roc_auc"][0],
            "auc_std": summary["roc_auc"][1],
            "ks_mean": summary["ks"][0],
            "n_features_mean": float(np.mean(n_selected)),
            "auc_diff_vs_freeze_v1_mean": float(diff.mean()),
            "auc_diff_vs_freeze_v1_min": float(diff.min()),
            "auc_diff_vs_freeze_v1_max": float(diff.max()),
        })
        print(
            f"{name:15s} AUC {summary['roc_auc'][0]:.4f} ± {summary['roc_auc'][1]:.4f}  "
            f"KS {summary['ks'][0]:.4f}  n_features {np.mean(n_selected):.1f}  "
            f"diff vs freeze_v1 {diff.mean():+.4f}"
        )

    report_dir = ROOT_DIR / "reports"
    pd.DataFrame(results).to_csv(report_dir / "correlation_experiment_auc.csv", index=False, float_format="%.6f")
    pd.DataFrame(
        selection_rows,
        columns=["config", "fold", "feature", "iv", "reason", "correlated_with", "correlation"],
    ).to_csv(report_dir / "correlation_feature_selection.csv", index=False, float_format="%.6f")
    print("\nWrote reports/correlation_experiment_auc.csv, reports/correlation_feature_selection.csv")


if __name__ == "__main__":
    main()
