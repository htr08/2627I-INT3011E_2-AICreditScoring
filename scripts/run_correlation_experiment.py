import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.config import load_config
from src.data_split import load_split_data
from src.features import build_features
from src.pipelines import build_scorecard_pipeline
from src.train import run_cv, summarize_folds, N_SPLITS


def main():
    cfg = load_config()
    target_col = cfg.get("data", {}).get(
        "target_col", "default.payment.next.month"
    )
    random_state = cfg.get("random_state", 42)

    train_df, _, _ = load_split_data()
    X, y = build_features(train_df, target_col=target_col)

    results = []

    for enabled in (False, True):
        print(f"\nCorrelation filter enabled: {enabled}")

        pipeline = build_scorecard_pipeline(
            drop_sensitive=True,
            enable_feature_selection=True,
            enable_correlation_filter=enabled,
        )

        folds = run_cv(
            pipeline,
            X,
            y,
            n_splits=N_SPLITS,
            random_state=random_state,
        )

        summary = summarize_folds(folds)
        auc_mean, auc_std = summary["roc_auc"]

        results.append({
            "correlation_filter": enabled,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "note": "Experiment only; official feature freeze remains v1",
        })

        print(f"AUC: {auc_mean:.4f} +/- {auc_std:.4f}")

    report_dir = ROOT_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    output_file = report_dir / "correlation_experiment_auc.csv"
    pd.DataFrame(results).to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved results to: {output_file}")
    print("Official feature freeze v1 remains unchanged.")


if __name__ == "__main__":
    main()