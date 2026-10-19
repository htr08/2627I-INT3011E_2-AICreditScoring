"""Run the locked final evaluation on Test once (Week 3 – T3, ML Engineer)."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibrate import PlattCalibrator, evaluate_calibration, predict_margin
from src.data_split import load_split_data
from src.evaluate import (
    bootstrap_auc_diff_ci, bootstrap_metric_ci, compute_brier_score,
    compute_confusion_matrix_metrics, compute_expected_cost, compute_gini,
    compute_ks, compute_pr_auc, compute_roc_auc, delong_test,
)
from src.features import build_features
from src.pipelines import make_pipeline
from src.scoring import pd_to_score
from src.tune import fit_tuned_pipeline

logger = logging.getLogger(__name__)
FINAL_STUDY = "lightgbm_monotone"
FINAL_THRESHOLD = 0.09  # Valid EAD-weighted OOF mean 0.088, rounded before Test.
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 42
OUTPUT_DIR = PROJECT_ROOT / "reports" / "test_evaluation"
RUN_MARKER = OUTPUT_DIR / "test_evaluation_started.json"
RESULT_MARKERS = (
    RUN_MARKER,
    OUTPUT_DIR / "test_evaluation.md",
    OUTPUT_DIR / "test_metrics.csv",
)


def _metric_interval(y_true, y_prob, metric_fn):
    return bootstrap_metric_ci(
        y_true, y_prob, metric_fn=metric_fn, n_bootstraps=BOOTSTRAPS,
        ci=0.95, random_state=BOOTSTRAP_SEED,
    )


def _evaluate(y_true, y_prob):
    calibration = evaluate_calibration(y_true, y_prob, n_bins=10)
    metrics = {
        "roc_auc": compute_roc_auc(y_true, y_prob),
        "gini": compute_gini(y_true, y_prob),
        "ks": compute_ks(y_true, y_prob),
        "pr_auc": compute_pr_auc(y_true, y_prob),
        "brier_score": calibration["brier_score"],
        "log_loss": calibration["log_loss"],
        "ece": calibration["ece"],
        "mean_predicted_pd": float(np.mean(y_prob)),
    }
    for key, fn in {
        "roc_auc": compute_roc_auc,
        "pr_auc": compute_pr_auc,
        "ks": compute_ks,
        "brier_score": compute_brier_score,
    }.items():
        _, low, high = _metric_interval(y_true, y_prob, fn)
        metrics[f"{key}_ci95_low"] = low
        metrics[f"{key}_ci95_high"] = high
    return metrics, calibration


def _write_report(rows, comparison, decision, y_test):
    lines = [
        "# Tuần 3 – T3: Đánh giá cuối trên tập Test", "",
        "Test được đánh giá một lần. Mô hình, calibrator và ngưỡng đã khóa trước khi đánh giá: LightGBM tuned + monotonic, Platt fit trên Valid, ngưỡng 0.09 (làm tròn từ trung bình OOF EAD-weighted 0.088). Không điều chỉnh tham số theo kết quả Test.", "",
        f"Tỷ lệ default thực tế trên Test: **{np.mean(y_test):.2%}** ({len(y_test):,} hồ sơ).", "",
        "## Chỉ số phân tách và hiệu chuẩn", "",
        "| Mô hình/đầu ra | ROC-AUC (bootstrap 95% CI) | Gini | KS (bootstrap 95% CI) | PR-AUC (bootstrap 95% CI) | Brier (bootstrap 95% CI) | Log Loss | ECE | Mean PD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {roc_auc:.4f} ({roc_auc_ci95_low:.4f}, {roc_auc_ci95_high:.4f}) | "
            "{gini:.4f} | {ks:.4f} ({ks_ci95_low:.4f}, {ks_ci95_high:.4f}) | "
            "{pr_auc:.4f} ({pr_auc_ci95_low:.4f}, {pr_auc_ci95_high:.4f}) | "
            "{brier_score:.4f} ({brier_score_ci95_low:.4f}, {brier_score_ci95_high:.4f}) | "
            "{log_loss:.4f} | {ece:.4f} | {mean_predicted_pd:.2%} |".format(**row)
        )
    lines.extend([
        "", "## So sánh cuối với baseline Logistic Regression", "",
        f"- DeLong paired test: ΔAUC (LightGBM Platt − baseline) = **{comparison['diff']:.5f}**, 95% CI [{comparison['ci_lower']:.5f}, {comparison['ci_upper']:.5f}], p = **{comparison['p_value']:.5g}**.",
        f"- Paired bootstrap 95% CI cho ΔAUC: [{comparison['bootstrap_ci_lower']:.5f}, {comparison['bootstrap_ci_upper']:.5f}].",
        f"- Kết luận mức 5%: **{'có khác biệt có ý nghĩa' if comparison['is_significant'] else 'chưa có bằng chứng khác biệt có ý nghĩa'}**.",
        "", "## Kết quả tại ngưỡng đã khóa trên Valid", "",
        f"Ngưỡng PD = **0.09**; cutoff PDO = **{decision['cutoff_score']:.1f}** điểm.", "",
        "| Chỉ số | Ngưỡng 0.09 | Ngưỡng 0.50 |", "|---|---:|---:|",
        f"| Precision | {decision['cm_threshold']['precision']:.2%} | {decision['cm_05']['precision']:.2%} |",
        f"| Recall | {decision['cm_threshold']['recall']:.2%} | {decision['cm_05']['recall']:.2%} |",
        f"| Flag rate | {decision['cm_threshold']['flag_rate']:.2%} | {decision['cm_05']['flag_rate']:.2%} |",
        f"| Unit expected cost/customer | {decision['unit_cost']:.5f} | {decision['unit_cost_05']:.5f} |",
        f"| EAD-weighted cost (NT$/customer) | {decision['ead_cost']:.2f} | {decision['ead_cost_05']:.2f} |",
        "", "## Đồ thị", "", "![ROC, PR và calibration curves](test_curves.png)", "",
        "Chi tiết số liệu và khoảng tin cậy: `test_metrics.csv` và `test_comparison.csv`. Kết quả Test chỉ dùng nghiệm thu, không dùng đổi calibrator, ngưỡng hoặc mô hình.", "",
    ])
    (OUTPUT_DIR / "test_evaluation.md").write_text("\n".join(lines), encoding="utf-8")


def run_final_test_evaluation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    previous_run = next((path for path in RESULT_MARKERS if path.exists()), None)
    if previous_run is not None:
        raise RuntimeError(f"Test evaluation has already started ({previous_run}); do not rerun on Test.")
    RUN_MARKER.write_text(json.dumps({
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": FINAL_STUDY, "calibration": "platt", "threshold": FINAL_THRESHOLD,
        "bootstrap_replicates": BOOTSTRAPS, "status": "started",
    }, indent=2), encoding="utf-8")

    with (PROJECT_ROOT / "configs" / "config.yaml").open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    target_col = cfg["data"]["target_col"]
    cost_fn = float(cfg["cost_matrix"]["cost_fn"])
    cost_fp = float(cfg["cost_matrix"]["cost_fp"])

    train_df, valid_df, test_df = load_split_data()
    X_train, y_train = build_features(train_df, target_col=target_col)
    X_valid, y_valid = build_features(valid_df, target_col=target_col)
    X_test, y_test = build_features(test_df, target_col=target_col)
    y_train, y_valid, y_test = (y_train.to_numpy(dtype=int), y_valid.to_numpy(dtype=int), y_test.to_numpy(dtype=int))

    final_model = fit_tuned_pipeline(FINAL_STUDY, X_train, y_train)
    baseline_model = make_pipeline(LogisticRegression(max_iter=1000, random_state=cfg.get("random_state", 42))).fit(X_train, y_train)
    calibrator = PlattCalibrator().fit(predict_margin(final_model, X_valid), y_valid)

    probabilities = [
        ("LightGBM tuned + Monotonic (raw)", final_model.predict_proba(X_test)[:, 1]),
        ("LightGBM tuned + Monotonic (Platt)", calibrator.transform(predict_margin(final_model, X_test))),
        ("Logistic Regression baseline", baseline_model.predict_proba(X_test)[:, 1]),
    ]
    rows, curves = [], {}
    for name, p in probabilities:
        metrics, curve = _evaluate(y_test, p)
        rows.append({"model": name, **metrics})
        curves[name] = curve

    p_final = probabilities[1][1]
    p_baseline = probabilities[2][1]
    comparison = delong_test(y_test, p_final, p_baseline)
    boot_diff = bootstrap_auc_diff_ci(
        y_test, p_final, p_baseline, n_bootstraps=BOOTSTRAPS,
        ci=0.95, random_state=BOOTSTRAP_SEED,
    )
    comparison["bootstrap_ci_lower"] = boot_diff["ci_lower"]
    comparison["bootstrap_ci_upper"] = boot_diff["ci_upper"]

    ead_test = np.maximum(test_df["BILL_AMT1"].to_numpy(dtype=float), 0.0)
    decision = {
        "cm_threshold": compute_confusion_matrix_metrics(y_test, p_final, FINAL_THRESHOLD),
        "cm_05": compute_confusion_matrix_metrics(y_test, p_final, 0.5),
        "unit_cost": compute_expected_cost(y_test, p_final, FINAL_THRESHOLD, cost_fn, cost_fp),
        "unit_cost_05": compute_expected_cost(y_test, p_final, 0.5, cost_fn, cost_fp),
        "ead_cost": compute_expected_cost(y_test, p_final, FINAL_THRESHOLD, cost_fn, cost_fp, ead=ead_test),
        "ead_cost_05": compute_expected_cost(y_test, p_final, 0.5, cost_fn, cost_fp, ead=ead_test),
        "cutoff_score": float(pd_to_score(FINAL_THRESHOLD)),
    }

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False, float_format="%.8f")
    pd.DataFrame([{
        **comparison, "threshold": FINAL_THRESHOLD, "cost_fn": cost_fn,
        "cost_fp": cost_fp, "ead_currency": "TWD", "bootstrap_replicates": BOOTSTRAPS,
    }]).to_csv(OUTPUT_DIR / "test_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, p in probabilities:
        fpr, tpr, _ = roc_curve(y_test, p)
        precision, recall, _ = precision_recall_curve(y_test, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={compute_roc_auc(y_test, p):.3f})")
        axes[1].plot(recall, precision, label=f"{name} (AP={compute_pr_auc(y_test, p):.3f})")
        axes[2].plot(curves[name]["prob_pred"], curves[name]["prob_true"], marker="o", label=name)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axes[1].set(title="Precision–Recall Curve", xlabel="Recall", ylabel="Precision")
    axes[2].plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    axes[2].set(title="Calibration Curve", xlabel="Mean predicted PD", ylabel="Observed default rate")
    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_curves.png", dpi=180)
    plt.close(fig)

    _write_report(rows, comparison, decision, y_test)
    marker = json.loads(RUN_MARKER.read_text(encoding="utf-8"))
    marker.update({
        "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_test": int(len(y_test)),
        "artifacts": ["test_metrics.csv", "test_comparison.csv", "test_curves.png", "test_evaluation.md"],
    })
    RUN_MARKER.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    logger.info("Test evaluated once. Results saved in %s", OUTPUT_DIR)
    return {"metrics": rows, "comparison": comparison, "decision": decision}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_final_test_evaluation()


if __name__ == "__main__":
    main()
