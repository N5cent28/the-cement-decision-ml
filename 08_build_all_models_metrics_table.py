"""
Step 8: Consolidated All-Models Metrics Table
==============================================
Walks every scenario's model_results.json (classification and vote-fraction
regression alike) and builds one master table with accuracy, precision,
recall, F1, and ROC-AUC for every model in every scenario, so no report
figure has to be hand-assembled from 13 separate JSON files.

Classification scenarios report these metrics directly. Vote-fraction
(regression) scenarios report precision/recall/F1/accuracy/ROC-AUC computed
at the 0.5 vote-fraction threshold (see the `binary_*` fields written by
04b/04l/04m) alongside their native regression metrics (RMSE, MAE, R2),
which have no classification equivalent and are left blank in those columns.

Every row also carries a 95% confidence interval on ROC-AUC (and, for
regression scenarios, RMSE), pulled from the repeated patient-grouped-split
distribution computed by 07_repeated_grouped_split_eval.py (30 resplits per
scenario) rather than the single train/test split used for the point
estimate. Single-split point estimates on test sets this small (19-43 hips)
are noisy; the CI columns show how much that noise actually is.

Run:
  python 07_repeated_grouped_split_eval.py   # regenerates the CI source data
  python 08_build_all_models_metrics_table.py
"""

import json
import os

import pandas as pd

REPORT_DIR = "reports"
CI_SUMMARY_PATH = "results_repeated_splits_all_scenarios/repeated_split_metrics_summary.csv"

# (scenario_number, ground_truth, feature_set, results_dir, is_regression)
SCENARIOS = [
    (1, "Majority vote", "CT-only", "results", False),
    (3, "Majority vote", "CT + demographics", "results_majority_with_demographics_same_split", False),
    (4, "Majority vote", "Demographics-only", "results_majority_demographics_only_same_split", False),
    (2, "Vote fraction", "CT-only", "results_vote_fraction", True),
    (13, "Vote fraction", "CT + demographics", "results_vote_fraction_with_demographics_same_split", True),
    (12, "Vote fraction", "Demographics-only", "results_vote_fraction_demographics_only_same_split", True),
    (5, "Halldor+3D agree (2/3)", "CT-only", "results_scenario5_agree_ct_only", False),
    (7, "Unanimous (3/3)", "CT-only", "results_unanimous_ct_only", False),
    (9, "Unanimous (3/3)", "CT + demographics", "results_unanimous_with_demographics_same_split", False),
    (8, "Unanimous (3/3)", "Demographics-only", "results_unanimous_demographics_only_same_split", False),
    (10, "Original surgeon only", "CT-only", "results_original_ct_only", False),
    (6, "Original surgeon only", "CT + demographics", "results_scenario6_original_ct_plus_demo", False),
    (11, "Original surgeon only", "Demographics-only", "results_original_demographics_only_same_split", False),
]


def load_scenario_rows(scenario_num, ground_truth, feature_set, results_dir, is_regression):
    path = os.path.join(results_dir, "model_results.json")
    with open(path) as f:
        data = json.load(f)

    n_train = data.get("preprocessing", {}).get("n_train")
    n_test = data.get("preprocessing", {}).get("n_test")

    rows = []
    for model_name, metrics in data["test_set"].items():
        if is_regression:
            row = {
                "scenario": scenario_num,
                "ground_truth": ground_truth,
                "feature_set": feature_set,
                "model": model_name,
                "task_type": "regression",
                "n_train": n_train,
                "n_test": n_test,
                "accuracy": metrics.get("binary_accuracy_at_0p5"),
                "precision": metrics.get("binary_precision_at_0p5"),
                "recall": metrics.get("binary_recall_at_0p5"),
                "f1": metrics.get("binary_f1_at_0p5"),
                "roc_auc": metrics.get("binary_roc_auc_vs_thresholded_truth"),
                "pr_auc": None,
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "r2": metrics.get("r2"),
                "note": "precision/recall/F1/accuracy/ROC-AUC computed at 0.5 vote-fraction threshold",
            }
        else:
            row = {
                "scenario": scenario_num,
                "ground_truth": ground_truth,
                "feature_set": feature_set,
                "model": model_name,
                "task_type": "classification",
                "n_train": n_train,
                "n_test": n_test,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "rmse": None,
                "mae": None,
                "r2": None,
                "note": "",
            }
        rows.append(row)
    return rows


def load_ci_summary():
    """Load the repeated-split 95% CI summary, keyed by (scenario_number, model)."""
    if not os.path.exists(CI_SUMMARY_PATH):
        return None
    ci = pd.read_csv(CI_SUMMARY_PATH)
    return ci.set_index(["scenario_number", "model"])


def attach_ci(df, ci_df):
    if ci_df is None:
        df["roc_auc_95ci_lower"] = None
        df["roc_auc_95ci_upper"] = None
        df["rmse_95ci_lower"] = None
        df["rmse_95ci_upper"] = None
        df["ci_repeats"] = None
        return df

    lowers, uppers, rmse_lowers, rmse_uppers, repeats = [], [], [], [], []
    for _, r in df.iterrows():
        key = (r["scenario"], r["model"])
        if key in ci_df.index:
            row = ci_df.loc[key]
            lowers.append(round(row["roc_auc_ci95_lower"], 4))
            uppers.append(round(row["roc_auc_ci95_upper"], 4))
            repeats.append(int(row["repeats"]))
            if r["task_type"] == "regression":
                rmse_lowers.append(round(row["rmse_ci95_lower"], 4))
                rmse_uppers.append(round(row["rmse_ci95_upper"], 4))
            else:
                rmse_lowers.append(None)
                rmse_uppers.append(None)
        else:
            lowers.append(None)
            uppers.append(None)
            rmse_lowers.append(None)
            rmse_uppers.append(None)
            repeats.append(None)

    df["roc_auc_95ci_lower"] = lowers
    df["roc_auc_95ci_upper"] = uppers
    df["rmse_95ci_lower"] = rmse_lowers
    df["rmse_95ci_upper"] = rmse_uppers
    df["ci_repeats"] = repeats
    return df


def save_markdown_table(df, path, title):
    cols = [
        "scenario", "ground_truth", "feature_set", "model", "n_train", "n_test",
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "roc_auc_95ci_lower", "roc_auc_95ci_upper", "ci_repeats",
    ]
    lines = [f"# {title}", ""]
    lines.append(
        "95% CIs are computed from 30 repeated patient-grouped train/test "
        "splits (see `results_repeated_splits_all_scenarios/`), not from the "
        "single split used for the point-estimate columns. A blank CI means "
        "the repeated-split run hasn't been regenerated for that scenario/model."
    )
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    all_rows = []
    for scenario_num, ground_truth, feature_set, results_dir, is_regression in SCENARIOS:
        all_rows.extend(
            load_scenario_rows(scenario_num, ground_truth, feature_set, results_dir, is_regression)
        )

    df = pd.DataFrame(all_rows).sort_values(["scenario", "model"]).reset_index(drop=True)
    ci_df = load_ci_summary()
    df = attach_ci(df, ci_df)

    csv_path = os.path.join(REPORT_DIR, "all_models_metrics_table.csv")
    df.to_csv(csv_path, index=False)

    md_path = os.path.join(REPORT_DIR, "all_models_metrics_table.md")
    save_markdown_table(df, md_path, "All Models: Accuracy, Precision, Recall, F1, ROC-AUC by Scenario")

    print(f"Saved {len(df)} model x scenario rows to:\n  {csv_path}\n  {md_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
