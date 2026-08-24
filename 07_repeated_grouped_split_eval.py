"""
Repeated grouped/stratified split evaluation for all 13 scenarios, with
95% confidence intervals.

Purpose:
  A single train/test split is a noisy estimate of model performance,
  especially with test sets as small as 19-43 hips. This script re-splits
  the patient-grouped data many times (default 30), retrains every model on
  each split, and reports the resulting distribution of metrics per
  scenario/model — including an empirical 95% CI (2.5th/97.5th percentile
  across repeats) — instead of relying on any single-split point estimate.

Classification scenarios (1, 3, 4, 5, 6, 7, 8, 9, 10, 11) report
accuracy/precision/recall/F1/ROC-AUC/PR-AUC. Vote-fraction regression
scenarios (2, 12, 13) report RMSE/MAE/R2 plus the same
precision/recall/F1/accuracy/ROC-AUC computed at a 0.5 vote-fraction
threshold, matching the convention used in 04b/04l/04m.

Cleaning, ground-truth definitions, split logic, imputation, and feature
engineering all come from `common_preprocessing.py` — this script only adds
the "repeat N times and summarize the distribution" layer on top, and owns
the SCENARIOS list (which ground truth + feature set each scenario uses).

Run examples:
  python 07_repeated_grouped_split_eval.py
  python 07_repeated_grouped_split_eval.py --repeats 50 --test-size 0.2
"""

import argparse
import json
import os
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC, SVR

import common_preprocessing as cp

matplotlib.use("Agg")

OUT_DIR = "results_repeated_splits_all_scenarios"
FIG_DIR = os.path.join(OUT_DIR, "figures")
BINARY_THRESHOLD = 0.5


@dataclass
class ScenarioDef:
    number: int
    name: str
    target_col: str
    use_ct: bool
    use_demo: bool
    is_regression: bool = False


SCENARIOS = [
    ScenarioDef(1, "scenario1_majority_ct_only", "gt_majority", use_ct=True, use_demo=False),
    ScenarioDef(3, "scenario3_majority_ct_plus_demo", "gt_majority", use_ct=True, use_demo=True),
    ScenarioDef(4, "scenario4_majority_demo_only", "gt_majority", use_ct=False, use_demo=True),
    ScenarioDef(2, "scenario2_vote_fraction_ct_only", "gt_vote_fraction", use_ct=True, use_demo=False, is_regression=True),
    ScenarioDef(13, "scenario13_vote_fraction_ct_plus_demo", "gt_vote_fraction", use_ct=True, use_demo=True, is_regression=True),
    ScenarioDef(12, "scenario12_vote_fraction_demo_only", "gt_vote_fraction", use_ct=False, use_demo=True, is_regression=True),
    ScenarioDef(5, "scenario5_h3d_agree_ct_only", "gt_h3d_agree", use_ct=True, use_demo=False),
    ScenarioDef(7, "scenario7_unanimous_ct_only", "gt_unanimous", use_ct=True, use_demo=False),
    ScenarioDef(9, "scenario9_unanimous_ct_plus_demo", "gt_unanimous", use_ct=True, use_demo=True),
    ScenarioDef(8, "scenario8_unanimous_demo_only", "gt_unanimous", use_ct=False, use_demo=True),
    ScenarioDef(10, "scenario10_original_ct_only", "gt_original", use_ct=True, use_demo=False),
    ScenarioDef(6, "scenario6_original_ct_plus_demo", "gt_original", use_ct=True, use_demo=True),
    ScenarioDef(11, "scenario11_original_demo_only", "gt_original", use_ct=False, use_demo=True),
]


def build_classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, min_samples_leaf=3, random_state=42
        ),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    }


def build_regressors():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=250, max_depth=3, learning_rate=0.05, min_samples_leaf=3, random_state=42
        ),
        "SVR (RBF)": SVR(kernel="rbf", C=1.0, epsilon=0.05),
    }


def preprocess_for_split(train, test, scenario):
    """Engineer/impute/scale the feature set this scenario needs. Delegates
    every actual operation to common_preprocessing; only decides which
    operations apply (CT, demographics, or both)."""
    feature_cols = []

    if scenario.use_ct:
        train, test = cp.cohort_knn_impute(train, test, cp.CT_FEATURE_COLS)
        train = cp.engineer_ct_features(train)
        test = cp.engineer_ct_features(test)
        ct_cols = cp.ct_feature_list()
        train, test, _ = cp.scale_features(train, test, ct_cols)
        feature_cols.extend(ct_cols)

    if scenario.use_demo:
        train, test = cp.impute_and_scale_demographics(train, test, cp.DEMO_FEATURES)
        feature_cols.extend(cp.DEMO_FEATURES)

    return train, test, feature_cols


def run_single_repeat(df_base, scenario, seed, test_size):
    df = df_base[df_base[scenario.target_col].notna()].copy()

    train, test, split_strategy = cp.grouped_split(
        df, scenario.target_col, test_size=test_size, random_state=seed, is_regression=scenario.is_regression
    )
    train, test, features = preprocess_for_split(train, test, scenario)

    X_train = train[features].values
    X_test = test[features].values

    rows = []
    if scenario.is_regression:
        y_train = train[scenario.target_col].astype(float).values
        y_test = test[scenario.target_col].astype(float).values
        y_test_bin = (y_test >= BINARY_THRESHOLD).astype(int)

        for model_name, model in build_regressors().items():
            model.fit(X_train, y_train)
            y_pred_cont = np.clip(model.predict(X_test), 0.0, 1.0)
            y_pred_bin = (y_pred_cont >= BINARY_THRESHOLD).astype(int)
            rows.append({
                "scenario": scenario.name,
                "scenario_number": scenario.number,
                "repeat_seed": seed,
                "model": model_name,
                "n_train": len(train),
                "n_test": len(test),
                "split_stratification_mode": split_strategy,
                "accuracy": accuracy_score(y_test_bin, y_pred_bin),
                "precision": precision_score(y_test_bin, y_pred_bin, zero_division=0),
                "recall": recall_score(y_test_bin, y_pred_bin, zero_division=0),
                "f1": f1_score(y_test_bin, y_pred_bin, zero_division=0),
                "roc_auc": roc_auc_score(y_test_bin, y_pred_cont),
                "pr_auc": np.nan,
                "rmse": mean_squared_error(y_test, y_pred_cont) ** 0.5,
                "mae": mean_absolute_error(y_test, y_pred_cont),
                "r2": r2_score(y_test, y_pred_cont),
            })
    else:
        y_train = train[scenario.target_col].astype(int).values
        y_test = test[scenario.target_col].astype(int).values

        for model_name, model in build_classifiers().items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            rows.append({
                "scenario": scenario.name,
                "scenario_number": scenario.number,
                "repeat_seed": seed,
                "model": model_name,
                "n_train": len(train),
                "n_test": len(test),
                "split_stratification_mode": split_strategy,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "pr_auc": average_precision_score(y_test, y_proba),
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
            })
    return rows


def summarize_metrics(df):
    def ci_lower(s):
        return s.quantile(0.025)

    def ci_upper(s):
        return s.quantile(0.975)

    summary = (
        df.groupby(["scenario_number", "scenario", "model"], as_index=False)
        .agg(
            repeats=("roc_auc", "count"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            roc_auc_median=("roc_auc", "median"),
            roc_auc_ci95_lower=("roc_auc", ci_lower),
            roc_auc_ci95_upper=("roc_auc", ci_upper),
            accuracy_mean=("accuracy", "mean"),
            accuracy_ci95_lower=("accuracy", ci_lower),
            accuracy_ci95_upper=("accuracy", ci_upper),
            precision_mean=("precision", "mean"),
            precision_ci95_lower=("precision", ci_lower),
            precision_ci95_upper=("precision", ci_upper),
            recall_mean=("recall", "mean"),
            recall_ci95_lower=("recall", ci_lower),
            recall_ci95_upper=("recall", ci_upper),
            f1_mean=("f1", "mean"),
            f1_ci95_lower=("f1", ci_lower),
            f1_ci95_upper=("f1", ci_upper),
            rmse_mean=("rmse", "mean"),
            rmse_ci95_lower=("rmse", ci_lower),
            rmse_ci95_upper=("rmse", ci_upper),
        )
        .sort_values(["scenario_number", "roc_auc_median"], ascending=[True, False])
    )
    return summary


def plot_roc_auc_distributions(df):
    os.makedirs(FIG_DIR, exist_ok=True)
    for scenario_number in sorted(df["scenario_number"].unique()):
        sub = df[df["scenario_number"] == scenario_number]
        scenario_name = sub["scenario"].iloc[0]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=sub, x="model", y="roc_auc", ax=ax)
        ax.set_title(f"{scenario_name}: ROC-AUC distribution across repeated splits")
        ax.set_xlabel("")
        ax.set_ylabel("ROC-AUC (binary-thresholded for regression scenarios)")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"{scenario_name}_roc_auc_boxplot.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30, help="Number of repeated splits per scenario.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test fraction per repeated split.")
    parser.add_argument("--seed-start", type=int, default=100, help="Starting random seed.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df_base = cp.load_and_clean_raw()
    all_rows = []
    for scenario in SCENARIOS:
        for i in range(args.repeats):
            seed = args.seed_start + i
            all_rows.extend(run_single_repeat(df_base, scenario, seed=seed, test_size=args.test_size))

    raw_df = pd.DataFrame(all_rows)
    summary_df = summarize_metrics(raw_df)
    split_usage_df = (
        raw_df.groupby(["scenario_number", "scenario", "split_stratification_mode"], as_index=False)
        .size()
        .rename(columns={"size": "model_rows"})
        .sort_values(["scenario_number", "model_rows"], ascending=[True, False])
    )

    raw_path = os.path.join(OUT_DIR, "repeated_split_metrics_raw.csv")
    summary_path = os.path.join(OUT_DIR, "repeated_split_metrics_summary.csv")
    split_usage_path = os.path.join(OUT_DIR, "split_stratification_usage.csv")
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    split_usage_df.to_csv(split_usage_path, index=False)
    plot_roc_auc_distributions(raw_df)

    config = {
        "repeats": args.repeats,
        "test_size": args.test_size,
        "seed_start": args.seed_start,
        "scenarios": [s.name for s in SCENARIOS],
    }
    with open(os.path.join(OUT_DIR, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved repeated-split results ({args.repeats} repeats x {len(SCENARIOS)} scenarios) to {OUT_DIR}")
    print(f"Raw metrics: {raw_path}")
    print(f"Summary:     {summary_path}")
    print(f"Split modes: {split_usage_path}")


if __name__ == "__main__":
    main()
