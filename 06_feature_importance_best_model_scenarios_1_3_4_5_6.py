"""
Best-model feature importance for scenarios 1, 3, 4, 5, 6, 7, 8, 9, 10, and 11
(all classification scenarios; vote-fraction scenarios 2, 12, 13 are
regression tasks and are out of scope for this script).

Method:
  - Pick best model per scenario by highest test ROC-AUC.
  - Compute test-set permutation importance (ROC-AUC scoring, n_repeats=30)
    for that model, regardless of model type. Earlier revisions used native
    feature_importances_ for tree-based models (Random Forest, Gradient
    Boosting), but that method is not comparable in scale to permutation
    importance and is biased toward high-cardinality/correlated features;
    all scenarios now use the same method so rankings are comparable
    scenario-to-scenario.

Note: scenarios 1, 3, and 4 previously pointed at stale results directories
(results_majority/, results_majority_CT+Demographics/,
results_majority_demographics_only/) generated before a split-stratification
fix; they now point at the current results/, results_majority_with_demographics_same_split/,
and results_majority_demographics_only_same_split/ directories, matching the
numbers reported in methods.md.

Run:
  python 06_feature_importance_best_model_scenarios_1_3_4_5_6.py
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

matplotlib.use("Agg")

OUT_DIR = "results_feature_importance_best_model"
FIG_DIR = os.path.join(OUT_DIR, "figures")

SCENARIOS = {
    "scenario1_majority_ct_only": {
        "results_json": "results/model_results.json",
        "processed_dir": "processed_data",
    },
    "scenario3_majority_ct_plus_demo": {
        "results_json": "results_majority_with_demographics_same_split/model_results.json",
        "processed_dir": "processed_data_demographics_same_split",
    },
    "scenario4_majority_demo_only": {
        "results_json": "results_majority_demographics_only_same_split/model_results.json",
        "processed_dir": "processed_data_demographics_only_same_split",
    },
    "scenario5_h3d_agree_ct_only": {
        "results_json": "results_scenario5_agree_ct_only/model_results.json",
        "processed_dir": "processed_data_scenario5_agree_ct_only",
    },
    "scenario6_original_ct_plus_demo": {
        "results_json": "results_scenario6_original_ct_plus_demo/model_results.json",
        "processed_dir": "processed_data_scenario6_original_ct_plus_demo",
    },
    "scenario7_unanimous_ct_only": {
        "results_json": "results_unanimous_ct_only/model_results.json",
        "processed_dir": "processed_data_unanimous_ct_only",
    },
    "scenario8_unanimous_demo_only": {
        "results_json": "results_unanimous_demographics_only_same_split/model_results.json",
        "processed_dir": "processed_data_unanimous_demographics_only_same_split",
    },
    "scenario9_unanimous_ct_plus_demo": {
        "results_json": "results_unanimous_with_demographics_same_split/model_results.json",
        "processed_dir": "processed_data_unanimous_with_demographics_same_split",
    },
    "scenario10_original_ct_only": {
        "results_json": "results_original_ct_only/model_results.json",
        "processed_dir": "processed_data_original_ct_only",
    },
    "scenario11_original_demo_only": {
        "results_json": "results_original_demographics_only_same_split/model_results.json",
        "processed_dir": "processed_data_original_demographics_only_same_split",
    },
}


def build_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=3, random_state=42, n_jobs=-1
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, min_samples_leaf=3, random_state=42
        )
    if name == "SVM (RBF)":
        return SVC(kernel="rbf", probability=True, random_state=42)
    raise ValueError(f"Unsupported model name: {name}")


def infer_target_col(variant):
    if "h3d_agree" in variant or "h3d_agreement" in variant:
        return "gt_h3d_agree"
    if "unanimous" in variant:
        return "gt_unanimous"
    if "original" in variant:
        return "gt_original"
    return "gt_majority"


def best_model_from_results(results_json):
    with open(results_json) as f:
        data = json.load(f)
    test_set = data["test_set"]
    best_name = max(test_set, key=lambda k: test_set[k]["roc_auc"])
    best_auc = test_set[best_name]["roc_auc"]
    variant = data.get("analysis_variant", "")
    return best_name, best_auc, variant


def load_split(processed_dir, feature_cols, target_col):
    train = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    X_train = train[feature_cols].values
    y_train = train[target_col].astype(int).values
    X_test = test[feature_cols].values
    y_test = test[target_col].astype(int).values
    return X_train, y_train, X_test, y_test


def compute_importance(model_name, model, X_train, y_train, X_test, y_test, feature_cols):
    # Always use test-set permutation importance, regardless of model type.
    # Native tree importance (feature_importances_) was used previously for
    # Random Forest / Gradient Boosting, but it is not comparable in scale to
    # permutation importance and is biased toward high-cardinality/correlated
    # features. Using one method for every scenario keeps the "best model"
    # feature-importance rankings comparable across scenarios regardless of
    # which of the 4 candidate algorithms happened to win.
    model.fit(X_train, y_train)
    perm = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=42, n_jobs=-1
    )
    imp = perm.importances_mean
    std = perm.importances_std
    method = "permutation_importance_test_auc"

    df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": imp,
            "importance_std": std,
            "abs_importance": np.abs(imp),
            "method": method,
        }
    ).sort_values("abs_importance", ascending=False)
    return df


def plot_top_features(df, scenario_name):
    top = df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(f"{scenario_name}: Top 15 Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"{scenario_name}_top15_importance.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    summary_rows = []
    figure_rows = []

    for scenario_name, conf in SCENARIOS.items():
        best_model_name, best_auc, variant = best_model_from_results(conf["results_json"])

        with open(os.path.join(conf["processed_dir"], "preprocessing_metadata.json")) as f:
            meta = json.load(f)
        feature_cols = meta["feature_columns"]
        target_col = infer_target_col(variant if variant else scenario_name)

        X_train, y_train, X_test, y_test = load_split(conf["processed_dir"], feature_cols, target_col)
        model = build_model(best_model_name)
        imp_df = compute_importance(best_model_name, model, X_train, y_train, X_test, y_test, feature_cols)

        csv_out = os.path.join(OUT_DIR, f"{scenario_name}_best_model_importance.csv")
        imp_df.to_csv(csv_out, index=False)
        fig_out = plot_top_features(imp_df, scenario_name)

        summary_rows.append(
            {
                "scenario": scenario_name,
                "analysis_variant": variant,
                "best_model": best_model_name,
                "best_test_roc_auc": best_auc,
                "target_col": target_col,
                "n_features": len(feature_cols),
                "importance_method": imp_df["method"].iloc[0],
                "importance_csv": csv_out,
                "importance_figure": fig_out,
            }
        )
        figure_rows.append({"scenario": scenario_name, "figure_path": fig_out})

    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "best_model_feature_importance_summary.csv"), index=False)
    pd.DataFrame(figure_rows).to_csv(os.path.join(OUT_DIR, "feature_importance_figures_index.csv"), index=False)
    print(f"Saved feature importance outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
