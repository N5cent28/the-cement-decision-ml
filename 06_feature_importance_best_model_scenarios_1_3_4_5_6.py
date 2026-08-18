"""
Best-model feature importance for scenarios 1, 3, 4, 5, and 6.

Method:
  - Pick best model per scenario by highest test ROC-AUC.
  - Tree models: native feature_importances_.
  - Non-tree models: permutation importance on test set (ROC-AUC scoring).

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
        "results_json": "results_majority/model_results.json",
        "processed_dir": "processed_data",
    },
    "scenario3_majority_ct_plus_demo": {
        "results_json": "results_majority_CT+Demographics/model_results.json",
        "processed_dir": "processed_data_demographics_same_split",
    },
    "scenario4_majority_demo_only": {
        "results_json": "results_majority_demographics_only/model_results.json",
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
    if "scenario5_h3d_agree_ct_only" in variant or "scenario5_h3d_agreement_ct_only" in variant:
        return "gt_h3d_agree"
    if "scenario6_original_ct_plus_demo" in variant:
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
    model.fit(X_train, y_train)
    if model_name in {"Random Forest", "Gradient Boosting"}:
        imp = model.feature_importances_
        std = np.zeros_like(imp)
        method = "native_tree_importance"
    else:
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
