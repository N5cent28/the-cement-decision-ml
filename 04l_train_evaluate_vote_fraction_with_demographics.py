"""
Scenario 13 Training/Evaluation: Vote fraction (continuous), CT + demographics.

Run:  python 04l_train_evaluate_vote_fraction_with_demographics.py
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.svm import SVR

matplotlib.use("Agg")

PROCESSED_DIR = "processed_data_vote_fraction_with_demographics_same_split"
RESULTS_DIR = "results_vote_fraction_with_demographics_same_split"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

TARGET_COL = "gt_vote_fraction"
BINARY_THRESHOLD = 0.5
N_CV_FOLDS = 5
RANDOM_STATE = 42


def load_processed():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    with open(os.path.join(PROCESSED_DIR, "preprocessing_metadata.json")) as f:
        meta = json.load(f)
    return train, test, meta["feature_columns"], meta


def prepare_xy(df, feature_cols):
    valid = df[TARGET_COL].notna()
    X = df.loc[valid, feature_cols].values
    y = df.loc[valid, TARGET_COL].astype(float).values
    groups = df.loc[valid, "Anonymize_ID"].values
    cohorts = df.loc[valid, "Cohort_group"].values
    return X, y, groups, cohorts


def build_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=250, max_depth=3, learning_rate=0.05, min_samples_leaf=3, random_state=RANDOM_STATE
        ),
        "SVR (RBF)": SVR(kernel="rbf", C=1.0, epsilon=0.05),
    }


def cross_validate_models(models, X, y, groups):
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    results = {}
    for name, model in models.items():
        neg_rmse = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring="neg_root_mean_squared_error")
        r2 = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring="r2")
        rmse = -neg_rmse
        results[name] = {
            "mean_rmse": round(rmse.mean(), 4),
            "std_rmse": round(rmse.std(), 4),
            "mean_r2": round(r2.mean(), 4),
            "std_r2": round(r2.std(), 4),
        }
    return results


def evaluate_on_test(models, X_train, y_train, X_test, y_test):
    out = {}
    y_test_bin = (y_test >= BINARY_THRESHOLD).astype(int)
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_cont = np.clip(model.predict(X_test), 0.0, 1.0)
        y_pred_bin = (y_pred_cont >= BINARY_THRESHOLD).astype(int)

        rmse = mean_squared_error(y_test, y_pred_cont) ** 0.5
        mae = mean_absolute_error(y_test, y_pred_cont)
        r2 = r2_score(y_test, y_pred_cont)
        acc = accuracy_score(y_test_bin, y_pred_bin)
        auc = roc_auc_score(y_test_bin, y_pred_cont)
        prec = precision_score(y_test_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_test_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)

        out[name] = {
            "metrics": {
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r2": round(r2, 4),
                "binary_accuracy_at_0p5": round(acc, 4),
                "binary_precision_at_0p5": round(prec, 4),
                "binary_recall_at_0p5": round(rec, 4),
                "binary_f1_at_0p5": round(f1, 4),
                "binary_roc_auc_vs_thresholded_truth": round(auc, 4),
            },
            "y_pred_cont": y_pred_cont,
            "y_pred_bin": y_pred_bin,
        }
    return out


def evaluate_by_cohort(models, X_test, y_test, cohorts):
    cohort_results = {}
    for cohort in sorted(np.unique(cohorts)):
        mask = cohorts == cohort
        X_c = X_test[mask]
        y_c = y_test[mask]
        y_c_bin = (y_c >= BINARY_THRESHOLD).astype(int)
        cohort_results[cohort] = {}
        for name, model in models.items():
            y_pred = np.clip(model.predict(X_c), 0.0, 1.0)
            rmse = mean_squared_error(y_c, y_pred) ** 0.5
            acc = accuracy_score(y_c_bin, (y_pred >= BINARY_THRESHOLD).astype(int))
            cohort_results[cohort][name] = {
                "rmse": round(rmse, 4),
                "binary_accuracy_at_0p5": round(acc, 4),
                "n_samples": int(mask.sum()),
            }
    return cohort_results


def plot_scatter_by_model(results, y_test):
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(4.8 * n_models, 4.2))
    if n_models == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        y_pred = res["y_pred_cont"]
        ax.scatter(y_test, y_pred, alpha=0.7, edgecolor="none")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("True vote fraction")
        ax.set_ylabel("Predicted vote fraction")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
    plt.suptitle("Scenario 13: Predicted vs True (CT+Demographics)", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "predicted_vs_true_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regression_metrics(results):
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        rows.extend([
            {"Model": name, "Metric": "RMSE", "Value": m["rmse"]},
            {"Model": name, "Metric": "MAE", "Value": m["mae"]},
            {"Model": name, "Metric": "R2", "Value": m["r2"]},
            {"Model": name, "Metric": "Binary Acc @0.5", "Value": m["binary_accuracy_at_0p5"]},
            {"Model": name, "Metric": "Binary ROC-AUC", "Value": m["binary_roc_auc_vs_thresholded_truth"]},
        ])
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="Metric", y="Value", hue="Model", ax=ax)
    ax.set_title("Scenario 13: Model Comparison (CT+Demographics)")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "metric_comparison.png"), dpi=150)
    plt.close(fig)


def plot_binary_view(results, y_test):
    y_true_bin = (y_test >= BINARY_THRESHOLD).astype(int)
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(4.8 * n_models, 4.2))
    if n_models == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_true_bin, res["y_pred_bin"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Non-cem.", "Pred Cemented"],
            yticklabels=["True Non-cem.", "True Cemented"], ax=ax,
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.suptitle("Scenario 13: Thresholded Binary View (0.5)", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "binary_view_confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_results(cv_results, test_results, cohort_results, meta):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = {
        "analysis_variant": meta.get("analysis_variant", "scenario13_vote_fraction_ct_plus_demo"),
        "cross_validation": cv_results,
        "test_set": {k: v["metrics"] for k, v in test_results.items()},
        "cohort_specific": cohort_results,
        "preprocessing": {
            "n_train": meta["n_train_rows"],
            "n_test": meta["n_test_rows"],
            "n_features": len(meta["feature_columns"]),
            "split_stratification_mode": meta.get("split_stratification_mode", "not_recorded"),
        },
    }
    with open(os.path.join(RESULTS_DIR, "model_results.json"), "w") as f:
        json.dump(data, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "results_summary.txt"), "w") as f:
        f.write("Scenario 13 (Vote fraction, CT+demographics) Results\n")
        f.write("=" * 54 + "\n\n")
        for name, row in cv_results.items():
            f.write(f"  {name}: RMSE={row['mean_rmse']:.4f} +/- {row['std_rmse']:.4f}, R2={row['mean_r2']:.4f} +/- {row['std_r2']:.4f}\n")
        f.write("\nTest set metrics:\n")
        for name, res in test_results.items():
            f.write(f"\n  {name}:\n")
            for k, v in res["metrics"].items():
                f.write(f"    {k}: {v}\n")


def main():
    train, test, feature_cols, meta = load_processed()
    X_train, y_train, groups_train, _ = prepare_xy(train, feature_cols)
    X_test, y_test, _, cohorts_test = prepare_xy(test, feature_cols)

    models = build_models()
    cv_results = cross_validate_models(models, X_train, y_train, groups_train)
    test_results = evaluate_on_test(models, X_train, y_train, X_test, y_test)
    cohort_results = evaluate_by_cohort(models, X_test, y_test, cohorts_test)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_scatter_by_model(test_results, y_test)
    plot_regression_metrics(test_results)
    plot_binary_view(test_results, y_test)

    save_results(cv_results, test_results, cohort_results, meta)
    print(f"Saved scenario 13 outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
