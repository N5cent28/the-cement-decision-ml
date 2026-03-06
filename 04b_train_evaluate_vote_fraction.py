"""
Step 4B: Model Training & Evaluation (Vote Fraction Ground Truth)
=================================================================
Trains regressors against gt_vote_fraction (continuous target in [0, 1]).
Outputs are saved to a separate results folder to avoid overwriting the
majority-vote analysis.

Run:  python 04b_train_evaluate_vote_fraction.py
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
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.svm import SVR

matplotlib.use("Agg")

PROCESSED_DIR = "processed_data_vote_fraction"
RESULTS_DIR = "results_vote_fraction"
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
            n_estimators=300,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        ),
        "SVR (RBF)": SVR(kernel="rbf", C=1.0, epsilon=0.05),
    }


def cross_validate_models(models, X, y, groups):
    print("=" * 70)
    print("GROUPED CROSS-VALIDATION (regression on vote fraction)")
    print("=" * 70)
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    results = {}

    for name, model in models.items():
        neg_rmse = cross_val_score(
            model,
            X,
            y,
            cv=gkf,
            groups=groups,
            scoring="neg_root_mean_squared_error",
        )
        r2 = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring="r2")
        rmse = -neg_rmse
        results[name] = {
            "mean_rmse": round(rmse.mean(), 4),
            "std_rmse": round(rmse.std(), 4),
            "mean_r2": round(r2.mean(), 4),
            "std_r2": round(r2.std(), 4),
        }
        print(
            f"  {name:<30s} RMSE={rmse.mean():.4f} +/- {rmse.std():.4f} "
            f"R2={r2.mean():.4f} +/- {r2.std():.4f}"
        )
    print()
    return results


def evaluate_on_test(models, X_train, y_train, X_test, y_test):
    print("=" * 70)
    print("TEST SET EVALUATION (regression + thresholded classification view)")
    print("=" * 70)

    out = {}
    y_test_bin = (y_test >= BINARY_THRESHOLD).astype(int)
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_cont = np.clip(model.predict(X_test), 0.0, 1.0)
        y_pred_bin = (y_pred_cont >= BINARY_THRESHOLD).astype(int)

        rmse = mean_squared_error(y_test, y_pred_cont) ** 0.5
        mae = mean_absolute_error(y_test, y_pred_cont)
        r2 = r2_score(y_test, y_pred_cont)

        # Secondary readout against binary majority-like boundary.
        acc = accuracy_score(y_test_bin, y_pred_bin)
        auc = roc_auc_score(y_test_bin, y_pred_cont)

        out[name] = {
            "metrics": {
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r2": round(r2, 4),
                "binary_accuracy_at_0p5": round(acc, 4),
                "binary_roc_auc_vs_thresholded_truth": round(auc, 4),
            },
            "y_pred_cont": y_pred_cont,
            "y_pred_bin": y_pred_bin,
        }

        print(f"\n  {name}:")
        for key, val in out[name]["metrics"].items():
            print(f"    {key:<35s} {val}")
    print()
    return out


def evaluate_by_cohort(models, X_test, y_test, cohorts):
    print("=" * 70)
    print("COHORT-SPECIFIC TEST PERFORMANCE (vote fraction)")
    print("=" * 70)
    cohort_results = {}

    for cohort in sorted(np.unique(cohorts)):
        mask = cohorts == cohort
        X_c = X_test[mask]
        y_c = y_test[mask]
        y_c_bin = (y_c >= BINARY_THRESHOLD).astype(int)

        cohort_results[cohort] = {}
        print(f"\n  {cohort} ({mask.sum()} test hips):")
        for name, model in models.items():
            y_pred = np.clip(model.predict(X_c), 0.0, 1.0)
            rmse = mean_squared_error(y_c, y_pred) ** 0.5
            acc = accuracy_score(y_c_bin, (y_pred >= BINARY_THRESHOLD).astype(int))
            cohort_results[cohort][name] = {
                "rmse": round(rmse, 4),
                "binary_accuracy_at_0p5": round(acc, 4),
                "n_samples": int(mask.sum()),
            }
            print(f"    {name:<30s} RMSE={rmse:.4f}  BinAcc@0.5={acc:.4f}")
    print()
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

    plt.suptitle("Vote-Fraction Regression: Predicted vs True", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "predicted_vs_true_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter plots to {path}")


def plot_regression_metrics(results):
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        rows.extend(
            [
                {"Model": name, "Metric": "RMSE", "Value": m["rmse"]},
                {"Model": name, "Metric": "MAE", "Value": m["mae"]},
                {"Model": name, "Metric": "R2", "Value": m["r2"]},
                {
                    "Model": name,
                    "Metric": "Binary Acc @0.5",
                    "Value": m["binary_accuracy_at_0p5"],
                },
                {
                    "Model": name,
                    "Metric": "Binary ROC-AUC",
                    "Value": m["binary_roc_auc_vs_thresholded_truth"],
                },
            ]
        )

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="Metric", y="Value", hue="Model", ax=ax)
    ax.set_title("Vote-Fraction Analysis: Model Comparison")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "metric_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved metric comparison to {path}")


def plot_binary_view(results, y_test):
    y_true_bin = (y_test >= BINARY_THRESHOLD).astype(int)
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(4.8 * n_models, 4.2))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_true_bin, res["y_pred_bin"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred Non-cem.", "Pred Cemented"],
            yticklabels=["True Non-cem.", "True Cemented"],
            ax=ax,
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.suptitle("Thresholded Binary View (at 0.5 vote fraction)", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "binary_view_confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved binary-view confusion matrices to {path}")


def save_results(cv_results, test_results, cohort_results, meta):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = {
        "analysis_variant": "vote_fraction",
        "cross_validation": cv_results,
        "test_set": {k: v["metrics"] for k, v in test_results.items()},
        "cohort_specific": cohort_results,
        "preprocessing": {
            "n_train": meta["n_train_rows"],
            "n_test": meta["n_test_rows"],
            "n_features": len(meta["feature_columns"]),
        },
    }
    json_path = os.path.join(RESULTS_DIR, "model_results.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved numeric results to {json_path}")

    summary_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("THA Cement Decision — Vote Fraction Analysis Summary\n")
        f.write("=" * 58 + "\n\n")
        f.write("Cross-validation (grouped):\n")
        for name, row in cv_results.items():
            f.write(
                f"  {name}: RMSE={row['mean_rmse']:.4f} +/- {row['std_rmse']:.4f}, "
                f"R2={row['mean_r2']:.4f} +/- {row['std_r2']:.4f}\n"
            )
        f.write("\nTest set metrics:\n")
        for name, res in test_results.items():
            f.write(f"\n  {name}:\n")
            for k, v in res["metrics"].items():
                f.write(f"    {k}: {v}\n")
    print(f"  Saved summary to {summary_path}\n")


def main():
    print("=" * 70)
    print("STEP 4B: MODEL TRAINING & EVALUATION (VOTE FRACTION)")
    print("=" * 70 + "\n")

    train, test, feature_cols, meta = load_processed()
    X_train, y_train, groups_train, _ = prepare_xy(train, feature_cols)
    X_test, y_test, _, cohorts_test = prepare_xy(test, feature_cols)

    print(f"Features: {len(feature_cols)}")
    print(f"Train: {len(y_train)} samples")
    print(f"Test:  {len(y_test)} samples")
    print(f"Train vote-fraction dist: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
    print(f"Test vote-fraction dist:  {pd.Series(y_test).value_counts().sort_index().to_dict()}\n")

    models = build_models()
    cv_results = cross_validate_models(models, X_train, y_train, groups_train)
    test_results = evaluate_on_test(models, X_train, y_train, X_test, y_test)
    cohort_results = evaluate_by_cohort(models, X_test, y_test, cohorts_test)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating plots:")
    plot_scatter_by_model(test_results, y_test)
    plot_regression_metrics(test_results)
    plot_binary_view(test_results, y_test)
    print()

    save_results(cv_results, test_results, cohort_results, meta)

    print("=" * 70)
    print("TRAINING & EVALUATION (VOTE FRACTION) COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
